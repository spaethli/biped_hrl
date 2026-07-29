"""Goal decode contract (``src/tasks/velocity/rl/hrl/goal_space.py``).

The goal scale defines what the high level's bounded ``g`` *means* (``V* = ref + scale*g``),
so it is a property of the trained policy, not of the env it is replayed in. Losing that
invariant cost the 2026-07-15 phantom "HL hold degeneracy" (see ``freeze_scale``).
"""

import math

import pytest
import torch

from src.tasks.velocity.mdp.rewards import track_angular_velocity, track_linear_velocity
from src.tasks.velocity.rl.hrl.goal_space import (
  DEFAULT_GOAL_COMPONENTS,
  GoalSpace,
  build_goal_space,
  goal_dim,
)


# --- scale: what `g` means travels with the policy ---------------------------


def test_frozen_scale_survives_a_command_range_collapse(env):
  """A checkpoint-baked scale decodes ``g`` identically no matter what command ranges
  the replay env carries — the fix for the ``--eval-cmd-vx`` goal-channel collapse."""
  space = build_goal_space()
  state = space.extract(env)
  g = torch.ones(env.num_envs, space.dim)

  trained_scale = space.scale(env)
  target_as_trained = space.to_target(env, state, g, mode="delta")

  # Replay under a held command: --eval-cmd-vx 0.5 pins the twist range to a point.
  space.freeze_scale(trained_scale)
  env.set_twist_ranges(lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))

  assert torch.equal(space.scale(env), trained_scale)
  assert torch.allclose(space.to_target(env, state, g, mode="delta"), target_as_trained)


def test_unfrozen_scale_tracks_the_live_twist_curriculum(env):
  """During training the scale must follow the command ranges as the curriculum widens
  — the reason the live derivation exists at all, and why it must be pinned afterwards."""
  space = build_goal_space()
  env.set_twist_ranges(lin_vel_x=(-0.5, 0.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0))
  assert space.scale(env)[0].item() == pytest.approx(0.5)

  env.set_twist_ranges(lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0))
  assert space.scale(env)[0].item() == pytest.approx(1.5)


def test_unfreezing_restores_the_live_derivation(env):
  """``learn()`` un-pins so a resumed run tracks the curriculum again."""
  space = build_goal_space()
  space.freeze_scale(torch.full((space.dim,), 99.0))
  assert space.scale_frozen

  space.freeze_scale(None)
  assert not space.scale_frozen
  assert space.scale(env)[0].item() == pytest.approx(1.0)


# --- the g -> V* decode ------------------------------------------------------


def test_delta_decode_is_relative_to_the_current_state(env):
  """``delta`` (HIRO, the default): ``V* = state + scale*g``, so ``g=0`` asks the low
  level to hold exactly where it is."""
  space = build_goal_space()
  env.robot_data.root_link_lin_vel_b = torch.tensor([[0.4, -0.1, 0.0]]).repeat(env.num_envs, 1)
  state = space.extract(env)

  zero = space.to_target(env, state, torch.zeros(env.num_envs, space.dim), mode="delta")
  assert torch.allclose(zero, state)

  g = torch.full((env.num_envs, space.dim), 0.5)
  assert torch.allclose(
    space.to_target(env, state, g, mode="delta"), state + 0.5 * space.scale(env)
  )


def test_absolute_decode_ignores_the_current_state(env):
  """``absolute``: ``V* = center + scale*g`` — a state-INDEPENDENT target, so the same
  ``g`` means the same velocity however the robot is currently moving."""
  space = build_goal_space()
  g = torch.full((env.num_envs, space.dim), 0.5)

  env.robot_data.root_link_lin_vel_b = torch.zeros(env.num_envs, 3)
  at_rest = space.to_target(env, space.extract(env), g, mode="absolute")

  env.robot_data.root_link_lin_vel_b = torch.tensor([[0.9, 0.3, 0.0]]).repeat(env.num_envs, 1)
  while_moving = space.to_target(env, space.extract(env), g, mode="absolute")

  assert torch.allclose(at_rest, while_moving)


def test_absolute_g_spans_exactly_the_command_range(env):
  """``g`` in [-1,1] sweeps the velocity command range end to end (center + scale*g)."""
  space = build_goal_space()
  env.set_twist_ranges(lin_vel_x=(-0.5, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0))
  state = space.extract(env)

  lo = space.to_target(env, state, -torch.ones(env.num_envs, space.dim), mode="absolute")
  hi = space.to_target(env, state, torch.ones(env.num_envs, space.dim), mode="absolute")

  assert lo[:, 0].tolist() == pytest.approx([-0.5] * env.num_envs)
  assert hi[:, 0].tolist() == pytest.approx([1.5] * env.num_envs)


@pytest.mark.parametrize("mode", ["delta", "absolute"])
def test_to_g_inverts_to_target(env, mode):
  """``to_g`` recovers the ``g`` whose target equals the achieved state — the HIRO
  relabel map. Must round-trip, or relabeling rewrites goals to the wrong ones."""
  space = build_goal_space()
  state = space.extract(env)
  g = torch.tensor([[0.5, -0.25, 0.75, 0.1, -0.1, 0.2, -0.6]]).repeat(env.num_envs, 1)

  achieved = space.to_target(env, state, g, mode=mode)
  assert torch.allclose(space.to_g(env, state, achieved, mode=mode), g, atol=1e-6)


def test_to_g_clamps_unreachable_goals_into_the_bounded_action_range(env):
  """A relabel target outside one window's reach still has to be a legal HL action."""
  space = build_goal_space()
  state = space.extract(env)
  far = state + 1000.0

  g = space.to_g(env, state, far, mode="delta")
  assert torch.all(g <= 1.0) and torch.all(g >= -1.0)


def test_task_only_pins_posture_to_nominal_not_to_a_sagged_state(env):
  """``hl_velocity_goals_only``: the high level emits velocity columns only, and
  orientation/height are pinned to NOMINAL. Under a plain delta decode a sagged height
  with ``g=0`` is a rewarded steady state, which is how the co-trained walkers sank into
  bent knees (2026-07-09)."""
  space = build_goal_space()
  env.robot_data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.80]]).repeat(env.num_envs, 1)
  env.robot_data.projected_gravity_b = torch.tensor([[0.3, 0.0, -0.95]]).repeat(env.num_envs, 1)
  state = space.extract(env)
  g_task = torch.zeros(env.num_envs, space.task_dim)

  target = space.to_target(env, state, g_task, mode="delta", task_only=True)

  assert target.shape == (env.num_envs, space.dim)
  # Orientation target is upright and the height target is the default, NOT the sag.
  assert torch.allclose(target[:, 3:6], torch.tensor([0.0, 0.0, -1.0]))
  assert target[:, 6].tolist() == pytest.approx([1.02] * env.num_envs)


def test_task_dim_is_the_velocity_columns_only(env):
  space = build_goal_space()
  assert space.task_dim == 3
  assert space.dim == 7


# --- reward kernels ----------------------------------------------------------


def _reachable_states(n: int) -> torch.Tensor:
  """Physically reachable goal-space states, including thrashing and fallen ones:
  velocity well past the command range, projected gravity on the unit sphere (any
  attitude, incl. face-down), height from collapsed to airborne."""
  vel = (torch.rand(n, 3) - 0.5) * 12.0
  orient = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
  height = torch.rand(n, 1) * 1.6
  return torch.cat([vel, orient, height], dim=-1)


def test_exp_kernel_is_strictly_positive_over_the_reachable_state_space(env):
  """The exp kernel pairs with a TRUE-terminal ``fell_over``, so it must stay positive
  for ANY goal an infant high level emits — death has to forfeit a positive stream.
  The original quadratic orientation/height penalties drove this to ~-3.5 and re-created
  the suicide attractor (co-train collapse 2026-07-06, ep_len 7)."""
  space = build_goal_space()
  torch.manual_seed(0)
  achieved, garbage_target = _reachable_states(2048), _reachable_states(2048)

  r = space.reward(garbage_target, achieved, kernel="exp")
  assert torch.all(r > 0.0), f"exp kernel went non-positive: min={r.min().item()}"
  assert torch.all(r <= 4.0), f"exp kernel exceeded its bound: max={r.max().item()}"


def test_exp_positivity_depends_on_the_bounded_orientation_term(env):
  """WHY the guarantee above holds, made executable.

  ``exp(-d^2/sigma^2)`` underflows to *exactly* 0.0 in float32 once ``d^2/sigma^2`` exceeds
  ~88, which the velocity (|dv| > 4.7 m/s) and height (|dh| > 0.94 m) terms can reach. The
  sum survives only because projected gravity is a unit vector, so the orientation error is
  capped at ``2*sqrt(3)`` and that term can never underflow. A goal space without
  ``orientation`` would therefore break the exp <-> true-terminal pairing rule and could
  resurrect the suicide attractor — assert the dependency rather than trusting the docstring.
  """
  space = build_goal_space()
  far = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]])
  near = torch.zeros(1, space.dim)
  near[:, 5] = -1.0

  assert space.reward(far, near, kernel="exp").item() > 0.0

  vel_only = build_goal_space(("velocity",))
  # Both velocity sub-terms past their underflow thresholds (|dxy| > 4.7, |dyaw| > 6.6).
  underflowed = vel_only.reward(torch.tensor([[10.0, 0.0, 10.0]]), torch.zeros(1, 3), kernel="exp")
  assert underflowed.item() == 0.0, (
    "velocity-only goal space no longer underflows; re-check whether the exp kernel's "
    "positivity still rests on the orientation component"
  )


def test_exp_kernel_is_maximal_when_the_goal_is_reached(env):
  space = build_goal_space()
  s = torch.zeros(4, space.dim)
  assert space.reward(s, s, kernel="exp").tolist() == pytest.approx([4.0] * 4)


def test_l2_kernel_is_never_positive(env):
  """``l2`` (HIRO) is a strictly negative distance, which is why it needs the
  ``fell_over=time_out`` truncation bootstrap — never pair it with a true terminal."""
  space = build_goal_space()
  torch.manual_seed(0)
  achieved = torch.randn(256, space.dim)
  target = torch.randn(256, space.dim)

  r = space.reward(target, achieved, kernel="l2")
  assert torch.all(r <= 0.0)
  assert space.reward(achieved, achieved, kernel="l2").tolist() == pytest.approx([0.0] * 256)


def test_exp_velocity_kernel_matches_a0s_tracking_rewards(env):
  """RQ2 comparison cleanliness: A1's low-level velocity kernel is A0's
  ``track_linear_velocity`` + ``track_angular_velocity`` with ``V*`` substituted for the
  command. Checked on-axis (A0's extra z-linear / xy-angular penalties are outside the
  goal space, so they must be zero for the two to coincide)."""
  space = build_goal_space()
  asset_cfg = type("Cfg", (), {"name": "robot"})()
  achieved_vel = torch.tensor([[0.3, -0.2, 0.0]]).repeat(env.num_envs, 1)
  target_vel = torch.tensor([[0.8, 0.1, 0.4]]).repeat(env.num_envs, 1)

  env.robot_data.root_link_lin_vel_b = achieved_vel
  env.robot_data.root_link_ang_vel_b = torch.tensor([[0.0, 0.0, -0.2]]).repeat(env.num_envs, 1)
  env.set_command(target_vel)

  a0 = track_linear_velocity(env, std=0.5, command_name="twist", asset_cfg=asset_cfg)
  a0 = a0 + track_angular_velocity(
    env, std=math.sqrt(0.5), command_name="twist", asset_cfg=asset_cfg
  )

  achieved = space.extract(env)
  target = torch.cat([target_vel, achieved[:, 3:]], dim=-1)
  a1 = space.reward(target, achieved, kernel="exp")
  # Strip the orientation/height terms, which are at their maximum (target == achieved).
  a1 = a1 - 2.0

  assert torch.allclose(a1, a0, atol=1e-6)


# --- structure ---------------------------------------------------------------


def test_goal_dim_is_derived_from_the_component_list(env):
  """``goal_components`` is the single source of truth; nothing hardcodes a dimension."""
  for components in [("velocity",), ("velocity", "height"), DEFAULT_GOAL_COMPONENTS]:
    assert goal_dim(components) == build_goal_space(components).dim


def test_task_components_must_precede_non_task_ones():
  """The task_only slice is ``[:task_dim]``, so a non-task component in front would
  silently mis-slice. Fail loudly at construction instead."""
  from src.tasks.velocity.rl.hrl.goal_space import _FACTORY

  bad = [_FACTORY["orientation"](1.0), _FACTORY["velocity"](1.0)]
  with pytest.raises(ValueError, match="must precede"):
    GoalSpace(bad)
