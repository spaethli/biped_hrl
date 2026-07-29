"""Reward-term semantics (``src/tasks/velocity/mdp/rewards.py``).

These pin the *direction* and *gating* of the WL-D arm-6 terms and the gait schedule they
share. A reward that is merely mis-signed produces a confident, wrong experimental result
rather than a crash: the original ``ankle_pushoff_power`` rewarded a toe-lift 100% of the
time and cost a full retrain (2026-07-20).

The d(T) duty numbers are the calibrated values from the ADR-0004 schedule work.
"""

import pytest
import torch

from conftest import FakeAssetCfg
from src.tasks.velocity.mdp.rewards import ankle_pushoff_power, feet_gait

# Shared schedule: 2 legs in antiphase, 0.56 fixed duty, commanded (A1a) clock.
SCHEDULE = dict(
  period=0.6, offset=[0.0, 0.5], threshold=0.56,
  command_name="twist", command_threshold=0.1, use_commanded_phase=True,
)
PUSHOFF = dict(
  sensor_name="feet_ground_contact", w=0.2, p_scale=3.0,
  **{k: v for k, v in SCHEDULE.items()},
)


def _walking(env):
  """Commanded forward walk, phase parked in leg 0's terminal-stance window."""
  env.set_command(torch.tensor([[1.0, 0.0, 0.0]]).repeat(env.num_envs, 1))
  env.hrl_phase = torch.full((env.num_envs,), 0.5)


def _ankle(env, tau: float, qd: float, leg: int = 0):
  env.robot_data.qfrc_actuator = torch.zeros(env.num_envs, 2)
  env.robot_data.joint_vel = torch.zeros(env.num_envs, 2)
  env.robot_data.qfrc_actuator[:, leg] = tau
  env.robot_data.joint_vel[:, leg] = qd


# --- push-off direction: the 2026-07-20 defect -------------------------------


def test_pushoff_rewards_plantarflexion_not_a_toe_lift(env):
  """Both directions are concentric (``tau*qd > 0``); only plantarflexion (``qd > 0``,
  toe moves DOWN) is push-off. Gating on ``ReLU(power)`` alone cannot tell them apart,
  and the policy took the toe-lift every time."""
  _walking(env)
  cfg = FakeAssetCfg()

  _ankle(env, tau=10.0, qd=2.0)  # plantarflexion: propulsion
  plantarflexion = ankle_pushoff_power(env, asset_cfg=cfg, **PUSHOFF)

  _ankle(env, tau=-10.0, qd=-2.0)  # toe-lift: identical |power|, opposite direction
  toe_lift = ankle_pushoff_power(env, asset_cfg=cfg, **PUSHOFF)

  assert torch.all(plantarflexion > 0.5)
  assert torch.all(toe_lift == 0.0), "a toe-lift must earn nothing from a push-off reward"


def test_pushoff_requires_ground_contact(env):
  """Phase-only gating would let the low level harvest the bonus by driving the ankle in
  the air after an early liftoff. Power without contact is thrash, not propulsion."""
  _walking(env)
  _ankle(env, tau=10.0, qd=2.0)
  cfg = FakeAssetCfg()

  assert torch.all(ankle_pushoff_power(env, asset_cfg=cfg, **PUSHOFF) > 0.5)

  env.contact_data.current_contact_time = torch.zeros(env.num_envs, 2)
  assert torch.all(ankle_pushoff_power(env, asset_cfg=cfg, **PUSHOFF) == 0.0)


def test_pushoff_is_bounded_and_monotone_in_power(env):
  """Saturating by construction, so the term cannot be farmed by unbounded
  torque-cranking (the clamp discipline ``cost_of_transport_penalty`` established)."""
  _walking(env)
  cfg = FakeAssetCfg()

  rewards = []
  for qd in (0.5, 2.0, 20.0, 2000.0):
    _ankle(env, tau=10.0, qd=qd)
    rewards.append(ankle_pushoff_power(env, asset_cfg=cfg, **PUSHOFF)[0].item())

  assert rewards == sorted(rewards), f"not monotone in delivered power: {rewards}"
  assert rewards[0] > 0.0
  # Saturating: a 100x torque-crank past the scale buys essentially nothing more, and the
  # per-leg bonus never exceeds 1.0 however hard the ankle is driven.
  assert max(rewards) <= 1.0
  assert rewards[-1] - rewards[2] < 1e-6


def test_pushoff_is_gated_off_when_standing(env):
  """Command-gated like every other locomotion term: no walk command, no push-off bonus."""
  _walking(env)
  _ankle(env, tau=10.0, qd=2.0)
  env.set_command(torch.zeros(env.num_envs, 3))

  assert torch.all(ankle_pushoff_power(env, asset_cfg=FakeAssetCfg(), **PUSHOFF) == 0.0)


# --- the shared gait schedule ------------------------------------------------


def test_commanded_phase_follows_the_hrl_clock_not_the_episode_clock(env):
  """Under ``use_commanded_phase`` the schedule keys off ``env.hrl_phase``. A phase that
  never advances is the deploy-side "dead gait clock" signature (WL-B, 2026-07-21), which
  voided every live A1 result recorded before it was found."""
  env.set_command(torch.tensor([[1.0, 0.0, 0.0]]).repeat(env.num_envs, 1))

  env.hrl_phase = torch.zeros(env.num_envs)
  both_in_stance = feet_gait(env, sensor_name="feet_ground_contact", **SCHEDULE)

  env.hrl_phase = torch.full((env.num_envs,), 0.3)
  one_swinging = feet_gait(env, sensor_name="feet_ground_contact", **SCHEDULE)

  assert torch.all(both_in_stance == 1.0)
  assert torch.all(one_swinging == 0.5), "advancing hrl_phase must move the schedule"

  # The fixed-period episode clock must NOT leak in while commanded phase is active.
  env.episode_length_buf = torch.full((env.num_envs,), 999, dtype=torch.long)
  assert torch.equal(feet_gait(env, sensor_name="feet_ground_contact", **SCHEDULE), one_swinging)


def test_fixed_clock_advances_with_the_episode(env):
  """With ``use_commanded_phase`` off (the A0 path) the schedule runs off the episode
  clock, and must be unaffected by ``hrl_phase``."""
  env.set_command(torch.tensor([[1.0, 0.0, 0.0]]).repeat(env.num_envs, 1))
  fixed = dict(SCHEDULE, use_commanded_phase=False)

  env.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long)
  at_start = feet_gait(env, sensor_name="feet_ground_contact", **fixed)

  # 0.3 of a 0.6 s period -> half a cycle, leg 0 out of stance.
  env.episode_length_buf = torch.full((env.num_envs,), 15, dtype=torch.long)
  mid_cycle = feet_gait(env, sensor_name="feet_ground_contact", **fixed)

  assert torch.all(at_start == 1.0)
  assert torch.all(mid_cycle == 0.5)


@pytest.mark.parametrize(
  "period, expected_duty",
  [(0.5, 0.56), (0.8, 0.6125), (1.0, 0.69), (1.2, 0.70), (1.3, 0.70)],
)
def test_dt_duty_schedule_is_clamped_between_the_floor_and_human_slow_walk(
  env, period, expected_duty
):
  """ADR-0004 d(T): hold single support ~= the swing-time constant and let double stance
  absorb long periods, ``d = 1 - swing_time/T``, floored at the fixed threshold (fast band
  unchanged) and capped at 0.70 (human slow-walk duty; <= 0.5 would be running).

  Probed through the real ``feet_gait`` rather than the private helper: with both feet
  down, leg 0 is in stance exactly while its phase is below the duty, so the reward steps
  from 1.0 to 0.5 as the probe phase crosses d(T)."""
  env.set_command(torch.tensor([[1.0, 0.0, 0.0]]).repeat(env.num_envs, 1))
  env.hrl_period = torch.full((env.num_envs,), period)
  sched = dict(SCHEDULE, swing_time=0.31)

  def stance_at(phase: float) -> float:
    env.hrl_phase = torch.full((env.num_envs,), phase)
    return feet_gait(env, sensor_name="feet_ground_contact", **sched)[0].item()

  assert stance_at(expected_duty - 0.01) == 1.0
  assert stance_at(expected_duty + 0.01) == 0.5


def test_dt_duty_cap_is_configurable(env):
  """``duty_max`` is the cap actually applied (cadence_duty_range's upper end)."""
  env.set_command(torch.tensor([[1.0, 0.0, 0.0]]).repeat(env.num_envs, 1))
  env.hrl_period = torch.tensor([0.5, 0.8, 1.0, 1.2])
  env.hrl_phase = torch.full((env.num_envs,), 0.66)

  capped = feet_gait(
    env, sensor_name="feet_ground_contact", swing_time=0.31, duty_max=0.65, **SCHEDULE
  )
  assert torch.all(capped == 0.5), "every period's duty must be capped below 0.66"


def test_swing_time_zero_keeps_the_fixed_duty(env):
  """Default (``swing_time=0``) is byte-identical to the fixed-threshold behavior for
  every period — the d(T) schedule is opt-in and must not perturb the A0 baseline."""
  env.set_command(torch.tensor([[1.0, 0.0, 0.0]]).repeat(env.num_envs, 1))
  env.hrl_period = torch.tensor([0.5, 0.8, 1.0, 1.2])
  env.hrl_phase = torch.full((env.num_envs,), 0.60)  # past the 0.56 fixed duty

  assert torch.all(feet_gait(env, sensor_name="feet_ground_contact", **SCHEDULE) == 0.5)
