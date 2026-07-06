"""Declarative goal space for the hierarchical (A1) runner.

The goal space is a slice of the robot state the high level commands (HIRO-style).
It is declared as an ordered list of component *names* (serializable, so the runner
cfg still dumps to YAML); a registry turns names into ``GoalComponent`` objects with
the actual extractors at runtime. ``goal_dim`` is therefore *derived* from the
component list — nothing downstream hardcodes a dimension, and changing the goal
space (or ``c``) is a one-line config change.

Encoding is **delta (HIRO directional)**: the high level emits an absolute window
target ``V*`` (for learned HLs, ``V* = s_t + g``); the low level observes the
*remaining* delta ``V* - s_i`` each step and is rewarded ``-Σ_c w_c ||V*_c - s_i_c||``.

Default components (= 7 dims): base velocity (vx, vy, yaw_rate), base orientation
(projected gravity), base/torso height.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

# Component dimensions, known without an env (used to derive the obs term dim).
GOAL_COMPONENT_DIMS: dict[str, int] = {
  "velocity": 3,
  "orientation": 3,
  "height": 1,
  "posture": 27,  # available; not in the default space
}

DEFAULT_GOAL_COMPONENTS: tuple[str, ...] = ("velocity", "orientation", "height")


def goal_dim(components: tuple[str, ...]) -> int:
  """Total goal dimension for a component list (no env needed)."""
  return sum(GOAL_COMPONENT_DIMS[c] for c in components)


# --- Component extractors / nominal targets ---------------------------------


def _robot(env):
  return env.scene["robot"].data


def _vel_extract(env) -> torch.Tensor:
  d = _robot(env)
  return torch.cat([d.root_link_lin_vel_b[:, :2], d.root_link_ang_vel_b[:, 2:3]], dim=-1)


def _orient_extract(env) -> torch.Tensor:
  return _robot(env).projected_gravity_b


def _orient_nominal(env) -> torch.Tensor:
  g = torch.zeros(env.num_envs, 3, device=env.device)
  g[:, 2] = -1.0  # upright: gravity points down the body z-axis
  return g


def _height_extract(env) -> torch.Tensor:
  return _robot(env).root_link_pos_w[:, 2:3]


def _height_nominal(env) -> torch.Tensor:
  return _robot(env).default_root_state[:, 2:3]


# --- Per-dim goal scales (HIRO delta map V* = state + scale * g) -------------
# g is the high level's bounded Gaussian sample (a *directional* goal: the desired
# change in state over the window); scale sets the per-dim reach of that delta.
# Velocity scales track the live twist curriculum so editing the command ranges
# automatically rescales the goal.


def _vel_scale(env) -> torch.Tensor:
  r = env.command_manager.get_term("twist").cfg.ranges
  half = lambda lo, hi: max((hi - lo) / 2.0, 1e-3)  # noqa: E731
  return torch.tensor(
    [half(*r.lin_vel_x), half(*r.lin_vel_y), half(*r.ang_vel_z)], device=env.device
  )


def _orient_scale(env) -> torch.Tensor:
  return torch.ones(3, device=env.device)  # projected gravity lives in [-1, 1]


def _height_scale(env) -> torch.Tensor:
  return torch.tensor([0.2], device=env.device)  # ~0.2 m of height deviation


# --- Absolute-target centers (for the `absolute` HL map V* = center + scale*g) -----
# Only used when hl_target_mode == "absolute": `center` is the state-INDEPENDENT
# reference the bounded goal g perturbs around. Velocity uses the command-range
# midpoint so g in [-1,1] spans exactly the command range (center+scale*g sweeps
# [lo, hi]); non-task comps reuse their nominal value. Shape [N, dim] (broadcast).


def _vel_center(env) -> torch.Tensor:
  r = env.command_manager.get_term("twist").cfg.ranges
  mid = lambda lo, hi: (lo + hi) / 2.0  # noqa: E731
  c = torch.tensor(
    [mid(*r.lin_vel_x), mid(*r.lin_vel_y), mid(*r.ang_vel_z)], device=env.device
  )
  return c.expand(env.num_envs, 3)


@dataclass
class GoalComponent:
  """One slice of the goal space."""

  name: str
  dim: int
  extract: Callable[[object], torch.Tensor]
  """env -> current value of this slice, shape [N, dim]."""
  nominal: Callable[[object], torch.Tensor]
  """env -> nominal/default target (oracle uses this for non-task components)."""
  scale: Callable[[object], torch.Tensor]
  """env -> per-dim delta scale, shape [dim] (learned HL: V* = state + scale*g)."""
  is_task: bool
  """If True, the oracle targets the command instead of the nominal value."""
  weight: float = 1.0
  """Reward weight on this component's distance."""
  center: Callable[[object], torch.Tensor] | None = None
  """env -> absolute-target center, shape [N, dim] (learned HL, `absolute` mode:
  V* = center + scale*g). None -> falls back to ``nominal``."""


# Factory: name -> GoalComponent (given a reward weight).
_FACTORY: dict[str, Callable[[float], GoalComponent]] = {
  "velocity": lambda w: GoalComponent(
    "velocity", 3, _vel_extract, lambda e: torch.zeros(e.num_envs, 3, device=e.device),
    _vel_scale, True, w, center=_vel_center
  ),
  "orientation": lambda w: GoalComponent(
    "orientation", 3, _orient_extract, _orient_nominal, _orient_scale, False, w
  ),
  "height": lambda w: GoalComponent(
    "height", 1, _height_extract, _height_nominal, _height_scale, False, w
  ),
}


class GoalSpace:
  """An ordered set of goal components with extract / reward / oracle-target ops."""

  def __init__(self, components: list[GoalComponent]) -> None:
    self.components = components

  @property
  def dim(self) -> int:
    return sum(c.dim for c in self.components)

  def extract(self, env) -> torch.Tensor:
    """Current state slice s, shape [N, dim]."""
    return torch.cat([c.extract(env) for c in self.components], dim=-1)

  def scale(self, env) -> torch.Tensor:
    """Per-dim delta scale for the HIRO map V* = state + scale*g, shape [dim].

    Read from the env (velocity tracks the live twist curriculum), so it reflects
    the active command ranges if the curriculum stage changes between calls."""
    return torch.cat([c.scale(env) for c in self.components], dim=-1)

  def center(self, env) -> torch.Tensor:
    """Per-dim absolute-target center (velocity = command midpoint, others = nominal),
    shape [N, dim]. Only used by the `absolute` HL map."""
    return torch.cat(
      [(c.center or c.nominal)(env) for c in self.components], dim=-1
    )

  def to_target(self, env, state: torch.Tensor, g: torch.Tensor, mode: str) -> torch.Tensor:
    """Map a bounded goal g in [-1,1]^dim to the absolute window target V*.

    ``delta`` (HIRO, default): ``V* = state + scale*g`` (target relative to the current
    state). ``absolute``: ``V* = center + scale*g`` (state-independent — g spans the
    command range; the oracle's target structure). The LL still observes V*-s_i either
    way; only this map changes."""
    ref = self.center(env) if mode == "absolute" else state
    return ref + self.scale(env) * g

  def to_g(self, env, state: torch.Tensor, achieved: torch.Tensor, mode: str) -> torch.Tensor:
    """Inverse of :meth:`to_target`: the raw g whose target equals ``achieved``
    (the relabel empirical goal), clamped to [-1,1]."""
    ref = self.center(env) if mode == "absolute" else state
    return ((achieved - ref) / self.scale(env)).clamp(-1.0, 1.0)

  def reward(
    self, target: torch.Tensor, achieved: torch.Tensor, kernel: str = "l2"
  ) -> torch.Tensor:
    """LL intrinsic reward, shape [N].
    ``l2`` (HIRO): -Σ_c w_c ||target_c - achieved_c||_2 (strictly negative; needs
    ``fell_over=time_out`` or falling truncates the negative stream — suicide attractor).
    ``exp`` (A0-parity, for from-scratch training; pair with a true-terminal
    ``fell_over``): each component mirrors its A0 reward term with V* substituted for
    the command — velocity = track_linear_velocity + track_angular_velocity exp kernels,
    orientation = body_orientation_l2's quadratic penalty, height = quadratic (no A0
    analog; weight keeps the height goal dim from being gradient-dead). Positive-bounded
    (max 2), so death forfeits future value instead of ending a negative stream.
    Component ``weight`` (goal_weights) applies to ``l2`` only."""
    diff = target - achieved
    r = torch.zeros(diff.shape[0], device=diff.device)
    i = 0
    for c in self.components:
      d = diff[:, i : i + c.dim]
      if kernel == "l2":
        r = r - c.weight * d.norm(dim=-1)
      elif c.name == "velocity":
        r = r + torch.exp(-d[:, :2].square().sum(-1) / 0.25) + torch.exp(-d[:, 2].square() / 0.5)
      elif c.name == "orientation":
        r = r - d.square().sum(-1)
      elif c.name == "height":
        r = r - 10.0 * d.square().sum(-1)
      else:
        raise ValueError(f"no exp-kernel form for goal component '{c.name}'")
      i += c.dim
    return r

  def oracle_target(self, env, command: torch.Tensor) -> torch.Tensor:
    """Absolute window target for the oracle: command for the task component,
    nominal (upright / nominal height) for the rest. Shape [N, dim]."""
    parts = []
    for c in self.components:
      parts.append(command[:, : c.dim] if c.is_task else c.nominal(env))
    return torch.cat(parts, dim=-1)


def build_goal_space(
  components: tuple[str, ...] = DEFAULT_GOAL_COMPONENTS,
  weights: dict[str, float] | None = None,
) -> GoalSpace:
  weights = weights or {}
  return GoalSpace([_FACTORY[name](weights.get(name, 1.0)) for name in components])


def init_goal_buffer(env, dim: int, device: str) -> torch.Tensor:
  """Create the ``env.hrl_goal`` buffer the goal observation reads. Returns it."""
  env.hrl_goal = torch.zeros(env.num_envs, dim, device=device)
  return env.hrl_goal
