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


@dataclass
class GoalComponent:
  """One slice of the goal space."""

  name: str
  dim: int
  extract: Callable[[object], torch.Tensor]
  """env -> current value of this slice, shape [N, dim]."""
  nominal: Callable[[object], torch.Tensor]
  """env -> nominal/default target (oracle uses this for non-task components)."""
  is_task: bool
  """If True, the oracle targets the command instead of the nominal value."""
  weight: float = 1.0
  """Reward weight on this component's distance."""


# Factory: name -> GoalComponent (given a reward weight).
_FACTORY: dict[str, Callable[[float], GoalComponent]] = {
  "velocity": lambda w: GoalComponent(
    "velocity", 3, _vel_extract, lambda e: torch.zeros(e.num_envs, 3, device=e.device), True, w
  ),
  "orientation": lambda w: GoalComponent(
    "orientation", 3, _orient_extract, _orient_nominal, False, w
  ),
  "height": lambda w: GoalComponent("height", 1, _height_extract, _height_nominal, False, w),
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

  def reward(self, target: torch.Tensor, achieved: torch.Tensor) -> torch.Tensor:
    """HIRO intrinsic reward: -Σ_c w_c ||target_c - achieved_c||_2, shape [N]."""
    diff = target - achieved
    r = torch.zeros(diff.shape[0], device=diff.device)
    i = 0
    for c in self.components:
      r = r - c.weight * diff[:, i : i + c.dim].norm(dim=-1)
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
