"""High-level policy strategies for the hierarchical (A1) runner.

The ``HighLevel`` interface decouples the high-level decision-maker from the
co-training loop, so the same ``HierarchicalRunner`` drives any of:

  * ``OracleHighLevel`` (M1/M2)  — no learning; emits an absolute window target
    (command velocity + nominal upright/height) to validate the LL pipeline.
  * ``HighLevelPpo`` (M3, later) — on-policy PPO over the goal delta (2-level PPO).
  * ``HighLevelTd3`` (M4/M5, later) — off-policy TD3 + HIRO relabeling.

``act`` returns the **absolute window target** ``V*`` in goal space; the runner turns
it into the per-step remaining-delta observation and the intrinsic reward. For a
learned HL, ``V* = state + scale * g`` (g = the network's bounded delta output); the
oracle returns ``goal_space.oracle_target`` directly.

Only the oracle is implemented in milestone 1/2. The learned strategies are stubbed
so the runner contract is fixed now and the learners slot in without touching the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .goal_space import GoalSpace


class HighLevel(ABC):
  """Interface between the co-training loop and a high-level decision-maker."""

  def __init__(self, goal_space: GoalSpace) -> None:
    self.goal_space = goal_space

  @abstractmethod
  def act(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    """Fire the high level. Returns the absolute window target ``V*`` [N, goal_dim].

    ``obs`` is the current observation TensorDict (learned HLs read their input
    groups from it; the oracle ignores it). ``state`` is ``goal_space.extract(env)``
    at the fire step (learned HLs form ``V* = state + scale*g``; the oracle ignores it).
    """

  def begin_window(self, env, obs, state: torch.Tensor) -> None:
    """Called at HL fire, after :meth:`act`, to open a transition window."""

  def accumulate(self, task_reward: torch.Tensor) -> None:
    """Accumulate per-step task reward into the open HL transition."""

  def end_window(self, env, obs, state: torch.Tensor, dones: torch.Tensor) -> None:
    """Close the HL window and store/push the transition (no-op for oracle)."""

  def update(self) -> dict[str, float]:
    """Run HL learning. Returns a loss/metric dict (empty for oracle)."""
    return {}

  def state_dict(self) -> dict:
    return {}

  def load_state_dict(self, state_dict: dict) -> None:
    del state_dict

  def train_mode(self) -> None:
    pass

  def eval_mode(self) -> None:
    pass


class OracleHighLevel(HighLevel):
  """Non-learning HL: emit the absolute target = command velocity + nominal pose.

  Validates the entire LL pipeline (goal-space extraction, delta injection, intrinsic
  reward) and should reach ~A0 behaviour: the goal asks the body to track the
  commanded velocity while staying upright at nominal height.
  """

  def __init__(self, goal_space: GoalSpace, command_name: str = "twist") -> None:
    super().__init__(goal_space)
    self.command_name = command_name

  def act(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    del obs, state
    command = env.command_manager.get_command(self.command_name)
    return self.goal_space.oracle_target(env, command)
