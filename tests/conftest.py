"""CPU-only fakes for the RL seams under test.

These tests exercise the goal decode, reward-term semantics and checkpoint contract
as *pure tensor code* — no MuJoCo, no GPU, no env construction. The real
``ManagerBasedRlEnv`` is only ever read from (a handful of attributes), so a duck-typed
stand-in is enough and keeps the suite fast enough to run every session.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEnv:
  """Duck-typed ``ManagerBasedRlEnv`` exposing only what the code under test reads."""

  def __init__(self, num_envs: int = 4, device: str = "cpu") -> None:
    self.num_envs = num_envs
    self.device = device
    self.step_dt = 0.02
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    # A1a commanded-cadence clock (rewards._gait_schedule).
    self.hrl_phase = torch.zeros(num_envs)
    self.hrl_period = torch.full((num_envs,), 0.6)
    self._robot = SimpleNamespace(
      data=SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(num_envs, 3),
        root_link_ang_vel_b=torch.zeros(num_envs, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 1.02]]).repeat(num_envs, 1),
        default_root_state=torch.tensor([[0.0, 0.0, 1.02]]).repeat(num_envs, 1),
        joint_pos=torch.zeros(num_envs, 2),
        joint_vel=torch.zeros(num_envs, 2),
        qfrc_actuator=torch.zeros(num_envs, 2),
      )
    )
    # Two feet, both on the ground by default.
    self._contact = SimpleNamespace(
      data=SimpleNamespace(current_contact_time=torch.ones(num_envs, 2))
    )
    self.set_twist_ranges((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    self._command = torch.zeros(num_envs, 3)

  # --- scene / command manager surface ---------------------------------------

  @property
  def scene(self):
    return {"robot": self._robot, "feet_ground_contact": self._contact}

  def set_twist_ranges(self, lin_vel_x, lin_vel_y, ang_vel_z) -> None:
    """Rewrite the twist command ranges (what ``--eval-cmd-vx`` and deploy.yaml edit)."""
    ranges = SimpleNamespace(lin_vel_x=lin_vel_x, lin_vel_y=lin_vel_y, ang_vel_z=ang_vel_z)
    term = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
    self.command_manager = SimpleNamespace(
      get_term=lambda name: term,
      get_command=lambda name: self._command,
    )

  def set_command(self, cmd: torch.Tensor) -> None:
    self._command = cmd

  @property
  def robot_data(self):
    return self._robot.data

  @property
  def contact_data(self):
    return self._contact.data


class FakeAssetCfg:
  """Stand-in for ``SceneEntityCfg``: the reward terms only read ``name``/``joint_ids``."""

  def __init__(self, name: str = "robot", joint_ids=None) -> None:
    self.name = name
    self.joint_ids = slice(None) if joint_ids is None else joint_ids


@pytest.fixture
def env() -> FakeEnv:
  return FakeEnv()
