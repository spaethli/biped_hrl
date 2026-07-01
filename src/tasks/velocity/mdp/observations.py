from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def hrl_goal(env: ManagerBasedRlEnv, dim: int = 3) -> torch.Tensor:
  """High-level goal observation for HRL (A1).

  Reads the goal buffer the hierarchical runner writes to ``env.hrl_goal`` when the
  high level fires. The goal lives in the planar base-velocity subspace
  ``(vx, vy, yaw_rate)``; ``dim`` is configurable via the runner cfg (``goal_dim``).
  Returns zeros if the buffer is not yet initialised (e.g. during model construction
  before the runner sets it), so the obs group resolves cleanly.
  """
  goal = getattr(env, "hrl_goal", None)
  if goal is None:
    return torch.zeros(env.num_envs, dim, device=env.device)
  return goal


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    # A1a: if the HL commands a gait cadence, the per-env accumulated phase buffer
    # (env.hrl_phase, already in [0,1)) overrides the fixed-period clock. Absent (A0 /
    # non-cadence A1) -> the original fixed-period phase. Backward compatible.
    hp = getattr(env, "hrl_phase", None)
    global_phase = hp if hp is not None else (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

