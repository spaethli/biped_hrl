"""What every probe reads: the products of play.py's setup, plus the two helpers probes share."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class ProbeContext:
  cfg: Any                       # PlayConfig, after play.py's setup (eval_steps set by a schedule)
  env: Any                       # RslRlVecEnvWrapper
  runner: Any                    # None for the zero/random agents
  policy: Callable
  policy_wants_dones: bool       # only the hierarchical closure accepts (obs, dones)
  resume_path: Path | None
  agent_cfg: Any
  device: str
  dummy: bool                    # --agent zero|random
  seq_cmds: list | None = None   # --eval-cmd-seq, one (vx, vy, wz) per step
  cf_cols: tuple = ()            # --cf-e-col / --cf-e-val, parsed
  cf_vals: tuple = ()

  @property
  def label(self) -> str:
    return str(self.resume_path) if self.resume_path is not None else "unknown"


def pin_twist(env, vx, vy, wz, heading=None):
  """Hold the twist command at one point: collapse the ranges, write the live buffer, and turn
  standing/heading envs off, so a resample cannot move it."""
  t = env.unwrapped.command_manager.get_term("twist")
  t.cfg.ranges.lin_vel_x = (vx, vx)
  t.cfg.ranges.lin_vel_y = (vy, vy)
  t.cfg.rel_standing_envs = 0.0
  t.vel_command_b[:, 0], t.vel_command_b[:, 1], t.vel_command_b[:, 2] = vx, vy, wz
  t.is_standing_env[:] = False
  if heading is not None:
    # Heading hold: wz becomes a live P-correction toward the world heading. Keep the
    # ang_vel_z range OPEN — the controller clips its correction to it, so collapsing
    # it to (0,0) would silently disable the hold.
    t.cfg.heading_command = True
    t.cfg.ranges.heading = (heading, heading)
    t.heading_target[:] = heading
    t.is_heading_env[:] = True
  else:
    t.cfg.ranges.ang_vel_z = (wz, wz)
    t.cfg.heading_command = False
    t.is_heading_env[:] = False


def foot_pos_b(robot, foot_ids: torch.Tensor) -> torch.Tensor:
  """Foot site positions in the PELVIS frame, [B, 2, 3].

  A pure function of the encoders, so this is exactly what hardware FK would produce from
  `q` (`hrl::foot_site_b` / `leg_odom.foot_sites_b`, parity-checked). Used by the leg-odometry
  bench and by the stance-width metric, which is why it is a function and not a closure.
  """
  rel_w = robot.site_pos_w[:, foot_ids, :] - robot.root_link_pos_w[:, None, :]
  q = robot.root_link_quat_w                       # [B, 4] (w, x, y, z)
  qv, qw = q[:, 1:], q[:, 0:1]
  t = 2.0 * torch.cross(qv.unsqueeze(1).expand_as(rel_w), rel_w, dim=-1)
  return rel_w - qw.unsqueeze(1) * t + torch.cross(
    qv.unsqueeze(1).expand_as(t), t, dim=-1)
