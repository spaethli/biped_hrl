"""Unitree H1_2 A1 (HIRO) velocity environment configuration.

Reuses the A0 flat env unchanged (same reward / DR / terrain) and only restructures
the observation groups for the hierarchy:

  * ``policy``  = A0 actor proprio **minus the command** (shared LL/HL actor input).
  * ``command`` = the twist command on its own (HL actor sees it; LL does not).
  * ``goal``    = the 7-dim HL goal, read from ``env.hrl_goal``.
  * ``critic``  = A0 critic group (privileged), unchanged.

Removing the command from the low level is what prevents the hierarchy from
collapsing into A0: the goal becomes the only task channel reaching the LL. The
per-model group selection (which groups feed the LL vs HL) lives in ``rl_cfg.py``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.config.h1_2.env_cfgs import (
  apply_lean_reward,
  unitree_h1_2_flat_env_cfg,
)
from src.tasks.velocity.rl.hrl.goal_space import (
  DEFAULT_GOAL_COMPONENTS,
  goal_dim as compute_goal_dim,
)
from src.tasks.velocity.rl.hrl.leg_odom import leg_odometry


def _restructure_obs_groups(cfg: ManagerBasedRlEnvCfg, goal_dim: int) -> None:
  """Split the command out of the actor group and add the goal group (in place)."""
  actor_group = cfg.observations["actor"]
  actor_terms = dict(actor_group.terms)
  command_term = actor_terms.pop("command")  # KeyError here = upstream obs changed

  cfg.observations["policy"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=actor_group.enable_corruption,
    history_length=actor_group.history_length,
  )
  cfg.observations["command"] = ObservationGroupCfg(
    terms={"command": command_term},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )
  cfg.observations["goal"] = ObservationGroupCfg(
    terms={"goal": ObservationTermCfg(func=mdp.hrl_goal, params={"dim": goal_dim})},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )
  # The original combined actor group is no longer used (the LL/HL select policy/
  # command/goal explicitly via obs_groups); drop it to avoid ambiguity.
  del cfg.observations["actor"]


def unitree_h1_2_flat_a1_env_cfg(
  play: bool = False,
  goal_components: tuple[str, ...] = DEFAULT_GOAL_COMPONENTS,
  lean: bool = False,
) -> ManagerBasedRlEnvCfg:
  """A1 flat env: A0 flat env with hierarchical observation groups.

  The ``goal`` obs dim is derived from ``goal_components`` so it always matches the
  runner's goal space (both default to the same component list).

  ``lean=True`` strips the shaping reward terms (Track F): the env reward then matches
  lean-A0's, so the lean A0-vs-A1 comparison stays clean. Must warm-start lean-A1 from
  a *lean*-A0 checkpoint -> doc/hrl/hierarchy_benefit_roadmap.md Track F.
  """
  cfg = unitree_h1_2_flat_env_cfg(play=play)
  if lean:
    apply_lean_reward(cfg)
  _restructure_obs_groups(cfg, goal_dim=compute_goal_dim(goal_components))
  # Kernel pairing (default = exp): exp is all-positive so a TRUE terminal is safe and
  # correct (no suicide attractor). The old l2 kernel needed time_out=True (truncation)
  # to avoid gaming the always-negative reward — if switching back to l2, also set this
  # to True. Never mix (CLAUDE.md gotcha). exp+terminal validated 2026-07-09.
  cfg.terminations["fell_over"].time_out = False
  # WL-F (2026-08-09): simulated leg odometry, differenced at the PHYSICS rate. Registered
  # here because ``per_substep`` is the only hook mjlab exposes inside the decimation loop,
  # and 50 Hz differencing is a measured 2.4x worse. Runs unconditionally: it draws no
  # randomness and feeds nothing (the runner only reads ``env.leg_odom`` when
  # ``hl_vel_source='leg_odom'``), so runs that don't use it are unperturbed, while the
  # ``leg_odom_err`` metric stays available on every arm — including the HlVelJitter
  # comparator, which is what makes the two arms scorable against each other.
  cfg.metrics["leg_odom_err"] = MetricsTermCfg(func=leg_odometry, per_substep=True)
    # Upweight velocity tracking in the reward the HL optimizes (A1 only; A0 untouched).
  # With weight 1.0, tracking (~+0.6/ep) is swamped by the penalty terms (joint_pos_limits
  # ~-3.6, action_rate ~-3.4), so the learned HL's dominant gradient is "reduce penalties"
  # -> emit V*~=state -> don't track. 4x makes tracking the HL's dominant signal. NOTE:
  # this diverges A1's reward from A0's (RQ2 confound) -- an A1 HL design choice; revisit
  # if a clean A0-vs-A1 comparison needs matched weights (retrain A0 + warm-start).
  #cfg.rewards["track_linear_velocity"].weight = 5.0
  #cfg.rewards["track_angular_velocity"].weight = 5.0
  return cfg
