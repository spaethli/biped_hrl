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
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.config.h1_2.env_cfgs import unitree_h1_2_flat_env_cfg
from src.tasks.velocity.rl.hrl.goal_space import (
  DEFAULT_GOAL_COMPONENTS,
  goal_dim as compute_goal_dim,
)


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
  play: bool = False, goal_components: tuple[str, ...] = DEFAULT_GOAL_COMPONENTS
) -> ManagerBasedRlEnvCfg:
  """A1 flat env: A0 flat env with hierarchical observation groups.

  The ``goal`` obs dim is derived from ``goal_components`` so it always matches the
  runner's goal space (both default to the same component list).
  """
  cfg = unitree_h1_2_flat_env_cfg(play=play)
  _restructure_obs_groups(cfg, goal_dim=compute_goal_dim(goal_components))
  # Treat falling as a truncation (bootstrap), not a true terminal: the LL's
  # always-negative goal-distance reward would otherwise be gamed by terminating
  # early (suicide). A0 keeps fell_over as a true terminal (shared base cfg).
  cfg.terminations["fell_over"].time_out = True
    # Upweight velocity tracking in the reward the HL optimizes (A1 only; A0 untouched).
  # With weight 1.0, tracking (~+0.6/ep) is swamped by the penalty terms (joint_pos_limits
  # ~-3.6, action_rate ~-3.4), so the learned HL's dominant gradient is "reduce penalties"
  # -> emit V*~=state -> don't track. 4x makes tracking the HL's dominant signal. NOTE:
  # this diverges A1's reward from A0's (RQ2 confound) -- an A1 HL design choice; revisit
  # if a clean A0-vs-A1 comparison needs matched weights (retrain A0 + warm-start).
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_angular_velocity"].weight = 5.0
  return cfg
