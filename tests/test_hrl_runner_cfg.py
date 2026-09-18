"""Legality-check tests for HrlRunnerCfg.__post_init__ (architecture-review candidate E,
2026-09-18): the cross-field invariants that make a combination of HrlRunnerCfg fields
illegal used to live 250+ lines away, inside HierarchicalRunner.__init__, so testing them
required constructing a full runner against a real MuJoCo env (GPU, no CPU-only path).
Moved into __post_init__, each is now reachable from a bare HrlRunnerCfg(**kwargs)
construction -- no env, no GPU.

Each row overrides only the fields needed to isolate exactly ONE invariant, given the
checks' fixed order in __post_init__ (a row must not accidentally trip an earlier check
first, since __post_init__ raises on the first violation it finds). See rl_cfg.py's
__post_init__ for the canonical order these mirror.
"""

import pytest

from src.tasks.velocity.config.h1_2_a1.rl_cfg import HlVelJitterCfg, HrlRunnerCfg

# (kwargs overriding HrlRunnerCfg() defaults, expected message substring)
_CASES = [
  # 1. hl_cadence_source='hl' requires hl_cadence=True and hl_algorithm='td3'.
  (
    dict(hl_cadence_source="hl", hl_cadence=False),
    "hl_cadence_source='hl' requires hl_cadence=True and hl_algorithm='td3'",
  ),
  # 2. hl_velocity_goals_only is not implemented for hl_algorithm='ppo'. hl_cadence_source
  # must move off its default 'hl' too, else check 1 fires first (hl_algorithm='ppo' also
  # breaks the 'hl'-source pairing).
  (
    dict(hl_algorithm="ppo", hl_cadence_source="random"),
    "hl_velocity_goals_only is not implemented",
  ),
  # 3. warm_start_path and freeze_ll_path are mutually exclusive.
  (
    dict(warm_start_path="a0.pt", freeze_ll_path="a1_ll.pt"),
    "mutually exclusive",
  ),
  # 4. hl_vel_jitter requires hl_obs_vel=True.
  (
    dict(hl_vel_jitter=HlVelJitterCfg(enable=True), hl_obs_vel=False),
    "hl_vel_jitter requires hl_obs_vel=True",
  ),
  # 5. hl_vel_source validity.
  (
    dict(hl_vel_source="bogus"),
    "hl_vel_source must be one of",
  ),
  # 6. hl_vel_source != 'state' requires hl_obs_vel=True.
  (
    dict(hl_vel_source="leg_odom", hl_obs_vel=False),
    "hl_vel_source='leg_odom' requires hl_obs_vel=True",
  ),
  # 7. hl_vel_source != 'state' and hl_vel_jitter are mutually exclusive arms.
  (
    dict(hl_vel_source="leg_odom", hl_vel_jitter=HlVelJitterCfg(enable=True)),
    "hl_vel_source='leg_odom' and hl_vel_jitter are ARMS",
  ),
  # 8. hl_vel_residual enabled but every magnitude is 0 (silent no-op).
  (
    dict(hl_vel_source="leg_odom", hl_vel_residual=HlVelJitterCfg(enable=True)),
    "every magnitude is 0",
  ),
  # 9. hl_vel_residual only applies on top of a non-'state' estimator.
  (
    dict(hl_vel_residual=HlVelJitterCfg(enable=True)),
    "hl_vel_residual only applies on top of the simulated estimator",
  ),
  # 10. num_steps_per_env must be a multiple of c.
  (
    dict(num_steps_per_env=25),
    "must be a multiple of c",
  ),
]


@pytest.mark.parametrize("kwargs, match", _CASES)
def test_invalid_combination_raises_at_construction(kwargs, match):
  with pytest.raises(ValueError, match=match):
    HrlRunnerCfg(**kwargs)


def test_defaults_construct_without_raising():
  """The legal, all-defaults combination (hl_cadence_source='hl', hl_cadence=True,
  hl_algorithm='td3', hl_vel_source='state', num_steps_per_env=24, c=8, ...) must not
  trip any of the 10 invariants -- the same bare construction test_deploy_parity.py
  already relies on."""
  HrlRunnerCfg()
