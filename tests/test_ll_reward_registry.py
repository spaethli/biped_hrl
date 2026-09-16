"""Registry-shape tests for HierarchicalRunner._build_ll_reward_terms (Phase 4 of the
LL-reward-registry migration).

Follows tests/test_freeze_ll_load.py's precedent: a SimpleNamespace stub standing in
for `self`, no FakeEnv, no real env -- `_build_ll_reward_terms` is a pure method (only
attribute reads on `self`), so it is exercised the same way
`HierarchicalRunner._load_frozen_ll` already is.

Numeric equivalence with the pre-refactor hand-rolled loop is NOT re-proven here (the
2026-09-16 GPU smoke bracket already did that, iteration-0 bit-identical) -- these
tests only guard the registry's SHAPE: every term's func/weight/params wired to the
right `self` attribute, and the hl_cadence gate.
"""

import inspect
import re
from types import SimpleNamespace

from src.tasks.velocity.mdp import rewards as mdp_rewards
from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner
from mjlab.envs.mdp.rewards import joint_acc_l2, joint_pos_limits
from mjlab.managers.reward_manager import RewardTermCfg

# Distinct nonzero sentinels (never equal to each other) so a copy-paste field swap
# between two terms is unambiguously caught by the weight assertions below.
_COEFS = dict(
  ll_stand_still_coef=0.11,
  ll_angmom_coef=0.12,
  ll_footslip_coef=0.13,
  ll_footclear_coef=0.14,
  ll_energy_coef=0.15,
  ll_joint_acc_coef=0.16,
  ll_joint_limits_coef=0.17,
  ll_soft_landing_coef=0.18,
  ll_body_ang_vel_coef=0.19,
  ll_cadence_coef=0.20,
  ll_pitchref_coef=0.21,
  ll_pushoff_coef=0.22,
  ll_rollover_coef=0.23,
)

_FOOT_CFG = object()
_ANKLE_CFG = object()
_TORSO_CFG = object()
_ROLLOVER_CFG = object()
_GAIT_PARAMS = dict(
  period=0.6, offset=[0.0, 0.5], threshold=0.56, duty_max=0.70,
  command_threshold=0.1, command_name="twist", sensor_name="feet_ground_contact",
  swing_time=0.0,
)
_PITCHREF_GAIT_PARAMS = {k: v for k, v in _GAIT_PARAMS.items() if k != "sensor_name"}
_ROLLOVER_GEOM_IDS = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]]
_ROLLOVER_BODY_IDS = [14, 15]


def _stub(hl_cadence: bool) -> SimpleNamespace:
  return SimpleNamespace(
    hl_cadence=hl_cadence,
    **_COEFS,
    ll_pitchref_theta_hs=-0.253,
    ll_pitchref_theta_to=-0.275,
    ll_pitchref_k=2.0,
    ll_pitchref_sigma=0.15,
    ll_pushoff_w=0.175,
    ll_pushoff_p_scale=40.0,
    ll_rollover_heel_x_max=-0.03,
    ll_rollover_toe_x_min=0.08,
    ll_rollover_w=0.175,
    _foot_asset_cfg=_FOOT_CFG,
    _ankle_asset_cfg=_ANKLE_CFG,
    _torso_asset_cfg=_TORSO_CFG,
    _rollover_asset_cfg=_ROLLOVER_CFG,
    _gait_params=_GAIT_PARAMS,
    _pitchref_gait_params=_PITCHREF_GAIT_PARAMS,
    _rollover_geom_ids=_ROLLOVER_GEOM_IDS,
    _rollover_body_ids=_ROLLOVER_BODY_IDS,
  )


_UNGATED_NAMES = {
  "stand_still", "angmom", "footslip", "footclear", "energy",
  "joint_acc", "joint_limits", "soft_landing", "body_ang_vel",
}
_CADENCE_GATED_NAMES = {"cadence", "pitchref", "pushoff", "rollover"}


def test_ll_reward_registry_contains_all_terms_when_hl_cadence_true():
  terms = HierarchicalRunner._build_ll_reward_terms(_stub(hl_cadence=True))
  assert set(terms) == _UNGATED_NAMES | _CADENCE_GATED_NAMES


def test_ll_reward_registry_excludes_cadence_gated_terms_when_hl_cadence_false():
  terms = HierarchicalRunner._build_ll_reward_terms(_stub(hl_cadence=False))
  assert set(terms) == _UNGATED_NAMES


def test_ll_reward_registry_weights_and_funcs():
  stub = _stub(hl_cadence=True)
  terms = HierarchicalRunner._build_ll_reward_terms(stub)

  # The 9 non-cadence-family terms wrap non-negative COST functions that the old
  # hand-rolled loop always SUBTRACTED from r_lo (matching A0's own RewardManager,
  # which registers the identical functions with negative weights -- e.g. stand_still
  # -1.0, joint_pos_limits -10.0, velocity_env_cfg.py). `learn()` ADDS compute()'s
  # output unconditionally, so the registry weight must be the NEGATED coefficient.
  negated = {
    "stand_still": (mdp_rewards.stand_still, stub.ll_stand_still_coef),
    "angmom": (mdp_rewards.angular_momentum_penalty, stub.ll_angmom_coef),
    "footslip": (mdp_rewards.feet_slip, stub.ll_footslip_coef),
    "footclear": (mdp_rewards.feet_clearance, stub.ll_footclear_coef),
    "energy": (mdp_rewards.cost_of_transport_penalty, stub.ll_energy_coef),
    "joint_acc": (joint_acc_l2, stub.ll_joint_acc_coef),
    "joint_limits": (joint_pos_limits, stub.ll_joint_limits_coef),
    "soft_landing": (mdp_rewards.soft_landing, stub.ll_soft_landing_coef),
    "body_ang_vel": (mdp_rewards.body_angular_velocity_penalty, stub.ll_body_ang_vel_coef),
  }
  for name, (func, coef) in negated.items():
    assert terms[name].func is func, name
    assert terms[name].weight == -coef, name

  # The 4 cadence-family terms are genuine rewards the old loop ADDED, so they keep
  # the unnegated coefficient.
  added = {
    "cadence": (mdp_rewards.feet_gait, stub.ll_cadence_coef),
    "pitchref": (mdp_rewards.ankle_pushoff_pitchref, stub.ll_pitchref_coef),
    "pushoff": (mdp_rewards.ankle_pushoff_power, stub.ll_pushoff_coef),
    "rollover": (mdp_rewards.heel_toe_rollover_contact, stub.ll_rollover_coef),
  }
  for name, (func, coef) in added.items():
    assert terms[name].func is func, name
    assert terms[name].weight == coef, name


def test_ll_reward_registry_params_wire_the_right_attributes():
  stub = _stub(hl_cadence=True)
  terms = HierarchicalRunner._build_ll_reward_terms(stub)

  assert terms["stand_still"].params == {"command_name": "twist", "command_threshold": 0.1}
  assert terms["angmom"].params == {"sensor_name": "robot/root_angmom"}
  assert terms["footslip"].params == {
    "sensor_name": "feet_ground_contact", "command_name": "twist",
    "command_threshold": 0.1, "asset_cfg": _FOOT_CFG,
  }
  assert terms["footslip"].params["asset_cfg"] is stub._foot_asset_cfg
  assert terms["footclear"].params == {
    "target_height": 0.10, "command_name": "twist",
    "command_threshold": 0.1, "asset_cfg": _FOOT_CFG,
  }
  assert terms["footclear"].params["asset_cfg"] is stub._foot_asset_cfg
  assert terms["energy"].params == {"command_name": "twist", "command_threshold": 0.1}
  assert terms["joint_acc"].params == {}
  assert terms["joint_limits"].params == {}
  assert terms["soft_landing"].params == {
    "sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.1,
  }
  assert terms["body_ang_vel"].params == {"asset_cfg": _TORSO_CFG}
  assert terms["body_ang_vel"].params["asset_cfg"] is stub._torso_asset_cfg

  assert terms["cadence"].params == {"use_commanded_phase": True, **_GAIT_PARAMS}
  assert terms["pitchref"].params == {
    "asset_cfg": _ANKLE_CFG, "command_name": "twist", "command_threshold": 0.1,
    "theta_hs": stub.ll_pitchref_theta_hs, "theta_to": stub.ll_pitchref_theta_to,
    "k": stub.ll_pitchref_k, "sigma": stub.ll_pitchref_sigma,
    "use_commanded_phase": True, **_PITCHREF_GAIT_PARAMS,
  }
  assert terms["pitchref"].params["asset_cfg"] is stub._ankle_asset_cfg
  assert terms["pushoff"].params == {
    "asset_cfg": _ANKLE_CFG, "sensor_name": "feet_ground_contact",
    "command_name": "twist", "command_threshold": 0.1,
    "w": stub.ll_pushoff_w, "p_scale": stub.ll_pushoff_p_scale,
    "use_commanded_phase": True, **_PITCHREF_GAIT_PARAMS,
  }
  assert terms["rollover"].params == {
    "sensor_name": "foot_subgeom_contact", "asset_cfg": _ROLLOVER_CFG,
    "geom_ids": _ROLLOVER_GEOM_IDS, "body_ids": _ROLLOVER_BODY_IDS,
    "heel_x_max": stub.ll_rollover_heel_x_max, "toe_x_min": stub.ll_rollover_toe_x_min,
    "w": stub.ll_rollover_w, "use_commanded_phase": True, **_PITCHREF_GAIT_PARAMS,
  }
  assert terms["rollover"].params["asset_cfg"] is stub._rollover_asset_cfg
  assert terms["rollover"].params["geom_ids"] is stub._rollover_geom_ids
  assert terms["rollover"].params["body_ids"] is stub._rollover_body_ids


def test_ll_reward_registry_returns_a_plain_dict_not_a_manager():
  """`_build_ll_reward_terms` must stay pure (dict only) -- the manager is constructed
  separately in `__init__`. Every value is a RewardTermCfg, never a live term object."""
  terms = HierarchicalRunner._build_ll_reward_terms(_stub(hl_cadence=True))
  assert isinstance(terms, dict)
  assert all(isinstance(t, RewardTermCfg) for t in terms.values())


def test_ll_reward_manager_constructed_with_scale_by_dt_false():
  """The critical correctness requirement (plan decision 4): the hand-rolled loop this
  registry replaces applies NO dt scaling to any ll_* term, unlike A0's own
  RewardManager (scale_by_dt=True default) -- getting this wrong silently divides
  every migrated term by ~step_dt (~50x) without erroring.

  Constructing a real HierarchicalRunner needs a full mjlab env (scene, observation
  manager, action manager, ...) that a SimpleNamespace stub cannot stand in for --
  building one here would reintroduce exactly the FakeEnv/real-env instantiation the
  plan rules out. Inspecting __init__'s source for the literal `RewardManager(...)`
  call site is the source-text-level equivalent: it is what the paired
  check_test_sensitivity.py mutation (`scale_by_dt=False` -> `scale_by_dt=True`) edits,
  so this test and that mutation target the same text.
  """
  src = inspect.getsource(HierarchicalRunner.__init__)
  m = re.search(r"RewardManager\(([^)]*)\)", src, re.DOTALL)
  assert m is not None, "no RewardManager(...) construction found in __init__"
  call_args = m.group(1)
  assert "scale_by_dt=False" in call_args
  assert "scale_by_dt=True" not in call_args
