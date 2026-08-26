"""Deploy/ONNX parity (``deploy/robots/h1_2`` vs the training config).

The C++ controller re-declares, by hand, things the training side already knows: the
observation vector layout, the PD gains, the default pose, and every joint's position
limit. Those files carry "MUST stay in lockstep" / "MUST match the trained checkpoint"
comments and nothing enforced them — the 2026-07-20 audit found **15 of 27** joint
limits wrong, hand-entered from a documentation page instead of the URDF/XML.

A drift here is silent on both sides: sim keeps working, and the robot just behaves
slightly wrong. These tests make the lockstep executable.
"""

import re
from pathlib import Path

import mujoco
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy/robots/h1_2"
LIMITS_H = DEPLOY / "include/h1_2_limits.h"
HRL_YAML = DEPLOY / "config/policy/velocity_hrl/v0/params/deploy.yaml"
ROBOT_XML = REPO / "src/assets/robots/unitree_h1_2/xmls/h1_2.xml"

# ADR-0006 Spec B: joint_offset only ever belongs in the REAL-robot configs (hardware
# encoder-zero calibration); the sim configs' encoders are perfect by construction.
REAL_YAMLS = [
  DEPLOY / "config/policy/velocity/v0/params/deploy_real.yaml",
  DEPLOY / "config/policy/velocity_hrl/v0/params/deploy_real.yaml",
]
SIM_YAMLS = [
  DEPLOY / "config/policy/velocity/v0/params/deploy.yaml",
  HRL_YAML,
]

# The deploy YAML names observation terms after their mdp functions; training names them
# after the quantity. Same term, same order — only the label differs.
DEPLOY_TO_TRAINING_OBS = {
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "gait_phase_cmd": "phase",
  "joint_pos_rel": "joint_pos",
  "joint_vel_rel": "joint_vel",
  "last_action": "actions",
}


def _joint_xml_name(hardware_name: str) -> str:
  """Hardware/deploy joint name -> the name used in the MuJoCo model."""
  return "torso_joint" if hardware_name == "waist_yaw" else f"{hardware_name}_joint"


@pytest.fixture(scope="module")
def limits_header() -> tuple[list[str], list[tuple[float, float]]]:
  """(names, limits) parsed from the C++ header, both in hardware-ID order."""
  text = LIMITS_H.read_text()
  names_block = re.search(r"h1_2_joint_names\s*=\s*\{\{(.*?)\}\};", text, re.S).group(1)
  names = re.findall(r'"([^"]+)"', names_block)
  limits_block = re.search(r"h1_2_joint_limits\s*=\s*\{\{(.*?)\}\};", text, re.S).group(1)
  limits = [
    (float(lo), float(hi))
    for lo, hi in re.findall(r"\{\s*(-?[\d.]+)f\s*,\s*(-?[\d.]+)f\s*\}", limits_block)
  ]
  return names, limits


@pytest.fixture(scope="module")
def joint_trips() -> list[tuple[float, float]]:
  """(tau, dq) trip thresholds parsed from the C++ header, in hardware-ID order."""
  block = re.search(r"h1_2_joint_trips\s*=\s*\{\{(.*?)\}\};", LIMITS_H.read_text(), re.S)
  return [
    (float(tau), float(dq))
    for tau, dq in re.findall(r"\{\s*([\d.]+)f\s*,\s*([\d.]+)f\s*\}", block.group(1))
  ]


@pytest.fixture(scope="module")
def model_ranges() -> dict[str, tuple[float, float]]:
  model = mujoco.MjSpec.from_file(str(ROBOT_XML)).compile()
  out = {}
  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if model.jnt_limited[j]:
      out[name] = (float(model.jnt_range[j][0]), float(model.jnt_range[j][1]))
  return out


@pytest.fixture(scope="module")
def hrl_deploy() -> dict:
  return yaml.safe_load(HRL_YAML.read_text())


@pytest.fixture(scope="module")
def obs_groups() -> dict[str, list[str]]:
  """Observation term order per group, for both tasks. Loading the task configs costs a
  few seconds, so it is done once for the module."""
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  out = {}
  for label, task in (("a0", "Unitree-H1_2-Flat"), ("a1", "Unitree-H1_2-Flat-A1")):
    cfg = load_env_cfg(task)
    for group, g in cfg.observations.items():
      out[f"{label}.{group}"] = list(g.terms.keys())
  return out


# --- joint limits: the 15/27 defect ------------------------------------------


def test_deploy_joint_limits_match_the_training_model(limits_header, model_ranges):
  """Every limit the safety filter enforces on hardware must be the model's own range.
  15/27 of these were wrong before the 2026-07-20 audit; MuJoCo clamps joints silently,
  so neither the bridge nor the replica can observe a bad limit (they log trig_joint 0)."""
  names, limits = limits_header
  assert len(names) == 27 and len(limits) == 27

  mismatched = []
  for name, (lo, hi) in zip(names, limits):
    expected = model_ranges[_joint_xml_name(name)]
    if abs(lo - expected[0]) > 1e-3 or abs(hi - expected[1]) > 1e-3:
      mismatched.append(f"{name}: header ({lo}, {hi}) vs model {expected}")

  assert not mismatched, "deploy joint limits drifted from the model:\n" + "\n".join(mismatched)


def test_every_actuated_joint_has_a_deploy_limit(limits_header, model_ranges):
  """A joint missing from the header would be unprotected by the safety filter."""
  names, _ = limits_header
  covered = {_joint_xml_name(n) for n in names}
  assert covered == set(model_ranges), f"unprotected joints: {set(model_ranges) - covered}"


# --- ADR-0006 Spec B: joint_offset (encoder-zero correction) -----------------


@pytest.mark.parametrize("yaml_path", REAL_YAMLS)
def test_joint_offset_is_27_long_and_inert_when_absent(yaml_path):
  """Absent must mean a dead no-op (unitree_articulation.h and State_RL*.cpp both fall
  back to a 27-zero vector), and a present vector must be exactly 27 long — a short one
  would leave the tail joints uncorrected on one side of the read/write pair, which is
  precisely the sensor-vs-command inconsistency Spec B exists to remove."""
  cfg = yaml.safe_load(yaml_path.read_text())
  offset = cfg.get("joint_offset", [0.0] * 27)
  assert len(offset) == 27


@pytest.mark.parametrize("yaml_path", REAL_YAMLS)
def test_joint_offset_touches_only_leg_joints(yaml_path):
  """A non-zero joint_offset is a legitimate per-session encoder calibration (ADR-0006,
  kept from 2026-08-03 because it improves posture and walk stability, not just pitch).
  But only the 12 leg slots are ever calibrated: the waist and arms are held at
  default_joint_pos, so a non-zero entry there is a slot typo that silently shifts the
  held pose instead of correcting an encoder."""
  cfg = yaml.safe_load(yaml_path.read_text())
  offset = cfg.get("joint_offset", [0.0] * 27)
  assert all(v == 0.0 for v in offset[12:]), f"offset outside the legs: {offset[12:]}"


@pytest.mark.parametrize("yaml_path", REAL_YAMLS)
def test_joint_offset_magnitude_is_physically_sane(yaml_path):
  """Guards a decimal slip. Measured calibration values are <= 0.025 rad/joint (the
  ADR-0006 sweep), so 0.05 rad (2.9 deg) leaves 2x headroom over anything observed; a
  slipped 0.12-for-0.012 is 6.9 deg on one joint, enough to topple the robot on the first
  step. The required value also drifts between sessions, so a stale-but-plausible offset
  must stay recoverable — an implausible one must not reach the robot at all."""
  cfg = yaml.safe_load(yaml_path.read_text())
  offset = cfg.get("joint_offset", [0.0] * 27)
  assert all(abs(v) <= 0.05 for v in offset), f"implausible joint_offset: {offset}"


@pytest.mark.parametrize("yaml_path", REAL_YAMLS)
def test_joint_offset_is_not_a_policy_space_constant(yaml_path):
  """joint_offset is hardware calibration, not policy-space: it must never collapse onto
  default_joint_pos or the action offset (both trained constants), which would silently
  mean a Spec B edit also moved the trained pose/action mapping."""
  cfg = yaml.safe_load(yaml_path.read_text())
  if "joint_offset" not in cfg:
    pytest.skip("joint_offset not set in this config")
  offset = cfg["joint_offset"]
  assert len(offset) == 27
  assert offset != cfg["default_joint_pos"]
  assert offset != cfg["actions"]["JointPositionAction"]["offset"]


@pytest.mark.parametrize("yaml_path", SIM_YAMLS)
def test_joint_offset_absent_from_sim_configs(yaml_path):
  """Sim encoders are perfect by construction (ADR-0006 Spec B rationale) -- joint_offset
  belongs only in deploy_real.yaml, never in the keyboard/sim deploy.yaml twins."""
  cfg = yaml.safe_load(yaml_path.read_text())
  assert "joint_offset" not in cfg


# --- the sim bridge's base-state source (2026-08-05) --------------------------

EST_YAML = DEPLOY / "config/policy/velocity_hrl/v0/params/deploy_est.yaml"
# The three keys that switch the deploy off its sim-only privileged signals. The first two
# (2026-08-05) source the goal-space state `s`; the third (2026-08-06) sources the HIGH
# level's absolute velocity, which is a SEPARATE consumer -- `s` can be perfectly well-formed
# while the HL's input is dead, which is exactly what the 2026-08-05 bridge A/B measured.
BASE_STATE_KEYS = ("base_vel_from_imu", "base_height_from_fk", "hl_vel_from_leg_odom")


def test_estimator_bridge_config_is_a_pure_base_state_delta():
  """deploy_est.yaml is what deploy_readiness.py drives the A1 candidate arm with, so it has
  to be deploy.yaml in every respect EXCEPT the base-state source. If anything else drifts
  between them, the gate stops measuring the shipped config and nobody finds out -- the two
  files look interchangeable and are edited months apart.

  The estimator arm exists because the bridge's rt/sportmodestate is a SIM-ONLY privileged
  signal: the real robot publishes zeros there, so a gate driven by deploy.yaml validates a
  configuration the robot never runs."""
  sim = yaml.safe_load(HRL_YAML.read_text())
  est = yaml.safe_load(EST_YAML.read_text())
  for key in BASE_STATE_KEYS:
    assert est["hrl"].get(key) is True, f"deploy_est.yaml must set hrl.{key}: true"
    assert sim["hrl"].get(key) in (None, False), \
      f"deploy.yaml is the GROUND-TRUTH arm; it must not set hrl.{key}"
  for d in (sim, est):
    d["hrl"] = {k: v for k, v in d["hrl"].items() if k not in BASE_STATE_KEYS}
  assert sim == est, "deploy_est.yaml drifted from deploy.yaml beyond the base-state keys"


@pytest.mark.parametrize("yaml_path", REAL_YAMLS)
def test_real_config_gives_the_high_level_a_live_velocity_source(yaml_path):
  """`hl_obs_vel` makes the high level read an ABSOLUTE base velocity, and on the real H1-2
  there are only two candidate sources for it -- both dead unless leg odometry is on.

  rt/sportmodestate publishes zeros once this controller has command. The IMU increment is
  worse than it looks: `base_vel_increment(psi, w, 0, lev0)` is EXACTLY 0 at a window start
  by construction, and the HL fires exactly at window starts, so it reads 0.000000 at every
  single fire, at every speed. Measured on the bridge with the trip hold disabled: estimator
  3/3 falls against ground truth 0/3, with the fire rows confirmed as the exactly-zero est_vx
  rows (modal gap = c).

  This is the same class as the 2026-08-05 splay and the same reason it needs a test rather
  than a gate: in the bridge sportmodestate IS live, so a config missing this key behaves
  correctly there and fatally on hardware. The C++ refuses to construct on such a config;
  this asserts the repo cannot commit one."""
  hrl = yaml.safe_load(yaml_path.read_text()).get("hrl")
  if hrl is None or not hrl.get("hl_obs_vel"):
    pytest.skip("no hl_obs_vel -- the high level reads no absolute velocity")
  assert hrl.get("hl_vel_from_leg_odom") is True, (
    f"hrl.hl_obs_vel is set in {yaml_path.name} but hrl.hl_vel_from_leg_odom is "
    f"{hrl.get('hl_vel_from_leg_odom')!r}. On hardware that leaves the high level with no "
    f"live absolute velocity source: it would read exactly 0 at every fire.")


def test_a0_twin_of_the_estimator_config_exists():
  """State_RLBase loads whatever filename H1_2_DEPLOY_CFG names from its OWN params dir, and
  both FSM states are constructed at startup -- so a missing A0 twin kills the binary before
  State_RLHRL is even built. Found the hard way while testing the T1 refusal path."""
  assert (DEPLOY / "config/policy/velocity/v0/params/deploy_est.yaml").exists()


# --- per-joint torque / joint-velocity trip (2026-08-05) ----------------------

# Highest |tau_est| (Nm) and |dq| (rad/s) each joint GROUP reached during normal walking,
# over the union of both sources the thresholds are sized against: 389 s of healthy hardware
# walking (10 sessions 2026-07-20..2026-08-03, leg-active with the IMU tilt envelope under
# 12 deg for +-1 s so stumbles and falls are out) and 103 s of non-falling bridge walking.
# Regenerate with scripts/measure_joint_trip_thresholds.py.
#
# THE UNION IS THE POINT. The hardware pool is A0-dominated and A1 is measurably twitchier,
# so a hardware-only bound let the table hold right_hip_yaw during the A1 candidate's own
# normal bridge gait. A threshold at or below any of these numbers holds a joint mid-stride
# during healthy walking, which is a fall cause, not a fall guard.
WALKING_MAX_BY_GROUP = {
  "hip_yaw": ([0, 6], 174.1, 7.55),
  "hip_pitch": ([1, 7], 152.8, 6.93),
  "hip_roll": ([2, 8], 201.8, 8.13),
  "knee": ([3, 9], 295.9, 16.00),
  "ank_pitch": ([4, 10], 95.7, 21.48),
  "ank_roll": ([5, 11], 33.7, 20.53),
  "waist": ([12], 58.3, 5.18),
  "shoulder": ([13, 14, 15, 20, 21, 22], 29.4, 3.94),
  "elbow_wrist": ([16, 17, 18, 19, 23, 24, 25, 26], 18.0, 5.75),
}
# The 2026-08-05 leg splay: joints that must still trip, with their peak |tau_est| / |dq|
# over the event. RIGHT hip yaw (68.4 Nm / 6.55 rad/s) is deliberately NOT here -- clearing
# the bridge's 174 Nm hip-yaw torque pushed that threshold to 230 Nm, which costs its
# contribution to this event. Left hip roll is likewise absent: it peaked at 152.8 Nm /
# 8.71 rad/s and never carried the splay. Four joints still catch it, on the first
# anomalous sample, which is what the guard has to deliver.
SPLAY_PEAK = {0: (68.4, 13.65), 1: (218.2, 8.47), 7: (241.7, 15.20), 8: (329.6, 12.65)}


def test_every_joint_has_a_trip_threshold(joint_trips, limits_header):
  names, _ = limits_header
  assert len(joint_trips) == len(names) == 27


@pytest.mark.parametrize("group", sorted(WALKING_MAX_BY_GROUP))
def test_trip_thresholds_clear_measured_walking(joint_trips, limits_header, group):
  """Sized off WALKING, never off the stand. The stand sits at 36 Nm / 0.06 rad/s, so a
  stand-derived threshold would fire on every stride; walking reaches 296 Nm / 21.5 rad/s.
  A threshold below the measured walking peak holds a joint mid-stride during healthy
  walking, which is a fall cause, not a fall guard."""
  names, _ = limits_header
  ids, wtau, wdq = WALKING_MAX_BY_GROUP[group]
  for jid in ids:
    tau, dq = joint_trips[jid]
    assert tau > wtau, f"{names[jid]}: tau trip {tau} <= measured walking max {wtau} Nm"
    assert dq > wdq, f"{names[jid]}: dq trip {dq} <= measured walking max {wdq} rad/s"


@pytest.mark.parametrize("jid", sorted(SPLAY_PEAK))
def test_trip_thresholds_still_catch_the_2026_08_05_splay(joint_trips, limits_header, jid):
  """The other half of the sizing: headroom raised far enough to silence walking must not
  be raised so far that the failure it was built for slips through. These four hip joints
  carried the splay; each has to be over at least one of its two thresholds."""
  names, _ = limits_header
  tau, dq = joint_trips[jid]
  stau, sdq = SPLAY_PEAK[jid]
  assert stau > tau or sdq > dq, (
    f"{names[jid]}: splay peak ({stau} Nm, {sdq} rad/s) is under BOTH trips "
    f"({tau} Nm, {dq} rad/s) — the trigger would not have fired on this joint")


# --- the goal-space state source on the real robot ----------------------------

# Which deployable estimator each goal component depends on. `orientation` is absent on
# purpose: it comes from the IMU, which the real LowState does provide.
GOAL_COMPONENT_ESTIMATOR = {"velocity": "base_vel_from_imu", "height": "base_height_from_fk"}


@pytest.mark.parametrize("yaml_path", REAL_YAMLS)
def test_real_config_never_sources_the_goal_state_from_sportmodestate(yaml_path):
  """rt/sportmodestate is a SIM-ONLY privileged signal: on the real H1-2 it publishes
  identical zeros once our own low-level controller has command (verified over all 1182
  rows of trajectories/all_joints_2026-08-05_09-44-00.csv). State_RLHRL defaults both
  estimator keys to false when absent, so a real config that simply omits them runs the
  policy on velocity 0 / height 0 -- the height goal delta becomes nominal_root_height
  (1.3076 m) against a 0.2 m scale, which splayed the robot's legs within ~1 s of entering
  HRL mode on 2026-08-05.

  No bridge gate can catch this: in the bridge SportModeState IS live, so the identical
  config behaves correctly there. The C++ refuses to construct on such a config; this test
  is the same rule applied to the config in the repo, so the defect cannot be committed."""
  cfg = yaml.safe_load(yaml_path.read_text())
  hrl = cfg.get("hrl")
  if hrl is None:
    pytest.skip("no hierarchy in this config -- no goal-space state to source")
  for component in hrl.get("goal_components", []):
    key = GOAL_COMPONENT_ESTIMATOR.get(component)
    if key is None:
      continue
    assert hrl.get(key) is True, (
      f"goal component {component!r} needs hrl.{key}: true in {yaml_path.name}; "
      f"got {hrl.get(key)!r}. Absent means the goal state comes from the dead "
      f"rt/sportmodestate topic on hardware.")


# --- observation layout: the warm-start premise ------------------------------


def test_command_obs_sits_mid_vector_at_actor_cols_6_9(obs_groups):
  """A0's actor puts ``command`` at cols 6:9, AFTER base_ang_vel(3)+projected_gravity(3).
  The A1 warm start relies on this being an interior gap, not a trailing block."""
  order = obs_groups["a0.actor"]
  assert order[:3] == ["base_ang_vel", "projected_gravity", "command"]
  assert order.index("command") == 2, "command must stay the third actor term"


def test_a1_policy_group_is_the_a0_actor_minus_command(obs_groups):
  """A1's low level sees A0's proprio with ``command`` removed and nothing reordered —
  the exact premise of the gap-aware warm-start copy."""
  a0_without_command = [t for t in obs_groups["a0.actor"] if t != "command"]
  assert obs_groups["a1.policy"] == a0_without_command


def test_a1_critic_keeps_command_in_place(obs_groups):
  """The critic still carries ``command`` at the same offset, which is why its warm-start
  copy is a plain contiguous leading block with no gap."""
  a0 = obs_groups["a0.actor"]
  assert obs_groups["a1.critic"][: len(a0)] == a0


def test_deploy_yaml_observation_order_matches_training(obs_groups, hrl_deploy):
  """The C++ assembles the low-level input by concatenating these terms in YAML order.
  Any reordering feeds the policy a permuted observation vector — which trains and
  deploys without error, and simply walks worse."""
  deploy_terms = list(hrl_deploy["observations"]["policy"].keys())
  unknown = [t for t in deploy_terms if t not in DEPLOY_TO_TRAINING_OBS]
  assert not unknown, f"deploy.yaml declares observation terms training does not: {unknown}"

  translated = [DEPLOY_TO_TRAINING_OBS[t] for t in deploy_terms]
  assert translated == obs_groups["a1.policy"]

  assert list(hrl_deploy["observations"]["command"].keys()) == ["keyboard_velocity_commands"]


# --- gains / pose lockstep ---------------------------------------------------


def _training_gains() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
  from src.assets.robots.unitree_h1_2.h1_2_constants import H1_2_ARTICULATION

  kp, kd, effort = {}, {}, {}
  for act in H1_2_ARTICULATION.actuators:
    for expr in act.target_names_expr:
      kp[expr], kd[expr], effort[expr] = act.stiffness, act.damping, act.effort_limit
  return kp, kd, effort


def _resolve(per_expr: dict[str, float], joint: str) -> float:
  matches = [v for expr, v in per_expr.items() if re.fullmatch(expr, joint)]
  assert len(matches) == 1, f"{joint} matched {len(matches)} actuator patterns"
  return matches[0]


@pytest.mark.parametrize("field, index", [("stiffness", 0), ("damping", 1)])
def test_deploy_gains_match_the_training_constants(limits_header, hrl_deploy, field, index):
  """``deploy.yaml`` says these MUST stay in lockstep with ``h1_2_constants.py``. A gain
  mismatch is a sim-to-real gap that looks like a policy problem (ADR-0005 model v2)."""
  names, _ = limits_header
  kp, kd, _ = _training_gains()
  per_expr = kp if field == "stiffness" else kd

  drift = []
  for i, name in enumerate(names):
    expected = _resolve(per_expr, _joint_xml_name(name))
    actual = hrl_deploy[field][i]
    if abs(actual - expected) > 1e-6:
      drift.append(f"{name}: deploy {actual} vs training {expected}")

  assert not drift, f"{field} drifted from h1_2_constants.py:\n" + "\n".join(drift)


def test_deploy_default_pose_matches_the_training_home_keyframe(limits_header, hrl_deploy):
  """The action offset and PD target both key off this pose; a drift biases every
  commanded joint position by a constant."""
  from src.assets.robots.unitree_h1_2.h1_2_constants import HOME_KEYFRAME

  names, _ = limits_header
  drift = []
  for i, name in enumerate(names):
    joint = _joint_xml_name(name)
    expected = next(
      (v for expr, v in HOME_KEYFRAME.joint_pos.items() if re.fullmatch(expr, joint)), 0.0
    )
    if abs(hrl_deploy["default_joint_pos"][i] - expected) > 1e-6:
      drift.append(f"{name}: deploy {hrl_deploy['default_joint_pos'][i]} vs training {expected}")

  assert not drift, "default pose drifted:\n" + "\n".join(drift)


def test_deploy_action_scale_follows_the_derived_rule(limits_header, hrl_deploy):
  """Model v2 (ADR-0005 option B) derives the action scale as ``0.25 * effort / kp``
  rather than a flat 0.25 — the single largest delta from upstream."""
  names, _ = limits_header
  kp, _, effort = _training_gains()
  scales = hrl_deploy["actions"]["JointPositionAction"]["scale"]

  drift = []
  for i, name in enumerate(names):
    joint = _joint_xml_name(name)
    expected = 0.25 * _resolve(effort, joint) / _resolve(kp, joint)
    if abs(scales[i] - expected) > 1e-4:
      drift.append(f"{name}: deploy {scales[i]} vs derived {expected:.6f}")

  assert not drift, "action scale drifted from 0.25*effort/kp:\n" + "\n".join(drift)


# --- the goal-scale metadata path --------------------------------------------


def test_deploy_cadence_period_range_matches_training(hrl_deploy):
  """The HL's extra tanh dim is mapped affinely onto this range to get the stride period,
  on both sides. ``deploy.yaml`` says it MUST match ``agent.yaml``; a mismatch silently
  rescales every commanded cadence, which reads as a gait problem rather than a config one."""
  from src.tasks.velocity.config.h1_2_a1.rl_cfg import HrlRunnerCfg

  trained = tuple(HrlRunnerCfg().cadence_period_range)
  assert tuple(hrl_deploy["hrl"]["cadence_period_range"]) == trained


def test_exported_high_level_onnx_carries_the_baked_goal_scale():
  """Deploy prefers the ONNX-baked ``goal_scale`` and only falls back to deriving it from
  ``deploy.yaml``'s command ranges when the metadata is absent (State_RLHRL.cpp:217).

  This is what makes the deliberate command clamp safe. ``ang_vel_z`` is [-0.5, 0.5] here
  against a trained (-1.0, 1.0) ON PURPOSE: training wide teaches faster turning, deploy
  narrow keeps the joystick tame. The ranges are an operator knob and are NOT expected to
  match training — which is exactly why the goal scale must travel with the policy instead
  of being re-derived from them. On the legacy fallback path that same clamp would halve
  every yaw goal decode, so assert the metadata is present and the fallback unreachable."""
  import onnx

  onnx_path = DEPLOY / "config/policy/velocity_hrl/v0/exported/high_level.onnx"
  if not onnx_path.exists():
    pytest.skip("no exported high_level.onnx checked in")

  meta = {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}
  assert "goal_scale" in meta, (
    "exported high_level.onnx has no baked goal_scale -> deploy would derive it from "
    "deploy.yaml's command ranges, which do not match the trained ranges"
  )
  scale = [float(x) for x in meta["goal_scale"].split(",")]
  assert len(scale) == 7 and all(s > 0 for s in scale)


# --- ADR-0008 (Model v3): the commanded-target clip ---------------------------

ALL_DEPLOY_YAMLS = [
  DEPLOY / f"config/policy/{fam}/v0/params/{cfg}.yaml"
  for fam in ("velocity", "velocity_hrl")
  for cfg in ("deploy", "deploy_est", "deploy_est_pin0625", "deploy_real")
]


@pytest.fixture(scope="module")
def training_clip() -> dict[str, tuple[float, float]]:
  """The clip the training action term applies, keyed by anchored joint name."""
  from src.assets.robots.unitree_h1_2.h1_2_constants import H1_2_ACTION_CLIP

  return H1_2_ACTION_CLIP


def test_training_clip_covers_every_actuated_joint(training_clip, model_ranges):
  """A joint missing from the clip is trained UNCLIPPED while the robot truncates it —
  the exact asymmetry ADR-0008 exists to remove, and it would be silent."""
  clipped = {k.strip("^$") for k in training_clip}
  assert clipped == set(model_ranges), (
    f"clip/model mismatch: missing {set(model_ranges) - clipped}, "
    f"extra {clipped - set(model_ranges)}"
  )


def test_training_clip_matches_the_deploy_limit_header(training_clip, limits_header):
  """Three-way parity, leg 1: training clip == h1_2_limits.h.

  If these drift, sim trains against a different truncation than the robot applies, which
  is invisible in both places — sim keeps working and the robot just behaves wrong."""
  names, limits = limits_header
  mismatched = []
  for name, (lo, hi) in zip(names, limits):
    key = f"^{_joint_xml_name(name)}$"
    assert key in training_clip, f"{name}: not clipped in training"
    tlo, thi = training_clip[key]
    if abs(tlo - lo) > 1e-3 or abs(thi - hi) > 1e-3:
      mismatched.append(f"{name}: training ({tlo}, {thi}) vs header ({lo}, {hi})")
  assert not mismatched, "training clip drifted from the deploy header:\n" + "\n".join(
    mismatched
  )


@pytest.mark.parametrize("yaml_path", ALL_DEPLOY_YAMLS, ids=lambda p: p.parts[-3] + "/" + p.name)
def test_deploy_yaml_clip_matches_the_limit_header(yaml_path, limits_header):
  """Three-way parity, leg 2: every deploy yaml's clip == h1_2_limits.h, in joint_ids_map
  order (joint_actions.h:57 indexes ``_clip[i]`` by action index, not by name).

  All eight configs, not just the real-robot pair: a bridge config that clips differently
  from the robot reproduces the 2026-08-05 class of defect where sim and hardware disagree
  invisibly because the bridge is the only thing anyone watches."""
  cfg = yaml.safe_load(yaml_path.read_text())
  clip = cfg["actions"]["JointPositionAction"]["clip"]
  assert clip is not None, f"{yaml_path.name}: clip is null — deploy would not truncate"
  names, limits = limits_header
  assert len(clip) == len(names) == 27

  mismatched = []
  for i, (name, (lo, hi)) in enumerate(zip(names, limits)):
    assert len(clip[i]) == 2, f"{name}: clip[{i}] is not a [lo, hi] pair"
    if abs(clip[i][0] - lo) > 1e-3 or abs(clip[i][1] - hi) > 1e-3:
      mismatched.append(f"{name}: yaml {clip[i]} vs header ({lo}, {hi})")
  assert not mismatched, f"{yaml_path.name} clip drifted:\n" + "\n".join(mismatched)


def test_action_clip_excess_is_measured_before_the_clip():
  """The excess must be recomputed from ``raw_action * scale + offset``, NOT read back
  from ``processed_actions``.

  mjlab's ``BaseAction.process_actions`` overwrites ``_processed_actions`` with the CLAMPED
  value, so a term reading it back would be identically zero the moment the clip is
  configured — the penalty would silently never fire and ``--check-joint-limits`` would
  report a clean bill of health forever. That is the same measure-after-the-clamp defect
  ADR-0008 exists to fix, so it is exercised behaviourally: this stub reproduces mjlab's
  overwrite exactly, so a term that reads the clamped value returns 0.0 and fails here."""
  import torch

  from src.tasks.velocity.mdp.rewards import action_clip_excess

  class _Cfg:
    clip = {"^j$": (-1.0, 1.0)}

  class _Term:
    cfg = _Cfg()
    raw_action = torch.tensor([[2.0, -3.0]])   # -> preclip 2.0 / -3.0
    scale = 1.0
    offset = 0.0
    _clip = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    # mjlab overwrites this with the CLAMPED value; the excess is gone by reward time.
    _processed_actions = torch.tensor([[1.0, -1.0]])

  class _Env:
    num_envs = 1
    device = "cpu"

    class action_manager:
      @staticmethod
      def get_term(_name):
        return _Term()

  got = float(action_clip_excess(_Env()))
  # |2.0 - 1.0| + |-3.0 - (-1.0)| = 1.0 + 2.0
  assert got == pytest.approx(3.0), (
    f"expected the PRE-clip excess 3.0, got {got}; reading the clamped "
    f"processed_actions back would give 0.0"
  )


def test_a1_routes_the_clip_penalty_into_the_low_level_intrinsic():
  """A0's env reward term never reaches A1's low level (``ll_task_reward_coef = 0``), so
  the clip needs its own A1 routing or it is priced for A0 and free for A1 — an RQ2
  confound baked into the reward, not the architecture."""
  import inspect

  from src.tasks.velocity.config.h1_2_a1.rl_cfg import HrlRunnerCfg
  from src.tasks.velocity.rl.hrl import hrl_runner

  assert hasattr(HrlRunnerCfg(), "ll_action_clip_coef"), (
    "HrlRunnerCfg has no ll_action_clip_coef -> A1 trains with the clip unpriced"
  )
  src = inspect.getsource(hrl_runner)
  assert "self.ll_action_clip_coef" in src, "runner never reads ll_action_clip_coef"
  assert "action_clip_excess" in src, (
    "runner reads the coefficient but never calls action_clip_excess -> the term is "
    "configurable and inert"
  )


def test_the_built_env_actually_applies_the_clip(training_clip):
  """The clip constant existing is not the same as the action term USING it.

  Without this, deleting ``joint_pos_action.clip = H1_2_ACTION_CLIP`` leaves every other
  clip test green — the constant is still correct, the deploy yamls still match it — while
  training silently reverts to the unclipped v2 behaviour the ADR exists to end. Caught by
  the mutation harness as a MISSED before this test was added."""
  from mjlab.tasks.registry import load_env_cfg

  import src.tasks  # noqa: F401  (registers the task ids)

  for task in ("Unitree-H1_2-Flat", "Unitree-H1_2-Flat-A1"):
    cfg = load_env_cfg(task, play=False)
    clip = cfg.actions["joint_pos"].clip
    assert clip is not None, f"{task}: action term has no clip -> trains unclipped"
    assert clip == training_clip, f"{task}: action clip is not H1_2_ACTION_CLIP"
