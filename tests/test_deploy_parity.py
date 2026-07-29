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
