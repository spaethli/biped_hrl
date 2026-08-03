#!/usr/bin/env python3
"""Sensitivity check for tests/: reintroduce each historical defect, confirm it is caught.

A regression test written against already-correct code is worthless unless it can be
shown to fail against the bug it claims to guard. Each MUTATION below is a source edit
that recreates a real past defect; the suite MUST go red, and the named test must be
among the failures.

Run after adding tests or changing the code they guard:
    python scripts/check_test_sensitivity.py [substring-of-test-name]

It edits source files in place and restores them. It REFUSES to restore if a file
changed underneath it (concurrent session), leaving the mutation in place and saying
so, rather than silently reverting someone else's work.
"""
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
GS = REPO / "src/tasks/velocity/rl/hrl/goal_space.py"
RW = REPO / "src/tasks/velocity/mdp/rewards.py"
HR = REPO / "src/tasks/velocity/rl/hrl/hrl_runner.py"
LIM = REPO / "deploy/robots/h1_2/include/h1_2_limits.h"
YML = REPO / "deploy/robots/h1_2/config/policy/velocity_hrl/v0/params/deploy.yaml"
REAL_YML = REPO / "deploy/robots/h1_2/config/policy/velocity_hrl/v0/params/deploy_real.yaml"

# (label, file, old, new, test that must fail)
MUTATIONS = [
  ("goal scale re-derived live at inference (WL-C, 2026-07-16)", GS,
   "    if self._frozen_scale is not None:\n      return self._frozen_scale\n", "",
   "test_frozen_scale_survives_a_command_range_collapse"),

  ("task_only drops the nominal posture pin (posture sag, 2026-07-09)", GS,
   "    if not task_only:\n", "    if True:\n",
   "test_task_only_pins_posture_to_nominal_not_to_a_sagged_state"),

  ("exp orientation term back to a quadratic penalty (suicide attractor, 2026-07-06)", GS,
   'r = r + torch.exp(-d.square().sum(-1) / 0.25)\n      elif c.name == "height"',
   'r = r - d.square().sum(-1)\n      elif c.name == "height"',
   "test_exp_kernel_is_strictly_positive_over_the_reachable_state_space"),

  ("relabel goal left unclamped (illegal HL action)", GS,
   ".clamp(-1.0, 1.0)", "",
   "test_to_g_clamps_unreachable_goals_into_the_bounded_action_range"),

  ("absolute mode decodes against the state, not the center", GS,
   'ref = self.center(env) if mode == "absolute" else state\n      return ref',
   'ref = state\n      return ref',
   "test_absolute_decode_ignores_the_current_state"),

  ("task-ordering guard removed (silent mis-slice)", GS,
   'raise ValueError("task goal components must precede non-task ones.")',
   "pass",
   "test_task_components_must_precede_non_task_ones"),

  ("push-off direction gate removed (toe-lift rewarded, WL-D 2026-07-20)", RW,
   "directed_power = power * (qd > 0).float()", "directed_power = power",
   "test_pushoff_rewards_plantarflexion_not_a_toe_lift"),

  ("push-off contact gate removed (rewards airborne thrash)", RW,
   "gate = is_stance & (phi >= (1.0 - w)) & is_contact",
   "gate = is_stance & (phi >= (1.0 - w))",
   "test_pushoff_requires_ground_contact"),

  ("d(T) duty cap removed (duty above human slow-walk)", RW,
   ".clamp(threshold, duty_max)", ".clamp(threshold, 10.0)",
   "test_dt_duty_schedule_is_clamped_between_the_floor_and_human_slow_walk"),

  ("commanded-cadence clock ignored (dead gait phase, WL-B 2026-07-21)", RW,
   "global_phase = env.hrl_phase.unsqueeze(1)",
   "global_phase = torch.zeros_like(env.hrl_phase).unsqueeze(1)",
   "test_commanded_phase_follows_the_hrl_clock_not_the_episode_clock"),

  ("warm start treats the command gap as trailing (naive leading-block copy)", HR,
   "return list(range(start)) + list(range(start + width, src_in))",
   "return list(range(src_in - width))",
   "test_warm_start_maps_proprio_around_the_command_gap"),

  ("obs-normalizer stats not remapped (policy fed mis-scaled obs)", HR,
   '      "obs_normalizer._mean",\n      "obs_normalizer._var",\n      "obs_normalizer._std",\n',
   "",
   "test_warm_start_maps_the_normalizer_stats_the_same_way"),

  ("warm-start width mismatch no longer raises", HR,
   "if len(src_cols) != shared:", "if False:",
   "test_warm_start_fails_loudly_on_a_column_width_mismatch"),

  ("a deploy joint limit hand-entered from docs (WL-B0, 15/27 wrong 2026-07-20)", LIM,
   "{-0.43f,   0.43f  },  //  0 LEFT_HIP_YAW",
   "{-0.52f,   0.43f  },  //  0 LEFT_HIP_YAW",
   "test_deploy_joint_limits_match_the_training_model"),

  ("deploy PD gain drifts from h1_2_constants.py", YML,
   "stiffness: [200.0, 200.0, 200.0, 300.0,", "stiffness: [250.0, 200.0, 200.0, 300.0,",
   "test_deploy_gains_match_the_training_constants"),

  ("deploy action scale back to a flat 0.25 (pre-ADR-0005)", YML,
   "scale: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.166667,",
   "scale: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,",
   "test_deploy_action_scale_follows_the_derived_rule"),

  ("deploy default pose drifts from the training home keyframe", YML,
   "default_joint_pos: [0,-0.2,0,0.5,-0.3,0,", "default_joint_pos: [0,-0.35,0,0.5,-0.3,0,",
   "test_deploy_default_pose_matches_the_training_home_keyframe"),

  ("an observation term renamed in deploy.yaml (obs vector permuted)", YML,
   "    gait_phase_cmd:\n", "    gait_phase_typo:\n",
   "test_deploy_yaml_observation_order_matches_training"),

  ("deploy cadence range drifts from agent.yaml (cadence silently rescaled)", YML,
   "cadence_period_range: [0.35, 1.0]", "cadence_period_range: [0.35, 1.3]",
   "test_deploy_cadence_period_range_matches_training"),

  ("joint_offset (ADR-0006 Spec B) truncated to 26 entries", REAL_YML,
   "joint_offset: [0,0,0,0,0,0, 0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]",
   "joint_offset: [0,0,0,0,0,0, 0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0]",
   "test_joint_offset_is_27_long_and_inert_when_absent"),

  ("joint_offset (ADR-0006) lands on the waist slot — a slot typo shifting the held pose",
   REAL_YML,
   "joint_offset: [0,0,0,0,0,0, 0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]",
   "joint_offset: [0,0,0,0,0,0, 0,0,0,0,0,0, 0.0317, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]",
   "test_joint_offset_touches_only_leg_joints"),

  ("joint_offset (ADR-0006) decimal slip: 0.12 rad (6.9 deg) instead of 0.012", REAL_YML,
   "joint_offset: [0,0,0,0,0,0, 0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]",
   "joint_offset: [0,0.12,0,0,0,0, 0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]",
   "test_joint_offset_magnitude_is_physically_sane"),

  ("joint_offset (ADR-0006 Spec B) leaks into a sim config", YML,
   "# keep their old yaml unchanged.\nhrl:",
   "# keep their old yaml unchanged.\n"
   "joint_offset: [0,0,0,0,0,0, 0,0,0,0,0,0, 0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]\nhrl:",
   "test_joint_offset_absent_from_sim_configs"),
]


def run_suite() -> set[str]:
  p = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header", "--tb=no", "--color=no"],
    cwd=REPO, capture_output=True, text=True,
  )
  if "error" in p.stdout.lower() and " passed" not in p.stdout:
    return {"<COLLECTION-ERROR>"}
  return set(re.findall(r"FAILED tests/\S+::(\w+)", p.stdout))


def main() -> int:
  only = sys.argv[1] if len(sys.argv) > 1 else None
  baseline = run_suite()
  if baseline:
    print(f"BASELINE NOT GREEN: {sorted(baseline)}")
    return 1
  print("baseline: green\n")

  rows, bad = [], 0
  for label, path, old, new, expect in MUTATIONS:
    if only and only not in expect:
      continue
    src = path.read_text()
    if old not in src:
      rows.append(("SKIP", label, "mutation text not found - source drifted"))
      bad += 1
      continue
    mutated = src.replace(old, new, 1)
    path.write_text(mutated)
    try:
      failed = run_suite()
    finally:
      # Another session may be editing this repo. Only restore if the file still holds
      # exactly what we wrote - otherwise our copy is stale and writing it back would
      # silently revert their work.
      if path.read_text() == mutated:
        path.write_text(src)
      else:
        print(f"\nABORT: {path.name} changed underneath the mutation harness.\n"
              f"NOT restoring (would clobber a concurrent edit). The file currently\n"
              f"holds a MUTATED '{label}'. Restore by hand from git.")
        return 2
    if expect in failed:
      rows.append(("CAUGHT", label, f"{expect} (+{len(failed) - 1} other)"))
    else:
      rows.append(("MISSED", label, f"{expect} did NOT fail; failures={sorted(failed)}"))
      bad += 1

  print(f"{'verdict':8s} {'defect reintroduced':62s} detail")
  for v, label, detail in rows:
    print(f"{v:8s} {label:62s} {detail}")
  print(f"\n{sum(1 for r in rows if r[0] == 'CAUGHT')}/{len(rows)} defects caught")
  return 1 if bad else 0


if __name__ == "__main__":
  sys.exit(main())
