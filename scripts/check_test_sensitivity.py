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
OBS = REPO / "src/tasks/velocity/mdp/observations.py"
EC = REPO / "src/tasks/velocity/config/h1_2/env_cfgs.py"
TD3 = REPO / "src/tasks/velocity/rl/hrl/td3.py"
SN = REPO / "src/tasks/velocity/rl/hrl/state_noise.py"
LO = REPO / "src/tasks/velocity/rl/hrl/leg_odom.py"
LIM = REPO / "deploy/robots/h1_2/include/h1_2_limits.h"
HRLC = REPO / "deploy/robots/h1_2/src/State_RLHRL.cpp"
RBE = REPO / "scripts/replay_base_estimators.py"
YML = REPO / "deploy/robots/h1_2/config/policy/velocity_hrl/v0/params/deploy.yaml"
REAL_YML = REPO / "deploy/robots/h1_2/config/policy/velocity_hrl/v0/params/deploy_real.yaml"
DGA = REPO / "scripts/deploy_gate_analyzer.py"
PRV = REPO / "scripts/deploy_provenance.py"
BSN = REPO / "scripts/bridge_session.py"
PPS = REPO / "scripts/period_payload_stats.py"
PI = REPO / "src/tasks/velocity/mdp/payload_inertia.py"
BFR = REPO / "scripts/bench_flight_recorder.py"
SJH = REPO / "scripts/score_joint_hold.py"
HRLH = REPO / "deploy/robots/h1_2/include/FSM/State_RLHRL.h"
ENCH = REPO / "deploy/robots/h1_2/include/hrl/adapt_encoder.h"
A0CPP = REPO / "deploy/robots/h1_2/src/State_RLBase.cpp"
IOCH = REPO / "deploy/robots/h1_2/include/obs_contract.h"
ACS = REPO / "scripts/analyze_cadence_settling.py"

# (label, file, old, new, test that must fail)
MUTATIONS = [
  # ADR-0004 S1c: `CoT = energy / walked distance` is a ratio whose denominator the HL
  # controls, so uncapped it pays for walking FURTHER than commanded. Measured overshoot at
  # hl_cot_coef=5: +0.152 m/s at cmd 0.25 (61%), hl_err 0.198 > ll_err 0.117 (2026-09-05).
  ("CoT denominator credits distance beyond the command (overshoot pays)", HR,
   "torch.minimum(win_dist, win_cmd)", "torch.maximum(win_dist, win_cmd)",
   "test_cap_on_makes_overshoot_strictly_worse"),

  ("CoT commanded-distance cap silently disabled (flag becomes a no-op)", HR,
   "torch.minimum(win_dist, win_cmd) if cap_commanded else win_dist", "win_dist",
   "test_cap_on_credits_nothing_beyond_the_command"),

  # ADR-0009: the deploy clip is kept, so it stays pinned to h1_2_limits.h, and the yaml must
  # not carry a duplicate key (yaml-cpp keeps the FIRST, PyYAML the LAST -- three A0 hardware
  # sessions on 2026-08-27 ran clipped while the config read `clip: null`).
  ("deploy clip drifts from the limit header (left_ankle_pitch)", YML,
   "[-0.897334, 0.523598],", "[-1.2, 0.9],",
   "test_deploy_yaml_clip_matches_the_limit_header"),

  ("`clip: null` appended below the table (silent no-op on the robot, 2026-08-27)", YML,
   "    ]\n    joint_names: [.*]", "    ]\n    clip: null\n    joint_names: [.*]",
   "test_deploy_yaml_clip_matches_the_limit_header"),

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

  # Trip thresholds sized off the STAND (36 Nm / 0.06 rad/s) instead of measured WALKING --
  # the mistake the 2026-08-05 brief started from. Holds a hip mid-stride every step.
  ("torque trip threshold sized off the stand, not walking (2026-08-05)", LIM,
   "{ 265.0f,  11.0f},  //  2 LEFT_HIP_ROLL",
   "{  50.0f,  11.0f},  //  2 LEFT_HIP_ROLL",
   "test_trip_thresholds_clear_measured_walking"),

  # The A0-only calibration that held right_hip_yaw during the A1 candidate's own gait.
  ("hip-yaw trip back to the hardware-only (A0-dominated) bound, which the A1 candidate's "
   "bridge gait exceeds (2026-08-05)", LIM,
   "{ 230.0f,  10.0f},  //  0 LEFT_HIP_YAW",
   "{  60.0f,   6.0f},  //  0 LEFT_HIP_YAW",
   "test_trip_thresholds_clear_measured_walking"),

  # The opposite error: headroom raised until the failure it guards slips through.
  ("hip-roll trip raised above the splay peak it exists to catch (2026-08-05)", LIM,
   "{ 265.0f,  11.0f},  //  8 RIGHT_HIP_ROLL",
   "{ 400.0f,  20.0f},  //  8 RIGHT_HIP_ROLL",
   "test_trip_thresholds_still_catch_the_2026_08_05_splay"),

  # The 2026-08-05 hardware defect itself: the real config carried neither estimator key,
  # so both defaulted false and the goal-space state came from the dead rt/sportmodestate.
  ("real config back to sourcing base velocity from the dead rt/sportmodestate "
   "(2026-08-05 leg splay)", REAL_YML,
   "  base_vel_from_imu: true",
   "  # base_vel_from_imu: true",
   "test_real_config_never_sources_the_goal_state_from_sportmodestate"),

  ("real config back to sourcing base height from the dead rt/sportmodestate "
   "(2026-08-05 leg splay)", REAL_YML,
   "  base_height_from_fk: true",
   "  # base_height_from_fk: true",
   "test_real_config_never_sources_the_goal_state_from_sportmodestate"),

  # The 2026-08-06 half of the same defect class, one level up: `s` is well-formed but the
  # HIGH level's absolute velocity is not. Under base_vel_from_imu the increment is 0 at
  # every window start, which is every HL fire, so removing this key hands the HL a constant
  # 0.000000 velocity reading (bridge A/B: 3/3 falls vs 0/3 on ground truth).
  ("real config leaves the high level with no live absolute velocity source "
   "(2026-08-05 est-mode bridge blocker)", REAL_YML,
   "  hl_vel_from_leg_odom: true",
   "  # hl_vel_from_leg_odom: true",
   "test_real_config_gives_the_high_level_a_live_velocity_source"),

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

  # --- readiness pipeline (2026-08-04). Each mutation is a defect that actually shipped.
  ("analyzer dt back to median(diff(t)) — the ZeroDivisionError on quantised timestamps", DGA,
   "  return float(t[-1] - t[0]) / max(len(t) - 1, 1) if len(t) > 1 else 0.0",
   "  return float(np.median(np.diff(t))) if len(t) > 1 else 0.0",
   "test_row_dt_survives_the_quantised_timestamps_that_crashed_the_analyzer"),

  ("stride_proxy divides by a zero dt instead of returning NaN", DGA,
   "  if not (dt > 0) or not np.isfinite(dt):\n    return float(\"nan\")",
   "  if False:\n    return float(\"nan\")",
   "test_stride_proxy_returns_nan_rather_than_raising_on_a_degenerate_segment"),

  ("transition windows stop reporting falls (the 0.5->0 decel fall goes unseen)", DGA,
   '"fell": bool(fell_mask[m].any()),', '"fell": False,',
   "test_transition_window_catches_a_fall_on_the_deceleration"),

  ("HL-only velocity jitter mutates its input view in place (2026-08-05 leg-odometry "
   "probe) -- would silently corrupt state_n through shared storage, since hrl_runner.py "
   "passes state_n[:, 0:2], a VIEW, not a copy", SN,
   "    return v_est + self.bias + self.sample",
   "    v_est += self.bias + self.sample\n    return v_est",
   "test_hl_vel_jitter_does_not_mutate_its_input_view"),

  # The pre-fix code verbatim: any failing hold won (sticky-False) AND post-fall holds were
  # scored, so a robot that walked 7.1 m and then fell was reported as "band never
  # released" -- a real FALL masked as a harness artifact, i.e. NO-GO downgraded to
  # "this run tells you nothing".
  ("band check: sticky-False accumulator + post-fall holds scored (fall masked as INFRA)", DGA,
   "    if post_fall:\n      continue  # explained by the fall, not by the harness\n"
   "    band_ok = True if travel >= BAND_MIN_TRAVEL_M else (band_ok or False)",
   "    band_ok = (band_ok is not False) and travel >= BAND_MIN_TRAVEL_M",
   "test_band_check_ignores_holds_after_a_fall"),

  ("stand drift reports magnitude only, losing the backward/left direction", DGA,
   '"heading_deg": round(float(np.degrees(np.arctan2(dy, dx))), 1),',
   '"heading_deg": 0.0,',
   "test_stand_drift_reports_direction_not_just_magnitude"),

  ("joint clamp reports rate but not magnitude past the stop", DGA,
   '"max_past_stop_rad": round(float(over[:, i].max()), 4),',
   '"max_past_stop_rad": 0.0,',
   "test_joint_clamp_reports_magnitude_past_the_stop_not_only_rate"),

  ("HL dim expectation ignores hl_obs_vel (94 vs 92 mismatch reaches the C++)", PRV,
   '"high_level.onnx": 92 + (2 if hrl.get("hl_obs_vel") else 0),',
   '"high_level.onnx": 92,',
   "test_expected_dims_track_hl_obs_vel_and_goal_components"),

  ("reverse md5 index keeps only one path per hash (cannot name every holding run)", PRV,
   "    index.setdefault(md5(onnx_path), []).append(onnx_path)",
   "    index[md5(onnx_path)] = [onnx_path]",
   "test_index_maps_one_hash_to_every_run_that_holds_it"),

  # Must remove EVERY qualifying hold, not just the first line's -- the trailing
  # '5:12' also satisfies >=10 s at >=0.3, so a partial mutation leaves the test
  # correctly passing (it did, on the first attempt).
  ("the default bridge sequence loses its long hold (band check silently skipped)", BSN,
   "DEFAULT_SEQ = ('0:8,3:30,0:6,5:30,0:6,w:30,0:8,'\n"
   "               's:15,0:6,a:15,0:6,d:15,0:6,'\n"
   "               'q:15,0:6,e:15,0:6,'\n"
   "               '5:12,q:8,0:10')",
   "DEFAULT_SEQ = '0:8,3:5,0:6,5:5,0:6,w:5,0:8'",
   "test_band_thresholds_are_reachable_by_the_default_sequence"),

  # --- WL-F simulated leg odometry (2026-08-09). The estimator is a transcription of the
  # deployed C++; a silently different one teaches the policy to compensate for an error the
  # robot does not have, and sim would look fine throughout. These are the ways it can go
  # subtly wrong without crashing.
  ("stance foot chosen as the one HIGHER along gravity (swing leg differenced)", LO,
   "  stance = (along[:, 0] < along[:, 1]).long()",
   "  stance = (along[:, 0] >= along[:, 1]).long()",
   "test_stance_selection_is_the_foot_further_along_gravity"),

  ("the w x p term dropped (on hardware it carries most of the estimate)", LO,
   "  return -(ps - ps_prev) / dt - torch.cross(w_P, ps, dim=-1)",
   "  return -(ps - ps_prev) / dt",
   "test_omega_cross_p_term_is_present"),

  ("leg FK y-mirror lost — the right leg built as a second left leg", LO,
   '  sy = 1.0 if leg == 0 else -1.0',
   "  sy = 1.0",
   "test_matches_cpp_leg_odom_velocity_on_golden_vectors"),

  ("hip roll/pitch rotation order swapped in the FK chain", LO,
   "  R = R @ _rot(1, j[:, 1])\n  R = R @ _rot(0, j[:, 2])",
   "  R = R @ _rot(0, j[:, 2])\n  R = R @ _rot(1, j[:, 1])",
   "test_matches_cpp_leg_odom_velocity_on_golden_vectors"),

  # --- estimator bench (WL-G, docs/adr/0007) ------------------------------------------
  # The isolation invariant is what makes it defensible to run the known-divergent
  # position-only arm inside the 1 kHz loop, so a passive arm reaching the policy must fail.
  ("a PASSIVE estimator arm is routed into obs['hl_vel'] instead of the selected one", HRLC,
   "            lo_sum_ += out[est_arm_].cast<float>();",
   "            lo_sum_ += out[2].cast<float>();",
   "test_only_the_selected_arm_can_reach_the_hl_velocity"),

  ("arm A's accumulation loses its selection guard (double-counts when another arm runs)", HRLC,
   "            if (est_arm_ == 0) {          // selected: byte-identical to the pre-bench build",
   "            if (true) {",
   "test_only_the_selected_arm_can_reach_the_hl_velocity"),

  # The golden fixture is the only thing tying the C++ arms to every published number.
  ("the complementary filter's time constant drifts from the scored value", RBE,
   "def arm_b_complementary(s: Session, pre: dict, v_legodom: np.ndarray, tau: float = 0.20):",
   "def arm_b_complementary(s: Session, pre: dict, v_legodom: np.ndarray, tau: float = 0.25):",
   "test_python_arm_matches_golden_on_the_c8_window"),

  ("foot site offset dropped (the sole point substituted for the site point)", LO,
   "  return p + R @ _vec(offset, q)",
   "  return p",
   "test_matches_cpp_leg_odom_velocity_on_golden_vectors"),

  ("reset keeps the previous foot positions (episode boundary differenced as velocity)", LO,
   "    self.valid[env_ids] = False",
   "    self.valid[env_ids] = True",
   "test_reset_drops_the_cross_episode_difference"),

  # obs["hl_vel"] must be latched at the fire and held, like hl_vel_lo_ (State_RLHRL.cpp:444).
  # Refreshing it mid-window decorrelates the TD3 buffer's next_s from the value the actor
  # actually conditioned its action on -- the same reasoning that forbids a fresh jitter draw.
  ("obs['hl_vel'] recomputed mid-window instead of holding the latched value", HR,
   '      v = (uenv.leg_odom.fire() if fire else uenv.leg_odom.value).clone()',
   "      v = uenv.leg_odom.fire().clone()",
   "test_leg_odom_value_is_held_across_the_window_and_only_moves_on_a_fire"),

  ("obs['hl_vel'] hands out the accumulator's own storage (aliasing)", HR,
   '      v = (uenv.leg_odom.fire() if fire else uenv.leg_odom.value).clone()',
   "      v = uenv.leg_odom.fire() if fire else uenv.leg_odom.value",
   "test_leg_odom_output_does_not_alias_the_accumulator"),

  # Anchored on the RETURN line too: `if self.hl_vel_source == "leg_odom":` also appears in
  # __init__ (the config guards), and this harness replaces only the first occurrence — so
  # the bare condition would mutate the guard, which no CPU test exercises, and read green.
  ("the leg-odom source silently falls back to the goal state", HR,
   '    if self.hl_vel_source == "leg_odom":\n'
   '      v = (uenv.leg_odom.fire() if fire else uenv.leg_odom.value).clone()',
   "    if False:\n"
   "      v = (uenv.leg_odom.fire() if fire else uenv.leg_odom.value).clone()",
   "test_leg_odom_source_never_reads_or_writes_the_goal_state"),

  # WP2 (2026-08-31): env_latent_e reads payload/CoM as absolute values instead of
  # deltas from the un-randomized default -- would read the torso's nominal mass as
  # "payload" even with DR off.
  ("env_latent_e drops the payload/CoM default subtraction", OBS,
   "  payload = env.sim.model.body_mass[:, torso_gid] - mass_default[torso_gid]\n"
   "  com_delta = env.sim.model.body_ipos[:, torso_gid, :] - ipos_default[torso_gid]",
   "  payload = env.sim.model.body_mass[:, torso_gid]\n"
   "  com_delta = env.sim.model.body_ipos[:, torso_gid, :]",
   "test_env_latent_e_reads_payload_and_com_as_deltas_and_friction_as_absolute"),

  # WP2: payload DR must be opt-in -- registering base_mass unconditionally on the base
  # A0 task would shift the RNG stream for every event after it, even at a degenerate
  # range, breaking the inertness gate for every existing A0/A1 checkpoint.
  ("payload DR applied unconditionally to the base A0 task", EC,
   "  cfg.sim.njmax = 300",
   "  cfg.sim.njmax = 300\n  apply_payload_dr(cfg)",
   "test_apply_payload_dr_is_opt_in_not_on_the_base_tasks"),

  # WP2: obs_e_dim dropped from the HL's state-dim math -- obs["hl_e"] would still be
  # concatenated in _state_vec but the actor/critic/normalizer would be built one
  # column too narrow, a shape mismatch WP5's Phase 1 would hit at construction.
  ("obs_e_dim omitted from HighLevelTd3's state-dim math", TD3,
   "obs[\"policy\"].shape[-1] + obs[\"command\"].shape[-1] + obs_vel_dim + obs_e_dim",
   "obs[\"policy\"].shape[-1] + obs[\"command\"].shape[-1] + obs_vel_dim",
   "test_obs_e_dim_appends_hl_e_and_changes_state_dim"),

  # WP5 (2026-08-31): freezing an LL under WP2's widened critic obs group. The fix is
  # narrow and has an over-fix on BOTH sides -- reverting it blocks Phase 1 entirely,
  # while skipping the ACTOR silently freezes a random LL that still trains and scores.
  ("frozen-LL critic load back to strict (blocks WP5 Phase 1 on any pre-WP2 checkpoint)", HR,
   '    try:\n      critic.load_state_dict(ck["critic_state_dict"], strict=True)\n    except RuntimeError:',
   '    if True:\n      critic.load_state_dict(ck["critic_state_dict"], strict=True)\n    if False:',
   "test_frozen_ll_loads_actor_even_when_critic_obs_widened"),

  ("frozen-LL ACTOR load skipped -- freezes a randomly-initialised LL, silently", HR,
   '    actor.load_state_dict(ck["actor_state_dict"], strict=True)',
   '    pass  # MUTATION: actor never loaded',
   "test_frozen_ll_still_raises_on_actor_mismatch"),

  ("frozen-LL critic blanket-skipped even when shapes agree (over-fix)", HR,
   '      critic.load_state_dict(ck["critic_state_dict"], strict=True)',
   '      pass  # MUTATION: critic never loaded',
   "test_frozen_ll_loads_critic_when_shapes_agree"),

  # bench_flight_recorder (2026-09-01): scoring a hardware flight recorder into the sim
  # [BENCH] schema. Each defect below either silently changes what a key MEANS while it
  # keeps a sim name, or breaks the rate/alignment every derived number rests on.
  ("commanded span back to min..max (one transient excursion read as authority)", BFR,
   "        p1, p50, p99 = np.percentile(x, [1, 50, 99])",
   "        p1, p50, p99 = x.min(), np.median(x), x.max()",
   "test_ankle_roll_span_reproduces_adr_0009"),

  ("headroom measured to the FARTHER bound (inverts the clip-vs-penalty verdict)", BFR,
   '"headroom": round(float(min(j["max"] - p99, p1 - j["min"])), 4),',
   '"headroom": round(float(max(j["max"] - p99, p1 - j["min"])), 4),',
   "test_headroom_is_distance_to_the_nearer_bound"),

  ("pinned_any becomes a per-joint MEAN (~3x smaller, reads as a different robot)", BFR,
   "    return float((((np.abs(x - lo) < 1e-9) | (np.abs(x - hi) < 1e-9)).any(axis=1)).mean())",
   "    return float((((np.abs(x - lo) < 1e-9) | (np.abs(x - hi) < 1e-9))).mean())",
   "test_pinned_any_leg_is_a_union_not_a_mean"),

  ("regime gate back to a 3-vector norm (pure-yaw walking misread as standing)", BFR,
   "    total_command = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])",
   "    total_command = np.linalg.norm(cmd, axis=1)",
   "test_regime_gate_matches_the_reward_command_threshold"),

  ("regime gate back to score_joint_hold's 1e-6 stillness gate (a different question)", BFR,
   "CMD_THRESHOLD = 0.1 ", "CMD_THRESHOLD = 1e-6 ",
   "test_regime_gate_matches_the_reward_command_threshold"),

  ("alignment drops the clock-RATE term (constant offset: 40 steps off over 308 s)", BFR,
   "        m, b = np.polyfit(ts[keep], ls[keep], 1, w=cs[keep])",
   "        m, b = 0.0, float(np.average(ls[keep], weights=cs[keep]))",
   "test_affine_alignment_recovers_an_injected_clock_skew"),

  ("policy-step decode back to a fixed row stride (a stride of 20 spans TWO steps)", SJH,
   "    changed = np.any(np.diff(raw, axis=0) != 0.0, axis=1)\n    idx = np.flatnonzero(np.concatenate(([True], changed)))",
   "    idx = np.arange(0, len(raw), 10)",
   "test_policy_step_decode_tracks_changes_not_a_row_count"),

  ("a BENCH key silently dropped from the partition (neither emitted nor documented)", BFR,
   '    "height_dev": "body_height is in the dead sportmodestate block (identically zero)",\n',
   "",
   "test_every_bench_key_is_emitted_renamed_or_documented_as_omitted"),

  # bench_flight_recorder pairing/alignment (2026-09-14): the 2026-09-07 A1a session broke
  # both assumptions the tool had inherited from how the 2026-08-27 runs were operated.
  ("alignment coarse stage removed (a -10.3 s offset is outside the +-8 s fine search)", BFR,
   "    b0 = coarse_lag(t_f, knee_f, t_a, knee_a) or 0.0",
   "    b0 = 0.0",
   "test_alignment_on_real_session_pairs"),

  ("alignment rejection back to a fixed 0.5 s gate (bends round a transient: -1648 ppm)", BFR,
   "        refit = (np.abs(r) < max(0.05, 3.0 * mad)) & (cs > 0.7)",
   "        refit = (np.abs(r) < 0.5) & (cs > 0.7)",
   "test_alignment_on_real_session_pairs"),

  ("pairing back to a forward start-time window (encodes operator timing, broke at +125 s)", BFR,
   "        overlap = (min(end_f, end_a) - max(start_f, start_a)).total_seconds()",
   "        overlap = 120.0 - (start_a - start_f).total_seconds()",
   "test_find_pair_selects_by_recording_interval_overlap"),

  ("missing --traj-dir globs to nothing and reads as 'no pair for this session'", BFR,
   "    if not d.is_dir():",
   "    if False:",
   "test_missing_traj_dir_fails_loudly"),

  # A0 deploy I/O contract (2026-09-15). algorithms.h sizes the ORT tensor from the ONNX and
  # reads the obs vector's data pointer, so a wide model reads adjacent HEAP -> limp robot. The
  # A0 slot is filled by hand, and until now nothing checked it.
  ("A0 contract violation only WARNS at load, so a wrong-width policy.onnx still runs", A0CPP,
   '            throw std::runtime_error("[A0] REFUSING TO LOAD: " + violation);',
   '            spdlog::warn("[A0] " + violation);',
   "test_the_a0_contract_is_checked_at_load_and_refuses"),

  ("A0 contract check never called from the constructor", A0CPP,
   "        const std::string violation = h1_2::io_contract_violation(",
   "        const std::string violation = std::string(); (void)h1_2::io_contract_violation;(",
   "test_the_a0_contract_is_checked_at_load_and_refuses"),

  ("contract accepts a model WIDER than the obs vector (the heap-read direction)", IOCH,
   "        if (built != in.size)",
   "        if (built > in.size)",
   "test_the_cpp_contract_unit_test_passes"),

  ("contract never compares the output width (process_action slices past the end)", IOCH,
   "    if (onnx_output_size != action_dim)",
   "    if (false)",
   "test_the_cpp_contract_unit_test_passes"),

  # analyze_cadence_settling capture coverage (2026-09-15): np.interp CLAMPS, so rows past the
  # end of a capture inherit the last captured speed and a policy that came to rest reads as
  # never settling. Latent on 2026-09-14; ADR-0013's floors rest on this metric.
  ("settling scored over segments the capture does not cover (the clamp bias)", ACS,
   "    base_segs = [(i0, i1) for i0, i1 in segs if cov[i0:i1 + 1].all()]",
   "    base_segs = list(segs)",
   "test_a_segment_the_capture_does_not_cover_is_dropped_not_censored"),

  ("coverage tolerance unbounded, so any row counts as covered", ACS,
   "  return near <= MOCAP_GAP_S",
   "  return near <= 1e9",
   "test_covered_rows_rejects_an_internal_dropout"),

  ("NaN capture samples reach the smoother and smear across its window", ACS,
   "  ok = np.isfinite(t) & np.isfinite(speed)",
   "  ok = np.ones(len(t), bool)",
   "test_read_mocap_drops_non_finite_samples"),

  # bench_flight_recorder Run scoping (2026-09-15): one CSV per controller PROCESS, so a raw
  # log can hold several Runs under different experimental conditions (15-00-37.csv: a broom
  # push and two 7.5 kg payload Runs). Pooling them returns a confident number for no
  # experiment that was run, and nothing in the file name says so.
  ("a multi-Run flight recorder file is pooled instead of refused", BFR,
   "        elif len(seen) > 1:",
   "        elif False:",
   "test_a_multi_run_file_is_refused_rather_than_pooled"),

  ("--entry filters a mask but never slices, so positional indexing stays on the full file",
   BFR,
   '            df = df[df["entry"] == entry].reset_index(drop=True)',
   "            pass",
   "test_entry_selects_exactly_that_run"),

  # bench_flight_recorder energy gating (2026-09-15): np.interp CLAMPS, so telemetry recorded
  # after a Run that ENDED WALKING inherited the last gate (1) and the last speed. On
  # 2026-09-14_14-48-55 that credited 45.12 m against a true 11.5 m -- CoT 0.166 where the
  # session runs 0.55-0.68, the single most flattering number of the day, and it was fiction.
  ("Run-span mask dropped: trailing telemetry inherits the clamped gate and earns distance",
   BFR,
   " & inside\n",
   "\n",
   "test_energy_ignores_telemetry_recorded_outside_the_run"),

  ("Run-span mask applied to the telemetry's RAW clock, not the aligned one (the two differ "
   "by -3.2 to +1028 s, so it excludes real samples)", BFR,
   "    inside = (t_in_flight >= tf[0]) & (t_in_flight <= tf[-1])",
   "    inside = (ta >= tf[0]) & (ta <= tf[-1])",
   "test_energy_is_unchanged_when_the_telemetry_fits_inside_the_run"),


  # WP5 (2026-09-01): the Bar B statistic itself. None of these crash -- each returns a
  # plausible number and the thesis verdict is read off it.
  ("Bar B pools by WINDOW not by ENV (inflates n ~150x, shrinks every interval)", PPS,
   '    "period_sd_env": round(Te.std(unbiased=True).item(), 5),',
   '    "period_sd_env": round(per_w[keep_w].std(unbiased=True).item(), 5),',
   "test_aggregation_unit_is_the_env_not_the_window"),

  ("Bar B ignores the keep mask (reset-contaminated windows enter the per-env mean)", PPS,
   "  kf = keep_w.float()",
   "  kf = torch.ones_like(keep_w, dtype=torch.float)",
   "test_keep_mask_excludes_contaminated_windows"),

  ("Bar B reports r=0.0 instead of NaN when a regressor has no variance", PPS,
   '    return float("nan") if d < 1e-12 else round((xc @ yc).item() / d, 4)',
   '    return 0.0 if d < 1e-12 else round((xc @ yc).item() / d, 4)',
   "test_no_payload_variance_gives_nan_not_zero"),

  ("Bar B mis-slices e: com_dz reads the friction column, so the partial regression "
   "attributes the HL's response to the wrong physical cause", PPS,
   '                "com_dz": [round(v, 5) for v in com[ok, 2].tolist()]},',
   '                "com_dz": [round(v, 5) for v in fric[ok].tolist()]},',
   "test_per_env_com_columns_carry_the_right_latent_slots"),

  ("Bar B leaves the CoM columns UNMASKED, so per-env rows de-align from payload/period "
   "and every partial regression silently pairs the wrong envs", PPS,
   '                "com_dx": [round(v, 5) for v in com[ok, 0].tolist()],',
   '                "com_dx": [round(v, 5) for v in com[:, 0].tolist()],',
   "test_per_env_com_columns_carry_the_right_latent_slots"),

  # --- WP5d: H-adapt's deploy path (2026-09-02) --------------------------------------
  # phi's contract crosses two languages with no runtime error if the two sides disagree:
  # algorithms.h sizes the ORT input from the ONNX shape and never compares it against the
  # vector that was built, so a mismatch reads adjacent HEAP and the robot goes limp via the
  # safety hold. Each mutation below is one way that agreement can silently rot.

  ("a duplicate hl_obs_e appended below the hrl: block (yaml-cpp keeps the FIRST)", YML,
   "  hl_obs_e: false", "  hl_obs_e: false\nhl_obs_e: true",
   "test_hl_obs_e_appears_exactly_once"),

  ("hl_obs_e ships ON by default (a pre-WP5d checkpoint would then refuse to load)", YML,
   "  hl_obs_e: false", "  hl_obs_e: true",
   "test_hl_obs_e_is_inside_the_hrl_block_and_defaults_off"),

  ("phi's layout string swaps the policy/command widths", HR,
   'f"policy{policy_dim}+command{command_dim}"', 'f"policy{command_dim}+command{policy_dim}"',
   "test_phi_frame_is_the_deploy_policy_and_command_widths"),

  ("E_NAMES drifts from the order env_latent_e concatenates", OBS,
   'E_NAMES = ("payload_kg", "com_dx", "com_dy", "com_dz", "friction")',
   'E_NAMES = ("payload_kg", "friction", "com_dx", "com_dy", "com_dz")',
   "test_e_names_match_the_latent_the_env_actually_builds"),

  ("the exporter stops writing a key the deploy loader reads (z_cold)", HR,
   '      "z_cold": list(z_cold),\n', "",
   "test_export_writes_every_metadata_key_the_deploy_reads"),

  ("the cold-start prior is no longer checked against the clamp (friction 0 = frictionless)", HR,
   "      if not lo <= c <= hi:", "      if False:",
   "test_export_refuses_a_cold_start_prior_outside_the_clamp"),

  ("phi's input width is no longer tied to policy+command", HR,
   "    if input_dim != policy_dim + command_dim:", "    if False:",
   "test_export_refuses_an_input_dim_that_is_not_policy_plus_command"),

  ("a normalisation scale that rounds to 0.000 is written anyway (mutes the channel)", HR,
   "      if v != 0.0 and round(v, 3) == 0.0:", "      if False:",
   "test_export_refuses_a_scale_that_the_metadata_cannot_represent"),

  ("z bounds no longer have to cover every latent component", HR,
   "    if not (len(z_clip_lo) == len(z_clip_hi) == len(z_cold) == n):", "    if False:",
   "test_export_refuses_bounds_that_do_not_cover_every_component"),

  ("the encoder session runs unguarded (heap read -> limp robot)", HRLC,
   '                    if (!check_input("adapt_encoder.onnx", hist_win_.size(), enc_in_dim_))\n'
   "                        return;\n", "",
   "test_every_ort_session_is_length_checked_before_it_runs"),

  ("the runtime dimension guard THROWS from the policy thread (std::terminate = fail-dark)", HRLH,
   "        if (!dim_fault_) {\n            dim_fault_ = true;",
   "        if (!dim_fault_) {\n            throw std::runtime_error(\"dim\");",
   "test_the_runtime_guard_latches_instead_of_throwing"),

  ("the load-time contract check degrades to a warning instead of refusing", HRLC,
   "            throw std::runtime_error(\n"
   '                "[HRL] adapt_encoder.onnx: " + bad',
   '            spdlog::warn("[HRL] adapt_encoder.onnx: {}", bad); if (false) throw std::runtime_error(\n'
   '                "[HRL] adapt_encoder.onnx: " + bad',
   "test_the_load_time_contract_check_is_actually_called"),

  ("phi runs on a partially filled history (an input shape it never saw in training)", ENCH,
   "        if (!full()) return false;\n", "",
   "test_window_refuses_a_partial_buffer"),

  # --- ADR-0010: the coupled mounted-payload event (2026-09-02) -----------------------
  # The first is the silent class: nothing crashes, the CoM shift just disappears and `e`
  # keeps reporting a plausible number.

  ("payload event reads the DEFAULT com, so the base_com residual is silently erased "
   "(the Operation.add overwrite class)", PI,
   "  com = model.body_ipos[env_ids, bid, :]",
   '  com = env.sim.get_default_field("body_ipos")[bid].to(env.device).expand(n, 3)',
   "test_payload_shift_composes_on_top_of_the_base_com_residual"),

  ("rank-1 inertia left about the body ORIGIN: the parallel-axis shift back to the new "
   "CoM is dropped", PI,
   "  inertia_new = i_o_new - mass_new[:, None, None] * (cc_new * eye - c_outer_new)",
   "  inertia_new = i_o_new",
   "test_rank1_update_matches_the_independent_closed_form"),

  ("--eval-payload-kg pins MASS ONLY: eval physics becomes payload with no CoM shift, a "
   "configuration training never samples and the hardware cannot produce", PI,
   '    "d_ranges": tuple((v, v) for v in MOUNT_D_MEAN),',
   '    "d_ranges": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),',
   "test_eval_pin_reproduces_what_training_samples_at_that_mass"),

  ("payload mass never written back, so `e` reports a 0 payload beside a real CoM shift", PI,
   "  model.body_mass[env_ids, bid] = mass_new",
   "  model.body_mass[env_ids, bid] = mass",
   "test_env_latent_e_reports_realized_totals_after_the_coupled_event"),
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
