"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from src.tasks.velocity.eval import probes
from src.tasks.velocity.eval.context import ProbeContext, pin_twist
from src.tasks.velocity.mdp.payload_inertia import (
  MOUNT_D_MEAN, eval_pin_params, payload_mount)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_session  # noqa: E402


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  eval_steps: int = 0
  """Headless deterministic eval: if > 0, run this many steps (no viewer) and print the
  mean per-axis velocity-tracking error (vx, vy, yaw) = |command - achieved|, the survival metrics fall rate 
  and episode length, the smoothness metric action-rate and the stability metrics orienation deviation and body height deviation.
  All of them are averaged over steps and envs. Matches the prior A1 det-eval procedure (e.g. 64 envs / 600 steps)."""
  eval_seeds: int = 1
  """Number of independent rollouts (each with a distinct seed) to average over. Reports
  mean ± std per metric across the K rollouts."""
  diagnose_goals: int = 0
  """A1-only goal-achievability probe: if > 0, run this many deterministic steps and
  decompose the velocity tracking error per HL window into the HL goal error
  |command - V*| (is the HL asking for the commanded velocity?) and the LL reach error
  |V* - achieved| (does the LL deliver the target the HL set?). Also reports raw goal
  |g| / saturation and a forward-vs-backward vx split (the observed directional bias).
  Sibling to ``eval_steps``; reuses ``eval_seeds``. Requires a hierarchical runner."""
  check_vel_increment: int = 0
  """Deploy-feasibility bench: if > 0, run this many deterministic steps and measure how
  well IMU-only integration reproduces the WITHIN-WINDOW base-velocity increment
  ``v_b(i) - v_b(t0)``. Rationale: in ``delta`` mode the LL goal obs is
  ``V* - s_i = scale*g - (s_i - s_t0)``, so absolute velocity CANCELS -- deploy never needs
  a state estimator (E1's dead end), only the increment, which resets every ``c`` steps so
  drift cannot accumulate. Reports RMS reconstruction error per axis for a term ladder
  (raw / gravity-removed / +Coriolis) against both the pelvis-origin velocity the goal
  space uses and the co-located IMU-site velocity, so the dominant error term is
  identifiable, split by yaw-rate band. Gate on the gravity-removed pelvis column: <=0.05
  ship as-is, 0.05-0.15 retrain with the residual as state_noise DR, >0.15 blind-LL sweep.
  Sibling to ``eval_steps``; reuses ``eval_seeds``."""
  check_leg_odometry: int = 0
  """Deploy-feasibility bench for the HL's ABSOLUTE base velocity (the one thing
  ``delta`` mode does NOT cancel — see ``--check-vel-increment``). Estimates
  ``v_pelvis_b = -d(p_foot_b)/dt - w_b x p_foot_b`` from the stance foot, i.e. pure
  encoder kinematics + gyro, no integration anywhere, so unlike leg-odometry *position*
  it cannot drift. Stance is picked by the gravity-projected lower foot (``LowState`` has
  no foot force sensor). Scored against true ``root_link_lin_vel_b``, split by gait phase
  and speed, plus a 6.25 Hz-filtered column (the HL fires once per ``c`` steps, so it can
  average). Sibling to ``eval_steps``; reuses ``eval_seeds``."""
  diagnose_symmetry: int = 0
  """WL-D arm 10 probe: if > 0, run this many deterministic steps and report per-foot
  touchdown counts, the t_LR/t_RL step-time distributions and symmetry index
  (SI = |t_LR-t_RL|/(t_LR+t_RL)), same-foot "double-tap" repeat counts, the realized
  double-support fraction, per-foot gait_match (not averaged over feet), and whether
  play.py's own stride_period_s metric (which divides by raw touchdown count) is
  corrupted by double-taps vs an alternation-based estimate. Sibling to ``eval_steps``;
  reuses ``eval_seeds``."""
  diagnose_cadence: int = 0
  """Cadence-transmission probe (2026-08-31): if > 0, run this many deterministic steps and
  report the HL's COMMANDED stride period (``env.hrl_period``) sampled at every HL fire --
  its within-episode mean, the per-env std ACROSS windows, and the mean absolute
  window-to-window step. Motivation: across 17 co-trained arms the commanded period spans
  0.387-0.505 s while the realized stride sits at 0.3487 +- 0.0044 s (slope +0.010,
  r=+0.085), i.e. the LL does not follow the HL's period at all, while it follows a HELD
  random period with slope +0.839 (r=+0.996). The two candidate causes -- a period that
  jitters faster than a stride vs an LL that ignores a steady command -- imply opposite
  fixes, and only the within-episode jitter separates them. ``hl_cadence_source='random'``
  is the built-in control: it draws once per episode, so its std must read ~0. Sibling to
  ``eval_steps``; reuses ``eval_seeds``."""
  diagnose_action_rate: int = 0
  """Action-rate decomposition: if > 0, run this many deterministic steps and split the
  whole-body action rate ||a_t - a_{t-1}|| by (1) position within the HL window (bin
  ``step % c``; bin 0 = the HL fire step, where the goal obs jumps) and (2) joint group
  (legs / arms / waist). Separates goal-stepping twitch (a spike at the fire step) from a
  uniform smoothness deficit. Works on A0 (non-hierarchical) too, where c defaults to 8 and
  the flat profile is the control proving any A1 structure is real, not a binning artifact.
  Sibling to ``eval_steps``; reuses ``eval_seeds``."""
  reward_terms: bool = False
  """With ``--eval-steps``: the env reward's per-term value (``weight * raw``, per step), mean
  over envs/steps/seeds and split into walking / standing by the commanded-motion gate
  (|cmd_xy| + |cmd_yaw| > 0.1, ``bench_flight_recorder.walking_mask``). Read from the live
  RewardManager, so it is the training reward. Prints ``[REWARD] {json}`` and adds
  ``reward_terms`` to ``[BENCH]``. A1's LL trains on its intrinsic reward, so for A1 this is
  the DIAGNOSTIC env reward. Compared against bridge/hardware by ``score_env_reward.py``."""
  reward_dump: str = ""
  """With ``--reward-terms``: also write ``<path>.npz`` (post-step qpos/qvel/ctrl, command, gait
  phase and the live per-term reward of the first 8 envs, seed 0) and ``<path>.mjb`` (the compiled
  host model), the ground truth that ``score_env_reward.py --validate`` replays offline."""
  check_joint_limits: bool = False
  """Deploy parity check: with ``--eval-steps``, score the policy's COMMANDED joint
  position targets (``action_manager['joint_pos'].processed_actions``, the same quantity
  the C++ writes to ``motor_cmd().q()``) against the hardware position limits in
  ``deploy/robots/h1_2/include/h1_2_limits.h``. On hardware those bounds are enforced by a
  per-joint command clamp (``State_RLBase.cpp``), so a target past them is silently
  truncated and the joint never does what the policy intended. In sim nothing clamps the
  target, so the same policy can train against a trajectory it can never execute on the
  robot, and the defect is invisible until the flight recorder reports it. Reports the
  per-joint over-limit rate and worst overshoot in radians, directly comparable to
  ``safety_analyzer.py``'s ``raw_policy_violations`` on a real session. Costs one tensor
  compare per step; no extra rollout."""
  probe_lean: int = 0
  """Backward-lean sensitivity probe (ADR-0006 / WL-B0e): if > 0, run this many
  deterministic standing steps per swept bias value and report the steady-state base
  pitch, measuring d(lean)/d(encoder offset) in sim instead of inferring it. Injects a
  DETERMINISTIC ``encoder_bias`` on a leg joint group (bypassing the startup event's
  random sampling) and flips the ``joint_pos`` obs term to ``biased=True``. That flag is
  the whole probe: with it off (the training default) mjlab shows the policy the TRUE
  joint angle, so gravity and encoders never disagree and the bias is nulled by a constant
  action shift. Pair with ``--eval-cmd-vx 0`` to hold the commanded stand."""
  probe_lean_bias: str = "0,0.01,0.02,0.0317,-0.0317"
  """Comma-separated ``encoder_bias`` values (rad) to sweep. mjlab applies
  ``target = action - encoder_bias``, so a bias b reproduces the deploy-side ADR-0006
  ``joint_offset`` of -b; both are printed so the result maps onto Spec B directly."""
  probe_gravity_noise: str = ""
  """ADR-0006 discriminator: if set (comma-separated Unoise half-ranges, e.g.
  "0.05,0.1,0.2,0.4"), sweep corruption of the `projected_gravity` OBSERVATION instead of
  the encoder bias, holding the encoder clean. Separates the two readings of Spec A's
  hardware failure: if the biased-obs policy degrades faster than the keeper under the same
  gravity corruption, it really did shift its trust onto the IMU (the IMU-fragility
  reading); if both degrade alike, the failure was out-of-distribution encoder bias
  instead. 0.05 = the training default, i.e. the control. Requires --probe-lean."""
  probe_lean_joints: str = "ankle_pitch,knee,hip_pitch,all"
  """Comma-separated leg groups to place the bias on. ``all`` splits the swept value
  equally over the three sagittal joints so the CHAIN SUM equals it -- the quantity the
  feet-flat FK actually pins."""
  eval_cadence_period: float | None = None
  """A1a: pin the HL-commanded stride period (s) during eval, for the CoT(period) sweep
  (requires an hl_cadence runner). None -> the runner's mid-range default."""
  eval_cmd_vx: float | None = None
  """A1a: pin the commanded forward velocity (m/s) for the fixed-velocity stride-period sweep /
  replay. When set, the twist command is held constant (vy/wz default 0, standing+heading off),
  so varying --eval-cadence-period isolates the commanded period from the natural v->period map."""
  eval_cmd_vy: float | None = None
  eval_cmd_wz: float | None = None
  eval_cmd_heading: float | None = None
  """Hold a world-frame heading (rad; 0 = world +x) via the env's heading P-controller, so
  fixed-command replays walk STRAIGHT instead of slowly circling (the A0-inherited yaw drift
  curves the path when wz is just pinned to 0). Requires --eval-cmd-vx; overrides eval_cmd_wz
  (wz becomes the live heading correction, clipped to the task's ang_vel_z range)."""
  eval_cmd_seq: str = ""
  """Replay a sim-bridge command schedule at the bench: a ``bridge_session.py --seq`` spec
  (``KEY:SECONDS,...``; ``default`` = its ``DEFAULT_SEQ``, the 253 s battery). Every env steps
  the same schedule, so the bench and a bridge Run are one experiment on two plants. Sets
  ``eval_steps`` to the schedule length. The inline ``[BENCH]`` numbers then POOL regimes and
  are not the result: write ``--eval-trace`` and score it with ``bench_flight_recorder.py
  --trace``, the scorer the bridge and the robot go through."""
  eval_trace: str = ""
  """With ``--eval-steps``/``--eval-cmd-seq``: write ``<path>_s<seed>.npz``, one row per policy
  step per env in the flight recorder's quantities and DEPLOY slot order (command, commanded
  target, measured q/dq, actuator torque, IMU-body attitude and gyro, true base velocity, the
  observed gait phase, first-fall step; with ``--reward-terms`` also the per-term env reward)."""
  out: str = ""
  """Any probe: also write its `[TAG]` result to this JSON file, with what identifies the run:
  the checkpoint's md5, the git commit (and whether the tree was dirty), the task and every flag.
  The `[TAG]` line alone lives in terminal scrollback, which is how a number loses its run."""
  eval_payload_kg: float | None = None
  """WP1 payload pilot (2026-08-27): pin an exact torso payload mass (kg, added at the COM)
  for the whole eval rollout, instead of the training-time sampled range. Reuses the existing
  ``base_mass`` DR mechanism (``dr.body_mass``, same function as WL-D arm 8's wide-DR bundle)
  with a degenerate ``ranges=(kg, kg)`` -- the same pin idiom as --eval-cadence-period. None
  (default) -> no event added, byte-identical to current behavior. The checkpoint under test
  was NOT trained with payload DR, so this is a zero-shot generalization probe: does the
  optimal stride period shift with an injected payload the policy never saw."""
  cf_e_col: str | None = None
  """WP5 counterfactual-e disambiguator (2026-09-02). Index into the privileged latent
  ``e`` = (0 payload, 1 com_dx, 2 com_dy, 3 com_dz, 4 friction) whose value fed to the HL
  is FALSIFIED, leaving the physics untouched. Requires --cf-e-val and --diagnose-goals.

  Why this exists: the pinned-payload sweep measures corr(commanded T, true latent), which
  a memoryless policy responding proprioceptively to how the robot actually moves would
  reproduce exactly. It cannot tell "the HL reads e" from "the HL feels the consequences
  of e". Forcing a wrong value into the column the HL READS, while every body/geom
  property stays as sampled, separates them: if T moves, the HL is using the observation;
  if T does not, the correlation was proprioceptive and the privileged latent is inert.
  All envs are forced to the SAME value, so the true per-env spread becomes identical
  noise in every arm and the between-arm contrast stays clean."""
  cf_e_val: str | None = None
  """Value(s) forced into --cf-e-col of the HL's e observation. Physics UNCHANGED.
  Stay inside the trained DR range (payload 0-12 kg, com_d* +-0.05 m, friction 0.3-1.6):
  the HL's obs normalizer was fitted on that support, so an out-of-range value measures
  extrapolation, not sensitivity.

  Both flags accept a COMMA-SEPARATED list of equal length, which falsifies several
  columns at once (2026-09-02). That is required whenever the deployment couples two
  components: a backpack of mass m at horizontal offset d from the torso CoM shifts it
  by ``dx = m*d/(M_torso + m)`` (``M_torso`` = 17.789 kg), so payload and com_dx move
  TOGETHER on the robot and never independently as the two DR events sample them.
  Falsifying payload alone measures a direction the hardware cannot realize."""
  hl_obs_e_source: Literal["true", "phi"] = "true"
  """WP5 Phase 2's sim-side Phase-3-lite comparison (2026-09-07): 'true' (default,
  byte-identical) feeds the HL the privileged env_latent_e as always; 'phi' swaps in
  phi's reconstructed history estimate instead -- the sim-only decision input for
  whether Phase 3 (fine-tuning the HL against phi's imperfect estimate) is needed.
  Requires --phi-dir. Never touches deploy/ or the ONNX export path -- phi runs as a
  torch module directly, mirroring the C++ decode/cold-start math for realism."""
  phi_dir: str | None = None
  """Directory from train_phi.py holding phi_real.pt + norm.json (with a z_cold key).
  Required when --hl-obs-e-source phi."""
  video: bool = False
  video_length: int = 1000
  video_height: int = 1080 #| None = None
  video_width: int = 1920 #| None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  export_onnx: tyro.conf.UseCounterAction[int] = 0
  """Export the loaded checkpoint to policy.onnx next to the checkpoint file and exit.
  Bare flag, no value needed (counter type sidesteps mjlab's global
  FlagConversionOff, which would otherwise demand ``--export-onnx True``)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _deep_merge(default: dict, override: dict) -> dict:
  """Nested dict merge: ``override`` wins where present, ``default`` fills the rest.

  Needed because the dumped agent.yaml is written AFTER runner construction, and
  model construction pops keys from nested cfg dicts in place (e.g. MLPModel pops
  ``distribution_cfg["class_name"]``) — so saved cfgs can be missing keys that a
  rebuild requires. Older runs may also predate newly added cfg fields.
  """
  out = dict(default)
  for k, v in override.items():
    if isinstance(v, dict) and isinstance(out.get(k), dict):
      out[k] = _deep_merge(out[k], v)
    else:
      out[k] = v
  return out


def restore_run_structure(agent_cfg, env_cfg, resume_path: Path) -> None:
  """Rebuild ``agent_cfg`` (+ ``env_cfg``'s goal dim) with the structure the checkpoint at
  ``resume_path`` was trained with, not the task's current defaults: a checkpoint trained
  with a different goal space crashes the load (dim mismatch), and a different
  ``hl_algorithm`` silently builds the wrong high level (e.g. oracle instead of the
  trained ppo/td3, whose state is then ignored). ``train.py`` dumps the launch config to
  ``params/agent.yaml``; restore the structure-determining keys from there. Keys absent
  from the yaml (older runs) or not on the cfg (e.g. A0) keep the defaults. Shared by
  ``play.py`` and ``collect_phi_data.py`` (WP5 Phase 2), which both load frozen HRL
  checkpoints outside their task's bare-default structure.
  """
  params_yaml = resume_path.parent / "params" / "agent.yaml"
  if not params_yaml.exists():
    return
  saved = yaml.full_load(params_yaml.read_text())  # dump_yaml writes python/tuple tags
  structure_keys = ("c", "goal_components", "goal_weights", "hl_algorithm",
                    "hl_ppo", "hl_td3", "relabeling", "gamma_hi", "hl_target_mode",
                    "hl_obs_vel", "hl_obs_e",
                    # A1a cadence channel: without hl_cadence restored, eval rebuilt the
                    # runner with the channel OFF -> fixed 0.6 clock, --eval-cadence-period
                    # silently inert (the 2026-07-02 "no entrainment" false verdicts).
                    "hl_cadence", "cadence_period_range", "ll_cadence_coef",
                    "hl_cot_coef", "hl_cadence_source", "cadence_swing_time",
                    "cadence_duty_range", "hl_velocity_goals_only")
  restored = {k: saved[k] for k in structure_keys
              if k in saved and hasattr(agent_cfg, k)}
  # hl_velocity_goals_only defaulted to True on 2026-07-09: checkpoints saved
  # before the key existed trained full-goal HLs, so ABSENCE must restore False
  # (the "keys absent keep the defaults" rule would silently rebuild a task_dim
  # HL and crash/mis-load every pre-change TD3 checkpoint).
  if "hl_velocity_goals_only" not in saved and hasattr(agent_cfg, "hl_velocity_goals_only"):
    restored["hl_velocity_goals_only"] = False
  # Same absence rule for hl_obs_vel (defaulted to True 2026-07-10): checkpoints saved
  # before the key existed trained HLs without the +2 vel obs -> absence must restore
  # False or every pre-velobs TD3 checkpoint mis-builds 94-dim nets vs its saved 92.
  if "hl_obs_vel" not in saved and hasattr(agent_cfg, "hl_obs_vel"):
    restored["hl_obs_vel"] = False
  for k, v in restored.items():
    cur = getattr(agent_cfg, k)
    if isinstance(v, dict) and cur is not None and not isinstance(cur, dict):
      v = _deep_merge(asdict(cur), v)  # cur is a cfg dataclass (hl_ppo / hl_td3)
    setattr(agent_cfg, k, v)
  # The env's goal obs dim was baked from the default runner cfg at task
  # registration; re-derive it from the restored component list.
  if "goal_components" in restored and "goal" in env_cfg.observations:
    from src.tasks.velocity.rl.hrl.goal_space import goal_dim

    env_cfg.observations["goal"].terms["goal"].params["dim"] = goal_dim(
      tuple(restored["goal_components"])
    )
  if restored:
    print(f"[INFO]: Restored run structure from {params_yaml.name}: "
          f"{ {k: v for k, v in restored.items() if k in ('goal_components', 'hl_algorithm')} }")


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # WP1 payload pilot: pin an exact torso payload for the whole eval rollout (see
  # --eval-payload-kg docstring). Reuses dr.body_mass with a degenerate range instead of
  # adding a parallel mass mechanism.
  _cf_cols: list[int] = []
  _cf_vals: list[float] = []
  if cfg.cf_e_col is not None:
    if cfg.cf_e_val is None:
      raise SystemExit("--cf-e-col requires --cf-e-val (the value(s) to falsify it to).")
    _cf_cols = [int(c) for c in str(cfg.cf_e_col).split(",")]
    _cf_vals = [float(v) for v in str(cfg.cf_e_val).split(",")]
    if len(_cf_cols) != len(_cf_vals):
      raise SystemExit(f"--cf-e-col has {len(_cf_cols)} entries but --cf-e-val has "
                       f"{len(_cf_vals)}; they must pair up.")
    if len(set(_cf_cols)) != len(_cf_cols):
      raise SystemExit("--cf-e-col repeats a column; the last write would silently win.")
    if not all(0 <= c <= 4 for c in _cf_cols):
      raise SystemExit("--cf-e-col must index e = (0 payload, 1 com_dx, 2 com_dy, "
                       "3 com_dz, 4 friction).")
    if cfg.diagnose_goals <= 0:
      raise SystemExit("--cf-e-col only applies to the --diagnose-goals probe (the one "
                       "path that writes obs['hl_e'] itself).")
  if cfg.eval_payload_kg is not None:
    if "base_com" not in env_cfg.events:
      raise SystemExit("--eval-payload-kg: task has no 'base_com' event to borrow the "
                        "torso asset_cfg from.")
    # ADR-0010: the pin means "the RIG, loaded to m kg" -- mass, CoM shift and inertia at the
    # mean mount lever arm, i.e. a point ON the deployment ray. Pinning mass alone would put
    # eval physics outside the training distribution (payload with no CoM shift is a
    # configuration the coupled DR never samples and the hardware cannot produce).
    env_cfg.events.pop("base_mass", None)
    env_cfg.events["payload_mount"] = EventTermCfg(
      mode="startup",
      func=payload_mount,
      params=eval_pin_params(
        env_cfg.events["base_com"].params["asset_cfg"], cfg.eval_payload_kg),
    )
    _k = list(env_cfg.events)
    if _k.index("base_com") > _k.index("payload_mount"):
      raise SystemExit(f"--eval-payload-kg: payload_mount must follow base_com; order {_k}.")
    print(f"[WP1] Pinned mounted payload = {cfg.eval_payload_kg} kg at d={MOUNT_D_MEAN}.")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

    # Rebuild the runner with the structure the checkpoint was trained with, not the
    # current task defaults (see restore_run_structure's docstring).
    restore_run_structure(agent_cfg, env_cfg, resume_path)

  # Every headless eval/probe reports a DISTRIBUTIONAL mean, so the env count is part of the
  # measurement, not a perf knob. At 1 env a single command draw is the whole sample and the
  # result is bimodal: the same A0 checkpoint returned jacc_legs 24.75/25.14/25.81/26.23 and
  # 42.27 across five draws (+-26%), vs +-2.6% at 64 envs. The old `None -> scene default (1)`
  # silently produced non-comparable numbers twice (the 2026-07-15 goal probe, then the
  # 2026-08-09 jacc benches), so eval paths default to 64 while interactive play stays at 1.
  seq_cmds = None
  if cfg.eval_cmd_seq:
    seq_cmds = bridge_session.seq_commands(
      bridge_session.DEFAULT_SEQ if cfg.eval_cmd_seq == "default" else cfg.eval_cmd_seq)
    cfg = replace(cfg, eval_steps=len(seq_cmds))
  _eval_mode = (cfg.eval_steps > 0 or cfg.diagnose_goals > 0 or cfg.diagnose_symmetry > 0
                or cfg.diagnose_action_rate > 0 or cfg.check_vel_increment > 0
                or cfg.diagnose_cadence > 0)
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  elif _eval_mode:
    env_cfg.scene.num_envs = 64
    print("[INFO]: eval/probe mode without --num-envs -> defaulting to 64 "
          "(1 env is not a usable sample; pass --num-envs explicitly to override)")
  if _eval_mode and env_cfg.scene.num_envs < 8:
    print(f"[WARN]: num_envs={env_cfg.scene.num_envs} for an eval/probe run. Results are "
          "NOT comparable to the 64-env references and swing ~+-26% between repeats. "
          "Use --num-envs 64 for any number you intend to report.")
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  if cfg.probe_lean > 0:
    # Wire the OBSERVATION half of the encoder bias (see --probe-lean). Training leaves it
    # off, which makes the bias observable and therefore trivially correctable; the probe
    # needs the hardware failure mode, where the reported angle IS the command and only the
    # true joint is displaced. "critic" shares the same term objects as "actor".
    # A1 renames "actor" -> "policy" and deletes "actor", so scan every group by term
    # name rather than by group name, or the probe silently measures nothing on A1.
    wired = []
    for gname, grp_cfg in env_cfg.observations.items():
      term = grp_cfg.terms.get("joint_pos")
      if term is not None:
        term.params = dict(term.params or {})
        term.params["biased"] = True
        wired.append(gname)
    # Print it: this wiring IS the probe, and a silent miss (A1 renames the group) makes
    # the probe report a spurious zero sensitivity rather than failing.
    print(f"[LEANPROBE] joint_pos biased=True wired on obs groups: {wired}")
    assert wired, "no obs group has a joint_pos term -- probe would measure nothing"

  if cfg.probe_lean > 0 and cfg.probe_gravity_noise:
    # Play mode sets enable_corruption=False (h1_2 env_cfgs.py), which makes the obs manager
    # null every term's noise -- so the gravity sweep would have nothing to vary. Turn
    # corruption back on but strip the noise from every OTHER actor term, so gravity is the
    # only corrupted channel and the sweep stays single-variable.
    for gname, grp in env_cfg.observations.items():
      if gname == "critic" or "projected_gravity" not in grp.terms:
        continue
      grp.enable_corruption = True
      for name, term in grp.terms.items():
        if name != "projected_gravity":
          term.noise = None

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  # WP5 Phase 2 gave HierarchicalRunner.get_inference_policy an optional 2nd arg (`dones`,
  # for phi's history-buffer reset). A flat or dummy policy's callable takes `obs` only and
  # reads a 2nd positional as `masks` -> crash. Set True below once `policy` exists and the
  # runner is hierarchical.
  policy_wants_dones = False
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )

    # Absence shim: checkpoints saved before 2026-07-16 carry no baked `goal_scale`
    # (HierarchicalRunner.save), so pin the scale the run TRAINED with. Deriving it live
    # instead reads THIS env's command ranges — play mode narrows ang_vel_z (yaw scale 0.5
    # vs 1.0 trained) and `--eval-cmd-vx` collapses them to a point, which drops the derived
    # scale to its 1e-3 floor and pins V* = s + 1e-3*g ≈ s, inerting the whole goal channel
    # (the 2026-07-15 "HL hold degeneracy" artifact; see doc/hrl/A1_findings.md WL-C row).
    # Training ranges = the task's non-play cfg with the twist curriculum's stages applied.
    gsp = getattr(runner, "goal_space", None)
    if gsp is not None and not gsp.scale_frozen:
      train_cfg = load_env_cfg(task_id)  # non-play => the ranges training ran with
      train_ranges = train_cfg.commands["twist"].ranges
      for term in train_cfg.curriculum.values():
        for stage in (term.params or {}).get("velocity_stages", []):
          for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            if stage.get(axis) is not None:
              setattr(train_ranges, axis, stage[axis])
      live = env.unwrapped.command_manager.get_term("twist").cfg.ranges
      live_ranges = (live.lin_vel_x, live.lin_vel_y, live.ang_vel_z)
      # Reuse GoalSpace's own derivation against the training ranges, then restore.
      live.lin_vel_x, live.lin_vel_y, live.ang_vel_z = (
        train_ranges.lin_vel_x, train_ranges.lin_vel_y, train_ranges.ang_vel_z
      )
      gsp.freeze_scale(gsp.scale(env.unwrapped).clone())
      live.lin_vel_x, live.lin_vel_y, live.ang_vel_z = live_ranges
      print(f"[SHIM] no baked goal_scale in checkpoint -> pinned to the training value "
            f"{[round(x, 4) for x in gsp.scale(env.unwrapped).tolist()]}")

    # Export mode: write the ONNX policy/policies next to the checkpoint and exit.
    if cfg.export_onnx:
      assert log_dir is not None
      if hasattr(runner, "export_hierarchy_to_onnx"):
        # A1 hierarchy: two nets (HL fires every c steps -> goal -> LL).
        runner.export_hierarchy_to_onnx(str(log_dir) + os.sep)
        print(f"[INFO] Exported: {log_dir}/low_level.onnx (+ high_level.onnx unless oracle)")
        deploy = (
          "~/ramlab_ws/src/unitree_rl_mjlab/deploy/robots/h1_2/"
          "config/policy/velocity_hrl/v0/exported/"
        )
        print(f"[INFO] Copy to deploy:\n"
              f"  cp {log_dir}/low_level.onnx {log_dir}/high_level.onnx {deploy}")
      else:
        runner.export_policy_to_onnx(str(log_dir), filename="policy.onnx")
        onnx_path = log_dir / "policy.onnx"
        print(f"[INFO] Exported: {onnx_path}")
        deploy = (
          "~/ramlab_ws/src/unitree_rl_mjlab/deploy/robots/h1_2/"
          "config/policy/velocity/v0/exported/policy.onnx"
        )
        print(f"[INFO] Copy to deploy:\n  cp {onnx_path} {deploy}")
      env.close()
      return

    if cfg.eval_cadence_period is not None and hasattr(runner, "eval_cadence_period"):
      runner.eval_cadence_period = cfg.eval_cadence_period
      print(f"[A1a] Fixed stride period = {cfg.eval_cadence_period} s.")

    if cfg.hl_obs_e_source == "phi":
      if not getattr(runner, "hl_obs_e", False):
        raise SystemExit("--hl-obs-e-source phi: this checkpoint was not trained with "
                          "hl_obs_e -- there is no true-e channel to swap out.")
      if cfg.phi_dir is None:
        raise SystemExit("--hl-obs-e-source phi requires --phi-dir "
                          "(train_phi.py's output directory).")
      from phi_data import load_phi_for_inference
      phi_model, phi_norm = load_phi_for_inference(cfg.phi_dir, device)
      print(f"[INFO] hl_obs_e_source=phi: loaded {cfg.phi_dir}/phi_real.pt")
      policy = runner.get_inference_policy(device=device, phi_model=phi_model, phi_norm=phi_norm)
    else:
      policy = runner.get_inference_policy(device=device)
    policy_wants_dones = hasattr(runner, "hl")  # only the hierarchical closure accepts (obs, dones)

    # A1a fixed-command eval/replay: pin the twist command so a stride-period sweep isolates the
    # commanded period from the natural velocity->period mapping. Collapse ranges to a point +
    # disable standing/heading, on the live term (works in benchmark, viewer, and diagnose loops).
    if cfg.eval_cmd_vx is not None:
      vx, vy, wz = cfg.eval_cmd_vx, (cfg.eval_cmd_vy or 0.0), (cfg.eval_cmd_wz or 0.0)
      pin_twist(env, vx, vy, wz, cfg.eval_cmd_heading)
      if cfg.eval_cmd_heading is not None:
        print(f"[A1a] Fixed twist = ({vx} m/s, {vy} m/s, heading-hold {cfg.eval_cmd_heading} rad); walks straight.")
      else:
        print(f"[A1a] Fixed twist command = ({vx}, {vy}, {wz}) m/s; resampling collapsed to a point.")


  if DUMMY_MODE:
    runner = None
  probe = probes.select(cfg)
  if probe is not None:
    result = probe.run(ProbeContext(
      cfg=cfg, env=env, runner=runner, policy=policy, policy_wants_dones=policy_wants_dones,
      resume_path=resume_path, agent_cfg=agent_cfg, device=device, dummy=DUMMY_MODE,
      seq_cmds=seq_cmds, cf_cols=tuple(_cf_cols), cf_vals=tuple(_cf_vals)))
    if cfg.out:
      import hashlib, json, subprocess
      git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
      Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
      Path(cfg.out).write_text(json.dumps({
        "probe": probe.__name__.rsplit(".", 1)[-1], "result": result, "task": task_id,
        "checkpoint": str(resume_path) if resume_path else None,
        "checkpoint_md5": hashlib.md5(resume_path.read_bytes()).hexdigest() if resume_path else None,
        "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "flags": asdict(cfg)}, indent=1, default=str))
      print(f"[INFO] Wrote {cfg.out}")
    env.close()
    return

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
