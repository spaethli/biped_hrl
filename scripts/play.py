"""Script to play RL agent with RSL-RL."""

import math
import os
import sys
from dataclasses import asdict, dataclass
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
  diagnose_action_rate: int = 0
  """Action-rate decomposition: if > 0, run this many deterministic steps and split the
  whole-body action rate ||a_t - a_{t-1}|| by (1) position within the HL window (bin
  ``step % c``; bin 0 = the HL fire step, where the goal obs jumps) and (2) joint group
  (legs / arms / waist). Separates goal-stepping twitch (a spike at the fire step) from a
  uniform smoothness deficit. Works on A0 (non-hierarchical) too, where c defaults to 8 and
  the flat profile is the control proving any A1 structure is real, not a binning artifact.
  Sibling to ``eval_steps``; reuses ``eval_seeds``."""
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
  eval_payload_kg: float | None = None
  """WP1 payload pilot (2026-08-27): pin an exact torso payload mass (kg, added at the COM)
  for the whole eval rollout, instead of the training-time sampled range. Reuses the existing
  ``base_mass`` DR mechanism (``dr.body_mass``, same function as WL-D arm 8's wide-DR bundle)
  with a degenerate ``ranges=(kg, kg)`` -- the same pin idiom as --eval-cadence-period. None
  (default) -> no event added, byte-identical to current behavior. The checkpoint under test
  was NOT trained with payload DR, so this is a zero-shot generalization probe: does the
  optimal stride period shift with an injected payload the policy never saw."""
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
  if cfg.eval_payload_kg is not None:
    if "base_com" not in env_cfg.events:
      raise SystemExit("--eval-payload-kg: task has no 'base_com' event to borrow the "
                        "torso asset_cfg from.")
    env_cfg.events["base_mass"] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.body_mass,
      params={
        "asset_cfg": env_cfg.events["base_com"].params["asset_cfg"],
        "operation": "add",
        "ranges": (cfg.eval_payload_kg, cfg.eval_payload_kg),
      },
    )
    print(f"[WP1] Pinned torso payload = {cfg.eval_payload_kg} kg.")

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
    # current task defaults: a checkpoint trained with a different goal space crashes
    # the load (dim mismatch), and a different hl_algorithm silently builds the wrong
    # high level (e.g. oracle instead of the trained ppo/td3, whose state is then
    # ignored). train.py dumps the launch config to params/agent.yaml; restore the
    # structure-determining keys from there. Keys absent from the yaml (older runs)
    # or not on the cfg (e.g. A0) keep the defaults.
    params_yaml = resume_path.parent / "params" / "agent.yaml"
    if params_yaml.exists():
      saved = yaml.full_load(params_yaml.read_text())  # dump_yaml writes python/tuple tags
      structure_keys = ("c", "goal_components", "goal_weights", "hl_algorithm",
                        "hl_ppo", "hl_td3", "relabeling", "gamma_hi", "hl_target_mode",
                        "hl_obs_vel",
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

  # Every headless eval/probe reports a DISTRIBUTIONAL mean, so the env count is part of the
  # measurement, not a perf knob. At 1 env a single command draw is the whole sample and the
  # result is bimodal: the same A0 checkpoint returned jacc_legs 24.75/25.14/25.81/26.23 and
  # 42.27 across five draws (+-26%), vs +-2.6% at 64 envs. The old `None -> scene default (1)`
  # silently produced non-comparable numbers twice (the 2026-07-15 goal probe, then the
  # 2026-08-09 jacc benches), so eval paths default to 64 while interactive play stays at 1.
  _eval_mode = (cfg.eval_steps > 0 or cfg.diagnose_goals > 0 or cfg.diagnose_symmetry > 0
                or cfg.diagnose_action_rate > 0 or cfg.check_vel_increment > 0)
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
    for grp in ("actor", "critic"):
      term = env_cfg.observations.get(grp)
      term = term.terms.get("joint_pos") if term is not None else None
      if term is not None:
        term.params = dict(term.params or {})
        term.params["biased"] = True

  if cfg.probe_lean > 0 and cfg.probe_gravity_noise:
    # Play mode sets enable_corruption=False (h1_2 env_cfgs.py), which makes the obs manager
    # null every term's noise -- so the gravity sweep would have nothing to vary. Turn
    # corruption back on but strip the noise from every OTHER actor term, so gravity is the
    # only corrupted channel and the sweep stays single-variable.
    grp = env_cfg.observations.get("actor")
    if grp is not None:
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

    policy = runner.get_inference_policy(device=device)

    # A1a fixed-command eval/replay: pin the twist command so a stride-period sweep isolates the
    # commanded period from the natural velocity->period mapping. Collapse ranges to a point +
    # disable standing/heading, on the live term (works in benchmark, viewer, and diagnose loops).
    if cfg.eval_cmd_vx is not None:
      vx, vy, wz = cfg.eval_cmd_vx, (cfg.eval_cmd_vy or 0.0), (cfg.eval_cmd_wz or 0.0)
      t = env.unwrapped.command_manager.get_term("twist")
      t.cfg.ranges.lin_vel_x = (vx, vx)
      t.cfg.ranges.lin_vel_y = (vy, vy)
      t.cfg.rel_standing_envs = 0.0
      t.vel_command_b[:, 0], t.vel_command_b[:, 1], t.vel_command_b[:, 2] = vx, vy, wz
      t.is_standing_env[:] = False
      if cfg.eval_cmd_heading is not None:
        # Heading hold: wz becomes a live P-correction toward the world heading. Keep the
        # ang_vel_z range OPEN — the controller clips its correction to it, so collapsing
        # it to (0,0) would silently disable the hold.
        h = cfg.eval_cmd_heading
        t.cfg.heading_command = True
        t.cfg.ranges.heading = (h, h)
        t.heading_target[:] = h
        t.is_heading_env[:] = True
        print(f"[A1a] Fixed twist = ({vx} m/s, {vy} m/s, heading-hold {h} rad); walks straight.")
      else:
        t.cfg.ranges.ang_vel_z = (wz, wz)
        t.cfg.heading_command = False
        t.is_heading_env[:] = False
        print(f"[A1a] Fixed twist command = ({vx}, {vy}, {wz}) m/s; resampling collapsed to a point.")

  # Headless deterministic benchmark: roll out `eval_steps` steps x `eval_seeds` seeds.
  if cfg.eval_steps > 0:
    import json
    uenv = env.unwrapped
    robot = uenv.scene["robot"].data
    n_envs = uenv.num_envs
    tm = uenv.termination_manager
    has_fell = "fell_over" in tm.active_terms
    # Upper-body deploy-hygiene metrics (ADR-0002): arms+waist drift-from-default + speed.
    ub_ids, _ = uenv.scene["robot"].find_joints(
      [".*shoulder.*", ".*elbow.*", ".*wrist.*", ".*torso.*"])
    ub_ids = torch.as_tensor(ub_ids, device=robot.joint_pos.device)
    # Leg/arm-restricted smoothness+accel diagnostics (the deploy blocker is smoothness,
    # and action_rate is arm-dominated -- see ARDIAG's g_legs/g_arms below). Same per-
    # pattern try/except as that block's _find_group (find_joints raises if ANY pattern
    # in the list matches zero joints).
    def _grp(pats):
      ids = set()
      for p in pats:
        try:
          pids, _ = uenv.scene["robot"].find_joints([p])
        except ValueError:
          continue
        ids.update(pids)
      return torch.as_tensor(sorted(ids), device=robot.joint_pos.device) if ids else None
    leg_ids = _grp([".*hip.*", ".*knee.*", ".*ankle.*"])
    arm_ids = _grp([".*shoulder.*", ".*elbow.*", ".*wrist.*"])
    # Per-joint action scale kappa (rad per action unit), read from the live action term
    # rather than re-imported from the robot constants, so it can never desync from the
    # plant the checkpoint is actually being replayed on. Needed because every commanded-
    # side smoothness number is reported in RADIANS: kappa = 0.25*tau_max/Kp spans 6.7x
    # across the body (legs 0.25, shoulder_yaw 0.0375), so a raw-action-unit whole-body
    # norm physically over-weights the arms 3-6.7x. See CONTEXT.md "Physical units rule".
    _act_scale = uenv.action_manager.get_term("joint_pos").scale
    if not torch.is_tensor(_act_scale):
      _act_scale = torch.full((1, uenv.action_manager.total_action_dim),
                              float(_act_scale), device=robot.joint_pos.device)
    kappa = _act_scale[:1].to(robot.joint_pos.device)  # [1, A], env-invariant
    ctrl_dt = uenv.step_dt
    # CoT / cadence metrics (A1a M0): mechanical power, cost of transport, achieved stride period.
    try:
      contact_sensor = uenv.scene["feet_ground_contact"]
    except KeyError:
      contact_sensor = None
    step_dt = uenv.step_dt
    MASS_G = 75.0 * 9.81  # H1-2 ~75 kg; dimensionless CoT = energy / (m g distance)
    # WP1 payload pilot: copper-loss-corrected CoT variant, metric-only (never enters any
    # reward -- the trained/mechanical `power`/`cot` below are unchanged). k = 0.3 from
    # Yang et al. 2022 (CoRL), "Fast and Efficient Locomotion via Learned Gait Transitions"
    # (arXiv:2104.04644) eq. 3, Sum_i max(tau_i*omega_i + k*tau_i^2, 0), their "motor
    # parameter" following the MIT-Cheetah/Di Carlo actuator convention. NOT fit to the
    # H1-2's M107/GO2HV actuators specifically (no public winding-resistance/torque-constant
    # spec for them); a documented literature default.
    COPPER_LOSS_K = 0.3

    # Collect label / structure info for the JSON line.
    bench_meta: dict = {"label": str(resume_path) if resume_path is not None else "unknown"}
    for k in ("hl_algorithm", "goal_components", "c"):
      if hasattr(agent_cfg, k):
        v = getattr(agent_cfg, k)
        bench_meta[k] = list(v) if isinstance(v, (list, tuple)) else v

    seed_results: list[dict] = []
    for seed_idx in range(cfg.eval_seeds):
      seed = 42 + seed_idx
      torch.manual_seed(seed)
      # reset() does in-place buffer updates; must be inside inference_mode or it errors
      # on seed >= 1 (the prior rollout marked those env buffers as inference tensors).
      with torch.inference_mode():
        obs, _ = env.reset()

      errs_vx, errs_vy, errs_yaw = [], [], []
      fall_flags, ep_lens, action_rates, orient_devs, height_devs = [], [], [], [], []
      ub_pose_devs, ub_arm_vels, powers, gait_matches = [], [], [], []
      # WP1 payload pilot: copper-loss power (metric-only), body roll/pitch-rate magnitude
      # (wobble proxy), and per-step cross-env vx variance (F5 analogue -- the N=64 parallel
      # envs under a pinned --eval-cmd-vx ARE the "N repeats of an identical command").
      powers_copper, omega_xys, achieved_vx_vars = [], [], []
      # Leg-restricted action rate + whole-body/leg/arm joint-accel (deploy diagnostics).
      act_legs_list, jacc_list, jacc_legs_list, jacc_arms_list = [], [], [], []
      # LCP smoothness suite (Chen 2025), legs-only, in the paper's units. `ajit` is the
      # THIRD derivative of the commanded joint target (rad/s^3) and `qjit` the third
      # derivative of the realized joint position -- the two metrics the paper's ablation
      # ranks on, because they separate smoothing methods ~13x where first/second
      # derivatives (our action_rate / jacc) separate them only 1.2-1.7x.
      ajit_list, qjit_list, act_legs_rad_list = [], [], []
      act_hist: list[torch.Tensor] = []   # last 3 raw action tensors, newest last
      prev_jacc_signed: torch.Tensor | None = None
      achieved_vxs = []  # per-step env-mean achieved vx (ramp metric for --eval-cmd-vx holds)
      prev_actions: torch.Tensor | None = None
      # --check-joint-limits: hardware command bounds, read from the deploy header rather
      # than duplicated here (it is the audited source -- 5-source consensus, WL-B0 item 2).
      lim_lo = lim_hi = lim_names = None
      jl_over = jl_max = None
      if cfg.check_joint_limits:
        import re
        hdr = (Path(__file__).parent.parent
               / "deploy/robots/h1_2/include/h1_2_limits.h").read_text()
        blk = hdr.split("h1_2_joint_limits = {{", 1)[1].split("}};", 1)[0]
        rows = re.findall(r"\{\s*(-?[\d.]+)f?,\s*(-?[\d.]+)f?\s*\}\s*,?\s*//\s*\d+\s+(\w+)", blk)
        by_name = {n.lower(): (float(a), float(b)) for a, b, n in rows}
        at = uenv.action_manager.get_term("joint_pos")
        sim_names = uenv.scene["robot"].joint_names
        tgt = at._target_ids if hasattr(at, "_target_ids") else list(range(len(sim_names)))
        tgt = list(range(len(sim_names))) if tgt is None else list(tgt)
        # The sim model calls slot 12 `torso`, the deploy header calls it `waist_yaw`
        # (same joint; play.py's own group patterns already match both spellings).
        alias = {"torso": "waist_yaw"}
        lim_names = [alias.get(n, n) for n in
                     (sim_names[i].replace("_joint", "").lower() for i in tgt)]
        miss = [n for n in lim_names if n not in by_name]
        if miss:
          raise SystemExit(f"--check-joint-limits: no deploy limit for {miss}")
        lim_lo = torch.tensor([by_name[n][0] for n in lim_names], device=env.device)
        lim_hi = torch.tensor([by_name[n][1] for n in lim_names], device=env.device)
        jl_over = torch.zeros(len(lim_names), device=env.device)
        jl_max = torch.zeros(len(lim_names), device=env.device)
        jl_sum = torch.zeros(len(lim_names), device=env.device)  # mean excess/step (ADR-0008 sizing)
        jl_rew = {}  # term name -> summed |contribution|, for the reward-share table
        jl_cmds = []  # per-step POST-clip targets, for the commanded-range percentiles
      energy_eng = dist_eng = td_count = 0.0  # CoT numerator/denominator + footfall count
      energy_eng_copper = 0.0  # copper-loss CoT numerator (metric-only)
      prev_contact: torch.Tensor | None = None
      n_feet = 0
      gait_offsets = torch.tensor([0.0, 0.5], device=env.device).view(1, -1)
      # d(T) duty for gait_match: MUST mirror feet_gait's schedule, else a d(T) checkpoint
      # reads as "drifting" against the fixed floor (the structure_keys lesson, eval side).
      gait_swing = 0.0 if DUMMY_MODE else getattr(runner, "cadence_swing_time", 0.0)
      duty_lo, duty_hi = (0.56, 0.70) if DUMMY_MODE else getattr(
        runner, "cadence_duty_range", (0.56, 0.70))

      with torch.inference_mode():
        for _ in range(cfg.eval_steps):
          actions = policy(obs)
          obs, _, dones, extras = env.step(actions.to(env.device))

          # 1. Tracking error.
          if cfg.check_joint_limits:
            # RECOMPUTE the pre-clip target: raw*scale+offset, matching the C++ `q_cmd`
            # before its clamp. Reading `_processed_actions` back would report 0.0 forever
            # once ADR-0008's clip is configured, because mjlab's process_actions
            # overwrites it with the CLAMPED value -- the instrument would die at exactly
            # the moment it is needed, which is the same measure-after-the-clamp defect
            # ADR-0008 exists to fix.
            _t = uenv.action_manager.get_term("joint_pos")
            q_cmd = _t.raw_action * _t.scale + _t.offset
            over = torch.maximum(lim_lo - q_cmd, q_cmd - lim_hi).clamp(min=0.0)  # [B, D]
            jl_over += (over > 0).float().mean(dim=0)
            jl_sum += over.mean(dim=0)
            jl_max = torch.maximum(jl_max, over.amax(dim=0))
            # Post-clip target = what the deploy path actually sends, and what the flight
            # recorder holds: State_RLBase.cpp:221 logs `action` and the safety clamp at
            # :164 writes a local, so `raw_q` is post-yaml-clip. Kept per step because the
            # RANGE, not the excess, is the discriminating quantity -- `over` collapses to
            # ~0 on any clip-trained policy, which hides whether the clip merely removed an
            # unusable overshoot or the penalty retreated the policy off the bound.
            jl_cmds.append(torch.clamp(q_cmd, lim_lo, lim_hi).detach())
            # Reward shares (ADR-0008 sizing): the excess weight is set so `action_clip`
            # contributes what `joint_pos_limits` contributes at the baseline violation
            # level, so both must be measured on the SAME rollout. NOTE a term whose
            # weight is 0.0 is skipped by RewardManager.compute and reads 0 here -- that
            # is why the sizing uses the RAW mean excess above, not this table, for the
            # numerator; this table supplies the target contribution only.
            _rm = uenv.reward_manager
            _sr = _rm._step_reward
            for _i, _n in enumerate(_rm.active_terms):
              jl_rew[_n] = jl_rew.get(_n, 0.0) + float(_sr[:, _i].abs().mean())

          cmd = uenv.command_manager.get_command("twist")  # [B, >=3]
          achieved = torch.cat(
            [robot.root_link_lin_vel_b[:, :2], robot.root_link_ang_vel_b[:, 2:3]], dim=-1
          )
          ae = (cmd[:, :3] - achieved).abs()  # [B, 3]
          errs_vx.append(ae[:, 0].mean().item())
          errs_vy.append(ae[:, 1].mean().item())
          errs_yaw.append(ae[:, 2].mean().item())
          achieved_vxs.append(achieved[:, 0].mean().item())
          # WP1 stability proxy (F5 analogue): cross-env variance of achieved vx at this
          # step -- meaningful under a pinned --eval-cmd-vx, where the N envs are N
          # repeats of the identical command (same convention as ss_err_vx/t90 below).
          achieved_vx_vars.append(achieved[:, 0].var(dim=0).item())

          # 2. Survival: fall flag and episode length.
          if has_fell:
            fall_flags.append(tm.get_term("fell_over").float().mean().item())
          else:
            fall_flags.append(dones.float().mean().item())
          ep_lens.append(uenv.episode_length_buf.float().mean().item())

          # 3. Smoothness: action rate ||a_t - a_{t-1}||, whole-body + leg-restricted
          # (same slicing as ARDIAG's g_legs -- see the cross-check note on act_legs below).
          if prev_actions is not None:
            d_ar = actions - prev_actions
            action_rates.append(d_ar.norm(dim=-1).mean().item())
            act_legs_list.append(
              d_ar[:, leg_ids].norm(dim=-1).mean().item() if leg_ids is not None
              else float("nan"))
            if leg_ids is not None:
              act_legs_rad_list.append(
                (d_ar * kappa)[:, leg_ids].norm(dim=-1).mean().item())
          prev_actions = actions.clone()

          # 3c. Action jitter (LCP primary): d3(q_des)/dt3 in rad/s^3, legs-only.
          # q_des = q_def + kappa*a, and q_def is constant, so the third derivative of the
          # command is the third difference of kappa*a. Backward third difference
          # (a_t - 3a_{t-1} + 3a_{t-2} - a_{t-3}) / dt^3 -- exact here because the action
          # sequence carries no sensor noise, unlike the same operator on hardware logs.
          act_hist.append(actions.clone())
          if len(act_hist) > 4:
            act_hist.pop(0)
          if len(act_hist) == 4:
            a3, a2, a1, a0_ = act_hist  # oldest -> newest
            d3 = (a0_ - 3.0 * a1 + 3.0 * a2 - a3) * kappa / (ctrl_dt ** 3)
            ajit_list.append(
              d3[:, leg_ids].norm(dim=-1).mean().item() if leg_ids is not None
              else float("nan"))

          # 3b. Joint acceleration (deploy diagnostic): whole-body + leg/arm-restricted
          # mean |qddot|. p95 (below, post-rollout) separates contact-impulse spikes
          # from steady-state jitter -- the mean alone conflates the two.
          jacc = robot.joint_acc.abs()
          # 3d. DoF position jitter (LCP realized-side counterpart): one difference of the
          # SIGNED joint acceleration, not of |qddot| -- differencing the absolute value
          # would miss every sign flip, which is most of what jerk is.
          if prev_jacc_signed is not None and leg_ids is not None:
            qjit = (robot.joint_acc - prev_jacc_signed) / ctrl_dt
            qjit_list.append(qjit[:, leg_ids].norm(dim=-1).mean().item())
          prev_jacc_signed = robot.joint_acc.clone()
          jacc_list.append(jacc.mean().item())
          jacc_legs_list.append(
            jacc[:, leg_ids].mean().item() if leg_ids is not None else float("nan"))
          jacc_arms_list.append(
            jacc[:, arm_ids].mean().item() if arm_ids is not None else float("nan"))

          # 4. Stability: orientation deviation + height deviation.
          orient_devs.append(
            robot.projected_gravity_b[:, :2].norm(dim=-1).mean().item()
          )
          nom_h = robot.default_root_state[:, 2]  # [B] nominal height
          height_devs.append(
            (robot.root_link_pos_w[:, 2] - nom_h).abs().mean().item()
          )
          # WP1 stability proxy: body roll/pitch-rate magnitude (wobble), distinct from
          # orient_dev (a position-like projected-gravity deviation) and err_yaw (the
          # commanded-axis yaw-RATE tracking error, about z only).
          omega_xys.append(robot.root_link_ang_vel_b[:, :2].norm(dim=-1).mean().item())

          # 5. Upper-body deploy hygiene: arm+waist drift from default + joint speed.
          ub_pose_devs.append(
            (robot.joint_pos[:, ub_ids] - robot.default_joint_pos[:, ub_ids])
            .square().mean(dim=1).mean().item()
          )
          ub_arm_vels.append(robot.joint_vel[:, ub_ids].abs().mean().item())

          # 6. Mechanical power + cost of transport (gated by commanded linear speed > 0.1).
          power = (robot.qfrc_actuator * robot.joint_vel).abs().sum(dim=1)  # [B] watts
          powers.append(power.mean().item())
          # 6b. Copper-loss-corrected power (metric-only, WP1): adds k*tau^2 per joint
          # before summing -- see COPPER_LOSS_K comment above.
          power_copper = ((robot.qfrc_actuator * robot.joint_vel).abs()
                          + COPPER_LOSS_K * robot.qfrc_actuator.square()).sum(dim=1)
          powers_copper.append(power_copper.mean().item())
          lin_speed = robot.root_link_lin_vel_b[:, :2].norm(dim=-1)  # [B] achieved m/s
          eng = (cmd[:, :2].norm(dim=-1) > 0.1).float()  # commanded-motion gate
          energy_eng += (power * eng).sum().item() * step_dt
          energy_eng_copper += (power_copper * eng).sum().item() * step_dt
          dist_eng += (lin_speed * eng).sum().item() * step_dt
          # 7. Achieved stride period from footfall rising edges (same-foot touchdown interval).
          if contact_sensor is not None:
            is_contact = contact_sensor.data.current_contact_time > 0  # [B, n_feet]
            n_feet = is_contact.shape[1]
            if prev_contact is not None:
              td_count += (is_contact & ~prev_contact).float().sum().item()
            prev_contact = is_contact.clone()
            # 8. Gait match vs the commanded clock (H2 lock-in diagnostic): fraction of feet
            # whose contact agrees with the commanded stance schedule (same rule as feet_gait).
            hp = getattr(uenv, "hrl_phase", None)
            gait_thr = duty_lo
            if hp is None:  # A0 / non-cadence A1: the fixed 0.6 s clock
              hp = (uenv.episode_length_buf * step_dt) / 0.6
            elif gait_swing > 0.0:  # d(T) schedule, same map as feet_gait
              gait_thr = (1.0 - gait_swing / uenv.hrl_period).clamp(duty_lo, duty_hi).unsqueeze(1)
            leg_phase = (hp.unsqueeze(1) + gait_offsets) % 1.0
            gait_matches.append(((leg_phase < gait_thr) == is_contact).float().mean().item())

      def _m(lst): return float(torch.tensor(lst).mean())  # noqa: E731
      def _p95(lst):  # noqa: E731
        # NaN-poisoned lists (an empty leg/arm group) stay NaN rather than erroring out
        # of torch.quantile.
        return float("nan") if any(v != v for v in lst) else torch.quantile(torch.tensor(lst), 0.95).item()

      # Hold-eval split (stage D, 2026-07-14): steady-state errs over the last 2/3 of the
      # rollout separate the from-stand acceleration ramp from held tracking (the table-e
      # full-window numbers conflate them); t90 = time to first reach 90% of the commanded
      # vx (NaN when never reached, or for random-command aggregates where it's undefined).
      ss0 = cfg.eval_steps // 3
      if cfg.eval_cmd_vx is not None and abs(cfg.eval_cmd_vx) > 1e-6:
        sgn = 1.0 if cfg.eval_cmd_vx > 0 else -1.0
        thr = 0.9 * abs(cfg.eval_cmd_vx)
        t90 = next((i * step_dt for i, v in enumerate(achieved_vxs) if sgn * v >= thr),
                   float("nan"))
      else:
        t90 = float("nan")

      # Commanded RANGE per joint (2026-08-28, ADR-0008 follow-up). Percentiles of the
      # POST-clip target so they compare directly against the flight recorder. p1/p99
      # rather than min/max: one transient excursion is not authority, sustained range is.
      # `headroom` is the distance from the used range to the nearer bound -- the quantity
      # that separates a clip (headroom ~0, policy still reaches the stop) from a penalty
      # that pushed the policy inward (headroom ~ one action-sigma).
      jl_range = {}
      if cfg.check_joint_limits and jl_cmds:
        allc = torch.cat(jl_cmds, dim=0).float()  # [steps*B, D]
        p01, p50, p99 = torch.quantile(
          allc, torch.tensor([0.01, 0.5, 0.99], device=allc.device), dim=0)
        pin = (((allc - lim_lo).abs() < 1e-9)
               | ((allc - lim_hi).abs() < 1e-9)).float().mean(dim=0)
        jl_range = {n: {"p1": round(p01[i].item(), 4), "p50": round(p50[i].item(), 4),
                        "p99": round(p99[i].item(), 4),
                        "span": round((p99[i] - p01[i]).item(), 4),
                        "pinned": round(pin[i].item(), 5),
                        "headroom": round(min((lim_hi[i] - p99[i]).item(),
                                              (p01[i] - lim_lo[i]).item()), 4)}
                    for i, n in enumerate(lim_names)}

      def _span(n): return jl_range.get(n, {}).get("span", float("nan"))  # noqa: E731

      seed_results.append({
        "err_vx":      _m(errs_vx),
        "err_vy":      _m(errs_vy),
        "err_yaw":     _m(errs_yaw),
        "ss_err_vx":   _m(errs_vx[ss0:]),
        "ss_err_vy":   _m(errs_vy[ss0:]),
        "t90_s":       t90,
        "fall_rate":   _m(fall_flags),
        "mean_ep_len": _m(ep_lens),
        "action_rate": _m(action_rates) if action_rates else float("nan"),
        "act_legs":    _m(act_legs_list) if act_legs_list else float("nan"),
        # Ankle command SPAN (rad): the deploy-relevant authority measure, since the two
        # rolls are what reject a lateral push. Scalars here rather than print-only so
        # they get the per-seed line and the seed mean+-std every other metric gets.
        "ank_roll_span_l":  _span("left_ankle_roll"),
        "ank_roll_span_r":  _span("right_ankle_roll"),
        "ank_pitch_span_l": _span("left_ankle_pitch"),
        "ank_pitch_span_r": _span("right_ankle_pitch"),
        "jacc":          _m(jacc_list),
        "jacc_legs":     _m(jacc_legs_list),
        "jacc_arms":     _m(jacc_arms_list),
        "jacc_legs_p95": _p95(jacc_legs_list),
        "jacc_arms_p95": _p95(jacc_arms_list),
        # LCP suite, legs-only, rad/s^3 (published Unitree H1 refs: action jitter 0.44 in
        # MuJoCo, 1.11-1.20 on real ground; NOT rate-matched to our 50 Hz -- a sanity band,
        # never a pass/fail gate).
        "ajit_legs":     _m(ajit_list) if ajit_list else float("nan"),
        "ajit_legs_p95": _p95(ajit_list) if ajit_list else float("nan"),
        "qjit_legs":     _m(qjit_list) if qjit_list else float("nan"),
        "qjit_legs_p95": _p95(qjit_list) if qjit_list else float("nan"),
        # Commanded agitation in PHYSICAL units (rad/step): act_legs is raw action units.
        "act_legs_rad":  _m(act_legs_rad_list) if act_legs_rad_list else float("nan"),
        "orient_dev":  _m(orient_devs),
        "height_dev":  _m(height_devs),
        "ub_pose_dev": _m(ub_pose_devs),
        "ub_arm_vel":  _m(ub_arm_vels),
        "mech_power_w": _m(powers) if powers else float("nan"),
        "gait_match":   _m(gait_matches) if gait_matches else float("nan"),
        "cot":          energy_eng / (dist_eng * MASS_G + 1e-6),
        # WP1 payload pilot additions (2026-08-27): copper-loss CoT variant (metric-only,
        # see COPPER_LOSS_K), body roll/pitch-rate wobble, and the F5-analogue cross-env
        # vx spread under a pinned command (meaningful only with --eval-cmd-vx set).
        "mech_power_copper_w": _m(powers_copper) if powers_copper else float("nan"),
        "cot_copper":   energy_eng_copper / (dist_eng * MASS_G + 1e-6),
        "omega_xy":     _m(omega_xys) if omega_xys else float("nan"),
        "ss_vx_var":    _m(achieved_vx_vars[ss0:]) if achieved_vx_vars else float("nan"),
        "stride_period_s": (cfg.eval_steps * step_dt)
                           / max(td_count / max(n_envs * max(n_feet, 1), 1), 1e-6),
      })
      # WP1 payload pilot: a per-seed line, since [BENCH] below only ever prints the
      # seed-aggregated mean+-std. The sweep CSV needs one row per (T, payload, vx, seed)
      # for the T* seed-spread analysis, which the aggregate's std alone cannot give
      # (fitting T* per seed, then spreading THAT, is not the same as spreading the metric).
      print(f"[BENCH_SEED] {json.dumps({**bench_meta, 'seed': seed, **seed_results[-1]})}")

    # Aggregate across seeds.
    keys = list(seed_results[0].keys())
    vals = {k: torch.tensor([r[k] for r in seed_results]) for k in keys}
    means = {k: vals[k].mean().item() for k in keys}
    stds  = {k: vals[k].std().item() if cfg.eval_seeds > 1 else float("nan") for k in keys}

    def _fmt(k): return f"{means[k]:.4f} ± {stds[k]:.4f}" if cfg.eval_seeds > 1 else f"{means[k]:.4f}"  # noqa: E731

    print()
    print("=" * 58)
    if cfg.check_joint_limits and jl_over is not None:
      rate = (jl_over / cfg.eval_steps).tolist()
      mean_ex = (jl_sum / cfg.eval_steps).tolist()
      worst = sorted(zip(lim_names, rate, jl_max.tolist(), mean_ex), key=lambda r: -r[1])
      tot = sum(rate) / max(len(rate), 1)
      tot_ex = sum(mean_ex)
      print()
      print("=" * 78)
      print(f"  COMMANDED JOINT-LIMIT CHECK (last seed) | {cfg.eval_steps} steps x {n_envs} envs")
      print("=" * 78)
      print(f"  any-joint over-limit rate (mean per joint) = {tot:.5f}")
      print(f"  TOTAL mean excess/step (the reward term's raw value) = {tot_ex:.6f} rad")
      hit = [w for w in worst if w[1] > 0]
      if not hit:
        print("  no commanded target left the hardware limits")
      for n, r, m, e in hit[:8]:
        print(f"    {n:<20} rate {r:>8.5f}   max_over {m:>7.4f}   mean_excess {e:>8.6f} rad")
      # ADR-0008 sizing: `action_clip`'s weight should make it contribute what
      # `joint_pos_limits` contributes here. That target is read off this table; the
      # numerator is the RAW total above, because a weight-0.0 term is never evaluated
      # by RewardManager.compute and would otherwise read a misleading 0.
      if jl_rew:
        print("  -" * 39)
        print(f"  reward-term shares (|contribution| per step, {cfg.eval_steps} steps)")
        tot_r = sum(jl_rew.values()) or 1.0
        for n, v in sorted(jl_rew.items(), key=lambda kv: -kv[1])[:10]:
          mark = "  <-- sizing target" if n == "joint_pos_limits" else (
                 "  <-- 0.0 weight: NOT evaluated" if n == "action_clip" else "")
          print(f"    {n:<24} {v / cfg.eval_steps:>10.6f}  ({100 * v / tot_r:>5.2f}%){mark}")
        jpl = jl_rew.get("joint_pos_limits", 0.0) / cfg.eval_steps
        if tot_ex > 0:
          print(f"  => suggested |weight| = {jpl:.6f} / {tot_ex:.6f} = {jpl / tot_ex:.4f}")
      print("=" * 78)
      if jl_range:
        print("  -" * 39)
        print("  COMMANDED RANGE, post-clip (last seed)   span = p99-p1, "
              "headroom = gap to nearer bound")
        for n in ("left_ankle_roll", "right_ankle_roll",
                  "left_ankle_pitch", "right_ankle_pitch"):
          r = jl_range.get(n)
          if r is None:
            continue
          i = lim_names.index(n)
          print(f"    {n:<20} p1 {r['p1']:>8.4f}  p99 {r['p99']:>8.4f}  "
                f"span {r['span']:>7.4f}  headroom {r['headroom']:>7.4f}  "
                f"pinned {r['pinned']:>8.5f}   "
                f"bounds [{lim_lo[i].item():+.4f},{lim_hi[i].item():+.4f}]")
      jl_json = {"total_mean_excess_rad": round(tot_ex, 6),
                 "range": jl_range,
                 "per_joint": {n: {"rate": round(r, 6), "max_over_rad": round(m, 4),
                                   "mean_excess_rad": round(e, 6)}
                               for n, r, m, e in hit},
                 "reward_share": {n: round(v / cfg.eval_steps, 6) for n, v in jl_rew.items()}}
      print(f"[LIMITS] {json.dumps(jl_json)}")

    print(f"  BENCHMARK SCORECARD  |  {cfg.eval_steps} steps x {n_envs} envs x {cfg.eval_seeds} seed(s)")
    print("=" * 58)
    print(f"  Tracking  err_vx    : {_fmt('err_vx')}")
    print(f"  Tracking  err_vy    : {_fmt('err_vy')}")
    print(f"  Tracking  err_yaw   : {_fmt('err_yaw')}")
    print(f"  Survival  fall_rate : {_fmt('fall_rate')}")
    print(f"  Survival  ep_len    : {_fmt('mean_ep_len')}")
    print(f"  Smoothness act_rate : {_fmt('action_rate')}")
    print(f"  Smoothness act_legs : {_fmt('act_legs')}")
    print(f"  Accel     jacc      : {_fmt('jacc')}")
    print(f"  Accel     jacc_legs : {_fmt('jacc_legs')}")
    print(f"  Accel     jacc_arms : {_fmt('jacc_arms')}")
    print(f"  Accel     jacc_legs_p95: {_fmt('jacc_legs_p95')}")
    print(f"  Accel     jacc_arms_p95: {_fmt('jacc_arms_p95')}")
    print(f"  Stability orient_dev: {_fmt('orient_dev')}")
    print(f"  Stability height_dev: {_fmt('height_dev')}")
    print(f"  UpperBody pose_dev  : {_fmt('ub_pose_dev')}")
    print(f"  UpperBody arm_vel   : {_fmt('ub_arm_vel')}")
    print(f"  Energy    power_W   : {_fmt('mech_power_w')}")
    print(f"  Energy    CoT (norm): {_fmt('cot')}")
    print(f"  Energy    CoT copper: {_fmt('cot_copper')}")
    print(f"  Stability omega_xy  : {_fmt('omega_xy')}")
    print(f"  Stability ss_vx_var : {_fmt('ss_vx_var')}")
    print(f"  Gait      stride_s  : {_fmt('stride_period_s')}")
    print(f"  Gait      match     : {_fmt('gait_match')}")
    if cfg.eval_cmd_vx is not None:
      print(f"  Hold      ss_err_vx : {_fmt('ss_err_vx')}  (last 2/3)")
      print(f"  Hold      ss_err_vy : {_fmt('ss_err_vy')}")
      print(f"  Hold      t90_s     : {_fmt('t90_s')}")
    print("=" * 58)
    print()

    bench_out = {**bench_meta, **{k: round(means[k], 6) for k in keys},
                 **{f"{k}_std": round(stds[k], 6) for k in keys},
                 "eval_steps": cfg.eval_steps, "num_envs": n_envs, "eval_seeds": cfg.eval_seeds}
    print(f"[BENCH] {json.dumps(bench_out)}")
    env.close()
    return

  # WL-D arm 10 probe (2026-07-20, A1a_plan.md "Arm 10"): left/right gait symmetry.
  # feet_gait's own gait_match .mean(dim=1) dilutes one leg's systematic schedule
  # violation against the other's good match, and nothing in the aggregate bench
  # compares the two legs to each other or counts touchdowns per foot - this probe
  # does, plus checks whether play.py:498's stride_period_s (which divides eval time
  # by raw touchdown count) is corrupted by a double-tap inflating that count.
  if cfg.diagnose_symmetry > 0:
    import json
    uenv = env.unwrapped
    n_envs = uenv.num_envs
    contact_sensor = uenv.scene["feet_ground_contact"]
    step_dt = uenv.step_dt
    gait_offsets = torch.tensor([0.0, 0.5], device=env.device).view(1, -1)
    gait_swing = 0.0 if DUMMY_MODE else getattr(runner, "cadence_swing_time", 0.0)
    duty_lo, duty_hi = (0.56, 0.70) if DUMMY_MODE else getattr(
      runner, "cadence_duty_range", (0.56, 0.70))

    seed_results: list[dict] = []
    for seed_idx in range(cfg.eval_seeds):
      torch.manual_seed(42 + seed_idx)
      with torch.inference_mode():
        obs, _ = env.reset()

      elapsed = torch.zeros(n_envs, device=env.device)
      prev_contact: torch.Tensor | None = None
      # Per-env ordered touchdown event log: (time, foot) with foot 0=left, 1=right
      # (same left/right convention as gait_offsets / _gait_params's offset [0.0, 0.5]).
      td_events: list[list[tuple[float, int]]] = [[] for _ in range(n_envs)]
      gait_matches_foot: list[list[float]] = [[], []]
      both_contact_count = 0.0
      total_count = 0

      with torch.inference_mode():
        for _ in range(cfg.diagnose_symmetry):
          actions = policy(obs)
          obs, _, dones, _ = env.step(actions.to(env.device))

          is_contact = contact_sensor.data.current_contact_time > 0  # [B, 2]
          if prev_contact is not None:
            first_td = is_contact & ~prev_contact  # [B, 2]
            for foot in range(2):
              for i in first_td[:, foot].nonzero(as_tuple=True)[0].tolist():
                td_events[i].append((elapsed[i].item(), foot))
          prev_contact = is_contact.clone()

          both_contact_count += is_contact.all(dim=1).float().sum().item()
          total_count += n_envs

          # Per-foot gait_match: same schedule feet_gait/eval_steps use, NOT averaged
          # over feet (arm 10's whole premise: the average hides a one-leg violation).
          hp = getattr(uenv, "hrl_phase", None)
          gait_thr = duty_lo
          if hp is None:
            hp = (uenv.episode_length_buf * step_dt) / 0.6
          elif gait_swing > 0.0:
            gait_thr = (1.0 - gait_swing / uenv.hrl_period).clamp(duty_lo, duty_hi).unsqueeze(1)
          leg_phase = (hp.unsqueeze(1) + gait_offsets) % 1.0
          matches = (leg_phase < gait_thr) == is_contact  # [B, 2]
          gait_matches_foot[0].append(matches[:, 0].float().mean().item())
          gait_matches_foot[1].append(matches[:, 1].float().mean().item())

          elapsed += step_dt
          if dones.any():
            d = dones.bool()
            elapsed[d] = 0.0
            # Drop in-progress event lists for reset envs so a reset never splices
            # into a fake short/long interval across the episode boundary.
            for i in d.nonzero(as_tuple=True)[0].tolist():
              td_events[i] = []

      # Post-hoc per-env event-sequence analysis: classify each consecutive touchdown
      # pair (any foot) as alternating (LR/RL -> the symmetry-index intervals) or a
      # same-foot repeat (LL/RR -> exactly the user's "touches down twice before the other
      # lifts" stutter, counted separately rather than folded into t_LR/t_RL).
      t_lr, t_rl = [], []
      repeat_n = {0: 0, 1: 0}
      td_counts = [0, 0]
      for events in td_events:
        events.sort(key=lambda e: e[0])
        td_counts[0] += sum(1 for _, f in events if f == 0)
        td_counts[1] += sum(1 for _, f in events if f == 1)
        for (t0, f0), (t1, f1) in zip(events, events[1:]):
          dt = t1 - t0
          if dt <= 0:
            continue
          if f0 == 0 and f1 == 1:
            t_lr.append(dt)
          elif f0 == 1 and f1 == 0:
            t_rl.append(dt)
          else:  # f0 == f1: same-foot repeat, the double-tap signature
            repeat_n[f1] += 1

      def _stats(xs):
        if not xs:
          return {"n": 0, "mean": float("nan"), "std": float("nan")}
        t = torch.tensor(xs)
        return {"n": len(xs), "mean": t.mean().item(),
                "std": t.std().item() if len(xs) > 1 else 0.0}

      lr_stats, rl_stats = _stats(t_lr), _stats(t_rl)
      si = (abs(lr_stats["mean"] - rl_stats["mean"]) / (lr_stats["mean"] + rl_stats["mean"])
            if lr_stats["n"] and rl_stats["n"] else float("nan"))
      # (c) stride corruption check: play.py:498's stride_period_s divides eval time by
      # (raw touchdown count / (n_envs*n_feet)) - recompute that SAME metric here so it
      # is directly comparable, in this run, to an alternation-based full-cycle estimate
      # (mean t_LR + mean t_RL) that is immune to double-taps (those land in repeat_n
      # instead of t_LR/t_RL, so they can't shrink the alternation-based stride).
      total_td = td_counts[0] + td_counts[1]
      stride_play_metric = ((cfg.diagnose_symmetry * step_dt)
                             / max(total_td / max(n_envs * 2, 1), 1e-6))
      stride_alternation = (lr_stats["mean"] + rl_stats["mean"]
                             if lr_stats["n"] and rl_stats["n"] else float("nan"))

      seed_results.append({
        "td_count_left": float(td_counts[0]), "td_count_right": float(td_counts[1]),
        "t_lr_mean": lr_stats["mean"], "t_lr_std": lr_stats["std"], "t_lr_n": float(lr_stats["n"]),
        "t_rl_mean": rl_stats["mean"], "t_rl_std": rl_stats["std"], "t_rl_n": float(rl_stats["n"]),
        "si": si,
        "repeat_left_n": float(repeat_n[0]), "repeat_right_n": float(repeat_n[1]),
        "double_support_frac": both_contact_count / max(total_count, 1),
        "gait_match_left": float(torch.tensor(gait_matches_foot[0]).mean()),
        "gait_match_right": float(torch.tensor(gait_matches_foot[1]).mean()),
        "stride_play_metric": stride_play_metric,
        "stride_alternation": stride_alternation,
      })

    keys = list(seed_results[0].keys())
    means = {k: float(torch.tensor([r[k] for r in seed_results]).mean()) for k in keys}

    print()
    print("=" * 70)
    print(f"  SYMMETRY PROBE (WL-D arm 10) | {cfg.diagnose_symmetry} steps x {n_envs} envs "
          f"x {cfg.eval_seeds} seed(s)")
    print("=" * 70)
    print(f"  Touchdowns   left={means['td_count_left']:.0f}  right={means['td_count_right']:.0f}"
          f"  ratio(R/L)={means['td_count_right'] / max(means['td_count_left'], 1e-6):.3f}")
    print(f"  t_LR  mean={means['t_lr_mean']:.4f}s std={means['t_lr_std']:.4f} (n={means['t_lr_n']:.0f})")
    print(f"  t_RL  mean={means['t_rl_mean']:.4f}s std={means['t_rl_std']:.4f} (n={means['t_rl_n']:.0f})")
    print(f"  SI (symmetry index) = {means['si']:.4f}")
    print(f"  Same-foot repeats (double-taps): left={means['repeat_left_n']:.0f}"
          f"  right={means['repeat_right_n']:.0f}")
    print(f"  Double-support fraction (realized) = {means['double_support_frac']:.4f}"
          f"  (scheduled ~0.12 at duty {duty_lo:.2f})")
    print(f"  gait_match  left={means['gait_match_left']:.4f}  right={means['gait_match_right']:.4f}")
    print(f"  stride_period_s (play.py:498 metric) = {means['stride_play_metric']:.4f}")
    print(f"  stride (alternation t_LR+t_RL)       = {means['stride_alternation']:.4f}")
    print("=" * 70)
    print()

    diag_out = {"label": str(resume_path) if resume_path is not None else "unknown",
                **{k: round(v, 6) for k, v in means.items()},
                "eval_steps": cfg.diagnose_symmetry, "num_envs": n_envs,
                "eval_seeds": cfg.eval_seeds}
    print(f"[SYMDIAG] {json.dumps(diag_out)}")
    env.close()
    return

  # Action-rate decomposition: is A1's twitch concentrated at the HL fire step (goal obs
  # jumps every c steps) or spread uniformly (a smoothness-reward deficit)? Bin the
  # whole-body action delta by step % c and by joint group. A0 (no HL) is the flat control.
  if cfg.diagnose_action_rate > 0:
    import json
    uenv = env.unwrapped
    n_envs = uenv.num_envs
    c = getattr(runner, "c", 8)  # A0 has no window; c=8 makes it the flat-profile control.

    def _find_group(pats):
      # find_joints([...]) raises if ANY pattern in the list matches zero joints (e.g.
      # this robot has a torso joint but no "waist" joint); resolve patterns one at a
      # time so a non-matching pattern is just skipped instead of failing the group.
      ids = set()
      for p in pats:
        try:
          pids, _ = uenv.scene["robot"].find_joints([p])
        except ValueError:
          continue
        ids.update(pids)
      return sorted(ids) if ids else None

    # Joint groups (regex vs the robot's joint names), resolved once.
    grp_ids = {}
    for name, pats in (("legs", [".*hip.*", ".*knee.*", ".*ankle.*"]),
                       ("arms", [".*shoulder.*", ".*elbow.*", ".*wrist.*"]),
                       ("waist", [".*waist.*", ".*torso.*"])):
      ids = _find_group(pats)
      grp_ids[name] = torch.as_tensor(ids, device=env.device) if ids is not None else None

    seed_results: list[dict] = []
    for seed_idx in range(cfg.eval_seeds):
      torch.manual_seed(42 + seed_idx)
      with torch.inference_mode():
        obs, _ = env.reset()

      phase_sum = [0.0] * c
      phase_n = [0] * c
      grp_sum = {k: 0.0 for k in grp_ids}
      grp_n = 0
      ar_sum = 0.0
      prev_actions: torch.Tensor | None = None
      with torch.inference_mode():
        for step_i in range(cfg.diagnose_action_rate):
          actions = policy(obs)
          obs, _, dones, _ = env.step(actions.to(env.device))
          if prev_actions is not None:
            d = actions - prev_actions
            nrm = d.norm(dim=-1).mean().item()  # ||a_t - a_{t-1}|| over joints, env-mean
            b = step_i % c
            phase_sum[b] += nrm; phase_n[b] += 1
            ar_sum += nrm; grp_n += 1
            for k, ids in grp_ids.items():
              grp_sum[k] += (d[:, ids].norm(dim=-1).mean().item()
                             if ids is not None else float("nan"))
          prev_actions = actions.clone()

      phase = [phase_sum[i] / phase_n[i] if phase_n[i] else float("nan") for i in range(c)]
      rest = [phase[i] for i in range(1, c) if phase_n[i]]
      rest_mean = sum(rest) / len(rest) if rest else float("nan")
      grp_mean = {k: grp_sum[k] / max(grp_n, 1) for k in grp_ids}
      # Shares from summed squared group norms, so the three shares sum to 1.0.
      sq = {k: (grp_mean[k] ** 2 if grp_mean[k] == grp_mean[k] else 0.0) for k in grp_ids}
      tot_sq = sum(sq.values()) or 1.0
      res = {f"phase_{i}": phase[i] for i in range(c)}
      res.update({
        "fire_excess": phase[0] / rest_mean if rest_mean else float("nan"),
        "ar_mean": ar_sum / max(grp_n, 1),
        "g_legs": grp_mean["legs"], "g_arms": grp_mean["arms"], "g_waist": grp_mean["waist"],
        "share_legs": sq["legs"] / tot_sq, "share_arms": sq["arms"] / tot_sq,
        "share_waist": sq["waist"] / tot_sq,
      })
      seed_results.append(res)

    keys = list(seed_results[0].keys())
    means = {k: float(torch.tensor([r[k] for r in seed_results]).mean()) for k in keys}

    print()
    print("=" * 70)
    print(f"  ACTION-RATE DECOMPOSITION | {cfg.diagnose_action_rate} steps x {n_envs} envs "
          f"x {cfg.eval_seeds} seed(s) | c={c}")
    print("=" * 70)
    prof = "  ".join(f"[{i}]{means[f'phase_{i}']:.3f}" + ("*" if i == 0 else "")
                     for i in range(c))
    print(f"  window profile (* = HL fire step): {prof}")
    print(f"  fire_excess (phase0 / mean rest) = {means['fire_excess']:.3f}")
    print(f"  ar_mean (all steps)              = {means['ar_mean']:.4f}")
    print(f"  legs  mean={means['g_legs']:.4f}  share={means['share_legs']:.3f}")
    print(f"  arms  mean={means['g_arms']:.4f}  share={means['share_arms']:.3f}")
    print(f"  waist mean={means['g_waist']:.4f}  share={means['share_waist']:.3f}")
    print("=" * 70)
    print()

    diag_out = {"label": str(resume_path) if resume_path is not None else "unknown",
                **{k: round(v, 6) for k, v in means.items()},
                "eval_steps": cfg.diagnose_action_rate, "num_envs": n_envs,
                "eval_seeds": cfg.eval_seeds, "c": c}
    print(f"[ARDIAG] {json.dumps(diag_out)}")
    env.close()
    return

  # A1 goal-achievability probe: decompose tracking error into HL goal error vs LL
  # reach error, per HL window. Mirrors HierarchicalRunner.get_inference_policy goal
  # wiring (fire hl.act_inference every c steps; LL sees the remaining delta) but
  # captures s_fire / V* / command at each fire and `achieved` at window end.
  if cfg.diagnose_goals > 0:
    import json
    if not hasattr(runner, "hl"):
      raise ValueError("--diagnose-goals requires a hierarchical (A1) runner.")
    uenv = env.unwrapped
    gs = runner.goal_space
    c = runner.c
    ll_policy = runner.alg.get_policy().to(device)
    runner.alg.eval_mode(); runner.hl.eval_mode()
    # Goal-space layout: find the velocity (is_task) slice for the command comparison.
    offs, vel_slice = {}, None
    i = 0
    for comp in gs.components:
      offs[comp.name] = (i, i + comp.dim)
      if comp.is_task and vel_slice is None:
        vel_slice = (i, i + comp.dim)
      i += comp.dim
    assert vel_slice is not None and (vel_slice[1] - vel_slice[0]) == 3, \
      "probe assumes a 3-dim velocity is_task component (vx, vy, yaw)"
    D = gs.dim
    n_envs = uenv.num_envs
    n_windows = cfg.diagnose_goals // c

    # Per-(window,env) buffers accumulated across seeds; masked by within-window done.
    cmd_a, vstar_a, gabs_a, sfire_a, ach_a, keep_a = [], [], [], [], [], []
    period_a = []  # HL stride periods (cadence runs)
    for seed_idx in range(cfg.eval_seeds):
      torch.manual_seed(42 + seed_idx)
      with torch.inference_mode():
        obs, _ = env.reset()
      with torch.inference_mode():
        for _ in range(n_windows):
          s_fire = gs.extract(uenv)                          # [B, D]
          if getattr(runner, "hl_obs_vel", False):
            obs["hl_vel"] = s_fire[:, 0:2]  # clean lin-vel for the noise-free probe
          v_star = runner.hl.act_inference(uenv, obs, s_fire)  # [B, D]
          scale = gs.scale(uenv)                             # [D]
          g_raw = ((v_star - s_fire) / scale).abs()         # [B, D] recovered |g|
          cmd = uenv.command_manager.get_command("twist")[:, :3].clone()  # [B, 3]
          done_in_window = torch.zeros(n_envs, dtype=torch.bool, device=env.device)
          for k in range(c):
            cur = gs.extract(uenv)
            # Advance the cadence phase clock exactly like get_inference_policy does —
            # the phase obs reads env.hrl_phase, and a frozen clock de-entrains the LL
            # (pre-2026-07-15 probes on cadence checkpoints ran frozen: probe artifact).
            if getattr(runner, "hl_cadence", False):
              uenv.hrl_phase = (uenv.hrl_phase + uenv.step_dt / uenv.hrl_period) % 1.0
            delta = v_star - cur
            uenv.hrl_goal = delta
            obs["goal"] = delta
            obs, _, dones, _ = env.step(ll_policy(obs).to(env.device))
            done_in_window |= dones.bool()
          achieved = gs.extract(uenv)                        # [B, D] window end
          cmd_a.append(cmd.cpu()); vstar_a.append(v_star.cpu()); gabs_a.append(g_raw.cpu())
          sfire_a.append(s_fire.cpu()); ach_a.append(achieved.cpu())
          keep_a.append((~done_in_window).cpu())             # drop reset-contaminated windows
          # A1a: HL-commanded stride period per window (cadence runs)
          hp = getattr(uenv, "hrl_period", None)
          period_a.append(hp.clone().cpu() if hp is not None else torch.zeros(n_envs))

    cmd_t = torch.cat(cmd_a)            # [W, 3]  (W = windows*envs*seeds, flattened)
    vstar_t = torch.cat(vstar_a)        # [W, D]
    gabs_t = torch.cat(gabs_a)          # [W, D]
    sfire_t = torch.cat(sfire_a)        # [W, D]
    ach_t = torch.cat(ach_a)            # [W, D]
    keep = torch.cat(keep_a)            # [W] bool
    cmd_t, vstar_t, gabs_t, sfire_t, ach_t = (
      x[keep] for x in (cmd_t, vstar_t, gabs_t, sfire_t, ach_t)
    )
    vs, ve = vel_slice
    vstar_vel, sfire_vel, ach_vel = vstar_t[:, vs:ve], sfire_t[:, vs:ve], ach_t[:, vs:ve]
    gabs_vel = gabs_t[:, vs:ve]

    # Per-velocity-axis decomposition (signed sums close exactly: HL + LL = end).
    hl_err = (cmd_t - vstar_vel)                 # HL goal error  [W, 3]
    ll_err = (vstar_vel - ach_vel)               # LL reach error [W, 3]
    end_err = (cmd_t - ach_vel)                  # end error      [W, 3]
    axes = ("vx", "vy", "yaw")

    def _col(t, j): return t[:, j]
    # Forward (cmd_vx > 0.05) vs backward (cmd_vx < -0.05) split on vx only.
    fwd = cmd_t[:, 0] > 0.05
    bwd = cmd_t[:, 0] < -0.05

    print()
    print("=" * 70)
    print(f"  GOAL-ACHIEVABILITY PROBE | {cfg.diagnose_goals} steps x {n_envs} envs "
          f"x {cfg.eval_seeds} seed(s) | c={c} | windows kept={int(keep.sum())}/{keep.numel()}")
    print("=" * 70)
    print(f"  {'axis':<5} {'|HL goal err|':>13} {'|LL reach err|':>14} "
          f"{'|end err|':>10} {'|g|':>7} {'sat>0.95':>9}")
    for j, ax in enumerate(axes):
      print(f"  {ax:<5} {_col(hl_err, j).abs().mean():>13.4f} "
            f"{_col(ll_err, j).abs().mean():>14.4f} {_col(end_err, j).abs().mean():>10.4f} "
            f"{_col(gabs_vel, j).mean():>7.3f} {(_col(gabs_vel, j) > 0.95).float().mean():>9.3f}")
    print("  " + "-" * 66)
    print(f"  signed-closure check (mean): HL + LL == end  (should hold per axis)")
    for j, ax in enumerate(axes):
      print(f"    {ax:<4} HL {_col(hl_err, j).mean():+.4f}  + LL {_col(ll_err, j).mean():+.4f}"
            f"  = {_col(hl_err, j).mean() + _col(ll_err, j).mean():+.4f}  (end {_col(end_err, j).mean():+.4f})")
    print("  " + "-" * 66)
    print("  vx forward vs backward (the directional-bias check):")
    for label, m in (("forward", fwd), ("backward", bwd)):
      if m.any():
        print(f"    {label:<8} n={int(m.sum()):>6}  |HL|={hl_err[m, 0].abs().mean():.4f}  "
              f"|LL|={ll_err[m, 0].abs().mean():.4f}  |end|={end_err[m, 0].abs().mean():.4f}  "
              f"|g_vx|={gabs_vel[m, 0].mean():.3f}  sat={ (gabs_vel[m, 0] > 0.95).float().mean():.3f}")
    # Realized vs requested delta per velocity axis (under-reach if ratio << 1).
    req = (vstar_vel - sfire_vel).mean(0)        # mean requested delta [3]
    real = (ach_vel - sfire_vel).mean(0)         # mean realized delta  [3]
    print("  " + "-" * 66)
    print("  realized/requested delta (LL follow-through; <<1 = under-reach):")
    for j, ax in enumerate(axes):
      r = (real[j] / req[j]).item() if abs(req[j]) > 1e-3 else float("nan")
      print(f"    {ax:<4} requested {req[j]:+.4f}  realized {real[j]:+.4f}  ratio {r:+.3f}")
    print("=" * 70)
    print()

    # Pinned-command group diff (2026-07-15): split envs by late-window achieved vx and
    # print signed goals + period per group — the probe that exposed the velocity-hold HL.
    if cfg.eval_cmd_vx is not None:
      W0 = n_windows
      ach_w = torch.stack(ach_a[:W0])     # [W, B, D]
      vst_w = torch.stack(vstar_a[:W0])
      gab_w = torch.stack(gabs_a[:W0])
      per_w = torch.stack(period_a[:W0])  # [W, B]
      vx_ss = ach_w[-max(W0 // 3, 1):, :, vs].mean(0)  # [B] late-window achieved vx
      for nm, m in (("bwd(vx<0)", vx_ss < 0.0), ("fwd(vx>0.1)", vx_ss > 0.1)):
        if int(m.sum()) == 0:
          continue
        print(f"[HOLDDIAG] {nm} n={int(m.sum())}: goal_vx {vst_w[:, m, vs].mean():+.3f} "
              f"goal_vy {vst_w[:, m, vs + 1].mean():+.3f} "
              f"|g|vx {gab_w[:, m, vs].mean():.2f} |g|vy {gab_w[:, m, vs + 1].mean():.2f} "
              f"period {per_w[:, m].mean():.3f}s")
        traj_a = ach_w[:, m, vs].mean(1)
        traj_g = vst_w[:, m, vs].mean(1)
        print(f"[HOLDDIAG] {nm} ach_vx/win : "
              + " ".join(f"{v:+.2f}" for v in traj_a[:12].tolist()))
        print(f"[HOLDDIAG] {nm} goal_vx/win: "
              + " ".join(f"{v:+.2f}" for v in traj_g[:12].tolist()))

    diag_out = {"label": str(resume_path) if resume_path is not None else "unknown",
                "c": c, "kept_windows": int(keep.sum()), "total_windows": int(keep.numel())}
    for j, ax in enumerate(axes):
      diag_out[f"hl_err_{ax}"] = round(hl_err[:, j].abs().mean().item(), 6)
      diag_out[f"ll_err_{ax}"] = round(ll_err[:, j].abs().mean().item(), 6)
      diag_out[f"end_err_{ax}"] = round(end_err[:, j].abs().mean().item(), 6)
      diag_out[f"gabs_{ax}"] = round(gabs_vel[:, j].mean().item(), 6)
    if fwd.any():
      diag_out["fwd_hl_vx"] = round(hl_err[fwd, 0].abs().mean().item(), 6)
      diag_out["fwd_ll_vx"] = round(ll_err[fwd, 0].abs().mean().item(), 6)
      diag_out["fwd_gabs_vx"] = round(gabs_vel[fwd, 0].mean().item(), 6)
    if bwd.any():
      diag_out["bwd_hl_vx"] = round(hl_err[bwd, 0].abs().mean().item(), 6)
      diag_out["bwd_ll_vx"] = round(ll_err[bwd, 0].abs().mean().item(), 6)
      diag_out["bwd_gabs_vx"] = round(gabs_vel[bwd, 0].mean().item(), 6)
    print(f"[GOALDIAG] {json.dumps(diag_out)}")
    env.close()
    return

  if cfg.check_leg_odometry > 0:
    import json
    uenv = env.unwrapped
    robot = uenv.scene["robot"].data
    entity = uenv.scene["robot"]
    foot_ids, foot_names = entity.find_sites(["left_foot", "right_foot"], preserve_order=True)
    assert foot_names == ["left_foot", "right_foot"], f"foot site order: {foot_names}"
    dt = uenv.step_dt
    c = getattr(runner, "c", 8)
    n_envs = uenv.num_envs

    def _foot_b() -> torch.Tensor:
      """Foot positions in the PELVIS frame, [B, 2, 3] — a pure function of the encoders,
      so this is exactly what hardware FK would produce from `q`."""
      rel_w = robot.site_pos_w[:, foot_ids, :] - robot.root_link_pos_w[:, None, :]
      # world -> body: quat_rotate_inverse, done via the projected-gravity-free route of
      # rotating by the conjugate root quaternion.
      q = robot.root_link_quat_w                       # [B, 4] (w, x, y, z)
      qv, qw = q[:, 1:], q[:, 0:1]
      t = 2.0 * torch.cross(qv.unsqueeze(1).expand_as(rel_w), rel_w, dim=-1)
      return rel_w - qw.unsqueeze(1) * t + torch.cross(
        qv.unsqueeze(1).expand_as(t), t, dim=-1)

    est_a, true_a, keep_a, phase_a, spd_a = [], [], [], [], []
    orc_a, single_a, nct_a = [], [], []  # contact-oracle rungs

    # Physics-rate rung. d(p_foot)/dt is a finite difference, so at the 50 Hz control rate
    # it aliases exactly like the IMU integration did in --check-vel-increment (which was
    # 4x worse at 50 Hz than at 200 Hz). Hardware would run this in State_RLHRL::run() at
    # 1 kHz, so measure it at the substep rate before judging the estimator.
    fast_sum = torch.zeros(uenv.num_envs, 3, device=env.device)
    fast_n = torch.zeros((), device=env.device)
    fast_prev = {"p": None}
    _orig_substep = uenv.metrics_manager.compute_substep

    def _substep_hook():
      p_s = _foot_b()
      if fast_prev["p"] is not None:
        w_s = robot.root_link_ang_vel_b[:, None, :].expand_as(p_s)
        e_f = -(p_s - fast_prev["p"]) / uenv.physics_dt - torch.cross(w_s, p_s, dim=-1)
        d_s = (p_s * robot.projected_gravity_b[:, None, :]).sum(-1)
        i_s = d_s.argmax(dim=1).view(-1, 1, 1).expand(-1, 1, 3)
        fast_sum.add_(e_f.gather(1, i_s).squeeze(1))
        fast_n.add_(1.0)
      fast_prev["p"] = p_s
      _orig_substep()

    uenv.metrics_manager.compute_substep = _substep_hook
    fast_a = []
    with torch.inference_mode():
      for seed_idx in range(cfg.eval_seeds):
        torch.manual_seed(42 + seed_idx)
        obs, _ = env.reset()
        prev_p = _foot_b()
        for _ in range(cfg.check_leg_odometry):
          obs, _, dones, _ = env.step(policy(obs).to(env.device))
          p = _foot_b()                                  # [B, 2, 3]
          w_b = robot.root_link_ang_vel_b[:, None, :].expand_as(p)
          # v_p^b = -d(p_f^b)/dt - w^b x p_f^b   (stance foot fixed in the world), per foot
          est_f = -(p - prev_p) / dt - torch.cross(w_b, p, dim=-1)       # [B, 2, 3]
          # Stance = the foot further along gravity (hardware-computable: FK + IMU attitude).
          # MEASURED BETTER than an actuator-torque load proxy (2026-08-03): a soft-load
          # softmax regressed the gate 0.148 -> 0.241 and NO T in {1.9, 8, 20, inf, 0} beat
          # this rule. Both rules agree with a per-sample oracle only ~13% of the time while
          # the oracle itself scores 0.068 -- the headroom is real but neither depth nor
          # summed joint torque finds it. Deploy journal, 2026-08-03.
          depth = (p * robot.projected_gravity_b[:, None, :]).sum(-1)     # [B, 2]
          idx = depth.argmax(dim=1).view(-1, 1, 1).expand(-1, 1, 3)
          est = est_f.gather(1, idx).squeeze(1)
          fast_a.append((fast_sum / fast_n.clamp(min=1.0)).cpu())
          fast_sum.zero_(); fast_n.zero_()
          # CONTACT-ORACLE reference rungs: what a PERFECT contact signal would buy (a real
          # foot sensor, or a torque-transient touchdown detector). Unlike a lowest-error
          # oracle this is physically achievable, so it bounds the whole detector idea:
          # if these are not good, no contact detector can rescue leg odometry.
          from src.tasks.velocity.mdp import foot_contact
          ct = foot_contact(uenv, "feet_ground_contact") > 0.5            # [B, 2]
          single = ct.sum(-1) == 1
          idx_c = ct.float().argmax(dim=1).view(-1, 1, 1).expand(-1, 1, 3)
          est_c1 = est_f.gather(1, idx_c).squeeze(1)                     # the contacting foot
          w_c = ct.float() / ct.float().sum(-1, keepdim=True).clamp(min=1.0)
          est_cb = (w_c.unsqueeze(-1) * est_f).sum(dim=1)                # blend in double support
          orc_a.append(torch.stack([est_c1, est_cb], dim=1).cpu())
          single_a.append(single.cpu())
          nct_a.append((ct.sum(-1) == 0).cpu())
          est_a.append(est.cpu())
          true_a.append(robot.root_link_lin_vel_b.clone().cpu())
          keep_a.append((~dones.bool()).cpu())           # reset jumps the FK difference
          ph = getattr(uenv, "hrl_phase", None)
          phase_a.append(ph.clone().cpu() if ph is not None else torch.zeros(n_envs))
          spd_a.append(robot.root_link_lin_vel_b[:, :2].norm(dim=-1).cpu())
          prev_p = p

    # Window view BEFORE masking: the HL fires every c steps and can average the c
    # readings it has seen. [T, B, 3] -> [W, c, B, 3].
    est_t = torch.stack(est_a); true_t = torch.stack(true_a); keep_t = torch.stack(keep_a)
    fast_t = torch.stack(fast_a)
    n_w = est_t.shape[0] // c
    if n_w > 0:
      ew = est_t[: n_w * c].view(n_w, c, -1, 3)
      tw = true_t[: n_w * c].view(n_w, c, -1, 3)
      kw = keep_t[: n_w * c].view(n_w, c, -1).all(dim=1)          # [W, B] clean windows
      fw = fast_t[: n_w * c].view(n_w, c, -1, 3)
      fast_win = fw.mean(dim=1)[kw]       # physics-rate estimate, then c-averaged
      est_win = ew.mean(dim=1)[kw]        # averaged estimate the HL would consume
      true_win = tw.mean(dim=1)[kw]       # window-mean truth (filter quality)
      true_fire = tw[:, -1][kw]           # truth at fire time (what the HL needs)
    else:
      est_win = true_win = true_fire = fast_win = torch.zeros(0, 3)

    est = torch.cat(est_a); true = torch.cat(true_a)
    keep = torch.cat(keep_a); phase = torch.cat(phase_a); spd = torch.cat(spd_a)
    orc = torch.cat(orc_a); single = torch.cat(single_a); nct = torch.cat(nct_a)
    fast = torch.cat(fast_a)[keep]
    est, true, phase, spd = est[keep], true[keep], phase[keep], spd[keep]
    orc, single, nct = orc[keep], single[keep], nct[keep]
    err = est - true

    def _rms(t):
      return t.square().mean().sqrt().item()

    print()
    print("=" * 78)
    print(f"  LEG-ODOMETRY VELOCITY BENCH | {cfg.check_leg_odometry} steps x {n_envs} envs "
          f"x {cfg.eval_seeds} seed(s) | kept={int(keep.sum())}/{keep.numel()}")
    print("  Estimator: stance-foot kinematics + gyro. No integration -> no drift.")
    print("=" * 78)
    print(f"  {'split':<22} {'vx':>9} {'vy':>9} {'vz':>9}")
    print(f"  {'all steps':<22} " + " ".join(f"{_rms(err[:, j]):>9.4f}" for j in range(3)))
    # Gait-phase split: leg odometry is worst at flight/impact, which is exactly when the
    # HL may be firing -- a batch mean would hide it.
    for lo, hi, name in ((0.0, 0.25, "phase 0.00-0.25"), (0.25, 0.5, "phase 0.25-0.50"),
                         (0.5, 0.75, "phase 0.50-0.75"), (0.75, 1.0, "phase 0.75-1.00")):
      m = (phase >= lo) & (phase < hi)
      if m.any():
        print(f"  {name:<22} " + " ".join(f"{_rms(err[m, j]):>9.4f}" for j in range(3)))
    for lo, hi, name in ((0.0, 0.3, "speed <0.3 m/s"), (0.3, 0.8, "speed 0.3-0.8"),
                         (0.8, 9.9, "speed >0.8")):
      m = (spd >= lo) & (spd < hi)
      if m.any():
        print(f"  {name:<22} " + " ".join(f"{_rms(err[m, j]):>9.4f}" for j in range(3)))
    print(f"  {'depth @physics rate':<22} "
          + " ".join(f"{_rms(fast[:, j] - true[:, j]):>9.4f}" for j in range(3)))
    print("  " + "-" * 74)
    print(f"  {'ORACLE contact 1-foot':<22} "
          + " ".join(f"{_rms(orc[:, 0, j] - true[:, j]):>9.4f}" for j in range(3)))
    print(f"  {'ORACLE contact blend':<22} "
          + " ".join(f"{_rms(orc[:, 1, j] - true[:, j]):>9.4f}" for j in range(3)))
    m1 = single
    print(f"  {'  ..single-support only':<22} "
          + " ".join(f"{_rms(orc[m1, 0, j] - true[m1, j]):>9.4f}" for j in range(3))
          + f"   ({100.0 * m1.float().mean():.0f}% of steps)")
    print(f"  {'  ..double-support only':<22} "
          + " ".join(f"{_rms(orc[~m1 & ~nct, 1, j] - true[~m1 & ~nct, j]):>9.4f}" for j in range(3))
          + f"   ({100.0 * (~m1 & ~nct).float().mean():.0f}%)")
    print(f"  {'  ..flight (no contact)':<22} "
          + " ".join(f"{_rms(orc[nct, 1, j] - true[nct, j]):>9.4f}" for j in range(3))
          + f"   ({100.0 * nct.float().mean():.0f}%)" if nct.any() else "")
    print("  " + "-" * 74)
    if est_win.numel():
      ew_filt, ew_lag = est_win - true_win, est_win - true_fire
      print(f"  {f'c={c} avg vs win-mean':<22} "
            + " ".join(f"{_rms(ew_filt[:, j]):>9.4f}" for j in range(3))
            + "   <- filter quality")
      print(f"  {f'c={c} avg vs at-fire':<22} "
            + " ".join(f"{_rms(ew_lag[:, j]):>9.4f}" for j in range(3))
            + "   <- 50 Hz")
      fw_lag, fw_filt = fast_win - true_fire, fast_win - true_win
      print(f"  {f'c={c} avg, PHYSICS rate':<22} "
            + " ".join(f"{_rms(fw_lag[:, j]):>9.4f}" for j in range(3))
            + "   <- what the HL consumes")
      # Same rung scored against the WINDOW-MEAN truth instead of the truth at the fire
      # step: the difference between this row and the one above is purely the c/2-step
      # averaging LAG, with the estimator's own tracking error held fixed. Added
      # 2026-08-06 because the bridge measures the two apart (vy tracking 0.055 but
      # at-fire 0.10, i.e. lag-dominated) and the sim rung alone could not say whether
      # sim shares that split or has a genuinely quieter lateral channel.
      print(f"  {f'c={c} avg, PHYS filter':<22} "
            + " ".join(f"{_rms(fw_filt[:, j]):>9.4f}" for j in range(3))
            + "   <- tracking only, lag removed")
    print("=" * 78)
    gate = max(_rms(fw_lag[:, 0]), _rms(fw_lag[:, 1])) if est_win.numel() else float("nan")
    print(f"  per-step RMS  max(vx,vy) = {max(_rms(err[:, 0]), _rms(err[:, 1])):.4f} m/s")
    print(f"  gate (HL-consumed, c-averaged) = {gate:.4f} m/s")
    out = {"gate_rms": round(gate, 6), "steps": cfg.check_leg_odometry,
           "num_envs": n_envs, "eval_seeds": cfg.eval_seeds, "kept": int(keep.sum()),
           "windows": int(est_win.shape[0])}
    out["single_support_frac"] = round(single.float().mean().item(), 4)
    out["flight_frac"] = round(nct.float().mean().item(), 4)
    for j, ax in enumerate(("vx", "vy", "vz")):
      out[f"fast_{ax}"] = round(_rms(fast[:, j] - true[:, j]), 6)
      out[f"fastbias_{ax}"] = round((fast[:, j] - true[:, j]).mean().item(), 6)
      if est_win.numel():
        out[f"fasthl_{ax}"] = round(_rms(fw_lag[:, j]), 6)
        out[f"fastfilt_{ax}"] = round(_rms(fw_filt[:, j]), 6)
      out[f"orc_ct1_{ax}"] = round(_rms(orc[:, 0, j] - true[:, j]), 6)
      out[f"orc_ctb_{ax}"] = round(_rms(orc[:, 1, j] - true[:, j]), 6)
      out[f"rms_{ax}"] = round(_rms(err[:, j]), 6)
      out[f"bias_{ax}"] = round(err[:, j].mean().item(), 6)
      if est_win.numel():
        out[f"filt_{ax}"] = round(_rms(ew_filt[:, j]), 6)
        out[f"hlcons_{ax}"] = round(_rms(ew_lag[:, j]), 6)
    print(f"[LEGODOM] {json.dumps(out)}")
    env.close()
    return

  if cfg.check_vel_increment > 0:
    import json
    uenv = env.unwrapped
    robot = uenv.scene["robot"].data
    # Accelerometer + velocimeter share the `imu` site (h1_2.xml:236-238) and the site
    # carries no rotation, so both read in the pelvis ORIENTATION frame -- but at a point
    # 0.28 m above the pelvis ORIGIN whose velocity the goal space uses (goal_space.py:50).
    # That lever arm (w x r) is why the site column is reported next to the pelvis one.
    acc = uenv.scene["robot/imu_lin_acc"]
    gyro = uenv.scene["robot/imu_ang_vel"]
    vel_site = uenv.scene["robot/imu_lin_vel"]
    # MuJoCo's accelerometer includes gravity like a real IMU (verified: raw z ~ +9.95
    # against projected_gravity z ~ -0.996), so removal is a + proj_grav * |g|.
    g_mag = float(torch.tensor(uenv.cfg.sim.mujoco.gravity).norm())
    # The `imu` site sits on TORSO_LINK, behind the yaw-axis `torso_joint` whose anchor is
    # the pelvis origin (h1_2.xml:138-145) -- NOT on the pelvis whose velocity the goal
    # space uses. So pelvis <- torso is a pure Rz(psi) with coincident origins, and the
    # site offset r is a constant in the TORSO frame. Both psi and the gyro are in
    # LowState on hardware, so this correction is deployable:
    #   v_pelvis_P = Rz(psi) * (v_site_T - w_T x r_T)
    # Over a window only the INCREMENT of v_site_T is known (integrated accel), leaving an
    # unmodellable [Rz(psi_i)-Rz(psi_0)] * v_site_T(t0) residual that needs absolute
    # velocity. It is small only while the waist barely turns within the window; the RMS
    # below INCLUDES it, so this rung is what hardware would actually achieve.
    import mujoco
    _sid = mujoco.mj_name2id(uenv.sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, "robot/imu")
    assert _sid >= 0, "imu site not found; the lever-arm rung needs its torso-frame offset"
    r_T = torch.tensor(uenv.sim.mj_model.site_pos[_sid], device=env.device,
                       dtype=torch.float32).expand(uenv.num_envs, 3)
    _tj, _ = uenv.scene["robot"].find_joints([".*torso_joint.*"])
    assert len(_tj) == 1, f"expected exactly one torso_joint, got {_tj}"
    torso_idx = _tj[0]

    def _rz(psi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
      """Rotate v about z by psi (per-env). psi<0 gives the inverse."""
      c_, s_ = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
      x, y, z = v[:, 0:1], v[:, 1:2], v[:, 2:3]
      return torch.cat([c_ * x - s_ * y, s_ * x + c_ * y, z], dim=-1)
    c = getattr(runner, "c", 8)
    dt = uenv.step_dt
    n_envs = uenv.num_envs
    n_windows = cfg.check_vel_increment // c

    # Physics-rate integration probe. The decimation loop calls metrics_manager.
    # compute_substep() once per substep, right after scene.update refreshes sensordata
    # (manager_based_rl_env.py:421-427) -- the only hook that sees the 200 Hz signal a real
    # IMU publishes, vs the 50 Hz control rate the other rungs are stuck with. Tests whether
    # the residual is aliased foot-impact transients rather than true estimator error.
    dv_fast = torch.zeros(uenv.num_envs, 3, device=env.device)
    _orig_substep = uenv.metrics_manager.compute_substep

    def _substep_hook():
      psi_s = robot.joint_pos[:, torso_idx]
      dv_fast.add_((acc.data + _rz(-psi_s, robot.projected_gravity_b) * g_mag)
                   * uenv.physics_dt)
      _orig_substep()

    uenv.metrics_manager.compute_substep = _substep_hook

    # Per-(window, env) rows, accumulated across seeds.
    err_a, keep_a, yaw_a = [], [], []
    lever_a = []  # |site increment - pelvis increment|: the lever-arm share
    with torch.inference_mode():
      for seed_idx in range(cfg.eval_seeds):
        torch.manual_seed(42 + seed_idx)
        obs, _ = env.reset()
        for _ in range(n_windows):
          v0_root = robot.root_link_lin_vel_b.clone()
          v0_site = vel_site.data.clone()
          psi0 = robot.joint_pos[:, torso_idx].clone()
          w0_T = gyro.data.clone()
          dv_fast.zero_()  # in-place so the substep hook keeps writing the same buffer
          dv_raw = torch.zeros_like(v0_root)
          dv_grav = torch.zeros_like(v0_root)
          dv_cor = torch.zeros_like(v0_root)
          done_in_window = torch.zeros(n_envs, dtype=torch.bool, device=env.device)
          yaw_abs = torch.zeros(n_envs, device=env.device)
          for _ in range(c):
            obs, _, dones, _ = env.step(policy(obs).to(env.device))
            done_in_window |= dones.bool()
            a = acc.data
            psi = robot.joint_pos[:, torso_idx]
            # Gravity must be removed in the TORSO frame the accelerometer reads in;
            # projected_gravity_b is the PELVIS frame, so undo the waist yaw first.
            a_g = a + _rz(-psi, robot.projected_gravity_b) * g_mag
            w = robot.root_link_ang_vel_b
            # d(v_b)/dt = a_b - w x v_b. The Coriolis rung uses the TRUE v_b, so it is the
            # privileged floor (hardware has no absolute v), not a deployable variant.
            dv_raw = dv_raw + a * dt
            dv_grav = dv_grav + a_g * dt
            dv_cor = dv_cor + (a_g - torch.cross(w, robot.root_link_lin_vel_b, dim=-1)) * dt
            yaw_abs = yaw_abs + w[:, 2].abs() / c
          true_root = robot.root_link_lin_vel_b - v0_root
          true_site = vel_site.data - v0_site
          # Deployable rung: pelvis increment from the integrated torso-site increment,
          # the gyro and the waist encoder (all present in LowState). Nothing privileged.
          psi_e, w_e = robot.joint_pos[:, torso_idx], gyro.data
          lev = torch.cross(w_e, r_T, dim=-1)
          lev0 = _rz(psi0, torch.cross(w0_T, r_T, dim=-1))
          dv_waist = _rz(psi_e, dv_grav - lev) + lev0
          dv_waist_fast = _rz(psi_e, dv_fast - lev) + lev0
          err_a.append(torch.stack([
            dv_raw - true_root, dv_grav - true_root, dv_cor - true_root,
            dv_grav - true_site, dv_waist - true_root, dv_waist_fast - true_root,
          ], dim=1).cpu())                      # [B, 6 rungs, 3 axes]
          lever_a.append((true_site - true_root).cpu())
          keep_a.append((~done_in_window).cpu())  # drop reset-contaminated windows
          yaw_a.append(yaw_abs.cpu())

    err = torch.cat(err_a)      # [W, 4, 3]
    lever = torch.cat(lever_a)  # [W, 3]
    keep = torch.cat(keep_a)
    yaw = torch.cat(yaw_a)
    err, lever, yaw = err[keep], lever[keep], yaw[keep]
    rungs = ("raw (no grav)", "grav-removed", "+coriolis(true v)", "grav vs SITE v",
             "waist-corr @50Hz", "waist-corr @200Hz")
    axes = ("vx", "vy", "vz")

    def _rms(t):  # t: [W] -> scalar RMS
      return t.square().mean().sqrt().item()

    print()
    print("=" * 78)
    print(f"  IMU VELOCITY-INCREMENT BENCH | {cfg.check_vel_increment} steps x {n_envs} envs "
          f"x {cfg.eval_seeds} seed(s) | c={c} window={c * dt:.3f}s")
    print(f"  windows kept={int(keep.sum())}/{keep.numel()} | RMS error (m/s) vs the true "
          f"within-window increment")
    print("=" * 78)
    print(f"  {'reconstruction':<20} {'vx':>9} {'vy':>9} {'vz':>9}")
    for i, name in enumerate(rungs):
      print(f"  {name:<20} " + " ".join(f"{_rms(err[:, i, j]):>9.4f}" for j in range(3)))
    print("  " + "-" * 74)
    print(f"  {'lever arm (site-pelvis)':<20} "
          + " ".join(f"{_rms(lever[:, j]):>9.4f}" for j in range(3)))
    # Yaw-rate split: Coriolis and lever-arm both scale with |w|, so a gap here localizes them.
    hi = yaw > yaw.median()
    print(f"  {'waist-corr |w|<med':<20} "
          + " ".join(f"{_rms(err[~hi, 4, j]):>9.4f}" for j in range(3)))
    print(f"  {'waist-corr |w|>med':<20} "
          + " ".join(f"{_rms(err[hi, 4, j]):>9.4f}" for j in range(3)))
    print("=" * 78)

    # Verdict on the horizontal axes the goal channel carries, from the DEPLOYABLE rung.
    # That is the @200Hz one: the deploy integrator belongs in State_RLHRL::run(), the 1 kHz
    # control loop, NOT in the 50 Hz policy_step() -- and 200 Hz is a conservative stand-in
    # for 1 kHz. The 50 Hz rung is kept only to show how much of the residual is aliasing.
    # The other rungs are diagnostics; two of them use privileged state no hardware has.
    gate = max(_rms(err[:, 5, 0]), _rms(err[:, 5, 1]))
    verdict = ("SHIP (no retrain)" if gate <= 0.05 else
               "STATE_NOISE DR retrain" if gate <= 0.15 else "BLIND-LL SWEEP")
    print(f"  gate = max(vx, vy) waist-corrected = {gate:.4f} m/s  ->  {verdict}")
    print(f"  (bands: <=0.05 ship | 0.05-0.15 DR retrain | >0.15 blind sweep)")
    out = {"gate_rms": round(gate, 6), "verdict": verdict, "window_s": round(c * dt, 4),
           "windows": int(keep.sum()), "steps": cfg.check_vel_increment,
           "num_envs": n_envs, "eval_seeds": cfg.eval_seeds}
    for i, name in enumerate(("raw", "grav", "cor", "grav_site", "waist", "waist_fast")):
      for j, ax in enumerate(axes):
        out[f"{name}_{ax}"] = round(_rms(err[:, i, j]), 6)
    for j, ax in enumerate(axes):
      out[f"lever_{ax}"] = round(_rms(lever[:, j]), 6)
    print(f"[VELINC] {json.dumps(out)}")
    env.close()
    return

  if cfg.probe_lean > 0:
    import json
    uenv = env.unwrapped
    entity = uenv.scene["robot"]
    robot = entity.data
    biases = [float(x) for x in cfg.probe_lean_bias.split(",")]
    groups = [g.strip() for g in cfg.probe_lean_joints.split(",")]
    pats = {"ankle_pitch": [".*ankle_pitch.*"], "knee": [".*knee.*"],
            "hip_pitch": [".*hip_pitch.*"],
            "all": [".*hip_pitch.*", ".*knee.*", ".*ankle_pitch.*"]}
    # "all" spreads the swept value over the 3 sagittal joints so the chain sum matches it.
    n_chain = {"all": 3}

    print("=" * 78)
    print(f"  BACKWARD-LEAN SENSITIVITY PROBE | {cfg.probe_lean} steps x {uenv.num_envs} "
          f"envs x {cfg.eval_seeds} seed(s)")
    print("  pitch < 0 = leaning BACKWARD (same sign as the flight recorder's p_IMU).")
    print("  bias b <=> deploy joint_offset -b. Reference: real stand p_IMU = -0.0668 rad.")
    print("=" * 78)

    leg_ids, _ = entity.find_joints([".*hip.*", ".*knee.*", ".*ankle.*"], preserve_order=False)
    leg_ids = torch.as_tensor(sorted(leg_ids), device=env.device)

    def rollout(ids, per_joint):
      """One condition -> (steady-state pitch, mean leg joint speed, mean action rate),
      each averaged over the last third of the rollout and over seeds."""
      out = []
      for seed_idx in range(cfg.eval_seeds):
        torch.manual_seed(42 + seed_idx)
        with torch.inference_mode():
          obs, _ = env.reset()
          # After reset: encoder_bias is a startup event, so this overwrite is the value
          # that stands for the whole rollout.
          robot.encoder_bias[:] = 0.0
          if ids is not None:
            robot.encoder_bias[:, ids] = per_joint
          tail, dq, ar = [], [], []
          prev = None
          for step_i in range(cfg.probe_lean):
            act = policy(obs)
            obs, _, _, _ = env.step(act.to(env.device))
            if step_i >= (2 * cfg.probe_lean) // 3:  # steady state = last third
              g = robot.projected_gravity_b
              tail.append(torch.atan2(g[:, 0], -g[:, 2]).mean().item())
              dq.append(robot.joint_vel[:, leg_ids].abs().mean().item())
              if prev is not None:
                ar.append((act - prev).norm(dim=-1).mean().item())
            prev = act.clone()
        n = max(len(tail), 1)
        out.append((sum(tail) / n, sum(dq) / n, sum(ar) / max(len(ar), 1)))
      k = len(out)
      mean = [sum(o[i] for o in out) / k for i in range(3)]
      sd = (sum((o[0] - mean[0]) ** 2 for o in out) / k) ** 0.5
      return mean[0], sd, mean[1], mean[2]

    rows = []
    if cfg.probe_gravity_noise:
      # Gravity-corruption sweep (ADR-0006 discriminator). The obs manager reads
      # `term_cfg.noise` live every step, so mutating it here takes effect immediately.
      from mjlab.utils.noise.noise_cfg import UniformNoiseCfg
      om = uenv.observation_manager
      gterm = next(
        c for n, c in zip(om.active_terms["actor"],
                          om._group_obs_term_cfgs["actor"], strict=False)  # noqa: SLF001
        if n == "projected_gravity"
      )
      base = gterm.noise
      assert base is not None, "projected_gravity has no noise term to sweep"
      print("  MODE: projected_gravity corruption (encoder held CLEAN). 0.05 = training "
            "default = control.")
      print("-" * 78)
      for nz in [float(x) for x in cfg.probe_gravity_noise.split(",")]:
        gterm.noise = UniformNoiseCfg(n_min=-nz, n_max=nz, operation=base.operation)
        p, s, dq, ar = rollout(None, 0.0)
        rows.append({"grav_noise": nz, "pitch": p, "pitch_std": s, "leg_dq": dq,
                     "action_rate": ar})
        print(f"  grav_noise +-{nz:<6.3f} -> pitch {p:+.4f} +- {s:.4f} rad "
              f"({math.degrees(p):+6.2f} deg)  leg|dq| {dq:6.3f}  act_rate {ar:6.3f}")
      gterm.noise = base
    else:
      for grp in groups:
        ids, _ = entity.find_joints(pats[grp], preserve_order=False)
        ids = torch.as_tensor(sorted(ids), device=env.device)
        for b in biases:
          per_joint = b / n_chain.get(grp, 1)
          m, s, dq, ar = rollout(ids, per_joint)
          rows.append({"group": grp, "bias": b, "per_joint": per_joint,
                       "joint_offset": -b, "pitch": m, "pitch_std": s, "leg_dq": dq,
                       "action_rate": ar})
          print(f"  {grp:<11} bias {b:+.4f} (per-joint {per_joint:+.4f}, "
                f"joint_offset {-b:+.4f}) -> pitch {m:+.4f} +- {s:.4f} rad "
                f"({math.degrees(m):+.2f} deg)  leg|dq| {dq:5.3f}  act {ar:5.3f}")

    print("-" * 78)
    if cfg.probe_gravity_noise:
      # Degradation relative to the training-default noise: the quantity to compare
      # BETWEEN policies. A policy that leans on the IMU should degrade faster.
      ctl = min(rows, key=lambda r: abs(r["grav_noise"] - 0.05))
      print(f"  degradation vs the +-{ctl['grav_noise']:.3f} control "
            f"(leg|dq| {ctl['leg_dq']:.3f}, act {ctl['action_rate']:.3f}):")
      for r in rows:
        print(f"    grav_noise +-{r['grav_noise']:<6.3f} leg|dq| x{r['leg_dq']/ctl['leg_dq']:5.2f}  "
              f"act x{r['action_rate']/ctl['action_rate']:5.2f}  "
              f"pitch drift {math.degrees(r['pitch']-ctl['pitch']):+6.2f} deg")
      print("=" * 78)
      print(f"[LEANPROBE] {json.dumps({'label': str(resume_path) if resume_path else 'unknown', 'mode': 'gravity', 'rows': rows})}")
      env.close()
      return
    print("  sensitivity d(pitch)/d(bias), least-squares through the swept points:")
    for grp in groups:
      pts = [(r["bias"], r["pitch"]) for r in rows if r["group"] == grp]
      if len(pts) < 2:
        continue
      mb = sum(p[0] for p in pts) / len(pts)
      mp = sum(p[1] for p in pts) / len(pts)
      den = sum((p[0] - mb) ** 2 for p in pts)
      slope = sum((p[0] - mb) * (p[1] - mp) for p in pts) / den if den else float("nan")
      # What chain bias explains the real lean? Measured as a DELTA from this policy's own
      # unbiased stand, which is not itself at 0 (the sim policy has its own small offset).
      p0 = next((p[1] for p in pts if p[0] == 0.0), mp - slope * mb)
      need = (-0.0668 - p0) / slope if slope else float("nan")
      print(f"    {grp:<11} d(pitch)/d(bias) = {slope:+7.3f} rad/rad  "
            f"=> bias explaining the full -0.0668 rad lean: {need:+.4f} rad "
            f"({math.degrees(need):+.2f} deg)")
    print("=" * 78)
    print(f"[LEANPROBE] {json.dumps({'label': str(resume_path) if resume_path else 'unknown', 'rows': rows})}")
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
