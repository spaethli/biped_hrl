"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv
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
  diagnose_symmetry: int = 0
  """WL-D arm 10 probe: if > 0, run this many deterministic steps and report per-foot
  touchdown counts, the t_LR/t_RL step-time distributions and symmetry index
  (SI = |t_LR-t_RL|/(t_LR+t_RL)), same-foot "double-tap" repeat counts, the realized
  double-support fraction, per-foot gait_match (not averaged over feet), and whether
  play.py's own stride_period_s metric (which divides by raw touchdown count) is
  corrupted by double-taps vs an alternation-based estimate. Sibling to ``eval_steps``;
  reuses ``eval_seeds``."""
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

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

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
    # CoT / cadence metrics (A1a M0): mechanical power, cost of transport, achieved stride period.
    try:
      contact_sensor = uenv.scene["feet_ground_contact"]
    except KeyError:
      contact_sensor = None
    step_dt = uenv.step_dt
    MASS_G = 75.0 * 9.81  # H1-2 ~75 kg; dimensionless CoT = energy / (m g distance)

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
      achieved_vxs = []  # per-step env-mean achieved vx (ramp metric for --eval-cmd-vx holds)
      prev_actions: torch.Tensor | None = None
      energy_eng = dist_eng = td_count = 0.0  # CoT numerator/denominator + footfall count
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
          cmd = uenv.command_manager.get_command("twist")  # [B, >=3]
          achieved = torch.cat(
            [robot.root_link_lin_vel_b[:, :2], robot.root_link_ang_vel_b[:, 2:3]], dim=-1
          )
          ae = (cmd[:, :3] - achieved).abs()  # [B, 3]
          errs_vx.append(ae[:, 0].mean().item())
          errs_vy.append(ae[:, 1].mean().item())
          errs_yaw.append(ae[:, 2].mean().item())
          achieved_vxs.append(achieved[:, 0].mean().item())

          # 2. Survival: fall flag and episode length.
          if has_fell:
            fall_flags.append(tm.get_term("fell_over").float().mean().item())
          else:
            fall_flags.append(dones.float().mean().item())
          ep_lens.append(uenv.episode_length_buf.float().mean().item())

          # 3. Smoothness: action rate ||a_t - a_{t-1}||.
          if prev_actions is not None:
            action_rates.append((actions - prev_actions).norm(dim=-1).mean().item())
          prev_actions = actions.clone()

          # 4. Stability: orientation deviation + height deviation.
          orient_devs.append(
            robot.projected_gravity_b[:, :2].norm(dim=-1).mean().item()
          )
          nom_h = robot.default_root_state[:, 2]  # [B] nominal height
          height_devs.append(
            (robot.root_link_pos_w[:, 2] - nom_h).abs().mean().item()
          )

          # 5. Upper-body deploy hygiene: arm+waist drift from default + joint speed.
          ub_pose_devs.append(
            (robot.joint_pos[:, ub_ids] - robot.default_joint_pos[:, ub_ids])
            .square().mean(dim=1).mean().item()
          )
          ub_arm_vels.append(robot.joint_vel[:, ub_ids].abs().mean().item())

          # 6. Mechanical power + cost of transport (gated by commanded linear speed > 0.1).
          power = (robot.qfrc_actuator * robot.joint_vel).abs().sum(dim=1)  # [B] watts
          powers.append(power.mean().item())
          lin_speed = robot.root_link_lin_vel_b[:, :2].norm(dim=-1)  # [B] achieved m/s
          eng = (cmd[:, :2].norm(dim=-1) > 0.1).float()  # commanded-motion gate
          energy_eng += (power * eng).sum().item() * step_dt
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
        "orient_dev":  _m(orient_devs),
        "height_dev":  _m(height_devs),
        "ub_pose_dev": _m(ub_pose_devs),
        "ub_arm_vel":  _m(ub_arm_vels),
        "mech_power_w": _m(powers) if powers else float("nan"),
        "gait_match":   _m(gait_matches) if gait_matches else float("nan"),
        "cot":          energy_eng / (dist_eng * MASS_G + 1e-6),
        "stride_period_s": (cfg.eval_steps * step_dt)
                           / max(td_count / max(n_envs * max(n_feet, 1), 1), 1e-6),
      })

    # Aggregate across seeds.
    keys = list(seed_results[0].keys())
    vals = {k: torch.tensor([r[k] for r in seed_results]) for k in keys}
    means = {k: vals[k].mean().item() for k in keys}
    stds  = {k: vals[k].std().item() if cfg.eval_seeds > 1 else float("nan") for k in keys}

    def _fmt(k): return f"{means[k]:.4f} ± {stds[k]:.4f}" if cfg.eval_seeds > 1 else f"{means[k]:.4f}"  # noqa: E731

    print()
    print("=" * 58)
    print(f"  BENCHMARK SCORECARD  |  {cfg.eval_steps} steps x {n_envs} envs x {cfg.eval_seeds} seed(s)")
    print("=" * 58)
    print(f"  Tracking  err_vx    : {_fmt('err_vx')}")
    print(f"  Tracking  err_vy    : {_fmt('err_vy')}")
    print(f"  Tracking  err_yaw   : {_fmt('err_yaw')}")
    print(f"  Survival  fall_rate : {_fmt('fall_rate')}")
    print(f"  Survival  ep_len    : {_fmt('mean_ep_len')}")
    print(f"  Smoothness act_rate : {_fmt('action_rate')}")
    print(f"  Stability orient_dev: {_fmt('orient_dev')}")
    print(f"  Stability height_dev: {_fmt('height_dev')}")
    print(f"  UpperBody pose_dev  : {_fmt('ub_pose_dev')}")
    print(f"  UpperBody arm_vel   : {_fmt('ub_arm_vel')}")
    print(f"  Energy    power_W   : {_fmt('mech_power_w')}")
    print(f"  Energy    CoT (norm): {_fmt('cot')}")
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
