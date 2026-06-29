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
                        "hl_obs_vel")
      restored = {k: saved[k] for k in structure_keys
                  if k in saved and hasattr(agent_cfg, k)}
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

    policy = runner.get_inference_policy(device=device)

  # Headless deterministic benchmark: roll out `eval_steps` steps x `eval_seeds` seeds.
  if cfg.eval_steps > 0:
    import json
    uenv = env.unwrapped
    robot = uenv.scene["robot"].data
    n_envs = uenv.num_envs
    tm = uenv.termination_manager
    has_fell = "fell_over" in tm.active_terms

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
      prev_actions: torch.Tensor | None = None

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

      def _m(lst): return float(torch.tensor(lst).mean())  # noqa: E731

      seed_results.append({
        "err_vx":      _m(errs_vx),
        "err_vy":      _m(errs_vy),
        "err_yaw":     _m(errs_yaw),
        "fall_rate":   _m(fall_flags),
        "mean_ep_len": _m(ep_lens),
        "action_rate": _m(action_rates) if action_rates else float("nan"),
        "orient_dev":  _m(orient_devs),
        "height_dev":  _m(height_devs),
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
    print("=" * 58)
    print()

    bench_out = {**bench_meta, **{k: round(means[k], 6) for k in keys},
                 **{f"{k}_std": round(stds[k], 6) for k in keys},
                 "eval_steps": cfg.eval_steps, "num_envs": n_envs, "eval_seeds": cfg.eval_seeds}
    print(f"[BENCH] {json.dumps(bench_out)}")
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
    for seed_idx in range(cfg.eval_seeds):
      torch.manual_seed(42 + seed_idx)
      with torch.inference_mode():
        obs, _ = env.reset()
      with torch.inference_mode():
        for _ in range(n_windows):
          s_fire = gs.extract(uenv)                          # [B, D]
          v_star = runner.hl.act_inference(uenv, obs, s_fire)  # [B, D]
          scale = gs.scale(uenv)                             # [D]
          g_raw = ((v_star - s_fire) / scale).abs()         # [B, D] recovered |g|
          cmd = uenv.command_manager.get_command("twist")[:, :3].clone()  # [B, 3]
          done_in_window = torch.zeros(n_envs, dtype=torch.bool, device=env.device)
          for k in range(c):
            cur = gs.extract(uenv)
            delta = v_star - cur
            uenv.hrl_goal = delta
            obs["goal"] = delta
            obs, _, dones, _ = env.step(ll_policy(obs).to(env.device))
            done_in_window |= dones.bool()
          achieved = gs.extract(uenv)                        # [B, D] window end
          cmd_a.append(cmd.cpu()); vstar_a.append(v_star.cpu()); gabs_a.append(g_raw.cpu())
          sfire_a.append(s_fire.cpu()); ach_a.append(achieved.cpu())
          keep_a.append((~done_in_window).cpu())             # drop reset-contaminated windows

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
