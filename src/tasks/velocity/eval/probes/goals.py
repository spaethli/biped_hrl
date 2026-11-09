"""--diagnose-goals: the A1 goal-achievability probe. Decomposes tracking error into HL goal error
vs LL reach error, per HL window. Mirrors HierarchicalRunner.get_inference_policy goal wiring (fire
hl.act_inference every c steps; LL sees the remaining delta) but captures s_fire / V* / command at
each fire and `achieved` at window end. Keeps its own loop: it drives the high level itself
(ADR-0003, the deliberate clean-eval mirror). Also prints [HOLDDIAG] under a pinned command and
[PERIODDIAG] on cadence checkpoints."""

import json

import torch

from src.tasks.velocity.mdp.observations import env_latent_e


def run(ctx) -> dict:
  from period_payload_stats import period_payload_stats   # scripts/, on play.py's path
  cfg, env, runner, device = ctx.cfg, ctx.env, ctx.runner, ctx.device
  _cf_cols, _cf_vals = ctx.cf_cols, ctx.cf_vals
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
        # WP5: the probe calls act_inference directly, so it must write the privileged
        # latent itself -- get_inference_policy/learn do it at their own fire steps
        # (hrl_runner.py get_inference_policy / learn) and this path bypasses both. Without it
        # td3._state_vec raises KeyError('hl_e') on every hl_obs_e checkpoint.
        if getattr(runner, "hl_obs_e", False):
          obs["hl_e"] = env_latent_e(uenv, runner._e_torso_cfg, runner._e_foot_cfg)
          if cfg.cf_e_col is not None:
            for _c, _v in zip(_cf_cols, _cf_vals):
              obs["hl_e"][:, _c] = _v
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

  # WP5 pre-flight / Bar B: commanded stride period vs the privileged latent e.
  # `e` is drawn by mode="startup" DR (fires at construction, NOT on reset), so each
  # env's payload/CoM/friction is constant for the whole process and per-env pairing
  # across seeds is valid. Bar B correlates the PER-ENV MEAN commanded T against the
  # per-env true payload; the spread of that mean at fixed payload is its noise floor.
  per_w = torch.stack(period_a)                  # [W_total, B] commanded T at each fire
  keep_w = torch.stack(keep_a)                   # [W_total, B] reset-free window mask
  if per_w.abs().sum() > 0:
    e_cfg = uenv.observation_manager.get_term_cfg("critic", "env_latent_e").params
    e = env_latent_e(uenv, e_cfg["torso_cfg"], e_cfg["foot_cfg"]).cpu()      # [B, 5]
    pdiag = period_payload_stats(per_w, keep_w, e)
    # Stamp the falsification into the record: `e` here is the TRUE latent, so a
    # counterfactual run's per-env columns describe physics the HL never saw. Without
    # this the two arms' JSON is indistinguishable.
    pdiag["cf_e_col"] = cfg.cf_e_col
    pdiag["cf_e_val"] = cfg.cf_e_val
    if cfg.cf_e_col is not None:
      print(f"  [CF] HL read e[:, {list(_cf_cols)}] = {list(_cf_vals)} (FALSIFIED); physics unchanged.")
    print(f"  Period    mean {pdiag['period_mean']:.4f}s  sd_env {pdiag['period_sd_env']:.4f}  "
          f"sd_win {pdiag['period_sd_win']:.4f}  p5-p95 [{pdiag['period_p5']:.4f}, "
          f"{pdiag['period_p95']:.4f}]")
    print(f"  Bar B     r(T, payload) = {pdiag['r_period_payload']}  "
          f"slope {pdiag['slope_period_payload']} s/kg  "
          f"(payload {pdiag['payload_mean']:.2f} +- {pdiag['payload_sd']:.2f} kg)")
    print(f"[PERIODDIAG] {json.dumps(pdiag)}")

  diag_out = {"label": ctx.label,
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
  return diag_out
