"""--check-leg-odometry: the stance-foot kinematics + gyro velocity estimator against ground truth,
per step, per HL window (what the HL would consume) and at the physics rate (`[LEGODOM]`)."""

import json

import torch

from ..context import foot_pos_b
from ..rollout import Observer, rollout


class LegOdometry(Observer):

  def __init__(self, ctx):
    env = ctx.env
    uenv = self.uenv = env.unwrapped
    robot = self.robot = uenv.scene["robot"].data
    entity = uenv.scene["robot"]
    foot_ids, foot_names = entity.find_sites(["left_foot", "right_foot"], preserve_order=True)
    assert foot_names == ["left_foot", "right_foot"], f"foot site order: {foot_names}"
    self.foot_ids = foot_ids
    self.dt = uenv.step_dt
    self.c = getattr(ctx.runner, "c", 8)
    self.n_envs = uenv.num_envs

    self.est_a, self.true_a, self.keep_a, self.phase_a, self.spd_a = [], [], [], [], []
    self.orc_a, self.single_a, self.nct_a = [], [], []  # contact-oracle rungs

    # Physics-rate rung. d(p_foot)/dt is a finite difference, so at the 50 Hz control rate
    # it aliases exactly like the IMU integration did in --check-vel-increment (which was
    # 4x worse at 50 Hz than at 200 Hz). Hardware would run this in State_RLHRL::run() at
    # 1 kHz, so measure it at the substep rate before judging the estimator.
    self.fast_sum = fast_sum = torch.zeros(uenv.num_envs, 3, device=env.device)
    self.fast_n = fast_n = torch.zeros((), device=env.device)
    fast_prev = {"p": None}
    _orig_substep = uenv.metrics_manager.compute_substep

    def _substep_hook():
      p_s = self._foot_b()
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
    self.fast_a = []

  def _foot_b(self) -> torch.Tensor:
    return foot_pos_b(self.robot, self.foot_ids)

  def reset(self):
    self.prev_p = self._foot_b()

  def after_step(self, step_i, actions, dones):
    uenv, robot, dt = self.uenv, self.robot, self.dt
    p = self._foot_b()                                  # [B, 2, 3]
    w_b = robot.root_link_ang_vel_b[:, None, :].expand_as(p)
    # v_p^b = -d(p_f^b)/dt - w^b x p_f^b   (stance foot fixed in the world), per foot
    est_f = -(p - self.prev_p) / dt - torch.cross(w_b, p, dim=-1)       # [B, 2, 3]
    # Stance = the foot further along gravity (hardware-computable: FK + IMU attitude).
    # MEASURED BETTER than an actuator-torque load proxy (2026-08-03): a soft-load
    # softmax regressed the gate 0.148 -> 0.241 and NO T in {1.9, 8, 20, inf, 0} beat
    # this rule. Both rules agree with a per-sample oracle only ~13% of the time while
    # the oracle itself scores 0.068 -- the headroom is real but neither depth nor
    # summed joint torque finds it. Deploy journal, 2026-08-03.
    depth = (p * robot.projected_gravity_b[:, None, :]).sum(-1)     # [B, 2]
    idx = depth.argmax(dim=1).view(-1, 1, 1).expand(-1, 1, 3)
    est = est_f.gather(1, idx).squeeze(1)
    self.fast_a.append((self.fast_sum / self.fast_n.clamp(min=1.0)).cpu())
    self.fast_sum.zero_(); self.fast_n.zero_()
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
    self.orc_a.append(torch.stack([est_c1, est_cb], dim=1).cpu())
    self.single_a.append(single.cpu())
    self.nct_a.append((ct.sum(-1) == 0).cpu())
    self.est_a.append(est.cpu())
    self.true_a.append(robot.root_link_lin_vel_b.clone().cpu())
    self.keep_a.append((~dones.bool()).cpu())           # reset jumps the FK difference
    ph = getattr(uenv, "hrl_phase", None)
    self.phase_a.append(ph.clone().cpu() if ph is not None else torch.zeros(self.n_envs))
    self.spd_a.append(robot.root_link_lin_vel_b[:, :2].norm(dim=-1).cpu())
    self.prev_p = p


def run(ctx) -> dict:
  cfg = ctx.cfg
  probe = LegOdometry(ctx)
  rollout(ctx, cfg.check_leg_odometry, [probe])
  c, n_envs = probe.c, probe.n_envs

  # Window view BEFORE masking: the HL fires every c steps and can average the c
  # readings it has seen. [T, B, 3] -> [W, c, B, 3].
  est_t = torch.stack(probe.est_a); true_t = torch.stack(probe.true_a); keep_t = torch.stack(probe.keep_a)
  fast_t = torch.stack(probe.fast_a)
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

  est = torch.cat(probe.est_a); true = torch.cat(probe.true_a)
  keep = torch.cat(probe.keep_a); phase = torch.cat(probe.phase_a); spd = torch.cat(probe.spd_a)
  orc = torch.cat(probe.orc_a); single = torch.cat(probe.single_a); nct = torch.cat(probe.nct_a)
  fast = torch.cat(probe.fast_a)[keep]
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
  return out
