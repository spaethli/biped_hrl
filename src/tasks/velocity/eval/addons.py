"""Observers any rollout probe can attach: the schedule replay, the trace, the reward-term table,
the reward dump and the commanded joint-limit check. `build_addons` wires them from the flags in
the order rollout.py relies on: the trace records the command the policy saw BEFORE the schedule
pins the next one, and the reward add-on reads the command the step will be scored on AFTER it."""

import json

import torch

from .context import pin_twist
from .rollout import Observer


class Schedule(Observer):
  """--eval-cmd-seq: replay one (vx, vy, wz) per step in every env."""

  def __init__(self, ctx):
    self.env, self.seq_cmds = ctx.env, ctx.seq_cmds

  def seed_begin(self, seed_idx, seed):
    pin_twist(self.env, *self.seq_cmds[0])

  def before_step(self, step_i, actions):
    seq_cmds = self.seq_cmds
    if step_i + 1 < len(seq_cmds) and seq_cmds[step_i + 1] != seq_cmds[step_i]:
      pin_twist(self.env, *seq_cmds[step_i + 1])   # so the NEXT observation carries it


class Trace(Observer):
  """--eval-trace: per-step arrays a flight recorder would hold, one npz per seed."""

  def __init__(self, ctx):
    import bridge_session
    self.ctx = ctx
    env, uenv = ctx.env, ctx.env.unwrapped
    self.uenv, self.robot, self.runner = uenv, uenv.scene["robot"].data, ctx.runner
    self.n_envs = uenv.num_envs
    self.tr = None
    self.rew_names = None   # set by Reward when it adds the per-term reward row
    # --eval-trace: joints go out in DEPLOY slot order, matched by name (the sim calls slot 12
    # `torso`, the deploy header `waist_yaw`), so the scorer's LEG/UB slots mean the same joints.
    self._lim = _lim = bridge_session.deploy_limit_rows()
    _sim = [{"torso": "waist_yaw"}.get(n, n) for n in
            (x.replace("_joint", "").lower() for x in uenv.scene["robot"].joint_names)]
    _at = uenv.action_manager.get_term("joint_pos")
    _tgt = getattr(_at, "_target_ids", None)
    _act = [_sim[i] for i in (range(len(_sim)) if _tgt is None else _tgt)]
    self.tr_pj = torch.as_tensor([_sim.index(n) for _, _, n in _lim], device=env.device)
    self.tr_pa = torch.as_tensor([_act.index(n) for _, _, n in _lim], device=env.device)
    self.tr_torso = uenv.scene["robot"].find_joints([".*torso_joint.*"])[0][0]
    self.tr_gyro = uenv.scene["robot/imu_ang_vel"]
    _om = uenv.observation_manager
    self.tr_phase = _om.get_term_cfg(next(g for g, ts in _om.active_terms.items() if "phase" in ts), "phase")

  def loop_begin(self, steps):
    env, n_envs, runner = self.ctx.env, self.n_envs, self.runner
    self.steps = steps
    _z = lambda d: torch.zeros(steps, n_envs, d, device=env.device)  # noqa: E731
    tr = self.tr = {k: _z(d) for k, d in (("cmd", 3), ("raw_q", 27), ("q", 27), ("dq", 27), ("tau", 27),
                                          ("quat", 4), ("gyro", 3), ("ach_v", 2), ("phase", 2))}
    self.tr_fell = torch.full((n_envs,), -1, device=env.device)
    if hasattr(runner, "goal_space"):   # hierarchy: the low level's goal V* and achieved state
      tr["vstar"], tr["ach"] = _z(runner.goal_space.dim), _z(runner.goal_space.dim)

  def before_step(self, step_i, actions):
    tr, robot, uenv, tr_pj, tr_torso = self.tr, self.robot, self.uenv, self.tr_pj, self.tr_torso
    # The state the policy just acted on, paired with the command it saw: what one
    # flight-recorder tick holds. Attitude and gyro are the IMU body's (torso_link, behind
    # the waist yaw), not the pelvis the inline metrics read.
    tr["cmd"][step_i] = uenv.command_manager.get_command("twist")[:, :3]
    tr["q"][step_i] = robot.joint_pos[:, tr_pj]
    tr["dq"][step_i] = robot.joint_vel[:, tr_pj]
    tr["tau"][step_i] = robot.qfrc_actuator[:, tr_pj]
    _w, _x, _y, _zq = robot.root_link_quat_w.unbind(-1)
    _c, _s = torch.cos(robot.joint_pos[:, tr_torso] / 2), torch.sin(robot.joint_pos[:, tr_torso] / 2)
    tr["quat"][step_i] = torch.stack(
      [_w * _c - _zq * _s, _x * _c + _y * _s, _y * _c - _x * _s, _zq * _c + _w * _s], -1)
    tr["gyro"][step_i] = self.tr_gyro.data
    # Pelvis velocity expressed in the TORSO frame, v_T = Rz(-psi) v_P: what both
    # controllers log as `ach_v` on the bridge (they rotate by the IMU attitude).
    _vx, _vy = robot.root_link_lin_vel_b[:, 0], robot.root_link_lin_vel_b[:, 1]
    _cp, _sp = torch.cos(robot.joint_pos[:, tr_torso]), torch.sin(robot.joint_pos[:, tr_torso])
    tr["ach_v"][step_i] = torch.stack([_cp * _vx + _sp * _vy, _cp * _vy - _sp * _vx], -1)
    tr["phase"][step_i] = self.tr_phase.func(uenv, **self.tr_phase.params)
    if "vstar" in tr:   # the closure wrote hrl_goal = V* - s for THIS step
      tr["vstar"][step_i] = uenv.hrl_goal + self.runner.goal_space.extract(uenv)

  def after_step(self, step_i, actions, dones):
    tr, uenv = self.tr, self.uenv
    _t = uenv.action_manager.get_term("joint_pos")
    tr["raw_q"][step_i] = (_t.raw_action * _t.scale + _t.offset)[:, self.tr_pa]
    if "ach" in tr:     # post-step goal state, what training scores V* against
      tr["ach"][step_i] = self.runner.goal_space.extract(uenv)
    self.tr_fell[(dones.bool()) & (self.tr_fell < 0)] = step_i

  def seed_end(self, seed_idx, seed):
    import numpy as np
    cfg, tr, tr_fell, _lim = self.ctx.cfg, self.tr, self.tr_fell, self._lim
    np.savez_compressed(
      f"{cfg.eval_trace}_s{seed}.npz", step_dt=self.uenv.step_dt, seed=seed,
      label=self.ctx.label,
      cmd_seq=cfg.eval_cmd_seq, fell_step=tr_fell.cpu().numpy(),
      joint_names=np.array([n for _, _, n in _lim]),
      joint_min=np.array([a for a, _, _ in _lim]), joint_max=np.array([b for _, b, _ in _lim]),
      rew_names=np.array(self.rew_names if "rew" in tr else []),
      **{k: v.cpu().numpy().astype(np.float32) for k, v in tr.items()})
    print(f"[TRACE] {cfg.eval_trace}_s{seed}.npz  {self.steps} steps x {self.n_envs} envs, "
          f"{int((tr_fell >= 0).sum())} fell")


class Reward(Observer):
  """--reward-terms: every env reward term per policy step, split by regime, summed over seeds."""

  def __init__(self, ctx, trace: Trace | None):
    self.uenv, self.runner, self.trace = ctx.env.unwrapped, ctx.runner, trace
    self.rw_sum, self.rw_n = {}, {"walking": 0, "standing": 0}
    self.cmd = None   # the command this step is scored on; the reward dump records it

  def before_step(self, step_i, actions):
    self.cmd = _c = self.uenv.command_manager.get_command("twist")  # the command the reward is scored on
    self._walk = (_c[:, :2].norm(dim=1) + _c[:, 2].abs()) > 0.1

  def after_step(self, step_i, actions, dones):
    uenv, runner, rw_sum, rw_n, _walk = self.uenv, self.runner, self.rw_sum, self.rw_n, self._walk
    _rm = uenv.reward_manager
    # A1a: the env's foot_gait is A0's term on A0's fixed 0.6 s clock, which the LL never
    # sees; the LL trains its own feet_gait on the HL-commanded phase (hrl_runner `cadence`
    # term). `foot_gait_hl` = that term at the env weight, so the two can be swapped.
    _fg_hl = None
    if getattr(runner, "hl_cadence", False):
      from src.tasks.velocity.mdp import rewards as _mdp_rw
      _fg_hl = _rm.get_term_cfg("foot_gait").weight * _mdp_rw.feet_gait(
        uenv, use_commanded_phase=True, **runner._gait_params)
    for _reg, _m in (("walking", _walk), ("standing", ~_walk)):
      rw_n[_reg] += int(_m.sum())
      for _i, _n in enumerate(_rm.active_terms):
        rw_sum.setdefault(_n, {"walking": 0.0, "standing": 0.0})[_reg] += float(
          _rm._step_reward[_m, _i].sum())
      if _fg_hl is not None:
        rw_sum.setdefault("foot_gait_hl", {"walking": 0.0, "standing": 0.0})[_reg] += float(
          _fg_hl[_m].sum())
    if self.trace is not None:
      tr = self.trace.tr
      if "rew" not in tr:
        self.trace.rew_names = list(_rm.active_terms) + (["foot_gait_hl"] if _fg_hl is not None else [])
        tr["rew"] = torch.zeros(self.trace.steps, self.trace.n_envs, len(self.trace.rew_names),
                                device=tr["cmd"].device)
      tr["rew"][step_i] = _rm._step_reward if _fg_hl is None else torch.cat(
        [_rm._step_reward, _fg_hl[:, None]], dim=1)

  def finish(self, out):
    rw_sum, rw_n = self.rw_sum, self.rw_n
    out["reward_terms"] = {
      r: {n: round(v[r] / max(rw_n[r], 1), 6) for n, v in rw_sum.items()} for r in rw_n}
    out["reward_terms_steps"] = rw_n
    print(f"[REWARD] {json.dumps({'reward_terms': out['reward_terms'], 'env_steps': rw_n})}")


class RewardDump(Observer):
  """--reward-dump: the first seed's first 8 envs, raw, for an offline reward replay. Reads the
  command from the Reward add-on, so it needs --reward-terms."""

  def __init__(self, ctx, reward: Reward | None):
    self.ctx, self.uenv, self.reward = ctx, ctx.env.unwrapped, reward
    self.n_envs = self.uenv.num_envs
    self.dump = {k: [] for k in ("qpos", "qvel", "ctrl", "cmd", "phase", "period", "rew", "ct", "force")}
    self.seed_idx = 0

  def seed_begin(self, seed_idx, seed):
    self.seed_idx = seed_idx

  def before_step(self, step_i, actions):
    if self.seed_idx == 0:   # the phase the reward reads is set in policy()
      uenv, dump = self.uenv, self.dump
      dump["phase"].append(getattr(uenv, "hrl_phase", torch.full((self.n_envs,), float("nan")))[:8].tolist())
      dump["period"].append(float("nan") if not hasattr(uenv, "hrl_period")
                            else uenv.hrl_period[:8].tolist())
      dump["cmd"].append(self.reward.cmd[:8, :3].tolist())

  def after_step(self, step_i, actions, dones):
    if self.seed_idx == 0:
      uenv = self.uenv
      _d = uenv.sim.data
      for k, v in (("qpos", _d.qpos), ("qvel", _d.qvel), ("ctrl", _d.ctrl),
                   ("rew", uenv.reward_manager._step_reward),
                   ("ct", uenv.scene["feet_ground_contact"].data.current_contact_time),
                   ("force", uenv.scene["feet_ground_contact"].data.force)):
        self.dump[k].append(v[:8].tolist())

  def finish(self, out):
    import mujoco
    import numpy as np
    uenv, reward_dump = self.uenv, self.ctx.cfg.reward_dump
    _rm = uenv.reward_manager
    np.savez_compressed(reward_dump + ".npz", terms=np.array(_rm.active_terms),
                        weights=np.array([_rm.get_term_cfg(n).weight for n in _rm.active_terms]),
                        step_dt=uenv.step_dt,
                        gait_params=json.dumps(_rm.get_term_cfg("foot_gait").params, default=str),
                        **{k: np.array(v, dtype=float) for k, v in self.dump.items()})
    mujoco.mj_saveModel(uenv.sim.mj_model, reward_dump + ".mjb", None)
    print(f"[REWARD] dump -> {reward_dump}.npz/.mjb")


class JointLimits(Observer):
  """--check-joint-limits: the policy's COMMANDED targets against the deploy header's bounds,
  scored per seed ("last seed" is what the report prints), in safety_analyzer's shape."""

  def __init__(self, ctx):
    import bridge_session
    self.ctx, self.env, self.uenv = ctx, ctx.env, ctx.env.unwrapped
    self.n_envs = self.uenv.num_envs
    self._by_name = {n: (a, b) for a, b, n in bridge_session.deploy_limit_rows()}
    self.lim_lo = self.lim_hi = self.lim_names = None
    self.jl_over = self.jl_max = None
    self.jl_range = {}

  def reset(self):
    # --check-joint-limits: hardware command bounds, read from the deploy header rather
    # than duplicated here (it is the audited source -- 5-source consensus, WL-B0 item 2).
    env, uenv, by_name = self.env, self.uenv, self._by_name
    at = uenv.action_manager.get_term("joint_pos")
    sim_names = uenv.scene["robot"].joint_names
    tgt = at._target_ids if hasattr(at, "_target_ids") else list(range(len(sim_names)))
    tgt = list(range(len(sim_names))) if tgt is None else list(tgt)
    # The sim model calls slot 12 `torso`, the deploy header calls it `waist_yaw`
    # (same joint; play.py's own group patterns already match both spellings).
    alias = {"torso": "waist_yaw"}
    lim_names = self.lim_names = [alias.get(n, n) for n in
                                  (sim_names[i].replace("_joint", "").lower() for i in tgt)]
    miss = [n for n in lim_names if n not in by_name]
    if miss:
      raise SystemExit(f"--check-joint-limits: no deploy limit for {miss}")
    self.lim_lo = torch.tensor([by_name[n][0] for n in lim_names], device=env.device)
    self.lim_hi = torch.tensor([by_name[n][1] for n in lim_names], device=env.device)
    self.jl_over = torch.zeros(len(lim_names), device=env.device)
    self.jl_max = torch.zeros(len(lim_names), device=env.device)
    self.jl_sum = torch.zeros(len(lim_names), device=env.device)  # mean excess/step (ADR-0008 sizing)
    self.jl_rew = {}  # term name -> summed |contribution|, for the reward-share table
    self.jl_cmds = []  # per-step POST-clip targets, for the commanded-range percentiles

  def after_step(self, step_i, actions, dones):
    uenv, lim_lo, lim_hi = self.uenv, self.lim_lo, self.lim_hi
    # RECOMPUTE the pre-clip target: raw*scale+offset, matching the C++ `q_cmd`
    # before its clamp. Reading `_processed_actions` back would report 0.0 forever
    # once ADR-0008's clip is configured, because mjlab's process_actions
    # overwrites it with the CLAMPED value -- the instrument would die at exactly
    # the moment it is needed, which is the same measure-after-the-clamp defect
    # ADR-0008 exists to fix.
    _t = uenv.action_manager.get_term("joint_pos")
    q_cmd = _t.raw_action * _t.scale + _t.offset
    over = torch.maximum(lim_lo - q_cmd, q_cmd - lim_hi).clamp(min=0.0)  # [B, D]
    self.jl_over += (over > 0).float().mean(dim=0)
    self.jl_sum += over.mean(dim=0)
    self.jl_max = torch.maximum(self.jl_max, over.amax(dim=0))
    # Post-clip target = what the deploy path actually sends, and what the flight
    # recorder holds: State_RLBase.cpp:221 logs `action` and the safety clamp at
    # :164 writes a local, so `raw_q` is post-yaml-clip. Kept per step because the
    # RANGE, not the excess, is the discriminating quantity -- `over` collapses to
    # ~0 on any clip-trained policy, which hides whether the clip merely removed an
    # unusable overshoot or the penalty retreated the policy off the bound.
    self.jl_cmds.append(torch.clamp(q_cmd, lim_lo, lim_hi).detach())
    # Reward shares (ADR-0008 sizing): the excess weight is set so `action_clip`
    # contributes what `joint_pos_limits` contributes at the baseline violation
    # level, so both must be measured on the SAME rollout. NOTE a term whose
    # weight is 0.0 is skipped by RewardManager.compute and reads 0 here -- that
    # is why the sizing uses the RAW mean excess above, not this table, for the
    # numerator; this table supplies the target contribution only.
    _rm = uenv.reward_manager
    _sr = _rm._step_reward
    for _i, _n in enumerate(_rm.active_terms):
      self.jl_rew[_n] = self.jl_rew.get(_n, 0.0) + float(_sr[:, _i].abs().mean())

  def seed_end(self, seed_idx, seed):
    # Commanded RANGE per joint (2026-08-28, ADR-0008 follow-up). Percentiles of the
    # POST-clip target so they compare directly against the flight recorder. p1/p99
    # rather than min/max: one transient excursion is not authority, sustained range is.
    # `headroom` is the distance from the used range to the nearer bound -- the quantity
    # that separates a clip (headroom ~0, policy still reaches the stop) from a penalty
    # that pushed the policy inward (headroom ~ one action-sigma).
    lim_lo, lim_hi, lim_names = self.lim_lo, self.lim_hi, self.lim_names
    self.jl_range = {}
    if self.jl_cmds:
      allc = torch.cat(self.jl_cmds, dim=0).float()  # [steps*B, D]
      p01, p50, p99 = torch.quantile(
        allc, torch.tensor([0.01, 0.5, 0.99], device=allc.device), dim=0)
      pin = (((allc - lim_lo).abs() < 1e-9)
             | ((allc - lim_hi).abs() < 1e-9)).float().mean(dim=0)
      self.jl_range = {n: {"p1": round(p01[i].item(), 4), "p50": round(p50[i].item(), 4),
                           "p99": round(p99[i].item(), 4),
                           "span": round((p99[i] - p01[i]).item(), 4),
                           "pinned": round(pin[i].item(), 5),
                           "headroom": round(min((lim_hi[i] - p99[i]).item(),
                                                 (p01[i] - lim_lo[i]).item()), 4)}
                       for i, n in enumerate(lim_names)}

  def span(self, n):
    return self.jl_range.get(n, {}).get("span", float("nan"))

  def report(self):
    if self.jl_over is None:
      return
    steps = self.ctx.cfg.eval_steps
    lim_names, jl_rew, jl_range = self.lim_names, self.jl_rew, self.jl_range
    lim_lo, lim_hi = self.lim_lo, self.lim_hi
    rate = (self.jl_over / steps).tolist()
    mean_ex = (self.jl_sum / steps).tolist()
    worst = sorted(zip(lim_names, rate, self.jl_max.tolist(), mean_ex), key=lambda r: -r[1])
    tot = sum(rate) / max(len(rate), 1)
    tot_ex = sum(mean_ex)
    print()
    print("=" * 78)
    print(f"  COMMANDED JOINT-LIMIT CHECK (last seed) | {steps} steps x {self.n_envs} envs")
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
      print(f"  reward-term shares (|contribution| per step, {steps} steps)")
      tot_r = sum(jl_rew.values()) or 1.0
      for n, v in sorted(jl_rew.items(), key=lambda kv: -kv[1])[:10]:
        mark = "  <-- sizing target" if n == "joint_pos_limits" else (
               "  <-- 0.0 weight: NOT evaluated" if n == "action_clip" else "")
        print(f"    {n:<24} {v / steps:>10.6f}  ({100 * v / tot_r:>5.2f}%){mark}")
      jpl = jl_rew.get("joint_pos_limits", 0.0) / steps
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
               "reward_share": {n: round(v / steps, 6) for n, v in jl_rew.items()}}
    print(f"[LIMITS] {json.dumps(jl_json)}")


def build_addons(ctx) -> dict:
  """The add-ons the flags ask for, in hook order (see the module docstring)."""
  cfg, out = ctx.cfg, {}
  if cfg.eval_trace:
    out["trace"] = Trace(ctx)
  if ctx.seq_cmds is not None:
    out["schedule"] = Schedule(ctx)
  if cfg.reward_terms:
    out["reward"] = Reward(ctx, out.get("trace"))
  if cfg.reward_dump:
    out["dump"] = RewardDump(ctx, out.get("reward"))
  if cfg.check_joint_limits:
    out["limits"] = JointLimits(ctx)
  return out
