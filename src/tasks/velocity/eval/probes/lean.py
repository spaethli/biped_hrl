"""--probe-lean: backward-lean sensitivity (ADR-0006 / WL-B0e). Sweeps a deterministic encoder
bias (or, with --probe-gravity-noise, the projected-gravity corruption) and reports the steady-state
base pitch per condition (`[LEANPROBE]`). Keeps its own loop: one rollout per swept condition, with
the bias written after reset. Its observation wiring (joint_pos biased=True) happens in play.py's
setup, before the env is built."""

import json
import math

import torch


def run(ctx) -> dict:
  cfg, env, policy, resume_path = ctx.cfg, ctx.env, ctx.policy, ctx.resume_path
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

  label = str(resume_path) if resume_path else 'unknown'
  rows = []
  if cfg.probe_gravity_noise:
    # Gravity-corruption sweep (ADR-0006 discriminator). The obs manager reads
    # `term_cfg.noise` live every step, so mutating it here takes effect immediately.
    from mjlab.utils.noise.noise_cfg import UniformNoiseCfg
    om = uenv.observation_manager
    # A0 keeps one "actor" group; A1 splits it into policy/command/goal and DELETES
    # "actor" (h1_2_a1/env_cfgs.py), so find the actor-side group carrying the term.
    ggroup = next(
      g for g, terms in om.active_terms.items()
      if g != "critic" and "projected_gravity" in terms
    )
    gterm = next(
      c for n, c in zip(om.active_terms[ggroup],
                        om._group_obs_term_cfgs[ggroup], strict=False)  # noqa: SLF001
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
    out = {'label': label, 'mode': 'gravity', 'rows': rows}
    print(f"[LEANPROBE] {json.dumps(out)}")
    return out
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
  out = {'label': label, 'rows': rows}
  print(f"[LEANPROBE] {json.dumps(out)}")
  return out
