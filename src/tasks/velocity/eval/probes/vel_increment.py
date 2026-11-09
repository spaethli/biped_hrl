"""--check-vel-increment: IMU velocity-increment reconstruction per HL window, six rungs from raw
to the deployable waist-corrected @200 Hz one, scored against the true increment (`[VELINC]`).
`delta`-mode goals cancel absolute velocity, so deploy needs only this within-window increment."""

import json

import torch

from ..rollout import Observer, rollout


def _rz(psi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate v about z by psi (per-env). psi<0 gives the inverse."""
  c_, s_ = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
  x, y, z = v[:, 0:1], v[:, 1:2], v[:, 2:3]
  return torch.cat([c_ * x - s_ * y, s_ * x + c_ * y, z], dim=-1)


class VelIncrement(Observer):

  def __init__(self, ctx):
    env = self.env = ctx.env
    uenv = self.uenv = env.unwrapped
    robot = self.robot = uenv.scene["robot"].data
    # Accelerometer + velocimeter share the `imu` site (h1_2.xml:236-238) and the site
    # carries no rotation, so both read in the pelvis ORIENTATION frame -- but at a point
    # 0.28 m above the pelvis ORIGIN whose velocity the goal space uses (goal_space.py:50).
    # That lever arm (w x r) is why the site column is reported next to the pelvis one.
    acc = self.acc = uenv.scene["robot/imu_lin_acc"]
    self.gyro = uenv.scene["robot/imu_ang_vel"]
    self.vel_site = uenv.scene["robot/imu_lin_vel"]
    # MuJoCo's accelerometer includes gravity like a real IMU (verified: raw z ~ +9.95
    # against projected_gravity z ~ -0.996), so removal is a + proj_grav * |g|.
    g_mag = self.g_mag = float(torch.tensor(uenv.cfg.sim.mujoco.gravity).norm())
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
    self.r_T = torch.tensor(uenv.sim.mj_model.site_pos[_sid], device=env.device,
                            dtype=torch.float32).expand(uenv.num_envs, 3)
    _tj, _ = uenv.scene["robot"].find_joints([".*torso_joint.*"])
    assert len(_tj) == 1, f"expected exactly one torso_joint, got {_tj}"
    torso_idx = self.torso_idx = _tj[0]
    self.c = getattr(ctx.runner, "c", 8)
    self.dt = uenv.step_dt
    self.n_envs = uenv.num_envs

    # Physics-rate integration probe. The decimation loop calls metrics_manager.
    # compute_substep() once per substep, right after scene.update refreshes sensordata
    # (manager_based_rl_env.py:421-427) -- the only hook that sees the 200 Hz signal a real
    # IMU publishes, vs the 50 Hz control rate the other rungs are stuck with. Tests whether
    # the residual is aliased foot-impact transients rather than true estimator error.
    self.dv_fast = dv_fast = torch.zeros(uenv.num_envs, 3, device=env.device)
    _orig_substep = uenv.metrics_manager.compute_substep

    def _substep_hook():
      psi_s = robot.joint_pos[:, torso_idx]
      dv_fast.add_((acc.data + _rz(-psi_s, robot.projected_gravity_b) * g_mag)
                   * uenv.physics_dt)
      _orig_substep()

    uenv.metrics_manager.compute_substep = _substep_hook

    # Per-(window, env) rows, accumulated across seeds.
    self.err_a, self.keep_a, self.yaw_a = [], [], []
    self.lever_a = []  # |site increment - pelvis increment|: the lever-arm share

  def before_step(self, step_i, actions):
    if step_i % self.c:
      return
    # Window start. The policy call that precedes this hook only writes the hierarchy's goal
    # and phase buffers, never the robot state read here.
    robot, n_envs, device = self.robot, self.n_envs, self.env.device
    self.v0_root = robot.root_link_lin_vel_b.clone()
    self.v0_site = self.vel_site.data.clone()
    self.psi0 = robot.joint_pos[:, self.torso_idx].clone()
    self.w0_T = self.gyro.data.clone()
    self.dv_fast.zero_()  # in-place so the substep hook keeps writing the same buffer
    self.dv_raw = torch.zeros_like(self.v0_root)
    self.dv_grav = torch.zeros_like(self.v0_root)
    self.dv_cor = torch.zeros_like(self.v0_root)
    self.done_in_window = torch.zeros(n_envs, dtype=torch.bool, device=device)
    self.yaw_abs = torch.zeros(n_envs, device=device)

  def after_step(self, step_i, actions, dones):
    robot, c, dt, torso_idx = self.robot, self.c, self.dt, self.torso_idx
    self.done_in_window |= dones.bool()
    a = self.acc.data
    psi = robot.joint_pos[:, torso_idx]
    # Gravity must be removed in the TORSO frame the accelerometer reads in;
    # projected_gravity_b is the PELVIS frame, so undo the waist yaw first.
    a_g = a + _rz(-psi, robot.projected_gravity_b) * self.g_mag
    w = robot.root_link_ang_vel_b
    # d(v_b)/dt = a_b - w x v_b. The Coriolis rung uses the TRUE v_b, so it is the
    # privileged floor (hardware has no absolute v), not a deployable variant.
    self.dv_raw = self.dv_raw + a * dt
    self.dv_grav = self.dv_grav + a_g * dt
    self.dv_cor = self.dv_cor + (a_g - torch.cross(w, robot.root_link_lin_vel_b, dim=-1)) * dt
    self.yaw_abs = self.yaw_abs + w[:, 2].abs() / c
    if step_i % c != c - 1:
      return
    # Window end.
    true_root = robot.root_link_lin_vel_b - self.v0_root
    true_site = self.vel_site.data - self.v0_site
    # Deployable rung: pelvis increment from the integrated torso-site increment,
    # the gyro and the waist encoder (all present in LowState). Nothing privileged.
    psi_e, w_e = robot.joint_pos[:, torso_idx], self.gyro.data
    lev = torch.cross(w_e, self.r_T, dim=-1)
    lev0 = _rz(self.psi0, torch.cross(self.w0_T, self.r_T, dim=-1))
    dv_waist = _rz(psi_e, self.dv_grav - lev) + lev0
    dv_waist_fast = _rz(psi_e, self.dv_fast - lev) + lev0
    self.err_a.append(torch.stack([
      self.dv_raw - true_root, self.dv_grav - true_root, self.dv_cor - true_root,
      self.dv_grav - true_site, dv_waist - true_root, dv_waist_fast - true_root,
    ], dim=1).cpu())                      # [B, 6 rungs, 3 axes]
    self.lever_a.append((true_site - true_root).cpu())
    self.keep_a.append((~self.done_in_window).cpu())  # drop reset-contaminated windows
    self.yaw_a.append(self.yaw_abs.cpu())


def run(ctx) -> dict:
  cfg = ctx.cfg
  probe = VelIncrement(ctx)
  c, dt, n_envs = probe.c, probe.dt, probe.n_envs
  n_windows = cfg.check_vel_increment // c
  rollout(ctx, n_windows * c, [probe])

  err = torch.cat(probe.err_a)      # [W, 4, 3]
  lever = torch.cat(probe.lever_a)  # [W, 3]
  keep = torch.cat(probe.keep_a)
  yaw = torch.cat(probe.yaw_a)
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
  return out
