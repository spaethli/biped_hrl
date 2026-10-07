#!/usr/bin/env python3
"""Score the env-reward terms a flight recorder can reproduce (bridge sim + hardware), regime-split.

The training reward has 17 terms; a flight recorder carries enough state for the seven below.
`play.py --reward-terms` reads the same seven from the live RewardManager (the truth), so the
three environments compare on one set. Values are `weight * raw`, per policy step, the same unit
as RewardManager._step_reward.

  track_linear_velocity  exp(-(|cmd_xy - v_xy|^2 + 2 vz^2)/std^2). Truth: mocap (hardware) or the
                         plant's own velocity `ach_v*` (bridge). Hardware has no vz -> that term is
                         omitted there (`--vz-effect` on a bridge log prices the omission).
  track_angular_velocity exp(-((cmd_wz - gyro_z)^2 + 0.05 |gyro_xy|^2)/std^2)
  body_orientation_l2    |proj. gravity_xy|^2 of the torso (IMU quaternion; the IMU sits on torso_link)
  body_ang_vel           |gyro_xy|^2 -- sim reads the torso's WORLD-frame rate; the gyro is body-frame
                         (identical while upright, differs by cos(tilt))
  pose                   exp(-mean_j (q_j - q_def_j)^2 / std_j^2), std by the commanded speed regime
  action_rate_l2         sum_j (a_t - a_t-1)^2, a = (raw_q - offset)/scale  (raw_q is post yaml clip)
  stand_still            sum_j (q_j - q_def_j)^2 while the command is below threshold

The other 10 (foot_gait, foot_clearance, foot_slip, soft_landing, self_collisions, angular_momentum,
joint_acc_l2, joint_pos_limits, is_terminated) need contact, foot and whole-body state. On HARDWARE they
are absent (never zero). On the BRIDGE they are recomputed by PLANT REPLAY: the log carries the plant's
full generalized state at every policy step, so it is put back into the bridge's own MJCF and one
`mj_forward` per step (torque = the logged PD target, kp(q_des - q) - kd dq) gives contacts, forces,
foot sites, angular momentum and qacc as the plant computed them. Height: `gt_h` (A1a telemetry) or,
without it (A0), the lowest foot vertex at the plant's median stance penetration (-2.9 mm); on A1a that
approximation moves the 10-term sum by <= 0.0016 per step. `--validate`: a round-trip selftest of the
reconstruction and a replay of bench dumps (play.py --reward-dump) against the live RewardManager.

Weights, std tables and thresholds are read from the task config, not restated here.

Usage:
  python scripts/score_env_reward.py --out data/2026-09-25-env-reward/env_reward_hw_bridge.json
  python scripts/score_env_reward.py --out /dev/null --validate
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_flight_recorder import (CMD_THRESHOLD, attach_mocap, projected_gravity_xy,  # noqa: E402
                                   read_flight)

REPO = Path(__file__).resolve().parents[1]
DEPLOY_YAML = REPO / "deploy/robots/h1_2/config/policy/velocity/v0/params/deploy.yaml"
COMPARABLE = ("track_linear_velocity", "track_angular_velocity", "body_orientation_l2",
              "body_ang_vel", "pose", "action_rate_l2", "stand_still")


def sim_joint_name(slot_name: str) -> str:
  return "torso_joint" if slot_name == "waist_yaw" else slot_name + "_joint"


def task_cfg():
  from src.tasks.velocity.config.h1_2.env_cfgs import unitree_h1_2_flat_env_cfg
  return unitree_h1_2_flat_env_cfg().rewards


def pose_stds(rewards, names):
  """[3, n_joints] std for (standing, walking, running), resolved as mjlab does (regex over names)."""
  out = []
  for key in ("std_standing", "std_walking", "std_running"):
    row = []
    for n in names:
      hits = [v for pat, v in rewards["pose"].params[key].items() if re.fullmatch(pat, n)]
      assert len(hits) == 1, f"{key}: {n} matched {len(hits)} patterns"
      row.append(hits[0])
    out.append(row)
  return np.array(out, float)


# ------------------------------------------------------------ plant replay (bridge only) ----
# The other 10 terms need contact, foot and whole-body state. The bridge log carries the plant's
# full generalized state at every policy step (joint q/dq from exact sensors, torso orientation
# from the IMU framequat, pelvis COM velocity as `ach_v`, IMU-site height as `gt_h`), so the
# state is put back into the plant's own MJCF and `mj_forward` recomputes what the plant saw.
# Torque is recomputed from the logged PD target: the bridge applies kp(q_des - q) - kd dq.
BRIDGE_SCENE = Path("/opt/unitree_mujoco/unitree_robots/h1_2/scene_stress.xml")
REPLAY_TERMS = ("angular_momentum", "joint_acc_l2", "joint_pos_limits", "foot_gait",
                "foot_clearance", "foot_slip", "soft_landing", "self_collisions", "is_terminated")
FOOT_SITE_POS = (0.04, 0.0, -0.04)   # h1_2.xml `left_foot` / `right_foot`, in the ankle_roll_link frame
SELF_FORCE_N = 10.0                  # self_collision_cost force_threshold
DECIMATION = 4                       # physics substeps per policy step; self_collisions counts substeps


class Plant:
  """One MuJoCo model plus the lookups the reward terms need; works on the training model (a
  play.py dump, names prefixed `robot/`) and on the bridge MJCF (no foot sites: added in memory)."""

  def __init__(self, m):
    import mujoco
    self.mj, self.m, self.d = mujoco, m, mujoco.MjData(m)
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    bid = {name(mujoco.mjtObj.mjOBJ_BODY, i).split("/")[-1]: i for i in range(m.nbody)}
    self.pelvis = bid["pelvis"]
    self.torso = bid["torso_link"]
    self.angmom_body = self.torso   # `robot/root_angmom` is a subtreeangmom of torso_link, not pelvis
    self.feet = [bid["left_ankle_roll_link"], bid["right_ankle_roll_link"]]
    self.sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in
                  (s for s in (name(mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite))
                   if s.split("/")[-1] in ("left_foot", "right_foot"))]
    root = m.body_rootid
    self.robot_geom = np.array([root[m.geom_bodyid[g]] == self.pelvis for g in range(m.ngeom)])
    # foot index of each geom: the subtree of that ankle_roll_link (feet_ground_contact is a subtree match)
    self.geom_foot = np.full(m.ngeom, -1)
    for g in range(m.ngeom):
      b = m.geom_bodyid[g]
      while b > 0:
        if b in self.feet:
          self.geom_foot[g] = self.feet.index(b)
          break
        b = m.body_parentid[b]
    jn = [name(mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1] for j in range(m.njnt)]
    self.waist_jnt = jn.index("torso_joint")
    self.waist = self.waist_jnt - 1                        # index among the 27 actuated joints
    self.imu = next(i for i in range(m.nsite) if name(mujoco.mjtObj.mjOBJ_SITE, i).split("/")[-1] == "imu")
    self.foot_geoms = [g for g in range(m.ngeom) if self.geom_foot[g] >= 0
                       and (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] == 7]
    self.jnt_lo, self.jnt_hi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()

  @classmethod
  def bridge(cls):
    import mujoco
    spec = mujoco.MjSpec.from_file(str(BRIDGE_SCENE))
    for side in ("left", "right"):
      spec.body(f"{side}_ankle_roll_link").add_site(name=f"{side}_foot", pos=FOOT_SITE_POS)
    return cls(spec.compile())

  def forward(self, qpos, qvel, ctrl):
    """-> per-step plant quantities, after one mj_forward at (qpos, qvel, ctrl)."""
    mj, m, d = self.mj, self.m, self.d
    d.qpos[:], d.qvel[:], d.ctrl[:] = qpos, qvel, ctrl
    mj.mj_forward(m, d)
    found, force, self_max = np.zeros(2, bool), np.zeros((2, 3)), 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
      c = d.contact[i]
      g1, g2 = c.geom1, c.geom2
      mj.mj_contactForce(m, d, i, f6)
      fw = c.frame.reshape(3, 3).T @ f6[:3]          # force on geom2 from geom1, world frame
      r1, r2 = self.robot_geom[g1], self.robot_geom[g2]
      if r1 and r2:
        self_max = max(self_max, float(np.linalg.norm(fw)))
      elif r1 != r2:
        g, sign = (g1, -1.0) if r1 else (g2, 1.0)      # force ON the robot geom
        k = self.geom_foot[g]
        if k >= 0:
          found[k] = True
          force[k] += sign * fw
    site_p = np.array([d.site_xpos[s] for s in self.sites])
    site_v = np.zeros((2, 3))
    v6 = np.zeros(6)
    for k, s in enumerate(self.sites):
      mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_SITE, s, v6, 0)
      site_v[k] = v6[3:]
    mj.mj_subtreeVel(m, d)
    return {"found": found, "force": force, "self_max": self_max, "site_p": site_p,
            "site_v": site_v, "angmom": d.subtree_angmom[self.angmom_body].copy(),
            "qacc_j": d.qacc[6:].copy(), "q_j": d.qpos[7:].copy()}


def replay_terms(plant, qpos, qvel, ctrl, cmd, gait_phase, weights, rewards, stance_thr=None):
  """weight x raw for REPLAY_TERMS, per step. `gait_phase` = the global phase the reward reads
  (fixed-clock or commanded), [T]; `stance_thr` the duty threshold (fixed if None)."""
  T = len(qpos)
  P = [plant.forward(qpos[t], qvel[t], ctrl[t]) for t in range(T)]
  found = np.array([p["found"] for p in P])
  force = np.linalg.norm([p["force"] for p in P], axis=-1)
  first = found & ~np.vstack([np.zeros((1, 2), bool), found[:-1]])
  tot = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])
  gp = rewards["foot_gait"].params
  gate = lambda k: (tot > rewards[k].params["command_threshold"]).astype(float)
  leg_phase = (gait_phase[:, None] + np.array(gp["offset"])[None]) % 1.0
  thr = gp["threshold"] if stance_thr is None else stance_thr[:, None]
  sp = np.array([p["site_p"] for p in P])
  sv = np.array([p["site_v"] for p in P])
  vxy = np.linalg.norm(sv[..., :2], axis=-1)
  q = np.array([p["q_j"] for p in P])
  mid, half = 0.5 * (plant.jnt_lo + plant.jnt_hi), 0.5 * (plant.jnt_hi - plant.jnt_lo) * SOFT_LIMIT
  raw = {
    "angular_momentum": (np.array([p["angmom"] for p in P]) ** 2).sum(1),
    "joint_acc_l2": (np.array([p["qacc_j"] for p in P]) ** 2).sum(1),
    "joint_pos_limits": ((mid - half - q).clip(min=0) + (q - mid - half).clip(min=0)).sum(1),
    "foot_gait": ((leg_phase < thr) == found).mean(1) * gate("foot_gait"),
    "foot_clearance": (np.abs(sp[..., 2] - rewards["foot_clearance"].params["target_height"])
                       * vxy).sum(1) * gate("foot_clearance"),
    "foot_slip": (vxy ** 2 * found).sum(1) * gate("foot_slip"),
    "soft_landing": (force * first).sum(1) * gate("soft_landing"),
    "self_collisions": DECIMATION * np.array([p["self_max"] > SELF_FORCE_N for p in P], float),
    "is_terminated": np.zeros(T),
  }
  return {k: weights[k] * raw[k] for k in REPLAY_TERMS}, found


SOFT_LIMIT = 0.9   # h1_2_constants soft_joint_pos_limit_factor


def _qmul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rot(q):
  R = np.zeros(9)
  import mujoco
  mujoco.mju_quat2Mat(R, np.asarray(q, float))
  return R.reshape(3, 3)


def reconstruct(plant, quat_t, gyro_t, v_b, imu_z, q_j, dq_j):
  """Bridge log quantities of one step -> (qpos, qvel) of the plant.

  quat_t/gyro_t: the torso IMU (site frame = torso frame); v_b: `ach_v`, the pelvis COM velocity
  (frame_vel is a body-INERTIAL framelinvel) rotated into the IMU frame by the controller;
  imu_z: `gt_h`, the IMU site's world height (frame_pos)."""
  mj, m, d = plant.mj, plant.m, plant.d
  d.qpos[:3], d.qpos[3:7], d.qpos[7:] = 0.0, (1, 0, 0, 0), q_j
  mj.mj_kinematics(m, d)
  q_rel = d.xquat[plant.torso].copy()                   # torso wrt pelvis, pelvis at identity
  q_p = _qmul(quat_t, q_rel * np.array([1, -1, -1, -1]))
  d.qpos[3:7] = q_p
  mj.mj_kinematics(m, d)
  qpos = d.qpos.copy()
  qpos[2] = imu_z - d.site_xpos[plant.imu][2]
  R_t, R_p = _rot(quat_t), _rot(q_p)
  w_p = R_t @ gyro_t - dq_j[plant.waist] * d.xaxis[plant.waist_jnt]   # pelvis angular velocity, world
  mj.mj_comPos(m, d)
  r = d.xipos[plant.pelvis] - d.xpos[plant.pelvis]
  v_origin = R_t @ v_b - np.cross(w_p, r)
  qvel = np.concatenate([v_origin, R_p.T @ w_p, dq_j])
  return qpos, qvel


def selftest(plant, n=20, seed=0):
  """Round trip: plant state -> the quantities the bridge publishes and the controller logs ->
  reconstruct -> the same state. Returns the worst |error| over qpos and qvel."""
  mj, m = plant.mj, plant.m
  d = mj.MjData(m)
  sid = lambda k: m.sensor_adr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, k)]
  rng, worst = np.random.default_rng(seed), 0.0
  for _ in range(n):
    q = rng.normal(0, 1, 4) * [1, .1, .1, 1]
    d.qpos[:] = np.concatenate([[.3, -.2, 1.0], q / np.linalg.norm(q), rng.normal(0, .2, 27)])
    d.qvel[:] = rng.normal(0, .5, m.nv)
    mj.mj_forward(m, d)
    qt = d.sensordata[sid("imu_quat"):sid("imu_quat") + 4].copy()
    gy = d.sensordata[sid("imu_gyro"):sid("imu_gyro") + 3].copy()
    vw = d.sensordata[sid("frame_vel"):sid("frame_vel") + 3].copy()
    z = d.sensordata[sid("frame_pos") + 2]
    v_b = _rot(qt).T @ vw                                   # State_RLBase: root_quat_w.conjugate() * v_world
    qp, qv = reconstruct(plant, qt, gy, v_b, z, d.qpos[7:].copy(), d.qvel[6:].copy())
    worst = max(worst, np.abs(qp[2:] - d.qpos[2:]).max(), np.abs(qv - d.qvel).max())
  return worst


ANCHOR_PEN_M = -0.0029   # median lowest-foot-vertex z of the bridge plant (A1a/A1a_DR s42 with gt_h: -2.99/-2.73 mm)


def min_foot_z(plant, qpos):
  """Lowest world z over the foot collision meshes' vertices at qpos (plane-mesh contact distance)."""
  mj, m, d = plant.mj, plant.m, plant.d
  d.qpos[:] = qpos
  mj.mj_kinematics(m, d)
  lo = np.inf
  for g in plant.foot_geoms:
    a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
    lo = min(lo, float((m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T[:, 2]).min() + d.geom_xpos[g][2]))
  return lo


def bridge_replay(plant, capture, weights, rewards, gait_period, fixed_clock_phase, height="gt_h"):
  """A bridge capture -> {regime: {term: mean}} for REPLAY_TERMS. `fixed_clock_phase`: True when the
  policy's gait obs IS the reward's clock (A0); for A1a the reward's clock is the fixed 0.6 s episode
  clock, which a deploy session has no origin for -> started at the policy entry."""
  import pandas as pd
  import yaml
  F = read_flight(capture, assume_hold=False)
  df, sidx, steps = F["df"], F["sidx"], F["steps"]
  g = lambda cols: df[cols].to_numpy(float)[sidx]
  quat, gyro = g(["quat_w", "quat_x", "quat_y", "quat_z"]), g(["est_gyro_x", "est_gyro_y", "est_gyro_z"])
  v_b, q, dq = g(["ach_vx", "ach_vy", "ach_vz"]), g([f"meas_q{j}" for j in range(27)]), g([f"meas_dq{j}" for j in range(27)])
  cmd = F["cmd"][sidx]
  hrl = Path(capture).with_name(Path(capture).stem + "_hrl.csv")
  n, z = len(sidx) - 1, np.zeros(len(sidx))
  if height == "gt_h":
    if not hrl.exists():
      raise SystemExit(f"{capture}: no _hrl.csv -> no gt_h; use height='anchor'")
    # HRL row i is flight policy row i+1 (paired by index, not by `t`: the two tick clocks are offset,
    # and one step of height error is mm-scale penetration, i.e. the contact state). Guarded on hip pitch.
    H = pd.read_csv(hrl, usecols=["gt_h", "hip_pitch_l"])
    n = min(len(H), n)
    err = np.abs(H["hip_pitch_l"].to_numpy(float)[:n] - q[1:n + 1, 1]).max()
    if err > 1e-3:
      raise SystemExit(f"{capture}: _hrl.csv does not pair row-for-row with the flight log (hip pitch {err:.4f} rad)")
    z[1:n + 1] = H["gt_h"].to_numpy(float)[:n]
  y = yaml.safe_load(DEPLOY_YAML.read_text())
  kp, kd = np.array(y["stiffness"], float), np.array(y["damping"], float)
  lo, hi = plant.m.actuator_ctrlrange[:, 0], plant.m.actuator_ctrlrange[:, 1]
  T = min(len(sidx) - 1, n)
  qpos, qvel, ctrl = np.zeros((T, plant.m.nq)), np.zeros((T, plant.m.nv)), np.zeros((T, plant.m.nu))
  for k in range(T):                        # step k is scored at its END state (= row k+1) under target a_k
    qpos[k], qvel[k] = reconstruct(plant, quat[k + 1], gyro[k + 1], v_b[k + 1], z[k + 1], q[k + 1], dq[k + 1])
    if height == "anchor":   # no height in the log (A0): lowest foot vertex at the plant's stance penetration
      qpos[k, 2] += ANCHOR_PEN_M - min_foot_z(plant, qpos[k])
    ctrl[k] = np.clip(kp * (steps[k] - q[k + 1]) - kd * dq[k + 1], lo, hi)
  dt = 0.02
  if fixed_clock_phase:     # continuous clock from the obs (it is zeroed while standing): fit phi0
    s, c = g(["phase_sin"])[:, 0], g(["phase_cos"])[:, 0]
    live = (s ** 2 + c ** 2) > 0.5
    k = np.arange(len(sidx))
    phi_obs = np.arctan2(s, c) / (2 * np.pi)
    phi0 = np.angle(np.mean(np.exp(2j * np.pi * (phi_obs - k * dt / gait_period))[live])) / (2 * np.pi)
    phase = (phi0 + (k[:T] + 1) * dt / gait_period) % 1.0
  else:
    phase = ((np.arange(T) + 1) * dt / gait_period) % 1.0
  terms, found = replay_terms(plant, qpos, qvel, ctrl, cmd[:T], phase, weights, rewards)
  if not fixed_clock_phase and hrl.exists():
    # A1a's LL trains feet_gait on the HL-commanded phase (hrl_runner `cadence` term), not on the
    # env's fixed 0.6 s clock above. The logged gait obs IS that phase as the policy saw it; step k's
    # reward reads the phase already advanced for step k = the obs at row k+1. The obs is zeroed below
    # |cmd| 0.1, so those steps integrate the logged HL period (hrl row i = flight row i+1).
    import pandas as pd
    Tp = pd.read_csv(hrl, usecols=["period"])["period"].to_numpy(float)
    s_, c_ = g(["phase_sin"])[:, 0], g(["phase_cos"])[:, 0]
    live = (s_ ** 2 + c_ ** 2) > 0.5
    phi = np.zeros(len(sidx))
    for k in range(1, len(sidx)):
      phi[k] = (np.arctan2(s_[k], c_[k]) / (2 * np.pi)) % 1.0 if live[k] else \
               (phi[k - 1] + dt / Tp[min(k - 1, len(Tp) - 1)]) % 1.0
    gp = rewards["foot_gait"].params
    leg = (phi[1:T + 1, None] + np.array(gp["offset"])[None]) % 1.0
    tot = np.linalg.norm(cmd[:T, :2], axis=1) + np.abs(cmd[:T, 2])
    terms["foot_gait_hl"] = weights["foot_gait"] * ((leg < gp["threshold"]) == found).mean(1) * \
                            (tot > gp["command_threshold"])
  walk = F["walk_rows"][sidx][:T]
  out = {r: {k: float(v[m].mean()) for k, v in terms.items()} for r, m in (("walking", walk), ("standing", ~walk))}
  out["_n"] = {"walking": int(walk.sum()), "standing": int((~walk).sum())}
  out["_duty"] = [float(x) for x in found[walk].mean(0)]
  # `foot_gait_hl` is an ALTERNATIVE to foot_gait (A1a's trained clock), never an extra term: a
  # 17-term sum takes one or the other.
  return out


def validate(dump_path, rewards):
  """Replay a play.py --reward-dump through `replay_terms` and compare to the live RewardManager."""
  import mujoco
  z = np.load(f"{dump_path}.npz")
  plant = Plant(mujoco.MjModel.from_binary_path(f"{dump_path}.mjb"))
  terms = list(z["terms"])
  w = dict(zip(terms, z["weights"]))
  T, E = z["qpos"].shape[:2]
  gp = json.loads(str(z["gait_params"]))
  out = {}
  for e in range(E):
    cmd = z["cmd"][:, e]
    phase = ((np.arange(T) + 1) * float(z["step_dt"]) / gp["period"]) % 1.0   # episode_length_buf after +1
    got, found = replay_terms(plant, z["qpos"][:, e], z["qvel"][:, e], z["ctrl"][:, e], cmd, phase, w, rewards)
    live = {k: z["rew"][:, e, terms.index(k)] for k in REPLAY_TERMS}
    walk = (np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])) > 0.1
    for k in REPLAY_TERMS:
      out.setdefault(k, {"replay": [], "live": []})
      out[k]["replay"].append(got[k][walk])
      out[k]["live"].append(live[k][walk])
    out.setdefault("_contact_agree", []).append((found == (z["ct"][:, e] > 0)).mean())
  print(f"[VALIDATE] {Path(dump_path).name}: foot contact agrees with the live sensor on "
        f"{100 * np.mean(out.pop('_contact_agree')):.1f}% of foot-steps; walking means, weight x raw:")
  res = {}
  for k, v in out.items():
    r, l = np.concatenate(v["replay"]).mean(), np.concatenate(v["live"]).mean()
    res[k] = (r, l)
    print(f"[VALIDATE]   {k:18s} replay {r:+.5f}  live {l:+.5f}  diff {r - l:+.5f}")
  print(f"[VALIDATE]   {'SUM':18s} replay {sum(a for a, _ in res.values()):+.5f}  "
        f"live {sum(b for _, b in res.values()):+.5f}")
  return res


def score_flight(csv_path, rewards, entry=None, mocap=True):
  """-> {regime: {term: mean}, "_n": {regime: policy steps}, "_ref": velocity reference}."""
  F = read_flight(csv_path, assume_hold=False, entry=entry)
  df, sidx, steps, cmd = F["df"], F["sidx"], F["steps"], F["cmd"]
  ref, has_vz = "ach_v", bool(np.any(df["ach_vz"].to_numpy(float) != 0.0)) if "ach_vz" in df else False
  if not np.any(df[["ach_vx", "ach_vy"]].to_numpy(float) != 0.0):
    mo = attach_mocap(F, csv_path) if mocap else None
    if mo is None or mo["gt_coverage"] < 0.5:
      raise SystemExit(f"{csv_path}: no velocity truth (no ach_v, no usable mocap)")
    ref = "gt_v"
  v = df[["ach_vx", "ach_vy"] if ref == "ach_v" else ["gt_vx", "gt_vy"]].to_numpy(float)[sidx]
  gyro = df[["est_gyro_x", "est_gyro_y", "est_gyro_z"]].to_numpy(float)[sidx]
  q = df[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(float)[sidx]
  meas = df[[f"meas_q{j}" for j in range(27)]].to_numpy(float)[sidx]
  c = cmd[sidx]
  walk = F["walk_rows"][sidx]
  speed = np.linalg.norm(c[:, :2], axis=1) + np.abs(c[:, 2])   # the pose term's regime variable

  import yaml
  d = yaml.safe_load(DEPLOY_YAML.read_text())["actions"]["JointPositionAction"]
  scale, offset = np.array(d["scale"], float), np.array(d["offset"], float)
  dflt = F["default_pose"]
  names = [sim_joint_name(F["joints"][j]["name"]) for j in range(27)]
  std = pose_stds(rewards, names)
  th = rewards["pose"].params
  sel = np.where(speed < th["walking_threshold"], 0, np.where(speed < th["running_threshold"], 1, 2))
  std_row = std[sel, :]

  w = {k: rewards[k].weight for k in COMPARABLE}
  lin_err = ((c[:, :2] - v) ** 2).sum(1)
  if has_vz:
    lin_err = lin_err + 2.0 * df["ach_vz"].to_numpy(float)[sidx] ** 2
  vals = {
    "track_linear_velocity": np.exp(-lin_err / rewards["track_linear_velocity"].params["std"] ** 2),
    "track_angular_velocity": np.exp(-((c[:, 2] - gyro[:, 2]) ** 2 + 0.05 * (gyro[:, :2] ** 2).sum(1))
                                     / rewards["track_angular_velocity"].params["std"] ** 2),
    "body_orientation_l2": projected_gravity_xy(*q.T) ** 2,
    "body_ang_vel": (gyro[:, :2] ** 2).sum(1),
    "pose": np.exp(-np.mean((meas - dflt) ** 2 / std_row ** 2, axis=1)),
    "stand_still": np.where(~walk, ((meas - dflt) ** 2).sum(1), 0.0),
  }
  a = (steps - offset) / scale
  ar = np.full(len(steps), np.nan)
  ar[1:] = ((a[1:] - a[:-1]) ** 2).sum(1)
  vals["action_rate_l2"] = ar
  out = {"_ref": ref, "_vz_included": has_vz, "_n": {}}
  for regime, m in (("walking", walk), ("standing", ~walk)):
    out["_n"][regime] = int(m.sum())
    out[regime] = {}
    for k in COMPARABLE:
      x = vals[k].copy()
      ok = m & np.isfinite(x)
      if k == "action_rate_l2":     # a transition counts only when both endpoints are in the regime
        ok = ok & np.concatenate([[False], m[:-1]])
      if k in ("track_linear_velocity",):
        ok = ok & np.isfinite(v).all(1)
      out[regime][k] = float(w[k] * x[ok].mean()) if ok.sum() > 10 else None
  if ref == "ach_v" and has_vz:   # what dropping the vz term (hardware has none) would have cost
    lin2 = np.exp(-((c[:, :2] - v) ** 2).sum(1) / rewards["track_linear_velocity"].params["std"] ** 2)
    out["_vz_effect"] = {r: float(w["track_linear_velocity"] * (lin2[m] - vals["track_linear_velocity"][m]).mean())
                         for r, m in (("walking", walk), ("standing", ~walk))}
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", required=True)
  ap.add_argument("--hw-glob", default="logs/robot_logs/2026_09_16-*/")
  ap.add_argument("--bridge-dir", default="logs/sim_logs/chain")
  ap.add_argument("--dump-dir", default="data/2026-09-25-env-reward",
                  help="play.py --reward-dump outputs dump_A0_s42 / dump_A1a_s42: the live weights for the "
                       "plant replay, and the ground truth --validate replays")
  ap.add_argument("--validate", action="store_true",
                  help="only: round-trip selftest + replay the bench dumps against the live reward")
  args = ap.parse_args()
  rewards = task_cfg()
  dumps = {"A0": REPO / args.dump_dir / "dump_A0_s42", "A1a": REPO / args.dump_dir / "dump_A1a_s42"}
  if args.validate:
    print(f"[VALIDATE] reconstruction round trip, worst |error| {selftest(Plant.bridge()):.2e}")
    for p in dumps.values():
      validate(p, rewards)
    return 0
  plant = Plant.bridge()
  weights = {}
  for fam, p in dumps.items():
    z = np.load(f"{p}.npz")
    weights[fam] = (dict(zip(z["terms"], z["weights"])), json.loads(str(z["gait_params"]))["period"])
  res = {"hardware": {}, "bridge": {}}
  for d in sorted(REPO.glob(args.hw_glob)):
    if not (d / "run.json").exists():
      continue
    run = json.loads((d / "run.json").read_text())
    if float(run.get("payload_kg") or 0.0) > 0 or not (d / "mocap_aligned.csv").exists():
      continue
    try:
      res["hardware"][d.name] = {"policy": run["policy_tag"], **score_flight(d / run["flight_recorder"], rewards)}
    except SystemExit as e:
      print(f"[REWARD] skip {d.name}: {e}")
  for tag_dir in sorted((REPO / args.bridge_dir).glob("*/")):
    # `<tag>/bridge/` (the schedule-replay layout, 2026-10-02) holds several Runs per policy and
    # every one is scored; the older flat layout holds one capture per policy, the newest file.
    files = sorted((tag_dir / "bridge").glob("*.walking.json")) or sorted(tag_dir.glob("*.walking.json"))[-1:]
    for f in files:
      js = json.loads(f.read_text())
      fl = REPO / "logs/deploy_safety" / js["flight_recorder"]
      key = tag_dir.name if f.parent == tag_dir else f"{tag_dir.name}/{js['label']}"
      res["bridge"][key] = {"policy": tag_dir.name, "capture": js["flight_recorder"],
                            **score_flight(fl, rewards, mocap=False)}
      fam = "A0" if tag_dir.name.startswith("A0") else "A1a"
      height = "gt_h" if fl.with_name(fl.stem + "_hrl.csv").exists() else "anchor"
      w, period = weights[fam]
      try:
        rep = bridge_replay(plant, fl, w, rewards, period, fixed_clock_phase=(fam == "A0"), height=height)
      except SystemExit as e:
        print(f"[REWARD] bridge replay {key}: skipped -- {e}")
        continue
      res["bridge"][key]["replay"] = {**rep, "height": height}
      print(f"[REWARD] bridge replay {key:20s} height={height:6s} walk duty "
            f"{rep['_duty'][0]:.2f}/{rep['_duty'][1]:.2f}  walking 10-term sum {sum(rep['walking'][t] for t in REPLAY_TERMS):+.4f}"
            + (f"  (foot_gait on the HL clock {rep['walking']['foot_gait_hl']:+.4f} vs env clock "
               f"{rep['walking']['foot_gait']:+.4f})" if "foot_gait_hl" in rep["walking"] else ""))
  Path(args.out).write_text(json.dumps(res, indent=1))
  for env, rows in res.items():
    for k, r in rows.items():
      print(f"[REWARD] {env:8s} {k:34s} walk n={r['_n']['walking']:>5}  "
            + "  ".join(f"{t[:8]}={r['walking'][t]:+.4f}" if r['walking'][t] is not None else f"{t[:8]}=  n/a"
                        for t in COMPARABLE))
  del plant
  return 0


if __name__ == "__main__":
  sys.exit(main())
