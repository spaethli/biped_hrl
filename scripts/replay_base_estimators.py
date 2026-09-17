#!/usr/bin/env python3
"""Offline replay of base-velocity estimators against a deploy flight-recorder session.

Implements the experiment of ``doc/hrl/h1_2_ekf_design.md`` §10: every arm is replayed on
byte-identical logged data, so the robot, the policy, the session and the per-restart
encoder offset are all removed as confounds.

  A  deployed leg odometry (finite difference)           the incumbent, as shipped
  B  leg odometry + complementary filter                 "any filtering at all"     [gates]
  C  EKF, position-only (Bloesch core)                   the proposal               [primary]
  D  C + flat-foot orientation                           does Rotella's extension pay
  E  Jacobian leg odometry (-J qdot - w x p)             where the differentiation happens
  F  C consuming the vendor attitude                     the double-counting question

FRAMES (doc/hrl/h1_2_kinematics.md). The IMU is on `torso_link`, not the pelvis; the two
frame origins coincide and differ by the waist yaw psi alone. Leg FK is pelvis-referenced,
so a foot in the IMU frame is ``Rz(-psi) p^P - r_imu``. The EKF's body frame is the TORSO,
which is what lets Rotella's equations transfer unchanged; the pelvis conversion happens
once, at the output.

Run ``--selftest`` (no session needed) to check the measurement Jacobians against finite
differences and the filter against synthetic ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The FK is REUSED, not transcribed: src/tasks/velocity/rl/hrl/leg_odom.py is already a
# verified transcription of the deployed hrl::leg_fk (C++ parity 6.3e-7 m/s, golden-vector
# test tests/test_leg_odom_parity.py) and is batched, so it runs on a whole session at once.
import torch  # noqa: E402

from src.tasks.velocity.rl.hrl.leg_odom import (  # noqa: E402
  CONTACT_A,
  FOOT_SITE_A,
  leg_fk,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_flight_recorder import resolve_run_entry  # noqa: E402

R_IMU = np.array([-0.04452, -0.01891, 0.27756])  # imu site in the torso frame, h1_2.xml:145
GRAVITY = np.array([0.0, 0.0, -9.81])            # base_state.h kGravity

# ⚠️ MEASURED 2026-08-12, 655 s passive hang: the accelerometer reads |a| = 9.9030 m/s^2 at
# rest, 0.093 m/s^2 (0.95%) above the 9.81 the deploy stack removes gravity with. Whatever
# its origin (scale factor or z bias -- one static pose cannot separate them), the residual
# integrates to ~0.09 m/s of velocity error per second of open-loop integration, which is
# large next to the ~0.2 m/s the estimator is trying to resolve. The EKF's b_f state exists
# to absorb exactly this; arm B (complementary filter) has no such state, so this is a
# concrete, pre-registered reason to expect B to drift and a specific thing to check in the
# results rather than discover afterwards.
G_NORM_MEASURED = 9.9030

# ⚠️ The same log CANNOT tell us how much of the horizontal specific force is accelerometer
# bias and how much is the hang being off-level: acc_x = +0.0688 m/s^2 corresponds to a 0.4
# deg tilt, and a single static pose cannot separate a tilt from a body-fixed bias. That is
# the tilt/bias degeneracy of doc/hrl/h1_2_ekf_design.md §8.1, showing up in the calibration
# data itself. Separating them needs the robot measured in more than one orientation.


# ---------------------------------------------------------------------------------------
# small SO(3) helpers
# ---------------------------------------------------------------------------------------
def _longest_run(mask: np.ndarray) -> tuple[int, int]:
  """[start, stop) of the longest contiguous True run; (0, 0) if there is none."""
  idx = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
  if not len(idx):
    return 0, 0
  starts, stops = idx[::2], idx[1::2]
  i = int(np.argmax(stops - starts))
  return int(starts[i]), int(stops[i])


def skew(v: np.ndarray) -> np.ndarray:
  return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
  """Rodrigues. Exact at w -> 0 via the series."""
  th = float(np.linalg.norm(w))
  K = skew(w)
  if th < 1e-8:
    return np.eye(3) + K + 0.5 * K @ K
  return np.eye(3) + (np.sin(th) / th) * K + ((1.0 - np.cos(th)) / th**2) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
  c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
  th = float(np.arccos(c))
  if th < 1e-8:
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
  return (th / (2.0 * np.sin(th))) * np.array(
    [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
  )


def quat_to_mat(q: np.ndarray) -> np.ndarray:
  """(w, x, y, z) -> R. Element order verified against State_RLHRL.cpp:626-629, which feeds
  Eigen's 4-arg (w, x, y, z) constructor."""
  w, x, y, z = q / np.linalg.norm(q)
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def rz(psi: float) -> np.ndarray:
  c, s = np.cos(psi), np.sin(psi)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------------------
# kinematics: pelvis-frame FK -> torso/IMU frame
# ---------------------------------------------------------------------------------------
def feet_in_pelvis(q12: np.ndarray, offset) -> tuple[np.ndarray, np.ndarray]:
  """Batched. ``q12`` [N, 12] sdk order -> foot points [N, 2, 3] and orientations
  [N, 2, 3, 3], both in the PELVIS frame."""
  t = torch.tensor(q12, dtype=torch.float64)
  ps, Rs = [], []
  for leg in (0, 1):
    p, R = leg_fk(t, leg)
    off = torch.tensor(offset, dtype=torch.float64)
    ps.append((p + R @ off).numpy())
    Rs.append(R.numpy())
  return np.stack(ps, axis=1), np.stack(Rs, axis=1)


def to_imu_frame(p_P: np.ndarray, R_P: np.ndarray, psi: np.ndarray):
  """Pelvis-frame foot pose -> IMU/torso frame: ``p^I = Rz(-psi) p^P - r_imu``.
  Verified against MuJoCo to 6.7e-16 m (doc/hrl/h1_2_kinematics.md §D)."""
  c, s = np.cos(psi), np.sin(psi)
  Rzn = np.zeros((len(psi), 3, 3))
  Rzn[:, 0, 0] = c; Rzn[:, 0, 1] = s
  Rzn[:, 1, 0] = -s; Rzn[:, 1, 1] = c
  Rzn[:, 2, 2] = 1.0
  p_I = np.einsum("nij,nkj->nki", Rzn, p_P) - R_IMU
  R_I = np.einsum("nij,nkjl->nkil", Rzn, R_P)
  return p_I, R_I


# ---------------------------------------------------------------------------------------
# session loading
# ---------------------------------------------------------------------------------------
@dataclass
class Session:
  """One flight-recorder session, reduced to unique sensor samples."""

  name: str
  t: np.ndarray            # [N]
  acc: np.ndarray          # [N, 3] torso frame, gravity INCLUDED
  gyro: np.ndarray         # [N, 3] torso frame  (nan if absent -> no est block)
  quat: np.ndarray         # [N, 4] torso attitude (w, x, y, z)
  q: np.ndarray            # [N, 13] leg + waist, encoder-zero offset applied
  dq: np.ndarray           # [N, 13] (nan if absent)
  cmd: np.ndarray          # [N, 3] commanded vx, vy, wz (nan if absent)
  rate_hz: float
  has_est: bool
  has_dq: bool
  joint_offset: list = field(default_factory=list)

  @property
  def psi(self) -> np.ndarray:
    return self.q[:, 12]

  def _still(self, thresh: float = 0.3, win_s: float = 0.25) -> np.ndarray:
    """Samples where the legs are verifiably not moving, from the encoders alone.

    Quiet standing sits at p90 |dq| 0.05-0.08 rad/s across all three 2026-08-12 A0
    sessions; a set-down or a step reaches 1-22. A moving MAX blanks the neighbourhood of
    each spike, so the settling ring-down after a disturbance is excluded too.
    """
    if not np.isfinite(self.dq).all():
      return np.ones(len(self.t), bool)   # no est_dq -> cannot verify, don't pretend to
    m = np.abs(self.dq[:, :12]).max(axis=1)
    k = max(1, int(win_s * self.rate_hz))
    win = np.lib.stride_tricks.sliding_window_view(np.pad(m, k, mode="edge"), 2 * k + 1)
    return win.max(axis=1) < thresh

  def regimes(self, min_s: float = 5.0) -> list[tuple[str, np.ndarray]]:
    """Split into a LEADING zero-command segment and the commanded remainder.

    Scoring a session whole pools two regimes whose errors cancel: on 2026-08-12 a pooled
    drift of -0.026 m made arm E look like the standout, while on its standing segment
    alone E drifted -0.491 m, worse than the incumbent. Only the leading zero-command
    stretch carries a known truth (v == 0) -- after the first command the robot has moved,
    so a later cmd==0 stretch is a different, unknown pose with no displacement reference.
    """
    if not np.isfinite(self.cmd).all():
      return [("all (no command column)", np.ones(len(self.t), bool))]
    zero = (np.abs(self.cmd) < 1e-6).all(axis=1)
    n_lead = int(np.argmin(zero)) if not zero.all() else len(zero)
    out = []
    lead = np.zeros(len(zero), bool)
    lead[:n_lead] = True
    if n_lead and self.t[n_lead - 1] - self.t[0] >= min_s:
      still = lead & self._still()
      # cmd==0 is NOT proof the robot is still: on 2026-08-12_10-16-41 the set-down
      # transient in the first 5 s moved the feet 137 mm and it alone carried the
      # session's apparent noise (arm A std vx 0.141 over the whole lead vs 0.011 after
      # it). Score the longest VERIFIED-still run, and say how much was dropped.
      a, b = _longest_run(still)
      kept = np.zeros(len(zero), bool)
      kept[a:b] = True
      drop = self.t[n_lead - 1] - self.t[0] - (self.t[b - 1] - self.t[a]) if b > a else 0.0
      if b - a > 2 and self.t[b - 1] - self.t[a] >= min_s:
        out.append((f"standing (cmd==0 AND still, true v==0; {drop:.1f}s dropped)", kept))
      else:
        out.append(("standing (cmd==0, STILLNESS UNVERIFIED)", lead))
    rest = ~lead
    if rest.sum() and self.t[rest][-1] - self.t[rest][0] >= min_s:
      out.append(("commanded (no v truth)", rest))
    return out or [("all", np.ones(len(self.t), bool))]

  def summary(self) -> str:
    span = self.t[-1] - self.t[0]
    return (
      f"{self.name}: {len(self.t)} samples over {span:.1f} s = {self.rate_hz:.0f} Hz  "
      f"est_block={'yes' if self.has_est else 'NO'} est_dq={'yes' if self.has_dq else 'NO'}"
    )


def load_session(csv_path: Path, entry: int | None = None) -> Session:
  """One session. ``entry`` selects a single FSM entry from a file that holds several.

  A controller that is kept running while the operator switches to passive and re-enters
  APPENDS to the same csv (safety_logger.h: re-entry keeps ``tick_``), with only the
  ``entry`` column separating the runs. Replaying such a file whole would difference across
  the seam between two runs and inject the pose change as a false velocity. Default is
  therefore the LAST entry, not the concatenation.
  """
  import csv as _csv

  with open(csv_path) as fh:
    rows = list(_csv.reader(fh))
  hdr, data = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
  idx = {n: i for i, n in enumerate(hdr)}
  arr = np.array(data, dtype=np.float64)

  suffix = ""
  if "entry" in idx:
    raw_entries = arr[:, idx["entry"]]
    if len(np.unique(raw_entries)) > 1:
      resolved = resolve_run_entry(raw_entries, entry, csv_path.name, default="last")
      arr = arr[arr[:, idx["entry"]] == resolved]
      suffix = f"#entry{resolved}"

  meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
  joint_offset = []
  if meta_path.exists():
    with open(meta_path) as fh:
      joint_offset = json.load(fh).get("joint_offset", [])

  has_est = "est_gyro_x" in idx
  has_dq = "est_dq0" in idx

  def cols(names):
    return arr[:, [idx[n] for n in names]]

  if has_est:
    acc = cols([f"est_acc_{a}" for a in "xyz"])
    gyro = cols([f"est_gyro_{a}" for a in "xyz"])
    quat = cols([f"est_quat_{a}" for a in "wxyz"])
    q = cols([f"est_q{i}" for i in range(13)])
    dq = cols([f"est_dq{i}" for i in range(13)]) if has_dq else np.full_like(q, np.nan)
    key = np.concatenate([acc, gyro, q], axis=1)
  else:
    # DRY-RUN FALLBACK ONLY. These columns come from the articulation cache, which
    # policy_step() refreshes at 50 Hz, and there is no gyro at all. Good enough to exercise
    # the pipeline; NOT good enough to score arms (design doc §10 P0).
    acc = cols([f"acc_{a}" for a in "xyz"])
    quat = cols([f"quat_{a}" for a in "wxyz"])
    q = cols([f"meas_q{i}" for i in range(13)])
    gyro = np.full((len(arr), 3), np.nan)
    dq = cols([f"meas_dq{i}" for i in range(13)]) if "meas_dq0" in idx else np.full_like(q, np.nan)
    # Dedupe on the JOINTS alone here, not on acc. In this fallback the joints come from the
    # 50 Hz articulation cache while acc updates at ~500 Hz, so keying on acc would keep ~10
    # rows per joint sample and the kinematic arms would difference a staircase -- pure
    # quantisation, measured at std 2.9 m/s against the estimator's real ~0.2. Keying on the
    # limiting channel at least makes the dry run self-consistent.
    key = q

  # Keep only rows where the sensor block actually changed: the recorder logs at the loop
  # rate, but LowState arrives at ~500 Hz, so ~half the rows are exact repeats.
  keep = np.r_[True, np.any(np.diff(key, axis=0) != 0, axis=1)]
  cmd_names = ["cmd_vx", "cmd_vy", "cmd_wz"]
  cmd = (cols(cmd_names)[keep] if all(n in idx for n in cmd_names)
         else np.full((int(keep.sum()), 3), np.nan))
  t = arr[keep, idx["t"]]
  span = t[-1] - t[0]
  gyro = gyro[keep]
  quat = quat[keep]
  if not has_est:
    # DEGRADED FALLBACK, dry-run only. Reconstruct the body rate from successive attitudes:
    # dq = q_{k+1} (x) q_k^-1, w ~ 2 vec(dq)/dt. This is the exact handicap that made the
    # 2026-08-10 gate-2 study uninformative -- the |w x p| term carries half the leg-odometry
    # magnitude, so a 50 Hz cached reconstruction dominates the error it is meant to measure.
    gyro = _gyro_from_quat(quat, t)
  return Session(
    name=csv_path.stem + suffix, t=t, acc=acc[keep], gyro=gyro, quat=quat,
    q=q[keep], dq=dq[keep], cmd=cmd,
    rate_hz=(len(t) / span if span > 0 else float("nan")),
    has_est=has_est, has_dq=has_dq and has_est, joint_offset=joint_offset,
  )


def _gyro_from_quat(quat: np.ndarray, t: np.ndarray) -> np.ndarray:
  """Body-frame angular rate from successive attitude samples. Fallback only."""
  n = len(t)
  w = np.zeros((n, 3))
  for k in range(1, n):
    dt = t[k] - t[k - 1]
    if dt <= 0:
      w[k] = w[k - 1]
      continue
    Rprev, Rnow = quat_to_mat(quat[k - 1]), quat_to_mat(quat[k])
    w[k] = so3_log(Rprev.T @ Rnow) / dt
  w[0] = w[1] if n > 1 else 0.0
  return w


# ---------------------------------------------------------------------------------------
# derived per-sample quantities shared by every arm
# ---------------------------------------------------------------------------------------
def prepare(s: Session, offset=FOOT_SITE_A):
  """Feet in the pelvis and IMU frames, pelvis angular rate, projected gravity."""
  p_P, R_P = feet_in_pelvis(s.q[:, :12], offset)
  p_I, R_I = to_imu_frame(p_P, R_P, s.psi)

  # w_P = Rz(psi) w_T - psi_dot z   (State_RLHRL.cpp:652-654).
  c, s_ = np.cos(s.psi), np.sin(s.psi)
  Rzp = np.zeros((len(s.t), 3, 3))
  Rzp[:, 0, 0] = c; Rzp[:, 0, 1] = -s_
  Rzp[:, 1, 0] = s_; Rzp[:, 1, 1] = c
  Rzp[:, 2, 2] = 1.0
  w_P = np.einsum("nij,nj->ni", Rzp, s.gyro)
  dpsi = s.dq[:, 12] if s.has_dq else np.gradient(s.psi, s.t)
  w_P[:, 2] -= dpsi

  # projected gravity in the pelvis frame: Rz(psi) * (C * g_hat), C = R_WT^T
  C = np.stack([quat_to_mat(qq).T for qq in s.quat])       # world -> torso
  g_T = np.einsum("nij,j->ni", C, np.array([0.0, 0.0, -1.0]))
  cc, ss = np.cos(s.psi), np.sin(s.psi)
  grav_P = np.stack([cc * g_T[:, 0] - ss * g_T[:, 1], ss * g_T[:, 0] + cc * g_T[:, 1], g_T[:, 2]], 1)
  g_I = np.einsum("nij,j->ni", C, np.array([0.0, 0.0, -1.0]))  # gravity dir, torso frame
  return dict(p_P=p_P, R_P=R_P, p_I=p_I, R_I=R_I, w_P=w_P, grav_P=grav_P, C=C,
              g_I=g_I, offset=offset)


def stance_index(p_P: np.ndarray, grav_P: np.ndarray) -> np.ndarray:
  """Stance = the foot further along projected gravity, i.e. the LOWER one.
  C++ uses `>=` to pick foot 0 (base_state.h leg_odom_velocity)."""
  along = np.einsum("nkj,nj->nk", p_P, grav_P)
  return np.where(along[:, 0] >= along[:, 1], 0, 1)


# ---------------------------------------------------------------------------------------
# ARM A -- the deployed leg odometry, replayed exactly
# ---------------------------------------------------------------------------------------
def arm_a_legodom(s: Session, pre: dict) -> np.ndarray:
  """v = -(p_stance - p_stance_prev)/dt - w_P x p_stance, in the PELVIS frame.

  Transcription of hrl::leg_odom_velocity. Differencing is PER FOOT and never across a
  stance switch, which would inject the ~0.33 m inter-foot gap as a false spike.
  """
  p, w_P = pre["p_P"], pre["w_P"]
  st = stance_index(p, pre["grav_P"])
  n = len(s.t)
  v = np.full((n, 3), np.nan)
  dt = np.diff(s.t)
  ps = p[np.arange(n), st]
  for k in range(1, n):
    if dt[k - 1] <= 0 or st[k] != st[k - 1]:
      continue  # no previous sample for THIS foot: the C++ has the same guard
    v[k] = -(ps[k] - p[k - 1, st[k]]) / dt[k - 1] - np.cross(w_P[k], ps[k])
  return v


# ---------------------------------------------------------------------------------------
# ARM E -- Jacobian leg odometry (same information, no position differencing)
# ---------------------------------------------------------------------------------------
def _rot_batch(axis: int, a: np.ndarray) -> np.ndarray:
  """Batched elementary rotation about x/y/z. Mirrors leg_odom._rot."""
  c, s = np.cos(a), np.sin(a)
  o, z = np.ones_like(a), np.zeros_like(a)
  if axis == 0:
    rows = [(o, z, z), (z, c, -s), (z, s, c)]
  elif axis == 1:
    rows = [(c, z, s), (z, o, z), (-s, z, c)]
  else:
    rows = [(c, -s, z), (s, c, z), (z, z, o)]
  return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def leg_jacobian_pelvis(q12: np.ndarray, leg: int, offset=FOOT_SITE_A):
  """Geometric Jacobian of one foot point w.r.t. that leg's 6 joints, PELVIS frame.

  Returns ``(J_v [n, 3, 6], J_w [n, 3, 6], p_foot [n, 3])``. Columns are
  ``a_i x (p_foot - o_i)`` and ``a_i``; verified against finite differences to 4e-8
  (doc/hrl/h1_2_kinematics.md §E.2). Joint order and link table follow ``leg_fk``.
  """
  n = len(q12)
  sy = 1.0 if leg == 0 else -1.0
  j = q12[:, 6 * leg : 6 * leg + 6]
  axes = np.zeros((n, 6, 3))
  origins = np.zeros((n, 6, 3))
  R = _rot_batch(2, j[:, 0])
  o = np.tile(np.array([0.0, sy * 0.0875, -0.1632]), (n, 1))
  axes[:, 0] = np.array([0.0, 0.0, 1.0])
  origins[:, 0] = o
  p = o + np.einsum("nij,j->ni", R, np.array([0.0, sy * 0.0755, 0.0]))
  origins[:, 1] = p
  axes[:, 1] = np.einsum("nij,j->ni", R, np.array([0.0, 1.0, 0.0]))
  R = R @ _rot_batch(1, j[:, 1])
  origins[:, 2] = p
  axes[:, 2] = np.einsum("nij,j->ni", R, np.array([1.0, 0.0, 0.0]))
  R = R @ _rot_batch(0, j[:, 2])
  p = p + np.einsum("nij,j->ni", R, np.array([0.0, 0.0, -0.4]))
  origins[:, 3] = p
  axes[:, 3] = np.einsum("nij,j->ni", R, np.array([0.0, 1.0, 0.0]))
  R = R @ _rot_batch(1, j[:, 3])
  p = p + np.einsum("nij,j->ni", R, np.array([0.0, 0.0, -0.4]))
  origins[:, 4] = p
  axes[:, 4] = np.einsum("nij,j->ni", R, np.array([0.0, 1.0, 0.0]))
  R = R @ _rot_batch(1, j[:, 4])
  p = p + np.einsum("nij,j->ni", R, np.array([0.0, 0.0, -0.02]))
  origins[:, 5] = p
  axes[:, 5] = np.einsum("nij,j->ni", R, np.array([1.0, 0.0, 0.0]))
  R = R @ _rot_batch(0, j[:, 5])
  pf = p + np.einsum("nij,j->ni", R, np.array(offset))
  J_v = np.transpose(np.cross(axes, (pf[:, None, :] - origins)), (0, 2, 1))
  J_w = np.transpose(axes, (0, 2, 1))
  return J_v, J_w, pf


def foot_jacobians_imu(q12: np.ndarray, psi: np.ndarray, offset=FOOT_SITE_A):
  """Both feet's contact-frame Jacobians in the IMU/TORSO frame, 3x7 per foot.

  Columns 0-5 are that leg's joints, column 6 is the WAIST -- the seventh kinematic DoF
  between the IMU and the feet, which Rotella's leg-only chain has no analogue for:

      J_v^I = [ Rz(-psi) J_v^P | -z_hat x (p^I + r_imu) ]
      J_w^I = [ Rz(-psi) J_w^P | -z_hat                 ]

  using d/dpsi Rz(-psi) = -[z_hat]x Rz(-psi) (doc/hrl/h1_2_kinematics.md §E.2). Returns
  ``(J_v [n, 2, 3, 7], J_w [n, 2, 3, 7])``.
  """
  n = len(q12)
  Rzn = _rot_batch(2, -psi)
  zhat = np.array([0.0, 0.0, 1.0])
  J_v = np.zeros((n, 2, 3, 7))
  J_w = np.zeros((n, 2, 3, 7))
  for leg in (0, 1):
    Jv_P, Jw_P, pf_P = leg_jacobian_pelvis(q12, leg, offset)
    J_v[:, leg, :, :6] = np.einsum("nij,njk->nik", Rzn, Jv_P)
    J_w[:, leg, :, :6] = np.einsum("nij,njk->nik", Rzn, Jw_P)
    p_I = np.einsum("nij,nj->ni", Rzn, pf_P) - R_IMU
    J_v[:, leg, :, 6] = -np.cross(zhat, p_I + R_IMU)
    J_w[:, leg, :, 6] = -zhat
  return J_v, J_w


def arm_e_jacobian(s: Session, pre: dict) -> np.ndarray:
  """v = -(J qdot) - w_P x p_stance. Same information as arm A with the position
  differencing removed -- but dq is differentiated on the MOTOR BOARD, so this measures
  where the differentiation happens, not its absence (design doc §10)."""
  if not s.has_dq:
    return np.full((len(s.t), 3), np.nan)
  st = stance_index(pre["p_P"], pre["grav_P"])
  v = np.full((len(s.t), 3), np.nan)
  for leg in (0, 1):
    m = st == leg
    if not m.any():
      continue
    J, _, pf = leg_jacobian_pelvis(s.q[m, :12], leg)
    qd = s.dq[m, 6 * leg : 6 * leg + 6]
    v[m] = -np.einsum("nij,nj->ni", J, qd) - np.cross(pre["w_P"][m], pf)
  return v


# ---------------------------------------------------------------------------------------
# ARM B -- complementary filter: the cheap control that gates adoption
# ---------------------------------------------------------------------------------------
def arm_b_complementary(s: Session, pre: dict, v_legodom: np.ndarray, tau: float = 0.20):
  """High-pass the gravity-free accelerometer, low-pass the leg odometry, in the PELVIS
  frame. One time constant, no states, no contact logic -- if the EKF cannot beat this it
  has not earned 27 states (design doc §9)."""
  n = len(s.t)
  v = np.zeros((n, 3))
  g_hat_T = np.einsum("nij,j->ni", pre["C"], np.array([0.0, 0.0, -1.0]))
  a_free_T = s.acc + g_hat_T * 9.81          # a + proj_grav*|g|  (State_RLHRL.cpp:606)
  cc, ss = np.cos(s.psi), np.sin(s.psi)
  a_P = np.stack(
    [cc * a_free_T[:, 0] - ss * a_free_T[:, 1], ss * a_free_T[:, 0] + cc * a_free_T[:, 1],
     a_free_T[:, 2]], axis=1)
  last = np.zeros(3)
  ok = ~np.isnan(v_legodom).any(axis=1)
  for k in range(1, n):
    dt = s.t[k] - s.t[k - 1]
    if dt <= 0:
      v[k] = last
      continue
    pred = last + a_P[k] * dt
    if ok[k]:
      alpha = dt / (tau + dt)
      last = (1.0 - alpha) * pred + alpha * v_legodom[k]
    else:
      last = pred
    v[k] = last
  return v


# ---------------------------------------------------------------------------------------
# the EKF (arms C, D, F)
# ---------------------------------------------------------------------------------------
@dataclass
class MeasCfg:
  """Everything that goes into R.

  MEASURED 2026-08-12 from the 655 s passive hang (all_joints_2026-08-12_08-27-18_10min_hang):
  per-joint noise taken as ``std(diff(q))/sqrt(2)``, which is immune to the slow settling
  drift that inflates a raw std (the knee reads std 1.98e-4 but high-frequency 1.6e-5 --
  that difference is the leg creeping while it hangs, which is real motion, not noise).

  ⚠️ These are a LOWER BOUND: the hang is unloaded and vibration-free. Under load with the
  motors active, encoder noise can be materially higher, so treat a large fitted residual as
  a reason to re-measure during a policy run rather than to inflate sigma_slip.

  CONSEQUENCE, stated plainly: at these values the ``G Sigma_q G^T`` term is ~3 orders below
  sigma_slip/sigma_model, so R is dominated by foot slip and kinematic model error, and the
  FK-Jacobian construction contributes almost nothing numerically. It is kept because it is
  the correct structure and because sigma_q under load is not yet known -- not because it is
  currently moving the answer.
  """

  sigma_q: float = 1.64e-5     # rad, per leg joint   MEASURED (max over 12, high-frequency)
  sigma_psi: float = 2.61e-5   # rad, waist encoder   MEASURED
  sigma_slip: float = 3.0e-3   # m,  real foot motion while nominally planted   [assumed]
  sigma_model: float = 3.0e-3  # m,  link-length + sole-geometry error          [assumed]
  sigma_rot: float = 2.0e-2    # rad, foot orientation model error              [assumed]


def to_world(v_pelvis: np.ndarray, s: Session, pre: dict) -> np.ndarray:
  """Pelvis-frame arm output -> world, inverting the output conversion in ``run_ekf``.

  Displacement is the E2 metric (drift vs a tape measure), and it only equals the integral
  of the reported velocity in the WORLD frame: integrating pelvis-frame vx over a walk that
  turns measures nothing physical. Yaw is unobservable, so the world frame here is "yaw as
  the vendor filter believes it" -- fine over a straight there-and-back, not over a spin.
  """
  v_T = np.einsum("nij,nj->ni", _rot_batch(2, -s.psi), v_pelvis) + np.cross(s.gyro, R_IMU)
  return np.einsum("nji,nj->ni", pre["C"], v_T)


def build_R(J_v: np.ndarray, J_w: np.ndarray, alpha: np.ndarray, cfg: MeasCfg,
            use_ori: bool) -> np.ndarray:
  """Measurement covariance for one sample, BUILT from the kinematics rather than tuned.

  The clean way to get every correlation right at once is to propagate the joint-space
  covariance through one stacked kinematic Jacobian ``G`` (m x 13, the 12 leg joints plus
  the waist)::

      R = G Sigma_q G^T  +  diag(slip, model, rot)  +  contact inflation

  This automatically produces the two couplings a per-foot diagonal R would silently drop:

    * **cross-foot.** Both feet's Jacobians contain the SAME waist column, so their
      measurement noises are correlated -- ``cov(n_L, n_R) = J_psi,L sigma_psi^2 J_psi,R^T``.
      Rotella assumes uncorrelated feet; on the H1-2 that is false because the IMU sits
      behind the waist joint (design doc §3.2).
    * **position/orientation within a foot**, driven by the same encoder noise.

  Row order is [L_pos, L_rot, R_pos, R_rot] (rot rows only when ``use_ori``).
  """
  per = 6 if use_ori else 3
  m = 2 * per
  G = np.zeros((m, 13))
  for leg in (0, 1):
    r0 = leg * per
    G[r0 : r0 + 3, 6 * leg : 6 * leg + 6] = J_v[leg][:, :6]
    G[r0 : r0 + 3, 12] = J_v[leg][:, 6]
    if use_ori:
      G[r0 + 3 : r0 + 6, 6 * leg : 6 * leg + 6] = J_w[leg][:, :6]
      G[r0 + 3 : r0 + 6, 12] = J_w[leg][:, 6]
  Sq = np.diag(np.r_[np.full(12, cfg.sigma_q**2), cfg.sigma_psi**2])
  R = G @ Sq @ G.T
  for leg in (0, 1):
    r0 = leg * per
    # §7.1: a foot we are less sure about gets a larger measurement covariance, and the
    # same alpha simultaneously frees its process noise (see BaseEkf.predict).
    infl = (1.0 / max(alpha[leg], 1e-3) - 1.0) ** 2 * cfg.sigma_slip**2
    R[r0 : r0 + 3, r0 : r0 + 3] += np.eye(3) * (cfg.sigma_slip**2 + cfg.sigma_model**2 + infl)
    if use_ori:
      R[r0 + 3 : r0 + 6, r0 + 3 : r0 + 6] += np.eye(3) * (cfg.sigma_rot**2 + infl)
  return R


class BaseEkf:
  """Rotella-style error-state EKF, body frame = TORSO/IMU (design doc §1-§5).

  State  x  = [r, v, R_WB, p_L, p_R, R_L, R_R, b_f, b_w]
  Error  dx = [dr, dv, dphi, dp_L, dp_R, dth_L, dth_R, db_f, db_w]  in R^27
              (R^21 when use_orientation=False: the dth blocks are dropped)

  Attitude convention, derived and checked by finite difference in --selftest:
  ``R_true = R_hat exp([dphi]x)`` (right-multiplied, body-frame error), which reproduces
  Rotella's F_c and H exactly. ``C = R^T`` maps world -> body, as in the paper.
  """

  def __init__(self, use_orientation: bool = False, sig=None):
    self.use_ori = use_orientation
    self.nf = 2
    # error-state layout
    self.i_r, self.i_v, self.i_phi = 0, 3, 6
    self.i_p = 9                       # p_L, p_R
    self.i_th = 15 if use_orientation else None
    self.i_bf = (21 if use_orientation else 15)
    self.i_bw = self.i_bf + 3
    self.n = self.i_bw + 3
    # MEASURED 2026-08-12 from the 655 s passive hang, as noise DENSITIES (per sqrt(Hz)):
    # high-frequency std / sqrt(f_s), which is what Q = sigma^2 * dt wants. The previous
    # guesses (acc 0.05, gyro 0.005) were ~3x too LARGE, which makes the filter distrust its
    # own prediction and lean on the kinematic measurement -- i.e. degenerate toward the very
    # finite difference it is supposed to improve on.
    #   gyro x/y/z density 0.00164 / 0.00131 / 0.00144   -> 0.0016
    #   acc  x/y/z density 0.01451 / 0.00576 / 0.00676   -> 0.0145 (worst axis, x)
    # ⚠️ UPPER bounds: the 50 Hz hang log is decimated from ~500 Hz with no anti-aliasing, so
    # high-frequency noise folds down into the measured band. ba/bw remain assumptions -- the
    # bias-instability fit needs the Allan curve, not just the white-noise floor.
    s = dict(acc=0.0145, gyro=0.0016, ba=1e-4, bw=1e-5, foot_p=1e-3, foot_th=1e-2,
             meas_p=5e-3, meas_th=5e-2, att=5e-2)
    if sig:
      s.update(sig)
    self.s = s
    self.reset()

  def reset(self, r0=None, v0=None, R0=None):
    self.r = np.zeros(3) if r0 is None else np.array(r0, float)
    self.v = np.zeros(3) if v0 is None else np.array(v0, float)
    self.R = np.eye(3) if R0 is None else np.array(R0, float)
    self.p = np.zeros((2, 3))
    self.Rf = np.stack([np.eye(3), np.eye(3)])
    self.bf = np.zeros(3)
    self.bw = np.zeros(3)
    self.P = np.eye(self.n) * 1e-2
    self.P[self.i_r : self.i_r + 3] *= 1e2     # absolute position is unobservable anyway
    self.anchored = np.zeros(2, dtype=bool)

  # -- measurement model -----------------------------------------------------------
  def h_pos(self, i: int) -> np.ndarray:
    """Predicted foot position in the body frame: C (p_i - r)."""
    return self.R.T @ (self.p[i] - self.r)

  def h_rot(self, i: int) -> np.ndarray:
    """Predicted foot orientation, body frame: R^T R_foot."""
    return self.R.T @ self.Rf[i]

  def H_pos(self, i: int) -> np.ndarray:
    H = np.zeros((3, self.n))
    C = self.R.T
    H[:, self.i_r : self.i_r + 3] = -C
    H[:, self.i_phi : self.i_phi + 3] = skew(self.h_pos(i))
    H[:, self.i_p + 3 * i : self.i_p + 3 * i + 3] = C
    return H

  def H_rot(self, i: int) -> np.ndarray:
    H = np.zeros((3, self.n))
    H[:, self.i_phi : self.i_phi + 3] = -self.h_rot(i).T
    H[:, self.i_th + 3 * i : self.i_th + 3 * i + 3] = np.eye(3)
    return H

  # -- prediction ------------------------------------------------------------------
  def predict(self, acc: np.ndarray, gyro: np.ndarray, dt: float, alpha: np.ndarray):
    f = acc - self.bf
    w = gyro - self.bw
    a_w = self.R @ f + GRAVITY
    self.r = self.r + self.v * dt + 0.5 * a_w * dt * dt
    self.v = self.v + a_w * dt
    self.R = self.R @ so3_exp(w * dt)

    F = np.eye(self.n)
    F[self.i_r : self.i_r + 3, self.i_v : self.i_v + 3] = np.eye(3) * dt
    F[self.i_v : self.i_v + 3, self.i_phi : self.i_phi + 3] = -self.R @ skew(f) * dt
    F[self.i_v : self.i_v + 3, self.i_bf : self.i_bf + 3] = -self.R * dt
    F[self.i_phi : self.i_phi + 3, self.i_phi : self.i_phi + 3] = so3_exp(-w * dt)
    F[self.i_phi : self.i_phi + 3, self.i_bw : self.i_bw + 3] = -np.eye(3) * dt

    Q = np.zeros((self.n, self.n))
    Q[self.i_v : self.i_v + 3, self.i_v : self.i_v + 3] = np.eye(3) * (self.s["acc"] ** 2) * dt
    Q[self.i_phi : self.i_phi + 3, self.i_phi : self.i_phi + 3] = np.eye(3) * (self.s["gyro"] ** 2) * dt
    Q[self.i_bf : self.i_bf + 3, self.i_bf : self.i_bf + 3] = np.eye(3) * (self.s["ba"] ** 2) * dt
    Q[self.i_bw : self.i_bw + 3, self.i_bw : self.i_bw + 3] = np.eye(3) * (self.s["bw"] ** 2) * dt
    # A foot is free to move exactly as far as it is NOT in contact (design doc §7.1).
    for i in range(2):
      free = (1.0 - alpha[i]) * 1.0 + 1e-6
      Q[self.i_p + 3 * i : self.i_p + 3 * i + 3, self.i_p + 3 * i : self.i_p + 3 * i + 3] = (
        np.eye(3) * (self.s["foot_p"] ** 2 + free**2) * dt)
      if self.use_ori:
        Q[self.i_th + 3 * i : self.i_th + 3 * i + 3, self.i_th + 3 * i : self.i_th + 3 * i + 3] = (
          np.eye(3) * (self.s["foot_th"] ** 2 + free**2) * dt)
    self.P = F @ self.P @ F.T + Q

  # -- update ----------------------------------------------------------------------
  def update_feet(self, s_p: np.ndarray, s_R: np.ndarray | None, alpha: np.ndarray,
                  R_meas: np.ndarray, gate: float = 25.0) -> bool:
    """Both feet in ONE update, so the cross-foot waist correlation in ``R_meas`` is
    actually used (updating foot-at-a-time would discard those off-diagonal blocks).

    ``s_p`` [2, 3] and ``s_R`` [2, 3, 3] are the FK measurements in the body frame;
    ``R_meas`` is ``build_R``'s stacked matrix with row order [L_pos, L_rot, R_pos, R_rot].
    Returns False if nothing was applied (no contact, or the innovation was gated).
    """
    per = 6 if self.use_ori else 3
    rows, e_parts, H_parts = [], [], []
    for i in (0, 1):
      if alpha[i] <= 1e-3:
        continue
      if not self.anchored[i]:
        # First contact: place the foot where FK says it is rather than fighting a stale
        # pose. Rotella gets the same effect from the inflated covariance; doing it
        # explicitly avoids one large transient innovation per touchdown.
        self.p[i] = self.r + self.R @ s_p[i]
        if s_R is not None:
          self.Rf[i] = self.R @ s_R[i]
        self.anchored[i] = True
        continue
      r0 = i * per
      rows.extend(range(r0, r0 + 3))
      e_parts.append(s_p[i] - self.h_pos(i))
      H_parts.append(self.H_pos(i))
      if self.use_ori and s_R is not None:
        rows.extend(range(r0 + 3, r0 + 6))
        e_parts.append(so3_log(self.h_rot(i).T @ s_R[i]))
        H_parts.append(self.H_rot(i))
    if not rows:
      return False
    e = np.concatenate(e_parts)
    H = np.vstack(H_parts)
    Rm = R_meas[np.ix_(rows, rows)]
    S = H @ self.P @ H.T + Rm
    try:
      Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
      return False
    if float(e @ Sinv @ e) > gate * len(e):      # Mahalanobis gate (design doc §7.2)
      return False
    K = self.P @ H.T @ Sinv
    self._inject(K @ e)
    IKH = np.eye(self.n) - K @ H
    self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
    return True

  def update_gravity(self, acc: np.ndarray, sigma: float = 0.05,
                     tol: float = 0.5) -> bool:
    """Accelerometer as a direct TILT observation, applied only while near-static.

    WHY THIS EXISTS (it is not in Rotella or Bloesch). In the strapdown formulation the
    accelerometer is an INPUT to the prediction, never a measurement, and attitude is
    corrected only indirectly through the foot-position term ``(C(p-r))x dphi``. On this
    robot that coupling proved far too weak: measured 2026-08-12 on a real stepping session,
    the position-only arm's TILT error ran to 72.8 deg, which leaks gravity into horizontal
    specific force and diverged the velocity by tens of m/s. Neither the contact-detection
    fix nor the innovation gate touched it (95% of updates were applied).

    While the body is not accelerating, ``f ~ C * (0,0,+|g|)``, so the measured direction
    observes two of the three attitude DoF. Yaw is untouched by construction: the Jacobian
    ``[h]x`` is rank 2 with its nullspace along h, so this cannot invent yaw observability
    that the system does not have.

    Gated on ``| |f| - |g| | < tol`` so it is skipped exactly when the assumption fails
    (impacts, swing). Returns whether it was applied.
    """
    n = float(np.linalg.norm(acc))
    if abs(n - 9.81) > tol:
      return False
    m = acc / n                                   # measured gravity direction, body frame
    h = self.R.T @ np.array([0.0, 0.0, 1.0])      # predicted, = C * z_world
    H = np.zeros((3, self.n))
    H[:, self.i_phi : self.i_phi + 3] = skew(h)
    Rm = np.eye(3) * sigma**2
    S = H @ self.P @ H.T + Rm
    K = self.P @ H.T @ np.linalg.inv(S)
    self._inject(K @ (m - h))
    IKH = np.eye(self.n) - K @ H
    self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
    return True

  def update_attitude(self, R_meas: np.ndarray):
    """Arm F only: the vendor quaternion as a direct attitude measurement."""
    e = so3_log(self.R.T @ R_meas)
    H = np.zeros((3, self.n))
    H[:, self.i_phi : self.i_phi + 3] = np.eye(3)
    Rm = np.eye(3) * self.s["att"] ** 2
    S = H @ self.P @ H.T + Rm
    K = self.P @ H.T @ np.linalg.inv(S)
    self._inject(K @ e)
    IKH = np.eye(self.n) - K @ H
    self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T

  def _inject(self, dx: np.ndarray):
    self.r += dx[self.i_r : self.i_r + 3]
    self.v += dx[self.i_v : self.i_v + 3]
    self.R = self.R @ so3_exp(dx[self.i_phi : self.i_phi + 3])
    for i in range(2):
      self.p[i] += dx[self.i_p + 3 * i : self.i_p + 3 * i + 3]
      if self.use_ori:
        self.Rf[i] = self.Rf[i] @ so3_exp(dx[self.i_th + 3 * i : self.i_th + 3 * i + 3])
    self.bf += dx[self.i_bf : self.i_bf + 3]
    self.bw += dx[self.i_bw : self.i_bw + 3]


def contact_alpha(p_I: np.ndarray, g_I: np.ndarray, hyst: float = 0.04) -> np.ndarray:
  """Continuous contact confidence from foot height alone, by FK relative to the OTHER foot
  (design doc §7.2/§7.3: never from the filter's own state, which would close a bad-state ->
  wrong-contact -> worse-state loop). 1 = the lower foot, falling off as the other lifts.

  ⚠️ "Lower" means FURTHER ALONG PROJECTED GRAVITY, not smaller body-frame z. Using the body
  z axis was a real bug here (2026-08-12): under torso tilt the two are different axes, the
  detector reported both feet planted 100% of the time on a stepping session, and feeding a
  swinging foot to the filter as "stationary" drove the position-only arm's TILT error to
  72.8 deg -- which then leaks gravity straight into horizontal velocity. The deployed C++
  has always used projected gravity (base_state.h leg_odom_velocity); this now matches it.
  """
  along = np.einsum("nkj,nj->nk", p_I, g_I)                     # larger = further down
  d = along.max(axis=1, keepdims=True) - along                  # 0 for the lower foot
  return np.clip(1.0 - d / hyst, 0.0, 1.0)


def run_ekf(s: Session, pre: dict, use_ori: bool, use_att: bool, sig=None,
            meas: MeasCfg | None = None, v0_pelvis: np.ndarray | None = None,
            use_grav: bool = False) -> np.ndarray:
  """Returns the base velocity in the PELVIS frame, comparable with arms A/B/E.

  ``v0_pelvis`` WARM-STARTS the velocity state from the incumbent's first valid sample.
  This is not a courtesy to the filter, it is required for short runs to mean anything: from
  a cold start the filter needs ~2-4 s to pull v in through the r/p coupling alone (measured
  in --selftest), so on a 2.5 s recording a cold-started EKF would be scored entirely on its
  transient, against a stateless finite difference that has none. Warm-starting removes a
  handicap that is an artifact of the replay, not a property of the estimator -- online, the
  filter would have been running for minutes. Reported both ways in --selftest.
  """
  n = len(s.t)
  meas = meas or MeasCfg()
  ekf = BaseEkf(use_orientation=use_ori, sig=sig)
  ekf.reset(R0=pre["C"][0].T)
  if v0_pelvis is not None and np.all(np.isfinite(v0_pelvis)):
    # pelvis -> torso -> world, inverting the output conversion below
    v_T = rz(-s.psi[0]) @ v0_pelvis + np.cross(s.gyro[0], R_IMU)
    ekf.v = pre["C"][0].T @ v_T
    ekf.P[ekf.i_v : ekf.i_v + 3, ekf.i_v : ekf.i_v + 3] = np.eye(3) * 0.25**2
  alpha = contact_alpha(pre["p_I"], pre["g_I"])
  # The kinematic Jacobians depend only on the encoders, so they are precomputed for the
  # whole session in one batched pass rather than rebuilt inside the filter loop.
  J_v, J_w = foot_jacobians_imu(s.q[:, :12], s.psi, offset=pre["offset"])
  out = np.full((n, 3), np.nan)
  for k in range(n):
    dt = s.t[k] - s.t[k - 1] if k else 0.0
    if k and 0 < dt < 0.1:
      ekf.predict(s.acc[k], s.gyro[k], dt, alpha[k])
    if use_grav:
      ekf.update_gravity(s.acc[k])
    if use_att:
      ekf.update_attitude(pre["C"][k].T)
    R_meas = build_R(J_v[k], J_w[k], alpha[k], meas, use_ori)
    ekf.update_feet(pre["p_I"][k], pre["R_I"][k] if use_ori else None, alpha[k], R_meas)
    # world velocity -> torso -> pelvis:  v_P = Rz(psi) (v_I^T - w_T x r_imu)
    v_T = ekf.R.T @ ekf.v
    v_P = rz(s.psi[k]) @ (v_T - np.cross(s.gyro[k] - ekf.bw, R_IMU))
    out[k] = v_P
  return out


# ---------------------------------------------------------------------------------------
# self-tests: the Jacobians and the filter, with no hardware involved
# ---------------------------------------------------------------------------------------
def selftest() -> int:
  rng = np.random.default_rng(0)
  fails = 0

  # (1) measurement Jacobians vs finite differences of the true measurement functions
  for use_ori in (False, True):
    ekf = BaseEkf(use_orientation=use_ori)
    ekf.r = rng.normal(size=3)
    ekf.v = rng.normal(size=3)
    ekf.R = so3_exp(rng.normal(size=3) * 0.4)
    ekf.p = rng.normal(size=(2, 3))
    ekf.Rf = np.stack([so3_exp(rng.normal(size=3) * 0.3) for _ in range(2)])
    for i in (0, 1):
      base_p, base_R = ekf.h_pos(i), ekf.h_rot(i)
      Jp = np.zeros((3, ekf.n))
      Jr = np.zeros((3, ekf.n))
      eps = 1e-6
      for c in range(ekf.n):
        d = np.zeros(ekf.n)
        d[c] = eps
        pert = BaseEkf(use_orientation=use_ori)
        pert.r, pert.v, pert.R = ekf.r.copy(), ekf.v.copy(), ekf.R.copy()
        pert.p, pert.Rf = ekf.p.copy(), ekf.Rf.copy()
        pert.bf, pert.bw = ekf.bf.copy(), ekf.bw.copy()
        pert._inject(d)
        Jp[:, c] = (pert.h_pos(i) - base_p) / eps
        Jr[:, c] = so3_log(base_R.T @ pert.h_rot(i)) / eps
      ep = np.abs(Jp - ekf.H_pos(i)).max()
      print(f"  H_pos  foot {i} use_ori={use_ori}: max|analytic - FD| = {ep:.2e}")
      fails += ep > 1e-5
      if use_ori:
        er = np.abs(Jr - ekf.H_rot(i)).max()
        print(f"  H_rot  foot {i}: max|analytic - FD| = {er:.2e}")
        fails += er > 1e-5

  # (2) the filter on synthetic truth: constant world velocity, both feet planted, perfect
  #     measurements, level attitude, cold start (v_hat = 0 against v_true = 0.3).
  #     There is no acceleration signal here, so velocity is recovered purely through the
  #     coupling of r and p_i in the position measurement -- the mechanism of §F.1. The
  #     assertion is therefore about CONVERGENCE, not about a single early sample: a filter
  #     with a sign or frame error plateaus at a finite offset (or diverges) instead of
  #     decaying, which is what the ratio check below catches.
  n, dt = 4000, 0.002
  v_true = np.array([0.30, -0.12, 0.0])
  ekf = BaseEkf()
  ekf.reset()
  p_feet_w = np.array([[0.1, 0.16, 0.0], [0.1, -0.16, 0.0]])
  R_eye = np.eye(6) * (5e-3**2)   # perfect-measurement case: R is not under test here
  err = np.zeros(n)
  for k in range(n):
    r_true = v_true * (k * dt)
    ekf.predict(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt, np.ones(2))
    ekf.update_feet(p_feet_w - r_true, None, np.ones(2), R_eye)
    err[k] = np.abs(ekf.v - v_true).max()
  e_end, e_mid = err[-500:].max(), err[n // 4 : n // 2].max()
  print(f"  synthetic constant-velocity recovery: converged max|v_hat - v_true| = "
        f"{e_end:.2e} m/s  (mid-run {e_mid:.2e}, ratio {e_mid / max(e_end, 1e-12):.1f}x)")
  fails += e_end > 1e-3          # converged accuracy
  # Not plateaued on a bias. Anchored to the FIRST window, which always contains the cold
  # start, rather than to the mid window -- better tuning converges sooner, and a mid-vs-end
  # ratio then trips on a filter that is behaving better, not worse (seen 2026-08-12 when Q
  # was set from the measured Allan densities).
  fails += not (err[:100].max() > 10 * e_end)

  # (2b) how long is the cold-start transient, and does warm-starting remove it? This sets
  #      the minimum usable run length: any recording shorter than the settling time scores
  #      the transient rather than the estimator.
  def settle(warm: bool, thresh: float = 5e-3) -> float:
    e = BaseEkf()
    e.reset()
    if warm:
      e.v = v_true.copy()
      e.P[e.i_v : e.i_v + 3, e.i_v : e.i_v + 3] = np.eye(3) * 0.25**2
    for k in range(n):
      e.predict(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt, np.ones(2))
      e.update_feet(p_feet_w - v_true * (k * dt), None, np.ones(2), R_eye)
      if np.abs(e.v - v_true).max() < thresh:
        return k * dt
    return float("inf")
  t_cold, t_warm = settle(False), settle(True)
  print(f"  time to settle below 5 mm/s: cold start {t_cold:.2f} s -> warm start "
        f"{t_warm:.2f} s   (minimum usable run length)")
  fails += not (t_warm <= t_cold)

  # (3) gravity/frame sanity: stationary, level, both feet planted -> v stays 0
  ekf = BaseEkf()
  ekf.reset()
  for k in range(200):
    ekf.predict(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt, np.ones(2))
    ekf.update_feet(p_feet_w, None, np.ones(2), R_eye)
  e_s = np.abs(ekf.v).max()
  print(f"  stationary drift over 0.4 s: max|v| = {e_s:.2e} m/s")
  fails += e_s > 1e-6

  # (4) R built from the FK Jacobian: symmetry, positive-definiteness, and the two
  #     couplings that a per-foot diagonal R would silently drop.
  q12 = rng.normal(size=(1, 12)) * 0.3
  psi = np.array([0.35])
  Jv, Jw = foot_jacobians_imu(q12, psi, offset=CONTACT_A)
  for use_ori in (False, True):
    cfg = MeasCfg()
    R = build_R(Jv[0], Jw[0], np.ones(2), cfg, use_ori)
    per = 6 if use_ori else 3
    sym = np.abs(R - R.T).max()
    eig = np.linalg.eigvalsh(R).min()
    cross = np.abs(R[0:3, per : per + 3]).max()            # L_pos vs R_pos
    R0 = build_R(Jv[0], Jw[0], np.ones(2), MeasCfg(sigma_psi=0.0), use_ori)
    cross0 = np.abs(R0[0:3, per : per + 3]).max()
    # Scaling check rather than a magnitude threshold: the cross-foot block comes ONLY from
    # the shared waist column, so it must grow as sigma_psi^2 exactly. An absolute threshold
    # here would silently encode whatever sigma_psi happened to be provisional at the time --
    # and it did: the measured 2.61e-5 dropped this block ~1500x below the old 1e-3 guess.
    R2 = build_R(Jv[0], Jw[0], np.ones(2), MeasCfg(sigma_psi=2 * MeasCfg().sigma_psi), use_ori)
    ratio = np.abs(R2[0:3, per : per + 3]).max() / max(cross, 1e-300)
    print(f"  R(use_ori={use_ori}): sym {sym:.1e}  min eig {eig:.2e}  "
          f"cross-foot {cross:.2e} (sigma_psi=0 -> {cross0:.1e}, x2 -> {ratio:.2f}x)")
    fails += sym > 1e-12
    fails += eig <= 0
    fails += not (cross > 0.0)       # the shared waist column MUST couple the feet
    fails += cross0 != 0.0           # ...and must vanish exactly when the waist is noiseless
    fails += abs(ratio - 4.0) > 1e-6  # ...and scale as sigma_psi^2
    # a foot losing contact must inflate its own block, not its neighbour's
    Rin = build_R(Jv[0], Jw[0], np.array([1.0, 0.05]), cfg, use_ori)
    grew_self = Rin[per, per] > R[per, per] * 5
    grew_other = Rin[0, 0] > R[0, 0] * 1.001
    print(f"    contact inflation: own block grew={grew_self} other grew={grew_other}")
    fails += not grew_self
    fails += grew_other

  print("SELFTEST:", "PASS" if fails == 0 else f"FAIL ({fails})")
  return int(fails > 0)


# ---------------------------------------------------------------------------------------
def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("session", nargs="?", type=Path, help="deploy flight-recorder <base>.csv")
  ap.add_argument("--selftest", action="store_true", help="run the built-in checks and exit")
  ap.add_argument("--arms", default="A,B,C,D,E,F")
  ap.add_argument("--entry", type=int, default=None,
                  help="which FSM entry to replay (default: the last one in the file)")
  args = ap.parse_args()

  if args.selftest:
    return selftest()
  if args.session is None:
    ap.error("a session csv is required unless --selftest")

  s = load_session(args.session, entry=args.entry)
  print(s.summary())
  if not s.has_est:
    print("\n  !! No estimator block in this session: no gyro, joints from the 50 Hz cache.")
    print("     DRY RUN ONLY -- arms must not be scored from this (design doc §10 P0).\n")

  pre = prepare(s)
  arms = {}
  want = set(a.strip().upper() for a in args.arms.split(","))
  if "A" in want:
    arms["A legodom"] = arm_a_legodom(s, pre)
  if "E" in want:
    arms["E jacobian"] = arm_e_jacobian(s, pre)
  if "B" in want:
    arms["B compl"] = arm_b_complementary(s, pre, arms.get("A legodom", arm_a_legodom(s, pre)))
  if want & {"C", "D", "F"}:
    pre_c = prepare(s, offset=CONTACT_A)   # the EKF uses the SOLE-PLANE contact frame
    va = arms.get("A legodom")
    if va is None:
      va = arm_a_legodom(s, pre)
    ok = np.isfinite(va).all(axis=1)
    v0 = va[ok][0] if ok.any() else None   # warm start, see run_ekf
    if "C" in want:
      arms["C ekf"] = run_ekf(s, pre_c, use_ori=False, use_att=False, v0_pelvis=v0)
      arms["C+grav"] = run_ekf(s, pre_c, use_ori=False, use_att=False, v0_pelvis=v0,
                               use_grav=True)
    if "D" in want:
      arms["D ekf+rot"] = run_ekf(s, pre_c, use_ori=True, use_att=False, v0_pelvis=v0)
    if "F" in want:
      arms["F ekf+att"] = run_ekf(s, pre_c, use_ori=False, use_att=True, v0_pelvis=v0)

  for label, seg in s.regimes():
    span = s.t[seg][-1] - s.t[seg][0]
    print(f"\n-- {label}: {int(seg.sum())} samples, {span:.1f} s")
    print(f"{'arm':14s} {'mean vx':>9s} {'mean vy':>9s} {'std vx':>9s} {'std vy':>9s} "
          f"{'drift x':>9s} {'drift y':>9s} {'disp X':>9s} {'disp Y':>9s} {'|disp|':>9s}")
    for name, v in arms.items():
      ok = seg & ~np.isnan(v).any(axis=1)
      if ok.sum() < 2:
        print(f"{name:14s} {'-- unavailable --':>40s}")
        continue
      vv, tt = v[ok], s.t[ok]
      dx = np.trapezoid(vv[:, 0], tt), np.trapezoid(vv[:, 1], tt)
      w = to_world(v, s, pre)[ok]            # world displacement = the tape-measure metric
      pw = np.trapezoid(w[:, 0], tt), np.trapezoid(w[:, 1], tt)
      print(f"{name:14s} {vv[:,0].mean():9.4f} {vv[:,1].mean():9.4f} "
            f"{vv[:,0].std():9.4f} {vv[:,1].std():9.4f} {dx[0]:9.3f} {dx[1]:9.3f} "
            f"{pw[0]:9.3f} {pw[1]:9.3f} {np.hypot(*pw):9.3f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
