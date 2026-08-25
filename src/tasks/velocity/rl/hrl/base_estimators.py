"""Simulated base-velocity estimator bank — arms B-F as training-time HL inputs (2026-08-24).

Mirrors ``deploy/robots/h1_2/include/hrl/base_estimators.h`` in batched torch, the same
relationship ``leg_odom.py`` (arm A) has to ``base_state.h``: **the C++ is the
specification.** Unlike arm A, this is NOT bit-parity tested against the deployed
implementation (explicit call, 2026-08-24: training only needs a representative error
distribution, not the 1e-4 golden-fixture bar ``tests/test_estimator_parity.py`` holds arm A
to). ``tests/test_base_estimators_sanity.py`` instead checks this against
``scripts/replay_base_estimators.py`` — the numpy reference this WAS ported from, already
1e-4-verified against the C++ — on the real 400-sample DDS fixture, for the parts that can
be: the Jacobians, ``contact_alpha``, ``build_R`` and the ``BaseEkf`` recursion, fed the
SAME logged acc/gyro so the filter math is checked independent of the synthetic sensors
below.

  B  complementary filter          ComplementaryFilter
  C  EKF, position-only            BaseEkf(use_ori=False)
  C+ C + gravity tilt update       BaseEkf(use_ori=False), use_grav
  D  C + flat-foot orientation     BaseEkf(use_ori=True)
  E  Jacobian leg odometry         jacobian_velocity()
  F  C + vendor attitude           BaseEkf(use_ori=False), use_att

(Arm A, ``legodom``, stays in ``leg_odom.py`` untouched — same reasoning the C++ comment
gives: the validated keeper path must not be disturbed by work on the new arms.)

SYNTHETIC IMU — the one piece with no precedent anywhere in this codebase. Training has no
accelerometer: there is no ``<sensor>`` block in ``h1_2.xml`` and no linear-acceleration
property on ``EntityData`` (only per-joint ``qacc`` for revolute joints, ``data.py:381``).
``root_link_ang_vel_b`` (the gyro-equivalent) and ``root_link_quat_w`` (attitude) ARE already
standard, so specific force is the only new derivation, built from three verified pieces:

  1. MuJoCo's documented free-joint convention (linear qvel/qacc in WORLD frame, angular in
     BODY frame — the same split ``leg_odometry.__call__`` already relies on for ``qvel``).
  2. Standard rigid-body point-acceleration kinematics:
     ``a_point_body = R^T a_origin_world + alpha_body x r_body + w_body x (w_body x r_body)``.
  3. ``R_torso_W = R_pelvis_W @ Rz(psi)`` — ALGEBRAICALLY CROSS-CHECKED, not assumed: it
     reproduces ``leg_odom.py``'s own direct ``grav_P`` computation exactly when composed with
     ``prepare()``'s ``Rz(psi) @ (C @ g_hat)`` formula (the two routes agree termwise), so the
     pelvis/torso rotation relation this whole module leans on is verified independent of any
     new test being added here. The joint's XML name is ``torso_joint`` (checked against
     ``h1_2.xml``, not "waist_yaw_joint" as its physical role would suggest).

APPROXIMATION, stated once and explicitly (do not silently refine this without noting it
here): the lever-arm correction treats pelvis+torso as ONE rigid body, i.e. it uses the
PELVIS's own w/alpha at the IMU offset and ignores the waist joint's own (small) angular
velocity/acceleration contribution to the true IMU motion. Given fp32/approximate fidelity
was the explicit brief for this port, this was judged not worth the second-joint term; revisit
if a policy trained on an EKF/compl arm shows accelerometer-attributable defects under fast
waist yaw.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

from src.tasks.velocity.rl.hrl.leg_odom import (
  CONTACT_A,
  FOOT_SITE_A,
  LEG_JOINT_NAMES,
  LegOdometry,
  _rot,
  foot_sites_b,
  leg_fk,
)

# Leg encoders (12, sdk order) + the waist yaw joint (13th, XML name ``torso_joint``) — the
# same 13-dof layout ``Session.q``/``est_q0..12`` uses on the deploy side.
LEG_WAIST_JOINT_NAMES: tuple[str, ...] = LEG_JOINT_NAMES + ("torso_joint",)

# IMU site in the torso frame (doc/hrl/h1_2_kinematics.md; base_estimators.h kRimu).
R_IMU: tuple[float, float, float] = (-0.04452, -0.01891, 0.27756)
GRAVITY_W: tuple[float, float, float] = (0.0, 0.0, -9.81)

# Real accelerometer/gyro saturation ranges (~20g / ~2000 deg/s, generous vs typical MEMS
# full-scale limits). synthetic_imu() clamps to these -- not because the robot ever legitimately
# gets here, but because an njmax constraint-buffer overflow (h1_2/env_cfgs.py njmax=300, MuJoCo
# silently drops excess contact rows under early random-policy chaos) can leave one env's qacc
# huge-but-FINITE for a tick. That passes every isfinite() guard in BaseEkf/ComplementaryFilter
# (2026-08-24 fix) untouched and lands in obs["hl_vel"] as a legitimate-looking outlier --
# 2026-08-25, ekf_rot local run, same "RuntimeError: normal expects std >= 0.0" crash from a
# different cause (physics-magnitude corruption, not filter divergence). Saturating at the
# source is the one place that protects every downstream consumer (compl AND all four ekf* arms)
# at once.
ACC_SATURATION: float = 200.0   # m/s^2
GYRO_SATURATION: float = 35.0   # rad/s

# 2026-08-25: acc/gyro saturation alone did not stop a repeat "std >= 0.0" crash (ekf_rot,
# same run started AFTER that fix landed). Root cause: BaseEkf.update_feet's inputs
# (p_I/Jv/Jw, from joint positions q12, not qacc) have their own, separate unclamped path --
# an njmax overflow can drop a joint-limit constraint row too, letting q12 swing extreme for
# a tick. Its own Mahalanobis gate (`mahal <= gate*m`) doesn't save it: if the corrupted
# measurement inflates S along with e, the normalized distance can stay "small" while dx
# itself is huge. A raw magnitude bound on the injected velocity correction is a physical
# sanity check that no covariance-relative test can be fooled around -- applied both where
# update_feet injects dx and in predict()'s existing finite-guard (so it also catches
# anything update_gravity/update_attitude leave behind, the next tick).
EKF_V_MAX: float = 20.0   # m/s -- no H1-2 base velocity estimate is legitimately this large

ARM_NAMES: tuple[str, ...] = ("compl", "ekf", "ekf_grav", "ekf_rot", "jacobian", "ekf_att")
"""The six arms this module adds. ``legodom``/``state`` are handled elsewhere (leg_odom.py /
ground truth) — see ``hl_vel_source`` in rl_cfg.py for the full 8-value vocabulary."""


def _vec(v: tuple[float, float, float], like: torch.Tensor) -> torch.Tensor:
  return torch.tensor(v, device=like.device, dtype=like.dtype)


def _eye(n: int, N: int, like: torch.Tensor) -> torch.Tensor:
  return torch.eye(n, device=like.device, dtype=like.dtype).expand(N, n, n).clone()


# ------------------------------------------------------------------------------- SO(3) ----

def skew(v: torch.Tensor) -> torch.Tensor:
  """[N, 3] -> [N, 3, 3]."""
  z = torch.zeros_like(v[:, 0])
  return torch.stack([
    torch.stack([z, -v[:, 2], v[:, 1]], -1),
    torch.stack([v[:, 2], z, -v[:, 0]], -1),
    torch.stack([-v[:, 1], v[:, 0], z], -1),
  ], -2)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
  """Rodrigues, batched. [N, 3] -> [N, 3, 3]. Series branch at th->0, mirroring the C++/numpy
  reference exactly (base_estimators.h so3_exp / replay_base_estimators.py so3_exp)."""
  th = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
  K = skew(w)
  I = _eye(3, w.shape[0], w)
  KK = K @ K
  small = (th.squeeze(-1) < 1e-8)
  reg = I + (torch.sin(th) / th).unsqueeze(-1) * K + ((1.0 - torch.cos(th)) / th**2).unsqueeze(-1) * KK
  ser = I + K + 0.5 * KK
  return torch.where(small.view(-1, 1, 1), ser, reg)


def so3_log(R: torch.Tensor) -> torch.Tensor:
  """[N, 3, 3] -> [N, 3]. Series branch at th->0."""
  tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
  c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
  th = torch.acos(c)
  u = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
  small = th < 1e-8
  ser = u * 0.5
  reg = (th / (2.0 * torch.sin(th).clamp(min=1e-12))).unsqueeze(-1) * u
  return torch.where(small.unsqueeze(-1), ser, reg)


def rz(psi: torch.Tensor) -> torch.Tensor:
  """[N] -> [N, 3, 3]. Identical to leg_odom.py's ``_rot(2, psi)``; kept as a named alias
  since every caller here reads it as "rz", matching the C++/numpy reference's own naming."""
  return _rot(2, psi)


# --------------------------------------------------------------------- kinematics/frames ----

def leg_jacobian_pelvis(q12: torch.Tensor, leg: int, offset: tuple[float, float, float]
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Geometric Jacobian of one foot point w.r.t. that leg's 6 joints, PELVIS frame.

  Transcription of ``scripts/replay_base_estimators.py``'s ``leg_jacobian_pelvis`` (itself
  1e-4-verified against the C++), batched over envs instead of samples. Returns
  ``(J_v [N, 3, 6], J_w [N, 3, 6], p_foot [N, 3])``. Columns are ``a_i x (p_foot - o_i)`` and
  ``a_i``; link table and joint order follow ``leg_fk`` (leg_odom.py).
  """
  N = q12.shape[0]
  sy = 1.0 if leg == 0 else -1.0
  j = q12[:, 6 * leg : 6 * leg + 6]
  axes = [None] * 6
  origins = [None] * 6
  R = _rot(2, j[:, 0])
  o = _vec((0.0, sy * 0.0875, -0.1632), q12).expand(N, 3)
  axes[0] = _vec((0.0, 0.0, 1.0), q12).expand(N, 3)
  origins[0] = o
  p = o + torch.einsum("nij,j->ni", R, _vec((0.0, sy * 0.0755, 0.0), q12))
  origins[1] = p
  axes[1] = torch.einsum("nij,j->ni", R, _vec((0.0, 1.0, 0.0), q12))
  R = R @ _rot(1, j[:, 1])
  origins[2] = p
  axes[2] = torch.einsum("nij,j->ni", R, _vec((1.0, 0.0, 0.0), q12))
  R = R @ _rot(0, j[:, 2])
  p = p + torch.einsum("nij,j->ni", R, _vec((0.0, 0.0, -0.4), q12))
  origins[3] = p
  axes[3] = torch.einsum("nij,j->ni", R, _vec((0.0, 1.0, 0.0), q12))
  R = R @ _rot(1, j[:, 3])
  p = p + torch.einsum("nij,j->ni", R, _vec((0.0, 0.0, -0.4), q12))
  origins[4] = p
  axes[4] = torch.einsum("nij,j->ni", R, _vec((0.0, 1.0, 0.0), q12))
  R = R @ _rot(1, j[:, 4])
  p = p + torch.einsum("nij,j->ni", R, _vec((0.0, 0.0, -0.02), q12))
  origins[5] = p
  axes[5] = torch.einsum("nij,j->ni", R, _vec((1.0, 0.0, 0.0), q12))
  R = R @ _rot(0, j[:, 5])
  pf = p + torch.einsum("nij,j->ni", R, _vec(offset, q12))
  axes_t = torch.stack(axes, dim=1)        # [N, 6, 3]
  origins_t = torch.stack(origins, dim=1)  # [N, 6, 3]
  Jv = torch.cross(axes_t, pf.unsqueeze(1) - origins_t, dim=-1).transpose(1, 2)  # [N, 3, 6]
  Jw = axes_t.transpose(1, 2)  # [N, 3, 6]
  return Jv, Jw, pf


def foot_jacobians_imu(q12: torch.Tensor, psi: torch.Tensor,
                       offset: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
  """Both feet's contact-frame Jacobians in the IMU/TORSO frame, 3x7 each. Column 6 is the
  waist DoF. Returns ``(J_v [N, 2, 3, 7], J_w [N, 2, 3, 7])``."""
  N = q12.shape[0]
  Rzn = rz(-psi)
  zhat = _vec((0.0, 0.0, 1.0), q12)
  r_imu = _vec(R_IMU, q12)
  Jv = torch.zeros(N, 2, 3, 7, device=q12.device, dtype=q12.dtype)
  Jw = torch.zeros(N, 2, 3, 7, device=q12.device, dtype=q12.dtype)
  for leg in (0, 1):
    Jv_P, Jw_P, pf_P = leg_jacobian_pelvis(q12, leg, offset)
    Jv[:, leg, :, :6] = Rzn @ Jv_P
    Jw[:, leg, :, :6] = Rzn @ Jw_P
    p_I = torch.einsum("nij,nj->ni", Rzn, pf_P) - r_imu
    Jv[:, leg, :, 6] = -torch.cross(zhat.expand(N, 3), p_I + r_imu, dim=-1)
    Jw[:, leg, :, 6] = -zhat.expand(N, 3)
  return Jv, Jw


def feet_in_imu(q12: torch.Tensor, psi: torch.Tensor, offset: tuple[float, float, float]
               ) -> tuple[torch.Tensor, torch.Tensor]:
  """Both feet's pose in the IMU/torso frame: ``p^I = Rz(-psi) p^P - r_imu``. Returns
  ``(p_I [N, 2, 3], R_I [N, 2, 3, 3])``."""
  N = q12.shape[0]
  Rzn = rz(-psi)
  r_imu = _vec(R_IMU, q12)
  off = _vec(offset, q12)
  p_I = torch.zeros(N, 2, 3, device=q12.device, dtype=q12.dtype)
  R_I = torch.zeros(N, 2, 3, 3, device=q12.device, dtype=q12.dtype)
  for leg in (0, 1):
    p_P, R_P = leg_fk(q12, leg)
    pf_P = p_P + torch.einsum("nij,j->ni", R_P, off)
    p_I[:, leg] = torch.einsum("nij,nj->ni", Rzn, pf_P) - r_imu
    R_I[:, leg] = Rzn @ R_P
  return p_I, R_I


def contact_alpha(p_I: torch.Tensor, g_I: torch.Tensor, hyst: float = 0.04) -> torch.Tensor:
  """Continuous contact confidence from foot height alone (FURTHER ALONG PROJECTED GRAVITY,
  never body z — see base_estimators.h's note on the 2026-08-12 bug this avoids). ``p_I``
  [N, 2, 3], ``g_I`` [N, 3] -> alpha [N, 2], 1 = the lower foot."""
  along = torch.einsum("nkj,nj->nk", p_I, g_I)
  d = along.max(dim=1, keepdim=True).values - along
  return (1.0 - d / hyst).clamp(0.0, 1.0)


class MeasCfg:
  """Measurement-noise constants for ``build_R``. MEASURED 2026-08-12, 655 s passive hang
  (see ``replay_base_estimators.py``'s ``MeasCfg`` docstring for the full derivation and the
  "these dominate R, the Jacobian structure barely moves the answer numerically" caveat)."""

  sigma_q: float = 1.64e-5
  sigma_psi: float = 2.61e-5
  sigma_slip: float = 3.0e-3
  sigma_model: float = 3.0e-3
  sigma_rot: float = 2.0e-2


def build_R(Jv: torch.Tensor, Jw: torch.Tensor, alpha: torch.Tensor, cfg: MeasCfg,
           use_ori: bool) -> torch.Tensor:
  """[N, 2, 3, 7] x2, [N, 2] -> R [N, m, m], m = 2*(6 if use_ori else 3). Row order
  [L_pos, L_rot, R_pos, R_rot] (rot rows only when use_ori). See build_R's docstring in
  ``replay_base_estimators.py`` for why this is built from the kinematics (cross-foot
  correlation via the shared waist column) rather than a tuned per-foot diagonal."""
  N = Jv.shape[0]
  per = 6 if use_ori else 3
  m = 2 * per
  G = torch.zeros(N, m, 13, device=Jv.device, dtype=Jv.dtype)
  for leg in (0, 1):
    r0 = leg * per
    G[:, r0 : r0 + 3, 6 * leg : 6 * leg + 6] = Jv[:, leg, :, :6]
    G[:, r0 : r0 + 3, 12] = Jv[:, leg, :, 6]
    if use_ori:
      G[:, r0 + 3 : r0 + 6, 6 * leg : 6 * leg + 6] = Jw[:, leg, :, :6]
      G[:, r0 + 3 : r0 + 6, 12] = Jw[:, leg, :, 6]
  Sq = torch.diag(torch.cat([
    torch.full((12,), cfg.sigma_q**2, device=Jv.device, dtype=Jv.dtype),
    torch.full((1,), cfg.sigma_psi**2, device=Jv.device, dtype=Jv.dtype),
  ]))
  R = G @ Sq @ G.transpose(1, 2)
  I3 = _eye(3, N, Jv)
  for leg in (0, 1):
    r0 = leg * per
    a = alpha[:, leg].clamp(min=1e-3)
    infl = (1.0 / a - 1.0) ** 2 * cfg.sigma_slip**2
    R[:, r0 : r0 + 3, r0 : r0 + 3] = R[:, r0 : r0 + 3, r0 : r0 + 3] + \
      I3 * (cfg.sigma_slip**2 + cfg.sigma_model**2 + infl).view(-1, 1, 1)
    if use_ori:
      R[:, r0 + 3 : r0 + 6, r0 + 3 : r0 + 6] = R[:, r0 + 3 : r0 + 6, r0 + 3 : r0 + 6] + \
        I3 * (cfg.sigma_rot**2 + infl).view(-1, 1, 1)
  return R


def jacobian_velocity(q12: torch.Tensor, dq12: torch.Tensor, stance: torch.Tensor,
                      w_P: torch.Tensor) -> torch.Tensor:
  """Arm E: v = -(J qdot) - w_P x p_stance, PELVIS frame. ``stance`` [N] in {0, 1}."""
  N = q12.shape[0]
  v = torch.zeros(N, 3, device=q12.device, dtype=q12.dtype)
  for leg in (0, 1):
    m = stance == leg
    if not m.any():
      continue
    Jv, _, pf = leg_jacobian_pelvis(q12[m], leg, FOOT_SITE_A)
    qd = dq12[m, 6 * leg : 6 * leg + 6]
    v[m] = -torch.einsum("nij,nj->ni", Jv, qd) - torch.cross(w_P[m], pf, dim=-1)
  return v


def stance_index(p_P: torch.Tensor, grav_P: torch.Tensor) -> torch.Tensor:
  """Stance = the foot further along projected gravity. ``p_P`` [N, 2, 3] -> [N] in {0, 1}."""
  along = torch.einsum("nkj,nj->nk", p_P, grav_P)
  return torch.where(along[:, 0] >= along[:, 1], 0, 1)


# ---------------------------------------------------------------------------- BaseEkf ----

class BaseEkf:
  """Rotella-style error-state EKF, body frame = TORSO/IMU, batched over ``num_envs``.

  State x = [r, v, R_WB, p_L, p_R, R_L, R_R, b_f, b_w]; error dx in R^27 (R^21 when
  ``use_ori=False``). Transcription of ``replay_base_estimators.py``'s ``BaseEkf``
  (itself the reference the C++ ``BaseEkf<UseOri>`` was ported to), with every per-sample
  Python-loop operation replaced by a batched torch one and the ragged
  "how many feet contributed this tick" list replaced by the SAME fixed-size neutralisation
  the C++ uses (comment on ``update_feet``, matching ``base_estimators.h``): an inactive
  foot's row gets H=0, e=0, and its R block set to identity, so K's contribution there is
  provably zero rather than the system changing size per env.
  """

  def __init__(self, num_envs: int, device: str, use_orientation: bool,
              dtype: torch.dtype = torch.float32):
    self.device, self.dtype = device, dtype
    self.use_ori = use_orientation
    self.i_r, self.i_v, self.i_phi = 0, 3, 6
    self.i_p = 9
    self.i_th = 15 if use_orientation else None
    self.i_bf = 21 if use_orientation else 15
    self.i_bw = self.i_bf + 3
    self.n = self.i_bw + 3
    self.s = dict(acc=0.0145, gyro=0.0016, ba=1e-4, bw=1e-5, foot_p=1e-3, foot_th=1e-2, att=5e-2)
    N = num_envs
    self.r = torch.zeros(N, 3, device=device, dtype=dtype)
    self.v = torch.zeros(N, 3, device=device, dtype=dtype)
    self.R = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3).clone()
    self.p = torch.zeros(N, 2, 3, device=device, dtype=dtype)
    self.Rf = torch.eye(3, device=device, dtype=dtype).expand(N, 2, 3, 3).clone()
    self.bf = torch.zeros(N, 3, device=device, dtype=dtype)
    self.bw = torch.zeros(N, 3, device=device, dtype=dtype)
    self.P = torch.eye(self.n, device=device, dtype=dtype).expand(N, self.n, self.n).clone() * 1e-2
    self.P[:, self.i_r : self.i_r + 3, self.i_r : self.i_r + 3] *= 1e2
    self.anchored = torch.zeros(N, 2, dtype=torch.bool, device=device)

  def reset(self, env_ids: torch.Tensor, R0: torch.Tensor) -> None:
    """Re-seed the given envs from GROUND-TRUTH attitude (an advantage sim has over the
    offline replay, which only had a noisy vendor quaternion). Velocity resets to 0 --
    training episodes run far longer than the offline clips that needed the arm-A warm
    start to be scored fairly, so a several-second cold-start transient is a small fraction
    of an episode and the warm start is deliberately NOT reproduced here."""
    if env_ids is None or len(env_ids) == 0:
      return
    self.r[env_ids] = 0.0
    self.v[env_ids] = 0.0
    self.R[env_ids] = R0[env_ids]
    self.p[env_ids] = 0.0
    self.Rf[env_ids] = torch.eye(3, device=self.device, dtype=self.dtype)
    self.bf[env_ids] = 0.0
    self.bw[env_ids] = 0.0
    P0 = torch.eye(self.n, device=self.device, dtype=self.dtype) * 1e-2
    P0[self.i_r : self.i_r + 3, self.i_r : self.i_r + 3] *= 1e2
    self.P[env_ids] = P0
    self.anchored[env_ids] = False

  def h_pos(self, i: int) -> torch.Tensor:
    return torch.einsum("nji,nj->ni", self.R, self.p[:, i] - self.r)

  def h_rot(self, i: int) -> torch.Tensor:
    return self.R.transpose(1, 2) @ self.Rf[:, i]

  def H_pos(self, i: int) -> torch.Tensor:
    N = self.r.shape[0]
    H = torch.zeros(N, 3, self.n, device=self.device, dtype=self.dtype)
    C = self.R.transpose(1, 2)
    H[:, :, self.i_r : self.i_r + 3] = -C
    H[:, :, self.i_phi : self.i_phi + 3] = skew(self.h_pos(i))
    H[:, :, self.i_p + 3 * i : self.i_p + 3 * i + 3] = C
    return H

  def H_rot(self, i: int) -> torch.Tensor:
    N = self.r.shape[0]
    H = torch.zeros(N, 3, self.n, device=self.device, dtype=self.dtype)
    H[:, :, self.i_phi : self.i_phi + 3] = -self.h_rot(i).transpose(1, 2)
    H[:, :, self.i_th + 3 * i : self.i_th + 3 * i + 3] = torch.eye(3, device=self.device, dtype=self.dtype)
    return H

  def predict(self, acc: torch.Tensor, gyro: torch.Tensor, dt: float, alpha: torch.Tensor) -> None:
    N = self.r.shape[0]
    r0, v0, R0, P0 = self.r, self.v, self.R, self.P  # pre-tick, for the finite-guard below
    f = acc - self.bf
    w = gyro - self.bw
    a_w = torch.einsum("nij,nj->ni", self.R, f) + _vec(GRAVITY_W, acc)
    self.r = self.r + self.v * dt + 0.5 * a_w * dt * dt
    self.v = self.v + a_w * dt
    self.R = self.R @ so3_exp(w * dt)

    I3 = _eye(3, N, acc)
    F = _eye(self.n, N, acc)
    F[:, self.i_r : self.i_r + 3, self.i_v : self.i_v + 3] = I3 * dt
    F[:, self.i_v : self.i_v + 3, self.i_phi : self.i_phi + 3] = -self.R @ skew(f) * dt
    F[:, self.i_v : self.i_v + 3, self.i_bf : self.i_bf + 3] = -self.R * dt
    F[:, self.i_phi : self.i_phi + 3, self.i_phi : self.i_phi + 3] = so3_exp(-w * dt)
    F[:, self.i_phi : self.i_phi + 3, self.i_bw : self.i_bw + 3] = -I3 * dt

    Q = torch.zeros(N, self.n, self.n, device=self.device, dtype=self.dtype)
    Q[:, self.i_v : self.i_v + 3, self.i_v : self.i_v + 3] = I3 * (self.s["acc"] ** 2 * dt)
    Q[:, self.i_phi : self.i_phi + 3, self.i_phi : self.i_phi + 3] = I3 * (self.s["gyro"] ** 2 * dt)
    Q[:, self.i_bf : self.i_bf + 3, self.i_bf : self.i_bf + 3] = I3 * (self.s["ba"] ** 2 * dt)
    Q[:, self.i_bw : self.i_bw + 3, self.i_bw : self.i_bw + 3] = I3 * (self.s["bw"] ** 2 * dt)
    for i in range(2):
      free = (1.0 - alpha[:, i]) + 1e-6
      Q[:, self.i_p + 3 * i : self.i_p + 3 * i + 3, self.i_p + 3 * i : self.i_p + 3 * i + 3] = (
        I3 * (self.s["foot_p"] ** 2 + free**2).view(-1, 1, 1) * dt)
      if self.use_ori:
        Q[:, self.i_th + 3 * i : self.i_th + 3 * i + 3, self.i_th + 3 * i : self.i_th + 3 * i + 3] = (
          I3 * (self.s["foot_th"] ** 2 + free**2).view(-1, 1, 1) * dt)
    self.P = F @ self.P @ F.transpose(1, 2) + Q

    # A non-finite result -- e.g. a transient NaN in qacc under early random-policy
    # exploration (mjlab ships its own nan_guard.py for exactly this: qpos/qvel/qacc CAN
    # go non-finite mid-training), or any other numerical blow-up in this filter itself --
    # must not silently poison the state forever. Without this, one bad tick's NaN sits in
    # self.R/self.v/self.P and every subsequent tick re-derives from it, so the corruption
    # never heals until an episode reset; hundreds of iterations later it surfaces as
    # "RuntimeError: normal expects std >= 0.0" deep inside PPO's actor.sample(), with no
    # estimator code anywhere in the traceback (2026-08-24, ekf_rot/ekf_grav cluster jobs).
    # Holding the pre-tick state is a no-op whenever the tick was clean (the common case).
    ok = (torch.isfinite(self.r).all(-1) & torch.isfinite(self.v).all(-1)
          & torch.isfinite(self.R).flatten(-2).all(-1) & torch.isfinite(self.P).flatten(-2).all(-1)
          & (self.v.norm(dim=-1) < EKF_V_MAX))
    self.r = torch.where(ok.unsqueeze(-1), self.r, r0)
    self.v = torch.where(ok.unsqueeze(-1), self.v, v0)
    self.R = torch.where(ok.view(-1, 1, 1), self.R, R0)
    self.P = torch.where(ok.view(-1, 1, 1), self.P, P0)

  def update_feet(self, s_p: torch.Tensor, s_R: torch.Tensor | None, alpha: torch.Tensor,
                  R_meas: torch.Tensor, gate: float = 25.0) -> None:
    """``s_p`` [N, 2, 3], ``s_R`` [N, 2, 3, 3] or None, ``alpha`` [N, 2], ``R_meas`` [N, m, m]
    (``build_R``'s output, row order [L_pos, L_rot, R_pos, R_rot]).

    Fixed-size system (batch-friendly, matches the C++): every env always carries the full
    ``m``-row measurement; an inactive foot (not yet anchored, or currently airborne) has its
    rows zeroed in H/e and its R block set to identity, so it changes nothing. First contact
    (alpha active AND not yet anchored) anchors the foot at the current FK estimate instead of
    contributing a KF residual, exactly as the reference does.
    """
    N = self.r.shape[0]
    per = 6 if self.use_ori else 3
    m = 2 * per
    active = alpha > 1e-3
    was_anchored = self.anchored.clone()
    first_contact = active & ~was_anchored
    for i in (0, 1):
      fc = first_contact[:, i]
      if fc.any():
        new_p = self.r + torch.einsum("nij,nj->ni", self.R, s_p[:, i])
        self.p[:, i] = torch.where(fc.unsqueeze(-1), new_p, self.p[:, i])
        if s_R is not None:
          new_Rf = self.R @ s_R[:, i]
          self.Rf[:, i] = torch.where(fc.view(-1, 1, 1), new_Rf, self.Rf[:, i])
    self.anchored = self.anchored | active

    e = torch.zeros(N, m, device=self.device, dtype=self.dtype)
    H = torch.zeros(N, m, self.n, device=self.device, dtype=self.dtype)
    row_active = torch.zeros(N, m, dtype=torch.bool, device=self.device)
    for i in (0, 1):
      row_i = active[:, i] & was_anchored[:, i]
      r0 = i * per
      e[:, r0 : r0 + 3] = torch.where(row_i.unsqueeze(-1), s_p[:, i] - self.h_pos(i),
                                       torch.zeros_like(s_p[:, i]))
      H[:, r0 : r0 + 3, :] = torch.where(row_i.view(-1, 1, 1), self.H_pos(i),
                                          torch.zeros_like(H[:, r0 : r0 + 3, :]))
      row_active[:, r0 : r0 + 3] = row_i.unsqueeze(-1)
      if self.use_ori and s_R is not None:
        e_rot = so3_log(self.h_rot(i).transpose(1, 2) @ s_R[:, i])
        e[:, r0 + 3 : r0 + 6] = torch.where(row_i.unsqueeze(-1), e_rot, torch.zeros_like(e_rot))
        H[:, r0 + 3 : r0 + 6, :] = torch.where(row_i.view(-1, 1, 1), self.H_rot(i),
                                                torch.zeros_like(H[:, r0 + 3 : r0 + 6, :]))
        row_active[:, r0 + 3 : r0 + 6] = row_i.unsqueeze(-1)
    # Neutralise inactive rows in Rm: identity on that row/col so S stays well-conditioned
    # and the (zero) e/H for that row contributes nothing to K.
    inactive = ~row_active
    Im = _eye(m, N, e)
    Rm = torch.where(inactive.unsqueeze(-1) | inactive.unsqueeze(-2), Im, R_meas)

    S = H @ self.P @ H.transpose(1, 2) + Rm
    try:
      Sinv = torch.linalg.inv(S)
    except RuntimeError:
      return
    mahal = torch.einsum("ni,nij,nj->n", e, Sinv, e)
    any_active = row_active.any(dim=-1)
    K = self.P @ H.transpose(1, 2) @ Sinv
    dx = torch.einsum("nij,nj->ni", K, e)
    I_n = _eye(self.n, N, e)
    IKH = I_n - K @ H
    P_new = IKH @ self.P @ IKH.transpose(1, 2) + K @ Rm @ K.transpose(1, 2)
    # mahal is already NaN-safe (any comparison against NaN is False, so a NaN e/Sinv is
    # excluded by the gate below on its own) -- but a merely ILL-conditioned S can still
    # produce a huge-but-finite K/dx that passes the Mahalanobis test (if the corruption that
    # inflated e also inflated S, the normalized distance can stay "small"), or a P_new with a
    # stray NaN from the matrix products above; require both finite AND a physically plausible
    # velocity correction before applying (see EKF_V_MAX's comment).
    dx_v = dx[:, self.i_v : self.i_v + 3]
    finite = (torch.isfinite(dx).all(-1) & torch.isfinite(P_new).flatten(-2).all(-1)
              & (dx_v.norm(dim=-1) < EKF_V_MAX))
    apply = any_active & (mahal <= gate * m) & finite
    if not apply.any():
      return
    dx = torch.where(apply.unsqueeze(-1), dx, torch.zeros_like(dx))
    self._inject(dx)
    self.P = torch.where(apply.view(-1, 1, 1), P_new, self.P)

  def update_gravity(self, acc: torch.Tensor, sigma: float = 0.05, tol: float = 0.5) -> None:
    """Accelerometer as a direct tilt observation (yaw-unobservable by construction). Gated
    per env on ``||acc| - |g|| < tol``. See replay_base_estimators.py's docstring for why
    this exists (the position-only arm's tilt error otherwise ran to 72.8 deg)."""
    N = self.r.shape[0]
    n = acc.norm(dim=-1)
    phys_gate = (n - 9.81).abs() < tol
    if not phys_gate.any():
      return
    meas = acc / n.clamp(min=1e-6).unsqueeze(-1)
    h = torch.einsum("nji,j->ni", self.R, _vec((0.0, 0.0, 1.0), acc))  # R^T @ z_world
    H = torch.zeros(N, 3, self.n, device=self.device, dtype=self.dtype)
    H[:, :, self.i_phi : self.i_phi + 3] = skew(h)
    Rm = _eye(3, N, acc) * sigma**2
    S = H @ self.P @ H.transpose(1, 2) + Rm
    try:
      Sinv = torch.linalg.inv(S)
    except RuntimeError:
      return
    K = self.P @ H.transpose(1, 2) @ Sinv
    dx = torch.einsum("nij,nj->ni", K, meas - h)
    I_n = _eye(self.n, N, acc)
    IKH = I_n - K @ H
    P_new = IKH @ self.P @ IKH.transpose(1, 2) + K @ Rm @ K.transpose(1, 2)
    # See update_feet's matching comment: an ill-conditioned (not necessarily singular) S
    # can pass the physical gate above and still produce a non-finite dx/P_new.
    finite = torch.isfinite(dx).all(-1) & torch.isfinite(P_new).flatten(-2).all(-1)
    gate = phys_gate & finite
    if not gate.any():
      return
    dx = torch.where(gate.unsqueeze(-1), dx, torch.zeros_like(dx))
    self._inject(dx)
    self.P = torch.where(gate.view(-1, 1, 1), P_new, self.P)

  def update_attitude(self, R_meas: torch.Tensor) -> None:
    """Arm F only: the (in sim, ground-truth) attitude as a direct measurement."""
    N = self.r.shape[0]
    e = so3_log(self.R.transpose(1, 2) @ R_meas)
    H = torch.zeros(N, 3, self.n, device=self.device, dtype=self.dtype)
    H[:, :, self.i_phi : self.i_phi + 3] = _eye(3, N, e)
    Rm = _eye(3, N, e) * self.s["att"] ** 2
    S = H @ self.P @ H.transpose(1, 2) + Rm
    try:
      Sinv = torch.linalg.inv(S)
    except RuntimeError:
      return
    K = self.P @ H.transpose(1, 2) @ Sinv
    dx = torch.einsum("nij,nj->ni", K, e)
    I_n = _eye(self.n, N, e)
    IKH = I_n - K @ H
    P_new = IKH @ self.P @ IKH.transpose(1, 2) + K @ Rm @ K.transpose(1, 2)
    # This update had no gate at all before -- it always fires when called (unlike
    # update_feet's contact gate or update_gravity's physical gate), so finiteness is the
    # ONLY thing standing between a bad tick and permanently poisoning self.R/self.P.
    ok = torch.isfinite(dx).all(-1) & torch.isfinite(P_new).flatten(-2).all(-1)
    if not ok.any():
      return
    dx = torch.where(ok.unsqueeze(-1), dx, torch.zeros_like(dx))
    self._inject(dx)
    self.P = torch.where(ok.view(-1, 1, 1), P_new, self.P)

  def _inject(self, dx: torch.Tensor) -> None:
    # Backstop, not the primary guard: every call site above already gates on finiteness
    # before calling this, but a bad dx must never move r/v/R/p/Rf/bf/bw regardless of
    # what any future call site does or forgets to check.
    dx = torch.where(torch.isfinite(dx).all(-1, keepdim=True), dx, torch.zeros_like(dx))
    self.r = self.r + dx[:, self.i_r : self.i_r + 3]
    self.v = self.v + dx[:, self.i_v : self.i_v + 3]
    self.R = self.R @ so3_exp(dx[:, self.i_phi : self.i_phi + 3])
    for i in range(2):
      self.p[:, i] = self.p[:, i] + dx[:, self.i_p + 3 * i : self.i_p + 3 * i + 3]
      if self.use_ori:
        self.Rf[:, i] = self.Rf[:, i] @ so3_exp(dx[:, self.i_th + 3 * i : self.i_th + 3 * i + 3])
    self.bf = self.bf + dx[:, self.i_bf : self.i_bf + 3]
    self.bw = self.bw + dx[:, self.i_bw : self.i_bw + 3]


class ComplementaryFilter:
  """Arm B: high-pass the gravity-free accelerometer, low-pass leg odometry, PELVIS frame.
  One time constant, no covariance, no contact logic -- if the EKF cannot beat this it has
  not earned its 27 states (design doc §9)."""

  def __init__(self, num_envs: int, device: str, tau: float = 0.20,
              dtype: torch.dtype = torch.float32):
    self.tau = tau
    self.v = torch.zeros(num_envs, 3, device=device, dtype=dtype)

  def reset(self, env_ids: torch.Tensor) -> None:
    if env_ids is None or len(env_ids) == 0:
      return
    self.v[env_ids] = 0.0

  def step(self, a_P: torch.Tensor, v_legodom: torch.Tensor, ok: torch.Tensor,
          dt: float) -> torch.Tensor:
    """``a_P`` [N, 3] pelvis-frame gravity-free acceleration, ``v_legodom`` [N, 3] the
    instantaneous (unwindowed) legodom reading this tick, ``ok`` [N] bool (legodom valid)."""
    pred = self.v + a_P * dt
    alpha = dt / (self.tau + dt)
    fused = (1.0 - alpha) * pred + alpha * v_legodom
    v_new = torch.where(ok.unsqueeze(-1), fused, pred)
    # Same reasoning as BaseEkf.predict: self.v is persistent state that feeds next tick's
    # `pred`, so a single non-finite a_P/v_legodom (e.g. a transient NaN from the
    # synthetic-IMU input) would otherwise poison every subsequent tick permanently.
    finite = torch.isfinite(v_new).all(-1)
    self.v = torch.where(finite.unsqueeze(-1), v_new, self.v)
    return self.v


# ------------------------------------------------------------------------ synthetic IMU ----

def synthetic_imu(free_q: torch.Tensor, free_v: torch.Tensor, free_a: torch.Tensor,
                  psi: torch.Tensor, dpsi: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Derive a torso-mounted accelerometer/gyro reading from privileged sim state -- see the
  module docstring for the three verified pieces this is built from and the stated
  pelvis+torso-as-one-rigid-body approximation.

  Args (all straight off the free joint, mid-substep, matching leg_odometry.__call__'s own
  staleness discipline):
    free_q [N, 7]  pos(3) ++ quat_wxyz(4), world frame
    free_v [N, 6]  linear(world) ++ angular(body/pelvis), MuJoCo's native qvel split
    free_a [N, 6]  SAME split, from qacc (d/dt of the qvel generalized coordinates)
    psi    [N]     waist yaw joint angle
    dpsi   [N]     waist yaw joint rate

  Returns (acc_T [N, 3] specific force incl. gravity, gyro_T [N, 3], C [N, 3, 3] world->torso,
  R_torso_W [N, 3, 3]). acc_T/gyro_T are saturated to ACC_SATURATION/GYRO_SATURATION.
  """
  quat = free_q[:, 3:7]
  R_pelvis_W = matrix_from_quat(quat)
  R_torso_W = R_pelvis_W @ rz(psi)
  C = R_torso_W.transpose(1, 2)

  w_pelvis = free_v[:, 3:6]           # body(pelvis)-frame angular velocity, ground truth
  alpha_pelvis = free_a[:, 3:6]       # body(pelvis)-frame angular acceleration
  a0_world = free_a[:, 0:3]           # WORLD-frame linear acceleration of the pelvis origin

  r_imu_pelvis = torch.einsum("nij,j->ni", rz(psi), _vec(R_IMU, quat))
  a0_pelvis = torch.einsum("nji,nj->ni", R_pelvis_W, a0_world)  # R_pelvis_W^T @ a0_world
  a_imu_pelvis = (a0_pelvis
                  + torch.cross(alpha_pelvis, r_imu_pelvis, dim=-1)
                  + torch.cross(w_pelvis, torch.cross(w_pelvis, r_imu_pelvis, dim=-1), dim=-1))
  a_imu_torso = torch.einsum("nij,nj->ni", rz(-psi), a_imu_pelvis)

  g_T = torch.einsum("nij,j->ni", C, _vec((0.0, 0.0, -1.0), quat))
  acc_T = a_imu_torso - g_T * 9.81   # specific force = a - g; g_T*9.81 is g expressed in torso

  # Direct inversion of prepare()'s w_P = Rz(psi) w_T - dpsi*z (verified against that formula
  # algebraically, see the module docstring): w_T = Rz(-psi) (w_P + dpsi*z).
  z = _vec((0.0, 0.0, 1.0), quat)
  gyro_T = torch.einsum("nij,nj->ni", rz(-psi), w_pelvis + dpsi.unsqueeze(-1) * z)

  # Saturate to real-sensor range -- see ACC_SATURATION's comment. nan_to_num first (clamp
  # alone passes NaN through unchanged); a no-op on every physically-plausible tick.
  acc_T = torch.nan_to_num(acc_T, nan=0.0, posinf=ACC_SATURATION, neginf=-ACC_SATURATION)
  acc_T = acc_T.clamp(-ACC_SATURATION, ACC_SATURATION)
  gyro_T = torch.nan_to_num(gyro_T, nan=0.0, posinf=GYRO_SATURATION, neginf=-GYRO_SATURATION)
  gyro_T = gyro_T.clamp(-GYRO_SATURATION, GYRO_SATURATION)

  return acc_T, gyro_T, C, R_torso_W


# --------------------------------------------------------------------------- the bank ----

class EstimatorBank:
  """Lazily-activated per_substep term publishing ``env.estimator_bank``.

  Registered UNCONDITIONALLY (mirrors ``leg_odometry``), but constructs no filter state and
  does no work until ``.activate(arm_name)`` is called -- ``HierarchicalRunner`` does this
  once it knows ``hl_vel_source`` from ``train_cfg``, mirroring the existing
  ``hasattr(env.unwrapped, "leg_odom")`` guard for arm A. This matters for cost, not just
  laziness: unlike arm A's pure kinematics, the four EKF variants are a batched Kalman
  filter (persistent [N, up to 27, 27] covariance, a batched matrix inversion at every
  contact update) -- running all six unconditionally would tax every A1 run, including
  ``hl_vel_source='state'`` baselines that never read this bank's output.
  """

  def __init__(self, cfg, env) -> None:
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(LEG_WAIST_JOINT_NAMES), preserve_order=True)
    self.joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self.num_envs = env.num_envs
    self.device = env.device
    self.active_arm: str | None = None
    self._impl = None  # constructed on activate()
    # Per-env pending-reset mask for the EKF arms: reset() knows WHICH envs reset but not
    # their ground-truth attitude (only available in __call__, from the just-stepped sim
    # state); __call__ knows the attitude but must reset ONLY the envs actually pending, not
    # the whole batch (a global bool here would wipe every other env's filter state on the
    # next tick after any single env reset -- caught before this shipped).
    self._pending_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.estimator_bank = self
    # windowed value the runner reads at HL fires, mirroring LegOdometry's value/fire.
    self.sum = torch.zeros(env.num_envs, 2, device=env.device)
    self.n = torch.zeros(env.num_envs, device=env.device)
    self.value = torch.zeros(env.num_envs, 2, device=env.device)

  def activate(self, arm: str) -> None:
    if arm not in ARM_NAMES:
      raise ValueError(f"EstimatorBank.activate: {arm!r} not in {ARM_NAMES}")
    if self.active_arm is not None and self.active_arm != arm:
      raise ValueError(f"EstimatorBank already activated for {self.active_arm!r}, "
                       f"cannot also activate {arm!r} -- one bank, one arm per run.")
    self.active_arm = arm
    if arm == "compl":
      self._impl = ComplementaryFilter(self.num_envs, self.device)
      self._legodom = LegOdometry(self.num_envs, self.device)
    elif arm == "jacobian":
      self._impl = None
    else:
      use_ori = arm == "ekf_rot"
      self._impl = BaseEkf(self.num_envs, self.device, use_orientation=use_ori)
      self._grav = arm == "ekf_grav"
      self._att = arm == "ekf_att"
    self._pending_reset[:] = True  # every env needs its initial reset

  def __call__(self, env) -> torch.Tensor:
    diag = torch.zeros(env.num_envs, device=env.device)
    if self.active_arm is None:
      return diag
    d = env.scene["robot"].data
    free_q = d.data.qpos[:, d.indexing.free_joint_q_adr]
    free_v = d.data.qvel[:, d.indexing.free_joint_v_adr]
    free_a = d.data.qacc[:, d.indexing.free_joint_v_adr]
    q13 = d.joint_pos[:, self.joint_ids]
    dq13 = d.joint_vel[:, self.joint_ids]
    q12, psi, dpsi = q13[:, :12], q13[:, 12], dq13[:, 12]

    grav_P = quat_apply_inverse(free_q[:, 3:7], _vec((0.0, 0.0, -1.0), free_q).expand(free_q.shape[0], 3))
    w_P = free_v[:, 3:6]

    if self.active_arm == "jacobian":
      p_P = foot_sites_b(q12)
      st = stance_index(p_P, grav_P)
      v = jacobian_velocity(q12, dq13[:, :12], st, w_P)
      self._accumulate(v[:, :2])
      return diag

    acc_T, gyro_T, C, R_torso_W = synthetic_imu(free_q, free_v, free_a, psi, dpsi)

    if self.active_arm == "compl":
      v_lo = self._legodom.step(q12, env.physics_dt, w_P, grav_P)
      ok = self._legodom.valid
      g_hat_T = torch.einsum("nij,j->ni", C, _vec((0.0, 0.0, -1.0), q12))
      a_free_T = acc_T + g_hat_T * 9.81
      a_P = torch.einsum("nij,nj->ni", rz(psi), a_free_T)
      v = self._impl.step(a_P, v_lo, ok, env.physics_dt)
      self._accumulate(v[:, :2])
      return diag

    # EKF arms (ekf / ekf_grav / ekf_rot / ekf_att)
    ekf: BaseEkf = self._impl
    if self._pending_reset.any():
      pending_ids = self._pending_reset.nonzero(as_tuple=True)[0]
      ekf.reset(pending_ids, R_torso_W)
      self._pending_reset[pending_ids] = False
    p_I, R_I = feet_in_imu(q12, psi, CONTACT_A)
    g_I = torch.einsum("nij,j->ni", C, _vec((0.0, 0.0, -1.0), q12))
    alpha = contact_alpha(p_I, g_I)
    ekf.predict(acc_T, gyro_T, env.physics_dt, alpha)
    if self._grav:
      ekf.update_gravity(acc_T)
    if self._att:
      ekf.update_attitude(R_torso_W)
    Jv, Jw = foot_jacobians_imu(q12, psi, CONTACT_A)
    R_meas = build_R(Jv, Jw, alpha, MeasCfg(), ekf.use_ori)
    ekf.update_feet(p_I, R_I if ekf.use_ori else None, alpha, R_meas)
    v_T = torch.einsum("nji,nj->ni", ekf.R, ekf.v)
    r_imu = _vec(R_IMU, q12).expand(q12.shape[0], 3)
    v_P = torch.einsum("nij,nj->ni", rz(psi), v_T - torch.cross(gyro_T - ekf.bw, r_imu, dim=-1))
    self._accumulate(v_P[:, :2])
    return diag

  def _accumulate(self, v_xy: torch.Tensor) -> None:
    # Central backstop: every arm (including the stateless jacobian one, which has no
    # persistent state of its own to guard) funnels through here before its window
    # average can reach obs["hl_vel"]. Summing one non-finite tick would poison the WHOLE
    # window's average (NaN + anything = NaN); skip that tick's contribution instead --
    # `fire()`'s existing `got = self.n > 0` already handles "no samples this window" by
    # holding the last latched value, so this degrades to that same, already-safe path.
    ok = torch.isfinite(v_xy).all(-1)
    self.sum = torch.where(ok.unsqueeze(-1), self.sum + v_xy, self.sum)
    self.n = torch.where(ok, self.n + 1.0, self.n)

  def fire(self) -> torch.Tensor:
    got = self.n > 0
    avg = self.sum / self.n.clamp(min=1).unsqueeze(-1)
    self.value = torch.where(got.unsqueeze(-1), avg, self.value)
    self.sum = torch.zeros_like(self.sum)
    self.n = torch.zeros_like(self.n)
    return self.value

  def reset(self, env_ids) -> None:
    if env_ids is None:
      return
    self.sum[env_ids] = 0.0
    self.n[env_ids] = 0.0
    self.value[env_ids] = 0.0
    if self.active_arm == "compl":
      self._impl.reset(env_ids)
      self._legodom.reset(env_ids)
    elif self.active_arm not in ("jacobian", None):
      # BaseEkf needs the CURRENT ground-truth torso attitude to reseed with; __call__
      # recomputes it every tick anyway, so mark just THESE envs pending rather than
      # resetting now from a possibly-stale env snapshot (or, as a first draft of this got
      # wrong, resetting every env instead of only the ones that actually reset).
      self._pending_reset[env_ids] = True
