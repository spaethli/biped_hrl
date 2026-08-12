"""Simulated leg odometry — the HL's absolute base-velocity estimate, computed the way
the robot computes it (WL-F, 2026-08-09).

Mirrors ``deploy/robots/h1_2/include/hrl/base_state.h`` (``leg_fk`` / ``foot_site_b`` /
``leg_odom_velocity``) in batched torch. **That C++ is the specification** — every constant
and every frame convention here is a transcription, not a re-derivation. Parity against the
real-robot log is the acceptance test (``check_leg_odom_parity`` in the WL-F data dir).

WHY THIS EXISTS. ``HlVelJitter`` models the HL's velocity-estimate error as i.i.d. per-window
Gaussian noise, i.e. as *exogenous*. On hardware it is not: leg odometry is computed **from
the legs**, so its error is a function of the policy's own leg motion, and the loop closes
(2026-08-09, 424 HL fires at zero command: ``corr(|lo_vx|_t, |tgt0|_t) = +0.703``,
``corr(|tgt0|_t, |lo_vx|_t+1) = +0.609``, escalating). Under jitter, thrashing the feet is
free in sim; under this estimator it costs the policy its own velocity read.

FRAMES. Leg FK is pelvis-referenced, so the returned velocity is already in the PELVIS frame
— no rotation on the way out (unlike the torso-mounted IMU path). ``w_P`` is the pelvis
angular velocity *in the pelvis frame*; on the robot the caller owes
``Rz(psi)*w_gyro - psi_dot*z`` to get there from the torso gyro, in sim it is read directly.
``grav_P`` is projected gravity in the pelvis frame (unit vector, +z down-ish), used only to
pick the stance foot as the one further along gravity.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

# Site "left_foot"/"right_foot" pos in the ANKLE_ROLL frame (h1_2.xml:80,121). base_state.h
# kFootSiteA. NOT the sole point ``lowest_foot_z`` uses — see its note there.
FOOT_SITE_A: tuple[float, float, float] = (0.04, 0.0, -0.04)

# EKF contact frame (doc/hrl/h1_2_ekf_design.md §C): the same point dropped 5 mm onto the
# sole plane. x is the measured area centroid of the sole hull (+0.03999, i.e. FOOT_SITE_A's
# x to 0.01 mm) and z is the measured sole plane (-0.045, identical across all five model
# variants). Origin on the sole plane is what makes "a contacting foot does not translate"
# literally true — the ankle_roll origin sits 45 mm above it, so foot roll/pitch translates
# it. Used by the offline estimator replay, NOT by the shipped leg odometry, which keeps
# FOOT_SITE_A so its calibration history stays comparable.
CONTACT_A: tuple[float, float, float] = (0.04, 0.0, -0.045)

# Link offsets, left leg (right mirrors the y sign) — base_state.h ``leg_fk``.
_HIP_YAW = (0.0, 0.0875, -0.1632)
_HIP_PITCH = (0.0, 0.0755, 0.0)
_THIGH = (0.0, 0.0, -0.4)
_SHANK = (0.0, 0.0, -0.4)
_ANKLE = (0.0, 0.0, -0.02)


def _rot(axis: int, a: torch.Tensor) -> torch.Tensor:
  """Batched elementary rotation matrix about ``axis`` (0=x, 1=y, 2=z). ``a`` is [N]."""
  c, s = torch.cos(a), torch.sin(a)
  o, z = torch.ones_like(a), torch.zeros_like(a)
  if axis == 0:
    rows = [(o, z, z), (z, c, -s), (z, s, c)]
  elif axis == 1:
    rows = [(c, z, s), (z, o, z), (-s, z, c)]
  else:
    rows = [(c, -s, z), (s, c, z), (z, z, o)]
  return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def _vec(v: tuple[float, float, float], like: torch.Tensor) -> torch.Tensor:
  return torch.tensor(v, device=like.device, dtype=like.dtype)


def leg_fk(q: torch.Tensor, leg: int) -> tuple[torch.Tensor, torch.Tensor]:
  """One leg's ANKLE_ROLL frame (origin and orientation) in the PELVIS frame.

  ``q`` is [N, 12] in sdk order (0-5 left, 6-11 right; yaw, pitch, roll, knee, ankle_pitch,
  ankle_roll). Returns ``(p [N, 3], R [N, 3, 3])``. Transcription of ``base_state.h``
  ``leg_fk``, and split out for the same reason the C++ splits it: both consumers start
  here and differ only in which POINT of the foot they then evaluate, so the link table
  has exactly one copy.
  """
  sy = 1.0 if leg == 0 else -1.0
  j = q[:, 6 * leg : 6 * leg + 6]
  R = _rot(2, j[:, 0])
  p = _vec((_HIP_YAW[0], sy * _HIP_YAW[1], _HIP_YAW[2]), q).expand(q.shape[0], 3).clone()
  p = p + R @ _vec((_HIP_PITCH[0], sy * _HIP_PITCH[1], _HIP_PITCH[2]), q)
  R = R @ _rot(1, j[:, 1])
  R = R @ _rot(0, j[:, 2])
  p = p + R @ _vec(_THIGH, q)
  R = R @ _rot(1, j[:, 3])
  p = p + R @ _vec(_SHANK, q)
  R = R @ _rot(1, j[:, 4])
  p = p + R @ _vec(_ANKLE, q)
  R = R @ _rot(0, j[:, 5])
  return p, R


def foot_site_b(
  q: torch.Tensor, leg: int, offset: tuple[float, float, float] = FOOT_SITE_A
) -> torch.Tensor:
  """Foot-site position in the PELVIS frame from one leg's 6 encoders. [N, 3].

  ``offset`` is the point evaluated in the ankle_roll frame; the default reproduces the
  shipped estimator exactly (``kFootSiteA``). Pass ``CONTACT_A`` for the EKF contact frame.
  """
  p, R = leg_fk(q, leg)
  return p + R @ _vec(offset, q)


def foot_sites_b(q: torch.Tensor) -> torch.Tensor:
  """Both feet's site positions in the pelvis frame. ``q`` [N, 12] -> [N, 2, 3]."""
  return torch.stack([foot_site_b(q, 0), foot_site_b(q, 1)], dim=1)


def leg_odom_velocity(
  p: torch.Tensor, p_prev: torch.Tensor, dt: float, w_P: torch.Tensor, grav_P: torch.Tensor
) -> torch.Tensor:
  """Instantaneous pelvis-frame base linear velocity from the stance foot.

      v = -d(p_stance)/dt - w_P x p_stance

  exact while the stance foot is fixed in the world. ``p``/``p_prev`` are [N, 2, 3] (both
  feet, now and one tick ago); differencing is PER FOOT, never across the stance switch,
  which would inject the ~0.3 m inter-foot gap as a false velocity spike. Stance = the foot
  further along projected gravity; no contact sensing (the robot's LowState has no foot-force
  field, and a contact oracle was measured to buy ~nothing).
  """
  along = (p * grav_P.unsqueeze(1)).sum(-1)  # [N, 2]
  stance = (along[:, 0] < along[:, 1]).long()  # C++: >= picks foot 0
  idx = stance.view(-1, 1, 1).expand(-1, 1, 3)
  ps = p.gather(1, idx).squeeze(1)
  ps_prev = p_prev.gather(1, idx).squeeze(1)
  return -(ps - ps_prev) / dt - torch.cross(w_P, ps, dim=-1)


class LegOdometry:
  """Per-env leg-odometry accumulator: differences at the *physics* rate and c-averages
  per HL window, exactly as ``State_RLHRL`` does at 1 kHz (``hl_vel_lo_ = lo_sum_/lo_n_``).

  Differencing rate matters and is not a free choice: the deploy bench measured the identical
  reconstruction 2.4x worse when ``d(p_foot)/dt`` was taken at the 50 Hz control rate instead
  of 1 kHz (per-step vx 0.374 -> 0.176, c-averaged 0.147 -> 0.072). Feed this the physics
  step, never the control step.

  Sum-then-average makes the estimate insensitive to the sampling rate *of the difference
  term alone* — it telescopes to ``-(p_end - p_start)/(n*dt)`` between stance switches — so
  the rate penalty lands on the ``w x p`` term and on stance-switch attribution.
  """

  def __init__(self, num_envs: int, device: str) -> None:
    self.device = device
    self.p_prev = torch.zeros(num_envs, 2, 3, device=device)
    self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.sum = torch.zeros(num_envs, 3, device=device)
    self.n = torch.zeros(num_envs, device=device)
    self.value = torch.zeros(num_envs, 2, device=device)  # last c-averaged (vx, vy)

  def step(self, q: torch.Tensor, dt: float, w_P: torch.Tensor, grav_P: torch.Tensor
           ) -> torch.Tensor:
    """Accumulate one physics tick. ``q`` [N, 12] leg encoders in sdk order. Returns the
    instantaneous per-tick estimate, 0 for envs with no previous sample yet."""
    p = foot_sites_b(q)
    m = self.valid.unsqueeze(-1)
    v = torch.where(m, leg_odom_velocity(p, self.p_prev, dt, w_P, grav_P), 0.0)
    self.sum = self.sum + v
    self.n = self.n + self.valid.to(self.n.dtype)
    self.p_prev = p
    self.valid = torch.ones_like(self.valid)
    return v

  def fire(self) -> torch.Tensor:
    """Close the window: latch ``sum/n`` as the HL's (vx, vy) and reset the accumulator.
    Envs with no samples keep their previous value (C++: ``if (lo_n_ > 0)``)."""
    got = self.n > 0
    avg = self.sum[:, :2] / self.n.clamp(min=1).unsqueeze(-1)
    self.value = torch.where(got.unsqueeze(-1), avg, self.value)
    self.sum = torch.zeros_like(self.sum)
    self.n = torch.zeros_like(self.n)
    return self.value

  def reset(self, env_ids) -> None:
    """Drop the cross-episode difference for reset envs. The reset writes default joint
    positions, so differencing across it would inject the whole pose change as a false
    velocity spike; and a fresh episode has no history, so the latched value goes to 0."""
    if env_ids is None:
      return
    self.valid[env_ids] = False
    self.sum[env_ids] = 0.0
    self.n[env_ids] = 0.0
    self.value[env_ids] = 0.0


# Leg joints in SDK order — the order ``leg_fk`` indexes (0-5 left, 6-11 right; yaw, pitch,
# roll, knee, ankle_pitch, ankle_roll). Identical to the deploy `joint_ids_map`, which is
# the identity, and to the XML declaration order.
LEG_JOINT_NAMES: tuple[str, ...] = tuple(
  f"{side}_{j}_joint"
  for side in ("left", "right")
  for j in ("hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
)


class leg_odometry:
  """Metrics term that runs the estimator once per PHYSICS substep and publishes the
  accumulator as ``env.leg_odom`` for the runner to read at HL fires.

  Registered with ``per_substep=True``, which is the only seam mjlab offers inside the
  decimation loop — and the rate is the whole point: the deploy bench measured this same
  reconstruction 2.4x worse when differenced at the 50 Hz control rate. Sim gets 200 Hz
  (``timestep=0.005``), still 5x coarser than the robot's ~991 Hz.

  MID-LOOP STALENESS. Inside the decimation loop only the integrated state (qpos, qvel) is
  current; every derived quantity (xpos, xquat, cvel, site_xpos) is one substep stale. So
  this reads the free joint's qpos/qvel directly rather than ``root_link_quat_w`` /
  ``root_link_ang_vel_b``, which are cvel/xquat-backed. MuJoCo stores a free joint's angular
  velocity in the BODY frame already (``write_root_velocity`` converts world->body before
  writing it), and the root body is the pelvis, so ``qvel[3:6]`` IS ``w_P`` with no rotation.

  The returned metric is the instantaneous estimator error ``|v_lo - v_true|`` (xy), logged
  as ``Episode_Metrics/leg_odom_err``. It is diagnostic only: nothing in the reward, the
  observations or the dynamics reads it, so this term is RQ2-neutral even though A0 has no
  counterpart.
  """

  def __init__(self, cfg, env) -> None:
    robot = env.scene["robot"]
    # preserve_order: the FK indexes q by SDK position, so the resolved ids must come back
    # in LEG_JOINT_NAMES order, not in model order.
    ids, _ = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
    self.leg_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self.odom = LegOdometry(env.num_envs, env.device)
    env.leg_odom = self.odom  # the runner's handle; see hrl_runner's hl_vel_source

  def __call__(self, env) -> torch.Tensor:
    d = env.scene["robot"].data
    free_q = d.data.qpos[:, d.indexing.free_joint_q_adr]  # [N, 7] pos ++ quat (w, x, y, z)
    free_v = d.data.qvel[:, d.indexing.free_joint_v_adr]  # [N, 6] lin (world) ++ ang (body)
    quat = free_q[:, 3:7]
    grav_P = quat_apply_inverse(quat, d.gravity_vec_w.expand(quat.shape[0], 3))
    v_lo = self.odom.step(
      d.joint_pos[:, self.leg_ids], env.physics_dt, free_v[:, 3:6], grav_P
    )
    v_true = quat_apply_inverse(quat, free_v[:, 0:3])
    return (v_lo[:, :2] - v_true[:, :2]).norm(dim=-1)

  def reset(self, env_ids) -> None:
    self.odom.reset(env_ids)
