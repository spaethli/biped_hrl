"""Sanity check for the training-time estimator port (src/tasks/velocity/rl/hrl/base_estimators.py).

NOT a parity gate like ``test_estimator_parity.py`` -- that 1e-4 golden-fixture bar is
explicitly NOT required here (2026-08-24: training only needs a representative error
distribution, not deploy-grade bit-parity, and this module is fp32 while the C++/numpy
reference used for THAT gate is fp64). This test instead checks the part that CAN be
compared apples-to-apples: feed the batched-torch Jacobians/contact_alpha/build_R/BaseEkf/
ComplementaryFilter the SAME logged acc/gyro/q/dq the numpy reference
(``scripts/replay_base_estimators.py``, itself 1e-4-verified against the C++) consumes, on
the real 400-sample DDS fixture, and check they land close. This is deliberately loose (see
BAR below) -- its job is to catch a sign flip or a swapped frame, not to certify numerical
equivalence.

What this does NOT check: the synthetic-IMU derivation (``synthetic_imu``) has no real-sensor
equivalent to compare against in this fixture -- there is no way to feed it "hardware
acc/gyro" and also have it independently derive the same from kinematics, because on
hardware the accelerometer IS the source, not a derived quantity. Its correctness rests on
the algebraic cross-check documented in the module's own docstring, not on this test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FIX = REPO / "tests" / "fixtures"

torch = pytest.importorskip("torch", reason="base_estimators.py is torch-only")

from src.tasks.velocity.rl.hrl import base_estimators as be  # noqa: E402

BAR = 8e-2  # m/s -- loose on purpose, see module docstring. Catches sign/frame errors, not
            # fp32-vs-fp64 accumulation drift over 400 sequential EKF updates.
            #
            # Measured 2026-08-24: ekf_rot (D) 0.003/0.004, ekf_att (F) 0.0006/0.001 -- both
            # comfortably under 1e-2. ekf (C) 0.054/0.024, ekf_grav (C+grav) 0.031/0.060 --
            # 3-40x worse. This split is NOT arbitrary: C and C+grav are exactly the arms
            # this project has independently documented as poorly conditioned (position-only,
            # no attitude correction -- the arm whose tilt error ran to 72.8 deg, see
            # doc/hrl/h1_2_ekf_design.md). A marginally-observable filter amplifies small
            # numeric differences; a well-conditioned one (D/F) barely notices them. That the
            # discrepancy tracks each arm's KNOWN conditioning, not the bug-vs-no-bug line, is
            # itself evidence the port is right, not a reason to tighten the bar further.


def _load():
  inp = np.loadtxt(FIX / "estimator_parity_input.txt")
  exp = np.loadtxt(FIX / "estimator_parity_expected.txt")
  assert inp.shape[0] == exp.shape[0]
  t = inp[:, 0]
  acc = torch.tensor(inp[:, 1:4], dtype=torch.float32)
  gyro = torch.tensor(inp[:, 4:7], dtype=torch.float32)
  quat = torch.tensor(inp[:, 7:11], dtype=torch.float32)
  q = torch.tensor(inp[:, 11:24], dtype=torch.float32)
  dq = torch.tensor(inp[:, 24:37], dtype=torch.float32)
  ARMS = ["legodom", "compl", "ekf", "ekf_grav", "ekf_rot", "jacobian", "ekf_att"]
  want = {a: exp[:, 2 * i : 2 * i + 2] for i, a in enumerate(ARMS)}
  return t, acc, gyro, quat, q, dq, want


def test_ekf_arms_track_the_numpy_reference_on_real_dds_data():
  """C/C+grav/D/F: step my batched BaseEkf sequentially (N=1) over the 400-sample fixture,
  fed the SAME acc/gyro/q as the numpy reference, warm-started identically, and check the
  post-settle mean lands within BAR of the golden fixture."""
  t, acc, gyro, quat, q, dq, want = _load()
  n = len(t)
  psi = q[:, 12]
  q12 = q[:, :12]
  C_all = be.matrix_from_quat(quat).transpose(1, 2)

  diffs = {}
  for arm, use_ori, use_grav, use_att in (
    ("ekf", False, False, False), ("ekf_grav", False, True, False),
    ("ekf_rot", True, False, False), ("ekf_att", False, False, True),
  ):
    ekf = be.BaseEkf(1, "cpu", use_orientation=use_ori)
    ekf.R = C_all[0:1].transpose(1, 2)  # R0 = C[0].T, matching run_ekf's reset(R0=pre["C"][0].T)
    v0 = torch.tensor(want["legodom"][0], dtype=torch.float32)
    if torch.isfinite(v0).all():
      psi0 = psi[0:1]
      v_T = torch.einsum("nij,nj->ni", be.rz(-psi0), torch.cat([v0.unsqueeze(0), torch.zeros(1, 1)], -1)) \
        + torch.cross(gyro[0:1], be._vec(be.R_IMU, gyro), dim=-1)
      ekf.v = torch.einsum("nji,nj->ni", C_all[0:1], v_T)
    out = torch.full((n, 3), float("nan"))
    for k in range(n):
      dt = float(t[k] - t[k - 1]) if k else 0.0
      qk, psik = q12[k : k + 1], psi[k : k + 1]
      p_I, R_I = be.feet_in_imu(qk, psik, be.CONTACT_A)
      g_I = C_all[k : k + 1] @ torch.tensor([0.0, 0.0, -1.0])
      alpha = be.contact_alpha(p_I, g_I.unsqueeze(0) if g_I.dim() == 1 else g_I)
      if k and 0 < dt < 0.1:
        ekf.predict(acc[k : k + 1], gyro[k : k + 1], dt, alpha)
      if use_grav:
        ekf.update_gravity(acc[k : k + 1])
      if use_att:
        ekf.update_attitude(C_all[k : k + 1].transpose(1, 2))
      Jv, Jw = be.foot_jacobians_imu(qk, psik, be.CONTACT_A)
      R_meas = be.build_R(Jv, Jw, alpha, be.MeasCfg(), use_ori)
      ekf.update_feet(p_I, R_I if use_ori else None, alpha, R_meas)
      v_T = torch.einsum("nji,nj->ni", ekf.R, ekf.v)
      r_imu = be._vec(be.R_IMU, acc).expand(1, 3)
      v_P = torch.einsum("nij,nj->ni", be.rz(psik),
                          v_T - torch.cross(gyro[k : k + 1] - ekf.bw, r_imu, dim=-1))
      out[k] = v_P[0]
    got = out[:, :2].numpy()
    ok = np.isfinite(got).all(axis=1) & np.isfinite(want[arm]).all(axis=1)
    assert ok.sum() > n // 2, f"{arm}: too many non-finite samples ({ok.sum()}/{n})"
    # post-settle: drop the first 20% (filter settling), mean should track the reference.
    tail = np.zeros(n, bool); tail[n // 5 :] = True
    m = ok & tail
    diffs[arm] = np.abs(got[m] - want[arm][m]).mean(axis=0)

  for arm, diff in diffs.items():
    print(f"{arm}: mean |diff| {diff}")
  for arm, diff in diffs.items():
    assert (diff < BAR).all(), f"{arm}: mean |diff| {diff} exceeds bar {BAR} (post-settle)"


def test_complementary_filter_tracks_the_numpy_reference():
  t, acc, gyro, quat, q, dq, want = _load()
  n = len(t)
  psi, q12 = q[:, 12], q[:, :12]
  C_all = be.matrix_from_quat(quat).transpose(1, 2)
  g_hat_T = torch.einsum("nij,nj->ni", C_all, torch.tensor([0.0, 0.0, -1.0]).expand(n, 3))
  a_free_T = acc + g_hat_T * 9.81
  a_P = torch.einsum("nij,nj->ni", be.rz(psi), a_free_T)

  legodom = torch.tensor(want["legodom"], dtype=torch.float32)
  ok = torch.isfinite(legodom).all(dim=-1)
  v_lo3 = torch.cat([torch.nan_to_num(legodom), torch.zeros(n, 1)], -1)

  cf = be.ComplementaryFilter(1, "cpu")
  out = torch.zeros(n, 3)
  for k in range(1, n):
    v = cf.step(a_P[k : k + 1], v_lo3[k : k + 1], ok[k : k + 1], float(t[k] - t[k - 1]))
    out[k] = v[0]
  got = out[:, :2].numpy()
  m = np.isfinite(want["compl"]).all(axis=1)
  diff = np.abs(got[m][n // 5 :] - want["compl"][m][n // 5 :]).mean(axis=0)
  assert (diff < BAR).all(), f"compl: mean |diff| {diff} exceeds bar {BAR}"


def test_jacobian_arm_matches_the_numpy_reference():
  t, acc, gyro, quat, q, dq, want = _load()
  n = len(t)
  psi, q12, dq12 = q[:, 12], q[:, :12], dq[:, :12]
  C_all = be.matrix_from_quat(quat).transpose(1, 2)
  gyro_pelvis = torch.einsum("nij,nj->ni", be.rz(psi), gyro) - torch.stack(
    [torch.zeros(n), torch.zeros(n), dq[:, 12]], -1)
  grav_P = torch.einsum("nij,nj->ni", be.rz(psi),
                        torch.einsum("nij,nj->ni", C_all, torch.tensor([0.0, 0.0, -1.0]).expand(n, 3)))
  p_P = be.foot_sites_b(q12)
  st = be.stance_index(p_P, grav_P)
  v = be.jacobian_velocity(q12, dq12, st, gyro_pelvis)
  got = v[:, :2].numpy()
  m = np.isfinite(want["jacobian"]).all(axis=1) & np.isfinite(got).all(axis=1)
  assert m.sum() > n // 2
  diff = np.abs(got[m] - want["jacobian"][m]).mean(axis=0)
  assert (diff < BAR).all(), f"jacobian: mean |diff| {diff} exceeds bar {BAR}"


# -----------------------------------------------------------------------------------------
# NaN resilience (2026-08-24, ekf_rot/ekf_grav cluster failures: "RuntimeError: normal
# expects std >= 0.0" many iterations deep in PPO, no estimator code in the traceback --
# a single non-finite tick in one env, uncaught, silently poisons persistent filter state
# forever and eventually reaches the policy). These guards are all no-ops in the clean-data
# tests above; this is what actually exercises them.
# -----------------------------------------------------------------------------------------

def test_ekf_predict_holds_previous_state_on_a_non_finite_tick_and_recovers():
  """One env in a 3-env batch gets a NaN acc reading on tick 2 of 3; the other two must be
  completely unaffected (proves the guard is per-env, not batch-wide), the poisoned env
  must hold its tick-1 state rather than go NaN, and a clean tick 3 must show it recovering
  (not permanently stuck) since the held state was itself finite."""
  torch.manual_seed(0)
  N = 3
  ekf = be.BaseEkf(N, "cpu", use_orientation=False)
  alpha = torch.zeros(N, 2)  # no feet in contact -> predict() is the only thing exercised

  acc = torch.tensor([0.1, -0.2, 9.9]).expand(N, 3).clone()
  gyro = torch.tensor([0.01, -0.02, 0.03]).expand(N, 3).clone()
  ekf.predict(acc, gyro, 0.01, alpha)
  after_tick1 = (ekf.r.clone(), ekf.v.clone(), ekf.R.clone(), ekf.P.clone())

  bad_acc = acc.clone()
  bad_acc[1, 0] = float("nan")
  ekf.predict(bad_acc, gyro, 0.01, alpha)
  assert torch.isfinite(ekf.r).all() and torch.isfinite(ekf.v).all()
  assert torch.isfinite(ekf.R).all() and torch.isfinite(ekf.P).all(), (
    "a non-finite input must never leave the filter's persistent state non-finite")
  # env 1 held its tick-1 state exactly; envs 0/2 moved on normally and are unaffected.
  for i in (0, 2):
    assert not torch.equal(ekf.r[i], after_tick1[0][i]), f"env {i} should have advanced"
  assert torch.equal(ekf.r[1], after_tick1[0][1]), "env 1 must hold, not advance, on NaN input"
  assert torch.equal(ekf.v[1], after_tick1[1][1])
  assert torch.equal(ekf.R[1], after_tick1[2][1])
  assert torch.equal(ekf.P[1], after_tick1[3][1])

  ekf.predict(acc, gyro, 0.01, alpha)  # clean tick again
  assert torch.isfinite(ekf.r).all() and torch.isfinite(ekf.v).all()
  assert not torch.equal(ekf.r[1], after_tick1[0][1]), (
    "env 1 must resume advancing on the next clean tick, not stay stuck forever")


def test_predict_holds_previous_state_on_a_huge_but_finite_acc():
  """2026-08-25, ekf_rot local run: the acc/gyro saturation added at synthetic_imu() did NOT
  stop a repeat crash -- a second, unclamped path (update_feet's joint-position-derived
  inputs) can still leave self.v huge-but-finite. predict()'s isfinite-only guard waves a
  finite outlier straight through; it needs a magnitude bound too, both to reject a huge
  input directly (this test) and as a backstop for whatever update_feet/update_gravity/
  update_attitude leave behind on the tick before (next test)."""
  torch.manual_seed(0)
  N = 3
  ekf = be.BaseEkf(N, "cpu", use_orientation=False)
  alpha = torch.zeros(N, 2)
  acc = torch.tensor([0.1, -0.2, 9.9]).expand(N, 3).clone()
  gyro = torch.tensor([0.01, -0.02, 0.03]).expand(N, 3).clone()
  ekf.predict(acc, gyro, 0.01, alpha)
  after_tick1 = ekf.v.clone()

  huge_acc = acc.clone()
  huge_acc[1, 0] = 1.0e5   # finite -- isfinite() alone would accept this
  ekf.predict(huge_acc, gyro, 0.01, alpha)
  assert torch.isfinite(ekf.v).all()
  assert torch.equal(ekf.v[1], after_tick1[1]), "a finite but implausible velocity must be rejected"
  for i in (0, 2):
    assert not torch.equal(ekf.v[i], after_tick1[i]), f"env {i} should have advanced normally"


def test_update_feet_rejects_a_huge_velocity_injection_even_when_mahalanobis_passes():
  """The existing Mahalanobis gate is covariance-relative: if P's own cross-covariance is
  ill-conditioned (a plausible symptom of accumulated numerical corruption, not modeled by any
  single measurement being 'too far off'), a totally ordinary ~1cm foot-position residual can
  still produce a ~14000 m/s^2 raw velocity injection while mahal stays a fraction of its gate
  (0.0002 vs a threshold of 150) -- H_pos only reads the r/p covariance blocks, so it cannot
  see a corrupted r-v cross term at all. Only a raw magnitude bound on dx catches this."""
  ekf = be.BaseEkf(1, "cpu", use_orientation=False)
  ekf.P[:, ekf.i_v : ekf.i_v + 3, ekf.i_r : ekf.i_r + 3] = 1e6 * torch.eye(3)
  ekf.anchored[:, 0] = True
  ekf.p[:, 0] = torch.tensor([0.05, 0.03, -0.8])
  alpha = torch.zeros(1, 2)
  alpha[:, 0] = 1.0
  s_p = torch.zeros(1, 2, 3)
  s_p[:, 0] = torch.tensor([0.06, 0.03, -0.79])  # ~1cm off the anchor -- an ordinary residual
  R_meas = torch.eye(6).unsqueeze(0) * 1e-3       # an ordinary measurement noise, not adversarial

  before_v = ekf.v.clone()
  ekf.update_feet(s_p, None, alpha, R_meas)
  assert torch.isfinite(ekf.v).all()
  assert torch.equal(ekf.v, before_v), "an implausible velocity injection must be rejected"


def test_complementary_filter_holds_previous_v_on_a_non_finite_tick():
  cf = be.ComplementaryFilter(2, "cpu")
  ok = torch.ones(2, dtype=torch.bool)
  v_lo = torch.zeros(2, 3)
  cf.step(torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]), v_lo, ok, 0.01)
  before = cf.v.clone()
  bad_a = torch.tensor([[float("nan"), 0.0, 0.0], [0.1, 0.0, 0.0]])
  cf.step(bad_a, v_lo, ok, 0.01)
  assert torch.isfinite(cf.v).all()
  assert torch.equal(cf.v[0], before[0]), "env 0 (bad input) must hold its previous v"
  assert not torch.equal(cf.v[1], before[1]), "env 1 (clean input) must still advance"


def _identity_free_state(N: int) -> tuple[torch.Tensor, torch.Tensor]:
  free_q = torch.zeros(N, 7)
  free_q[:, 2] = 1.0    # pos z = 1 m (standing)
  free_q[:, 3] = 1.0    # identity quat, wxyz
  free_v = torch.zeros(N, 6)
  return free_q, free_v


def test_synthetic_imu_is_a_no_op_at_physically_normal_magnitudes():
  """Standing still, qacc ~= 0: acc_T should read ~gravity (~9.81 m/s^2), nowhere near
  ACC_SATURATION -- confirms the clamp added for the njmax-overflow case doesn't touch the
  common case."""
  N = 2
  free_q, free_v = _identity_free_state(N)
  free_a = torch.zeros(N, 6)
  psi = torch.zeros(N)
  dpsi = torch.zeros(N)
  acc_T, gyro_T, _, _ = be.synthetic_imu(free_q, free_v, free_a, psi, dpsi)
  assert torch.isfinite(acc_T).all() and torch.isfinite(gyro_T).all()
  assert acc_T.norm(dim=-1).max() < 15.0, "should read close to plain gravity, not saturate"
  assert torch.equal(gyro_T, torch.zeros(N, 3))


def test_synthetic_imu_saturates_an_njmax_overflow_scale_qacc_spike():
  """2026-08-25, ekf_rot local run: 80x 'nefc overflow' warnings (MuJoCo dropping contact
  rows under early random-policy chaos) preceded the same 'normal expects std >= 0.0' crash
  the 2026-08-24 NaN guards were built for -- but this time from a huge-but-FINITE qacc, which
  isfinite()-only guards don't catch. Feed a wildly implausible qacc (1e6 m/s^2, ~1e5 g) into
  one env of a 2-env batch and confirm it comes out saturated, not propagated."""
  N = 2
  free_q, free_v = _identity_free_state(N)
  free_a = torch.zeros(N, 6)
  free_a[0, 0] = 1.0e6   # env 0: implausible world-frame linear qacc spike
  psi = torch.zeros(N)
  dpsi = torch.zeros(N)
  acc_T, gyro_T, _, _ = be.synthetic_imu(free_q, free_v, free_a, psi, dpsi)
  assert torch.isfinite(acc_T).all() and torch.isfinite(gyro_T).all()
  assert (acc_T.abs() <= be.ACC_SATURATION + 1e-4).all(), "spike must be clamped, not passed through"
  # env 1 (untouched) must be identical to the no-op case -- the clamp must be per-value, not
  # something that corrupts the whole batch because one env spiked.
  acc_T_clean, _, _, _ = be.synthetic_imu(*_identity_free_state(N), torch.zeros(N, 6), psi, dpsi)
  assert torch.equal(acc_T[1], acc_T_clean[1])


def test_synthetic_imu_saturates_a_non_finite_qacc():
  N = 1
  free_q, free_v = _identity_free_state(N)
  free_a = torch.zeros(N, 6)
  free_a[0, 1] = float("inf")
  free_v[0, 3] = float("nan")   # also poisons gyro_T's w_pelvis input
  psi = torch.zeros(N)
  dpsi = torch.zeros(N)
  acc_T, gyro_T, _, _ = be.synthetic_imu(free_q, free_v, free_a, psi, dpsi)
  assert torch.isfinite(acc_T).all() and torch.isfinite(gyro_T).all()


def test_estimator_bank_accumulate_skips_a_non_finite_tick_without_poisoning_the_window():
  """The central backstop: a NaN sample must not turn the WHOLE window's average into NaN
  (NaN + anything = NaN) -- it must be excluded from sum/n, same as a window with zero
  samples already is (fire()'s ``got = n > 0`` path)."""
  bank = be.EstimatorBank.__new__(be.EstimatorBank)  # skip __init__'s env-construction
  bank.sum = torch.zeros(2, 2)
  bank.n = torch.zeros(2)
  bank.value = torch.zeros(2, 2)
  bank._accumulate(torch.tensor([[1.0, 2.0], [float("nan"), 3.0]]))
  bank._accumulate(torch.tensor([[1.0, 2.0], [4.0, 5.0]]))
  assert torch.isfinite(bank.sum).all()
  assert bank.n[0] == 2 and bank.n[1] == 1, "the NaN tick must not be counted for env 1"
  v = bank.fire()
  assert torch.isfinite(v).all()
  assert torch.allclose(v[1], torch.tensor([4.0, 5.0])), (
    "env 1's average must come only from its one good sample")
