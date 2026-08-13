"""T2: the estimator-arm parity gate (doc/hrl/h1_2_ekf_design.md §13.5).

The robot runs the arms in C++ (`deploy/robots/h1_2/include/hrl/base_estimators.h`); every
number this project has published about them came from the Python harness
(`scripts/replay_base_estimators.py`). If those two disagree, a hardware result cannot be
attributed between the port and the robot, so both are pinned to ONE golden fixture:

    tests/fixtures/estimator_parity_input.txt      400 real DDS samples
    tests/fixtures/estimator_parity_expected.txt   vx,vy per arm

This file guards the PYTHON side. `deploy/robots/h1_2/test/base_state_estimator_test.cpp`
guards the C++ side against the same golden, and that pairing is what makes the two
comparable without pytest needing a C++ toolchain.

BAR, and why it is stated on the window rather than per sample: the HL consumes a c=8
window average of the estimate, never an individual tick, and `run()` ticks at ~1 kHz while
LowState changes at ~500 Hz, so the deployed filter sees each measurement once but predicts
twice. The window average is the quantity that both implementations can and must agree on.

INITIALISATION, measured 2026-08-13 and the reason this is worth a comment. Offline, the
harness warm-starts velocity at sample 0 from arm A's first finite reading -- it can look
ahead. Online, arm A needs a previous foot position, so the warm start necessarily lands one
tick later. For a contracting filter that difference decays (arms B/D/E/F agree to 1e-5 or
better after the settle); for the position-only arms it does NOT, because a divergent filter
amplifies any initial difference -- C and C+grav separate to ~2-4e-4. That is a property of
those arms, not of the port: driven from an identical initial condition, all seven agree to
~6e-8. So this test pins the arms from an identical initial condition, which is what
isolates a porting error from an initialisation convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FIX = REPO / "tests" / "fixtures"

ARMS = ["legodom", "compl", "ekf", "ekf_grav", "ekf_rot", "jacobian", "ekf_att"]
WINDOW = 160          # c=8 policy steps at ~500 Hz DDS
WINDOW_BAR = 1e-4     # m/s, on the c=8 window average
MEAN_BAR = 1e-5       # m/s, on the session mean

torch = pytest.importorskip("torch", reason="replay_base_estimators uses torch for batched FK")


def _load():
  inp = np.loadtxt(FIX / "estimator_parity_input.txt")
  exp = np.loadtxt(FIX / "estimator_parity_expected.txt")
  assert inp.shape[0] == exp.shape[0], "fixture rows disagree"
  assert exp.shape[1] == 2 * len(ARMS), f"expected {2 * len(ARMS)} output cols"
  return inp, exp


def _arms_from_fixture(inp):
  """Run every Python arm on the golden input, exactly as run_ekf/main do."""
  sys.path.insert(0, str(REPO / "scripts"))
  import replay_base_estimators as R

  s = R.Session(
    name="fixture", t=inp[:, 0], acc=inp[:, 1:4], gyro=inp[:, 4:7], quat=inp[:, 7:11],
    q=inp[:, 11:24], dq=inp[:, 24:37], cmd=np.zeros((len(inp), 3)),
    rate_hz=len(inp) / (inp[-1, 0] - inp[0, 0]), has_est=True, has_dq=True, joint_offset=[],
  )
  pre = R.prepare(s)
  pre_c = R.prepare(s, offset=R.CONTACT_A)
  va = R.arm_a_legodom(s, pre)
  v0 = va[np.isfinite(va).all(axis=1)][0]
  return {
    "legodom": va,
    "compl": R.arm_b_complementary(s, pre, va),
    "ekf": R.run_ekf(s, pre_c, use_ori=False, use_att=False, v0_pelvis=v0),
    "ekf_grav": R.run_ekf(s, pre_c, use_ori=False, use_att=False, v0_pelvis=v0, use_grav=True),
    "ekf_rot": R.run_ekf(s, pre_c, use_ori=True, use_att=False, v0_pelvis=v0),
    "jacobian": R.arm_e_jacobian(s, pre),
    "ekf_att": R.run_ekf(s, pre_c, use_ori=False, use_att=True, v0_pelvis=v0),
  }


@pytest.mark.parametrize("arm", ARMS)
def test_python_arm_matches_golden_on_the_c8_window(arm):
  """The published numbers must stay reproducible: a silent change to any arm's math moves
  the window average and is caught here before it reaches a hardware comparison."""
  inp, exp = _load()
  got = _arms_from_fixture(inp)[arm][:, :2]
  want = exp[:, 2 * ARMS.index(arm): 2 * ARMS.index(arm) + 2]
  ok = np.isfinite(got).all(axis=1) & np.isfinite(want).all(axis=1)
  assert ok.sum() > WINDOW, f"{arm}: too few finite samples to form a window"
  n = (ok.sum() // WINDOW) * WINDOW
  for axis, name in ((0, "vx"), (1, "vy")):
    g = got[ok, axis][:n].reshape(-1, WINDOW).mean(1)
    w = want[ok, axis][:n].reshape(-1, WINDOW).mean(1)
    assert np.abs(g - w).max() < WINDOW_BAR, (
      f"{arm} {name}: c=8 window average drifted "
      f"{np.abs(g - w).max():.3e} m/s from the golden (bar {WINDOW_BAR})")
    assert abs(got[ok, axis].mean() - want[ok, axis].mean()) < MEAN_BAR, (
      f"{arm} {name}: session mean drifted (bar {MEAN_BAR})")


def test_fixture_is_a_real_dds_stream_not_the_50hz_cache():
  """The whole point of the EstSample channel is that it is NOT the 50 Hz articulation
  cache. A fixture accidentally regenerated from the cached columns would hold each joint
  value for ~10 rows and silently make every arm look better than it is."""
  inp, _ = _load()
  dt = np.diff(inp[:, 0])
  assert 300.0 < 1.0 / np.median(dt) < 1200.0, "fixture is not at the DDS rate"
  held = (np.diff(inp[:, 11:24], axis=0) == 0).all(axis=1).mean()
  assert held < 0.25, f"{held:.1%} of rows repeat their joints — this looks like the 50 Hz cache"


def test_only_the_selected_arm_can_reach_the_hl_velocity():
  """T6, routing half (docs/adr/0007). A passive arm must reach the flight recorder and
  NOTHING else -- that invariant is what makes it defensible to run the known-divergent
  position-only arm inside the ~1 kHz control loop.

  `lo_sum_` is the accumulator `policy_step()` drains into `obs["hl_vel"]`, so every write to
  it is a path from an estimator to the policy. There must be exactly two: the shipped
  guarded arm-A path (taken only when arm 0 is selected) and the selected arm. A third write
  would mean a passive arm reached the policy. The C++ side of T6 (cross-arm isolation) lives
  in deploy/robots/h1_2/test/base_state_estimator_test.cpp.
  """
  src = (REPO / "deploy" / "robots" / "h1_2" / "src" / "State_RLHRL.cpp").read_text()
  writes = [ln.strip() for ln in src.splitlines() if "lo_sum_ +=" in ln]
  assert len(writes) == 2, f"expected exactly 2 writes to lo_sum_, found {len(writes)}: {writes}"
  assert any("v_a" in w for w in writes), "the shipped guarded arm-A accumulation is gone"
  assert any("out[est_arm_]" in w for w in writes), (
    "no write routes the SELECTED arm into lo_sum_ — selection is not wired")
  # The arm-A write must stay behind its selection guard, or arm 0 would double-count when
  # another arm is selected.
  assert "if (est_arm_ == 0)" in src, "arm A's accumulation lost its selection guard"
  # And nothing may write the accumulator from the passive path.
  assert "lo_sum_ += out[" not in src.replace("lo_sum_ += out[est_arm_]", ""), (
    "a non-selected arm writes lo_sum_")


def test_every_arm_is_present_and_distinct():
  """kEstArmNames order is a one-way door: it indexes both the config key and every logged
  column, so a reorder silently reinterprets past sessions."""
  _, exp = _load()
  cols = {a: exp[:, 2 * i:2 * i + 2] for i, a in enumerate(ARMS)}
  for a in ARMS:
    assert np.isfinite(cols[a]).any(), f"{a} column is entirely non-finite"
  # The EKF variants must not be accidentally identical: ekf_rot carries the orientation
  # block, ekf_att the vendor attitude, ekf_grav the tilt update.
  for other in ("ekf_rot", "ekf_att", "ekf_grav"):
    d = np.nanmax(np.abs(cols["ekf"] - cols[other]))
    assert d > 1e-6, f"ekf and {other} produced identical output — a flag is not wired through"
