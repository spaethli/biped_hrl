"""analyze_cadence_settling: base-speed settling must only be scored where motion capture
actually covers the Run (2026-09-15).

np.interp CLAMPS outside its range, so flight rows past the end of a capture used to inherit
the last captured speed. If the capture stops while the robot is still moving, a policy that
then came to rest reads as never settling. Latent on 2026-09-14 (every scored segment was
covered; the floors reproduce to the last digit), but the next capture with a dropout would
bias exactly the metric ADR-0013 rests on.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_cadence_settling as acs  # noqa: E402


def _bundle(tmp_path, capture_end_s, run_s=60.0):
  """One long standing Run. The robot moves at 0.30 m/s until 45 s, then is still."""
  b = tmp_path / "2026_09_14-_12_00_synthetic"
  b.mkdir()
  (b / "run.json").write_text(json.dumps({"policy_tag": "synthetic"}))
  t = np.arange(0.0, run_s, 0.02)
  cols = ["t", "cmd_vx", "cmd_vy", "cmd_wz"] + [f"meas_dq{j}" for j in acs.UB_SLOTS]
  rows = np.column_stack([t] + [np.zeros_like(t)] * (len(cols) - 1))
  np.savetxt(b / "2026-09-14_12-00-00.csv", rows, delimiter=",", header=",".join(cols),
             comments="")
  tm = np.arange(0.0, capture_end_s, 0.002)
  speed = np.where(tm < 45.0, 0.30, 0.0)
  np.savetxt(b / "mocap_aligned.csv", np.column_stack([tm, speed, np.zeros_like(tm)]),
             delimiter=",", header="t,gt_vx,gt_vy", comments="")
  return b


def test_a_segment_the_capture_does_not_cover_is_dropped_not_censored(tmp_path):
  """Capture ends at 40 s while the robot is still moving. Clamped, the last 20 s all read
  0.30 m/s and the segment is scored as never settling. It must be excluded and counted."""
  base = acs.analyze(_bundle(tmp_path, capture_end_s=40.0))["base"]
  assert base["n_segments_uncovered"] == 1
  assert all(v["n"] == 0 for k, v in base.items() if k.startswith("t_settle_")), (
    "an uncovered segment was scored -- a capture gap reads as the robot never settling")


def test_a_fully_covered_segment_is_still_scored(tmp_path):
  """The control: the same Run with the capture spanning it must settle, or the guard above
  would pass by dropping everything."""
  base = acs.analyze(_bundle(tmp_path, capture_end_s=60.0))["base"]
  assert base["n_segments_uncovered"] == 0
  got = base[f"t_settle_{acs.BASE_THRESHOLDS[0]}"]
  assert got["n"] == 1 and got["censored"] == 0
  assert 44.0 < got["median"] < 47.0          # it moved until 45 s, then settled
  assert base["floor_m_s"] < 1e-3


def test_covered_rows_rejects_an_internal_dropout():
  tm = np.concatenate([np.arange(0.0, 10.0, 0.002), np.arange(12.0, 20.0, 0.002)])
  tf = np.array([5.0, 11.0, 15.0, 25.0])
  assert acs.covered_rows(tf, tm).tolist() == [True, False, True, False]


def test_read_mocap_drops_non_finite_samples(tmp_path):
  """A NaN reaching moving_average would be smeared across its whole SMOOTH_S window."""
  (tmp_path / "mocap_aligned.csv").write_text("t,gt_vx,gt_vy\n0.0,0.1,0\n0.002,nan,0\n0.004,0.1,0\n")
  M = acs.read_mocap(tmp_path)
  assert len(M["t"]) == 2 and np.isfinite(M["speed"]).all()
