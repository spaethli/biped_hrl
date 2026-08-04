"""The readiness-pipeline seams, chosen the same way as the rest of tests/: each one is a
place where a defect yields a confident WRONG RESULT rather than a crash.

All four failures these guard are real and dated:
  - a parked ONNX ran for a day because nothing compared what was deployed against what
    was requested (2026-08-03)
  - a print artifact in the `t` column was read as a 100 Hz control stall, and the same
    duplicate timestamps crashed the analyzer with ZeroDivisionError (2026-08-03)
  - a fall on the 0.5 -> 0 deceleration was invisible to held-segment averages (2026-08-03)
  - the elastic band is a silent toggle, so a harness-bound session scores as a clean walk

CPU-only, no MuJoCo/GPU/checkpoints, in keeping with the rest of the suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import deploy_gate_analyzer as dga  # noqa: E402
import deploy_provenance as prov  # noqa: E402


# --------------------------------------------------------------- row_dt / stride_proxy


def test_row_dt_survives_the_quantised_timestamps_that_crashed_the_analyzer():
  """The pre-2026-08-04 `t` column was written with 4 SIGNIFICANT digits, so past 10 s it
  quantised to 0.01 s bins and ~71% of consecutive rows shared a printed timestamp.
  `median(diff(t))` was therefore exactly 0.0 and `stride_proxy` divided by it."""
  t = np.round(np.arange(10.0, 20.0, 0.002), 2)  # 5 identical stamps per 0.01 s bin
  assert np.median(np.diff(t)) == 0.0, "fixture must reproduce the degenerate median"
  dt = dga.row_dt(t)
  # rel=1e-3 absorbs the n-1 endpoint effect (mean spacing over intervals, not samples).
  # The point is that it lands on the true spacing at all, against a median of exactly 0.
  assert dt == pytest.approx(0.002, rel=1e-3)
  assert dt > 0, "must be usable as a divisor -- this is what ZeroDivisionError'd"


def test_stride_proxy_returns_nan_rather_than_raising_on_a_degenerate_segment():
  assert np.isnan(dga.stride_proxy(np.zeros(500), 0.0))
  assert np.isnan(dga.stride_proxy(np.zeros(500), float("nan")))


def test_row_dt_matches_a_uniform_clock():
  assert dga.row_dt(np.arange(0, 5, 0.001)) == pytest.approx(0.001, rel=1e-9)


# --------------------------------------------------------------- transition windows


def _ramp(hold_v, n=200, dt=0.01):
  t = np.arange(n) * dt
  return t


def test_transition_window_catches_a_fall_on_the_deceleration():
  """The 2026-08-03 signature: walking cleanly at 0.5, then on the 0.5 -> 0 command the
  achieved velocity LURCHES upward while commanded to zero, and the fall triggers fire.
  A held-segment mean cannot see it -- the lurch is a transient inside the segment."""
  t = np.arange(400) * 0.01
  cmd = np.zeros((400, 3))
  cmd[:200, 0] = 0.5                      # held 0.5, then commanded to 0 at index 200
  ach = np.zeros((400, 3))
  ach[:200, 0] = 0.5
  ach[200:260, 0] = 3.0                   # the lurch
  fell = np.zeros(400, bool)
  fell[240:] = True
  out = dga.transition_windows(t, cmd, ach, [0, 200, 400], fell)
  assert len(out) == 1
  assert out[0]["fell"] is True
  assert out[0]["peak_overshoot"] == pytest.approx(3.0, abs=1e-6)
  assert out[0]["axis"] == "vx"


def test_transition_window_is_clean_when_the_decel_is_clean():
  t = np.arange(400) * 0.01
  cmd = np.zeros((400, 3)); cmd[:200, 0] = 0.5
  ach = np.zeros((400, 3)); ach[:200, 0] = 0.5; ach[200:230, 0] = 0.2
  out = dga.transition_windows(t, cmd, ach, [0, 200, 400], np.zeros(400, bool))
  assert out[0]["fell"] is False
  assert out[0]["peak_overshoot"] < 0.5


# --------------------------------------------------------------- band-release detector


def test_band_detector_flags_a_session_where_the_robot_never_travelled():
  """A swallowed band release leaves the robot on the harness: FSM transitions correct,
  CSV written, commands held, and zero translation. Must be distinguishable from a walk."""
  n = 2000
  t = np.arange(n) * 0.01                      # 20 s
  cmd = np.zeros((n, 3)); cmd[:, 0] = 0.3
  held = np.zeros((n, 3))                      # band attached: no translation
  walked = np.zeros((n, 3)); walked[:, 0] = 0.3  # ~6 m over 20 s
  dt = np.diff(t, prepend=t[0])
  assert abs((held[:, 0] * dt).sum()) < dga.BAND_MIN_TRAVEL_M
  assert abs((walked[:, 0] * dt).sum()) > dga.BAND_MIN_TRAVEL_M


def _write_base_csv(path: Path, cmd_vx, ach_vx, trig_tilt, dt=0.01, njoints=2):
  """Minimal SafetyLogger-schema CSV, enough for analyze_base_csv to run end to end."""
  n = len(cmd_vx)
  cols = (["t", "t_wall"] + [f"raw_q{i}" for i in range(njoints)]
          + [f"meas_q{i}" for i in range(njoints)] + ["cmd_vx", "cmd_vy", "cmd_wz",
             "ach_vx", "ach_vy", "ach_vz", "alpha", "trig_joint", "trig_tilt",
             "trig_fall", "entry", "phase_sin", "phase_cos"])
  rows = []
  for i in range(n):
    r = {c: 0.0 for c in cols}
    r["t"] = r["t_wall"] = i * dt
    r["cmd_vx"], r["ach_vx"] = cmd_vx[i], ach_vx[i]
    r["trig_tilt"] = float(trig_tilt[i])
    r["alpha"] = float(trig_tilt[i])
    rows.append(r)
  path.write_text(",".join(cols) + "\n"
                  + "\n".join(",".join(f"{r[c]}" for c in cols) for r in rows) + "\n")


def test_band_check_ignores_holds_after_a_fall(tmp_path):
  """A fallen robot does not travel either. Scoring every hold made a genuine FALL look
  like a harness artifact -- downgrading a NO-GO to 'this run tells you nothing', the worst
  way this gate can be wrong. Caught 2026-08-04 on the keeper's first full-battery run: it
  walked 7.1 m, then fell, and the two later 0.0 m holds flipped band_released to False.

  Goes through analyze_base_csv itself, not a local reimplementation -- a test that
  restates the logic it is guarding cannot fail when that logic breaks.
  """
  n = 6000                                   # 60 s at 100 Hz
  cmd = np.zeros(n); ach = np.zeros(n)
  cmd[:2000] = 0.3; ach[:2000] = 0.3         # 20 s walking -> ~6 m: band clearly released
  cmd[3000:] = 0.3                           # commanded again, but it is down and cannot move
  tilt = np.zeros(n, bool); tilt[2500:] = True

  csv = tmp_path / "sess.csv"
  _write_base_csv(csv, cmd, ach, tilt)
  out = dga.analyze_base_csv(str(csv))

  assert out["any_fall"] is True, "the fall itself must still be reported"
  assert out["band_released"] is True, (
    "a pre-fall hold that travelled proves the band released; post-fall zero-travel holds "
    "are explained by the fall and must not mask it as an infrastructure failure")
  assert any(e["after_fall"] for e in out["band_evidence"])


def test_band_thresholds_are_reachable_by_the_default_sequence():
  """The detector needs a hold of >= BAND_MIN_HOLD_S at >= BAND_MIN_CMD to run at all;
  a sequence without one would silently skip the check (reported as None, not a pass)."""
  import bridge_session
  keyvals = {"3": 0.3, "5": 0.5, "w": 1.0}
  ok = [float(s.split(":")[1]) >= dga.BAND_MIN_HOLD_S
        for s in bridge_session.DEFAULT_SEQ.split(",")
        if keyvals.get(s.split(":")[0], 0.0) >= dga.BAND_MIN_CMD]
  assert any(ok), "default sequence has no hold long/fast enough to test the band"


# --------------------------------------------------------------- stand drift


def test_stand_drift_reports_direction_not_just_magnitude():
  """Backward drift at zero command is the bridge echo of the hardware backward lean, so
  the sign carries the meaning -- a magnitude-only metric would hide it."""
  n = 1000
  t = np.arange(n) * 0.01
  cmd = np.zeros((n, 3))
  ach = np.zeros((n, 3)); ach[:, 0] = -0.05     # drifting backwards
  out = dga.stand_drift(t, cmd, ach, [0, n])
  assert len(out) == 1
  assert out[0]["drift_x_m"] < 0
  assert abs(out[0]["heading_deg"]) > 170       # ~180 deg = backward


def test_stand_drift_skips_commanded_segments():
  n = 1000
  t = np.arange(n) * 0.01
  cmd = np.zeros((n, 3)); cmd[:, 0] = 0.5
  ach = np.zeros((n, 3)); ach[:, 0] = 0.5
  assert dga.stand_drift(t, cmd, ach, [0, n]) == []


# --------------------------------------------------------------- joint clamp


def test_joint_clamp_reports_magnitude_past_the_stop_not_only_rate(tmp_path):
  """Rate alone hides how hard a pinned joint is driven. rolloverfix0p5 commanded
  ankle_roll ~2.0 rad past a +-0.2618 stop; the keeper ~0.5. Same rate class, very
  different mechanical demand."""
  meta = tmp_path / "x_meta.json"
  meta.write_text(json.dumps({"joints": [
    {"slot": 0, "name": "left_ankle_roll", "min": -0.2618, "max": 0.2618},
    {"slot": 1, "name": "left_knee", "min": -0.26, "max": 2.05}]}))
  raw = np.zeros((100, 2))
  raw[:50, 0] = 2.0                              # half the ticks, far past the stop
  rep = dga.joint_clamp_report(raw, meta)
  assert rep["available"]
  assert rep["any_joint_rate"] == pytest.approx(0.5)
  top = rep["per_joint"][0]
  assert top["joint"] == "left_ankle_roll"
  assert top["max_past_stop_rad"] == pytest.approx(2.0 - 0.2618, abs=1e-6)


def test_joint_clamp_is_absent_not_wrong_when_meta_is_missing(tmp_path):
  assert dga.joint_clamp_report(np.zeros((10, 2)), tmp_path / "nope.json") == {
    "available": False}


# --------------------------------------------------------------- provenance


def _fake_onnx(path: Path, payload: bytes):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(payload)


def test_reverse_index_names_the_source_run(tmp_path):
  """The capability that was missing on 2026-08-03: identify an unknown deployed file
  rather than only confirm or deny one guess."""
  run = tmp_path / "logs" / "2026-07-17_keeper"
  _fake_onnx(run / "low_level.onnx", b"KEEPER-WEIGHTS")
  index = prov.build_index(tmp_path / "logs")
  assert index[prov.md5(run / "low_level.onnx")][0].parent.name == "2026-07-17_keeper"


def test_index_maps_one_hash_to_every_run_that_holds_it(tmp_path):
  for name in ("run_a", "run_b"):
    _fake_onnx(tmp_path / "logs" / name / "policy.onnx", b"SAME")
  index = prov.build_index(tmp_path / "logs")
  assert len(next(iter(index.values()))) == 2


def test_expected_dims_track_hl_obs_vel_and_goal_components(tmp_path):
  """The C++ throws on a dim mismatch at FSM entry. 94 vs 92 is the hl_obs_vel switch;
  getting it wrong means the hierarchy silently will not load."""
  pdir = tmp_path / "v0"
  (pdir / "params").mkdir(parents=True)
  (pdir / "params" / "deploy.yaml").write_text(
    "hrl:\n  hl_obs_vel: true\n  goal_components: [velocity, orientation, height]\n")
  assert prov.expected_dims(pdir) == {"high_level.onnx": 94, "low_level.onnx": 96}
  (pdir / "params" / "deploy.yaml").write_text(
    "hrl:\n  hl_obs_vel: false\n  goal_components: [velocity]\n")
  assert prov.expected_dims(pdir) == {"high_level.onnx": 92, "low_level.onnx": 92}
