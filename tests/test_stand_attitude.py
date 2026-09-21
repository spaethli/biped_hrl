"""Quiet-stand attitude scoring (``scripts/score_stand_attitude.py``).

This tool exists to stop a confounded number reaching the thesis, so the tests are built
around the ways it could silently produce a plausible wrong one.

**Golden values.** The two runs that `docs/adr/0006` Amendment (2026-09-21) rests on: the
leaning A0 seed and the hierarchical arm at matched stillness. The fixture is the 10x
(policy-rate) reduction of the two captures, which reproduces the full-rate statistic to
0.0003 deg of pitch and 0.03 stillness points -- the captures themselves are 115-127 MB and
gitignored, so without the fixture the guard on the ADR's headline would skip everywhere.

**Seam invariants.** Each gate in the statistic has flipped a verdict on this project
before, so each is pinned on synthetic rows where the right answer is known by
construction rather than by re-running the decode:

- ``alpha == 0`` -- a held run reports the safety filter's posture, not the policy's.
- dq-filtering -- the whole ADR-0006 confound is that motion moves the reading.
- legs-only stillness -- ``hold_joint_ids`` freezes waist+arms on hardware.
- the 0.05 command gate -- the bench uses 0.1 for regime, the ADR uses 0.05 for stand.
- the pitch sign -- a backward lean must read negative or every table flips.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests/fixtures/stand_attitude"
sys.path.insert(0, str(REPO / "scripts"))

import score_stand_attitude as ssa  # noqa: E402

# docs/adr/0006 Amendment (2026-09-21), the per-RUN rows behind the per-arm table.
# (stem, still_pct, mean_maxdq, pitch_deg, offset is non-zero)
GOLDEN = {
    "2026-09-16_13-03-28": (84.881, 0.18554, -4.3505, True),   # A0_s123 run 16
    "2026-09-16_11-06-52": (79.593, 0.18276, +0.5718, False),  # A1a_DR_s42 run 11
}

# These are the FULL-RATE numbers the ADR quotes; the fixture is the 10x policy-rate
# reduction. Tolerances are the measured reduction error with ~2x margin, not values picked
# to make the test pass: over the two runs the reduction moves still_pct by <=0.028 pts,
# mean_maxdq by <=8.4e-4 and pitch_deg by <=2.4e-4.
TOL = {"still_pct": 0.05, "mean_maxdq": 2e-3, "pitch_deg": 2e-3}


def fixture_frame(stem):
    return pd.read_csv(FIX / f"{stem}.stand.csv.gz")


def synth(pitch_deg, dq, n=4000, cmd=0.0, alpha=0.0, ub_dq=0.0):
    """Rows with a pitch and a leg speed known by construction.

    A pure rotation about +y by theta gives q = (cos(t/2), 0, sin(t/2), 0), so any correct
    decode must return exactly theta -- independent of how the tool computes it.
    """
    th = np.radians(np.full(n, pitch_deg, dtype=float))
    d = {"quat_w": np.cos(th / 2), "quat_x": 0.0, "quat_y": np.sin(th / 2), "quat_z": 0.0,
         "cmd_vx": cmd, "cmd_vy": 0.0, "cmd_wz": 0.0, "alpha": alpha, "entry": 0}
    for i in range(12):
        d[f"meas_dq{i}"] = dq
    for i in range(12, 27):          # present so a legs-only -> all-27 change is expressible
        d[f"meas_dq{i}"] = ub_dq
    return pd.DataFrame({k: np.full(n, v, dtype=float) if np.isscalar(v) else v
                         for k, v in d.items()})


# --------------------------------------------------------------------- golden values

@pytest.mark.parametrize("stem", sorted(GOLDEN))
def test_golden_run_reproduces_adr_0006_amendment(stem):
    still, maxdq, pitch, _ = GOLDEN[stem]
    r = ssa.score_entry(fixture_frame(stem), 0)
    assert r["still_pct"] == pytest.approx(still, abs=TOL["still_pct"])
    assert r["mean_maxdq"] == pytest.approx(maxdq, abs=TOL["mean_maxdq"])
    assert r["pitch_deg"] == pytest.approx(pitch, abs=TOL["pitch_deg"])


def test_the_leaning_arm_is_the_more_still_one():
    """The ADR's whole argument: the confound points the right way and is too small.

    If this inverts, the 2026-08-03 stillness confound WOULD explain the lean and the
    amendment's conclusion is wrong. Stated as the ordering, not as the numbers.
    """
    lean = ssa.score_entry(fixture_frame("2026-09-16_13-03-28"), 0)
    ref = ssa.score_entry(fixture_frame("2026-09-16_11-06-52"), 0)
    assert lean["pitch_deg"] < ref["pitch_deg"] - 3.0      # leans by degrees, not tenths
    assert lean["still_pct"] > ref["still_pct"]            # while being the STILLER arm


def test_confound_slope_matches_the_adr_table():
    """Refit, never hardcoded -- but pinned, so a corrupted table is caught."""
    slope, r = ssa.confound_slope()
    assert slope == pytest.approx(-0.0739, abs=5e-4)
    assert r == pytest.approx(-0.958, abs=5e-4)


def test_joint_offset_is_read_from_the_run_not_the_yaml():
    """The config asymmetry the amendment rests on: A0 flew calibrated, the hierarchy did not."""
    for stem, (_, _, _, nonzero) in GOLDEN.items():
        off = json.loads((FIX / f"{stem}.meta.json").read_text())["joint_offset"]
        assert any(off) is nonzero
        assert len(off) == 27


# --------------------------------------------------------------------- seam invariants

def test_held_rows_are_excluded():
    """alpha>0 rows report the filter's posture. Held rows lean hard the other way."""
    free = synth(+0.5, dq=0.01, n=3000)
    held = synth(-40.0, dq=0.01, n=3000, alpha=0.7)
    r = ssa.score_entry(pd.concat([free, held], ignore_index=True), 0)
    assert r["pitch_deg"] == pytest.approx(0.5, abs=1e-3)
    assert r["alpha_frac"] == pytest.approx(0.5, abs=1e-6)


def test_pitch_is_dq_filtered():
    """Moving rows must not enter `pitch_deg`, but must enter `pitch_all_deg`.

    Both halves matter: the first proves the filter is applied, the second proves the tool
    did not simply drop the moving rows on read (which would make the filter untestable).
    """
    still = synth(+1.0, dq=0.01, n=3000)
    moving = synth(-9.0, dq=5.0, n=1000)
    r = ssa.score_entry(pd.concat([still, moving], ignore_index=True), 0)
    assert r["pitch_deg"] == pytest.approx(1.0, abs=1e-3)
    assert r["pitch_all_deg"] == pytest.approx(0.75 * 1.0 + 0.25 * -9.0, abs=1e-3)
    assert r["still_pct"] == pytest.approx(75.0, abs=1e-6)


def test_stillness_ignores_the_upper_body():
    """`hold_joint_ids` freezes waist+arms on hardware, so their motion is not the policy's."""
    d = synth(+0.5, dq=0.01, n=3000, ub_dq=50.0)
    assert ssa.score_entry(d, 0)["still_pct"] == pytest.approx(100.0, abs=1e-6)


def test_command_gate_is_the_adr_value_not_the_bench_regime_value():
    """Rows between the two thresholds are walking rows carrying a different attitude.

    ADR-0006 stands on |cmd|<0.05; the sim bench uses 0.1 for its regime split. Swapping
    them silently mixes walking rows into a stand statistic.
    """
    stand = synth(+0.5, dq=0.01, n=3000, cmd=0.0)
    between = synth(-12.0, dq=0.01, n=3000, cmd=0.08)
    r = ssa.score_entry(pd.concat([stand, between], ignore_index=True), 0)
    assert r["n"] == 3000
    assert r["pitch_deg"] == pytest.approx(0.5, abs=1e-3)


def test_backward_lean_reads_negative():
    """The sign convention every table in ADR-0006 is written against."""
    assert ssa.score_entry(synth(-3.5, dq=0.01), 0)["pitch_deg"] == pytest.approx(-3.5, abs=1e-3)
    assert ssa.score_entry(synth(+3.5, dq=0.01), 0)["pitch_deg"] == pytest.approx(+3.5, abs=1e-3)


def test_a_thin_run_scores_nothing_rather_than_a_mean():
    """Fail closed: too few zero-command rows must not silently produce a confident number."""
    assert ssa.score_entry(synth(+0.5, dq=0.01, n=ssa.MIN_ROWS - 1), 0) is None
