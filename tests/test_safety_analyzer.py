"""Run-selection + n_ticks correctness for `scripts/safety_analyzer.py` (ADR-0012).

Before this change the script had zero `entry` awareness -- exactly the silent-pooling
failure ADR-0012 was written to close -- and its `n_ticks` reconstruction assumed the
`t` column always starts at 0, which `safety_logger.h` confirms is only true for
`entry 0` (`tick_` never resets on FSM re-entry within one controller process). Both
are fixed together here: a Run after the first needs the entry guard to pick it AND a
duration formula that does not silently overcount once it does.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import safety_analyzer as sa  # noqa: E402

CONTROL_DT = 0.02
JOINTS = [{"slot": 0, "name": "j0", "min": -1.0, "max": 1.0},
          {"slot": 1, "name": "j1", "min": -1.0, "max": 1.0}]


def _safety_base(tmp_path, entries, n=5, name="run"):
    """A minimal but COMPLETE safety-logger pair: every column `main()` reads, plus
    `entry`, with `t` continuing tick-for-tick across entries (never resetting to 0 for
    entry > 0) -- the exact condition `safety_logger.h:147-148` describes and the one
    that catches a reversion to the old `t[-1]/control_dt` formula.
    """
    rows = n * len(entries)
    t = np.arange(rows) * CONTROL_DT  # tick_ never resets: t keeps counting across entries
    cols = {
        "t": t,
        "alpha": np.zeros(rows),
        "trig_joint": np.zeros(rows, dtype=int),
        "trig_tilt": np.zeros(rows, dtype=int),
        "trig_fall": np.zeros(rows, dtype=int),
        "entry": np.repeat(entries, n),
    }
    for j in JOINTS:
        cols[f"raw_q{j['slot']}"] = np.zeros(rows)
        cols[f"meas_q{j['slot']}"] = np.zeros(rows)
    base = tmp_path / name
    pd.DataFrame(cols).to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / f"{name}_meta.json").write_text(json.dumps(
        {"control_dt": CONTROL_DT, "joints": JOINTS}))
    return base


def test_multi_entry_file_without_entry_is_refused(tmp_path, monkeypatch):
    base = _safety_base(tmp_path, entries=[0, 1])
    monkeypatch.setattr(sys, "argv", ["safety_analyzer.py", str(base)])
    with pytest.raises(SystemExit, match=r"holds 2 Runs"):
        sa.main()


def test_entry_filters_and_n_ticks_uses_t0_of_the_filtered_slice(tmp_path, monkeypatch, capsys):
    n = 5
    base = _safety_base(tmp_path, entries=[0, 1], n=n)
    monkeypatch.setattr(sys, "argv", ["safety_analyzer.py", str(base), "--entry", "1"])
    sa.main()
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[SAFETY] "))
    report = json.loads(line[len("[SAFETY] "):])

    # entry 1's rows are t = [5,6,7,8,9] * CONTROL_DT: t[0] != 0, so the two formulas
    # disagree -- this is a numeric check, not just "did it raise".
    t1 = np.arange(n, 2 * n) * CONTROL_DT
    correct = int(round(float(t1[-1] - t1[0]) / CONTROL_DT)) + 1
    buggy = int(round(float(t1[-1]) / CONTROL_DT)) + 1
    assert correct != buggy, "fixture must make the two formulas disagree"
    assert report["n_ticks"] == correct
    assert report["n_rows_logged"] == n


def test_single_entry_file_needs_no_selector(tmp_path, monkeypatch, capsys):
    base = _safety_base(tmp_path, entries=[0])
    monkeypatch.setattr(sys, "argv", ["safety_analyzer.py", str(base)])
    sa.main()
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[SAFETY] "))
    report = json.loads(line[len("[SAFETY] "):])
    assert report["n_rows_logged"] == 5


def test_absent_entry_is_refused(tmp_path, monkeypatch):
    base = _safety_base(tmp_path, entries=[0, 1])
    monkeypatch.setattr(sys, "argv", ["safety_analyzer.py", str(base), "--entry", "7"])
    with pytest.raises(SystemExit, match=r"no entry 7"):
        sa.main()
