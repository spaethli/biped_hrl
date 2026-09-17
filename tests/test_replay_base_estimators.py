"""Run-selection for `scripts/replay_base_estimators.py`'s `load_session` (ADR-0012).

`load_session` reads a raw csv via Python's `csv` module into a numpy array (not
pandas) and has its own deliberate policy, kept unlike `read_flight`'s: default to the
LAST entry rather than refusing, because this is an interactive replay tool, not a
scoring one. It now routes that policy through the shared `resolve_run_entry`
(Phase 3), with one deliberate deviation from a naive mirror of that migration: the
`suffix` that feeds `Session.name` must stay empty on an ordinary single-entry file,
so the ambiguity guard (`len(entries) > 1`) is kept even though `resolve_run_entry`
itself would happily resolve a single-entry file too.
"""
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import replay_base_estimators as rbe  # noqa: E402

# Columns load_session's DRY-RUN FALLBACK branch reads (no est_* block -> has_est=False):
# t, entry, acc_{xyz}, quat_{wxyz}, meas_q{0..12}. Values vary per row so the
# sensor-changed dedupe (`keep`) does not collapse everything to one row.
FIELDS = (["t", "entry", "acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"]
          + [f"meas_q{i}" for i in range(13)])


def _replay_csv(tmp_path, entries, name="session.csv"):
    p = tmp_path / name
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for i, e in enumerate(entries):
            row = {"t": i * 0.02, "entry": e, "acc_x": 0.0, "acc_y": 0.0, "acc_z": -9.81,
                   "quat_w": 1.0, "quat_x": 0.0, "quat_y": 0.0, "quat_z": 0.0}
            for j in range(13):
                row[f"meas_q{j}"] = i * 0.01  # varies per row -> nothing gets deduped away
            w.writerow([row[f] for f in FIELDS])
    return p


def test_multi_entry_file_picks_the_last_entry(tmp_path):
    entries = [0, 0, 1, 1, 1]
    p = _replay_csv(tmp_path, entries)
    sess = rbe.load_session(p, entry=None)
    assert sess.name == "session#entry1"
    assert len(sess.t) == entries.count(1)


def test_single_entry_file_has_no_suffix(tmp_path):
    p = _replay_csv(tmp_path, [0, 0, 0, 0])
    sess = rbe.load_session(p, entry=None)
    assert sess.name == "session"
    assert len(sess.t) == 4


def test_notice_prints_exactly_once(tmp_path, capsys):
    p = _replay_csv(tmp_path, [0, 0, 1, 1, 1])
    rbe.load_session(p, entry=None)
    out = capsys.readouterr().out
    assert out.count("FSM entries present") == 1


def test_single_entry_file_with_explicit_wrong_entry_is_silently_ignored(tmp_path):
    """Pre-migration behavior, verified by reading the original code before writing this:

    the original guard was ``if len(entries) > 1:`` -- for a single-entry file that whole
    block, including the ``pick not in entries`` validation, was never reached at all, no
    matter what ``entry`` was passed. So a single-entry file with an explicit but WRONG
    ``--entry`` silently used the whole file rather than raising. The migration preserves
    this exact (if slightly surprising) legacy behavior: no elif branch was added to
    validate `entry` in the single-entry case, since that would be a new check nothing
    asked for.
    """
    p = _replay_csv(tmp_path, [0, 0, 0, 0])
    sess = rbe.load_session(p, entry=99)  # entry 99 does not exist in this file
    assert sess.name == "session"         # no suffix, no SystemExit
    assert len(sess.t) == 4               # whole file kept, not filtered to nothing
