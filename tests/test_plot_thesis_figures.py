"""Thesis figure script (``scripts/plot_thesis_figures.py``): the seams that went blank.

The 2026-09-16 session first drew grey, empty figures for three independent reasons, none of
which raised: a cadence file hardcoded to another session, a colour table keyed on full tags,
and one NaN in the ground truth turning a whole RMS or drift into ``nan``. Each is pinned on
synthetic data whose right answer is known by construction.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import plot_thesis_figures as ptf  # noqa: E402

# Every tag the two hardware sessions produced, plus a loaded one.
TAGS = ["A1a", "A0_DR_s123", "A1a_DR_s123", "A1a_DR_cotcap_s42",
        "A1a_s42", "A1a_s123", "A0_s42", "A0_s123", "A1a_DR_s42", "A1a_DR_cotcap_s123",
        "A1a +7.5kg", "A0_DR_s123 +7.5kg"]


@pytest.mark.parametrize("tag", TAGS)
def test_every_session_tag_has_a_palette_colour(tag):
    # Keyed on the full tag, every seed nobody had listed fell back to the grey default.
    assert ptf.policy_color(tag) != "0.4"


def test_seeds_share_the_variant_colour_and_differ_in_marker():
    a, b = ptf.policy_style("A1a_s42"), ptf.policy_style("A1a_s123")
    assert a["color"] == b["color"] and a["marker"] != b["marker"]
    # the load rides on the line style, so a loaded Run is still its variant's colour
    assert ptf.policy_style("A1a +7.5kg")["color"] == ptf.policy_color("A1a")
    assert ptf.policy_style("A1a +7.5kg")["linestyle"] != ptf.policy_style("A1a")["linestyle"]
    # different variants must not collapse onto one colour
    assert len({ptf.policy_color(v) for v in ("A0_s42", "A0_DR_s123", "A1a_s42",
                                              "A1a_DR_s42", "A1a_DR_cotcap_s42")}) == 5


def test_sort_key_orders_variant_then_seed_then_load():
    shuffled = ["A1a_DR_s123", "A1a_s123", "A0_s42", "A1a +7.5kg", "A1a_s42", "A0_s123"]
    assert sorted(shuffled, key=ptf.policy_sort_key) == [
        "A0_s42", "A0_s123", "A1a_s42", "A1a +7.5kg", "A1a_s123", "A1a_DR_s123"]


def test_rms_ignores_samples_without_truth():
    n = 200
    est = np.zeros(n)
    truth = np.full(n, 0.3)
    truth[:60] = np.nan                        # a dropout, including the first sample
    m = np.ones(n, bool)
    assert ptf._rms(est, truth, m) == pytest.approx(0.3)


def test_rms_with_no_truth_at_all_is_nan_not_zero():
    n = 100
    assert np.isnan(ptf._rms(np.zeros(n), np.full(n, np.nan), np.ones(n, bool)))


def test_gap_pct_states_the_dropout():
    g = np.zeros(200)
    g[:50] = np.nan
    assert ptf._gap_pct(g) == pytest.approx(25.0)


def test_cadence_rows_merge_across_sessions_newest_wins(tmp_path):
    for session, rows in (("2026-09-14-x", [{"bundle": "b14", "v": 1}, {"bundle": "shared", "v": 1}]),
                          ("2026-09-16-x", [{"bundle": "b16", "v": 2}, {"bundle": "shared", "v": 2}])):
        (tmp_path / "data" / session).mkdir(parents=True)
        (tmp_path / "data" / session / "cadence_settling.json").write_text(json.dumps(rows))
    got = {r["bundle"]: r["v"] for r in ptf.load_cadence(tmp_path)}
    # a Run of either session has its row; the later file re-scores a shared one
    assert got == {"b14": 1, "b16": 2, "shared": 2}
    only = ptf.load_cadence(tmp_path, str(tmp_path / "data/2026-09-14-x/cadence_settling.json"))
    assert {r["bundle"] for r in only} == {"b14", "shared"}


def test_mocap_note_does_not_carry_another_sessions_excuse():
    B = lambda tag, mocap: ptf.Bundle(dir=Path(tag), run={}, policy="A1a", tag=tag,
                                      flight=Path("f"), hrl=None, joints=None, mocap=mocap)
    all_have = ptf._mocap_note([B("run 1", Path("m")), B("run 6", Path("m"))])
    assert "cannot appear" not in all_have and "corrupt" not in all_have
    some = ptf._mocap_note([B("run 1", Path("m")), B("run 7", None)])
    assert "run 7" in some.split("cannot appear")[1]


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pgf text metrics need LaTeX")
def test_stand_drift_survives_a_dropout_at_the_start_of_the_hold(tmp_path):
    """The displacement was differenced from the hold's FIRST sample, so a NaN there voided
    the whole Run; it is now the net displacement between the first and last CAPTURED one."""
    t = np.arange(0.0, 30.0, 0.002)
    pd.DataFrame({"t": t, "cmd_vx": 0.0, "cmd_vy": 0.0, "cmd_wz": 0.0}).to_csv(
        tmp_path / "f.csv", index=False)
    px = 0.03 * t                                # 0.03 m/s drift along x, nothing along y
    px[:500] = np.nan                            # the first second of the hold is not captured
    pd.DataFrame({"t": t, "gt_px": px, "gt_py": 0.0}).to_csv(tmp_path / "m.csv", index=False)
    b = ptf.Bundle(dir=tmp_path, run={}, policy="A1a_s42", tag="run 1",
                   flight=tmp_path / "f.csv", hrl=None, joints=None, mocap=tmp_path / "m.csv")
    ptf.fig_stand([b], [], tmp_path / "out")
    row = pd.read_csv(tmp_path / "out" / "f_stand_displacement.csv").iloc[0]
    # first captured sample is t = 1.0 s, last is t = 29.998 s
    assert row["total_displacement_m"] == pytest.approx(0.03 * (29.998 - 1.0), abs=1e-3)
    assert row["truth_missing_pct"] == pytest.approx(100 * 500 / len(t), abs=0.1)


def test_track_line_styles_keep_all_three_series_distinguishable():
    """The operator command is a solid line, the estimate dashed, the truth dotted (the thin
    dashed grey command was the one series nobody could find under the estimate)."""
    st = ptf.TRACK_STYLE
    assert st["command"][1] == "-"
    assert st["estimate"][1] != st["truth"][1] and st["estimate"][1] != "-" != st["truth"][1]
    assert st["command"][2] > st["estimate"][2]          # thickest, so it shows under the others
    # f_hier draws a fourth series, the HL target, and it must not reuse another's pattern
    assert len({repr(st[k][1]) for k in ("command", "target", "estimate", "truth")}) == 4


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pgf text metrics need LaTeX")
def test_track_runs_writes_one_pdf_per_run_with_truth_and_skips_the_rest(tmp_path):
    t = np.arange(0.0, 12.0, 0.002)
    z = np.zeros_like(t)
    flight = pd.DataFrame({"t": t, "cmd_vx": z, "cmd_vy": z, "cmd_wz": z,
                           "est_v_compl_x": 0.1 + z, "est_v_compl_y": z, "est_gyro_z": z})
    flight.to_csv(tmp_path / "f.csv", index=False)
    pd.DataFrame({"t": t, "gt_vx": z, "gt_vy": z, "gt_vz": z, "gt_wz": z, "gt_px": z,
                  "gt_py": z, "gt_pz": z, "gt_yaw": z}).to_csv(tmp_path / "m.csv", index=False)
    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()

    def bundle(d, mocap):
        return ptf.Bundle(dir=tmp_path / d, run={}, policy="A1a_s42", tag=d,
                          flight=tmp_path / "f.csv", hrl=None, joints=None, mocap=mocap)

    ptf.fig_track_runs([bundle("with", tmp_path / "m.csv"), bundle("without", None)],
                       tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").glob("*.pdf")) == ["with.pdf"]
    row = pd.read_csv(tmp_path / "out" / "track_runs_summary.csv").iloc[0]
    # a constant 0.1 m/s estimate against a truth of zero, all of it standing
    assert row["rms_vx_standing"] == pytest.approx(0.1)
    assert row["rms_vy_standing"] == pytest.approx(0.0)
    assert row["truth_missing_pct"] == 0.0


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pgf text metrics need LaTeX")
def test_hier_runs_need_both_ground_truth_and_hrl_telemetry(tmp_path):
    n = 600                                       # 12 s at the 50 Hz policy rate
    t = np.arange(1, n + 1) * 0.02
    z = np.zeros(n)
    pd.DataFrame({"t": t, "cmd_vx": z, "cmd_vy": z, "cmd_wz": z, "est_gyro_z": z}).to_csv(
        tmp_path / "f.csv", index=False)
    pd.DataFrame({"t": t, "gt_vx": z, "gt_vy": z, "gt_vz": z, "gt_wz": z, "gt_px": z,
                  "gt_py": z, "gt_pz": z, "gt_yaw": z}).to_csv(tmp_path / "m.csv", index=False)
    inc = np.full(n, 0.1)
    inc[::8] = 0.0                                # est_vx/vy reset to 0 at every HL fire, c = 8
    pd.DataFrame({"t": t, "cmd_vx": z, "cmd_vy": z, "cmd_wz": z, "tgt0": z, "tgt1": z,
                  "tgt2": z, "lo_vx": z, "lo_vy": z, "est_vx": inc, "est_vy": inc}).to_csv(
        tmp_path / "h.csv", index=False)
    for d in ("full", "no_hrl", "no_mocap"):
        (tmp_path / d).mkdir()

    def bundle(d, hrl, mocap):
        return ptf.Bundle(dir=tmp_path / d, run={}, policy="A1a_s42", tag=d,
                          flight=tmp_path / "f.csv", hrl=hrl, joints=None, mocap=mocap)

    m, h = tmp_path / "m.csv", tmp_path / "h.csv"
    ptf.fig_hier_runs([bundle("full", h, m), bundle("no_hrl", None, m),
                       bundle("no_mocap", h, None)], tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").glob("*.pdf")) == ["full.pdf"]
    row = pd.read_csv(tmp_path / "out" / "hier_runs_summary.csv").iloc[0]
    assert row["hl_fires"] == n // 8
