#!/usr/bin/env python3
"""Bundle one hardware RUN into a self-contained folder under logs/robot_logs/.

A run is an FSM `entry`, NOT a file. The flight recorder writes one CSV per controller
PROCESS (safety_logger.h); going Passive -> walk again without restarting the controller
appends a new `entry` value to the same CSV. 2026-09-14 is 12 entries across 8 files, and
`15-00-37.csv` alone holds three runs under two different experimental conditions. Scoring
such a file whole pools distinct policies into one number.

`t` in the flight recorder is a TICK COUNTER, not a clock -- it advances one control_dt per
run() iteration regardless of real duration, so the gaps between entries (time spent in
Passive/FixStand) are invisible in it. `t_wall` is the only column that places an entry on
the wall clock, and it is what this tool uses.

Pairing with the joint telemetry log is by KNEE CROSS-CORRELATION, never by filename.
Neither filename is a first-sample time (the flight recorder's is process launch; logging
starts when the RL state is entered), and on 2026-09-14 the interval heuristic alone leaves
4 entries and 3 telemetry files unmatched even though they sit adjacent in time. An entry
with no confident partner is bundled WITHOUT torque data and says so in run.json, rather
than being silently attached to whichever file is nearest.

Usage:
  # 1. look first: every entry, its wall window, and its best telemetry candidate
  python scripts/bundle_hardware_run.py --scan 2026-09-14

  # 2. write the folders, once the labels are confirmed
  python scripts/bundle_hardware_run.py --labels labels.json

labels.json maps "<flight stem>:<entry>" to the run's identity, e.g.
  {
    "2026-09-14_14-07-20:0": {"tag": "A1a"},
    "2026-09-14_15-00-37:1": {"tag": "A1a", "note": "7.5 kg backpack", "payload_kg": 7.5},
    "2026-09-14_15-16-46:0": {"tag": "A0_DR_s123"}
  }
Optional per-run keys: note, payload_kg, checkpoint, pin_period, skip (true = not a run),
trajectory (the joint-telemetry stem, e.g. "2026-09-16_10-08-24": the operator's pairing
overrides the correlation search; the knee fit still runs, for the pair statistics only).
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bench_flight_recorder import (  # noqa: E402
    KNEE_SLOT, TRAJ_DIR, fit_alignment,
)

REPO = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO / "logs" / "deploy_safety"
OUT_DIR = REPO / "logs" / "robot_logs"
PAIR_CORR_FLOOR = 0.70  # same floor bench_flight_recorder uses to accept a pair
MAX_SKEW = 0.02         # |rate| beyond 2% is a degenerate fit, not a clock
MIN_PAIR_S = 20.0       # coarse_lag's window; a shorter entry cannot be aligned at all
OFFSET_OUTLIER_S = 300  # flag a wall-clock offset this far from the session median


# --------------------------------------------------------------------------- entries

def entry_windows(csv_path):
    """[(entry, row_lo, row_hi, t_wall0, t_wall1)] for one flight recorder CSV.

    Row indices are into the DATA rows (header excluded), so a streaming slicer can use
    them directly. Entries are contiguous, so lo/hi is enough -- no row mask needed.
    """
    stamp = pd.to_datetime(csv_path.name[:19], format="%Y-%m-%d_%H-%M-%S")
    out, cur, lo, w0, w1, i = [], None, 0, None, None, -1
    with open(csv_path) as fh:
        r = csv.reader(fh)
        hdr = next(r)
        ec, wc = hdr.index("entry"), hdr.index("t_wall")
        for i, row in enumerate(r):
            try:
                e, w = int(float(row[ec])), float(row[wc])
            except (ValueError, IndexError):
                continue
            if e != cur:
                if cur is not None:
                    out.append((cur, lo, i - 1, w0, w1))
                cur, lo, w0 = e, i, w
            w1 = w
    if cur is not None:
        out.append((cur, lo, i, w0, w1))
    return [(e, a, b, stamp + pd.Timedelta(seconds=s), stamp + pd.Timedelta(seconds=t))
            for e, a, b, s, t in out]


def entry_knee(csv_path, lo, hi):
    """(t, knee) for one entry, on the recorder's own non-uniform row grid."""
    t, knee = [], []
    with open(csv_path) as fh:
        r = csv.reader(fh)
        hdr = next(r)
        tc, kc = hdr.index("t"), hdr.index(f"meas_q{KNEE_SLOT}")
        for i, row in enumerate(r):
            if i < lo:
                continue
            if i > hi:
                break
            try:
                t.append(float(row[tc])); knee.append(float(row[kc]))
            except (ValueError, IndexError):
                continue
    return np.asarray(t), np.asarray(knee)


# --------------------------------------------------------------------------- pairing

def best_trajectory(t_f, knee_f, traj_dir, cache):
    """Best joint-telemetry partner for ONE entry, decided by knee correlation.

    Every candidate in the directory is tried. The wall-clock intervals are deliberately
    NOT used to pre-filter: on 2026-09-14 they exclude the correct answer for 4 of 12
    entries.

    Scoring goes through `fit_alignment`, not a constant lag, because the two loggers'
    clocks differ in RATE by ~-2600 ppm. Over a 170 s run that is 0.45 s of drift against a
    ~0.5 s stride period, so a correctly-offset pair scored at a single fixed lag ends up a
    full stride out of phase and correlates at nothing. `fit_alignment` fits offset AND rate
    (with stride-period alias resolution) and reports its own `xcorr`, which is the number
    bench_flight_recorder gates on.

    Returns (path, align_dict) or None.
    """
    best = None
    for p in sorted(Path(traj_dir).glob("all_joints_*.csv")):
        if p not in cache:
            try:
                d = np.genfromtxt(p, delimiter=",", names=True)
                t_a = np.atleast_1d(d["time"])
                k_a = np.atleast_1d(d[f"q{KNEE_SLOT}"])
                # The two 2026-02-19 captures predate the current schema (4 literal column
                # names for 82 columns) and parse to nothing; so does a truncated capture.
                cache[p] = (t_a, k_a) if t_a.size > 100 else None
            except Exception:
                cache[p] = None
        if cache[p] is None:
            continue
        t_a, knee_a = cache[p]
        try:
            al = fit_alignment(t_f, knee_f, t_a, knee_a)
        except Exception:
            continue
        if al is None or not np.isfinite(al.get("xcorr", np.nan)):
            continue
        # A fitted rate of more than 2% is not a clock. Real skew on this rig is thousands
        # of ppm (measured -3.3e-3 to -6.6e-3 this session); a 12.8 s entry, being shorter
        # than coarse_lag's 20 s window, produced a "pair" at 1.2 rate and 0.0 ms residual,
        # i.e. the fit absorbed everything into the rate and matched noise.
        if abs(al["m"]) > MAX_SKEW:
            continue
        if best is None or al["xcorr"] > best[1]["xcorr"]:
            best = (p, al)
    return best if best and best[1]["xcorr"] >= PAIR_CORR_FLOOR else None


# --------------------------------------------------------------------------- writing

def slice_csv(src, dst, lo, hi):
    """Copy header + data rows [lo, hi] verbatim. Bytes are preserved, not reformatted."""
    with open(src) as fi, open(dst, "w") as fo:
        fo.write(fi.readline())
        for i, line in enumerate(fi):
            if i < lo:
                continue
            if i > hi:
                break
            fo.write(line)


def hrl_entry_bounds(path, entry):
    """(lo, hi) data-row range of `entry` in an _hrl.csv, or None when absent."""
    lo = hi = None
    with open(path) as fh:
        r = csv.reader(fh)
        hdr = next(r)
        if "entry" not in hdr:
            return None
        ec = hdr.index("entry")
        for i, row in enumerate(r):
            try:
                e = int(float(row[ec]))
            except (ValueError, IndexError):
                continue
            if e == entry:
                lo = i if lo is None else lo
                hi = i
    return None if lo is None else (lo, hi)


def labelled_pair(flight, lo, hi, stem, traj_dir):
    """(path, align or None) for an operator-named telemetry log.

    The label decides identity, so the pair is kept even when the knee fit cannot score it
    (a short entry, or a rate outside MAX_SKEW); run.json then carries no pair statistics.
    """
    p = Path(traj_dir) / f"all_joints_{stem}.csv"
    if not p.exists():
        raise SystemExit(f"labelled trajectory {p} does not exist")
    d = np.genfromtxt(p, delimiter=",", names=True)
    t_f, knee_f = entry_knee(flight, lo, hi)
    try:
        al = fit_alignment(t_f, knee_f, np.atleast_1d(d["time"]), np.atleast_1d(d[f"q{KNEE_SLOT}"]))
    except Exception:
        al = None
    if al is not None and (not np.isfinite(al.get("xcorr", np.nan)) or abs(al["m"]) > MAX_SKEW):
        al = None
    return p, al


def bundle(flight, entry, lo, hi, w0, w1, label, pair, multi):
    stem = flight.stem
    tag = label.get("tag", "UNKNOWN")
    name = f"{w0.strftime('%Y_%m_%d')}-_{w0.strftime('%H_%M')}_{tag}"
    if multi:
        name += f"_e{entry}"
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    slice_csv(flight, out / flight.name, lo, hi)
    meta = flight.with_name(stem + "_meta.json")
    if meta.exists():
        shutil.copy2(meta, out / meta.name)
    hrl = flight.with_name(stem + "_hrl.csv")
    hrl_rows = hrl_entry_bounds(hrl, entry) if hrl.exists() else None
    if hrl_rows:
        slice_csv(hrl, out / hrl.name, *hrl_rows)

    run = {
        "policy_tag": tag,
        "flight_recorder": flight.name,
        "entry": entry,
        "wall_start": w0.isoformat(timespec="seconds"),
        "wall_end": w1.isoformat(timespec="seconds"),
        "duration_s": round((w1 - w0).total_seconds(), 1),
        "has_hrl_telemetry": bool(hrl_rows),
        "all_joints": None, "pair_lag_s": None, "pair_skew_ppm": None,
        "pair_xcorr": None, "pair_resid_ms": None, "pair_windows": None,
        "mocap": None,
        "note": label.get("note"),
        "payload_kg": label.get("payload_kg", 0.0),
        "checkpoint": label.get("checkpoint"),
        "pin_period": label.get("pin_period"),
    }
    if pair:
        p, al = pair
        for old in out.glob("all_joints_*.csv"):
            if old.name != p.name:
                old.unlink()
        shutil.copy2(p, out / p.name)
        run["all_joints"] = p.name
        if label.get("trajectory"):
            run["pair_source"] = "label"
        if al is not None:
            run.update(pair_lag_s=round(al["b"], 3),
                       pair_skew_ppm=round(al["m"] * 1e6, 1),
                       pair_xcorr=round(al["xcorr"], 4),
                       pair_resid_ms=round(al["resid_ms"], 2),
                       pair_windows=f"{al['windows']}/{al['windows_total']}")
        else:
            run["pair_note"] = "operator-labelled; the knee fit could not score this pair"
    else:
        run["torque_unavailable"] = (
            "no joint-telemetry log passed the knee cross-correlation floor; mech_power_w "
            "and cot cannot be computed for this run")
    # A re-bundle must not erase mocap_align's record of a slice it did not change: the
    # aligned csv is a function of (flight slice, mocap take) only.
    prev_path = out / "run.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    if (prev.get("flight_recorder"), prev.get("entry")) == (flight.name, entry):
        for k in ("mocap", "mocap_align"):
            if prev.get(k) is not None:
                run[k] = prev[k]
    prev_path.write_text(json.dumps(run, indent=2) + "\n")
    return out, run


# --------------------------------------------------------------------------- main

def collect(date, traj_dir):
    """[(flight, entry, lo, hi, w0, w1, pair)] for every entry on `date`."""
    cache, rows = {}, []
    for f in sorted(DEPLOY_DIR.glob(f"{date}_*.csv")):
        if f.name.endswith("_hrl.csv"):
            continue
        for entry, lo, hi, w0, w1 in entry_windows(f):
            t_f, knee_f = entry_knee(f, lo, hi)
            long_enough = len(t_f) > 100 and (t_f[-1] - t_f[0]) >= MIN_PAIR_S
            pair = best_trajectory(t_f, knee_f, traj_dir, cache) if long_enough else None
            rows.append((f, entry, lo, hi, w0, w1, pair))
    return rows, cache


def wall_offset(pair, w0):
    """Telemetry filename stamp minus this entry's wall start, in seconds.

    A pair is accepted on the knee correlation alone, which is right, but the offsets across
    one session should still cluster: the two loggers are started by the same hands minutes
    apart. An offset that sits far outside the rest is worth a human look even at corr 0.98,
    because a quasi-periodic walking knee plus a free rate parameter can fit a wrong bout.
    """
    stamp = pd.to_datetime(pair[0].name[11:30], format="%Y-%m-%d_%H-%M-%S")
    return (stamp - w0).total_seconds()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", metavar="DATE", help="list entries for a date, write nothing")
    ap.add_argument("--labels", type=Path, help="labels json -> write the bundles")
    ap.add_argument("--date", help="date for --labels (default: inferred from the keys)")
    ap.add_argument("--traj-dir", default=TRAJ_DIR)
    ap.add_argument("--accept-outliers", action="store_true",
                    help="keep pairs whose wall-clock offset is far from the session median")
    args = ap.parse_args()

    if not args.scan and not args.labels:
        ap.error("give --scan DATE or --labels labels.json")

    date = args.scan
    labels = {}
    if args.labels:
        labels = json.loads(args.labels.read_text())
        date = args.date or sorted(labels)[0].split(":")[0][:10]

    rows, cache = collect(date, args.traj_dir)
    per_file = {}
    for f, e, *_ in rows:
        per_file[f] = per_file.get(f, 0) + 1

    offs = [wall_offset(p, w0) for *_, w0, _, p in
            [(f, e, a, b, w0, w1, p) for f, e, a, b, w0, w1, p in rows] if p]
    med = float(np.median(offs)) if offs else 0.0

    print(f"{'flight file':<26} {'ent':>3} {'start':<8} {'dur':>7}  "
          f"{'telemetry partner':<38} {'corr':>6} {'ppm':>7} {'ms':>5} {'d_wall':>8}")
    flagged = []
    for f, entry, lo, hi, w0, w1, pair in rows:
        pn = pair[0].name if pair else "-- NONE (no torque) --"
        sc = ""
        if pair:
            off = wall_offset(pair, w0)
            mark = ""
            if abs(off - med) > OFFSET_OUTLIER_S:
                mark = " <-- OFFSET OUTLIER"
                flagged.append((f.stem, entry, pair[0].name, off))
            sc = (f"{pair[1]['xcorr']:>6.3f} {pair[1]['m']*1e6:>7.0f} "
                  f"{pair[1]['resid_ms']:>5.1f} {off:>+8.0f}{mark}")
        print(f"{f.name:<26} {entry:>3} {w0.strftime('%H:%M:%S'):<8} "
              f"{(w1-w0).total_seconds():7.1f}  {pn:<38} {sc}")
    if flagged:
        print(f"\n⚠ {len(flagged)} pair(s) sit >{OFFSET_OUTLIER_S:.0f}s from the session "
              f"median wall offset ({med:+.0f}s). High knee correlation does not rule out a "
              f"different walking bout with a similar cadence -- confirm by hand:")
        for stem, e, name, off in flagged:
            print(f"    {stem}:{e} -> {name} ({off:+.0f}s)")

    claimed = {p[0].name for *_, p in rows if p}
    orphan = [p.name for p in sorted(Path(args.traj_dir).glob(f"all_joints_{date}_*.csv"))
              if p.name not in claimed]
    if orphan:
        print(f"\nunclaimed telemetry logs ({len(orphan)}): {', '.join(orphan)}")

    if flagged and not args.accept_outliers:
        drop = {(s, e) for s, e, _, _ in flagged}
        rows = [(f, e, a, b, w0, w1, None if (f.stem, e) in drop else p)
                for f, e, a, b, w0, w1, p in rows]
        print("\nOutlier pairs DROPPED (torque unavailable for those runs). "
              "Pass --accept-outliers to keep them.")

    if not args.labels:
        print("\nscan only, nothing written. Write a labels json and rerun with --labels.")
        return

    print()
    for f, entry, lo, hi, w0, w1, pair in rows:
        key = f"{f.stem}:{entry}"
        lab = labels.get(key)
        if lab is None:
            print(f"[skip] {key}: not in labels json")
            continue
        if lab.get("skip"):
            print(f"[skip] {key}: marked skip ({lab.get('note', '')})")
            continue
        if lab.get("trajectory"):
            auto = pair[0].name if pair else None
            pair = labelled_pair(f, lo, hi, lab["trajectory"], args.traj_dir)
            if auto != pair[0].name:
                print(f"[label] {key}: trajectory {pair[0].name} from the label "
                      f"(correlation search picked {auto})")
            if pair[1] is None:
                print(f"[label] {key}: ⚠ knee fit could not score {pair[0].name}; "
                      f"attached on the label alone")
        out, run = bundle(f, entry, lo, hi, w0, w1, lab, pair, per_file[f] > 1)
        print(f"[bundle] {out.name}  ({run['duration_s']}s, "
              f"hrl={'y' if run['has_hrl_telemetry'] else 'n'}, "
              f"traj={'y' if run['all_joints'] else 'NONE'})")


if __name__ == "__main__":
    main()
