#!/usr/bin/env python3
"""Offline safety analysis for h1_2 deploy flight-recorder logs.

Reads the two files written by the C++ SafetyLogger (deploy/robots/h1_2):
  <base>.csv        per-tick time series (raw policy intent, measured state, triggers)
  <base>_meta.json  limits / names / thresholds that were active during the run

and reports how often the *raw* (pre-filter) policy would have violated joint
limits, how often the filter actually engaged, and per-joint statistics. The raw
violation rate is the architecture-comparison metric: it quantifies how unsafe a
policy is independent of the filter that masks it at runtime.

Usage:
  python scripts/safety_analyzer.py <base>            # base path, no extension
  python scripts/safety_analyzer.py run.csv run_meta.json
Emits <base>_report.json and a one-line  [SAFETY] {json}  summary to stdout.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_flight_recorder import resolve_run_entry  # noqa: E402


def _resolve_paths(args):
    if len(args) == 1:
        base = args[0]
        # allow passing "<base>.csv" or "<base>"
        if base.endswith(".csv"):
            base = base[:-4]
        return Path(f"{base}.csv"), Path(f"{base}_meta.json"), base
    elif len(args) == 2:
        csv, meta = Path(args[0]), Path(args[1])
        base = str(csv)[:-4] if str(csv).endswith(".csv") else str(csv)
        return csv, meta, base
    raise SystemExit("expected <base> OR <csv> <meta.json>")


def _count_events(mask: np.ndarray) -> int:
    """Number of contiguous True runs (rising edges)."""
    m = mask.astype(int)
    return int(np.sum((m[1:] == 1) & (m[:-1] == 0)) + (m[0] if m.size else 0))


def _violation_stats(df, prefix, joints, t):
    """Per-joint and total out-of-range stats for the raw_q / meas_q columns."""
    per_joint = {}
    total = 0
    for j in joints:
        col = f"{prefix}{j['slot']}"
        x = df[col].to_numpy()
        below = j["min"] - x
        above = x - j["max"]
        over = np.maximum(np.maximum(below, above), 0.0)  # 0 if in range
        viol = over > 0.0
        n = int(viol.sum())
        total += n
        if n:
            first_idx = int(np.argmax(viol))
            per_joint[j["name"]] = {
                "count": n,
                "rate": round(n / len(x), 5),
                "max_over_rad": round(float(over.max()), 4),
                "first_t_s": round(float(t[first_idx]), 3),
            }
    return total, per_joint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="<base>  OR  <csv> <meta.json>")
    ap.add_argument("--entry", type=int, default=None,
                    help="select one FSM Run when the file holds several; refused if omitted")
    args = ap.parse_args()

    csv_path, meta_path, base = _resolve_paths(args.paths)
    if not csv_path.exists() or not meta_path.exists():
        raise SystemExit(f"missing {csv_path} or {meta_path}")

    meta = json.loads(meta_path.read_text())
    df = pd.read_csv(csv_path)
    if "entry" in df.columns:
        resolved = resolve_run_entry(df["entry"].to_numpy(), args.entry, csv_path.name,
                                      default="refuse")
        df = df[df["entry"] == resolved].reset_index(drop=True)
    joints = meta["joints"]
    t = df["t"].to_numpy()
    control_dt = meta["control_dt"]
    n_rows = len(df)
    # True control-tick count, NOT the logged row count: the flight recorder (2026-07-27)
    # decimates quiet ticks to ~500 Hz while always logging a tick in full when a safety
    # trigger fires, so n_rows < true tick count outside trigger bursts. tick_ increments
    # by exactly 1 every control cycle regardless of decimation, so t[-1] = last_tick *
    # control_dt reconstructs the true count exactly (and matches n_rows on older,
    # pre-decimation CSVs where every tick was logged).
    n_ticks = int(round(float(t[-1] - t[0]) / control_dt)) + 1 if n_rows else 0

    # --- filter engagement ---
    alpha = df["alpha"].to_numpy()
    engaged = alpha > 0.0
    trig_joint = df["trig_joint"].to_numpy().astype(bool)
    trig_tilt = df["trig_tilt"].to_numpy().astype(bool)
    trig_fall = df["trig_fall"].to_numpy().astype(bool)

    # --- violations: raw policy intent vs what actually happened ---
    raw_total, raw_per = _violation_stats(df, "raw_q", joints, t)
    meas_total, meas_per = _violation_stats(df, "meas_q", joints, t)

    report = {
        "source": str(csv_path),
        "n_ticks": n_ticks,
        "n_rows_logged": n_rows,
        "duration_s": round(float(t[-1]) if n_rows else 0.0, 3),
        "control_dt": control_dt,
        "filter": {
            "engaged_ticks": int(engaged.sum()),
            "engaged_frac": round(float(engaged.sum()) / n_ticks, 5) if n_ticks else 0.0,
            "engaged_time_s": round(float(engaged.sum()) * control_dt, 3),
            "trig_joint_ticks": int(trig_joint.sum()),
            "trig_tilt_ticks": int(trig_tilt.sum()),
            "trig_fall_ticks": int(trig_fall.sum()),
            "n_joint_events": _count_events(trig_joint),
            "n_tilt_events": _count_events(trig_tilt),
            "n_fall_events": _count_events(trig_fall),
        },
        "raw_policy_violations": {
            "total": raw_total,
            "rate": round(raw_total / n_ticks, 5) if n_ticks else 0.0,
            "per_joint": raw_per,
        },
        "measured_violations": {
            "total": meas_total,
            "rate": round(meas_total / n_ticks, 5) if n_ticks else 0.0,
            "per_joint": meas_per,
        },
    }

    out = Path(f"{base}_report.json")
    out.write_text(json.dumps(report, indent=2))

    # human-readable summary
    f = report["filter"]
    print(f"Analyzed {n_ticks} ticks ({report['duration_s']} s, {n_rows} rows logged) "
          f"from {csv_path.name}")
    print(f"  Filter engaged: {f['engaged_frac']*100:.1f}% of ticks "
          f"({f['engaged_time_s']} s)  "
          f"[joint x{f['n_joint_events']}, tilt x{f['n_tilt_events']}, fall x{f['n_fall_events']}]")
    print(f"  Raw policy limit violations: {raw_total} ticks "
          f"({report['raw_policy_violations']['rate']*100:.2f}%)")
    if raw_per:
        worst = sorted(raw_per.items(), key=lambda kv: -kv[1]["count"])[:5]
        for name, s in worst:
            print(f"    {name:24s} {s['count']:6d} ticks  max_over={s['max_over_rad']} rad")
    print(f"  Measured limit violations:   {meas_total} ticks "
          f"({report['measured_violations']['rate']*100:.2f}%)")
    print(f"Wrote {out}")
    # machine-readable one-liner (matches [BENCH]/[GOALDIAG] convention)
    print("[SAFETY] " + json.dumps(report))


if __name__ == "__main__":
    main()
