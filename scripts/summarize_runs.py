"""Summarize a BATCH of bridge/readiness runs into one table, and order their triggers.

`deploy_gate_analyzer.py` scores ONE csv. Every multi-run question -- a 2x2 plant/hold
matrix, an est-vs-gt A/B, "does this filter precede the fall" -- was previously answered
by a fresh ad-hoc grep or throwaway script each time. That is how the 2026-08-05 session
produced nine one-off scripts for three recurring questions, and how a `grep` over a doc
with a 20 kB single-line table cell cost more context than analysing 2.5 GB of CSV
(the analysis runs out-of-process and only its digest is ever read; a grep returns whole
lines).

So: compact digest to stdout, full detail to `--json` on disk. Read the file only if a
number in the digest looks wrong.

Scoring is delegated to `deploy_gate_analyzer.analyze_base_csv` -- this tool computes no
verdicts of its own, exactly like `deploy_readiness.py`. Two failure modes it exists to
avoid, both paid for already:
  * scraping prose for a verdict a tool already publishes ("fall" in out.lower() vs "FELL")
  * reading `trig_*` booleans as "quiet" on a capture written before those columns existed

Usage:
  python scripts/summarize_runs.py t3_                      # every run whose tag has t3_
  python scripts/summarize_runs.py 'ab_' --group 'ab_(\\w+?)_r'   # fall counts per cell
  python scripts/summarize_runs.py nh_ --events             # trigger ordering per run
  python scripts/summarize_runs.py t3_ --json /tmp/t3.json  # full detail to disk
"""

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy_gate_analyzer as dga  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs" / "deploy_safety"


def find_runs(pattern: str, arm: str) -> list[Path]:
  """Candidate CSVs matching `pattern`. `_hrl.csv` is the HRL sidecar, not a base capture."""
  hits = [Path(p) for p in sorted(glob.glob(str(LOGS / f"*{pattern}*_{arm}.csv")))
          if not p.endswith("_hrl.csv")]
  return hits


def tag_of(path: Path, arm: str) -> str:
  """Strip the timestamp prefix and the arm suffix, leaving the --tag the run was given."""
  return re.sub(rf"^\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d_|_{arm}$", "", path.stem)


def first_events(path: Path) -> dict:
  """When each trigger channel first fired, and whether the trip led the fall.

  This is the question that decided whether the per-joint torque/velocity hold was a fall
  GUARD or a fall CAUSE (2026-08-05: it led in 5 of 6 falls, so it ships detect-only).
  Ordering alone never proves causation -- it discriminates 'consequence of the fall' from
  'precedes the fall', which is the cheap half of the question. The other half is a control
  run with the mechanism disabled.
  """
  hdr = open(path).readline().strip().split(",")
  want = ["t", "alpha", "trig_tilt", "trig_fall"]
  opt = [c for c in ("trig_torque", "trig_dq", "alpha_trip") if c in hdr]
  d = np.genfromtxt(path, delimiter=",", names=True, usecols=want + opt, dtype=float)

  def first(mask):
    i = np.where(mask)[0]
    return round(float(d["t"][i[0]]), 3) if len(i) else None

  fall = first((d["trig_tilt"] != 0) | (d["trig_fall"] != 0))
  trip = first((d["trig_torque"] != 0) | (d["trig_dq"] != 0)) if opt else None
  out = {"first_fall": fall, "first_trip": trip,
         "first_hold": first(d["alpha"] > 0),
         "first_trip_hold": first(d["alpha_trip"] >= 1.0) if "alpha_trip" in opt else None,
         "trip_channel": bool(opt)}
  out["trip_lead_ms"] = round((fall - trip) * 1000, 0) if (fall and trip) else None
  return out


def summarize(path: Path, events: bool) -> dict:
  out = dga.analyze_base_csv(str(path))
  jt = out.get("joint_trip", {})
  row = {
    "file": str(path),
    "duration_s": out["duration_s"],
    "fall": out["any_fall"],
    "transition_fall": out["any_transition_fall"],
    "alpha_max": max((s["alpha_max"] for s in out["segments"]), default=0.0),
    # None, never 0, when the columns are absent: unscored is not the same as quiet.
    "trip_available": jt.get("available", False),
    "alpha_trip_max": jt.get("alpha_trip_max"),
    "torque_rate": jt.get("torque_rate"),
    "dq_rate": jt.get("dq_rate"),
    "worst_trip_joint": next(iter(jt.get("worst_joint_share", {})), None),
    "band_released": out.get("band_released"),
    "loop_rate_hz": out.get("loop_rate_hz"),
    "segments": out["segments"],
    "joint_trip": jt,
  }
  if events:
    row["events"] = first_events(path)
  return row


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("pattern", help="substring of the run --tag")
  ap.add_argument("--arm", default="candidate", choices=["candidate", "a0control"])
  ap.add_argument("--group", default=None,
                  help="regex with ONE capture group over the tag; aggregates fall counts "
                       "per captured cell (e.g. 'ab_(\\w+?)_r' -> est/gt)")
  ap.add_argument("--events", action="store_true", help="add trigger-ordering columns")
  ap.add_argument("--json", dest="json_out", default=None,
                  help="write full per-run detail here; stdout stays a digest")
  args = ap.parse_args()

  paths = find_runs(args.pattern, args.arm)
  if not paths:
    print(f"no {args.arm} runs matching {args.pattern!r} under {LOGS}")
    return 1

  rows = {}
  for p in paths:
    try:
      rows[tag_of(p, args.arm)] = summarize(p, args.events)
    except SystemExit as e:  # analyzer refuses pre-2026-07-21 captures by design
      print(f"{tag_of(p, args.arm):<24} SKIPPED — {e}")

  hdr = f'{"run":<24}{"fall":>6}{"trans":>7}{"alpha":>7}{"trip":>7}{"tau%":>7}{"dq%":>7}'
  if args.events:
    hdr += f'{"trip_t":>9}{"fall_t":>9}{"lead_ms":>9}'
  print(hdr)
  for tag, r in rows.items():
    trip = "n/a" if not r["trip_available"] else f'{r["alpha_trip_max"]:.2f}'
    tau = "-" if r["torque_rate"] is None else f'{r["torque_rate"]:.2%}'
    dq = "-" if r["dq_rate"] is None else f'{r["dq_rate"]:.2%}'
    line = (f'{tag:<24}{str(r["fall"]):>6}{str(r["transition_fall"]):>7}'
            f'{r["alpha_max"]:>7.2f}{trip:>7}{tau:>7}{dq:>7}')
    if args.events:
      e = r["events"]
      # Formatted before interpolation, not inside it: nested same-quote f-strings are a
      # 3.12+ feature and this repo runs 3.11.
      t_trip = "-" if e["first_trip"] is None else f'{e["first_trip"]:.2f}'
      t_fall = "-" if e["first_fall"] is None else f'{e["first_fall"]:.2f}'
      lead = "-" if e["trip_lead_ms"] is None else f'{e["trip_lead_ms"]:+.0f}'
      line += f"{t_trip:>9}{t_fall:>9}{lead:>9}"
    print(line)

  if args.group:
    cells = defaultdict(lambda: [0, 0])
    for tag, r in rows.items():
      m = re.search(args.group, tag)
      if not m:
        continue
      c = cells[m.group(1)]
      c[0] += int(bool(r["fall"]))
      c[1] += 1
    print(f'\n{"cell":<24}{"falls":>10}')
    for k in sorted(cells):
      f, n = cells[k]
      print(f"{k:<24}{f:>5} / {n:<4}")

  n_fell = sum(1 for r in rows.values() if r["fall"])
  print(f"\n{len(rows)} runs, {n_fell} with a fall"
        + (f"; full detail -> {args.json_out}" if args.json_out else ""))
  if args.json_out:
    Path(args.json_out).write_text(json.dumps(rows, indent=2, default=str))
  return 0


if __name__ == "__main__":
  sys.exit(main())
