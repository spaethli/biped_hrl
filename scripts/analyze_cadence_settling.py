#!/usr/bin/env python3
"""Commanded cadence and settling time per hardware _Run_, the two quantities that separate
the A1a line from its payload-DR arms when the pre-registered metrics do not.

WHY THESE TWO. `act_legs_rad` and `ub_arm_vel` average an AMPLITUDE over the whole standing
regime. A policy whose gait clock runs faster at a standstill moves more, more often, at the
same amplitude, and those metrics are blind to it: on 2026-09-14 both DR Runs landed inside
A1a's own 2.4x within-session band. Two things can see it.

  1. COMMANDED CADENCE. With `pin_period: 0.0` the HL owns the gait clock and its commanded
     period is logged per policy step. Split by _Regime_ it is a direct read of what the HL
     asks for while the robot is supposed to be still. ⚠ A pinned session logs the pin, not
     the HL's choice (`State_RLHRL.cpp` decodes the period only when `pin_period <= 0`), so a
     run whose period is constant to the float is reported as PINNED, never as a measurement.

  2. SETTLING TIME. Time from the command dropping to zero until the robot is actually still,
     which is the F1 fidelity metric and is a DURATION, not an amplitude.

THRESHOLD DISCIPLINE. A settling time needs a stillness threshold, and this project has been
burned once by a pre-registered bar scored against an assumed floor rather than a measured
one (WP1b). So: the steady-state floor is MEASURED here (the late part of long standing
segments), and settling time is reported at SEVERAL thresholds. A verdict that survives the
threshold sweep is a result; one that flips is an artifact of the threshold.

CHANNELS. Only the main body is motion captured, so the two channels have different sources:
  - arm velocity   -- `meas_dq` over slots 12..26, the instantaneous form of `ub_arm_vel`.
                      Always available.
  - base speed     -- mocap `gt_vx/gt_vy`, present only once `mocap_align.py` has run.
                      Skipped with a note when absent; the robot has no other true base speed.

Usage:
  python scripts/analyze_cadence_settling.py                      # all 2026-09-14 bundles
  python scripts/analyze_cadence_settling.py --bundles logs/robot_logs/2026_09_14-*/
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_flight_recorder import CMD_THRESHOLD, UB_SLOTS  # noqa: E402

SMOOTH_S = 0.20        # moving average before thresholding; |dq| per-sample is encoder noise
HOLD_S = 2.0           # the signal must stay under the threshold this long to count as settled
FLOOR_TAIL_S = 5.0     # late part of a standing segment, used to MEASURE the settled floor
FLOOR_MIN_SEG_S = 10.0 # a standing segment shorter than this cannot show a settled tail
MIN_SETTLE_SEG_S = 10.0 # ...nor a settle; see settling_times()
ARM_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)   # rad/s, swept
BASE_THRESHOLDS = (0.03, 0.05, 0.10)        # m/s, swept
PERIOD_RANGE = (0.35, 1.0)                  # cadence_period_range in the deploy yaml
PIN_TOL = 1e-6         # a period constant to this is the pin, not a choice


def moving_average(t, y, win_s):
  """Time-weighted box filter on a NON-uniform grid (recorder rows are ~500 Hz, decimated)."""
  if len(t) < 3:
    return y
  grid = np.arange(t[0], t[-1], np.median(np.diff(t)))
  if len(grid) < 3:
    return y
  yi = np.interp(grid, t, y)
  n = max(int(win_s / (grid[1] - grid[0])), 1)
  k = np.ones(n) / n
  sm = np.convolve(yi, k, mode="same")
  return np.interp(t, grid, sm)


def read_flight(path):
  """t, total command, and mean |dq| over the upper body, from one Run's flight recorder."""
  cols = {}
  with open(path) as fh:
    r = csv.reader(fh)
    hdr = next(r)
    idx = {n: i for i, n in enumerate(hdr)}
    want = ["t", "cmd_vx", "cmd_vy", "cmd_wz"] + [f"meas_dq{j}" for j in UB_SLOTS]
    missing = [w for w in want if w not in idx]
    if missing:
      raise SystemExit(f"{Path(path).name}: missing {missing[:4]}")
    take = [idx[w] for w in want]
    rows = []
    for row in r:
      try:
        rows.append([float(row[i]) for i in take])
      except (ValueError, IndexError):
        continue
  a = np.asarray(rows)
  cols["t"] = a[:, 0]
  # Same gate form as walking_mask / rewards.py: the 2-norm of the linear pair plus |wz|,
  # deliberately not a 3-vector norm.
  cols["cmd"] = np.linalg.norm(a[:, 1:3], axis=1) + np.abs(a[:, 3])
  cols["arm"] = np.abs(a[:, 4:]).mean(axis=1)
  return cols


def read_mocap(bundle):
  p = Path(bundle) / "mocap_aligned.csv"
  if not p.exists():
    return None
  d = np.genfromtxt(p, delimiter=",", names=True)
  return {"t": np.atleast_1d(d["t"]),
          "speed": np.hypot(np.atleast_1d(d["gt_vx"]), np.atleast_1d(d["gt_vy"]))}


def standing_segments(t, cmd):
  """[(i0, i1)] of contiguous runs with command at or below the threshold."""
  still = cmd <= CMD_THRESHOLD
  segs, i = [], 0
  while i < len(still):
    if not still[i]:
      i += 1
      continue
    j = i
    while j + 1 < len(still) and still[j + 1]:
      j += 1
    segs.append((i, j))
    i = j + 1
  return segs


def settled_floor(t, sig, segs):
  """Median of the signal over the late tail of every LONG standing segment.

  Measured, not assumed, and pooled over segments so one quiet moment cannot set it. This is
  the number a stillness threshold has to be interpreted against.
  """
  tail = []
  for i0, i1 in segs:
    if t[i1] - t[i0] < FLOOR_MIN_SEG_S:
      continue
    m = t[i0:i1 + 1] >= t[i1] - FLOOR_TAIL_S
    tail.append(sig[i0:i1 + 1][m])
  return float(np.median(np.concatenate(tail))) if tail else float("nan")


def settling_times(t, sig, segs, thresh):
  """Per standing segment: seconds from the command dropping to sustained sub-threshold.

  ⚠ Only segments of at least MIN_SETTLE_SEG_S are used. A 3 s standing gap between two
  walking bouts cannot show a 5 s settle, so including short segments makes settling time a
  measurement of how the operator drove the stick rather than of the policy. The first pass
  of this analysis did include them and every run came back right-censored.

  A long segment that still never settles contributes its own length as a right-CENSORED
  observation and is counted, because dropping those would bias against exactly the policies
  this metric exists to detect.
  """
  out, censored = [], 0
  for i0, i1 in segs:
    ts, ys = t[i0:i1 + 1], sig[i0:i1 + 1]
    if ts[-1] - ts[0] < MIN_SETTLE_SEG_S:
      continue
    under = ys <= thresh
    got = None
    for k in range(len(ts)):
      if not under[k]:
        continue
      m = (ts >= ts[k]) & (ts <= ts[k] + HOLD_S)
      if ts[m][-1] - ts[k] >= HOLD_S * 0.9 and under[m].all():
        got = ts[k] - ts[0]
        break
    if got is None:
      censored += 1
      out.append(ts[-1] - ts[0])
    else:
      out.append(got)
  return out, censored


def cadence(bundle):
  """Commanded stride period by regime from the HRL telemetry, or a reason it is unavailable."""
  h = sorted(Path(bundle).glob("*_hrl.csv"))
  if not h:
    return {"available": False, "why": "no HRL telemetry (flat policy)"}
  P, C = [], []
  with open(h[0]) as fh:
    for row in csv.DictReader(fh):
      try:
        P.append(float(row["period"]))
        C.append(abs(float(row["cmd_vx"])) + abs(float(row["cmd_vy"])) + abs(float(row["cmd_wz"])))
      except (ValueError, KeyError, TypeError):
        continue
  P, C = np.asarray(P), np.asarray(C)
  if P.size == 0:
    return {"available": False, "why": "empty HRL telemetry"}
  if P.max() - P.min() < PIN_TOL:
    return {"available": False, "why": f"PINNED at {P[0]:.4f}s (pin_period set; the HL's "
                                       f"period was never decoded)", "pinned_at": float(P[0])}
  w = C > CMD_THRESHOLD
  lo, hi = PERIOD_RANGE
  out = {"available": True, "n": int(P.size)}
  for name, m in (("standing", ~w), ("walking", w)):
    if not m.any():
      out[name] = None
      continue
    p = P[m]
    out[name] = {
      "mean": float(p.mean()), "std": float(p.std()),
      "p5": float(np.percentile(p, 5)), "p95": float(np.percentile(p, 95)),
      # How much of the time the HL is asking for the most extreme cadence it can.
      "frac_at_ceiling": float((p > hi - 0.01).mean()),
      "frac_at_floor": float((p < lo + 0.01).mean()),
      "n": int(p.size),
    }
  return out


def analyze(bundle):
  b = Path(bundle)
  # A bundle IS a run.json. Sibling directories under logs/robot_logs/ that match the date
  # glob but carry no manifest are not Runs (e.g. parked motion capture for an aborted
  # attempt) and must be skipped, not crashed on.
  if not (b / "run.json").exists():
    return None
  run = json.loads((b / "run.json").read_text())
  flight = sorted(p for p in b.glob("2026-*.csv") if not p.name.endswith("_hrl.csv"))
  if not flight:
    return None
  F = read_flight(flight[0])
  segs = standing_segments(F["t"], F["cmd"])
  arm = moving_average(F["t"], F["arm"], SMOOTH_S)

  res = {"bundle": b.name, "policy": run["policy_tag"], "note": run.get("note"),
         "n_standing_segments": len(segs),
         "standing_s": float(sum(F["t"][j] - F["t"][i] for i, j in segs)),
         "cadence": cadence(b), "arm": {}, "base": {}}

  res["arm"]["floor_rad_s"] = settled_floor(F["t"], arm, segs)
  for th in ARM_THRESHOLDS:
    v, cen = settling_times(F["t"], arm, segs, th)
    res["arm"][f"t_settle_{th}"] = {
      "median": float(np.median(v)) if v else None,
      "max": float(np.max(v)) if v else None,
      "n": len(v), "censored": cen}

  M = read_mocap(b)
  if M is None:
    res["base"] = {"available": False,
                   "why": "no mocap_aligned.csv; the robot has no true base speed"}
  else:
    sp = moving_average(M["t"], M["speed"], SMOOTH_S)
    sp_on_f = np.interp(F["t"], M["t"], sp)
    res["base"]["floor_m_s"] = settled_floor(F["t"], sp_on_f, segs)
    for th in BASE_THRESHOLDS:
      v, cen = settling_times(F["t"], sp_on_f, segs, th)
      res["base"][f"t_settle_{th}"] = {
        "median": float(np.median(v)) if v else None,
        "max": float(np.max(v)) if v else None,
        "n": len(v), "censored": cen}
  return res


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--bundles", nargs="*", default=None)
  ap.add_argument("--out", default="data/2026-09-14-hardware-session/cadence_settling.json")
  args = ap.parse_args()
  bundles = args.bundles or sorted(glob.glob("logs/robot_logs/2026_09_14-*/"))

  rows = [r for r in (analyze(b) for b in bundles) if r]

  print("COMMANDED CADENCE  (s; range %.2f-%.2f)" % PERIOD_RANGE)
  print("%-30s %-20s %8s %8s %7s | %8s %8s" %
        ("run", "policy", "stand", "std", "ceil%", "walk", "std"))
  for r in rows:
    c = r["cadence"]
    if not c.get("available"):
      print("%-30s %-20s   %s" % (r["bundle"][11:], r["policy"], c["why"]))
      continue
    s, w = c.get("standing"), c.get("walking")
    print("%-30s %-20s %8.3f %8.3f %6.0f%% | %8.3f %8.3f" % (
      r["bundle"][11:], r["policy"],
      s["mean"] if s else float("nan"), s["std"] if s else float("nan"),
      100 * s["frac_at_ceiling"] if s else float("nan"),
      w["mean"] if w else float("nan"), w["std"] if w else float("nan")))

  print("\nARM-VELOCITY SETTLING  (s to stay under threshold for %.1fs; * = right-censored)"
        % HOLD_S)
  print("%-30s %-20s %9s %4s %s" % ("run", "policy", "floor", "set/n", "".join(
    "%11s" % f"th={t}" for t in ARM_THRESHOLDS)))
  for r in rows:
    cells = ""
    for th in ARM_THRESHOLDS:
      d = r["arm"][f"t_settle_{th}"]
      cells += "%11s" % ("--" if d["median"] is None else
                         f"{d['median']:.1f}{'*' if d['censored'] else ''}")
    n = r["arm"][f"t_settle_{ARM_THRESHOLDS[0]}"]["n"]
    cen = r["arm"][f"t_settle_{ARM_THRESHOLDS[0]}"]["censored"]
    print("%-30s %-20s %9.4f %4s %s" % (r["bundle"][11:], r["policy"],
                                        r["arm"]["floor_rad_s"], f"{n-cen}/{n}", cells))

  if all(not r["base"].get("floor_m_s") for r in rows):
    print("\nBASE-SPEED SETTLING: unavailable (no mocap_aligned.csv in any bundle yet)")

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(rows, indent=2) + "\n")
  print(f"\n[WROTE] {out}")


if __name__ == "__main__":
  main()
