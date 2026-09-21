#!/usr/bin/env python3
"""Score quiet-stand torso attitude from flight recorders, WITH the stillness control that
`docs/adr/0006` makes mandatory for any cross-policy lean comparison.

WHY THIS EXISTS. Stand pitch looks like a policy property and is not. ADR-0006's 2026-08-03
read found stillness predicts the lean at r=-0.96 across three A0 variants: a policy that
fidgets settles into a less-drooped stance and reads less lean, regardless of calibration.
So a bare pitch mean ranks policies by how much they move, not by how they stand. The ADR's
rule is that stillness is reported alongside or the comparison is not made. This script is
that rule in code, so the same statistic is computed the same way across sessions.

THE STATISTIC, as ADR-0006's own table defines it:
  rows      zero-command (`|cmd_vx|,|cmd_vy|,|cmd_wz| < 0.05`) and `alpha == 0`, so the
            safety-hold blend can never contribute -- a held run reports the FILTER's
            posture, not the policy's (`safety_hold_causes_the_fall`).
  still     fraction of those rows with `max|dq|` over the 12 leg slots below 0.1 rad/s.
            Legs only: `hold_joint_ids` freezes waist+arms on hardware, so upper-body
            motion is not the policy's (`leg_only_action_rate`).
  pitch     `asin(2(wy - zx))` from the recorder's `quat_*`, i.e. the policy's OWN IMU
            observation, dq-filtered to the still rows. The IMU sits on `torso_link`
            (`doc/hrl/h1_2_kinematics.md`); this is torso attitude, not pelvis.

It also reads `joint_offset` out of each run's `_meta.json` rather than the deploy yaml,
because the yaml is the current file and the meta is what that run actually applied. The
two diverge whenever a config changed between sessions.

Usage:
  python scripts/score_stand_attitude.py logs/deploy_safety/2026-09-16_*.csv \
      --labels data/2026-09-16-hardware-session/labels.json
  python scripts/score_stand_attitude.py logs/deploy_safety/<stem>.csv --assume-hold ""
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score_joint_hold import LEG_SLOTS, load  # noqa: E402

CMD_STILL = 0.05    # zero-command gate, ADR-0006's value (the bench uses 0.1 for regime)
DQ_STILL = 0.1      # rad/s, `still` threshold on max|dq| over the legs
MIN_ROWS = 500      # below this a run cannot carry a mean; same floor score_joint_hold uses

# ADR-0006 2026-08-03, the three-A0-variant table the confound was measured on:
# (still %, dq-filtered stand pitch deg). Kept as data so the slope is refit, not hardcoded.
ADR0006_CONFOUND = np.array([[91.6, -6.13], [62.8, -3.97], [76.0, -5.51]])


def confound_slope():
  """deg of pitch per stillness-point, refit from ADR-0006's own table."""
  s, p = ADR0006_CONFOUND[:, 0], ADR0006_CONFOUND[:, 1]
  return float(np.polyfit(s, p, 1)[0]), float(np.corrcoef(s, p)[0, 1])


def pitch_rad(df):
  w, x, y, z = (df[f"quat_{k}"].values for k in "wxyz")
  return np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))


def score_entry(df, entry):
  """One FSM entry -> the ADR-0006 statistic, or None if it cannot carry a mean."""
  d = df[df.entry == entry] if "entry" in df.columns else df
  if len(d) == 0:
    return None
  cmd = np.abs(d[["cmd_vx", "cmd_vy", "cmd_wz"]].values).max(axis=1)
  dq = np.abs(d[[f"meas_dq{i}" for i in LEG_SLOTS]].values).max(axis=1)
  z = (cmd < CMD_STILL) & (d.alpha.values == 0)
  if z.sum() < MIN_ROWS:
    return None
  still = dq[z] < DQ_STILL
  p = np.degrees(pitch_rad(d)[z])
  return dict(n=int(z.sum()), still_pct=100 * float(still.mean()),
              mean_maxdq=float(dq[z].mean()), pitch_all_deg=float(p.mean()),
              pitch_deg=float(p[still].mean()) if still.sum() >= MIN_ROWS // 2 else np.nan,
              alpha_frac=float((d.alpha.values > 0).mean()))


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("csv", nargs="+", help="flight recorder csv(s), logs/deploy_safety/<stem>.csv")
  ap.add_argument("--labels", help="session labels.json mapping '<stem>:<entry>' -> {tag,...}")
  ap.add_argument("--assume-hold", default=None,
                  help="fallback hold_joint_ids for logs predating the meta key")
  a = ap.parse_args()

  labels = json.loads(Path(a.labels).read_text()) if a.labels else {}
  rows = []
  for path in sorted(a.csv):
    path = Path(path)
    if path.stem.endswith("_hrl") or "meta" in path.stem:
      continue
    df, _hold, _src = load(path, a.assume_hold)
    meta_path = path.with_name(path.stem + "_meta.json")
    off = json.loads(meta_path.read_text()).get("joint_offset") if meta_path.exists() else None
    entries = sorted(df.entry.unique()) if "entry" in df.columns else [0]
    for entry in entries:
      key = f"{path.stem}:{entry}"
      lab = labels.get(key, {})
      if lab.get("skip"):
        continue
      r = score_entry(df, entry)
      if r is None:
        continue
      # A run whose offset is non-zero was calibrated; one whose offset is all-zero was not.
      # Reported per run because the yaml is the current file, not the one that flew.
      r.update(run=key, tag=lab.get("tag", path.stem), note=lab.get("note", ""),
               # sum over all 27 slots, so it is 2x the per-leg chain the ADR quotes
               offset=("zero" if off is None or not any(off) else f"{sum(off):.3f} sum27"))
      rows.append(r)
  if not rows:
    raise SystemExit("no run carried enough zero-command alpha==0 rows to score")

  r = pd.DataFrame(rows)
  cols = ["run", "tag", "note", "n", "still_pct", "mean_maxdq", "pitch_all_deg", "pitch_deg", "offset"]
  pd.set_option("display.width", 220)
  print(r[cols].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

  g = r.groupby("tag").agg(runs=("n", "size"), still_pct=("still_pct", "mean"),
                           mean_maxdq=("mean_maxdq", "mean"), pitch_deg=("pitch_deg", "mean"),
                           offset=("offset", lambda s: "/".join(sorted(set(s)))))
  print(f"\n=== per-arm (|cmd|<{CMD_STILL}, alpha==0, dq-filtered) ===")
  print(g.sort_values("pitch_deg").to_string(float_format=lambda x: f"{x:7.3f}"))

  # The ADR-0006 gate: say how much of any ranking the stillness confound can account for.
  slope, r_adr = confound_slope()
  v = g.dropna(subset=["pitch_deg"])
  print(f"\n=== ADR-0006 stillness control ===")
  print(f"confound slope (refit from the 2026-08-03 table): {slope:+.4f} deg/pt, r={r_adr:+.3f}")
  if len(v) >= 3:
    r_here = float(np.corrcoef(v.still_pct, v.pitch_deg)[0, 1])
    print(f"this session, over {len(v)} arms:                     r={r_here:+.3f}")
  lo, hi = v.pitch_deg.idxmin(), v.pitch_deg.idxmax()
  gap = v.pitch_deg[hi] - v.pitch_deg[lo]
  if len(v) < 2 or gap == 0:
    print("single arm scored: no cross-policy comparison, so no confound to control for")
  else:
    pred = slope * (v.still_pct[hi] - v.still_pct[lo])
    print(f"widest pair: {lo} {v.pitch_deg[lo]:+.2f} deg (still {v.still_pct[lo]:.1f}%) vs "
          f"{hi} {v.pitch_deg[hi]:+.2f} deg (still {v.still_pct[hi]:.1f}%)")
    print(f"  observed gap {gap:+.2f} deg; stillness accounts for {pred:+.2f} deg; "
          f"UNEXPLAINED {gap - pred:+.2f} deg ({100 * abs((gap - pred) / gap):.0f}%)")
    if abs(pred) > 0.5 * abs(gap):
      print("  ⚠ the confound carries most of this gap -- do NOT report it as a policy effect")
  print("\n[STANDATT] " + json.dumps({k: {kk: (None if pd.isna(vv) else vv) for kk, vv in row.items()}
                                      for k, row in g.to_dict("index").items()}, default=str))


if __name__ == "__main__":
  main()
