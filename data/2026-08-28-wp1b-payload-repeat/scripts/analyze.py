#!/usr/bin/env python3
"""WP1b analysis -- bowl depth vs MEASURED noise floor, T* with bootstrap
uncertainty, k-sensitivity on copper loss, objective comparison table.

Rules that fix WP1's fit artifact:
  * uniform 0.05 T grid -> a local quadratic vertex is unbiased. The fit uses a
    LOCAL window (grid-argmin +/- 2 points, uniform), never the whole grid, so a
    steep far arm cannot pull the vertex.
  * the noise unit is an independent play.py PROCESS, not an --eval-seeds index
    (step-1 probe B: internal seeds understate run-to-run spread ~8x). Per-T
    SEM = std(repeats)/sqrt(R).
  * "local bracket depth" = spread of the 3 points centred on the grid argmin,
    as % of the min -- the exact quantity WP1's CORRECTION scored against the
    floor. T* is reported ONLY when that depth > 2 * pooled-per-T SEM (%).
    Otherwise: "not-resolvable" (a valid outcome, not a failure).
  * cot_copper(k) = cot_mech + (k/0.3)*(cot_copper_0.3 - cot_mech)  -- exact, the
    copper term k*tau^2 is linear in k; recomputed from the two logged metrics,
    no re-run.

Usage: analyze.py [sweep.csv [sweep_ext.csv ...]]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/iams/ramlab_ws/src/unitree_rl_mjlab")
OUT_DIR = REPO / "data/2026-08-28-wp1b-payload-repeat"

T_GRID = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
PAYLOADS = [0, 4, 8, 12]
VXS = [0.5, 0.6, 0.7]
KS = [0.1, 0.3, 1.0]
NBOOT = 3000
RNG = np.random.default_rng(0)


def cot_copper_k(df, k):
  return df["cot"] + (k / 0.3) * (df["cot_copper"] - df["cot"])


def cell_curve(df, payload, vx, metric):
  sub = df[(df.payload_kg == payload) & (df.vx_pinned == vx)]
  Ts, means, sems, raw = [], [], [], []
  for T in T_GRID:
    v = sub[np.isclose(sub.T_pinned, T)][metric].to_numpy()
    if len(v) == 0:
      continue
    Ts.append(float(T))
    means.append(v.mean())
    sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
    raw.append(v)
  return np.array(Ts), np.array(means), np.array(sems), raw


def quad_vertex(Ts, ys):
  a, b, _ = np.polyfit(Ts, ys, 2)
  if a <= 0:
    return np.nan, False
  return -b / (2 * a), True


def local_window(Ts, i, half=2):
  """Indices of the uniform window Ts[i] +/- half, contiguous & evenly spaced."""
  lo = max(0, i - half)
  hi = min(len(Ts), i + half + 1)
  idx = list(range(lo, hi))
  # keep only the run of uniform 0.05 spacing through i
  d = np.diff(Ts[idx])
  if len(d) and not np.allclose(d, d[0]):
    # trim to the largest uniform sub-run containing i
    idx = [j for j in idx if abs(Ts[j] - Ts[i]) <= half * 0.05 + 1e-9]
  return idx


def analyse_metric(df, metric, label, want_min=True):
  print(f"\n{'='*80}\n{label}\n{'='*80}")
  rows = []
  for vx in VXS:
    for payload in PAYLOADS:
      Ts, means, sems, raw = cell_curve(df, payload, vx, metric)
      if len(Ts) < 3:
        print(f"  vx={vx} p={payload:2d}: <3 T points, skip")
        continue
      i = int(np.argmin(means)) if want_min else int(np.argmax(means))
      ymin = means[i]
      argmin_T = Ts[i]
      interior = 0 < i < len(Ts) - 1

      # local bracket depth (3 points centred on argmin) -- the WP1-CORRECTION quantity
      if interior:
        br = [means[i - 1], means[i], means[i + 1]]
        loc_depth_pct = 100 * (max(br) - min(br)) / abs(ymin)
      else:
        loc_depth_pct = np.nan
      full_depth_pct = 100 * (means.max() - means.min()) / abs(ymin)
      pooled_sem_pct = 100 * np.nanmean(sems) / abs(ymin)
      floor2 = 2 * pooled_sem_pct
      resolvable = interior and (loc_depth_pct > floor2)

      tstar = tstar_lo = tstar_hi = np.nan
      fit_pts = None
      status = "not-resolvable"
      if not interior:
        status = f"argmin@edge({argmin_T:.2f})"
      elif not resolvable:
        status = "not-resolvable"
      else:
        idx = local_window(Ts, i, half=2)
        if len(idx) < 3:
          status = "window<3"
        else:
          Tw = Ts[idx]
          fit_pts = len(idx)
          tv, cx = quad_vertex(Tw, means[idx])
          if cx and Tw[0] - 1e-9 <= tv <= Tw[-1] + 1e-9:
            verts = []
            for _ in range(NBOOT):
              ys_b = np.array([RNG.choice(raw[j], size=len(raw[j])).mean() for j in idx])
              v, c2 = quad_vertex(Tw, ys_b)
              if c2 and Tw[0] - 1e-9 <= v <= Tw[-1] + 1e-9:
                verts.append(v)
            if len(verts) > NBOOT * 0.5:
              tstar = float(np.median(verts))
              tstar_lo, tstar_hi = np.percentile(verts, [16, 84])
              status = f"fit({fit_pts}pt)"
            else:
              status = "vertex-unstable"
          else:
            status = "vertex-out-of-window" if cx else "not-convex"

      rows.append(dict(
        vx=vx, payload=payload, argmin_T=round(argmin_T, 3), y_min=ymin,
        loc_depth_pct=None if np.isnan(loc_depth_pct) else round(loc_depth_pct, 3),
        full_depth_pct=round(full_depth_pct, 3),
        floor_2sem_pct=round(floor2, 3),
        depth_over_floor=None if np.isnan(loc_depth_pct) else round(loc_depth_pct / floor2, 2),
        Tstar=None if np.isnan(tstar) else round(tstar, 4),
        Tstar_lo=None if np.isnan(tstar_lo) else round(tstar_lo, 4),
        Tstar_hi=None if np.isnan(tstar_hi) else round(tstar_hi, 4),
        status=status))
      ld = "  n/a" if np.isnan(loc_depth_pct) else f"{loc_depth_pct:5.2f}%"
      ts = "n/a" if np.isnan(tstar) else f"{tstar:.4f} [{tstar_lo:.4f},{tstar_hi:.4f}]"
      print(f"  vx={vx} p={payload:2d}kg | argmin={argmin_T:.2f} y={ymin:.6g} | "
            f"loc_depth={ld} full={full_depth_pct:6.2f}% floor(2SEM)={floor2:5.2f}% "
            f"| {status:18s} T*={ts}")
  return pd.DataFrame(rows)


def payload_trend(sub):
  f = sub[sub.status.str.startswith("fit")].sort_values("payload")
  if len(f) < 3:
    return f"only {len(f)}/4 payloads give a resolvable interior T* -> inconclusive"
  ts = f.Tstar.to_numpy()
  ci = (f.Tstar_hi.to_numpy() - f.Tstar_lo.to_numpy()) / 2
  span = ts[-1] - ts[0]
  diffs = np.diff(ts)
  mono = np.all(diffs >= -ci[1:]) or np.all(diffs <= ci[1:])
  typ_ci = float(np.nanmean(ci))
  return (f"{len(f)}/4 resolvable; T* {ts[0]:.3f}->{ts[-1]:.3f} "
          f"(dT*={span:+.4f} over {int(f.payload.iloc[-1]-f.payload.iloc[0])}kg), "
          f"typ CI +/-{typ_ci:.4f} -> {'MONOTONIC' if mono else 'non-monotonic'}, "
          f"|span|/CI={abs(span)/typ_ci:.1f}x")


def main():
  paths = [Path(p) for p in sys.argv[1:]] or [OUT_DIR / "wp1b_sweep.csv",
                                              OUT_DIR / "wp1b_sweep_ext.csv"]
  paths = [p for p in paths if p.exists()]
  if not paths:
    sys.exit("no sweep CSV yet")
  df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
  n_per = df.groupby(["T_pinned", "payload_kg", "vx_pinned"]).size()
  print(f"sources: {[p.name for p in paths]}")
  print(f"{len(df)} rows | cells {len(n_per)} | repeats/cell "
        f"min={n_per.min()} max={n_per.max()} | eval_steps={sorted(df.eval_steps.unique())}")
  print(f"T grid present: {sorted(df.T_pinned.unique())}")

  # ---- level response: payload IS a latent ----
  print(f"\n{'='*80}\nLEVEL RESPONSE TO PAYLOAD  (mean over the {0.40}-{0.60} core T grid)\n{'='*80}")
  core = df[df.T_pinned <= 0.601]
  lvl = []
  for vx in VXS:
    for m in ["mech_power_w", "mech_power_copper_w", "cot", "cot_copper", "err_vx", "ss_vx_var"]:
      s = core[core.vx_pinned == vx]
      m0, m12 = s[s.payload_kg == 0][m].mean(), s[s.payload_kg == 12][m].mean()
      lvl.append(dict(vx=vx, metric=m, at0=m0, at12=m12, pct_rise=round(100*(m12-m0)/abs(m0), 2)))
      print(f"  vx={vx} {m:22s} 0kg={m0:.6g}  12kg={m12:.6g}  {100*(m12-m0)/abs(m0):+.1f}%")
  pd.DataFrame(lvl).to_csv(OUT_DIR / "wp1b_level_response.csv", index=False)

  res = {}
  res["cot"] = analyse_metric(df, "cot", "OBJ 1: mechanical CoT  Sum|tau*qdot| / (m g d)")
  res["cot"].to_csv(OUT_DIR / "wp1b_tstar_cot.csv", index=False)

  for k in KS:
    df[f"cc_k{k}"] = cot_copper_k(df, k)
    d = analyse_metric(df, f"cc_k{k}", f"OBJ 2: copper-loss CoT k={k}  Sum(|tau*qdot| + {k}*tau^2)")
    d.to_csv(OUT_DIR / f"wp1b_tstar_cot_copper_k{k}.csv", index=False)
    res[f"cc_k{k}"] = d

  res["err_vx"] = analyse_metric(df, "err_vx", "OBJ 3: tracking error err_vx")
  res["err_vx"].to_csv(OUT_DIR / "wp1b_tstar_err_vx.csv", index=False)
  res["ss_vx_var"] = analyse_metric(df, "ss_vx_var", "OBJ 4: cross-env vx variance ss_vx_var (F5 analogue)")
  res["ss_vx_var"].to_csv(OUT_DIR / "wp1b_tstar_ss_vx_var.csv", index=False)

  # ---- k-sensitivity: does the 0/4-edge -> 8/12-interior regime shift survive? ----
  print(f"\n{'='*80}\nSTEP 5 -- k-SENSITIVITY  (copper-loss CoT argmin_T / status, vx=0.5)\n{'='*80}")
  print(f"  {'k':>5} | " + " | ".join(f"{'p='+str(p)+'kg':>22s}" for p in PAYLOADS))
  for k in KS:
    d5 = res[f"cc_k{k}"]
    d5 = d5[d5.vx == 0.5].set_index("payload")
    cells = []
    for p in PAYLOADS:
      if p in d5.index:
        r = d5.loc[p]
        ts = "" if r.Tstar is None or (isinstance(r.Tstar, float) and np.isnan(r.Tstar)) else f" T*={r.Tstar}"
        cells.append(f"{r.argmin_T:.2f}/{r.status}{ts}")
      else:
        cells.append("--")
    print(f"  {k:>5} | " + " | ".join(f"{c:>22s}" for c in cells))

  # ---- step 6: objective comparison ----
  print(f"\n{'='*80}\nSTEP 6 -- OBJECTIVE COMPARISON  (report only, no recommendation)\n{'='*80}")
  for name, d in [("mech cot", res["cot"]), ("cot_copper k=0.3", res["cc_k0.3"]),
                  ("err_vx", res["err_vx"]), ("ss_vx_var", res["ss_vx_var"])]:
    print(f"\n  {name}")
    for vx in VXS:
      sub = d[d.vx == vx]
      nfit = int(sub.status.str.startswith("fit").sum())
      nres = int((~sub.status.isin(["not-resolvable"]) & ~sub.status.str.startswith("argmin@edge")).sum())
      print(f"    vx={vx}: interior T* resolvable in {nfit}/4 payloads "
            f"(any real structure {nres}/4) | {payload_trend(sub)}")

  print(f"\nCSVs -> {OUT_DIR}/")


if __name__ == "__main__":
  main()
