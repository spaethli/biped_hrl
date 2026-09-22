#!/usr/bin/env python3
"""Thesis figures for the chapter on what the simulation benchmark cannot see.

Separate from `plot_thesis_figures.py` because the data domain is different: that script
draws hardware _Run_ bundles, these draw ADR result tables and sim-bench logs. The OUTPUT
convention is shared, not re-invented -- `save_pgf` and `write_panel_csv` are imported, so
every figure in the thesis emits the same `.pgf` + per-panel `.csv` pair and inherits the
same rcParams and policy palette.

  f_blind_v3      ADR-0009. Every pre-registered metric IMPROVED under Model v3 while the
                  commanded ankle roll range -- the joint that rejects lateral pushes --
                  collapsed to ~0.45x, in sim and on hardware, and the robot could not walk.
  f_blind_scale   The goal-scale eval artifact. The SAME unchanged checkpoint, before and
                  after the harness fix. Nothing about the policy changed.
  f_blind_band    Training replicate spread at identical config AND seed, by regime, with
                  the seed-to-seed spread and the re-measurement floor overlaid to show
                  that neither is the dominant term.
  f_blind_control The matched control. Standing `act_legs` crosses parity once the A0
                  anchor trains at the same standing fraction, while `ajit_legs_p95` does
                  not move -- so one gap was confounded and the other is architectural.

PROVENANCE. `f_blind_control` is computed from `[BENCH]` lines in
`data/2026-09-21-rse-matched-baseline/*.log`. The other three carry their numbers as
literals, because their sources (the 2026-08-27 v3 hardware sessions, the WL-C probe, the
2026-08-27 replicate pairs) are 115-470 MB captures that are not in git. Every literal
block names the document it came from, and the panel CSVs are the archival record, the same
contract `tests/test_stand_attitude.py` uses for its golden values.

Usage:
  python scripts/plot_blindness_figures.py                      # all four
  python scripts/plot_blindness_figures.py --only v3 control
  python scripts/plot_blindness_figures.py --out figures/benchmark_blindness
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

# Reuse the thesis figure contract (pgf backend, rcParams, palette, emission helpers)
# rather than defining a second one. Importing is safe: that module's work is under a
# `__main__` guard and module level only sets constants and rcParams.
from plot_thesis_figures import (CHAIN_COLOR, FIG_W, bottom_legend,  # noqa: E402
                                 save_pgf, write_panel_csv)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

OUT_DEFAULT = "figures/benchmark_blindness"
MATCHED_GLOB = "data/2026-09-21-rse-matched-baseline/*stand_pay0.log"

GREY = "#5A5A5A"
BAD = "#D55E00"      # the quantity that was lost
GOOD = "#0072B2"     # the quantities that looked fine
PARITY = "#999999"


# --------------------------------------------------------------------- f_blind_v3

# docs/adr/0009-revert-model-v3-action-clip.md, "Evidence" + the bench-cannot-see bullet.
# Pre-registered metrics: sim, 600 steps x 64 envs x 2 seeds. fall_rate was 0 for BOTH, so
# it has no ratio and is annotated instead of drawn -- a 0/0 bar would be an invented number.
V3_PREREG = [                     # (metric, v2, v3)  lower is better in all four
  ("err_vx", 0.0941, 0.0903),
  ("act_legs", 0.600, 0.567),
  ("jacc_legs_p95", 78.6, 62.1),
]
V3_ROLL = [                       # (plant, side, v2 span, v3 span)
  ("sim", "L", 0.345, 0.154),
  ("sim", "R", 0.331, 0.141),
  ("hardware", "L", (0.413 + 0.436) / 2, 0.192),
  ("hardware", "R", (0.505 + 0.373) / 2, 0.223),
]


def fig_v3(out_dir, name="f_blind_v3"):
  fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.7), sharey=True)
  rows = []

  ax = axes[0]
  labels = [m for m, _, _ in V3_PREREG]
  ratios = [b / a for _, a, b in V3_PREREG]
  ax.bar(range(len(ratios)), ratios, color=GOOD, width=0.55)
  for i, (lab, v2, v3) in enumerate(V3_PREREG):
    ax.text(i, ratios[i] + 0.03, f"{ratios[i]:.2f}", ha="center", fontsize=6)
    rows.append(dict(panel="prereg", metric=lab, v2=v2, v3=v3, ratio=round(v3 / v2, 4)))
  ax.set_xticks(range(len(labels)))
  ax.set_xticklabels(labels, fontsize=6.5)
  ax.set_title("pre-registered metrics\n(all improved under v3)", fontsize=7)
  ax.set_ylabel("v3 / v2   (lower = better)", fontsize=7)

  ax = axes[1]
  rr = [v3 / v2 for _, _, v2, v3 in V3_ROLL]
  xs = range(len(rr))
  ax.bar(xs, rr, color=BAD, width=0.55)
  for i, ((plant, side, v2, v3), r) in enumerate(zip(V3_ROLL, rr)):
    ax.text(i, r + 0.03, f"{r:.2f}", ha="center", fontsize=6)
    rows.append(dict(panel="ankle_roll", metric=f"{plant}_{side}", v2=round(v2, 4),
                     v3=v3, ratio=round(r, 4)))
  ax.set_xticks(list(xs))
  ax.set_xticklabels([f"{p}\n{s}" for p, s, _, _ in V3_ROLL], fontsize=6.5)
  ax.set_title("commanded ankle roll range\n(the reserve that was lost)", fontsize=7)

  for ax in axes:
    ax.axhline(1.0, color=PARITY, lw=0.8, ls="--", zorder=0)
    ax.set_ylim(0, 1.25)
    ax.tick_params(labelsize=6.5)
  fig.suptitle("Model v3: every scored metric improved; the robot could not walk",
               fontsize=8)
  fig.tight_layout(rect=(0, 0.07, 1, 0.93))
  # figure-level, not in-axes: at these ratios every in-axes position is behind a bar.
  fig.text(0.5, 0.015, "fall_rate 0 for both, so it has no ratio and is not drawn",
           ha="center", fontsize=6, style="italic", color=GREY)
  write_panel_csv(out_dir, name, "ratios", ["panel", "metric", "v2", "v3", "ratio"], rows)
  return save_pgf(fig, out_dir, name)


# --------------------------------------------------------------------- f_blind_scale

# A1 findings ledger (research KB), WL-C row, 2026-07-16. SAME checkpoint either side: the
# fix is to where the goal scale is READ, not to the policy.
SCALE_ROWS = [                    # (label, before, after, lower_is_better)
  ("hold ss_err_vx @0.5", 0.304, 0.044, True),
  ("hold ss_err_vx @1.0", 0.882, 0.097, True),
  ("hl_err_vx", 0.912, 0.132, True),
  ("|g|_vx saturation", 0.956, 0.046, True),   # fraction of steps at the bound
]


def fig_scale(out_dir, name="f_blind_scale"):
  fig, ax = plt.subplots(figsize=(FIG_W, 2.5))
  rows = []
  ys = np.arange(len(SCALE_ROWS))
  for y, (lab, before, after, _) in zip(ys, SCALE_ROWS):
    ax.plot([before, after], [y, y], color=GREY, lw=1.0, zorder=1)
    ax.scatter([before], [y], s=34, color=BAD, zorder=2,
               label="harness artifact present" if y == 0 else None)
    ax.scatter([after], [y], s=34, color=GOOD, zorder=2,
               label="scale read from the checkpoint" if y == 0 else None)
    ax.annotate(f"{before / after:.1f}x", xy=(np.sqrt(before * after), y + 0.22),
                ha="center", fontsize=6, color=GREY)
    rows.append(dict(metric=lab, before=before, after=after,
                     factor=round(before / after, 3)))
  ax.set_yticks(ys)
  ax.set_yticklabels([r[0] for r in SCALE_ROWS], fontsize=6.5)
  ax.set_xscale("log")
  ax.set_xlabel("value (log scale); lower is better on every row", fontsize=7)
  ax.tick_params(labelsize=6.5)
  ax.set_ylim(-0.6, len(SCALE_ROWS) - 0.2)
  ax.set_title("The same unchanged checkpoint, before and after the harness fix",
               fontsize=8)
  h, _ = ax.get_legend_handles_labels()
  bottom_legend(fig, h, ncol=2, height=0.15)
  fig.text(0.5, 0.155, "direction split 38 fwd / 19 bwd  ->  64 / 64 fwd",
           ha="center", fontsize=6, style="italic", color=GREY)
  write_panel_csv(out_dir, name, "beforeafter", ["metric", "before", "after", "factor"], rows)
  return save_pgf(fig, out_dir, name)


# --------------------------------------------------------------------- f_blind_band

# `standing_metrics_replicate_spread` memory, 2026-08-27/28. Two accidental
# config-identical, SAME-SEED-42 replicate pairs. Values are max/min ratios.
BAND = {
  "walking": [("act_legs", 1.08, 1.10), ("ajit", 1.02, 1.31), ("jacc_legs", 1.03, 1.08),
              ("aggregate err_vx", 1.12, 1.94), ("ss_err_vx (hold)", 1.38, 3.32)],
  "standing": [("act_legs", 1.21, 2.23), ("ajit", 1.19, 4.15), ("ajit_p95", 1.00, 5.16),
               ("jacc_legs", 1.74, 2.43), ("touchdowns", 2.30, 4.22), ("|g|vy", 1.37, 3.57)],
}
SEED_SPREAD = (1.40, 2.68)     # standing ajit across seeds 42/123, rse 0.20 and 0.15
MEASURE_FLOOR = (1.016, 1.042)  # SAME checkpoint re-scored


def fig_band(out_dir, name="f_blind_band"):
  fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
  rows, y, ticks, labels = [], 0, [], []
  for regime in ("walking", "standing"):
    colour = GOOD if regime == "walking" else BAD
    for lab, lo, hi in BAND[regime]:
      ax.plot([lo, hi], [y, y], color=colour, lw=2.6, solid_capstyle="round",
              label=regime if lab == BAND[regime][0][0] else None, zorder=3)
      ticks.append(y)
      labels.append(f"{lab}")
      rows.append(dict(regime=regime, metric=lab, lo=lo, hi=hi))
      y -= 1
    y -= 0.6
  ax.axvspan(*MEASURE_FLOOR, color=PARITY, alpha=0.5, zorder=0)
  ax.axvspan(*SEED_SPREAD, color="#F0C674", alpha=0.35, zorder=0)
  ax.axvline(1.0, color="#333333", lw=0.8, zorder=1)
  ax.set_yticks(ticks)
  ax.set_yticklabels(labels, fontsize=6.5)
  ax.set_xscale("log")
  ax.set_xlim(0.95, 6.0)
  ax.set_xticks([1, 1.5, 2, 3, 4, 5])
  ax.set_xticklabels(["1x", "1.5x", "2x", "3x", "4x", "5x"], fontsize=6.5)
  ax.xaxis.set_minor_formatter(NullFormatter())   # else the log locator adds "6x10^0"
  ax.set_xlabel("max/min across two replicates at IDENTICAL config and seed", fontsize=7)
  ax.set_title("What a repeated training run alone can move", fontsize=8)
  # blended transform: x in data (the ratio axis), y in axes fraction -- the y axis here
  # is a metric INDEX, so a data-space y put these labels on top of the title.
  bl = ax.get_xaxis_transform()
  ax.text(np.sqrt(MEASURE_FLOOR[0] * MEASURE_FLOOR[1]), 0.02, "re-measurement floor",
          transform=bl, rotation=90, ha="center", va="bottom", fontsize=5.4, color=GREY)
  ax.text(np.sqrt(SEED_SPREAD[0] * SEED_SPREAD[1]), 0.97, "seed 42 vs 123", transform=bl,
          ha="center", va="top", fontsize=5.8, color="#9A7B1F")
  h, _ = ax.get_legend_handles_labels()
  bottom_legend(fig, h, ncol=2, height=0.16)
  for panel, regime in (("walking", "walking"), ("standing", "standing")):
    write_panel_csv(out_dir, name, panel, ["regime", "metric", "lo", "hi"],
                    [r for r in rows if r["regime"] == regime])
  return save_pgf(fig, out_dir, name)


# --------------------------------------------------------------------- f_blind_control

def load_matched(repo=REPO):
  """Standing `[BENCH]` scalars per arm from the 2026-09-21 matched-baseline sweep."""
  out = {}
  for path in sorted(repo.glob(MATCHED_GLOB)):
    arm = path.name.split("__")[0]
    hits = re.findall(r"\[BENCH\] (\{.*\})", path.read_text())
    if not hits:
      continue
    out[arm] = json.loads(hits[-1].replace("NaN", "null"))
  return out


# Which A0 each A1a arm is scored against. The point of the figure is that this CHOICE,
# not the policy, decides whether A1a "wins" on act_legs.
CONTROL = {"unmatched (rse 0.05)": {"s42": "a0_rse05_s42", "s123": "a0_rse05_s42"},
           "matched (rse 0.12)": {"s42": "a0_rse12_s42", "s123": "a0_rse12_s123"}}
CTRL_METRICS = [("act_legs", "standing act_legs"), ("ajit_legs_p95", "standing ajit_p95")]


def fig_control(out_dir, name="f_blind_control"):
  bench = load_matched()
  missing = [a for m in CONTROL.values() for a in m.values() if a not in bench]
  if missing:
    raise SystemExit(f"matched-baseline logs missing arms: {sorted(set(missing))}")
  fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.8))
  rows = []
  for ax, (key, title) in zip(axes, CTRL_METRICS):
    for i, (ctrl_label, per_seed) in enumerate(CONTROL.items()):
      vals = []
      for seed in ("s42", "s123"):
        arm = f"a1a_rse12_{seed}"
        if arm not in bench:
          continue
        r = bench[arm][key] / bench[per_seed[seed]][key]
        vals.append(r)
        rows.append(dict(metric=key, control=ctrl_label, seed=seed, arm=arm,
                         a1a=bench[arm][key], a0=bench[per_seed[seed]][key],
                         ratio=round(r, 4)))
      ax.bar(i, float(np.mean(vals)), width=0.5,
             color=(BAD if key == "act_legs" else GOOD), alpha=0.75, zorder=2)
      ax.scatter([i] * len(vals), vals, s=22, color="#222222", zorder=3,
                 label="per seed" if i == 0 and key == "act_legs" else None)
      # offset in POINTS above the highest per-seed dot: the dots straddle the mean, so a
      # data-space position lands on one of them at any ratio.
      ax.annotate(f"{np.mean(vals):.2f}x", xy=(i, max(vals + [float(np.mean(vals))])),
                  xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                  fontsize=6.5, fontweight="bold")
    ax.axhline(1.0, color=PARITY, lw=0.9, ls="--", zorder=1)
    ax.set_xticks(range(len(CONTROL)))
    ax.set_xticklabels(list(CONTROL), fontsize=6.5)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(labelsize=6.5)
  axes[0].set_ylabel("A1a / A0", fontsize=7)
  axes[0].set_ylim(0, 1.55)
  axes[1].set_ylim(0, max(r["ratio"] for r in rows if r["metric"] == "ajit_legs_p95") * 1.25)
  for ax, note in zip(axes, ("crosses parity when matched", "unmoved: architectural")):
    ax.text(0.5, 0.93, note, transform=ax.transAxes, ha="center", va="top", fontsize=6,
            style="italic", color=GREY)
  fig.suptitle("Matching the control on the axis the claim names", fontsize=8)
  h, _ = axes[0].get_legend_handles_labels()
  bottom_legend(fig, h, ncol=1, height=0.14)
  fig.subplots_adjust(top=0.84)
  write_panel_csv(out_dir, name, "ratios",
                  ["metric", "control", "seed", "arm", "a1a", "a0", "ratio"], rows)
  return save_pgf(fig, out_dir, name)


FIGS = {"v3": fig_v3, "scale": fig_scale, "band": fig_band, "control": fig_control}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", default=OUT_DEFAULT, help=f"output dir (default {OUT_DEFAULT})")
  ap.add_argument("--only", nargs="+", choices=sorted(FIGS), help="subset of figures")
  a = ap.parse_args()
  out_dir = (REPO / a.out) if not Path(a.out).is_absolute() else Path(a.out)
  for key in (a.only or list(FIGS)):
    FIGS[key](out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
