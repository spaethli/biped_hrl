#!/usr/bin/env python3
"""Radar/spider chart comparing policies (runs/checkpoints) across `[BENCH]` metrics.

One point of view per figure: everything is normalized **relative to a baseline run**
(typically A0), which traces the regular polygon at radius 1.0. Every other run reads
directly off each axis as "better/worse than baseline" -- outward is better, except the
one deliberately non-scored axis (see below).

Input: one file per run -- either a raw `play.py` stdout capture containing exactly one
`[BENCH] {...}` line (`play.py ... > run.log`), or a bare JSON file with the same schema.
Multiple/zero `[BENCH]` lines in one file is refused rather than guessed at (fail-closed,
same convention as deploy_readiness.py/deploy_gate_analyzer.py).

Normalization:
  - lower-is-better metrics (err_*, act_legs, jacc_*, cot, fall_rate, ...): axis =
    baseline/value. If baseline == 0: axis = 1.0 when value is also 0 (tied-perfect),
    else 0 (any nonzero reading against a perfect baseline is worst-case).
  - higher-is-better metrics (gait_match): axis = value/baseline (mirrored zero handling).
  - `stride_period_s` is DESCRIPTIVE, not scored: plain value/baseline ratio, no
    better/worse judgement (longer stride than baseline plots further out, shorter plots
    inside) -- it says nothing on its own about whether that's good, since a longer
    stride doesn't imply lower CoT. Its axis label is italicized "(descriptive)" so it
    isn't misread as a scored axis.
  - every axis is clipped to [0, 2] so one standout metric can't squash the rest of the
    chart; points that hit the cap are marked with "dagger" and the true ratio is listed
    in a footnote.
  - a custom `--metrics` key not in the built-in direction table needs an explicit
    --higher-is-better/--informative flag, or the script refuses to guess its direction.

Usage:
  # A0 vs two A1 variants, default 8-axis comparator, A0 as baseline (first file)
  python scripts/plot_radar.py logs/deploy_safety/a0.log logs/deploy_safety/aa1_v1.log logs/deploy_safety/aa1_v2.log \\
      --labels A0,A1-v1,A1-v2

  # thesis figure: vector output, explicit baseline, custom title
  python scripts/plot_radar.py a0.json a1.json a2.json a3.json a4.json \\
      --baseline A0 --format pdf --title "Architecture comparison vs A0"

Up to ~5-6 runs is the readable range for this chart form -- color alone stops reliably
separating series past 3 (see dataviz palette notes), which is why every run also gets a
distinct line style + marker; past 8 runs the fixed 8-slot categorical palette runs out of
slots and the script refuses rather than cycling hues.

Writes pdf (default; --format png/svg for LaTeX-ready vector output) next to the first
input file as `<first_input_stem>_radar.<ext>`, unless --out overrides it.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_RE = re.compile(r"\[BENCH\]\s+(\{.*\})")

# Direction table for the built-in BENCH scorecard fields (scripts/play.py "BENCHMARK
# SCORECARD"). Anything not listed here needs an explicit --higher-is-better/--informative
# flag from the caller -- the script never guesses a metric's direction.
LOWER_IS_BETTER = {
  "err_vx", "err_vy", "err_yaw", "fall_rate",
  "action_rate", "act_legs", "act_arms",
  "jacc", "jacc_legs", "jacc_arms", "jacc_legs_p95", "jacc_arms_p95",
  "orient_dev", "height_dev", "ub_pose_dev", "ub_arm_vel",
  "mech_power_w", "cot",
  "ss_err_vx", "ss_err_vy", "t90_s",
}
HIGHER_IS_BETTER = {"gait_match"}
INFORMATIVE = {"stride_period_s"}

DEFAULT_METRICS = [
  "err_vx", "err_vy", "err_yaw", "act_legs", "jacc_legs_p95",
  "cot", "fall_rate", "stride_period_s",
]

AXIS_CAP = 2.0

# palette.md categorical order (validated adjacent CVD/normal-vision separation); past
# slot 3 the all-pairs separation degrades, mitigated here by the paired line style/marker.
LIGHT = {
  "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
  "grid": "#e1e0d9",
  "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
             "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
}
DARK = {
  "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
  "grid": "#2c2c2a",
  "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}

# (linestyle, marker) per series slot -- secondary encoding so identity doesn't rely on
# hue alone once more than ~3 series are on screen at once (radar lines all cross, unlike
# a stacked bar's fixed adjacency, so this is the "all-pairs" case in dataviz terms).
STYLES = [
  ("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"),
  ((0, (3, 1, 1, 1)), "*"), ((0, (5, 1)), "P"), ((0, (1, 1)), "X"), ("--", "v"),
]


def load_bench(path: Path) -> dict:
  text = path.read_text()
  stripped = text.strip()
  if stripped.startswith("{"):
    return json.loads(stripped)
  hits = BENCH_RE.findall(text)
  if len(hits) != 1:
    sys.exit(f"{path}: found {len(hits)} '[BENCH]' lines, need exactly 1 "
              "(isolate a single benchmark capture per file)")
  return json.loads(hits[0])


def default_label(path: Path, bench: dict) -> str:
  raw = bench.get("label")
  if raw and raw != "unknown":
    # bench_meta["label"] is normally a checkpoint path like
    # logs/rsl_rl/h1_2_velocity/<run_dir>/model_N.pt -- the run dir is the useful name.
    run_dir = Path(raw).parent.name
    if run_dir:
      return run_dir
    return Path(raw).stem
  return path.stem


def classify(metric: str, higher: set, informative: set) -> str:
  if metric in informative:
    return "informative"
  if metric in higher:
    return "higher"
  if metric in LOWER_IS_BETTER:
    return "lower"
  if metric in HIGHER_IS_BETTER:
    return "higher"
  if metric in INFORMATIVE:
    return "informative"
  sys.exit(f"metric {metric!r} has no known direction -- pass --higher-is-better "
            f"{metric} or --informative {metric} to classify it")


def normalize(metric: str, direction: str, base_val: float, val: float) -> tuple:
  """Returns (axis_value_clipped, raw_ratio, capped)."""
  if direction == "informative":
    ratio = val / base_val if base_val != 0 else (1.0 if val == 0 else float("inf"))
  elif base_val == 0:
    ratio = 1.0 if val == 0 else (0.0 if direction == "lower" else float("inf"))
  elif direction == "lower":
    ratio = base_val / val if val != 0 else float("inf")
  else:  # higher
    ratio = val / base_val
  clipped = max(0.0, min(AXIS_CAP, ratio))
  capped = ratio > AXIS_CAP
  return clipped, ratio, capped


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("inputs", nargs="+", help="one file per run: play.py log capture "
                   "(containing one [BENCH] line) or a bare BENCH json")
  ap.add_argument("--labels", help="comma-separated legend labels, one per input, "
                   "in order (default: derived from each run's checkpoint dir name)")
  ap.add_argument("--baseline", help="label to normalize against (default: the first "
                   "input's label)")
  ap.add_argument("--metrics", help=f"comma-separated BENCH json keys, in axis order "
                   f"(default: {','.join(DEFAULT_METRICS)})")
  ap.add_argument("--higher-is-better", help="comma-separated custom metric keys to "
                   "classify as higher-is-better")
  ap.add_argument("--informative", help="comma-separated custom metric keys to classify "
                   "as descriptive-only (plotted, not scored)")
  ap.add_argument("--theme", choices=["light", "dark"], default="light")
  ap.add_argument("--format", choices=["png", "pdf", "svg"], default="pdf")
  ap.add_argument("--out", help="output file path (default: <first_input_stem>_radar.<ext> "
                   "next to the first input)")
  ap.add_argument("--title")
  args = ap.parse_args()

  paths = [Path(p) for p in args.inputs]
  benches = [load_bench(p) for p in paths]

  if args.labels:
    labels = [s.strip() for s in args.labels.split(",")]
    if len(labels) != len(paths):
      sys.exit(f"--labels has {len(labels)} entries, need {len(paths)} (one per input)")
  else:
    labels = [default_label(p, b) for p, b in zip(paths, benches)]
  if len(set(labels)) != len(labels):
    sys.exit(f"labels must be unique, got {labels}")
  runs = dict(zip(labels, benches))

  if len(runs) > len(STYLES):
    sys.exit(f"{len(runs)} runs given, only {len(STYLES)} series slots defined "
              "(colors are never cycled/reused -- cut runs or split into two figures)")

  baseline_label = args.baseline or labels[0]
  if baseline_label not in runs:
    sys.exit(f"--baseline {baseline_label!r} not among the resolved labels {labels}")

  metrics = (args.metrics.split(",") if args.metrics else DEFAULT_METRICS)
  higher_override = set((args.higher_is_better or "").split(",")) - {""}
  informative_override = set((args.informative or "").split(",")) - {""}
  directions = {m: classify(m, higher_override, informative_override) for m in metrics}

  for label, bench in runs.items():
    missing = [m for m in metrics if m not in bench]
    if missing:
      sys.exit(f"{label}: missing metric(s) {missing} in its BENCH json "
                f"(available: {sorted(bench.keys())})")

  if len(labels) > 3:
    print(f"WARN: {len(labels)} runs on one radar -- color separation degrades past 3 "
          "series (mitigated here by per-run line style/marker, but consider splitting "
          "into multiple figures for the thesis if it reads cluttered).", file=sys.stderr)

  # ---- normalize ----
  base_bench = runs[baseline_label]
  axis_values: dict[str, dict[str, float]] = {}  # label -> metric -> clipped value
  capped_footnotes: list[str] = []
  for label, bench in runs.items():
    axis_values[label] = {}
    for m in metrics:
      clipped, ratio, capped = normalize(m, directions[m], base_bench[m], bench[m])
      axis_values[label][m] = clipped
      if capped:
        ratio_str = "off-scale (baseline is 0)" if np.isinf(ratio) else f"{ratio:.2f}x"
        capped_footnotes.append(f"{label}/{m}: {ratio_str}")

  # ---- plot ----
  pal = DARK if args.theme == "dark" else LIGHT
  plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": pal["ink"], "axes.edgecolor": pal["grid"],
    "xtick.color": pal["ink2"], "ytick.color": pal["muted"],
  })
  fig = plt.figure(figsize=(9.5, 7.5), facecolor=pal["surface"])
  ax = fig.add_axes((0.08, 0.08, 0.62, 0.80), projection="polar", facecolor=pal["surface"])

  n = len(metrics)
  angles = [2 * np.pi * i / n for i in range(n)]
  angles_closed = angles + angles[:1]

  ax.set_theta_offset(np.pi / 2)
  ax.set_theta_direction(-1)
  ax.set_ylim(0, AXIS_CAP * 1.18)  # headroom so a capped point/dagger clears the axis labels

  # rings: hairline gray, except the baseline ring at 1.0 in the baseline/axis chrome color
  ring_vals = [v for v in (0.5, 1.0, 1.5, 2.0)]
  ax.set_rgrids(ring_vals, labels=[f"{v:g}" for v in ring_vals], angle=0,
                color=pal["muted"], fontsize=7)
  ax.yaxis.grid(True, color=pal["grid"], linewidth=0.8, linestyle="-")
  ax.xaxis.grid(True, color=pal["grid"], linewidth=0.8, linestyle="-")
  baseline_circle = plt.Circle((0, 0), 1.0, transform=ax.transData._b,
                                fill=False, edgecolor=pal["ink2"], linewidth=1.8, zorder=1.5)
  ax.add_artist(baseline_circle)
  ax.spines["polar"].set_color(pal["grid"])

  ax.set_xticks(angles)
  tick_labels = [f"{m}\n(descriptive)" if directions[m] == "informative" else m
                 for m in metrics]
  ax.set_xticklabels(tick_labels, fontsize=8.5, color=pal["ink2"])
  for tick, m in zip(ax.get_xticklabels(), metrics):
    if directions[m] == "informative":
      tick.set_fontstyle("italic")
      tick.set_color(pal["muted"])

  for i, (label, bench) in enumerate(runs.items()):
    color = pal["series"][i]
    linestyle, marker = STYLES[i]
    vals = [axis_values[label][m] for m in metrics]
    vals_closed = vals + vals[:1]
    ax.plot(angles_closed, vals_closed, color=color, linestyle=linestyle,
            linewidth=2, marker=marker, markersize=6.5,
            markerfacecolor=color, markeredgecolor=pal["surface"], markeredgewidth=1.2,
            label=label, zorder=3)
    ax.fill(angles_closed, vals_closed, color=color, alpha=0.08, zorder=2)
    for ang, m, v in zip(angles, metrics, vals):
      _, ratio, capped = normalize(m, directions[m], base_bench[m], bench[m])
      if capped:
        ax.annotate("†", (ang, AXIS_CAP - 0.10), color=pal["muted"],
                    fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)

  ax.legend(loc="upper left", bbox_to_anchor=(1.20, 1.05), frameon=False,
            labelcolor=pal["ink"], fontsize=9)

  title = args.title or f"Policy comparison vs {baseline_label} (baseline = 1.0 ring)"
  ax.set_title(title, color=pal["ink"], fontsize=12, pad=28)

  footnote_lines = []
  if any(directions[m] == "informative" for m in metrics):
    footnote_lines.append("Italic axis is descriptive only (not scored): "
                           "farther out just means a larger raw value than baseline.")
  if capped_footnotes:
    footnote_lines.append("† off-scale, clipped to the outer ring: "
                           + "; ".join(capped_footnotes))
  if footnote_lines:
    fig.text(0.5, 0.02, "\n".join(footnote_lines), ha="center", va="bottom",
              fontsize=7.5, color=pal["muted"], wrap=True)

  out = Path(args.out) if args.out else paths[0].with_name(
      f"{paths[0].stem}_radar.{args.format}")
  fig.savefig(out, dpi=200 if args.format == "png" else None,
              facecolor=pal["surface"], bbox_inches="tight")
  print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
