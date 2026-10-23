#!/usr/bin/env python3
"""Report-ready figures into figures/report_figures/ (a .pgf + .pdf + CSV sidecars each).

A selection and styling layer over plot_thesis_figures.py (which keeps the drawing code and the
thesis look): this file decides WHICH Runs go into each figure, applies the report style (sans-serif,
8 pt, no grid, no top/right spines, plain names: A0 -> Flat, A1a -> Hierarchical). The
figures carry NO title (owner, 2026-10-02): the take-away and the detail live in the captions in
report_figures.tex.

  f_track_{A1a,A0,A1a_DR}   command vs onboard estimate vs motion-capture truth, one Run each
  f_hier_{A1a,A1a_DR}       the same Runs: command -> HL target -> LL delivery -> truth
  f_chain_clean             walking metrics at training bench / bridge sim / robot (dumbbells)
  f_upper_3pol              upper-body cost per policy, every unloaded Run pooled over seeds and sessions
  f_reward_return, f_reward_terms_{walking,standing}
                            the env reward at the three stages (dumbbells)

Stages are CATEGORICAL (different command profiles, different instruments), so every cross-stage
figure is a dumbbell: one row per policy, the value on the x axis, one marker per stage and a bar over
the row's range. A line drawn across the stages would imply a trend between them.

Usage:  python scripts/plot_report_figures.py [--out figures/report_figures]
"""
import argparse
import csv
import dataclasses
import json
import textwrap
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plot_thesis_figures as ptf  # noqa: E402  (sets the pgf backend + the thesis rcParams)
import bench_flight_recorder as bfr  # noqa: E402

plt = ptf.plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Report style (SCIFIG rubric, owner's calls 2026-09-29): sans-serif for readability, nothing below
# 8 pt, no grid lines or top/right box. `\sffamily` is what the pgf backend emits for sans-serif, so
# the including document's sans font is used (report_figures.tex loads Latin Modern Sans, as the report does).
SANS = r"\usepackage{lmodern}" "\n" r"\usepackage[sc,osf]{mathpazo}"   # the report's fonts: sans is Latin Modern Sans, math Palatino
plt.rcParams.update({
  "font.family": "sans-serif",
  "pgf.preamble": plt.rcParams["pgf.preamble"] + "\n" + SANS,
  "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
  "xtick.labelsize": 8, "ytick.labelsize": 8,
  "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
  "savefig.bbox": "tight", "savefig.pad_inches": 0,   # crop to the ink, no margin: the figure is the plot
})
NAMES = {"A0": "Flat", "A1a": "Hierarchical", "A1a_DR": "Hierarchical + DR"}
ptf.DISPLAY_NAME.update(NAMES)
ptf.FIT_W = ptf.FIG_W
ptf.AXIS_LABEL.update({"vx": "sagittal", "vy": "lateral", "wz": "yaw rate"})


def name(tag, seed=False):
  """`A1a_DR_s123` -> "Hierarchical + DR" (or "... s123" when seeds are shown)."""
  variant, s = ptf._variant_seed(tag)
  return NAMES.get(variant, variant) + (f" s{s}" if seed else "")


# One Run per figure: the s42 Run of 2026-09-16 with the least motion-capture dropout, so the
# ground-truth line is continuous. (Per-Run gaps: figures/tracking_runs/track_runs_summary.csv.)
RUNS = {"A1a": "2026_09_16-_09_45_A1a_s42",        # 0.4 % missing
        "A0": "2026_09_16-_10_48_A0_s42",          # 7.8 % (the other two A0 s42 Runs: 56-66 %)
        "A1a_DR": "2026_09_16-_11_12_A1a_DR_s42"}  # 2.8 %

# The one hardware Run left out of f_chain: on 6 of the 8 metrics it sits 1.25-2.9x above its
# variant's median (mech_power_w 642 W against a 188-342 W band for the same policy). Owner's call.
CHAIN_EXCLUDE = {"2026_09_16-_10_07_A1a_s42"}
CHAIN_VARIANTS = ("A0", "A1a", "A1a_DR")      # cotcap removed from the report (owner, 2026-09-29)
CHAIN_LABEL = {
  "cot": "cost of transport (-)",
  "mech_power_w": "mechanical power (W)",
  "stride_period_s": "stride period (s)",
  "act_legs_rad": "leg action rate (rad)",
  "jacc_legs": "leg joint accel. (rad/s\\textsuperscript{2})",
  "err_yaw": "yaw-rate error (rad/s)",
  "orient_dev": "orientation deviation (-)",
  "omega_xy": "roll/pitch rate (rad/s)",
  "err_vx": "sagittal velocity error (m/s)",
  "err_vy": "lateral velocity error (m/s)",
  "body_joint_vel": "joint speed (rad/s)",
}
# The schedule-replay chain (2026-10-02): bench and bridge run ONE 253 s command schedule and go
# through ONE scorer (ptf.chain_seq_units). The simulation figures add the ground-truth velocity
# errors, which only the two simulated stages can measure; standing has no energy metrics.
CHAIN_TAGS = ("A0_s42", "A0_s123", "A1a_s42", "A1a_s123", "A1a_DR_s42", "A1a_DR_s123")
# Panel order (owner, 2026-10-08): energy and gait clock on top, then the velocity errors, then the
# smoothness and posture metrics. The upper body has its own figure (f_upper_3pol). The robot figure has
# no sagittal/lateral error: most Runs have motion-capture truth, but the operator's commands differed
# per policy (flat driven to 0.8-1.0 m/s, the hierarchies mostly 0.3-0.5) in walking segments of a few
# seconds, so the error measures the command profile, not the policy. Standing has no energy and no
# stride clock.
CHAIN_SEED_TAGS = CHAIN_TAGS + ("A0_s7", "A1a_s7", "A1a_DR_s7")   # seed 7 has bench and bridge but no robot Runs
CHAIN_KEYS = ["cot", "mech_power_w", "stride_period_s", "err_yaw", "act_legs_rad", "jacc_legs", "orient_dev",
              "omega_xy", "body_joint_vel"]
SIM_KEYS = {"walking": ["cot", "mech_power_w", "stride_period_s", "err_vx", "err_vy", "err_yaw", "act_legs_rad",
                        "jacc_legs", "orient_dev", "omega_xy", "body_joint_vel"],
            "standing": ["err_vx", "err_vy", "err_yaw", "act_legs_rad", "jacc_legs", "orient_dev"]}   # no roll/pitch rate: 2 columns x 3 rows on one page (owner, 2026-10-15)
# f_chain_sim_walking is split over two pages (owner, 2026-10-15): two floats of one figure number, in 2 columns.
# Part a: energy, gait clock, then the velocity errors; part b: smoothness and posture.
CHAIN_SIM_PARTS = (("a", ["cot", "mech_power_w", "stride_period_s", "err_yaw", "err_vx", "err_vy"]),
                   ("b", ["act_legs_rad", "jacc_legs", "orient_dev", "omega_xy", "body_joint_vel"]))
CHAIN_CLEAN_PARTS = (("a", ["cot", "mech_power_w", "stride_period_s", "err_yaw", "act_legs_rad", "jacc_legs"]),
                     ("b", ["orient_dev", "omega_xy", "body_joint_vel"]))     # the robot figure, split like f_chain_sim_walking
CONTINUED = {"f_chain_sim_walking_b", "f_chain_clean_b"}     # emitted with \ContinuedFloat: "Figure N (continued)"
UPPER_VARIANTS = ("A0", "A1a", "A1a_DR")

REWARD_SEQ = "data/2026-10-02-chain-seq/env_reward_seq.json"   # score_env_reward.py: robot + schedule-replay bridge Runs
# Schedule B (2026-10-07), the replication of the schedule-replay chain on other commands:
# data/2026-10-07-chain-seq-B/score_replication.py scores the pre-registered criteria into this json.
REPLICATION = "data/2026-10-07-chain-seq-B/replication.json"
REWARD_SEEDS = ("42", "123", "7")
REWARD_TAGS = tuple(f"{v}_s{sd}" for v in ("A0", "A1a", "A1a_DR") for sd in REWARD_SEEDS)   # a policy = its seeds, averaged
STAGE_KEYS = ("bench", "bridge", "hardware")
STAGE_LABEL = {"bench": "training plant", "bridge": "deployment plant", "hardware": "robot"}
# (marker, filled): hollow for the two simulated stages, filled for the robot
STAGE_MARKER = {"bench": ("o", False), "bridge": ("s", False), "hardware": ("D", True)}
TERM_ORDER = ("track_linear_velocity", "track_angular_velocity", "body_orientation_l2",
              "body_ang_vel", "pose", "action_rate_l2", "stand_still")
TERMS_IN = {"walking": TERM_ORDER[:-1], "standing": TERM_ORDER}    # stand_still is 0 while walking
TERM_LABEL = {"track_linear_velocity": "linear-velocity tracking (-)",
              "track_angular_velocity": "yaw-rate tracking (-)",
              "body_orientation_l2": "torso tilt penalty (-)",
              "body_ang_vel": "roll/pitch rate penalty (-)",
              "pose": "default-pose reward (-)",
              "action_rate_l2": "action-rate penalty (-)",
              "stand_still": "stand-still penalty (-)"}
EPISODE_S = 20.0    # velocity_env_cfg episode_length_s; W&B's mean reward is a per-EPISODE return


def pick(bundles, key):
  hit = [b for b in bundles if b.dir.name == RUNS[key]]
  if not hit:
    raise SystemExit(f"{RUNS[key]} not found under logs/robot_logs/")
  return hit[0]


# ----------------------------------------------------------------------- dumbbells ----

LOSS_GREY = "0.4"    # the loss column and its legend entry: secondary to the data, still above 3:1 on white


def merge_seeds(rows):
  """rows: [(label, colour, {stage: [unit values]}, tag)] with one row per SEED of a policy (same label).
  -> ([(label, colour, {stage: [one mean per seed]})], {label: {stage: [(seed, units)]}}). A policy's value
  at a stage is the mean of its seeds' own means, so a seed with many units (64 bench environments) does not
  outweigh one with few; a seed with no unit at the stage (every bridge Run fell) is absent, not zero."""
  merged, detail = {}, {}
  for label, colour, stages, tag in rows:
    seed = ptf._variant_seed(tag)[1]
    m = merged.setdefault(label, (label, colour, {st: [] for st in stages}))
    d = detail.setdefault(label, {st: [] for st in stages})
    for st, v in stages.items():
      if len(v):
        m[2][st].append(float(np.mean(v)))
        d[st].append((seed, len(v)))
  return list(merged.values()), detail


def dumbbell(ax, rows, xlabel, show_labels=True, nticks=4, better=None, none_note=None, vs="bridge"):
  """rows: [(label, colour, {stage: [values]})], top to bottom. One marker per stage at the stage
  mean and a light bar over the row's range of means; the bench's many envs also get a 5th-95th
  percentile whisker. Single Runs are not drawn (owner, 2026-10-03): bridge Runs agree to <1% and sat
  under the marker, and robot ticks read the same as bridge ticks. They stay in the CSV, one row per
  unit.

  better ("lower" | "higher"): which direction of the metric is good. Given it, every row with both a
  bench and a bridge mean gets a grey number in the right margin: the relative LOSS from bench to
  `vs` ("bridge" | "hardware"; the robot where the figure plots it), (vs - bench) / |bench| in %, signed so that positive is always worse (the sign flips
  for a higher-is-better metric). Absolute values are not comparable between policies; this is.
  A row with a bench value and NO value at `vs` prints `none_note` in the same column ("fell"), so a
  missing number is not read as a zero loss."""
  out = []
  detail = None
  if rows and len(rows[0]) == 4:      # seed rows: merge to one row per policy, no whisker
    rows, detail = merge_seeds(rows)
  n = len(rows)
  for i, (label, colour, stages) in enumerate(rows):
    y = n - 1 - i
    means = {s: float(np.mean(v)) for s, v in stages.items() if len(v)}
    if len(means) > 1:
      ax.hlines(y, min(means.values()), max(means.values()), color=colour, linewidth=2.2,
                alpha=0.3, zorder=1)
    loss = None
    if better and "bench" in means and vs in means and means["bench"]:
      loss = 100 * (means[vs] - means["bench"]) / abs(means["bench"]) * (1 if better == "lower" else -1)
      r = round(loss)
      ax.text(1.03, y, "0 %" if r == 0 else f"{'+' if r > 0 else '$-$'}{abs(r):d} %", color=LOSS_GREY,
              transform=ax.get_yaxis_transform(), ha="left", va="center")
    elif better and none_note and "bench" in means and vs in stages and not len(stages[vs]):
      ax.text(1.03, y, none_note, color=LOSS_GREY, transform=ax.get_yaxis_transform(), ha="left", va="center")
    for s, m in means.items():
      mk, filled = STAGE_MARKER.get(s, ("o", False))
      ax.plot(m, y, marker=mk, markersize=5, linestyle="", markeredgewidth=1.0, color=colour,
              markerfacecolor=colour if filled else "white", zorder=3)
      if len(stages[s]) > 1 and detail is None and s == "bench":
        lo, hi = np.percentile(stages[s], [5, 95])
        ax.plot([lo, hi], [y, y], marker="|", markersize=4, linewidth=0.9, color=colour, zorder=2)
      out += [{"row": label, "stage": s, "value": round(float(v), 6),
               "n": len(stages[s]), "stage_mean": round(m, 6),
               f"loss_pct_bench_to_{vs}": "" if loss is None else round(loss, 2),
               **({"seed": detail[label][s][k][0], "units": detail[label][s][k][1]} if detail else {})}
              for k, v in enumerate(stages[s])]
  ax.set_yticks(range(n))
  if show_labels:   # on shared y axes the labels are set ONCE: an empty list on a sibling wipes them all
    ax.set_yticklabels([ptf.tex(r[0]).replace(" + ", "\n+ ") for r in reversed(rows)])   # two lines: the longest row label sets the left margin
  ax.tick_params(axis="y", length=0, labelleft=show_labels, pad=5)   # row labels must not touch the axis
  from matplotlib.ticker import MaxNLocator
  ax.xaxis.set_major_locator(MaxNLocator(nticks))
  ax.set_ylim(-0.7, n - 0.3)
  ax._thin_ticks = True   # fit_inside drops every other tick where the labels touch
  ax.set_xlabel(textwrap.fill(xlabel, 18, break_long_words=False, break_on_hyphens=False))   # three narrow panels per row: a long label runs into its neighbour
  ax.margins(x=0.08)
  return out


def loss_vs(stages):
  """The stage the grey loss column compares with the bench: the robot where the figure plots it."""
  return "hardware" if "hardware" in stages else "bridge"


def stage_handles(stages=STAGE_KEYS, runs=True, seeds=False):
  h = [Line2D([], [], marker=STAGE_MARKER[s][0], markersize=5, linestyle="", color="0.3",
              markerfacecolor="0.3" if STAGE_MARKER[s][1] else "white", label=STAGE_LABEL[s])
       for s in stages]
  if runs and not seeds:      # seed rows carry no whisker: the spread between seeds is too small to see
    if "bench" in stages:
      h.append(Line2D([], [], marker="|", markersize=4, linewidth=0.9, color="0.3",
                      label="training plant: 5th to 95th\npercentile of environments"))
  return h


def one_row_legend(fig, handles, old_frac, pad=0.08):
  """One-row legend under the axes in a band exactly as tall as the legend (+ pad, inches). `old_frac` is the
  band fraction the figure was tuned with; the figure shrinks by the slack so the axes keep their size."""
  leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
  fig.canvas.draw()
  band = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).height + pad
  leg.remove()
  w, h = fig.get_size_inches()
  fig.set_size_inches(w, h * (1 - old_frac) + band)
  ptf.bottom_legend(fig, handles, ncol=len(handles), height=band / fig.get_figheight())


def write_rows(out, name_, rows):
  out.mkdir(parents=True, exist_ok=True)
  if not rows:
    return
  with open(out / f"{name_}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


# ------------------------------------------------------------------------ f_chain ----

def chain_points(kept, root, regime):
  """-> (rows(key, stages), {(tag, stage): (n scored, n fell)}). Bench and bridge are the schedule-
  replay units (ptf.chain_seq_units). The robot's Runs keep their own assembly (ptf.fig_chain: the
  regime jsons and the hardware jacc recomputation) and are read back from its panel CSVs."""
  hw = {}
  if regime == "walking":
    with tempfile.TemporaryDirectory() as tmp:
      save_pdf, ptf.SAVE_PDF = ptf.SAVE_PDF, False
      ptf.fig_chain(kept, Path(tmp), name="chain", with_notes=False)
      ptf.SAVE_PDF = save_pdf
      import pandas as pd
      for k in CHAIN_KEYS:
        if k == "stride_period_s":      # not a panel of ptf.fig_chain: the Run's realized period, from its regime json
          hw[k] = {t: [float(b.regimes[regime][k]) for b in kept if b.policy == t and regime in b.regimes
                       and np.isfinite(b.regimes[regime].get(k, np.nan))] for t in CHAIN_TAGS}
          continue
        if k == "body_joint_vel":       # not in the robot regime jsons (scored before the key existed): the scorer's rows
          def body_vel(b):
            F = bfr.read_flight(b.flight, None, None, b.run.get("entry"))
            rows = F["walk_rows"] if regime == "walking" else ~F["walk_rows"]
            return float(np.abs(F["df"][[f"meas_dq{j}" for j in range(27)]].to_numpy(float)[rows]).mean())
          hw[k] = {t: [body_vel(b) for b in kept if b.policy == t and regime in b.regimes] for t in CHAIN_TAGS}
          continue
        d = pd.read_csv(Path(tmp) / f"chain_{k}.csv")
        d = d[(d["stage"] == "hardware") & d["value"].notna()]
        hw[k] = {t: d[d.policy == t]["value"].astype(float).tolist() for t in CHAIN_TAGS}
  units, counts = {}, {}
  for t in CHAIN_SEED_TAGS:
    for st in ("bench", "bridge"):
      units[t, st], n, fell = ptf.chain_seq_units(root, t, st, regime)
      counts[t, st] = (n, fell)

  def rows(key, stages):
    return [(name(t), ptf.policy_color(t),
             {st: hw.get(key, {}).get(t, []) if st == "hardware" else
                  [float(u[key]) for u in units[t, st] if u.get(key) is not None and np.isfinite(u[key])]
              for st in stages}, t) for t in CHAIN_SEED_TAGS]     # 4-tuples: dumbbell averages the seeds
  return rows, counts


def fig_chain_seq(rows, keys, stages, out, fname, ncol=3):
  vs = loss_vs(stages)
  nrow = -(-len(keys) // ncol)
  npol = len({r[0] for r in rows(keys[0], stages)})
  extra = 0.5 if nrow == 1 else 0     # a one-row figure has too little height for the legend band under it
  fig, axes = plt.subplots(nrow, ncol, figsize=(ptf.FIG_W, ((0.95 if ncol == 3 else 0.65) + 0.3 * npol) * nrow + extra),
                           sharey=True)
  axes = list(np.atleast_1d(axes).flat)
  for i, (ax, key) in enumerate(zip(axes, keys)):
    write_rows(out, f"{fname}_{key}",
               dumbbell(ax, rows(key, stages), CHAIN_LABEL[key], show_labels=(i % ncol == 0),
                        nticks=2, better="lower", none_note="fell", vs=vs))
  spare = axes[len(keys):]
  for ax in spare:
    ax.set_axis_off()
  fig.subplots_adjust(hspace=0.55, wspace=0.8)   # room for the grey loss column
  if spare:     # the empty grid cell carries the legend instead of a band under the figure
    # anchored right of the cell's left edge: the previous panel's grey numbers hang into the cell's left margin
    spare[0].legend(handles=stage_handles(stages, seeds=True), loc="center left", bbox_to_anchor=(0.28, 0.5),
                    frameon=False, borderaxespad=0)
  else:
    one_row_legend(fig, stage_handles(stages, seeds=True), 0.11 + extra / fig.get_figheight())
  ptf.save_pgf(fig, out, fname)


# ------------------------------------------------------------------------ rewards ----

def load_reward_points(root):
  """{tag: {stage: [{regime: {term: value}} per unit]}} for REWARD_TAGS. A unit is one bench env
  (the live RewardManager on the schedule replay), one bridge Run or one robot Run (both
  score_env_reward.py). Units that fell are excluded, as in the chain figures."""
  if not (root / REWARD_SEQ).exists():
    raise SystemExit(f"{REWARD_SEQ} is missing: run data/2026-10-02-chain-seq/driver_bridge.sh (the "
                     "bridge Runs of the schedule-replay chain and their env-reward scores)")
  hb = json.loads((root / REWARD_SEQ).read_text())
  out = {}
  for tag in REWARD_TAGS:
    bench = {}
    for regime in ("walking", "standing"):
      for u in ptf.chain_seq_units(root, tag, "bench", regime)[0]:
        bench.setdefault(u["label"], {})[regime] = dict(u["reward_terms"])
    bridge = [r for r in hb["bridge"].values() if r["policy"] == tag
              and ptf._tilt_fall_s(root / "logs/deploy_safety" / r["capture"]) is None]
    hw = [r for r in hb["hardware"].values() if r["policy"] == tag]
    # A1a: score foot_gait on the HL-commanded clock the LL trains it on (`foot_gait_hl`), not on
    # A0's fixed 0.6 s clock the env term inherits -- swapped in, never added.
    for terms in [u[r] for u in bench.values() for r in u] + [b["replay"][r] for b in bridge
                                                               if "replay" in b for r in ("walking", "standing")]:
      if "foot_gait_hl" in terms:
        terms["foot_gait"] = terms.pop("foot_gait_hl")
    out[tag] = {"bench": list(bench.values()), "bridge": bridge, "hardware": hw}
  return out


HL_RETURN = "data/2026-10-02-chain-seq/hl_return.json"   # score_env_reward.py --hl-return
HL_ROW = "{}"   # the row says which policy; the caption says which objective


def hl_points(root):
  """{tag: {stage: [{regime: {hl_r, hl_track, hl_cot, hl_windows}} per unit]}} for the hierarchical
  REWARD_TAGS: the reward their HIGH level trains on (bench_flight_recorder.hl_return). Fallen bench envs
  and bridge Runs are excluded as everywhere else; robot Runs without motion capture use the onboard
  estimate for the velocity."""
  if not (root / HL_RETURN).exists():
    raise SystemExit(f"{HL_RETURN} is missing: python scripts/score_env_reward.py --bridge-dir "
                     "logs/sim_logs/chain_seq --hl-return " + HL_RETURN)
  res = json.loads((root / HL_RETURN).read_text())
  out = {}
  for tag in REWARD_TAGS:
    if tag.startswith("A0"):
      continue
    d = res[tag]
    out[tag] = {"bench": [u for u in d["bench"].values() if not u["fell"]],
                "bridge": [u for u in d["bridge"].values()
                           if ptf._tilt_fall_s(root / "logs/deploy_safety" / u["capture"]) is None],
                "hardware": list(d["hardware"].values())}
  return out


def stage_values(pts, stage, regime, terms):
  """The `terms` of every unit of one stage that has them in this regime."""
  out = []
  for r in pts[stage]:
    d = r.get(regime)
    if d is None or any(d.get(t) is None for t in terms):
      continue                                     # a unit without that regime's data is absent, not zero
    out.append({t: d[t] for t in terms})
  return out


def reward_rows(points, regime, key, terms, stages, hl=None):
  """One row per policy. With `hl` (hl_points) a hierarchical policy is scored on the reward its high
  level trains on instead of the environment reward."""
  return [(HL_ROW.format(name(t)) if hl and t in hl else name(t), ptf.policy_color(t),
           {s: [u[regime]["hl_r"] for u in hl[t][s] if regime in u] if hl and t in hl else
               [key(v) for v in stage_values(p, s, regime, terms)] for s in stages}, t)
          for t, p in points.items()]


def full_active(pts, regime):
  """All 16 ACTIVE terms (the 17th, cost_of_transport, has weight 0): live at the bench; 7 log terms + the
  9 plant-replay terms at the bridge."""
  return {"bench": [sum(u[regime].values()) for u in pts["bench"] if regime in u],
          "bridge": [sum(b[regime][t] for t in TERMS_IN[regime]) + sum(b["replay"][regime].values())
                     for b in pts["bridge"] if "replay" in b]}


def seed_facts(points, hl, root):
  """The sentences the seed-averaged reward captions quote, computed from the plotted units."""
  from collections import Counter
  hb = json.loads((root / REWARD_SEQ).read_text())
  total = Counter(r["policy"] for r in hb["bridge"].values())
  sd = lambda t: ptf._variant_seed(t)[1]
  span = lambda v: str(min(v)) if min(v) == max(v) else f"{min(v)} to {max(v)}"
  fell = [f"{name(t)} seed {sd(t)} ({len(points[t]['bridge'])} of {total[t]} stayed upright)"
          for t in points if len(points[t]["bridge"]) < total[t]]
  def robot(units):    # {tag: [units]} -> "Flat: 1 Run (seed 42), 2 (seed 123); ..."
    out = []
    for v in dict.fromkeys(ptf._variant_seed(t)[0] for t in units):
      got = [f"{len(units[t])} Run{'s' * (len(units[t]) != 1)} of seed {sd(t)}" for t in units
             if ptf._variant_seed(t)[0] == v and len(units[t])]
      out.append(f"{name(v + '_s42')}: " + (", ".join(got) if got else "none"))
    return "; ".join(out)
  return {"bench": span([len(p["bench"]) for p in points.values()]),
          "bridge": span(list(total[t] for t in points)),
          "fell": "; ".join(fell),
          "robot_env": robot({t: p["hardware"] for t, p in points.items()}),
          "robot_hl": robot({t: v["hardware"] for t, v in hl.items()}),
          "est": sum(u["ref"] == "est_v_compl" for v in hl.values() for u in v["hardware"])}


def sim_rows(points, hl, regime, stages, scale=1.0):
  """The two simulated stages, one row per policy on ITS objective: Flat on the extrinsic (environment)
  reward over all 16 active terms, a hierarchical policy on the reward its high level trains on. The
  captions say the objectives differ. `scale` turns per step into per episode."""
  rows = []
  for t, p in points.items():
    if t in hl:
      vals = {s: [scale * u[regime]["hl_r"] for u in hl[t][s] if regime in u] for s in stages}
    else:
      full = full_active(p, regime)
      vals = {s: [scale * v for v in full[s]] for s in stages}
    rows.append((name(t), ptf.policy_color(t), vals, t))
  return rows


# The hierarchies on the objective their LOW LEVEL trains on (2026-10-06): goal_space.reward's exp
# kernels on V*_i - s_{i+1}. Bench = a separate rollout (`bench_ll`, same schedule) with V* and the
# post-step state in its trace; bridge and robot = the controller's `_hrl.csv`.
LL_TAGS = tuple(t for t in CHAIN_TAGS if not t.startswith("A0"))
# The low level's reward r_lo = r_track + r_shape + r_cad (the formula catalogue's decomposition).
# r_shape = these nine env terms (same function, parameters and weight in the keeper's LL registry)
# + the keeper's posture and energy terms (bench_flight_recorder.ll_posture_energy); r_cad = the env
# gait term on the HL-commanded clock (`foot_gait_hl`, = the LL `cadence` term). The robot has no
# foot contact, so the hardware figure uses only the CONTACT-FREE part of r_shape, at every stage.
LL_SHAPE_ENV = ("stand_still", "body_ang_vel", "action_rate_l2", "joint_acc_l2", "joint_pos_limits",
                "angular_momentum", "foot_clearance", "foot_slip", "soft_landing")
LL_SHAPE_CF = ("stand_still", "body_ang_vel", "action_rate_l2")      # + posture + energy
LL_ROWS = {"sim": (("r_track", "goal tracking $r^\\mathrm{track}$ (-)"),
                   ("r_shape", "gait shaping $r^\\mathrm{shape}$ (-)"),
                   ("r_cad", "cadence $r^\\mathrm{cad}$ (-)"),
                   ("r_lo", "low-level reward $r^\\mathrm{lo}$ (-)")),
           "hw": (("r_track", "goal tracking $r^\\mathrm{track}$ (-)"),
                  ("r_shape_cf", "gait shaping, contact-free part (-)"))}


def _ll_parts(track, env, extra):
  """track: ll_goal_* dict; env: env reward terms of this regime (or None); extra: ll_posture/energy."""
  out = {"r_track": track.get("ll_goal_r")}
  if env is not None and "ll_posture" in extra and "ll_energy" in extra:
    pe = extra["ll_posture"] + extra["ll_energy"]
    if all(k in env for k in LL_SHAPE_CF):
      out["r_shape_cf"] = sum(env[k] for k in LL_SHAPE_CF) + pe
    if all(k in env for k in LL_SHAPE_ENV + ("foot_gait_hl",)):
      out["r_shape"] = sum(env[k] for k in LL_SHAPE_ENV) + pe
      out["r_cad"] = env["foot_gait_hl"]
      if out["r_track"] is not None:
        out["r_lo"] = out["r_track"] + out["r_shape"] + out["r_cad"]
  return {k: v for k, v in out.items() if v is not None}


def ll_points(kept, root):
  """{(tag, regime): {stage: [per-unit {r_track, r_shape, r_cad, r_lo, r_shape_cf}]}}. Fallen bench
  envs and bridge Runs are excluded as everywhere else (ptf.chain_seq_units)."""
  import score_env_reward as ser
  hb = json.loads((root / REWARD_SEQ).read_text())
  rewards = ser.task_cfg()
  hw_env = {}
  for b in kept:
    if b.policy in LL_TAGS and b.hrl is not None:
      try:
        hw_env[b.dir.name] = ser.score_flight(b.flight, rewards)   # needs a velocity truth for its own
        # tracking terms; the contact-free shaping terms used here do not depend on it
      except SystemExit as e:
        print(f"[REPORT] ll: {b.dir.name}: no env terms ({e})")
  pts = {}
  for t in LL_TAGS:
    for regime in ("walking", "standing"):
      bench = [_ll_parts(u, u.get("reward_terms"), u)   # a unit's reward_terms are already this regime's
               for u in ptf.chain_seq_units(root, t, "bench_ll", regime)[0] if "ll_goal_r" in u]
      bridge = []
      for u in ptf.chain_seq_units(root, t, "bridge", regime)[0]:
        fl = root / "logs/deploy_safety" / u["flight_recorder"]
        h = fl.with_name(fl.stem + "_hrl.csv")
        e = hb["bridge"].get(f"{t}/{u['label']}")
        env = {**e[regime], **e["replay"][regime]} if e and "replay" in e else None
        bridge.append(_ll_parts(bfr.ll_goal_from_hrl(h, regime) if h.exists() else {}, env, u))
      hw = []
      for b in kept:
        if b.policy != t or b.hrl is None:
          continue
        js = b.regimes.get(regime, {})
        tel = b.dir / js["all_joints"] if js.get("all_joints") else None
        extra = bfr.ll_shaping_for_run(b.flight, tel, js.get("pair_lag_s", 0.0), js.get("pair_skew_ppm", 0.0),
                                       regime) if js else {}
        env = hw_env.get(b.dir.name, {}).get(regime)
        hw.append(_ll_parts(bfr.ll_goal_from_hrl(b.hrl, regime), env, extra))
      pts[t, regime] = {"bench": bench, "bridge": bridge, "hardware": hw}
  return pts


def fig_ll_reward(pts, out, stages=STAGE_KEYS, sfx=""):
  rows_def = LL_ROWS["hw" if "hardware" in stages else "sim"]
  fig, axes = plt.subplots(len(rows_def), 2, figsize=(ptf.FIG_W, 1.6 * len(rows_def) + 1.0), sharey=True)
  rows = []
  for j, regime in enumerate(("walking", "standing")):
    for i, (key, lab) in enumerate(rows_def):
      if key == "r_cad" and regime == "standing":
        # feet_gait is gated on a command above 0.1: ~0 by construction while standing, and a
        # relative loss of ~0 is noise (it printed +100 %)
        axes[i, j].text(0.5, 0.5, "the gait term is switched off\nwhile standing (command < 0.1)",
                        transform=axes[i, j].transAxes, ha="center", va="center", color="0.4")
        axes[i, j].set_axis_off()
        continue
      r = dumbbell(axes[i, j], [(name(t), ptf.policy_color(t),
                                 {s: [u[key] for u in pts[t, regime][s] if key in u] for s in stages})
                                for t in LL_TAGS],
                   lab, show_labels=(j == 0), nticks=2, better="higher", none_note="fell", vs=loss_vs(stages))
      rows += [{**x, "regime": regime, "term": key} for x in r]
    axes[0, j].set_title(regime.capitalize())
  write_rows(out, f"f_ll_reward{sfx}_all", rows)
  fig.subplots_adjust(hspace=1.0, wspace=0.85)
  one_row_legend(fig, stage_handles(stages), 0.16 if len(rows_def) < 4 else 0.11)
  ptf.save_pgf(fig, out, f"f_ll_reward{sfx}")


def fig_reward_return(points, hl, out, stages=STAGE_KEYS, sfx=""):
  """On W&B's scale: return = per-step reward x episode length. Simulated stages only: every policy on
  all 16 active environment terms, hierarchies also on their high-level reward. With the robot: the 7
  terms reproducible at all three stages (Flat also all 16 at the bench and bridge), hierarchies on the
  high-level reward."""
  nv, nh = len(points) // len(REWARD_SEEDS), len(hl) // len(REWARD_SEEDS)
  nrow = nv + (0 if "hardware" not in stages else nv - nh)   # with the robot, Flat has two rows
  fig, axes = plt.subplots(1, 2, figsize=(ptf.FIG_W, 1.4 + 0.3 * nrow), sharex=True, sharey=True)
  rows = []
  for i, (ax, regime) in enumerate(zip(axes, ("walking", "standing"))):
    dr = sim_rows(points, hl, regime, stages, EPISODE_S) if "hardware" not in stages else []
    for t, p in ([] if dr else points.items()):
      if t in hl:    # a hierarchical policy: the return its high level trains on, one row
        dr.append((HL_ROW.format(name(t)), ptf.policy_color(t),
                   {s: [EPISODE_S * u[regime]["hl_r"] for u in hl[t][s] if regime in u] for s in stages}, t))
        continue
      seven = {s: [EPISODE_S * sum(v.values()) for v in stage_values(p, s, regime, TERMS_IN[regime])]
               for s in stages}
      full = {s: [EPISODE_S * v for v in vals] for s, vals in full_active(p, regime).items()}
      dr += [(f"{name(t)}, 7 terms", ptf.policy_color(t), seven, t),
             (f"{name(t)}, all 16", ptf.policy_color(t), full, t)]
    r = dumbbell(ax, dr, f"return: per-step mean $\\times$ {EPISODE_S:g} s (-)", show_labels=(i == 0), better="higher",
                 none_note="fell", vs=loss_vs(stages))
    rows += [{**x, "regime": regime} for x in r]
    ax.set_title(regime.capitalize())
  write_rows(out, f"f_reward_return{sfx}_all", rows)
  one_row_legend(fig, stage_handles(stages, seeds=True), 0.26)
  ptf.save_pgf(fig, out, f"f_reward_return{sfx}")


def fig_reward_terms(points, out, regime, stages=STAGE_KEYS, sfx=""):
  terms = TERMS_IN[regime]
  ncol = 3   # 4 columns left ~2.5 cm per panel: 4-decimal tick labels of the small penalties collided
  nrow = -(-len(terms) // ncol)
  fig, axes = plt.subplots(nrow, ncol, figsize=(ptf.FIG_W, 1.75 * nrow + 0.6), sharey=True)
  axes = list(np.atleast_1d(axes).flat)
  rows = []
  for i, (ax, t) in enumerate(zip(axes, terms)):
    r = dumbbell(ax, reward_rows(points, regime, lambda v, t=t: v[t], (t,), stages), TERM_LABEL[t],
                 show_labels=(i % ncol == 0), nticks=2, better="higher", vs=loss_vs(stages))
    rows += [{**x, "term": t} for x in r]
  spare = axes[len(terms):]
  for ax in spare:
    ax.set_axis_off()
  write_rows(out, f"f_reward_terms_{regime}{sfx}_all", rows)
  fig.subplots_adjust(hspace=0.7, wspace=0.8)   # room for the grey loss column
  if spare:     # the empty grid cell carries the legend instead of a band under the figure
    # anchored right of the cell's left edge: the previous panel's grey numbers hang into the cell's left margin
    spare[0].legend(handles=stage_handles(stages, seeds=True), loc="center left", bbox_to_anchor=(0.28, 0.5),
                    frameon=False, borderaxespad=0)
  else:
    one_row_legend(fig, stage_handles(stages, seeds=True), 0.18)
  ptf.save_pgf(fig, out, f"f_reward_terms_{regime}{sfx}")


# ------------------------------------------------------------------------ payload ----
# The 2026-09-23 backpack Runs: one frozen low level, the s42 high level with and without payload DR.
# Scores are the bundles re-scored against the onboard EKF the robot ran (`ekf_rot`) instead of the
# scorer's default complementary filter: bench_flight_recorder.py --robot --vel-reference est_v_ekf_rot
# --traj-dir <bundle> --out-dir data/2026-10-05-payload/<bundle> (logs beside them). No motion capture.
PAYLOAD_SCORES = "data/2026-10-05-payload"
PAYLOAD_VARIANTS = ("A1a", "A1a_DR")
PAYLOAD_TRAIN_MAX = 12.0     # PAYLOAD_DR_RANGE, config/h1_2/env_cfgs.py
ROBOT_KG = 75.0              # bench_flight_recorder.MASS_G's mass; cot here is re-normalised by robot + payload
PAYLOAD_PANELS = (("standing", "act_legs_rad", "standing leg\naction rate (rad)"),
                  ("standing", "err_vx_est", "standing sagittal\nvelocity error (m/s)"),
                  ("walking", "cot", "walking cost\nof transport (-)"))


def payload_points(root):
  """One row per (Run, regime) of the loaded comparison: tag, kg, thin flag and the scored metrics."""
  rows = []
  for d in sorted((root / PAYLOAD_SCORES).glob("2026_09_23-*/")):
    run = json.loads((root / "logs/robot_logs" / d.name / "run.json").read_text())
    variant, seed = ptf._variant_seed(run["policy_tag"])
    if variant not in PAYLOAD_VARIANTS:
      continue
    kg = float(run.get("payload_kg") or 0.0)
    for f in d.glob("*.json"):
      regime = f.name.rsplit(".", 2)[1]
      m = json.loads(f.read_text())
      if m.get("cot") is not None:
        m["cot"] *= ROBOT_KG / (ROBOT_KG + kg)
      rows.append({"variant": variant, "seed": seed, "run": d.name, "payload_kg": kg, "regime": regime,
                   "thin": bool(m["thin"]), "duration_s": m["duration_s"], **{k: m.get(k) for _, k, _ in PAYLOAD_PANELS}})
  return rows


def fig_payload(rows, out, name_="f_payload"):
  """Three metrics against the carried payload, one marker per Run; hollow = < 10 s in the regime
  (drawn, never in the mean); a short bar per cell = mean of the full-length Runs."""
  fig, axes = plt.subplots(1, 3, figsize=(ptf.FIG_W, 2.3))
  kgs = sorted({r["payload_kg"] for r in rows})
  facts = {}
  for ax, (regime, key, ylabel) in zip(axes, PAYLOAD_PANELS):
    ax.axvspan(PAYLOAD_TRAIN_MAX, max(kgs) + 2.5, color="0.92", linewidth=0, zorder=0)
    panel = []
    for k, v in enumerate(PAYLOAD_VARIANTS):
      col = ptf.CHAIN_COLOR[v]
      for kg in kgs:
        cell = [r for r in rows if r["variant"] == v and r["payload_kg"] == kg and r["regime"] == regime
                and r[key] is not None]
        x0 = kg + (k - 0.5) * 1.3
        jit = np.linspace(-0.3, 0.3, len(cell)) if len(cell) > 1 else [0.0]
        for dx, r in zip(jit, cell):
          ax.plot(x0 + dx, r[key], marker="o", markersize=3.4, linestyle="", markeredgewidth=0.7,
                  markeredgecolor=col, markerfacecolor="white" if r["thin"] else col, zorder=3)
          panel.append({**{c: r[c] for c in ("variant", "seed", "run", "payload_kg", "thin", "duration_s")},
                        "x": round(x0 + dx, 3), "value": round(float(r[key]), 6)})
        full = [r[key] for r in cell if not r["thin"]]
        if full:
          m = float(np.mean(full))
          ax.plot([x0 - 0.55, x0 + 0.55], [m, m], color=col, linewidth=1.4, zorder=2)
          facts[key, v, kg] = (m, len(full))
        if not cell and regime == "walking":
          ax.text(x0, 0.03, "n/a", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=8, color=col)
    ax.set_xticks(kgs)
    ax.set_xticklabels([f"{kg:g}" for kg in kgs])
    ax.set_xlim(min(kgs) - 2.5, max(kgs) + 2.5)
    ax.set_xlabel("payload (kg)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.12)
    ptf.write_panel_csv(out, name_, key, ["variant", "seed", "run", "payload_kg", "thin", "duration_s", "x",
                                          "value"], panel)
  from matplotlib.patches import Patch
  handles = [Line2D([], [], marker="o", linestyle="", color=ptf.CHAIN_COLOR[v], markersize=3.4, label=NAMES[v])
             for v in PAYLOAD_VARIANTS]
  handles += [Line2D([], [], marker="o", linestyle="", color="0.3", markerfacecolor="white", markersize=3.4,
                     label="$<$ 10 s in regime"),
              Patch(facecolor="0.92", label=f"beyond training range ($>$ {PAYLOAD_TRAIN_MAX:g} kg)")]
  ptf.bottom_legend(fig, handles, ncol=2, height=0.21)
  ptf.save_pgf(fig, out, name_)
  return facts


# ----------------------------------------------------------------- standing share ----
# The crossed rel_standing_envs design (2026-09-21): Flat and Hierarchical trained at 5 / 12 / 20 % standing
# environments, each benched at zero command (the stand_pay0 cell, 64 envs x 600 steps x 2 rollout seeds).
# Read from the [BENCH] line of every .log (crossed_bench.json lacks the Hierarchical 5 % arms).
# Seed 7 (12 % only) was benched later on the same protocol; that batch's seed-42 re-benches were a
# cross-session check and are not plotted.
SHARE_SRC = (("data/2026-09-21-rse-crossed", "*__stand_pay0*.log"), ("data/2026-10-09-rse-s7", "*_s7__stand_pay0*.log"))
SHARE_MARKER = {"42": "o", "123": "^", "7": "D"}
SHARE_EVAL_S = 600 * 0.02    # eval_steps x step_dt of the stand_pay0 cell
SHARE_PANELS = (("act_legs_rad", "standing leg\naction rate (rad)"),
                ("td_per_foot", "touchdowns per foot\nin a 12 s stand (-)"))


def share_points(root):
  """One row per TRAINED checkpoint: re-benches of the same weights (`__rep2`, `__rep3`) are averaged
  into their checkpoint, an independent retrain (`_repro2`) is its own row."""
  import re
  per = {}
  for f in sorted(f for d, g in SHARE_SRC for f in (root / d).glob(g)):
    m = re.match(r"(a0|a1a)_rse(\d\d)_(s\d+(?:_repro\d)?)__stand_pay0", f.name)
    line = next(l for l in f.read_text().splitlines() if l.startswith("[BENCH] "))
    b = json.loads(line[len("[BENCH] "):])
    per.setdefault(m.groups(), []).append(b)
  rows = []
  for (arm, rse, run), bs in per.items():
    rows.append({"variant": {"a0": "A0", "a1a": "A1a"}[arm], "rse": int(rse) / 100, "run": run,
                 "seed": run.split("_")[0][1:], "n_bench": len(bs),
                 "act_legs_rad": float(np.mean([b["act_legs_rad"] for b in bs])),
                 "td_per_foot": float(np.mean([SHARE_EVAL_S / b["stride_period_s"] for b in bs]))})
  return rows


def fig_standing_share(rows, out, name_="f_standing_share"):
  """Standing quality against the training share of standing environments, one marker per checkpoint
  (marker = seed), a thin line through the per-architecture means (the share is an ordered dose)."""
  fig, axes = plt.subplots(1, 2, figsize=(ptf.FIG_W, 2.4))
  shares = sorted({r["rse"] for r in rows})
  facts = {}
  for ax, (key, ylabel) in zip(axes, SHARE_PANELS):
    panel = []
    for k, v in enumerate(("A0", "A1a")):
      col = ptf.CHAIN_COLOR[v]
      means = []
      for s in shares:
        cell = [r for r in rows if r["variant"] == v and r["rse"] == s]
        x0 = 100 * s + (k - 0.5) * 1.2
        jit = np.linspace(-0.35, 0.35, len(cell)) if len(cell) > 1 else [0.0]
        for dx, r in zip(jit, cell):
          ax.plot(x0 + dx, r[key], marker=SHARE_MARKER[r["seed"]], markersize=3.6, linestyle="",
                  color=col, zorder=3)
          panel.append({**{c: r[c] for c in ("variant", "rse", "run", "seed", "n_bench")},
                        "x": round(x0 + dx, 3), "value": round(r[key], 6)})
        means.append((x0, float(np.mean([r[key] for r in cell]))))
        facts[key, v, s] = (means[-1][1], len(cell))
      ax.plot(*zip(*means), color=col, linewidth=0.8, zorder=2)
    if key == "act_legs_rad":
      ax.set_yscale("log")
      ax.set_yticks([0.003, 0.01, 0.03, 0.1])
      ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
      ax.yaxis.set_minor_formatter(plt.NullFormatter())
    else:
      ax.set_ylim(bottom=0)
    ax.set_xticks([100 * s for s in shares])
    ax.set_xticklabels([f"{100 * s:g}" for s in shares])
    ax.set_xlim(100 * shares[0] - 3, 100 * shares[-1] + 3)
    ax.set_xlabel("standing environments\nin training (%)")
    ax.set_ylabel(ylabel)
    ptf.write_panel_csv(out, name_, key, ["variant", "rse", "run", "seed", "n_bench", "x", "value"], panel)
  handles = [Line2D([], [], color=ptf.CHAIN_COLOR[v], marker="o", markersize=3.4, linewidth=0.8, label=NAMES[v])
             for v in ("A0", "A1a")]
  handles += [Line2D([], [], color="0.3", marker=m, linestyle="", markersize=3.4, label=f"seed {s}")
              for s, m in SHARE_MARKER.items()]
  ptf.bottom_legend(fig, handles, ncol=5, height=0.15)
  ptf.save_pgf(fig, out, name_)
  return facts


# --------------------------------------------------------------------------- lean ----
# Quiet-stand torso pitch on the robot (score_stand_attitude.score_entry, the ADR-0006 statistic) for every
# unloaded Run of 2026-09-16 and -23; joint_offset read from each Run's own _meta.json. The offset changes
# with a controller restart, not with the session, so the two sessions pool. Panel b: the simulated
# pitch-per-encoder-bias slopes from play.py --probe-lean (logs in data/2026-10-05-lean/).
LEAN_GLOBS = ("logs/robot_logs/2026_09_16-*/", "logs/robot_logs/2026_09_23-*/")
LEAN_PROBES = {"A0": "data/2026-10-05-lean/probe_A0_s42.log", "A1a": "data/2026-10-05-lean/probe_A1a_s42.log"}
LEAN_CATS = (("A0", True, "Flat, with offset"), ("A0", False, "Flat, no offset"),
             ("A1a", False, "Hierarchical"), ("A1a_DR", False, "Hierarchical + DR"))
LEAN_GROUPS = (("ankle_pitch", "ankle"), ("knee", "knee"), ("hip_pitch", "hip"), ("all", "all three"))


def lean_points(root):
  import pandas as pd
  from score_stand_attitude import score_entry
  rows = []
  for d in sorted({d for g in LEAN_GLOBS for d in root.glob(g)}):
    if not (d / "run.json").exists():
      continue
    run = json.loads((d / "run.json").read_text())
    variant, seed = ptf._variant_seed(run["policy_tag"])
    if variant not in ("A0", "A1a", "A1a_DR") or (run.get("payload_kg") or 0) > 0:
      continue
    flight = d / run["flight_recorder"]
    off = json.loads(flight.with_name(flight.stem + "_meta.json").read_text()).get("joint_offset") or []
    r = score_entry(pd.read_csv(flight), run.get("entry", 0))
    if r is None:
      continue
    rows.append({"variant": variant, "seed": seed, "run": d.name, "offset": bool(any(off)),
                 "pitch_deg": r["pitch_deg"], "still_pct": r["still_pct"], "n_rows": r["n"]})
  return rows


def lean_slopes(root):
  """d(stand pitch)/d(encoder bias) per joint group, least squares over the probe's bias sweep."""
  out = {}
  for v, f in LEAN_PROBES.items():
    line = next(l for l in (root / f).read_text().splitlines() if l.startswith("[LEANPROBE] {"))
    rows = json.loads(line[len("[LEANPROBE] "):])["rows"]
    for g, _ in LEAN_GROUPS:
      r = [x for x in rows if x["group"] == g]
      out[v, g] = float(np.polyfit([x["bias"] for x in r], [x["pitch"] for x in r], 1)[0])
  return out


def fig_lean(rows, slopes, out, name_="f_lean", left_only=False):
  if left_only:
    fig, ax = plt.subplots(figsize=(ptf.FIG_W, 3.0))
  else:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(ptf.FIG_W, 3.0), gridspec_kw={"width_ratios": [3, 2]})
  ax.axhline(0, color="0.6", linewidth=0.6, zorder=0)
  panel, facts = [], {}
  seed_look = {"42": "standing", "123": "walking"}   # f_upper's two looks (solid / hatched) carry the two seeds here
  w = 0.38
  for x, (v, on, _) in enumerate(LEAN_CATS):
    col = ptf.CHAIN_COLOR[v]
    for k, seed in enumerate(("42", "123")):
      cell = [r for r in rows if r["variant"] == v and r["offset"] == on and r["seed"] == seed
              and not np.isnan(r["pitch_deg"])]
      if not cell:
        continue
      x0 = x + (k - 0.5) * w
      m = float(np.mean([r["pitch_deg"] for r in cell]))
      ax.bar(x0, m, width=w * 0.92, color=col, alpha=ptf.REGIME_ALPHA[seed_look[seed]], edgecolor=col,
             linewidth=0.6, hatch=ptf.REGIME_HATCH[seed_look[seed]], zorder=1)
      jit = np.linspace(-0.11, 0.11, len(cell)) if len(cell) > 1 else [0.0]
      for dx, r in zip(jit, cell):
        ax.plot(x0 + dx, r["pitch_deg"], marker="o", markersize=3.2, linestyle="", markeredgecolor="white",
                markerfacecolor="0.15", markeredgewidth=0.5, zorder=3)   # a white rim keeps a dot readable on a dark bar
        panel.append({**r, "x": round(x0 + dx, 3)})
      facts[v, on, seed] = (m, len(cell))
  ax.set_xticks(range(len(LEAN_CATS)))
  ax.set_xticklabels([c[2] for c in LEAN_CATS], rotation=30, ha="right", rotation_mode="anchor")
  ax.set_xlim(-0.6, len(LEAN_CATS) - 0.4)
  ax.margins(y=0.14)   # a Run point on the axis edge reads as clipped
  ax.set_ylabel("quiet-stand torso pitch (deg)\n(negative = leaning back)")
  ptf.write_panel_csv(out, name_, "pitch", ["variant", "seed", "run", "offset", "pitch_deg", "still_pct", "n_rows",
                                            "x"], panel)
  if not left_only:
    bx.axhline(0, color="0.6", linewidth=0.6, zorder=0)
    w = 0.38
    srows = []
    for k, v in enumerate(("A0", "A1a")):
      xs = np.arange(len(LEAN_GROUPS)) + (k - 0.5) * w
      ys = [slopes[v, g] for g, _ in LEAN_GROUPS]
      bx.bar(xs, ys, width=w * 0.92, color=ptf.CHAIN_COLOR[v], zorder=2)
      srows += [{"variant": v, "group": g, "slope_rad_per_rad": round(y, 4)} for (g, _), y in zip(LEAN_GROUPS, ys)]
    bx.set_xticks(range(len(LEAN_GROUPS)))
    bx.set_xticklabels([l for _, l in LEAN_GROUPS], rotation=30, ha="right", rotation_mode="anchor")
    bx.set_xlabel("biased encoder(s)")
    bx.set_ylabel("simulated pitch per\nencoder bias (rad/rad)")
    ptf.write_panel_csv(out, name_, "slope", ["variant", "group", "slope_rad_per_rad"], srows)
  handles = [Patch(facecolor="0.55", edgecolor="0.3", alpha=ptf.REGIME_ALPHA[seed_look[sd]],
                   hatch=ptf.REGIME_HATCH[seed_look[sd]], label=f"seed {sd}") for sd in ("42", "123")]
  handles.append(Line2D([], [], color="0.15", marker="o", markersize=2.8, linestyle="", label="one run"))
  ptf.bottom_legend(fig, handles, ncol=3, height=0.12)
  ptf.save_pgf(fig, out, name_)
  return facts


# ----------------------------------------------------------------------- training ----
# Survival during training, every run on one machine (IAMS-ROS01, RTX 5070, 4096 envs, 10000 iterations)
# so wall-clock compares. The deployed keeper trained on the cluster, so its exact-config replicates
# (2026-10-05) stand in for it. "HL only": a new high level trained on the keeper's frozen low level.
TRAIN_RUNS = (
  ("A0", "42", "h1_2_velocity_v2/2026-09-14_16-33-58_a0_standing12_s42"),
  ("A0", "123", "h1_2_velocity_v2/2026-09-15_14-43-03_a0_standing12_s123"),
  ("A0", "7", "h1_2_velocity_v2/2026-10-05_17-02-36_a0_standing12_s7"),
  ("A1a", "42", "h1_2_velocity_a1_v2/2026-10-05_13-45-00_a1a_fullmirror_cot5_standing12_s42_repro3"),
  ("A1a", "7", "h1_2_velocity_a1_v2/2026-10-05_19-54-19_a1a_fullmirror_cot5_standing12_s7"),
  ("A1a", "123", "h1_2_velocity_a1_v2/2026-10-05_23-11-59_a1a_fullmirror_cot5_standing12_s123_repro2"),
  ("A1a_DR", "42", "h1_2_velocity_a1_v2/2026-09-09_11-05-04_a1a_hmem_cot5_keeperLL_standing12_s42"),
  ("A1a_DR", "123", "h1_2_velocity_a1_v2/2026-09-09_13-56-21_a1a_hmem_cot5_keeperLL_standing12_s123"),
  ("A1a_DR", "7", "h1_2_velocity_a1_v2/2026-10-06_14-30-54_a1a_hmem_cot5_keeperLL_standing12_s7"),
)
TRAIN_SMOOTH = 100          # iterations in the rolling mean, for the curve and the 90 % mark alike
TRAIN_XMAX = 4000           # every run stays within 1 % of its final value after this
TRAIN_LABEL = {"A0": "Flat", "A1a": "Hierarchical", "A1a_DR": "Hierarchical + DR (high level only)"}
TRAIN_SEED_LINE = {"42": "-", "123": "--", "7": ":"}   # seed rides on the line style: markers mean other things elsewhere


def train_points(root):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  rows = []
  for v, seed, d in TRAIN_RUNS:
    (f,) = (root / "logs/rsl_rl" / d).glob("events.out.tfevents*")
    ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
    ea.Reload()
    ev = ea.Scalars("Train/mean_episode_length")
    it = np.array([e.step for e in ev])[TRAIN_SMOOTH - 1:]
    ep = np.convolve([e.value * 0.02 for e in ev], np.ones(TRAIN_SMOOTH) / TRAIN_SMOOTH, "valid")  # steps -> s
    rows.append({"variant": v, "seed": seed, "run": Path(d).name, "host": f.name.split(".")[4],
                 "it": it, "ep_s": ep, "it90": int(it[np.argmax(ep >= 0.9 * ep[-1])]),
                 "wall_min": (ev[-1].wall_time - ev[0].wall_time) / 60})
  return rows


def fig_training(rows, out, name_="f_training"):
  fig, ax = plt.subplots(figsize=(ptf.FIG_W, 2.5))
  csv_rows = []
  for r in rows:
    col = ptf.CHAIN_COLOR[r["variant"]]
    m = r["it"] <= TRAIN_XMAX
    ax.plot(r["it"][m], r["ep_s"][m], color=col, linewidth=1.1, linestyle=TRAIN_SEED_LINE[r["seed"]], zorder=2)
    csv_rows += [{"variant": r["variant"], "seed": r["seed"], "run": r["run"], "iteration": int(i),
                  "episode_length_s": round(float(e), 4)} for i, e in zip(r["it"][m][::10], r["ep_s"][m][::10])]
  # each run's it90 is 90 % of ITS final value; the finals differ by < 0.1 s, so one line at 90 % of their mean
  ax.axhline(0.9 * np.mean([r["ep_s"][-1] for r in rows]), color="0.55", linewidth=0.7, zorder=1)
  ax.set_xlim(0, TRAIN_XMAX)
  ax.set_ylim(0, 21)
  ax.set_xlabel("training iteration (-)")
  ax.set_ylabel(f"episode length (s),\n{TRAIN_SMOOTH}-iteration mean")
  ptf.write_panel_csv(out, name_, "episode_length", ["variant", "seed", "run", "iteration", "episode_length_s"],
                      csv_rows)
  write_rows(out, f"{name_}_runs", [{k: r[k] for k in ("variant", "seed", "run", "host", "it90")}
                                    | {"wall_min": round(r["wall_min"], 1)} for r in rows])
  handles = [Line2D([], [], color=ptf.CHAIN_COLOR[v], linewidth=1.2, label=TRAIN_LABEL[v]) for v in TRAIN_LABEL]
  handles += [Line2D([], [], color="0.3", linewidth=1.1, linestyle=ls, label=f"seed {s}")
              for s, ls in TRAIN_SEED_LINE.items()]
  handles.append(Line2D([], [], color="0.55", linewidth=0.7, label="90 \\% of final value"))
  ptf.bottom_legend(fig, handles, ncol=3, height=0.22)
  ptf.save_pgf(fig, out, name_)


# -------------------------------------------------------------------- time series ----

ZOOM_S, ZOOM_LEAD_S = 20.0, 2.0   # the zoom window, and how much standing it keeps before walking starts


STARTUP_S, MIN_WINDOW_S = 10.0, 10.0   # the start-up window; the narrowest per-segment window


def walking_segments(b):
  """-> (segments, context arrays). A segment is a walking stretch of the command (gaps under 1 s
  merged): (t_start, t_end, description of the command in it, from its signed extremes)."""
  import pandas as pd
  df = pd.read_csv(b.flight, usecols=["t", "cmd_vx", "cmd_vy", "cmd_wz"])
  t = df["t"].to_numpy(float) - float(df["t"].iloc[0])
  cmd = df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float)
  walk = ptf.walking_mask(cmd)
  edges = np.flatnonzero(np.diff(np.r_[0, walk.astype(int), 0]))
  merged = []
  for a, e in zip(edges[::2], edges[1::2]):
    if merged and t[a] - merged[-1][1] < 1.0:
      merged[-1][1:3] = [t[e - 1], e]
    else:
      merged.append([t[a], t[e - 1], e, a])
  segs = []
  for t0, t1, e, a in merged:
    c = cmd[a:e]
    ext = [c[np.argmax(np.abs(c[:, k])), k] for k in range(3)]
    what = []
    if abs(ext[0]) > 0.05:
      what.append(f"{'forward' if ext[0] > 0 else 'backward'} {abs(ext[0]):.2f} m/s")
    if abs(ext[1]) > 0.05:
      what.append(f"{'left' if ext[1] > 0 else 'right'} sideways {abs(ext[1]):.2f} m/s")
    if abs(ext[2]) > 0.05:
      what.append(f"turn {'left' if ext[2] > 0 else 'right'} {abs(ext[2]):.2f} rad/s")
    segs.append((t0, t1, ", ".join(what)))
  G = ptf.read_mocap(b, df["t"].to_numpy(float))
  return segs, (t, cmd[:, 0], G["gt_vx"])


def zoom_window(b):
  """(t0, t1, context arrays): ZOOM_S around the Run's longest walking segment, starting
  ZOOM_LEAD_S before its onset so the onset is in view."""
  segs, ctx = walking_segments(b)
  a, _, _ = max(segs, key=lambda s_: s_[1] - s_[0])
  t0 = max(0.0, a - ZOOM_LEAD_S)
  return t0, t0 + ZOOM_S, ctx


def variant_windows(b):
  """Every walking segment (+/- ZOOM_LEAD_S, at least MIN_WINDOW_S wide) and the start-up, for the
  owner to choose from: [(tag, t0, t1, description)]."""
  segs, ctx = walking_segments(b)
  T = float(ctx[0][-1])
  out = [("startup", 0.0, STARTUP_S, "the policy start-up, standing")]
  for i, (a, e, what) in enumerate(segs, 1):
    t0, t1 = a - ZOOM_LEAD_S, e + ZOOM_LEAD_S
    if t1 - t0 < MIN_WINDOW_S:
      mid = 0.5 * (t0 + t1)
      t0, t1 = mid - MIN_WINDOW_S / 2, mid + MIN_WINDOW_S / 2
    t0, t1 = max(0.0, t0), min(T, t1)
    out.append((f"w{i}", t0, t1, f"walking segment {i} of {len(segs)}: {what}"))
  return out


def zoom(fig, t0, t1, ctx):
  """Crop the (shared) time axis to [t0, t1] and add a thin strip ABOVE the panels showing the whole
  Run (sagittal command + truth) with the window shaded."""
  main = [ax for ax in fig.axes if ax.get_visible()]
  for ax in main:
    ax.set_xlim(t0, t1)
  H = fig.get_figheight()
  top, bottom = max(ax.get_position().y1 for ax in main), min(ax.get_position().y0 for ax in main)
  strip, gap = 0.42 / H, 0.42 / H
  k = (top - strip - gap - bottom) / (top - bottom)
  for ax in main:
    p = ax.get_position()
    ax.set_position([p.x0, bottom + (p.y0 - bottom) * k, p.width, p.height * k])
  p = main[0].get_position()
  cax = fig.add_axes([p.x0, top - strip, p.width, strip])
  t, cmd, gt = ctx
  cmd_c, cmd_ls, _, _ = ptf.TRACK_STYLE["command"]
  tt, yy = ptf.envelope_downsample(t, gt, True)
  cax.plot(tt, yy, color=ptf.TRACK_STYLE["truth"][0], linewidth=0.4)
  cax.plot(t, cmd, color=cmd_c, linestyle=cmd_ls, linewidth=0.8)
  cax.axvspan(t0, t1, color="0.85", zorder=0, linewidth=0, gid="window")
  cax.set_xlim(t[0], t[-1])
  cax.set_yticks([])
  cax.set_ylabel("whole run", rotation=0, ha="right", va="center")
  cax.spines["left"].set_visible(False)
  cax.tick_params(axis="x", pad=1)
  return main, cax


def set_window(main, cax, t0, t1):
  """Move an already-zoomed figure to another window: panel x-limits and the shaded span."""
  for ax in main:
    ax.set_xlim(t0, t1)
  for p in [p for p in cax.patches if p.get_gid() == "window"]:
    p.remove()
  cax.axvspan(t0, t1, color="0.85", zorder=0, linewidth=0, gid="window")


def walk_errors(b):
  """(robot vs command, HL target vs command), planar RMS over the walking regime, m/s."""
  import pandas as pd
  df = pd.read_csv(b.flight, usecols=["t", "cmd_vx", "cmd_vy", "cmd_wz"])
  G = ptf.read_mocap(b, df["t"].to_numpy(float))
  cmd = df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float)
  walk = ptf.walking_mask(cmd)
  e = (cmd[:, 0] - G["gt_vx"]) ** 2 + (cmd[:, 1] - G["gt_vy"]) ** 2
  robot = float(np.sqrt(np.nanmean(e[walk])))
  if b.hrl is None:
    return robot, None
  H = pd.read_csv(b.hrl, usecols=["cmd_vx", "cmd_vy", "cmd_wz", "tgt0", "tgt1"])
  hw = ptf.walking_mask(H[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float))
  t = (H["tgt0"] - H["cmd_vx"]) ** 2 + (H["tgt1"] - H["cmd_vy"]) ** 2
  return robot, float(np.sqrt(t[hw].mean()))


# -------------------------------------------------------------------------- LaTeX ----

TEX_HEAD = r"""%% Generated by scripts/plot_report_figures.py -- edit the captions THERE, this file is overwritten.
% Typography of the IST report template (KOMA scrbook, A5, 9 pt, Palatino with Latin Modern Sans), without
% its bibliography and tikz-external machinery, so the preview compiles alone. The figures' pgf text is measured
% with the same fonts (SANS below), otherwise a label that fits in the preview overruns in the report.
\documentclass[english,paper=a5,headsepline=true,9pt,DIV=12,BCOR=0.7cm]{scrbook}
\usepackage{typearea}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[sc,osf]{mathpazo}
\usepackage[T1]{fontenc}
\usepackage[main=english]{babel}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx,pgf}
\usepackage{microtype}
\frenchspacing
\usepackage{caption}
\DeclareCaptionLabelFormat{cont}{#1~#2 (continued)}
\captionsetup[ContinuedFloat]{labelformat=cont}
\setcapindent{0em}
\addtokomafont{captionlabel}{\bfseries}
\pagestyle{headings}
\usepackage{booktabs,tabularx,array}
\begin{document}
\section*{Report figures}
Hardware runs are from 2026-09-16, seed 42 and unloaded unless stated. Flat = the flat PPO policy (A0),
Hierarchical = the HIRO hierarchy (A1a), Hierarchical + DR = A1a trained with payload domain randomisation.
"""


def zoom_note(w):
  t0, t1, T = w
  return (f"Panels show {t1 - t0:.0f}\\,s of the {T:.0f}\\,s run (t = {t0:.0f} to {t1:.0f}\\,s, the "
          "longest walking segment); the strip on top is the whole run (sagittal command and ground truth, "
          "the window shaded). Error and saturation figures are over the whole run.")


CHAIN_LOSS_NOTE = ("Grey numbers: loss from training plant to deployment plant, (deployment $-$ training) / |training| of the two stage "
                   "means, in percent and signed so that positive is worse (all panels here are lower-is-better). "
                   "Where some runs at the deployment plant fell the number covers the runs that stayed upright; ``fell'' marks a "
                   "policy none of whose deployment-plant runs did.")
CHAIN_LOSS_NOTE_ROBOT = ("Grey numbers: loss from training plant to robot, (robot $-$ training) / |training| of the two stage "
                         "means, in percent and signed so that positive is worse (all panels here are "
                         "lower-is-better). The robot ran operator commands, not the replayed schedule, so this "
                         "number mixes the sim-to-real gap with the difference in commands; the deployment plant is "
                         "not part of it.")
REWARD_LOSS_NOTE = ("Grey numbers: loss from training plant to deployment plant, (training $-$ deployment) / |training| of the two stage "
                    "means, in percent; positive means a lower reward at the deployment plant.")
REWARD_LOSS_NOTE_ROBOT = ("Grey numbers: loss from training plant to robot, (training $-$ robot) / |training| of the two stage "
                          "means, in percent; positive means a lower reward on the robot. The robot ran operator "
                          "commands, not the replayed schedule, so this number mixes the sim-to-real gap with the "
                          "difference in commands; the deployment plant is not part of it.")


def replication_table(root):
  """Schedule A vs schedule B on the pre-registered criteria, every number from REPLICATION."""
  if not (root / REPLICATION).exists():
    print(f"[REPORT] {REPLICATION} missing: replication table skipped (run score_replication.py)")
    return ""
  R = json.loads((root / REPLICATION).read_text())
  pc = lambda x: "$0$" if round(100 * x) == 0 else f"${100 * x:+.0f}$"
  span = lambda r, fmt=pc, unit="\\,\\%": f"{fmt(r[0])} to {fmt(r[1])}{unit}"
  num = lambda x, n=2: f"${x:.{n}f}$"
  rr = lambda r, n=2: f"{num(r[0], n)}--{num(r[1], n)}"
  x_ = lambda r: f"${r[0]:.2f}$--${r[1]:.2f}\\times$"

  def where(cmd):
    vx, vy, wz = cmd
    return ("at a stop" if max(abs(vx), abs(vy), abs(wz)) <= 0.1 else "walking sideways" if abs(vy) > 0.1
            else "turning" if abs(wz) > 0.1 else f"at {vx:g}\\,m/s")

  def col(s):
    x = R[s]
    speeds = ", ".join(f"{v:g}" for v in x["speeds"])
    n_runs = sorted(set(x["bridge_runs"].values()))
    hier = [t for t in x["bridge_runs"] if t.startswith("A1a_s")]    # Hierarchical without DR: the one that falls
    others = sum(v for t, v in x["bridge_runs"].items() if t not in hier)
    fw = {}
    for t, fl in x["fall_where"].items():
      for _, cmd in fl:
        fw.setdefault(t, []).append(where(cmd))
    wh = lambda t: f" ({', '.join(f'{fw[t].count(w)} {w}' for w in dict.fromkeys(fw[t]))})" if t in fw else ""
    seeds = lambda v: [t for t in x["ll"] if ptf._variant_seed(t)[0] == v and x["ll"][t] is not None]
    lls = lambda v: (f"{name(v)} s{' / s'.join(ptf._variant_seed(t)[1] for t in seeds(v))} "
                     f"{' / '.join(pc(x['ll'][t][0]) for t in seeds(v))}\\,\\%")
    ll = x["ll"]
    share = min(v[1] for v in ll.values() if v is not None and v[0] < 0)
    return [
      f"{x['duration_s']:.0f}\\,s ({x['walking_s']:.0f}\\,s walking); sagittal {speeds}\\,m/s; "
      f"{x['bench_envs']} training-plant environments, {'/'.join(map(str, n_runs))} deployment-plant runs per policy",
      f"flat {span(x['cot_loss']['flat'])}, hierarchies {span(x['cot_loss']['hier'])}",
      f"training plant: hierarchies {rr(x['cot']['bench']['hier'])} $<$ flat {rr(x['cot']['bench']['flat'])}"
      f"\\newline deployment plant: hierarchies {rr(x['cot']['bridge']['hier'])} $>$ flat {rr(x['cot']['bridge']['flat'])}",
      ", ".join(f"Hierarchical s{ptf._variant_seed(t)[1]} ${x['falls'][t]}/{x['bridge_runs'][t]}${wh(t)}"
                for t in hier) + f", all others ${sum(x['falls'][t] for t in x['falls'] if t not in hier)}/{others}$"
      f"\\newline training plant: ${x['bench_falls'][0]}$ of ${x['bench_falls'][1]}$ environments",
      f"flat {x_(x['stand_ratio']['flat'])}, hierarchies {x_(x['stand_ratio']['hier'])}",
      f"deployment plant: flat {rr(x['err_vx']['bridge']['flat'], 3)} $<$ hierarchies "
      f"{rr(x['err_vx']['bridge']['hier'], 3)}\\,m/s",
      f"{lls('A1a')}, {lls('A1a_DR')}; goal tracking carries $\\geq {100 * share:.0f}\\,\\%$ of each loss",
    ]

  sg, sa = R["sign"]["figure"], R["sign"]["all"]
  metric = {k: v.split(" (")[0] for k, v in CHAIN_LABEL.items()}
  flips = "; ".join(f"{name(t)} s{ptf._variant_seed(t)[1]} {rg} "
                    f"{metric.get(k, k)} ({pc(a)} vs {pc(b)}\\,\\%)" for t, rg, k, a, b in sa["flips"])
  crit = ["Command schedule", "Walking cost of transport, training plant $\\rightarrow$ deployment plant",
          "Walking energy ranking", "Falls (deployment-plant runs)", "Standing leg action rate, deployment plant $\\div$ training plant",
          "Walking sagittal error vs.\\ true velocity", "Walking low-level reward $r^\\mathrm{lo}$, "
          "training plant $\\rightarrow$ deployment plant"]
  rows = list(zip(crit, col("A"), col("B")))
  rows.append(("Sign of the training $\\rightarrow$ deployment plant change, every metric",
               f"(reference: cells whose change exceeds the deployment-plant run-to-run spread, "
               f"{100 * R['thr']['walking']:.0f}\\,\\% walking, {100 * R['thr']['standing']:.0f}\\,\\% standing)",
               f"${sg['agree']}$ of ${sg['n']}$ cells (the {len(CHAIN_SEED_TAGS)} policies of Figs.~\\ref{{fig:chain_sim_walking}} "
               f"and \\ref{{fig:chain_sim_standing}})"
               + (f"; ${sa['agree']}$ of ${sa['n']}$ over all {len(R['A']['bridge_runs'])}" if sa != sg else "")
               + (f", the exception{'s' if len(sa['flips']) > 1 else ''}: {flips}" if flips else "")
               + f"; walking losses move by a median of {R['walk_diff_pp'][0]:.0f} (at most "
               f"{R['walk_diff_pp'][1]:.0f}) percentage points"))
  holds = R["holds"]["B"]
  tags = list(R["A"]["bridge_runs"])
  n_flat = sum(t.startswith("A0") for t in tags)
  new = sorted(set(R["B"]["speeds"]) - set(R["A"]["speeds"]))
  verdict = ("All criteria hold" if all(holds.values()) else
             "Criteria that do not hold: " + ", ".join(k for k, v in holds.items() if not v))
  body = " \\\\\n    \\midrule\n    ".join(" & ".join(r) for r in rows)
  return f"""
\\begin{{table}}[p]
  \\centering\\small
  \\begin{{tabularx}}{{\\textwidth}}{{>{{\\raggedright\\arraybackslash}}p{{3.2cm}} >{{\\raggedright\\arraybackslash}}X >{{\\raggedright\\arraybackslash}}X}}
    \\toprule
    \\textbf{{Criterion}} & \\textbf{{Schedule A}} & \\textbf{{Schedule B}} \\\\
    \\midrule
    {body} \\\\
    \\bottomrule
  \\end{{tabularx}}
  \\caption{{Replication of the comparison of training and deployment plant on a second command schedule (B: new speeds
  {", ".join(f"{v:g}" for v in new)}\\,m/s, speed and direction changes without a stop), scored on criteria fixed before
  any schedule-B data existed; {verdict[0].lower() + verdict[1:]}. Ranges span the {n_flat} flat and {len(tags) - n_flat}
  hierarchical policies (seeds 42, 123 and 7 each); a policy whose deployment-plant runs all fell has no deployment-plant value.}}
  \\label{{tab:replication}}
\\end{{table}}
\\clearpage
"""


def write_latex(out, facts):
  """report_figures.tex: every figure \\input with its caption. Caption numbers come from `facts`."""
  f = lambda x, n=2: f"{x:.{n}f}"
  figs = []
  for key in ("A1a", "A0", "A1a_DR"):
    r = facts[key]["rms"]
    figs.append((f"f_track_{key}", f"fig:track_{key}",
      f"Velocity tracking of the {NAMES[key]} policy on the robot (one run). Operator command "
      "(grey), onboard velocity estimate (ochre; for yaw rate the gyro, a direct measurement) and "
      "motion-capture ground truth (black). Estimator RMS error against ground truth while walking: "
      f"{f(r['vx']['walking'])} m/s sagittal, {f(r['vy']['walking'])} m/s lateral, "
      f"{f(r['wz']['walking'])} rad/s yaw rate; while standing {f(r['vx']['standing'])}, "
      f"{f(r['vy']['standing'])} and {f(r['wz']['standing'])}. Ground truth is missing for "
      f"{r['vx']['gap']:.1f}\\,\\% of samples. {zoom_note(facts[key]['window'])} "
      "Y-limits are shared across the three tracking figures."))
  for key in ("A1a", "A1a_DR"):
    x = facts[key]
    figs.append((f"f_hier_{key}", f"fig:hier_{key}",
      f"The hierarchy's error decomposition for the {NAMES[key]} policy, same run as "
      f"Fig.~\\ref{{fig:track_{key}}}: operator command (grey), high-level (HL) target (green), velocity "
      "delivered by the low level as read by the onboard estimate (ochre) and motion-capture ground truth "
      "(black). The sagittal and lateral HL target is the velocity CHANGE the HL requests over its next "
      "decision window, not a velocity: on the robot its velocity state is the IMU increment since its last "
      "decision, zero at each decision, so the green trace is read against zero, not against the command. "
      "The yaw-rate target is absolute. "
      f"Bottom: the HL's signed action $\\mathbf{{g}}$, the velocity change divided by the goal scale "
      f"fixed at training ({x['gs'][0]:.2f} m/s sagittal, {x['gs'][1]:.2f} m/s lateral); $\\pm 1$ (dotted lines) is the largest "
      f"change it can request in one decision, in either direction. Each component of $\\mathbf{{g}}$ is at that limit in {x['sat']['vx']:.0f}\\,\\% (sagittal) "
      f"and {x['sat']['vy']:.0f}\\,\\% (lateral) of walking steps. "
      f"{zoom_note(x['window'])}"))
  def units(regime):
    """(bench envs per policy, deployment-plant runs per policy, 'who fell' sentence), from the scored units."""
    c = facts["chain"][regime]
    span = lambda v: str(v[0]) if len(set(v)) == 1 else f"{min(v)} to {max(v)}"
    runs = lambda v: span(v) + (" run" if set(v) == {1} else " runs")
    fell = [f"{name(t, True)} at the {STAGE_LABEL[st]} ({k} of {n})" for (t, st), (n, k) in c.items() if k]
    return (span([c[t, "bench"][0] for t in CHAIN_SEED_TAGS]), runs([c[t, "bridge"][0] for t in CHAIN_SEED_TAGS]),
            "A unit that fell is excluded from the means: " + "; ".join(fell) + "." if fell
            else "No unit fell.")
  sim = ("Both stages run one identical 253\\,s command schedule (stand, sagittal 0.3, 0.5 and 1.0 m/s, "
         "backward, sideways, turning, walk-turn-stop) and are scored by the same code. Training plant: "
         "{nb} environments per seed in the training environment (circle; the environments differ in "
         "randomised foot friction, encoder bias and torso centre of mass). Deployment plant: the exported policy in the robot's C++ controller with the onboard "
         "estimator, against the shipped MuJoCo plant ({nr} per seed; square: mean of the runs, which "
         "repeat one plant; single runs are not drawn, the largest range between the runs of one seed is "
         "{bsp:.0f}\\,\\% of their mean on any panel here and every run is in the panel data).")
  sim_rest = (" {fell} Commanded joint targets are unclipped at the training plant and clipped to the joint limits by the "
              "controller at the deployment plant.")
  cseeds = ("Each policy is the mean over its three seeds (42, 123, 7) of the seeds' own means; a seed with no "
            "unit left at a stage drops out of that stage's mean, which then covers fewer seeds.")
  for regime in ("walking", "standing"):
    nb, nr, fell = units(regime)
    cap = (f"{regime.capitalize()} metrics at the two simulated deploy stages"
      + (", in two parts: this part shows cost of transport, mechanical power, stride period and the yaw-rate, "
         "sagittal and lateral velocity errors; the next shows the smoothness and posture metrics" if regime == "walking" else "") + ". "
      + sim.format(nb=nb, nr=nr, bsp=facts["bridge_spread_pct"][regime]))
    rest = (sim_rest.format(fell=fell) + " Velocity errors are against the simulator's true pelvis velocity, read in the IMU (torso) frame; "
      "yaw-rate error, orientation deviation" + (" and roll/pitch rate are" if regime == "walking" else " are")
      + " the IMU's. Leg joint acceleration is the second "
      "difference of measured joint position at the control rate."
      + (" Cost of transport is the undirected, diagnostic form. Stride period is the realized interval "
         "between the same foot's steps, recovered from the dominant period of the knee joint. Joint speed is the "
         "mean absolute measured joint velocity over all 27 joints." if regime == "walking" else "")
      + " " + cseeds + " " + CHAIN_LOSS_NOTE)
    cap_all = cap + rest
    if regime == "standing":
      figs.append((f"f_chain_sim_{regime}", f"fig:chain_sim_{regime}",
        "Standing metrics at the two simulated deploy stages; stages, markers and grey numbers as in "
        "Fig.~\\ref{fig:chain_sim_walking}. The largest range between the runs of one seed "
        f"at the deployment plant is {facts['bridge_spread_pct'][regime]:.0f}\\,\\% of their mean on any panel. "
        + fell + " Velocity errors are against the simulator's true pelvis velocity, read in the IMU (torso) frame; "
        "yaw-rate error and orientation deviation are the IMU's. Leg joint acceleration is the second difference of "
        "measured joint position at the control rate. " + cseeds))
    else:
      figs.append(("f_chain_sim_walking_a", "fig:chain_sim_walking", cap))
      figs.append(("f_chain_sim_walking_b", None, "Smoothness and posture metrics; rows and markers as in the first part." + rest))
  nb, nr, fell = units("walking")
  cap = (
    "Walking metrics at three deploy stages. Training plant and deployment plant as in "
    f"Fig.~\\ref{{fig:chain_sim_walking}} (one identical 253\\,s command schedule, one scorer; {nb} "
    f"environments at the training plant and {nr} at the deployment plant per seed). {fell} Robot: runs of 2026-09-16 under operator commands "
    "(diamond: mean of the runs), flown with seeds 42 and 123 only, so the robot mean covers two seeds and the "
    "simulated stages three. The robot is not command-matched to the simulated stages and a "
    "row shows a range, not a trend.")
  rest = (" " + cseeds + " One hierarchical s42 robot run is omitted as an outlier (1.25 to 2.9 "
    "times its policy median on six of eight metrics). Cost of transport is the undirected, diagnostic form; "
    "its distance is the simulator's true velocity in simulation and motion capture or the onboard estimate "
    "on the robot. Leg joint acceleration is the second difference of measured joint position at the control "
    "rate at all three stages. Joint speed is the mean absolute measured joint velocity over all 27 joints; on "
    "the robot it also follows the operator's walking speed, which is not matched across policies. " + CHAIN_LOSS_NOTE_ROBOT)
  figs.append(("f_chain_clean_a", "fig:chain", cap))
  figs.append(("f_chain_clean_b", None, "Orientation deviation, roll/pitch rate and joint speed; rows and markers as in the "
               "first part." + rest))
  n = facts["upper_n"]
  figs.append(("f_upper_3pol", "fig:upper",
    "Upper-body joint speed and pose deviation on the robot, standing and walking. Bars: mean over all "
    "unloaded runs of the policy, both seeds, sessions of 2026-09-14 and -16 "
    f"(n = {n['A0']} / {n['A1a']} / {n['A1a_DR']} runs); dots: single runs; hollow: less than 10 s in "
    "the regime."))
  ENV_OBJ = ("This is the environment reward, the training objective of the flat policies; the hierarchical "
             "policies are scored on it but do not train on it (their low level: "
             "Fig.~\\ref{fig:ll_reward_sim}; their high level trains on the two tracking terms shown here "
             "minus a windowed cost of transport).")
  ln = facts["ll_n"]
  lspan = lambda v: str(v[0]) if len(set(v)) == 1 else " / ".join(map(str, v))
  for sfx in ("_sim", ""):
    figs.append((f"f_ll_reward{sfx}", f"fig:ll_reward{sfx}",
      "The hierarchical policies on the objective their low level trains on, "
      "$r^\\mathrm{lo} = r^\\mathrm{track} + r^\\mathrm{shape} + r^\\mathrm{cad}$ per step, with the "
      "coefficients of the low level all four run (the DR policies freeze the Hierarchical s42 low level). "
      "$r^\\mathrm{track}$: four exponential kernels on the high-level target for a step against the goal "
      "state after it (planar velocity, yaw rate, orientation, height; between 0 and 4). "
      "$r^\\mathrm{shape}$: posture over all joints and energy (cost of transport) at the low level's own "
      "weights, plus nine terms equal to the environment's (stand-still, roll/pitch rate, action rate, joint "
      "acceleration, joint limits, angular momentum, foot clearance, foot slip, soft landing). "
      "$r^\\mathrm{cad}$: the gait term on the gait clock the high level commands. Higher is better. "
      f"Training plant: true state, {lspan(ln['bench'])} environments, a separate rollout on the 253\\,s "
      "schedule of Fig.~\\ref{fig:chain_sim_walking}; deployment plant: the state the controller feeds the low level "
      f"(onboard estimate) and the logged and plant-replayed terms, {lspan(ln['bridge'])} runs per policy that "
      "did not fall."
      + ("" if sfx else f" Robot: n = {lspan(ln['hardware'])} runs under operator commands. The robot has no "
         "foot contact sensing, so this figure shows only $r^\\mathrm{track}$ and the contact-free part of "
         "$r^\\mathrm{shape}$ (stand-still, roll/pitch rate, action rate, posture, energy), the same part at "
         "all three stages; the full decomposition is Fig.~\\ref{fig:ll_reward_sim}.")
      + ("" if not facts["ll_partial"] else " Goal tracking only (the shaping terms could not be recomputed): "
         + "; ".join(f"{k} {STAGE_LABEL[s]} run of {name(t, seed=True)}" for t, s, k in facts["ll_partial"]
                     if s != "bench") + ".")
      + " The yaw-rate goal state is the pelvis rate in training and at the training plant but the torso gyro (which "
      "also carries the waist-yaw motion) in the controller. The last step of every high-level decision "
      "window is not scored in $r^\\mathrm{track}$ at any stage (on the robot the velocity state resets at the "
      "next decision). " + (REWARD_LOSS_NOTE if sfx else REWARD_LOSS_NOTE_ROBOT)))
  HL_OBJ_TAIL = ("High-level reward: the linear-velocity and yaw-rate tracking terms per step minus 5 times the "
                 "cost of transport of each 0.16\\,s decision window (the high level's own decision times; the "
                 "distance is along the command; velocity from the simulator truth, or motion capture on the "
                 "robot). The objectives of Flat and the hierarchical policies differ, so absolute values do not "
                 "compare between them; the grey numbers do. The hierarchical low level trains on a third "
                 "objective, Fig.~\\ref{fig:ll_reward_sim}.")
  sf = facts["reward_seed"]
  seven = ("the seven reward terms that can be recomputed from a log (linear-velocity and yaw-rate tracking, "
           "torso tilt, roll/pitch rate, default pose, action rate, stand-still)")
  simsrc = (f"the training plant (the live reward, {sf['bench']} environments per seed stepping the 253\\,s "
            f"command schedule) and the deployment plant (recomputed from the logs of {sf['bridge']} runs per seed of the "
            "same schedule)")
  seeds = ("Each policy is the mean over its three seeds (42, 123, 7) of the seeds' own means. A stage is averaged over the seeds that have it: runs that fell are "
           "excluded. At the deployment plant " + (sf["fell"] if sf["fell"] else "no run fell")
           + "; a seed with no run left drops out of the deployment-plant mean, which then covers fewer seeds. The robot "
           "flew seeds 42 and 123 only.")
  for sfx, where, lab in (("_sim", simsrc + ".", "_sim"),
                          ("", simsrc + f", and the robot (recomputed from the logs; environment reward: "
                               f"{sf['robot_env']}; high-level reward: {sf['robot_hl']}).", "")):
    if sfx:
      full16 = ("Flat is scored on the environment (extrinsic) reward over all 16 active terms (17 are "
                "defined, cost of transport has weight 0): live at the training plant; at the deployment plant the "
                "seven terms that can be recomputed from the log (" + seven.split("(")[1].rstrip(")") + ") plus nine "
                "contact and whole-body terms recomputed by replaying the logged plant state in the deployment plant's own "
                "MuJoCo model (validated against the live reward on training-plant states). The hierarchical policies "
                "do not train on this reward: they are scored on the reward their high level trains on, so "
                "the rows have different objectives. ")
      cap_return = ("Episode return (per-step reward times the 20 s episode, the scale of the training return), "
                    f"at {where} " + full16 + HL_OBJ_TAIL + " " + seeds)
    else:
      cap_return = ("Episode return (per-step reward times the 20 s episode, the scale of the training return). "
                    "Flat has two rows: the seven terms reproducible at all three stages, and all 16 active "
                    "terms at the training plant and the deployment plant (the robot has no contact sensing, so no "
                    "16-term value); the hierarchical policies one row, the reward their high level trains on. "
                    + HL_OBJ_TAIL + " " + seeds + f" {sf['est']} of the hierarchical robot runs have no motion capture and "
                    "use the onboard velocity estimate.")
    figs.append((f"f_reward_return{sfx}", f"fig:reward_return{lab}", cap_return + " " + (REWARD_LOSS_NOTE if sfx else REWARD_LOSS_NOTE_ROBOT)))
    for regime in ("walking", "standing"):
      figs.append((f"f_reward_terms_{regime}{sfx}", f"fig:reward_terms_{regime}{lab}",
        f"The reward terms of Fig.~\\ref{{fig:reward_return{lab}}} one by one while {regime} (weight times raw "
        "value, per step; higher is better, penalties are negative). " + ENV_OBJ
        + ("" if sfx else " On the robot, linear-velocity tracking lacks the vertical-velocity part (not "
           "measured) and the roll/pitch rate is the body-frame gyro.") + " " + seeds + " " + (REWARD_LOSS_NOTE if sfx else REWARD_LOSS_NOTE_ROBOT)))
  tr = facts["train"]
  by = lambda v, k: [r[k] for r in tr if r["variant"] == v]
  span = lambda x, n=0: (f"{min(x):.{n}f}" if f"{min(x):.{n}f}" == f"{max(x):.{n}f}"
                         else f"{min(x):.{n}f} to {max(x):.{n}f}")
  wa, wh = np.mean(by("A0", "wall_min")), np.mean(by("A1a", "wall_min"))
  its = lambda v: ", ".join(str(r["it90"]) for r in sorted((r for r in tr if r["variant"] == v), key=lambda r: r["it90"]))
  figs.append(("f_training", "fig:training",
    "Survival during training: episode length (20\\,s = an episode that runs to its time limit), "
    f"{TRAIN_SMOOTH}-iteration rolling mean, one line per training run (colour: policy; solid, dashed, dotted: seed 42, 123, 7). The grey line is "
    f"90\\,\\% of the runs' mean final value (the finals span {np.ptp([r['ep_s'][-1] for r in tr]):.2f}\\,s); the iteration where each run first reaches "
    "90\\,\\% of its own final value is in the panel data. Every run trains from scratch except Hierarchical + DR, a new high level on the "
    "frozen low level of the hierarchical policy. The hierarchical runs reach 90\\,\\% at iterations "
    f"{its('A1a')}, the flat runs at {its('A0')}: two of three flat seeds stay near 6\\,s for about 1000 "
    "iterations, so convergence speed is decided by the seed for the flat policy, and the hierarchy is not "
    "slower. Hierarchical + DR reaches it at " f"{its('A1a_DR')}. Wall-clock for 10\\,000 iterations on one RTX 5070 with 4096 environments: flat "
    f"{span(by('A0', 'wall_min'))} min, hierarchical {span(by('A1a', 'wall_min'))} min "
    f"({100 * (wh / wa - 1):+.0f}\\,\\%), a new high level alone {span(by('A1a_DR', 'wall_min'))} min. One "
    "iteration is 4096 environments times 24 steps. The deployed hierarchical policy was trained on another "
    "machine; the three hierarchical runs here repeat its configuration on this one (seeds 42, 7, 123). After "
    f"iteration {TRAIN_XMAX} every run stays within 1\\,\\% of its final value."))
  c, sl = facts["lean"]["cell"], facts["lean"]["slope"]
  nl = lambda v, on: " / ".join(str(c.get((v, on, s), (0, 0))[1]) for s in ("42", "123"))
  figs.append(("f_lean", "fig:lean",
    "Standing upright needs a hand calibration for the flat policy and none for the hierarchy. Left: torso "
    "pitch on the robot at zero command (IMU, still rows only: no leg joint faster than 0.1\\,rad/s), every "
    "unloaded run of 2026-09-16 and -23; bars are the mean per policy and seed (solid: seed 42, hatched: seed 123), dots are single runs. Number of runs per seed (42 / 123): "
    f"Flat with offset {nl('A0', True)}, without {nl('A0', False)}, Hierarchical {nl('A1a', False)}, "
    f"Hierarchical + DR {nl('A1a_DR', False)}. Right: simulated standing pitch per unit of encoder bias on "
    "one sagittal leg joint or all three (slope over $\\pm0.032$\\,rad, seed 42, 64 environments); flat "
    f"{sl['A0', 'all']:+.2f}, hierarchical {sl['A1a', 'all']:+.2f}\\,rad/rad for all three joints."))
  figs.append(("f_lean_left", "fig:lean_left",
    "Standing upright needs a hand calibration for the flat policy and none for the hierarchy. Torso "
    "pitch on the robot at zero command (IMU, still rows only: no leg joint faster than 0.1\\,rad/s), every "
    "unloaded run of 2026-09-16 and -23; bars are the mean per policy and seed (solid: seed 42, hatched: seed 123), dots are single runs. Number of runs per seed (42 / 123): "
    f"Flat with offset {nl('A0', True)}, without {nl('A0', False)}, Hierarchical {nl('A1a', False)}, "
    f"Hierarchical + DR {nl('A1a_DR', False)}."))
  c, srow = facts["share"]["cell"], facts["share"]["rows"]
  lo, mid = min(r["rse"] for r in srow), sorted({r["rse"] for r in srow})[1]
  tds = [r["td_per_foot"] for r in srow if r["variant"] == "A1a" and r["rse"] == lo]
  flat = [c["act_legs_rad", "A0", s][0] for s in sorted({r["rse"] for r in srow})]
  rep = [r["act_legs_rad"] for r in srow if r["variant"] == "A0" and r["rse"] == mid]
  ncell = lambda v: " / ".join(str(c["act_legs_rad", v, s][1]) for s in sorted({r["rse"] for r in srow}))
  hi = max(r["rse"] for r in srow)
  hier = lambda s: [r for r in srow if r["variant"] == "A1a" and r["rse"] == s]
  span = lambda rs, k, f: f"{f.format(min(r[k] for r in rs))} to {f.format(max(r[k] for r in rs))}"
  worst = max(hier(mid), key=lambda r: r["act_legs_rad"])
  figs.append(("f_standing_share", "fig:standing_share",
    "Standing in simulation against the share of standing environments in training, a crossed design: both "
    "architectures trained at 5, 12 and 20\\,\\% standing environments, otherwise unchanged. Bench at zero "
    "command, 64 environments, 600 steps (12\\,s), two rollout seeds. With "
    f"{100 * lo:g}\\,\\% the hierarchical policy marches in place: standing leg action rate "
    f"{c['act_legs_rad', 'A1a', lo][0]:.3f} rad against the flat policy's {c['act_legs_rad', 'A0', lo][0]:.4f} "
    f"({c['act_legs_rad', 'A1a', lo][0] / c['act_legs_rad', 'A0', lo][0]:.0f} times), and "
    f"{min(tds):.0f} to {max(tds):.0f} touchdowns per foot where a still stand gives 1 (the floor). At "
    f"{100 * mid:g}\\,\\% it depends on the checkpoint: {span(hier(mid), 'act_legs_rad', '{:.4f}')} rad and "
    f"{span(hier(mid), 'td_per_foot', '{:.2f}')} touchdowns per foot across its {len(hier(mid))} checkpoints, the "
    f"seed-{worst['seed']} one standing with extra steps ({worst['act_legs_rad']:.4f} rad, "
    f"{worst['td_per_foot']:.2f} touchdowns per foot). At {100 * hi:g}\\,\\% its {len(hier(hi))} checkpoints "
    f"stand at {span(hier(hi), 'act_legs_rad', '{:.4f}')} rad and {span(hier(hi), 'td_per_foot', '{:.2f}')} "
    "touchdowns per foot. The flat policy never steps at any share and "
    f"moves within {min(flat):.4f} to {max(flat):.4f} rad, against {min(rep):.4f} to {max(rep):.4f} rad across "
    f"its {len(rep)} checkpoints at {100 * mid:g}\\,\\% alone. Touchdowns per foot = 12\\,s / the training plant's stride period. "
    "Markers: one trained checkpoint each (circle seed 42, triangle seed 123, diamond seed 7; repeated evaluations "
    "of one checkpoint averaged, an independent retrain of seed 42 drawn separately; the seed-7 checkpoints "
    "were evaluated in a later batch on the same protocol), n = "
    f"{ncell('A0')} Flat and {ncell('A1a')} Hierarchical; lines join the means. The flat 5\\,\\% "
    "checkpoint is an earlier training of the same recipe whose critic lacks the five privileged "
    "environment parameters; its actor configuration is identical. Leg action rate on a log axis."))
  c, pn = facts["payload"]["cell"], facts["payload"]["n"]
  kgs = sorted({kg for _, kg in pn})
  ns = lambda v: " / ".join(str(pn.get((v, kg), 0)) for kg in kgs)
  top = kgs[-1]
  a, b = c["act_legs_rad", "A1a", top][0], c["act_legs_rad", "A1a_DR", top][0]
  e, g = c["err_vx_est", "A1a", top][0], c["err_vx_est", "A1a_DR", top][0]
  figs.append(("f_payload", "fig:payload",
    "The robot carrying a backpack payload (runs of 2026-09-23). Both policies run the same frozen low level "
    "and differ only in the seed-42 high level, trained without (Hierarchical) or with (Hierarchical + DR) "
    f"payload randomisation over 0 to {PAYLOAD_TRAIN_MAX:g}\\,kg. At {top:g}\\,kg, beyond that range, the high "
    f"level trained without randomisation loses the stand: standing leg action rate {a:.3f} against {b:.3f} rad "
    f"({a / b:.1f} times) and sagittal velocity error {e:.2f} against {g:.2f} m/s at zero sagittal command. It "
    "could not be commanded to walk (n/a), and both its runs were stopped after less than 13\\,s to avoid a "
    "fall, so its values understate the failure. The operator log records that both Hierarchical + DR runs "
    "at that load fell against the wall, after standing and walking under command. Markers: single runs (n = "
    f"{ns('A1a')} Hierarchical and {ns('A1a_DR')} Hierarchical + DR at "
    + " / ".join(f"{kg:g}" for kg in kgs) + "\\,kg); bars: mean of the runs with at least 10\\,s in the "
    "regime; hollow: shorter, drawn but not in the mean. Velocity is the onboard EKF estimate (no motion "
    f"capture in this session). Cost of transport is normalised by robot plus payload mass ({ROBOT_KG:g}\\,kg + "
    "payload); the operator's commands were not speed-matched across runs."))
  var = []
  for kind, keys, ref in (("track", ("A1a", "A0", "A1a_DR"), "track"), ("hier", ("A1a", "A1a_DR"), "hier")):
    for key in keys:
      for tag, a, e, what in facts[key]["variants"]:
        var.append((f"f_{kind}_{key}_{tag}", f"fig:{kind}_{key}_{tag}",
          f"{NAMES[key]}, {'tracking' if kind == 'track' else 'hierarchy'}: t = {a:.0f} to {e:.0f}\\,s, {what}. "
          f"Otherwise as Fig.~\\ref{{fig:{ref}_{key}}}."))
  cont, lab = "\\ContinuedFloat", "\\label"
  body = "".join(f"""
\\begin{{figure}}[p]
  \\centering
  {cont if f in CONTINUED else ""}
  \\input{{{f}.pgf}}
  \\caption{{{c}}}
  {lab + "{" + l + "}" if l else ""}
\\end{{figure}}
\\clearpage
""" for f, l, c in figs)
  body += replication_table(ptf.REPO)
  body += ("\n\\section*{Zoom variants}\nEvery walking segment and the start-up of the runs in "
           "Figs.~\\ref{fig:track_A1a} to \\ref{fig:hier_A1a_DR}, to choose from. Same figures, same "
           "y-limits, only the window moves (each walking segment plus 2\\,s either side, at least 10\\,s).\n")
  body += "".join(f"""
\\begin{{figure}}[p]
  \\centering
  \\input{{{f}.pgf}}
  \\caption{{{c}}}
  \\label{{{l}}}
\\end{{figure}}
\\clearpage
""" for f, l, c in var)
  (out / "report_figures.tex").write_text(TEX_HEAD + body + "\\end{document}\n")
  print(f"[REPORT] wrote {out / 'report_figures.tex'} ({len(figs)} figures + {len(var)} zoom variants)")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", default=str(ptf.REPO / "figures/report_figures"))
  out = Path(ap.parse_args().out)
  ptf.SAVE_PDF = True
  ptf.set_decimation(10000)   # 2500 envelope buckets: 1000 turned stride-rate bursts into solid bands
  root = ptf.REPO
  day = ptf.load_bundles(root, ["logs/robot_logs/2026_09_16-*/"])
  both = ptf.load_bundles(root, ["logs/robot_logs/2026_09_14-*/", "logs/robot_logs/2026_09_16-*/"])

  # Built first, saved after: each set (3 tracking, 2 hierarchy figures) is read side by side, so its
  # panels share y-limits instead of each autoscaling (the wz range differed 2x across the triplet).
  sets = {"track": [], "hier": []}
  facts = {}   # numbers the captions quote, computed here with the figures
  for key in ("A1a", "A0", "A1a_DR"):
    b = pick(day, key)
    fig, rows, rms = ptf._track_fig(b, "", annotate=False, figsize=(ptf.FIG_W, 5.0))
    fig.suptitle("")   # report figures carry no title
    robot, tgt = walk_errors(b)
    for axis in rows:
      ptf.write_panel_csv(out, f"f_track_{key}", axis,
                          ["policy", "run", "bundle", "series", "t_s", "value", "unit"], rows[axis])
    print(f"[REPORT] track {key}: {b.dir.name}, truth missing {rms['vx']['gap']:.1f}%")
    t0, t1, ctx = zoom_window(b)
    zoomed = zoom(fig, t0, t1, ctx)
    wins = variant_windows(b)
    facts[key] = {"rms": rms, "robot": robot, "tgt": tgt, "run": b.dir.name,
                  "window": (t0, t1, float(ctx[0][-1])), "variants": wins}
    sets["track"].append((fig, f"f_track_{key}", zoomed, (t0, t1), wins))
    if key == "A0":
      continue
    fig, rows, sat_rows, info = ptf._hier_fig(b, "", figsize=(ptf.FIG_W, 5.6))
    fig.suptitle("")   # report figures carry no title
    zoomed = zoom(fig, t0, t1, ctx)
    for axis in rows:
      ptf.write_panel_csv(out, f"f_hier_{key}", axis,
                          ["policy", "run", "bundle", "series", "t_s", "value", "unit"], rows[axis])
    ptf.write_panel_csv(out, f"f_hier_{key}", "saturation",
                        ["policy", "run", "bundle", "series", "t_s", "value", "unit"], sat_rows)
    print(f"[REPORT] hier {key}: {info['fires']} HL fires")
    facts[key]["sat"] = info["sat_pct"]
    facts[key]["gs"] = info["goal_scale"]
    sets["hier"].append((fig, f"f_hier_{key}", zoomed, (t0, t1), wins))
  for figs in sets.values():
    for i in range(len(figs[0][0].axes)):
      lims = [f[0].axes[i].get_ylim() for f in figs]
      for f in figs:
        f[0].axes[i].set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
    for f, name_, (main_axes, cax), (t0, t1), wins in figs:
      # the variants re-save the SAME figure with another window: identical data, styling and y-limits
      for tag, a, e, _ in wins:
        set_window(main_axes, cax, a, e)
        ptf.save_pgf(f, out, f"{name_}_{tag}")
      set_window(main_axes, cax, t0, t1)
      ptf.save_pgf(f, out, name_)

  kept = [b for b in day if b.dir.name not in CHAIN_EXCLUDE
          and ptf._variant_seed(b.policy)[0] in CHAIN_VARIANTS]
  print(f"[REPORT] chain: dropped {sorted(CHAIN_EXCLUDE)} and the cotcap runs")
  rows_w, n_w = chain_points(kept, root, "walking")
  rows_s, n_s = chain_points(kept, root, "standing")
  for part, keys in CHAIN_CLEAN_PARTS:
    fig_chain_seq(rows_w, keys, STAGE_KEYS, out, f"f_chain_clean_{part}", ncol=2)
  for part, keys in CHAIN_SIM_PARTS:
    fig_chain_seq(rows_w, keys, STAGE_KEYS[:2], out, f"f_chain_sim_walking_{part}", ncol=2)
  fig_chain_seq(rows_s, SIM_KEYS["standing"], STAGE_KEYS[:2], out, "f_chain_sim_standing", ncol=2)
  facts["chain"] = {"walking": n_w, "standing": n_s}
  # Largest deployment-plant run-to-run spread, (max - min) / mean over the runs of one policy, over every
  # metric of the simulation figure: the number that justifies not drawing single deployment-plant runs.
  facts["bridge_spread_pct"] = {
    r: max(100 * (max(v) - min(v)) / abs(np.mean(v)) for k in SIM_KEYS[r]
           for _, _, st, _ in rows(k, ("bridge",)) for v in [st["bridge"]] if len(v) > 1 and np.mean(v))
    for r, rows in (("walking", rows_w), ("standing", rows_s))}
  print(f"[REPORT] chain: largest deployment-plant run-to-run spread {facts['bridge_spread_pct']} %")
  write_rows(out, "f_chain_units", [{"policy": t, "stage": st, "regime": r, "n_scored": n, "n_fell": k}
                                    for r, c in facts["chain"].items() for (t, st), (n, k) in c.items()])

  # Pool by VARIANT: `A1a_s42` / `A1a_s123` / the seedless 09-14 `A1a` are one policy, and
  # `policy_label` keeps loaded runs apart by their `+kg` suffix, which the variant filter drops.
  pooled = []
  for b in both:
    variant, _seed = ptf._variant_seed(b.policy)
    if variant in UPPER_VARIANTS and b.regimes:
      pooled.append(dataclasses.replace(b, policy=variant))
  n = {v: sum(b.policy == v for b in pooled) for v in UPPER_VARIANTS}
  print(f"[REPORT] upper: runs per policy {n}")
  facts["upper_n"] = n
  ptf.fig_upper(pooled, out, name="f_upper_3pol")

  points = load_reward_points(root)
  hlp = hl_points(root)
  print("[REPORT] reward: hardware runs per policy "
        + str({t: len(p["hardware"]) for t, p in points.items()}))
  for stages, sfx in ((STAGE_KEYS, ""), (STAGE_KEYS[:2], "_sim")):
    fig_reward_return(points, hlp, out, stages, sfx)
    for regime in ("walking", "standing"):
      fig_reward_terms(points, out, regime, stages, sfx)
  facts["reward_seed"] = seed_facts(points, hlp, root)
  llp = ll_points(kept, root)
  for stages, sfx in ((STAGE_KEYS, ""), (STAGE_KEYS[:2], "_sim")):
    fig_ll_reward(llp, out, stages, sfx)
  facts["ll_n"] = {s: [len(llp[t, "walking"][s]) for t in LL_TAGS] for s in STAGE_KEYS}
  # units that have goal tracking but not the shaping terms (a refused replay pair, no velocity truth)
  facts["ll_partial"] = [(t, s, sum("r_track" in u and "r_shape_cf" not in u for u in llp[t, "walking"][s]))
                         for t in LL_TAGS for s in STAGE_KEYS]
  facts["ll_partial"] = [x for x in facts["ll_partial"] if x[2]]
  print(f"[REPORT] low-level goal reward units per policy {facts['ll_n']}")

  trow = train_points(root)
  fig_training(trow, out)
  facts["train"] = trow
  lrow, lsl = lean_points(root), lean_slopes(root)
  facts["lean"] = {"cell": fig_lean(lrow, lsl, out), "slope": lsl, "n": len(lrow)}
  fig_lean(lrow, lsl, out, "f_lean_left", left_only=True)
  srow = share_points(root)
  facts["share"] = {"cell": fig_standing_share(srow, out), "rows": srow}
  prow = payload_points(root)
  facts["payload"] = {"cell": fig_payload(prow, out),
                      "n": {(r["variant"], r["payload_kg"]): sum(1 for q in prow if q["regime"] == "standing"
                                                                   and (q["variant"], q["payload_kg"])
                                                                   == (r["variant"], r["payload_kg"]))
                            for r in prow}}
  write_latex(out, facts)
  return 0


if __name__ == "__main__":
  sys.exit(main())
