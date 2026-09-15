#!/usr/bin/env python3
"""LaTeX-ready thesis figures from the 2026-09-14 hardware _Run_ bundles.

Five figures, each written TWICE: a `.pgf` to `\\input` into the thesis (matplotlib's own
pgf backend -- fonts are NOT baked, `pgf.rcfonts=False`, so the document's fonts are
inherited) and one `<name>_<panel>.csv` per panel holding exactly the points that were
drawn, as the archival record.

  f_cadence  commanded stride period (the _Gait reference_) while standing, per policy,
             against the deploy yaml's `cadence_period_range` bounds. Every _Run_ is its
             own point; n is 5/1/2 per hierarchical policy, so a bare mean is never drawn.
             The flat policy has no high level and is shown ABSENT, not as zero.
  f_settle   arm-velocity settling time across the whole stillness-threshold sweep
             (0.05-0.20 rad/s), one line per _Run_, right-CENSORED cells marked as the
             lower bounds they are, plus the MEASURED steady-state floor per Run.
  f_upper    the upper-body cost of the hierarchy: `ub_arm_vel` and `ub_pose_dev` per
             policy, split by _Regime_, every Run drawn over its bar.
  f_smooth   `act_legs_rad` and leg joint acceleration over time, both from the flight
             recorder's `raw_q`/`meas_q`, so the flat policy (no HRL telemetry) is
             scorable on the same axes.
  f_power    mechanical power `sum_j |tau_est_j * dq_j|` over time, with the standing and
             walking _Regimes_ shaded.

Sources (nothing here is recomputed that already exists):
  logs/robot_logs/2026_09_14-*/    one _Run_ bundle each: sliced flight recorder, optional
                                   HRL telemetry, optional _Joint telemetry log_, run.json
                                   (the ONLY authority on `policy_tag`), and the
                                   already-scored `*.standing.json` / `*.walking.json`.
  data/2026-09-14-hardware-session/cadence_settling.json
                                   scripts/analyze_cadence_settling.py's output; f_cadence
                                   and f_settle READ it, they do not re-derive it.

Metric names follow bench_flight_recorder's output schema and CONTEXT.md. Two deliberate
renames, both following that tool's own `err_vx -> err_vx_est` precedent of renaming when
the quantity is not definitionally sim's:
  jacc_legs_ctrl   leg joint acceleration from MEASURED q differenced at the control rate.
                   bench_flight_recorder OMITS `jacc_legs` precisely because sim's is
                   MuJoCo `qacc` at the physics rate; the suffix keeps the two apart.
  mech_power_w     kept unchanged -- same definition as bench's, measured `tau_est * dq`.
No cost of transport is plotted here; when one is added it must say which of CONTEXT.md's
two CoT terms it is (the bundles' `cot` key is the DIAGNOSTIC, undirected distance).

Usage:
  python scripts/plot_thesis_figures.py                      # all five into figures/
  python scripts/plot_thesis_figures.py --figures cadence,settle
  python scripts/plot_thesis_figures.py --out figures --check-latex
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
# Reuse: envelope decimation (these logs reach 440k rows), the policy-step decode, the
# regime gate and the affine session-pair alignment all already exist and are validated.
import plot_deploy_logs as pdl  # noqa: E402
from plot_deploy_logs import envelope_downsample  # noqa: E402
from bench_flight_recorder import (CTRL_DT, KNEE_SLOT, LEG_SLOTS, THIN_STEPS,  # noqa: E402
                                   fit_alignment, read_telemetry, walking_mask)
from score_joint_hold import policy_steps  # noqa: E402

import matplotlib  # noqa: E402
# plot_deploy_logs forces Agg at import; re-set before pyplot is touched.
matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BUNDLE_GLOB = "logs/robot_logs/2026_09_14-*/"
CADENCE_JSON = "data/2026-09-14-hardware-session/cadence_settling.json"
PERIOD_RANGE = (0.35, 1.0)      # analyze_cadence_settling.PERIOD_RANGE (deploy yaml)
ARM_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)  # rad/s, analyze_cadence_settling.ARM_THRESHOLDS
FIG_W = 5.9                     # in; ~15 cm thesis column
FIGURES = ("cadence", "settle", "upper", "smooth", "power")

# Okabe-Ito, colour-blind safe. One colour per policy, shared by every figure.
POLICY_COLOR = {
  "A0_DR_s123": "#000000",
  "A1a": "#0072B2",
  "A1a_DR_s123": "#D55E00",
  "A1a_DR_cotcap_s42": "#009E73",
}
POLICY_ORDER = ["A0_DR_s123", "A1a", "A1a_DR_s123", "A1a_DR_cotcap_s42"]
REGIME_HATCH = {"standing": "", "walking": "////"}
REGIME_ALPHA = {"standing": 1.0, "walking": 0.45}

plt.rcParams.update({
  "pgf.texsystem": "pdflatex",
  "pgf.rcfonts": False,      # inherit the thesis fonts instead of baking matplotlib's
  "pgf.preamble": "\n".join([r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}"]),
  "font.family": "serif",
  "font.size": 8,
  "axes.labelsize": 8,
  "axes.titlesize": 8,
  "legend.fontsize": 7,
  "xtick.labelsize": 7,
  "ytick.labelsize": 7,
  "axes.grid": True,
  "grid.alpha": 0.25,
  "grid.linewidth": 0.4,
  "lines.linewidth": 0.8,
  "savefig.bbox": "tight",
})



def set_decimation(max_points: int) -> None:
  """Retune plot_deploy_logs' envelope decimation for vector output.

  Its module defaults (no decimation below 20000 rows) are sized for on-screen PNGs, where
  a 9300-point line costs nothing. In a .pgf every point is path data the thesis recompiles
  on every build: the uncapped f_power was 7.4 MB. Same decimation, lower threshold --
  min/max envelope, so the spikes survive.
  """
  pdl.DOWNSAMPLE_THRESHOLD = max(int(max_points), 100)
  pdl.DOWNSAMPLE_BUCKETS = max(int(max_points) // 4, 50)


def tex(s: str) -> str:
  """Escape a literal name (policy tag, metric key) for the pgf backend.

  matplotlib's pgf writer escapes only `^` and `%`, so an underscore in `A1a_DR_s123` or
  `act_legs_rad` reaches LaTeX as subscript math and the figure fails to compile at all.
  Never apply it to a string that carries deliberate `$...$` math.
  """
  for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#"),
               ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
    s = s.replace(a, b)
  return s

# ------------------------------------------------------------------- run bundles ----

@dataclass
class Bundle:
  """One _Run_: an FSM entry, its sliced logs and its already-scored regime metrics."""
  dir: Path
  run: dict
  policy: str
  tag: str                      # short human id, e.g. "run 5"
  flight: Path
  hrl: Path | None
  joints: Path | None
  regimes: dict = field(default_factory=dict)

  @property
  def walk_s(self) -> float:
    return float(self.regimes.get("walking", {}).get("duration_s", 0.0))


def policy_label(run: dict) -> str:
  """Policy name with the _Mounted payload_ folded in, because load is not a nuisance here.

  Grouping on `policy_tag` alone silently pools loaded and unloaded Runs: on 2026-09-14 that
  would put A1a at 0 kg together with A1a under 7.5 kg, and compare both against an A0_DR
  that carried the backpack for BOTH its Runs. Every flat-versus-hierarchical number would
  then be flat-under-load against hierarchical-unloaded without the figure saying so. The
  load belongs in the label, where a reader cannot miss it.
  """
  kg = float(run.get("payload_kg") or 0.0)
  return run["policy_tag"] + (f" +{kg:g}kg" if kg > 0 else "")


def load_bundles(root: Path) -> list[Bundle]:
  out = []
  for d in sorted(root.glob(BUNDLE_GLOB)):
    # A bundle IS a run.json. A sibling directory matching the date glob without one is not a
    # Run (e.g. parked motion capture for an aborted attempt) and is skipped silently.
    if not (d / "run.json").exists():
      continue
    run = json.loads((d / "run.json").read_text())
    flight = d / run["flight_recorder"]
    if not flight.exists():
      print(f"[PLOT] WARNING: {d.name}: {run['flight_recorder']} missing -- bundle skipped")
      continue
    hrl = flight.with_name(flight.stem + "_hrl.csv")
    joints = d / run["all_joints"] if run.get("all_joints") else None
    b = Bundle(dir=d, run=run, policy=policy_label(run),
               # `note` is free text and can run to a paragraph (run 4 carries the
               # label-correction story); the point label is only ever its leading id.
               tag=(run.get("note") or d.name).split(",")[0].split(".")[0].strip()[:24],
               flight=flight, hrl=hrl if hrl.exists() else None,
               joints=joints if (joints and joints.exists()) else None)
    for regime in ("standing", "walking"):
      p = d / f"{flight.stem}.{regime}.json"
      if p.exists():
        b.regimes[regime] = json.loads(p.read_text())
    out.append(b)
  return out


def policies_present(bundles: list[Bundle]) -> list[str]:
  seen = {b.policy for b in bundles}
  known = [p for p in POLICY_ORDER if p in seen]
  return known + sorted(seen - set(known))


def pick_runs(bundles: list[Bundle], need_torque: bool) -> dict[str, Bundle]:
  """One representative _Run_ per policy: the one with the most WALKING time, since both
  time-series figures shade or compare regimes and a Run with a 1 s walking segment shows
  nothing. Ties/absent walking fall back to Run duration."""
  out = {}
  for p in policies_present(bundles):
    cands = [b for b in bundles if b.policy == p and (b.joints or not need_torque)]
    if not cands:
      why = "no _Joint telemetry log_ (run.json all_joints: null)" if need_torque else "no runs"
      print(f"[PLOT] NOTE: {p}: skipped -- {why}; mech_power_w cannot be computed")
      continue
    out[p] = max(cands, key=lambda b: (b.walk_s, b.run.get("duration_s", 0.0)))
  return out


# ------------------------------------------------------------------------ output ----

def save_pgf(fig, out_dir: Path, name: str) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"{name}.pgf"
  fig.savefig(path)
  plt.close(fig)
  # matplotlib emits `\mathdefault` for any mathtext tick label (a log axis, a scientific
  # offset) but defines it only in the standalone document IT writes, never in the raw .pgf.
  # Without this line an \input of the figure dies on "Undefined control sequence" in a
  # thesis that happens not to define it -- so the figure carries its own definition.
  src = path.read_text()
  if "\\mathdefault" in src and "providecommand\\mathdefault" not in src:
    path.write_text(src.replace("\\begingroup%", "\\providecommand\\mathdefault[1]{#1}%\n"
                                "\\begingroup%", 1))
  print(f"[PLOT] wrote {path}")
  return path


def write_panel_csv(out_dir: Path, name: str, panel: str, fields: list[str],
                    rows: list[dict]) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"{name}_{panel}.csv"
  with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
  print(f"[PLOT] wrote {path}")
  return path


def bottom_legend(fig, handles, ncol=3, height=0.13):
  """Legend UNDER the axes. In every one of these figures the data reaches the corners
  (points at the action-space ceiling, the tallest bar, the startup power spike), so an
  in-axes legend covers a value the figure exists to show."""
  fig.tight_layout(rect=(0, height, 1, 1))
  # tight_layout declines (with a warning) on some mixed-gridspec figures and then leaves
  # the axes sitting on the band the legend is about to occupy; enforce the band directly.
  if min(ax.get_position().y0 for ax in fig.axes) < height:
    fig.subplots_adjust(bottom=height + 0.02)
  fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
             bbox_to_anchor=(0.5, 0.0))


def policy_handles(policies):
  from matplotlib.lines import Line2D
  return [Line2D([], [], color=POLICY_COLOR.get(p, "0.4"), marker="o", markersize=3.5,
                 linewidth=1.0, label=tex(p)) for p in policies]


# =============================================================== f_cadence ====

def fig_cadence(bundles, cad_rows, out_dir, name="f_cadence"):
  """Commanded stride period (_Gait reference_) per _Run_, standing vs walking.

  The bounds matter more than the values: "saturated at the action-space ceiling" is the
  finding, and it is invisible without `cadence_period_range` drawn. A flat policy has no
  high level, so it carries no commanded period at all -- marked absent, never 0.
  """
  policies = policies_present(bundles)
  by_bundle = {r["bundle"]: r for r in cad_rows}
  fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.8), sharey=True)
  lo, hi = PERIOD_RANGE
  rows = {"standing": [], "walking": []}

  for ax, regime in zip(axes, ("standing", "walking")):
    ax.axhline(hi, color="0.35", linestyle="--", linewidth=0.7)
    ax.axhline(lo, color="0.35", linestyle="--", linewidth=0.7)
    for x, p in enumerate(policies):
      bs = [b for b in bundles if b.policy == p]
      cells = [(b, (by_bundle.get(b.dir.name, {}).get("cadence") or {})) for b in bs]
      live = [(b, c[regime]) for b, c in cells
              if c.get("available") and c.get(regime)]
      if not live:
        why = next((c.get("why", "unavailable") for _, c in cells if not c.get("available")),
                   "no data in this regime")
        ax.axvspan(x - 0.42, x + 0.42, color="0.92", zorder=0)
        ax.text(x, 0.5 * (lo + hi), "absent\n" + ("flat policy:\nno high level"
                if "flat" in why else why), ha="center", va="center", fontsize=6,
                color="0.35", rotation=0)
        rows[regime].append({"policy": p, "run": "", "bundle": "", "period_mean_s": "",
                             "period_p5_s": "", "period_p95_s": "", "frac_at_ceiling": "",
                             "n_policy_steps": "", "status": f"absent: {why}"})
        continue
      xs = np.linspace(-0.18, 0.18, len(live)) if len(live) > 1 else np.zeros(1)
      for dx, (b, s) in zip(xs, live):
        thin = s["n"] < THIN_STEPS
        ax.plot([x + dx, x + dx], [s["p5"], s["p95"]], color=POLICY_COLOR.get(p, "0.4"),
                linewidth=0.7, alpha=0.7, zorder=2)
        ax.plot(x + dx, s["mean"], marker="o", markersize=3.6, zorder=3,
                color=POLICY_COLOR.get(p, "0.4"),
                markerfacecolor="white" if thin else POLICY_COLOR.get(p, "0.4"))
        rows[regime].append({
          "policy": p, "run": b.tag, "bundle": b.dir.name,
          "period_mean_s": round(s["mean"], 5), "period_p5_s": round(s["p5"], 5),
          "period_p95_s": round(s["p95"], 5),
          "frac_at_ceiling": round(s["frac_at_ceiling"], 5), "n_policy_steps": s["n"],
          "status": "thin (<%d steps)" % THIN_STEPS if thin else "ok"})
      # Fraction AT the ceiling, the finding this figure exists to show. Sub-percent values
      # get a decimal so a 0.4% run never prints as a flat "0%", and a range collapses to
      # one number when both ends round the same way.
      def pct(v):
        return ("%.0f" % (100 * v)) if v >= 0.01 else ("%.1f" % (100 * v))
      cf = [s["frac_at_ceiling"] for _, s in live]
      a, z = pct(min(cf)), pct(max(cf))
      top = max(s["p95"] for _, s in live)
      ax.text(x, top + 0.03, (a if a == z else f"{a}-{z}") + "%", ha="center", va="bottom",
              fontsize=6, color=POLICY_COLOR.get(p, "0.4"))
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([tex(p) for p in policies], rotation=20, ha="right")
    ax.set_xlim(-0.6, len(policies) - 0.4)
    ax.set_title(regime)
  axes[0].set_ylabel("commanded stride period (s)")
  axes[0].set_ylim(lo - 0.10, hi + 0.22)
  # Bound labels in the LEFT panel only, BELOW their lines: the walking panel's whiskers
  # run the full height, so a label on that panel lands on data.
  axes[0].text(-0.5, hi - 0.015, "action-space ceiling %.2f s" % hi,
               fontsize=5.6, va="top", ha="left", color="0.35")
  axes[0].text(-0.5, lo + 0.015, "floor %.2f s" % lo, fontsize=5.6,
               va="bottom", ha="left", color="0.35")
  from matplotlib.lines import Line2D
  bottom_legend(fig, [
    Line2D([], [], color="0.3", marker="o", markersize=3.6, linestyle="",
           label="one Run (mean, p5-p95)"),
    Line2D([], [], color="0.3", marker="o", markersize=3.6, linestyle="",
           markerfacecolor="white",
           label=r"thin regime ($<$%d steps)" % THIN_STEPS)],
    ncol=2, height=0.19)
  fig.text(0.5, 0.125, "percentage above each cluster = policy steps AT the ceiling",
           ha="center", fontsize=6, color="0.35")
  fields = ["policy", "run", "bundle", "period_mean_s", "period_p5_s", "period_p95_s",
            "frac_at_ceiling", "n_policy_steps", "status"]
  for regime in ("standing", "walking"):
    write_panel_csv(out_dir, name, regime, fields, rows[regime])
  return save_pgf(fig, out_dir, name)


# ================================================================ f_settle ====

def fig_settle(bundles, cad_rows, out_dir, name="f_settle"):
  """Arm-velocity settling time across the FULL stillness-threshold sweep.

  The sweep is the result: the flat-vs-hierarchical gap holds at every threshold, the
  DR-vs-A1a ordering only at the tight ones. Collapsing to one threshold hides that. A
  right-censored cell (the segment never settled) is a LOWER BOUND and is drawn as one.
  """
  policies = policies_present(bundles)
  by_bundle = {r["bundle"]: r for r in cad_rows}
  fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.0),
                           gridspec_kw={"width_ratios": [2.1, 1.0]})
  ax, axf = axes
  sweep_rows, floor_rows = [], []

  for b in bundles:
    r = by_bundle.get(b.dir.name)
    if not r:
      continue
    col = POLICY_COLOR.get(b.policy, "0.4")
    xs, ys, cens = [], [], []
    for th in ARM_THRESHOLDS:
      d = r["arm"].get(f"t_settle_{th}") or {}
      if d.get("median") is None:
        continue
      xs.append(th)
      ys.append(d["median"])
      cens.append(bool(d.get("censored")))
      sweep_rows.append({"policy": b.policy, "run": b.tag, "bundle": b.dir.name,
                         "threshold_rad_s": th, "t_settle_median_s": round(d["median"], 4),
                         "t_settle_max_s": round(d["max"], 4), "n_segments": d["n"],
                         "censored_segments": d["censored"],
                         "bound": "lower (right-censored)" if d.get("censored") else "measured"})
    if not xs:
      continue
    ax.plot(xs, ys, color=col, linewidth=0.9, alpha=0.85, marker="o", markersize=3.0,
            markerfacecolor=col, zorder=2)
    cx = [x for x, c in zip(xs, cens) if c]
    cy = [y for y, c in zip(ys, cens) if c]
    if cx:
      ax.plot(cx, cy, linestyle="", marker="^", markersize=5.0, markerfacecolor="white",
              markeredgecolor=col, zorder=4)

    floor = r["arm"].get("floor_rad_s")
    if floor is not None and np.isfinite(floor):
      x = policies.index(b.policy)
      sib = [o for o in bundles if o.policy == b.policy]
      dx = (sib.index(b) - 0.5 * (len(sib) - 1)) * (0.30 / max(len(sib) - 1, 1))
      axf.plot(x + dx, floor, marker="o", markersize=3.4, color=col, zorder=3)
      floor_rows.append({"policy": b.policy, "run": b.tag, "bundle": b.dir.name,
                         "arm_vel_floor_rad_s": round(float(floor), 6),
                         "n_standing_segments": r.get("n_standing_segments")})

  ax.set_xticks(list(ARM_THRESHOLDS))
  ax.set_xlabel("stillness threshold on mean $|\\mathrm{dq}|$, upper body (rad/s)")
  ax.set_ylabel("arm-velocity settling time (s)")
  ax.set_title("settling time vs threshold (one line per Run)")
  ax.margins(y=0.12)

  for th in ARM_THRESHOLDS:
    axf.axhline(th, color="0.55", linestyle=":", linewidth=0.6)
  axf.set_yscale("log")
  axf.set_ylim(top=ARM_THRESHOLDS[-1] * 2.2)
  axf.text(-0.5, ARM_THRESHOLDS[-1] * 0.92, "swept thresholds", fontsize=5.6,
           va="top", ha="left", color="0.4")
  axf.set_xticks(range(len(policies)))
  axf.set_xticklabels([tex(p) for p in policies], rotation=30, ha="right", fontsize=6)
  axf.set_xlim(-0.6, len(policies) - 0.4)
  axf.set_ylabel("measured steady-state floor (rad/s)")
  axf.set_title("settled floor per Run")
  from matplotlib.lines import Line2D
  bottom_legend(fig, policy_handles(policies) + [
    Line2D([], [], color="0.3", marker="^", markersize=5.0, linestyle="",
           markerfacecolor="white", label="right-censored (lower bound)")],
    ncol=3, height=0.20)

  write_panel_csv(out_dir, name, "sweep",
                  ["policy", "run", "bundle", "threshold_rad_s", "t_settle_median_s",
                   "t_settle_max_s", "n_segments", "censored_segments", "bound"], sweep_rows)
  write_panel_csv(out_dir, name, "floor",
                  ["policy", "run", "bundle", "arm_vel_floor_rad_s", "n_standing_segments"],
                  floor_rows)
  return save_pgf(fig, out_dir, name)


# ================================================================= f_upper ====

def fig_upper(bundles, out_dir, name="f_upper"):
  """The upper-body cost of the hierarchy, from the already-scored regime jsons.

  Bars are the per-policy mean, but every _Run_ is drawn over its bar: A1a's standing
  spread is ~2.4x across its Runs and that spread is part of the result, not noise to be
  averaged away.
  """
  policies = policies_present(bundles)
  metrics = [("ub_arm_vel", "upper-body joint velocity\n" r"$\overline{|\mathrm{dq}|}$ (rad/s)"),
             ("ub_pose_dev", "upper-body pose deviation\n" r"$\overline{(q-q_\mathrm{def})^2}$ (rad$^2$)")]
  fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.9))
  regimes = ("standing", "walking")
  width = 0.38
  missing = [b for b in bundles if not b.regimes]
  for b in missing:
    print(f"[PLOT] NOTE: {b.dir.name} ({b.policy}, {b.tag}): no scored regime json in the "
          "bundle -- excluded from f_upper")

  for ax, (key, ylabel) in zip(axes, metrics):
    rows = []
    for x, p in enumerate(policies):
      for k, regime in enumerate(regimes):
        vals = [(b, b.regimes[regime][key]) for b in bundles
                if b.policy == p and regime in b.regimes and key in b.regimes[regime]]
        if not vals:
          continue
        xpos = x + (k - 0.5) * width
        m = float(np.mean([v for _, v in vals]))
        ax.bar(xpos, m, width=width * 0.92, color=POLICY_COLOR.get(p, "0.4"),
               alpha=REGIME_ALPHA[regime], edgecolor=POLICY_COLOR.get(p, "0.4"),
               linewidth=0.6, hatch=REGIME_HATCH[regime], zorder=1)
        jit = np.linspace(-0.11, 0.11, len(vals)) if len(vals) > 1 else np.zeros(1)
        for dx, (b, v) in zip(jit, vals):
          thin = bool(b.regimes[regime].get("thin"))
          ax.plot(xpos + dx, v, marker="o", markersize=2.8, color="0.15",
                  markerfacecolor="white" if thin else "0.15", markeredgewidth=0.6, zorder=3)
          rows.append({"policy": p, "regime": regime, "run": b.tag, "bundle": b.dir.name,
                       key: v, f"{key}_policy_mean": round(m, 6),
                       "n_policy_steps": b.regimes[regime]["n_steps"],
                       "status": "thin (<%d steps)" % THIN_STEPS if thin else "ok"})
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([tex(p) for p in policies], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.14)   # a Run point sitting on the axis top edge reads as clipped
    write_panel_csv(out_dir, name, key,
                    ["policy", "regime", "run", "bundle", key, f"{key}_policy_mean",
                     "n_policy_steps", "status"], rows)

  from matplotlib.patches import Patch
  from matplotlib.lines import Line2D
  bottom_legend(fig, [
    Patch(facecolor="0.55", edgecolor="0.3", label="standing"),
    Patch(facecolor="0.55", edgecolor="0.3", alpha=REGIME_ALPHA["walking"],
          hatch=REGIME_HATCH["walking"], label="walking"),
    Line2D([], [], color="0.15", marker="o", markersize=2.8, linestyle="", label="one Run"),
    Line2D([], [], color="0.15", marker="o", markersize=2.8, linestyle="",
           markerfacecolor="white", label="thin regime")], ncol=4, height=0.13)
  return save_pgf(fig, out_dir, name)


# ================================================================ f_smooth ====

def _flight_steps(b: Bundle):
  """Policy-step grid of one Run: time, commanded leg targets, measured leg positions."""
  cols = ["t", "cmd_vx", "cmd_vy", "cmd_wz"] + [f"raw_q{j}" for j in range(27)] \
         + [f"meas_q{j}" for j in LEG_SLOTS]
  df = pd.read_csv(b.flight, usecols=cols)
  raw = df[[f"raw_q{j}" for j in range(27)]].to_numpy(float)
  steps, sidx = policy_steps(raw)
  t = df["t"].to_numpy(float)
  meas = df[[f"meas_q{j}" for j in LEG_SLOTS]].to_numpy(float)[sidx]
  walk = walking_mask(df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float))
  return t, sidx, steps, meas, walk


def fig_smooth(bundles, out_dir, name="f_smooth", downsample=True, window_s=60.0):
  """Commanded leg smoothness and realised leg acceleration, one Run per policy.

  Both panels come from the flight recorder, NOT from `_hrl.csv`'s `act_rate`: the flat
  policy writes no HRL telemetry, and a figure that cannot show A0 cannot carry a
  cross-policy claim. Commanded side is in RADIANS (CONTEXT.md "Physical units rule") and
  legs-only, since `hold_joint_ids` can freeze the upper body on hardware.

  Only the first `window_s` of each Run is drawn (0 = the whole Run). Four Runs of 78-216 s
  overlaid at 50 Hz is a solid block at thesis width. The panel CSV is the archival record
  of what was DRAWN, so it carries that same window.
  """
  picks = pick_runs(bundles, need_torque=False)
  fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 3.6), sharex=True)
  rows = {"act_legs_rad": [], "jacc_legs_ctrl": []}
  for p, b in picks.items():
    t, sidx, steps, meas, _ = _flight_steps(b)
    ts = t[sidx] - t[sidx][0]
    col = POLICY_COLOR.get(p, "0.4")
    # act_legs_rad: L2 over the 12 leg slots of the per-policy-step difference of raw_q.
    act = np.linalg.norm(np.diff(steps, axis=0)[:, LEG_SLOTS], axis=1)
    ta = 0.5 * (ts[1:] + ts[:-1])
    # jacc_legs_ctrl: second difference of MEASURED leg q over the control step. Renamed
    # from sim's `jacc_legs`, which is MuJoCo qacc at the PHYSICS rate (bench OMITTED).
    jac = np.linalg.norm((meas[2:] - 2.0 * meas[1:-1] + meas[:-2]) / CTRL_DT ** 2, axis=1)
    tj = ts[1:-1]
    for ax, (key, tt0, yy0) in zip(axes, (("act_legs_rad", ta, act),
                                          ("jacc_legs_ctrl", tj, jac))):
      if window_s:
        keep = tt0 <= window_s
        tt0, yy0 = tt0[keep], yy0[keep]
      tt, yy = envelope_downsample(tt0, yy0, downsample)
      ax.plot(tt, yy, color=col, linewidth=0.45, alpha=0.85, label=tex(p))
      rows[key] += [{"policy": p, "run": b.tag, "bundle": b.dir.name,
                     "t_s": round(float(x), 4), key: round(float(y), 6)}
                    for x, y in zip(tt, yy)]
    print(f"[PLOT] f_smooth: {p} = {b.tag} ({b.dir.name}), {len(steps)} policy steps")
  axes[0].set_ylabel(tex("act_legs_rad") + " (rad)")
  axes[0].set_title("commanded leg action rate (legs only, radians)")
  axes[1].set_ylabel(tex("jacc_legs_ctrl") + " (rad/s$^2$)")
  axes[1].set_title("measured leg joint acceleration (control rate)")
  axes[1].set_xlabel("time since Run start (s)"
                     + (f", first {window_s:.0f} s" if window_s else ""))
  bottom_legend(fig, policy_handles(picks), ncol=4, height=0.11)
  for key in rows:
    write_panel_csv(out_dir, name, key, ["policy", "run", "bundle", "t_s", key], rows[key])
  return save_pgf(fig, out_dir, name)


# ================================================================= f_power ====

def _alignment(b: Bundle, t_f, knee_f, T):
  """t_flight = t_aj + (b + m t_aj): the Run's own _Session pair_ fit, refitted if absent."""
  if b.run.get("pair_lag_s") is not None and b.run.get("pair_skew_ppm") is not None:
    return {"b": float(b.run["pair_lag_s"]), "m": float(b.run["pair_skew_ppm"]) * 1e-6,
            "src": "run.json"}
  a = fit_alignment(t_f, knee_f, T["t"], T["knee"])
  if a is None:
    return None
  a["src"] = "refit"
  return a


def fig_power(bundles, out_dir, name="f_power", downsample=True):
  """Mechanical power `sum_j |tau_est_j * dq_j|` over time, regimes shaded.

  `tau_est` lives only in the _Joint telemetry log_, on its own clock: the two loggers
  differ by -3337 to -6613 ppm on this session, so the map is AFFINE (a constant offset is
  wrong by most of a stride period over a 170 s Run).
  """
  picks = pick_runs(bundles, need_torque=True)
  if not picks:
    raise SystemExit("f_power: no Run has a _Joint telemetry log_; tau_est is its only source")
  fig, axes = plt.subplots(len(picks), 1, figsize=(FIG_W, 1.05 * len(picks) + 1.0),
                           sharex=True, sharey=True, squeeze=False)
  axes = list(axes[:, 0])
  rows, drawn = [], []
  for ax, (p, b) in zip(axes, picks.items()):
    df = pd.read_csv(b.flight, usecols=["t", "cmd_vx", "cmd_vy", "cmd_wz",
                                        f"meas_q{KNEE_SLOT}"])
    t_f = df["t"].to_numpy(float)
    T = read_telemetry(b.joints)
    al = _alignment(b, t_f, df[f"meas_q{KNEE_SLOT}"].to_numpy(float), T)
    if al is None:
      print(f"[PLOT] NOTE: {b.dir.name}: no usable session-pair alignment -- skipped")
      continue
    ta = T["t"]
    t_in_flight = ta + (al["b"] + al["m"] * ta)
    keep = (t_in_flight >= t_f[0]) & (t_in_flight <= t_f[-1])
    if keep.sum() < 50:
      print(f"[PLOT] NOTE: {b.dir.name}: telemetry does not overlap the Run slice -- skipped")
      continue
    power = np.abs(T["tau"] * T["dq"]).sum(axis=1)[keep]
    tp = t_in_flight[keep] - t_f[0]
    col = POLICY_COLOR.get(p, "0.4")
    # One row per policy, each shaded with its OWN command trace. Overlaying the Runs and
    # shading from one of them would put another Run's regime boundaries behind every
    # trace, which is exactly the pooling CONTEXT.md's "Regime" entry forbids.
    walk = walking_mask(df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float))
    # One span per contiguous walking block, not a fill_between over all ~100k recorder
    # rows: the row-wise form writes every row as path data and was 6.5 MB of the .pgf.
    wi = np.flatnonzero(walk)
    for blk in np.split(wi, np.flatnonzero(np.diff(wi) != 1) + 1):
      if len(blk):
        ax.axvspan(t_f[blk[0]] - t_f[0], t_f[blk[-1]] - t_f[0], color="0.85", linewidth=0,
                   zorder=0)
    tt, yy = envelope_downsample(tp, power, downsample)
    ax.plot(tt, yy, color=col, linewidth=0.5, alpha=0.95, zorder=2)
    ax.set_title(f"{tex(p)}  ({tex(b.tag)})", loc="left", color=col)
    ax.set_ylabel("W")
    drawn.append(power)
    rows += [{"policy": p, "run": b.tag, "bundle": b.dir.name, "t_s": round(float(x), 4),
              "mech_power_w": round(float(y), 4)} for x, y in zip(tt, yy)]
    print(f"[PLOT] f_power: {p} = {b.tag} ({b.dir.name}), align {al['src']} "
          f"b={al['b']:+.3f}s m={al['m'] * 1e6:+.0f}ppm, {int(keep.sum())} telemetry rows")
  # The FSM-entry transient peaks ~4x the walking peaks and would flatten every Run; the
  # axis is capped at the pooled p99.9 and the cap is stated; the clipped samples are still
  # plotted (they run off the top) and are still in the panel CSV.
  cap = float(np.percentile(np.concatenate(drawn), 99.9))
  over = int(sum(int((d > cap).sum()) for d in drawn))
  axes[0].set_ylim(0, cap * 1.05)
  axes[-1].set_xlabel("time since Run start (s)")
  fig.suptitle(r"mechanical power $\sum_j |\tau_{\mathrm{est},j}\,\dot q_j|$ (measured), "
               "walking regime shaded", fontsize=8)
  from matplotlib.patches import Patch
  bottom_legend(fig, [Patch(facecolor="0.85", edgecolor="none", label="walking regime")],
                ncol=1, height=0.10)
  fig.subplots_adjust(top=0.90)
  print(f"[PLOT] f_power: y capped at p99.9 = {cap:.0f} W ({over} sample(s) above it)")
  write_panel_csv(out_dir, name, "mech_power_w",
                  ["policy", "run", "bundle", "t_s", "mech_power_w"], rows)
  return save_pgf(fig, out_dir, name)


# ============================================================ latex check ====

WRAPPER = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=15mm]{geometry}
\usepackage{pgf}
\usepackage{graphicx}
\begin{document}
\input{%s}
\end{document}
"""


def check_latex(pgf_paths: list[Path]) -> int:
  """Compile each .pgf standalone; a figure that cannot be \\input is not a deliverable."""
  bad = 0
  for p in pgf_paths:
    with tempfile.TemporaryDirectory() as td:
      tex = Path(td) / "wrap.tex"
      tex.write_text(WRAPPER % p.resolve().as_posix())
      r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                          "wrap.tex"], cwd=td, capture_output=True, text=True)
      pdf = Path(td) / "wrap.pdf"
      ok = r.returncode == 0 and pdf.exists()
      size = pdf.stat().st_size if pdf.exists() else 0
      print(f"[LATEX] {'OK  ' if ok else 'FAIL'} {p.name}  ({size} B pdf)")
      if not ok:
        bad += 1
        tail = [ln for ln in r.stdout.splitlines() if ln.startswith("!")][:5]
        print("        " + "\n        ".join(tail or r.stdout.splitlines()[-5:]))
  return bad


# ===================================================================== main ====

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--root", default=str(REPO), help="repo root holding logs/ and data/")
  ap.add_argument("--out", default=None, help="output directory (default: <root>/figures)")
  ap.add_argument("--figures", default=None,
                  help="comma-separated subset of: " + ",".join(FIGURES))
  ap.add_argument("--cadence-json", default=None,
                  help=f"analyze_cadence_settling.py output (default: {CADENCE_JSON})")
  ap.add_argument("--max-points-per-line", type=int, default=4000,
                  help="envelope-decimate any time series longer than this (default 4000; "
                       "keeps the .pgf small enough to recompile with the thesis)")
  ap.add_argument("--no-downsample", action="store_true",
                  help="disable the min/max-envelope decimation on the time-series figures")
  ap.add_argument("--check-latex", action="store_true",
                  help="compile every written .pgf standalone with pdflatex")
  args = ap.parse_args()

  root = Path(args.root)
  out_dir = Path(args.out) if args.out else root / "figures"
  bundles = load_bundles(root)
  if not bundles:
    raise SystemExit(f"no run bundles under {root / BUNDLE_GLOB}")
  print(f"[PLOT] {len(bundles)} Run bundle(s): " + ", ".join(
    f"{b.tag}={b.policy}" for b in bundles))

  cad_path = Path(args.cadence_json) if args.cadence_json else root / CADENCE_JSON
  cad_rows = json.loads(cad_path.read_text()) if cad_path.exists() else []
  names = args.figures.split(",") if args.figures else list(FIGURES)
  ds = not args.no_downsample
  set_decimation(args.max_points_per_line)

  written = []
  for n in names:
    if n not in FIGURES:
      raise SystemExit(f"unknown figure '{n}'; choices: {list(FIGURES)}")
    if n in ("cadence", "settle") and not cad_rows:
      raise SystemExit(f"f_{n} needs {cad_path} (run scripts/analyze_cadence_settling.py)")
    if n == "cadence":
      written.append(fig_cadence(bundles, cad_rows, out_dir))
    elif n == "settle":
      written.append(fig_settle(bundles, cad_rows, out_dir))
    elif n == "upper":
      written.append(fig_upper(bundles, out_dir))
    elif n == "smooth":
      written.append(fig_smooth(bundles, out_dir, downsample=ds))
    elif n == "power":
      written.append(fig_power(bundles, out_dir, downsample=ds))

  print(f"[PLOT] {len(written)} figure(s) in {out_dir}")
  if args.check_latex:
    bad = check_latex(written)
    if bad:
      print(f"[LATEX] {bad} figure(s) failed to compile")
      return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
