#!/usr/bin/env python3
"""LaTeX-ready thesis figures from the 2026-09-14 hardware _Run_ bundles.

Nine figures, each written TWICE: a `.pgf` to `\\input` into the thesis (matplotlib's own
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
  f_track    operator command vs onboard estimate vs MOTION-CAPTURE ground truth, one Run,
             with the estimator's RMS error against truth per _Regime_ -- the number every
             earlier hardware velocity claim lacked, having been scored against the
             estimator itself.
  f_stand    metric F1: world-frame drift during zero-command holds, from ground truth,
             plus the MEASURED residual speed at rest per Run.
  f_hier     the hierarchy's error decomposition: operator command vs the HL's target V*
             vs the LL's delivered velocity (vx/vy: whichever arm the deploy config's
             `hrl.base_estimator` feeds to obs["hl_vel"], read per-Run from its meta.json,
             not assumed; wz: the direct gyro reading, since the estimator bank is vx/vy
             only) vs ground truth, plus goal saturation.
  f_chain    training -> sim bench -> hardware, per metric, for metrics defined identically
             at all three stages. Carries its caveats on the figure.

Sources (nothing here is recomputed that already exists):
  logs/robot_logs/<glob>/          one _Run_ bundle each: sliced flight recorder, optional
                                   HRL telemetry, optional _Joint telemetry log_, run.json
                                   (the ONLY authority on `policy_tag`), and the
                                   already-scored `*.standing.json` / `*.walking.json`.
                                   `--bundles` sets the glob(s); default is the
                                   2026-09-14 session alone, so a later session (e.g. the
                                   seed-band runs) must be named explicitly or merged in.
  data/2026-09-14-hardware-session/cadence_settling.json
                                   scripts/analyze_cadence_settling.py's output; f_cadence,
                                   f_settle and f_stand's residual speed READ it, they do
                                   not re-derive it.
  <bundle>/mocap_aligned.csv       mocap_align.py's pelvis ground truth, already on the
                                   flight recorder's `t` grid. Present on 3 of 11 Runs; the
                                   figures that need it SAY which Runs lack it.
  data/2026-09-09-wp3-baseline-arms/*_bench.json
                                   the sim point of f_chain. Each arm carries its checkpoint
                                   path, so the training run is read FROM it -- the chain's
                                   three points are provably the same policy.
  logs/rsl_rl/**/events.out.tfevents*
                                   the training point of f_chain, via tensorboard's
                                   EventAccumulator (tbparse is not installed).

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
  python scripts/plot_thesis_figures.py                      # all nine into figures/
  python scripts/plot_thesis_figures.py --figures track --run "run 4"
  python scripts/plot_thesis_figures.py --figures cadence,settle
  python scripts/plot_thesis_figures.py --out figures --check-latex
  python scripts/plot_thesis_figures.py --bundles "logs/robot_logs/2026_09_14-*/" \\
      "logs/robot_logs/2026_09_2?-*/"                         # pool multiple sessions
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
from plot_deploy_logs import envelope_downsample, _window_starts  # noqa: E402
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
MOCAP_CSV = "mocap_aligned.csv"   # mocap_align.py output, on the flight recorder's `t` grid
PERIOD_RANGE = (0.35, 1.0)      # analyze_cadence_settling.PERIOD_RANGE (deploy yaml)
ARM_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)  # rad/s, analyze_cadence_settling.ARM_THRESHOLDS
FIG_W = 5.9                     # in; ~15 cm thesis column
FIGURES = ("cadence", "settle", "upper", "smooth", "power",
           "track", "stand", "hier", "chain")

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
  mocap: Path | None = None     # mocap_aligned.csv, present on 3 of the 11 Runs
  regimes: dict = field(default_factory=dict)
  # The arm feeding obs["hl_vel"] (safety_logger.h), i.e. what `lo_vx`/`lo_vy` actually
  # ARE -- "none" for A0 (no HL, bank runs passively). Read per-Run, never assumed: the
  # `lo_` column name predates the 7-arm bank and is not evidence of which one is live.
  base_estimator: str = "none"

  @property
  def walk_s(self) -> float:
    return float(self.regimes.get("walking", {}).get("duration_s", 0.0))


def policy_color(label: str) -> str:
  """Colour by BASE policy tag, so a loaded Run stays recognisably the same policy.

  `policy_label` folds the _Mounted payload_ into the name, which means `A1a +7.5kg` is not
  a key of POLICY_COLOR: keyed naively, every loaded Run falls back to the SAME default
  grey and two different policies become one colour.
  """
  return POLICY_COLOR.get(label.split(" +")[0], "0.4")


def policy_style(label: str) -> dict:
  """Colour by policy, dash + square marker when the Run is loaded. Colour alone cannot
  separate `A1a` from `A1a +7.5kg` in a line legend."""
  loaded = " +" in label
  return {"color": policy_color(label), "linestyle": "--" if loaded else "-",
          "marker": "s" if loaded else "o"}


def policy_sort_key(label: str):
  """POLICY_ORDER for the base tag, then payload -- an unloaded Run before its loaded one."""
  base = label.split(" +")[0]
  kg = float(label.split(" +")[1].rstrip("kg")) if " +" in label else 0.0
  idx = POLICY_ORDER.index(base) if base in POLICY_ORDER else len(POLICY_ORDER)
  return (idx, base, kg)


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


def load_bundles(root: Path, globs: list[str] = (BUNDLE_GLOB,)) -> list[Bundle]:
  out = []
  dirs = sorted({d for pattern in globs for d in root.glob(pattern)})
  for d in dirs:
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
    meta_path = flight.with_name(flight.stem + "_meta.json")
    base_estimator = "none"
    if meta_path.exists():
      base_estimator = json.loads(meta_path.read_text()).get("base_estimator", "none")
    b = Bundle(dir=d, run=run, policy=policy_label(run),
               # `note` is free text and can run to a paragraph (run 4 carries the
               # label-correction story); the point label is only ever its leading id.
               tag=(run.get("note") or d.name).split(",")[0].split(".")[0].strip()[:24],
               flight=flight, hrl=hrl if hrl.exists() else None,
               joints=joints if (joints and joints.exists()) else None,
               # The ALIGNED capture is the authority, not run.json's `mocap` field: run 6
               # names a trial whose capture was rejected, so the field is set and the
               # aligned file is absent.
               mocap=(d / MOCAP_CSV) if (d / MOCAP_CSV).exists() else None,
               base_estimator=base_estimator)
    for regime in ("standing", "walking"):
      p = d / f"{flight.stem}.{regime}.json"
      if p.exists():
        b.regimes[regime] = json.loads(p.read_text())
    out.append(b)
  return out


def policies_present(bundles: list[Bundle]) -> list[str]:
  return sorted({b.policy for b in bundles}, key=policy_sort_key)


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

DRAWN: dict[str, list[float]] = {}


def harvest_drawn(fig, name: str) -> None:
  """Record every y value actually drawn in DATA coordinates, for --verify-csv.

  Taken off the live figure rather than parsed back out of the .pgf, because a .pgf stores
  path coordinates in inches after the axis transform: recovering data values from it means
  re-deriving the transform, which is more machinery than the thing being checked.

  Only artists on `ax.transData` count. Reference lines (axhline/axvline: the action-space
  ceiling, the A0 anchor, the zero-command floor) sit on a BLENDED transform, and legend
  proxies carry no data -- both are annotations, not series, and neither belongs in a panel
  CSV. Filtering by transform says exactly that, where a point-count threshold would also
  silently drop a real two-Run policy from the scalar figures.
  """
  vals: list[float] = []
  for ax in fig.axes:
    for ln in ax.get_lines():
      if ln.get_transform() is not ax.transData:
        continue
      y = np.asarray(ln.get_ydata(), float)
      vals += [float(v) for v in y[np.isfinite(y)]]
    for cont in getattr(ax, "containers", []):
      for patch in getattr(cont, "patches", []):
        h = patch.get_height()
        if np.isfinite(h):
          vals.append(float(h))
  DRAWN[name] = vals


def check_panel_csv(out_dir: Path, names: list[str]) -> int:
  """Every value drawn must appear in that figure's panel CSVs. Returns the failure count.

  The CSVs are the archived numbers -- the thesis quotes them and a reader re-plots from
  them -- so a series that is drawn but not written is a figure nobody can reproduce, and a
  panel CSV built from a different array than the one plotted is worse than none. Values are
  matched, not rows: the CSV schema differs per figure (long form here, one row per bar
  there) and pinning this check to a schema would make it a restatement of the writer.
  """
  bad = 0
  for name in names:
    drawn = DRAWN.get(f"f_{name}")
    if drawn is None:
      continue
    pool: list[float] = []
    csvs = sorted(out_dir.glob(f"f_{name}_*.csv"))
    for p in csvs:
      with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
          for v in row.values():
            try:
              f = float(v)
            except (TypeError, ValueError):
              continue
            if np.isfinite(f):
              pool.append(f)
    arr = np.array(sorted(pool)) if pool else np.empty(0)
    missing = 0
    for v in drawn:
      if len(arr) == 0:
        missing += 1
        continue
      i = int(np.searchsorted(arr, v))
      near = min(abs(v - arr[j]) for j in (max(i - 1, 0), min(i, len(arr) - 1)))
      # The writers round to 4-6 decimals, so an exact match is not available; a genuinely
      # different series misses by orders of magnitude, not by a rounding step.
      if near > max(1e-3, 1e-3 * abs(v)):
        missing += 1
    ok = missing == 0 and len(csvs) > 0
    note = "no panel CSV written" if not csvs else f"{missing}/{len(drawn)} drawn values absent"
    print(f"[CSV] {'OK  ' if ok else 'FAIL'} f_{name}  {len(drawn)} drawn, "
          f"{len(pool)} CSV values, {len(csvs)} panel file(s)" + ("" if ok else f"  <-- {note}"))
    bad += 0 if ok else 1
  return bad


def save_pgf(fig, out_dir: Path, name: str) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"{name}.pgf"
  fig.savefig(path)
  harvest_drawn(fig, name)
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
  return [Line2D([], [], markersize=3.5, linewidth=1.0, label=tex(p), **policy_style(p))
          for p in policies]


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
        ax.plot([x + dx, x + dx], [s["p5"], s["p95"]], color=policy_color(p),
                linewidth=0.7, alpha=0.7, zorder=2)
        ax.plot(x + dx, s["mean"], marker="o", markersize=3.6, zorder=3,
                color=policy_color(p),
                markerfacecolor="white" if thin else policy_color(p))
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
              fontsize=6, color=policy_color(p))
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
    col = policy_color(b.policy)
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
        ax.bar(xpos, m, width=width * 0.92, color=policy_color(p),
               alpha=REGIME_ALPHA[regime], edgecolor=policy_color(p),
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
    col = policy_color(p)
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
    col = policy_color(p)
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


# ================================================================= f_track ====

def _mocap_note(bundles) -> str:
  """One line naming which Runs carry ground truth. A figure that just draws the Runs it
  could draw silently redefines the sample; the ones without motion capture are named."""
  have = [b.tag for b in bundles if b.mocap]
  miss = [b.tag for b in bundles if not b.mocap]
  return (f"motion-capture ground truth exists for {len(have)} of {len(bundles)} Runs: "
          + ", ".join(have) + ".\n"
          + f"The other {len(miss)} cannot appear here (run 6's capture was rejected as "
            "corrupt; the rest were not captured).")


def read_mocap(b: Bundle, t: np.ndarray) -> dict:
  """Ground truth on the flight recorder's own `t` grid.

  `mocap_align.py` writes it already aligned to that grid, so the columns line up 1:1 on
  every Run measured here. Interpolation is a fallback for a partial capture, never the
  normal path -- resampling a pose stream would quietly smooth the drift f_stand measures.
  """
  M = pd.read_csv(b.mocap)
  tm = M["t"].to_numpy(float)
  if len(tm) == len(t) and np.allclose(tm, t):
    return {c: M[c].to_numpy(float) for c in M.columns if c != "t"}
  print(f"[PLOT] NOTE: {b.dir.name}: mocap grid differs from the flight grid "
        f"({len(tm)} vs {len(t)} rows) -- interpolating onto the flight grid")
  return {c: np.interp(t, tm, M[c].to_numpy(float)) for c in M.columns if c != "t"}


def pick_mocap_run(bundles, need_hrl=False, run=None):
  """The mocap Run with the most WALKING time, or the one `--run` names."""
  cands = [b for b in bundles if b.mocap and (b.hrl or not need_hrl)]
  if not cands:
    raise SystemExit("no Run has mocap_aligned.csv" + (" and HRL telemetry" if need_hrl else ""))
  if run:
    hit = [b for b in cands if run in b.tag or run in b.dir.name]
    if not hit:
      raise SystemExit(f"--run {run!r} matches no mocap Run; have: "
                       + ", ".join(f"{b.tag} ({b.dir.name})" for b in cands))
    return hit[0]
  return max(cands, key=lambda b: b.walk_s)


def _rms(a, b, m):
  return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() > 10 else float("nan")


def fig_track(bundles, out_dir, name="f_track", run=None, downsample=True):
  """Command vs onboard estimate vs motion-capture ground truth, one _Run_.

  This is the figure the capture session exists for. Until it, every hardware velocity
  number on this project was scored against the estimator itself (`err_vx_est`), and the
  estimator's error is a function of the policy's own motion (WL-F), so a policy could look
  accurate by moving in a way its own estimator likes. The per-panel RMS against truth is
  the deliverable; it is split by _Regime_, never pooled.
  """
  b = pick_mocap_run(bundles, run=run)
  df = pd.read_csv(b.flight, usecols=["t", "cmd_vx", "cmd_vy", "cmd_wz",
                                      "est_v_compl_x", "est_v_compl_y", "est_gyro_z"])
  t = df["t"].to_numpy(float)
  G = read_mocap(b, t)
  walk = walking_mask(df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float))
  regimes = {"walking": walk, "standing": ~walk}

  # The wz panel is a different test from the other two and the labels say so: est_v_compl is
  # a fused ESTIMATE under scrutiny, while the gyro is a direct rate measurement, so its
  # agreement with capture checks the FRAME alignment, not estimator drift. One shared
  # legend entry would claim three estimator tests where there are two.
  panels = [
    ("vx", "cmd_vx", "est_v_compl_x", "gt_vx", "m/s", "$v_x$",
     "onboard estimate (fused leg odometry + IMU)"),
    ("vy", "cmd_vy", "est_v_compl_y", "gt_vy", "m/s", "$v_y$",
     "onboard estimate (fused leg odometry + IMU)"),
    ("wz", "cmd_wz", "est_gyro_z", "gt_wz", "rad/s", "$\\omega_z$",
     "gyro $z$ (a direct measurement, not an estimate)"),
  ]
  fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 4.4), sharex=True)
  rows = {}
  rms_report = {}
  for ax, (axis, cmd_c, est_c, gt_c, unit, sym, est_name) in zip(axes, panels):
    est, gt = df[est_c].to_numpy(float), G[gt_c]
    series = (("operator command", df[cmd_c].to_numpy(float), "0.25", "--", 0.8),
              (est_name, est, "#D55E00", "-", 0.7),
              ("motion-capture ground truth", gt, "#0072B2", "-", 0.7))
    rows[axis] = []
    for label, y, colr, ls, lw in series:
      tt, yy = envelope_downsample(t - t[0], y, downsample)
      ax.plot(tt, yy, color=colr, linestyle=ls, linewidth=lw, alpha=0.9,
              label=label if ax is axes[0] else None)
      rows[axis] += [{"policy": b.policy, "run": b.tag, "bundle": b.dir.name,
                      "series": label.replace("\\", ""), "t_s": round(float(x), 4),
                      "value": round(float(v), 6), "unit": unit}
                     for x, v in zip(tt, yy)]
    r = {k: _rms(est, gt, m) for k, m in regimes.items()}
    rms_report[axis] = r
    ax.text(0.005, 0.97, f"estimator RMS vs truth:  standing {r['standing']:.4f}, "
            f"walking {r['walking']:.4f} {unit}", transform=ax.transAxes, va="top",
            ha="left", fontsize=6, color="#D55E00")
    ax.set_ylabel(f"{sym} ({unit})")
    ax.margins(y=0.22)
  axes[-1].set_xlabel("time since Run start (s)")
  fig.suptitle(f"velocity tracking vs ground truth -- {tex(b.policy)}, {tex(b.tag)}",
               fontsize=8)
  bottom_legend(fig, [
    plt.Line2D([], [], color="0.25", linestyle="--", label="operator command"),
    plt.Line2D([], [], color="#D55E00", label="onboard estimate / gyro"),
    plt.Line2D([], [], color="#0072B2", label="motion-capture ground truth")],
    ncol=3, height=0.20)
  fig.text(0.5, 0.185, _mocap_note(bundles), ha="center", va="top", fontsize=5.6,
           color="0.35", linespacing=1.6)
  fig.subplots_adjust(top=0.93)
  for axis in rows:
    write_panel_csv(out_dir, name, axis,
                    ["policy", "run", "bundle", "series", "t_s", "value", "unit"], rows[axis])
  print(f"[PLOT] f_track: {b.policy} = {b.tag} ({b.dir.name}); estimator RMS vs truth "
        + "; ".join(f"{a} stand {v['standing']:.4f} walk {v['walking']:.4f}"
                    for a, v in rms_report.items()))
  # The RMS is the deliverable, and it is a property of each Run, not of the one drawn --
  # printed for every Run that has ground truth so a single figure does not become the
  # only place the number exists.
  for o in bundles:
    if o.mocap is None or o is b:
      continue
    od = pd.read_csv(o.flight, usecols=["t", "cmd_vx", "cmd_vy", "cmd_wz",
                                        "est_v_compl_x", "est_v_compl_y", "est_gyro_z"])
    ot = od["t"].to_numpy(float)
    OG = read_mocap(o, ot)
    ow = walking_mask(od[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float))
    txt = []
    for axis, _c, est_c, gt_c, _u, _s, _n in panels:
      e, g = od[est_c].to_numpy(float), OG[gt_c]
      txt.append(f"{axis} stand {_rms(e, g, ~ow):.4f} walk {_rms(e, g, ow):.4f}")
    print(f"[PLOT] f_track: (not drawn) {o.policy} = {o.tag} ({o.dir.name}); "
          "estimator RMS vs truth " + "; ".join(txt))
  return save_pgf(fig, out_dir, name)


# ================================================================= f_stand ====

STAND_MIN_SEG_S = 10.0   # analyze_cadence_settling.MIN_SETTLE_SEG_S, same reasoning


def _still_segments(t, walk, min_s):
  """Contiguous zero-command holds at least `min_s` long.

  Shorter holds measure how the operator drove the stick, not how well the robot stands --
  the same floor `analyze_cadence_settling.settling_times` applies, for the same reason.
  """
  idx = np.flatnonzero(~walk)
  out = []
  for blk in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
    if len(blk) and t[blk[-1]] - t[blk[0]] >= min_s:
      out.append(blk)
  return out


def fig_stand(bundles, cad_rows, out_dir, name="f_stand", min_seg_s=STAND_MIN_SEG_S,
              downsample=True):
  """Metric F1 -- how far the robot walks away while told to stand still, exactly.

  Every previous version of this number came from an estimator; this one is world-frame
  motion capture, so it is a distance, not an integrated estimate. Paths are re-zeroed per
  segment: absolute position says where the operator started the hold, the drift vector is
  the policy's.
  """
  by_bundle = {r["bundle"]: r for r in cad_rows}
  have = [b for b in bundles if b.mocap]
  if not have:
    raise SystemExit("f_stand needs mocap_aligned.csv; no Run has one")
  policies = sorted({b.policy for b in have}, key=policy_sort_key)
  fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.0),
                           gridspec_kw={"width_ratios": [1.25, 1.0]})
  axp, axd = axes
  path_rows, disp_rows = [], []

  for b in have:
    df = pd.read_csv(b.flight, usecols=["t", "cmd_vx", "cmd_vy", "cmd_wz"])
    t = df["t"].to_numpy(float)
    G = read_mocap(b, t)
    walk = walking_mask(df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float))
    segs = _still_segments(t, walk, min_seg_s)
    col = policy_color(b.policy)
    disp = []
    for si, blk in enumerate(segs):
      x = G["gt_px"][blk] - G["gt_px"][blk[0]]
      y = G["gt_py"][blk] - G["gt_py"][blk[0]]
      # A path is decimated by INDEX, not by the envelope: min/max on x and y separately
      # would reorder the samples and draw a shape the robot never walked.
      k = np.linspace(0, len(blk) - 1, min(len(blk), 400)).astype(int)  # a path, not a series
      axp.plot(x[k], y[k], color=col, linewidth=0.7, alpha=0.75, zorder=2)
      axp.plot(x[k][-1], y[k][-1], marker="o", markersize=2.6, color=col, zorder=3)
      d = float(np.hypot(x[-1], y[-1]))
      disp.append(d)
      path_rows += [{"policy": b.policy, "run": b.tag, "bundle": b.dir.name, "segment": si,
                     "hold_s": round(float(t[blk[-1]] - t[blk[0]]), 2),
                     "x_m": round(float(xv), 5), "y_m": round(float(yv), 5)}
                    for xv, yv in zip(x[k], y[k])]
    floor = ((by_bundle.get(b.dir.name, {}).get("base") or {}).get("floor_m_s"))
    xpos = policies.index(b.policy)
    sib = [o for o in have if o.policy == b.policy]
    dx = (sib.index(b) - 0.5 * (len(sib) - 1)) * (0.30 / max(len(sib) - 1, 1))
    total = float(np.sum(disp)) if disp else float("nan")
    axd.plot(xpos + dx, total, marker="o", markersize=4.0, color=col, zorder=3)
    if floor is not None:
      axd.annotate(f"{floor:.4f} m/s", (xpos + dx, total), textcoords="offset points",
                   xytext=(0, 6), ha="center", fontsize=5.6, color=col)
    disp_rows.append({"policy": b.policy, "run": b.tag, "bundle": b.dir.name,
                      "n_segments": len(segs), "min_segment_s": min_seg_s,
                      "total_displacement_m": round(total, 5),
                      "max_segment_displacement_m": round(float(max(disp)), 5) if disp else "",
                      "residual_speed_m_s": floor if floor is not None else ""})
    print(f"[PLOT] f_stand: {b.policy} {b.tag}: {len(segs)} hold(s) >= {min_seg_s:g} s, "
          f"total drift {total:.3f} m, residual speed {floor}")

  axp.axhline(0, color="0.8", linewidth=0.5)
  axp.axvline(0, color="0.8", linewidth=0.5)
  axp.set_aspect("equal", adjustable="datalim")
  axp.set_xlabel("world X drift (m)")
  axp.set_ylabel("world Y drift (m)")
  axp.set_title(f"zero-command holds $\\geq$ {min_seg_s:g} s, each re-zeroed", fontsize=7.5)
  axd.set_xticks(range(len(policies)))
  axd.set_xticklabels([tex(p) for p in policies], rotation=30, ha="right", fontsize=6)
  axd.set_xlim(-0.6, len(policies) - 0.4)
  axd.set_ylabel("total drift over all holds (m)")
  axd.set_title("per Run (label = residual speed at rest)", fontsize=7.5)
  axd.margins(y=0.3)
  bottom_legend(fig, policy_handles(policies), ncol=len(policies), height=0.30)
  fig.text(0.5, 0.275, _mocap_note(bundles) + "\nresidual speed is the MEASURED floor from "
           "analyze\\_cadence\\_settling.py (base.floor\\_m\\_s), not re-derived here",
           ha="center", va="top", fontsize=5.6, color="0.35", linespacing=1.6)
  write_panel_csv(out_dir, name, "path",
                  ["policy", "run", "bundle", "segment", "hold_s", "x_m", "y_m"], path_rows)
  write_panel_csv(out_dir, name, "displacement",
                  ["policy", "run", "bundle", "n_segments", "min_segment_s",
                   "total_displacement_m", "max_segment_displacement_m",
                   "residual_speed_m_s"], disp_rows)
  return save_pgf(fig, out_dir, name)


# ================================================================== f_hier ====

def fig_hier(bundles, out_dir, name="f_hier", run=None, downsample=True):
  """HL intent vs LL delivery vs ground truth -- the hierarchy's error decomposition.

  `plot_deploy_logs.view_goaltrack` draws the first three traces; the fourth (motion
  capture) is what makes the second half of the decomposition answerable. `lo_*` is
  window-latched, so it is sampled on the HL FIRE rows and drawn as the piecewise-constant
  signal it is -- reading it per row would invent intermediate values the LL never saw.

  `lo_vx`/`lo_vy` are NOT necessarily leg odometry: safety_logger.h feeds obs["hl_vel"]
  from whichever of the seven base-estimator arms the deploy config's `hrl.base_estimator`
  names (`none` for A0, which runs the bank passively), and only that column name is a
  holdover from before the bank existed. Every live A1 deploy config ships `ekf_rot`. The
  label below reads `b.base_estimator` (from the Run's own meta.json) rather than assuming.
  """
  b = pick_mocap_run(bundles, need_hrl=True, run=run)
  hdf = pd.read_csv(b.hrl)
  th = hdf["t"].to_numpy(float)
  df = pd.read_csv(b.flight, usecols=["t", "cmd_vx", "cmd_vy", "cmd_wz", "est_gyro_z"])
  t = df["t"].to_numpy(float)
  G = read_mocap(b, t)
  fire = _window_starts(hdf)
  ll_label = f"LL achieved ({b.base_estimator}, window-latched)"
  gyro_label = "LL achieved (gyro, direct measurement)"

  fig, axes = plt.subplots(4, 1, figsize=(FIG_W, 5.0), sharex=True)
  rows = {}
  # wz has no counterpart in the base-estimator bank (vx/vy only, safety_logger.h) --
  # est_gyro_z is a direct torso-IMU rate measurement, not a filtered "arm", and is read
  # straight off `df` at the flight recorder's own rate rather than window-latched off
  # `hdf`/`fire` like lo_vx/lo_vy: there is no HL window to latch it to.
  panels = [("vx", "cmd_vx", "tgt0", "lo_vx", "gt_vx", "m/s"),
            ("vy", "cmd_vy", "tgt1", "lo_vy", "gt_vy", "m/s"),
            ("wz", "cmd_wz", "tgt2", "est_gyro_z", "gt_wz", "rad/s")]
  for ax, (axis, cmd_c, tgt_c, lo_c, gt_c, unit) in zip(axes, panels):
    rows[axis] = []
    drawn = [("operator command", t - t[0], df[cmd_c].to_numpy(float), "0.25", "--", {}),
             ("HL target V*", th - t[0], hdf[tgt_c].to_numpy(float), "#009E73", "-", {})]
    if axis == "wz":
      if lo_c in df:
        drawn.append((gyro_label, t - t[0], df[lo_c].to_numpy(float), "#D55E00", "-", {}))
      else:
        ax.text(0.005, 0.03, "no gyro column in this flight recorder (with_estimator off)",
                transform=ax.transAxes, fontsize=5.6, color="#D55E00", va="bottom")
    elif lo_c and lo_c in hdf:
      drawn.append((ll_label, th[fire] - t[0],
                    hdf[lo_c].to_numpy(float)[fire], "#D55E00", "-",
                    dict(drawstyle="steps-post")))
    drawn.append(("motion-capture ground truth", t - t[0], G[gt_c], "#0072B2", "-", {}))
    for k, (label, tx, y, colr, ls, kw) in enumerate(drawn):
      tt, yy = envelope_downsample(tx, y, downsample)
      # Ground truth is drawn last and heaviest: the HL target swings wider than anything
      # the robot did, and under equal weights it simply paints over the reference.
      truth = label.startswith("motion-capture")
      ax.plot(tt, yy, color=colr, linestyle=ls, alpha=1.0 if truth else 0.75,
              linewidth=0.85 if truth else 0.6, zorder=4 if truth else 2 + k, **kw)
      rows[axis] += [{"policy": b.policy, "run": b.tag, "bundle": b.dir.name,
                      "series": label, "t_s": round(float(x), 4),
                      "value": round(float(v), 6), "unit": unit}
                     for x, v in zip(tt, yy)]
    ax.set_ylabel(f"{axis} ({unit})")
    ax.margins(y=0.2)

  # Saturation: |V*| against the bound the session actually used. The baked `goal_scale`
  # is a checkpoint property and is not in the log, so the observed bound is used, exactly
  # as plot_deploy_logs.view_goaltrack does -- and it is labelled as observed.
  sat_rows = []
  walk_h = walking_mask(hdf[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float))
  for tgt_c, axis, colr in (("tgt0", "vx", "#0072B2"), ("tgt1", "vy", "#D55E00")):
    a = np.abs(hdf[tgt_c].to_numpy(float))
    scale = float(np.round(a.max(), 2)) or 1.0
    sat = float((a[walk_h] > 0.98 * scale).mean()) if walk_h.any() else float("nan")
    tt, yy = envelope_downsample(th - t[0], a / scale, downsample)
    axes[3].plot(tt, yy, color=colr, linewidth=0.7, alpha=0.9,
                 label=f"$|V^*|$ {axis} / {scale:g} (walk saturated {100 * sat:.0f}%)")
    sat_rows += [{"policy": b.policy, "run": b.tag, "bundle": b.dir.name,
                  "series": f"|tgt_{axis}|/observed_bound", "t_s": round(float(x), 4),
                  "value": round(float(v), 6), "unit": f"fraction of {scale:g}"}
                 for x, v in zip(tt, yy)]
  axes[3].axhline(1.0, color="red", linestyle=":", linewidth=0.7)
  axes[3].set_ylabel("fraction of bound")
  axes[3].legend(fontsize=5.6, loc="upper right", framealpha=0.9)
  axes[3].set_xlabel("time since Run start (s)")
  fig.suptitle(f"HL goal vs LL delivery vs ground truth -- {tex(b.policy)}, {tex(b.tag)}",
               fontsize=8)
  bottom_legend(fig, [
    plt.Line2D([], [], color="0.25", linestyle="--", label="operator command"),
    plt.Line2D([], [], color="#009E73", label="HL target V*"),
    plt.Line2D([], [], color="#D55E00",
               label=f"LL achieved ({b.base_estimator} vx/vy / gyro wz)"),
    plt.Line2D([], [], color="#0072B2", label="motion-capture ground truth")],
    ncol=2, height=0.20)
  fig.text(0.5, 0.185, _mocap_note(bundles), ha="center", va="top", fontsize=5.6,
           color="0.35", linespacing=1.6)
  fig.subplots_adjust(top=0.94)
  for axis in rows:
    write_panel_csv(out_dir, name, axis,
                    ["policy", "run", "bundle", "series", "t_s", "value", "unit"], rows[axis])
  write_panel_csv(out_dir, name, "saturation",
                  ["policy", "run", "bundle", "series", "t_s", "value", "unit"], sat_rows)
  print(f"[PLOT] f_hier: {b.policy} = {b.tag} ({b.dir.name}), {len(fire)} HL fires")
  return save_pgf(fig, out_dir, name)


# ================================================================= f_chain ====

# Hardware policy -> the sim bench arm that ran the SAME checkpoint. The arm files carry the
# checkpoint path, so the training run is read from them rather than guessed: the chain's
# three points are then provably the same policy.
CHAIN_ARMS = {
  "A0_DR_s123": ("all_bench.json", "fmem_s123"),
  "A1a": ("nodr_bench.json", "nodr_s42"),
  "A1a_DR_s123": ("all_bench.json", "hmem_s123"),
  "A1a_DR_cotcap_s42": ("cotcap_bench.json", "cotcap_s42"),
}
SIM_BENCH_DIR = "data/2026-09-09-wp3-baseline-arms"
SIM_CONDITION = "base_p0"     # the unloaded baseline condition of each arm
KAPPA_LEG = 0.25              # action scale, leg joints (uniform) -- raw action units -> rad
TRAIN_TAIL = 500              # iterations averaged at the end of training

# (key, axis label, training tag, converter, stochastic-at-training?)
CHAIN_METRICS = [
  ("cot", "cot (diagnostic)", "Loss/metrics/cot", 1.0, False),
  ("mech_power_w", "mech\\_power\\_w (W)", None, 1.0, False),
  ("act_legs_rad", "act\\_legs\\_rad (rad)", "Loss/metrics/act_rate_legs", KAPPA_LEG, True),
  ("jacc_legs", "jacc\\_legs (rad/s$^2$)", "Loss/metrics/jacc_legs", 1.0, True),
  ("err_yaw", "err\\_yaw (rad/s)", "Metrics/twist/error_vel_yaw", 1.0, False),
  ("orient_dev", "orient\\_dev (-)", None, 1.0, False),
  ("omega_xy", "omega\\_xy (rad/s)", None, 1.0, False),
  ("ub_pose_dev", "ub\\_pose\\_dev (rad$^2$)", None, 1.0, False),
  ("ub_arm_vel", "ub\\_arm\\_vel (rad/s)", None, 1.0, False),
]
# Stated, never substituted (bench_flight_recorder.OMITTED carries the same reasons).
CHAIN_OMITTED = {
  "mean_ep_len": "no episode structure on hardware",
  "ss_vx_var": "variance across N parallel envs; N = 1 robot",
  "fall_rate": "no episode denominator on hardware",
  "gait_match": "needs foot contact state; foot_force_* is dead on the robot",
  "angular_momentum": "training-only Episode_Reward term; no sim-bench or hardware counterpart",
  "foot_slip / foot_clearance": "world-frame foot terms; no hardware observable",
}


def _train_scalars(run_dir: Path, tag: str, tail: int = TRAIN_TAIL):
  """Tail-mean of one tfevents scalar, or None when the tag or the reader is absent."""
  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  except ImportError:
    return None
  files = sorted(run_dir.glob("events.out.tfevents*"))
  if not files:
    return None
  ea = EventAccumulator(str(files[0]), size_guidance={"scalars": 0})
  ea.Reload()
  if tag not in ea.Tags()["scalars"]:
    return None
  v = np.array([s.value for s in ea.Scalars(tag)], float)
  return float(np.mean(v[-tail:])) if len(v) else None


def _hw_jacc_legs(b: Bundle, regime: str) -> float:
  """Hardware `jacc_legs`: second difference of MEASURED leg q at the control rate.

  bench_flight_recorder OMITS this key deliberately (sim uses analytic MuJoCo `qacc` at the
  physics rate), so the number does not exist in the regime jsons and is computed here --
  which is exactly why the figure carries the "trends, not levels" caveat on this row.
  """
  t, sidx, steps, meas, walk = _flight_steps(b)
  in_reg = walk[sidx] if regime == "walking" else ~walk[sidx]
  sel = in_reg[2:] & in_reg[1:-1] & in_reg[:-2]
  if sel.sum() < 10:
    return float("nan")
  jac = np.linalg.norm((meas[2:] - 2.0 * meas[1:-1] + meas[:-2]) / CTRL_DT ** 2, axis=1)
  return float(jac[sel].mean())


def fig_chain(bundles, out_dir, name="f_chain", regime="walking", root=REPO):
  """training -> sim bench -> hardware, per metric, for metrics defined the same way at all
  three points. One panel per metric, one line per policy; a stage a metric does not exist
  at is simply absent.

  Read the caveats printed on the figure before reading a slope: the training column is a
  STOCHASTIC policy, the hardware flat-policy Runs carried a 7.5 kg _Mounted payload_ while
  the sim column is unloaded, and hardware `jacc_legs` is differentiated encoder velocity
  against sim's analytic `qacc`.
  """
  policies = sorted({b.policy for b in bundles}, key=policy_sort_key)
  stages = ["training", "sim bench", "hardware"]
  sim_cache, train_cache, hw_jacc = {}, {}, {}

  for p in policies:
    base = p.split(" +")[0]
    arm = CHAIN_ARMS.get(base)
    if not arm:
      print(f"[PLOT] NOTE: f_chain: {p}: no sim-bench arm mapped -- sim/training absent")
      continue
    d = json.loads((root / SIM_BENCH_DIR / arm[0]).read_text())
    cell = d.get(f"{arm[1]}__{SIM_CONDITION}")
    if cell is None:
      print(f"[PLOT] NOTE: f_chain: {p}: {arm[1]}__{SIM_CONDITION} absent from {arm[0]}")
      continue
    sim_cache[p] = cell["bench"]
    ckpt = Path(str(cell["bench"].get("label", "")))
    train_cache[p] = (root / ckpt.parent) if ckpt.parent.name else None

  fig, axes = plt.subplots(3, 3, figsize=(FIG_W, 5.6))
  axes = list(axes.flat)
  for ax, (key, ylabel, tag, conv, stoch) in zip(axes, CHAIN_METRICS):
    rows = []
    for p in policies:
      style = policy_style(p)
      xs, ys = [], []
      tv = None
      if tag and train_cache.get(p):
        tv = _train_scalars(train_cache[p], tag)
        if tv is not None:
          tv *= conv
      if tv is not None:
        xs.append(0); ys.append(tv)
        rows.append({"policy": p, "stage": "training", "value": round(tv, 6),
                     "source": f"{tag} x{conv:g}, mean of last {TRAIN_TAIL} iters",
                     "run": train_cache[p].name if train_cache[p] else "",
                     "caveat": "stochastic (sampled) action" if stoch else ""})
      elif tag and train_cache.get(p):
        # An absent training point is a fact about the RUNNER, not about the policy: the
        # flat A0 baseline trains under rsl_rl's own runner, which logs no `Loss/metrics/*`
        # (those are HierarchicalRunner's). Recorded as absent with the reason, never 0.
        rows.append({"policy": p, "stage": "training", "value": "",
                     "source": f"{tag} not logged by this run's runner",
                     "run": train_cache[p].name, "caveat": "absent, not zero"})
      sv = sim_cache.get(p, {}).get(key)
      if sv is not None and np.isfinite(sv):
        xs.append(1); ys.append(float(sv))
        rows.append({"policy": p, "stage": "sim bench", "value": round(float(sv), 6),
                     "source": f"{CHAIN_ARMS[p.split(' +')[0]][0]}:{CHAIN_ARMS[p.split(' +')[0]][1]}__{SIM_CONDITION}",
                     "run": "", "caveat": "0 kg payload"})
      hv = []
      for b in bundles:
        if b.policy != p or regime not in b.regimes:
          continue
        if key == "jacc_legs":
          hw_jacc.setdefault(b.dir.name, _hw_jacc_legs(b, regime))
          v = hw_jacc[b.dir.name]
        else:
          v = b.regimes[regime].get(key)
        if v is None or not np.isfinite(float(v)):
          continue
        hv.append(float(v))
        ax.plot(2, float(v), marker="o", markersize=2.4, color=style["color"], alpha=0.55,
                zorder=2)
        cav = "encoder-differentiated" if key == "jacc_legs" else ""
        if key == "cot":
          # The CoT denominator is a measured distance, and only the three Runs with motion
          # capture have a true one; the rest integrate the onboard estimate, which
          # under-reads by ~3% on the Runs where both are available. Small, but it is a
          # different instrument per point in one column, so it travels with the number.
          cav = f"distance from {b.regimes[regime].get('cot_distance_reference', 'unknown')}"
        rows.append({"policy": p, "stage": "hardware", "value": round(float(v), 6),
                     "source": f"{b.dir.name} {regime}.json" if key != "jacc_legs"
                               else f"{b.dir.name} flight recorder (meas_q, control rate)",
                     "run": b.tag, "caveat": cav})
      if hv:
        xs.append(2); ys.append(float(np.mean(hv)))
        # The LINE's hardware point is this mean, not any one Run, so it has to be archived
        # too: without it the series a reader's eye follows across the three stages is the
        # one number the panel CSV does not contain. Caught by --verify-csv.
        rows.append({"policy": p, "stage": "hardware (mean)", "value": round(float(np.mean(hv)), 6),
                     "source": f"mean of {len(hv)} {regime} Run(s), the plotted line point",
                     "run": "", "caveat": "n=1, no spread" if len(hv) == 1 else
                                          f"n={len(hv)}, spread {min(hv):.4g}-{max(hv):.4g}"})
      if xs:
        ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markersize=3.0, linewidth=0.9, zorder=3)
    mark = ("$\\ast$" if stoch else "") + ("$\\dagger$" if key == "jacc_legs" else "")
    ax.set_title(f"{ylabel} {mark}", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(stages, rotation=20, ha="right", fontsize=6)
    ax.set_xlim(-0.35, 2.35)
    ax.margins(y=0.2)
    ax.tick_params(labelsize=6)
    write_panel_csv(out_dir, name, key,
                    ["policy", "stage", "value", "source", "run", "caveat"], rows)

  bottom_legend(fig, policy_handles(policies), ncol=len(policies), height=0.36)
  notes = [
    "$\\ast$ training metrics are computed on the SAMPLED action (exploration noise); the "
    "sim and hardware columns are deterministic inference. Read no sim-to-real meaning into "
    "that step on these rows.",
    "$\\dagger$ hardware jacc\\_legs is differentiated encoder velocity at the control rate; "
    "sim uses analytic qacc at the physics rate. TRENDS compare, levels do not.",
    "Hardware A0\\_DR\\_s123 Runs carried a 7.5 kg mounted payload; the sim-bench column is "
    "the 0 kg baseline condition for every policy. That column is loaded vs unloaded.",
    "Hardware column = " + regime + " regime only (regimes are never pooled); points are "
    "Runs, the line joins their mean. Omitted: "
    # The omitted KEYS are metric names, so they carry underscores: the pgf backend does not
    # escape those and the whole figure then fails to compile (it did).
    + "; ".join(f"{tex(k)} ({tex(v)})" for k, v in list(CHAIN_OMITTED.items())[:3]) + ".",
    "cot's hardware denominator is motion-capture distance on the three captured Runs and "
    "the onboard estimate on the rest (it under-reads $\\approx$3\\% where both exist); the "
    "per-point instrument is in the panel CSV's caveat column.",
    "A missing point is absent, never zero: the flat A0\\_DR baseline trains under rsl\\_rl's "
    "own runner, which logs no Loss/metrics/* -- so it has no training point for cot, "
    "act\\_legs\\_rad or jacc\\_legs. mech\\_power\\_w is logged at no training stage at all.",
  ]
  # `wrap=True` measures against the FIGURE, not the tight bbox savefig crops to, so long
  # caveat lines ran off the page. Wrapped explicitly at a width that fits 15 cm at 5 pt.
  import textwrap
  lines = [w for n in notes for w in textwrap.wrap(n, 118)]
  for i, n in enumerate(lines):
    fig.text(0.012, 0.310 - 0.019 * i, n, fontsize=5.0, color="0.3", ha="left", va="top")
  print(f"[PLOT] f_chain: training points read for "
        + ", ".join(f"{p}:{'yes' if train_cache.get(p) else 'no'}" for p in policies))
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
  ap.add_argument("--bundles", nargs="+", default=None,
                  help="one or more globs under --root for Run bundle directories "
                       f"(default: {BUNDLE_GLOB!r}, the 2026-09-14 session alone); pass "
                       "several to pool bundles from more than one hardware session, e.g. "
                       "the B0 seed-band runs")
  ap.add_argument("--out", default=None, help="output directory (default: <root>/figures)")
  ap.add_argument("--figures", default=None,
                  help="comma-separated subset of: " + ",".join(FIGURES))
  ap.add_argument("--cadence-json", default=None,
                  help=f"analyze_cadence_settling.py output (default: {CADENCE_JSON})")
  ap.add_argument("--run", default=None,
                  help="f_track / f_hier: draw this Run (matches the run tag or the bundle "
                       "directory name) instead of the mocap Run with the most walking time")
  ap.add_argument("--min-hold-s", type=float, default=STAND_MIN_SEG_S,
                  help="f_stand: shortest zero-command hold that can show a drift "
                       f"(default {STAND_MIN_SEG_S:g} s)")
  ap.add_argument("--chain-regime", default="walking", choices=["walking", "standing"],
                  help="f_chain: which hardware regime forms the third point (never pooled)")
  ap.add_argument("--max-points-per-line", type=int, default=4000,
                  help="envelope-decimate any time series longer than this (default 4000; "
                       "keeps the .pgf small enough to recompile with the thesis)")
  ap.add_argument("--no-downsample", action="store_true",
                  help="disable the min/max-envelope decimation on the time-series figures")
  ap.add_argument("--check-latex", action="store_true",
                  help="compile every written .pgf standalone with pdflatex")
  ap.add_argument("--verify-csv", action="store_true",
                  help="assert every value drawn in each figure appears in its panel CSVs")
  args = ap.parse_args()

  root = Path(args.root)
  out_dir = Path(args.out) if args.out else root / "figures"
  globs = args.bundles if args.bundles else [BUNDLE_GLOB]
  bundles = load_bundles(root, globs)
  if not bundles:
    raise SystemExit("no run bundles under " + ", ".join(str(root / g) for g in globs))
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
    if n in ("cadence", "settle", "stand") and not cad_rows:
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
    elif n == "track":
      written.append(fig_track(bundles, out_dir, run=args.run, downsample=ds))
    elif n == "stand":
      written.append(fig_stand(bundles, cad_rows, out_dir, min_seg_s=args.min_hold_s,
                               downsample=ds))
    elif n == "hier":
      written.append(fig_hier(bundles, out_dir, run=args.run, downsample=ds))
    elif n == "chain":
      written.append(fig_chain(bundles, out_dir, regime=args.chain_regime, root=root))

  print(f"[PLOT] {len(written)} figure(s) in {out_dir}")
  if args.verify_csv:
    bad = check_panel_csv(out_dir, names)
    if bad:
      print(f"[CSV] {bad} figure(s) have drawn values missing from their panel CSVs")
      return 1
  if args.check_latex:
    bad = check_latex(written)
    if bad:
      print(f"[LATEX] {bad} figure(s) failed to compile")
      return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
