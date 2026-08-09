#!/usr/bin/env python3
"""Plot the CSV logs written by the h1_2 deploy stack. Three schemas, auto-detected
by column presence (same convention as scripts/deploy_gate_analyzer.py):

  <base>.csv          flight-recorder base log (safety_logger.h): per-tick raw/measured
                       joint state, IMU, cmd-vs-achieved velocity, safety triggers.
                       Reads <base>_meta.json for joint names/limits when present.
  <base>_hrl.csv       HRL telemetry (hrl_telemetry.h, A1 only, 50 Hz): HL goal-space
                       state/target, estimator-vs-ground-truth velocity/height, act_rate.
                       s0..s6 / tgt0..tgt6 are ALREADY physical units (vx, vy m/s; wz
                       rad/s; idx 3-5 projected-gravity, unitless; idx 6 height m) — the
                       C++ deploy side decodes the tanh goal before logging.
  all_joints_*.csv     read_all_joints.cpp (sibling repo h1_2_low_level_controller,
                       ~/ramlab_ws/trajectories/): q[i]/dq[i]/tau_est[i] per joint. Only
                       the CURRENT schema is supported — the two 2026-02-19 example files
                       on disk predate it (4 literal columns) and are out of scope.

Usage:
  python scripts/plot_deploy_logs.py <csv> [<csv2> ...] [options]

  # quick glance: every default view for the detected log type
  python scripts/plot_deploy_logs.py logs/deploy_safety/2026-08-07_11-17-20.csv

  # thesis figure: one view, custom labels, vector output
  python scripts/plot_deploy_logs.py logs/deploy_safety/RUN_hrl.csv --views goals \
      --format pdf --title "HL velocity goal tracking" --ylabel "m/s"

  # compare two runs (same log type) on shared axes
  python scripts/plot_deploy_logs.py logs/deploy_safety/A0_RUN.csv logs/deploy_safety/A1_RUN.csv \
      --run-labels A0,A1 --views velocity

  # correlate HL goal behaviour with the LL/safety trace from the same session
  python scripts/plot_deploy_logs.py logs/deploy_safety/RUN.csv --combined

Multi-run overlay simplifies each view to its single comparison-relevant series per run
(e.g. measured joint position, not raw+measured) so N runs don't multiply into 2N lines
per subplot. Sessions can run to ~440k rows; series above a row-count threshold are
min/max-envelope downsampled per line before plotting (spikes/trigger edges survive).

Writes PNG (default; --format pdf/svg for LaTeX-ready vector output) to <base>_plots/
next to the first input CSV, unless --out overrides it.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Hardware-joint-ID order, matching deploy/robots/h1_2/include/h1_2_limits.h
# (h1_2_joint_names) and read_all_joints.cpp's q[i]/dq[i]/tau_est[i] indexing. Duplicated
# here rather than parsed from the header, same precedent as scripts/measure_joint_trip_thresholds.py.
JOINT_NAMES = [
  "left_hip_yaw", "left_hip_pitch", "left_hip_roll", "left_knee",
  "left_ankle_pitch", "left_ankle_roll",
  "right_hip_yaw", "right_hip_pitch", "right_hip_roll", "right_knee",
  "right_ankle_pitch", "right_ankle_roll", "waist_yaw",
  "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
  "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
  "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
  "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
JOINT_GROUPS = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")

# deploy.yaml goal_components: [velocity(3), orientation(3), height(1)] -- fixed index
# meaning for the A1a 7-dim goal space (see goal_space.py GOAL_COMPONENT_DIMS).
GOAL_LABELS = ["vx", "vy", "wz", "grav_x", "grav_y", "grav_z", "height"]
GOAL_UNITS = ["m/s", "m/s", "rad/s", "", "", "", "m"]
GOAL_VELOCITY_DIMS = (0, 1, 2)

DOWNSAMPLE_THRESHOLD = 20_000
DOWNSAMPLE_BUCKETS = 4_000  # -> up to ~4*BUCKETS points/line (first,min,max,last)


def joint_group(name: str) -> str:
  if name.startswith(("left_hip", "left_knee", "left_ankle")):
    return "left_leg"
  if name.startswith(("right_hip", "right_knee", "right_ankle")):
    return "right_leg"
  if name.startswith("waist"):
    return "waist"
  if name.startswith("left_"):
    return "left_arm"
  if name.startswith("right_"):
    return "right_arm"
  return "other"


# ---------------------------------------------------------------- schema detection ----

def detect_kind(header: list[str]) -> str:
  if "s0" in header and "tgt0" in header and "act_rate" in header:
    return "flight_hrl"
  if "raw_q0" in header:
    return "flight_base"
  if "q[0]" in header:
    return "joint_traj"
  raise SystemExit(
    f"unrecognized CSV schema (header starts with {header[:6]}...) -- expected a "
    "flight-recorder base/_hrl.csv or a current-schema read_all_joints.csv")


def read_header(path: Path) -> list[str]:
  with open(path) as f:
    return next(csv.reader(f))


def resolve_base(csv_path: Path) -> str:
  s = str(csv_path)
  if s.endswith("_hrl.csv"):
    return s[: -len("_hrl.csv")]
  if s.endswith(".csv"):
    return s[: -len(".csv")]
  return s


TIME_COL = {"flight_base": "t", "flight_hrl": "t", "joint_traj": "time"}


# ---------------------------------------------------------------------- data loading ----

@dataclass
class Run:
  path: Path
  kind: str
  label: str
  df: pd.DataFrame
  meta: dict | None  # <base>_meta.json, flight_base only


def load_run(path: Path, kind: str) -> Run:
  df = pd.read_csv(path)
  meta = None
  if kind == "flight_base":
    meta_path = Path(f"{resolve_base(path)}_meta.json")
    if meta_path.exists():
      meta = json.loads(meta_path.read_text())
    else:
      print(f"[PLOT] WARNING: {meta_path} not found -- joint names/limits fall back to "
            "hardware-ID order, per-joint min/max reference lines are skipped")
  stem = path.stem[:-4] if path.stem.endswith("_hrl") else path.stem
  return Run(path=path, kind=kind, label=stem, df=df, meta=meta)


def joint_index_name_pairs(run: Run) -> list[tuple[int, str]]:
  """(CSV column index, joint name) pairs in a stable, name-groupable order."""
  if run.kind == "joint_traj" or run.meta is None:
    return list(enumerate(JOINT_NAMES))
  return [(j["slot"], j["name"]) for j in sorted(run.meta["joints"], key=lambda j: j["slot"])]


# -------------------------------------------------------------------- downsampling ----

def envelope_downsample(t: np.ndarray, y: np.ndarray, enabled: bool = True) -> tuple[np.ndarray, np.ndarray]:
  """Per-line min/max-envelope decimation: keeps first/min/max/last sample of each of
  DOWNSAMPLE_BUCKETS buckets, so spikes (trigger edges, trips) survive. No-op below
  DOWNSAMPLE_THRESHOLD rows or when disabled."""
  n = len(t)
  if not enabled or n <= DOWNSAMPLE_THRESHOLD:
    return t, y
  edges = np.unique(np.linspace(0, n, DOWNSAMPLE_BUCKETS + 1).astype(int))
  idxs = []
  for a, b in zip(edges[:-1], edges[1:]):
    if b <= a:
      continue
    seg = y[a:b]
    idxs.extend({a, a + int(np.argmin(seg)), a + int(np.argmax(seg)), b - 1})
  idxs = np.array(sorted(set(idxs)))
  return t[idxs], y[idxs]


def event_start_times(t: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """First timestamp of each contiguous True run in mask (collapses a held trigger to
  one marker instead of one per tick)."""
  mask = mask.astype(bool)
  starts = mask & ~np.concatenate(([False], mask[:-1]))
  return t[starts]


# -------------------------------------------------------------------------- labels ----

@dataclass
class LabelOverrides:
  title: str | None = None
  xlabel: str | None = None
  ylabel: str | None = None


def parse_joint_filter(spec: str | None):
  if not spec:
    return None
  tokens = {t.strip() for t in spec.split(",") if t.strip()}

  def pred(name: str) -> bool:
    return name in tokens or joint_group(name) in tokens

  return pred


# --------------------------------------------------------------------------- saving ----

def save_fig(fig, out_dir: Path, name: str, fmt: str) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"{name}.{fmt}"
  fig.tight_layout(rect=(0, 0, 1, 0.96))  # leave room for suptitle; avoids row-to-row overlap
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"[PLOT] wrote {path}")


def legend_if_any(ax, **kw) -> None:
  """ax.legend() warns and draws an empty box when nothing was plotted on it (e.g. a
  column that's absent on an older-schema CSV) -- skip it instead."""
  if ax.get_legend_handles_labels()[0]:
    ax.legend(**kw)


def grid_axes(n: int, ncols: int = 3):
  nrows = -(-n // ncols)  # ceil
  fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows), squeeze=False)
  flat = axes.flat
  used, spare = list(flat[:n]), list(flat[n:])
  for ax in spare:
    ax.set_visible(False)
  return fig, used


# ============================================================ flight_base : views ====

def view_joints(runs, run_labels, out_dir, fmt, labels, joint_filter, downsample, **_):
  """Per-body-region joint tracking grid. Single run: raw vs measured position, one
  subplot/joint. Multi-run overlay: measured position only, one line per run."""
  overlay = len(runs) > 1
  pairs = joint_index_name_pairs(runs[0])
  if joint_filter:
    pairs = [(i, n) for i, n in pairs if joint_filter(n)]
  limits = {j["slot"]: j for j in runs[0].meta["joints"]} if runs[0].meta else {}

  for group in JOINT_GROUPS:
    cols = [(i, n) for i, n in pairs if joint_group(n) == group]
    if not cols:
      continue
    fig, axes = grid_axes(len(cols))
    for ax, (idx, name) in zip(axes, cols):
      for run, rlabel in zip(runs, run_labels):
        t = run.df[TIME_COL[run.kind]].to_numpy()
        if overlay:
          col = f"meas_q{idx}"
          if col not in run.df:
            continue
          tt, yy = envelope_downsample(t, run.df[col].to_numpy(), downsample)
          ax.plot(tt, yy, label=rlabel, linewidth=1)
        else:
          for series_label, col, style in (
            ("raw", f"raw_q{idx}", dict(alpha=0.55, linewidth=1)),
            ("measured", f"meas_q{idx}", dict(linewidth=1)),
          ):
            if col not in run.df:
              continue
            tt, yy = envelope_downsample(t, run.df[col].to_numpy(), downsample)
            ax.plot(tt, yy, label=series_label, **style)
      lim = limits.get(idx)
      if lim:
        ax.axhline(lim["min"], color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(lim["max"], color="gray", linestyle=":", linewidth=0.8)
      ax.set_title(name, fontsize=9)
      ax.set_xlabel(labels.xlabel or "t (s)", fontsize=8)
      ax.set_ylabel(labels.ylabel or "rad", fontsize=8)
      ax.tick_params(labelsize=7)
      legend_if_any(ax, fontsize=6)
    fig.suptitle(labels.title or f"Joint tracking -- {group}")
    save_fig(fig, out_dir, f"joints_{group}", fmt)


def view_velocity(runs, run_labels, out_dir, fmt, labels, downsample, **_):
  """cmd vs achieved base-frame velocity (vx, vy) + commanded yaw rate (no achieved-yaw
  column exists in this schema)."""
  fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
  for run, rlabel in zip(runs, run_labels):
    df = run.df
    t = df["t"].to_numpy()
    sim_only = "ach_vx" in df and bool(np.all(df["ach_vx"].to_numpy() == 0))
    if sim_only:
      print(f"[PLOT] NOTE: {run.path.name}: ach_v* is all-zero (real-hardware CSV, or no "
            "SportModeState publisher) -- plotting cmd_* only")
    for ax, axis, cmd_col, ach_col in (
      (axes[0], "vx", "cmd_vx", "ach_vx"), (axes[1], "vy", "cmd_vy", "ach_vy")):
      if cmd_col in df:
        tt, yy = envelope_downsample(t, df[cmd_col].to_numpy(), downsample)
        ax.plot(tt, yy, "--", label=f"{rlabel} cmd", linewidth=1)
      if ach_col in df and not sim_only:
        tt, yy = envelope_downsample(t, df[ach_col].to_numpy(), downsample)
        ax.plot(tt, yy, label=f"{rlabel} achieved", linewidth=1)
    if "cmd_wz" in df:
      tt, yy = envelope_downsample(t, df["cmd_wz"].to_numpy(), downsample)
      axes[2].plot(tt, yy, "--", label=f"{rlabel} cmd", linewidth=1)
  for ax, ylabel, title in zip(axes, ("m/s", "m/s", "rad/s"),
                               ("vx", "vy", "wz (commanded only)")):
    ax.set_ylabel(labels.ylabel or ylabel)
    ax.set_title(title, fontsize=9)
    legend_if_any(ax, fontsize=7)
  axes[-1].set_xlabel(labels.xlabel or "t (s)")
  fig.suptitle(labels.title or "Velocity tracking")
  save_fig(fig, out_dir, "velocity", fmt)


def view_imu(runs, run_labels, out_dir, fmt, labels, downsample, **_):
  fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
  for run, rlabel in zip(runs, run_labels):
    df, t = run.df, run.df["t"].to_numpy()
    for comp in ("quat_w", "quat_x", "quat_y", "quat_z"):
      if comp in df:
        tt, yy = envelope_downsample(t, df[comp].to_numpy(), downsample)
        axes[0].plot(tt, yy, label=f"{rlabel} {comp}", linewidth=1)
    for comp in ("acc_x", "acc_y", "acc_z"):
      if comp in df:
        tt, yy = envelope_downsample(t, df[comp].to_numpy(), downsample)
        axes[1].plot(tt, yy, label=f"{rlabel} {comp}", linewidth=1)
  axes[0].set_ylabel(labels.ylabel or "quaternion"); axes[0].set_title("orientation", fontsize=9)
  axes[1].set_ylabel(labels.ylabel or "m/s^2"); axes[1].set_title("acceleration", fontsize=9)
  axes[1].set_xlabel(labels.xlabel or "t (s)")
  for ax in axes:
    legend_if_any(ax, fontsize=6)
  fig.suptitle(labels.title or "IMU")
  save_fig(fig, out_dir, "imu", fmt)


def view_safety(runs, run_labels, out_dir, fmt, labels, downsample, **_):
  fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
  flag_cols = ["trig_joint", "trig_tilt", "trig_fall", "trig_torque", "trig_dq"]
  for run, rlabel in zip(runs, run_labels):
    df, t = run.df, run.df["t"].to_numpy()
    if "alpha" in df:
      tt, yy = envelope_downsample(t, df["alpha"].to_numpy(), downsample)
      axes[0].plot(tt, yy, label=f"{rlabel} alpha", linewidth=1)
    if "alpha_trip" in df:
      tt, yy = envelope_downsample(t, df["alpha_trip"].to_numpy(), downsample)
      axes[0].plot(tt, yy, label=f"{rlabel} alpha_trip", linewidth=1, linestyle="--")

    present_flags = [c for c in flag_cols if c in df]
    for i, col in enumerate(present_flags):
      mask = df[col].to_numpy().astype(bool)
      starts = event_start_times(t, mask)
      y = np.full_like(starts, i, dtype=float)
      axes[1].scatter(starts, y, s=8, label=f"{rlabel} {col}" if run is runs[0] else None,
                       marker="|")
    if present_flags:
      axes[1].set_yticks(range(len(present_flags)))
      axes[1].set_yticklabels(present_flags, fontsize=7)

    if "trip_ratio" in df:
      tt, yy = envelope_downsample(t, df["trip_ratio"].to_numpy(), downsample)
      axes[2].plot(tt, yy, label=f"{rlabel} trip_ratio", linewidth=1)
  if any("trip_ratio" in r.df for r in runs):
    axes[2].axhline(1.0, color="red", linestyle=":", linewidth=0.8, label="trip threshold")
  axes[0].set_ylabel(labels.ylabel or "hold ramp [0,1]"); axes[0].set_title("alpha / alpha_trip", fontsize=9)
  axes[1].set_ylabel("trigger"); axes[1].set_title("trigger timeline (event onsets)", fontsize=9)
  axes[2].set_ylabel(labels.ylabel or "ratio"); axes[2].set_title("trip_ratio (headroom to threshold)", fontsize=9)
  axes[2].set_xlabel(labels.xlabel or "t (s)")
  for ax in axes:
    legend_if_any(ax, fontsize=6)
  fig.suptitle(labels.title or "Safety triggers")
  save_fig(fig, out_dir, "safety", fmt)


FLIGHT_BASE_VIEWS = {"joints": view_joints, "velocity": view_velocity,
                     "imu": view_imu, "safety": view_safety}


# ============================================================= flight_hrl : views ====

def view_goals(runs, run_labels, out_dir, fmt, labels, all_goal_dims, downsample, **_):
  dims = range(7) if all_goal_dims else GOAL_VELOCITY_DIMS
  fig, axes = grid_axes(len(dims), ncols=1 if len(dims) <= 3 else 3)
  for ax, i in zip(axes, dims):
    for run, rlabel in zip(runs, run_labels):
      df, t = run.df, run.df["t"].to_numpy()
      for col, style, tag in ((f"s{i}", {}, "s"), (f"tgt{i}", dict(linestyle="--"), "tgt")):
        if col in df:
          tt, yy = envelope_downsample(t, df[col].to_numpy(), downsample)
          ax.plot(tt, yy, label=f"{rlabel} {tag}", linewidth=1, **style)
    unit = f" ({GOAL_UNITS[i]})" if GOAL_UNITS[i] else ""
    ax.set_title(f"{GOAL_LABELS[i]}{unit}", fontsize=9)
    ax.set_xlabel(labels.xlabel or "t (s)", fontsize=8)
    ax.set_ylabel(labels.ylabel or GOAL_UNITS[i] or "", fontsize=8)
    legend_if_any(ax, fontsize=6)
  fig.suptitle(labels.title or "HL goal-space trace")
  save_fig(fig, out_dir, "goals", fmt)


def view_estimator(runs, run_labels, out_dir, fmt, labels, downsample, **_):
  fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
  for run, rlabel in zip(runs, run_labels):
    df, t = run.df, run.df["t"].to_numpy()
    if "gt_vx" in df and bool(np.all(df["gt_vx"].to_numpy() == 0)) and "gt_vy" in df and bool(np.all(df["gt_vy"].to_numpy() == 0)):
      print(f"[PLOT] NOTE: {run.path.name}: gt_* is all-zero (real hardware -- only a "
            "bridge run scores the estimator against ground truth)")
    for ax, est_col, gt_col, title in (
      (axes[0], "est_vx", "gt_vx", "vx (within-window increment)"),
      (axes[1], "est_vy", "gt_vy", "vy (within-window increment)"),
      (axes[2], "est_h", "gt_h", "height (absolute)"),
    ):
      if est_col in df:
        tt, yy = envelope_downsample(t, df[est_col].to_numpy(), downsample)
        ax.plot(tt, yy, label=f"{rlabel} est", linewidth=1)
      if gt_col in df:
        tt, yy = envelope_downsample(t, df[gt_col].to_numpy(), downsample)
        ax.plot(tt, yy, "--", label=f"{rlabel} gt", linewidth=1)
      ax.set_title(title, fontsize=9)
      legend_if_any(ax, fontsize=6)
  axes[0].set_ylabel(labels.ylabel or "m/s"); axes[1].set_ylabel(labels.ylabel or "m/s")
  axes[2].set_ylabel(labels.ylabel or "m"); axes[2].set_xlabel(labels.xlabel or "t (s)")
  fig.suptitle(labels.title or "Estimator vs ground truth")
  save_fig(fig, out_dir, "estimator", fmt)


def view_command(runs, run_labels, out_dir, fmt, labels, downsample, **_):
  """cmd vs leg-odometry absolute velocity estimate (lo_*), + sim ground truth (gt_a*)
  when available. lo_* is window-latched (piecewise-constant); plotted as-is."""
  fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
  for run, rlabel in zip(runs, run_labels):
    df, t = run.df, run.df["t"].to_numpy()
    for ax, cmd_col, lo_col, gt_col in (
      (axes[0], "cmd_vx", "lo_vx", "gt_avx"), (axes[1], "cmd_vy", "lo_vy", "gt_avy")):
      if cmd_col in df:
        tt, yy = envelope_downsample(t, df[cmd_col].to_numpy(), downsample)
        ax.plot(tt, yy, "--", label=f"{rlabel} cmd", linewidth=1)
      if lo_col in df:
        tt, yy = envelope_downsample(t, df[lo_col].to_numpy(), downsample)
        ax.plot(tt, yy, label=f"{rlabel} leg-odom", linewidth=1)
      if gt_col in df and not bool(np.all(df[gt_col].to_numpy() == 0)):
        tt, yy = envelope_downsample(t, df[gt_col].to_numpy(), downsample)
        ax.plot(tt, yy, ":", label=f"{rlabel} gt", linewidth=1)
  axes[0].set_ylabel(labels.ylabel or "m/s"); axes[0].set_title("vx", fontsize=9)
  axes[1].set_ylabel(labels.ylabel or "m/s"); axes[1].set_title("vy", fontsize=9)
  axes[1].set_xlabel(labels.xlabel or "t (s)")
  for ax in axes:
    legend_if_any(ax, fontsize=6)
  fig.suptitle(labels.title or "Command tracking (leg odometry)")
  save_fig(fig, out_dir, "command", fmt)


def view_smoothness(runs, run_labels, out_dir, fmt, labels, downsample, **_):
  fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
  for run, rlabel in zip(runs, run_labels):
    df, t = run.df, run.df["t"].to_numpy()
    if "act_rate" in df:
      tt, yy = envelope_downsample(t, df["act_rate"].to_numpy(), downsample)
      axes[0].plot(tt, yy, label=rlabel, linewidth=1)
    if "period" in df:
      tt, yy = envelope_downsample(t, df["period"].to_numpy(), downsample)
      axes[1].plot(tt, yy, label=rlabel, linewidth=1)
    if "hip_pitch_l" in df:
      tt, yy = envelope_downsample(t, df["hip_pitch_l"].to_numpy(), downsample)
      axes[2].plot(tt, yy, label=f"{rlabel} left", linewidth=1)
    if "hip_pitch_r" in df:
      tt, yy = envelope_downsample(t, df["hip_pitch_r"].to_numpy(), downsample)
      axes[2].plot(tt, yy, label=f"{rlabel} right", linewidth=1)
  axes[0].set_ylabel(labels.ylabel or "act_rate"); axes[0].set_title("action rate", fontsize=9)
  axes[1].set_ylabel(labels.ylabel or "s"); axes[1].set_title("commanded stride period", fontsize=9)
  axes[2].set_ylabel(labels.ylabel or "rad"); axes[2].set_title("hip pitch symmetry", fontsize=9)
  axes[2].set_xlabel(labels.xlabel or "t (s)")
  for ax in axes:
    legend_if_any(ax, fontsize=6)
  fig.suptitle(labels.title or "Smoothness / timing")
  save_fig(fig, out_dir, "smoothness", fmt)


FLIGHT_HRL_VIEWS = {"goals": view_goals, "estimator": view_estimator,
                    "command": view_command, "smoothness": view_smoothness}


# ============================================================= joint_traj : views ====

def view_joint_traj(runs, run_labels, out_dir, fmt, labels, joint_filter, downsample, **_):
  """q / dq / tau_est grid, one row per joint (grouped by body region), q|dq|tau_est
  columns -- kept separate because the three quantities don't share a sensible y-scale."""
  overlay = len(runs) > 1
  names = [(i, n) for i, n in enumerate(JOINT_NAMES)]
  if joint_filter:
    names = [(i, n) for i, n in names if joint_filter(n)]

  for group in JOINT_GROUPS:
    cols = [(i, n) for i, n in names if joint_group(n) == group]
    if not cols:
      continue
    fig, axes = plt.subplots(len(cols), 3, figsize=(12, 2.2 * len(cols)), squeeze=False)
    for row, (idx, name) in enumerate(cols):
      for col_i, (quantity, unit) in enumerate((("q", "rad"), ("dq", "rad/s"), ("tau_est", "Nm"))):
        ax = axes[row][col_i]
        colname = f"{quantity}[{idx}]"
        for run, rlabel in zip(runs, run_labels):
          if colname not in run.df:
            continue
          t = run.df["time"].to_numpy()
          tt, yy = envelope_downsample(t, run.df[colname].to_numpy(), downsample)
          ax.plot(tt, yy, label=rlabel if overlay else None, linewidth=1)
        if col_i == 0:
          ax.set_ylabel(name, fontsize=8)
        if row == 0:
          ax.set_title(f"{quantity} ({unit})", fontsize=9)
        if row == len(cols) - 1:
          ax.set_xlabel(labels.xlabel or "t (s)", fontsize=8)
        ax.tick_params(labelsize=6)
        if overlay:
          legend_if_any(ax, fontsize=6)
    fig.suptitle(labels.title or f"Joint trajectory -- {group}")
    save_fig(fig, out_dir, f"joints_{group}", fmt)


JOINT_TRAJ_VIEWS = {"joints": view_joint_traj}

DEFAULT_VIEWS = {
  "flight_base": list(FLIGHT_BASE_VIEWS),
  "flight_hrl": list(FLIGHT_HRL_VIEWS),
  "joint_traj": list(JOINT_TRAJ_VIEWS),
}
VIEW_FUNCS = {"flight_base": FLIGHT_BASE_VIEWS, "flight_hrl": FLIGHT_HRL_VIEWS,
              "joint_traj": JOINT_TRAJ_VIEWS}


# =============================================================== combined view ====

def plot_combined(run: Run, out_dir: Path, fmt: str, labels: LabelOverrides, downsample: bool) -> None:
  hrl_path = Path(f"{resolve_base(run.path)}_hrl.csv")
  if not hrl_path.exists():
    raise SystemExit(f"--combined requires a matching {hrl_path} (not found)")
  hrl_df = pd.read_csv(hrl_path)
  df, t = run.df, run.df["t"].to_numpy()

  sim_only = "ach_vx" in df and bool(np.all(df["ach_vx"].to_numpy() == 0))
  if sim_only:
    print(f"[PLOT] NOTE: {run.path.name}: ach_v* is all-zero (real-hardware CSV, or no "
          "SportModeState publisher) -- plotting cmd_* only")
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
  for col, style, tag in (("cmd_vx", {}, "cmd vx"), ("ach_vx", {}, "ach vx"),
                          ("cmd_vy", dict(linestyle="--"), "cmd vy"),
                          ("ach_vy", dict(linestyle="--"), "ach vy")):
    if col in df and not (sim_only and col.startswith("ach_")):
      tt, yy = envelope_downsample(t, df[col].to_numpy(), downsample)
      ax1.plot(tt, yy, label=tag, linewidth=1, **style)
  ax1.set_ylabel(labels.ylabel or "m/s")
  ax1.set_title("velocity tracking (base log)", fontsize=9)
  legend_if_any(ax1, fontsize=7)

  th = hrl_df["t"].to_numpy()
  for i in GOAL_VELOCITY_DIMS:
    for col, style, tag in ((f"s{i}", {}, f"{GOAL_LABELS[i]} (s)"),
                            (f"tgt{i}", dict(linestyle="--"), f"{GOAL_LABELS[i]} (tgt)")):
      if col in hrl_df:
        tt, yy = envelope_downsample(th, hrl_df[col].to_numpy(), downsample)
        ax2.plot(tt, yy, label=tag, linewidth=1, **style)
  ax2.set_ylabel(labels.ylabel or "m/s, rad/s")
  ax2.set_xlabel(labels.xlabel or "t (s)")
  ax2.set_title("HL goal-space trace (_hrl.csv)", fontsize=9)
  legend_if_any(ax2, fontsize=7)

  for trig_col, color in (("trig_fall", "red"), ("trig_tilt", "orange")):
    if trig_col in df:
      for tv in event_start_times(t, df[trig_col].to_numpy().astype(bool)):
        ax1.axvline(tv, color=color, alpha=0.4, linewidth=1)
        ax2.axvline(tv, color=color, alpha=0.4, linewidth=1)

  fig.suptitle(labels.title or f"Combined session view -- {run.label}")
  save_fig(fig, out_dir, "combined", fmt)


# ===================================================================== main ====

def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("csvs", nargs="+",
                   help="one or more CSV logs of the SAME detected type (2+ overlays runs)")
  ap.add_argument("--views", help="comma-separated view names (default: all for the "
                                   "detected type)")
  ap.add_argument("--joints", help="comma-separated joint names or group names "
                                    f"({','.join(JOINT_GROUPS)}); default: all "
                                    "(joints/joint-trajectory views only)")
  ap.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
  ap.add_argument("--out", help="output directory (default: <base>_plots/ next to the "
                                 "first input CSV)")
  ap.add_argument("--title"); ap.add_argument("--xlabel"); ap.add_argument("--ylabel")
  ap.add_argument("--all-goal-dims", action="store_true",
                   help="HRL goals view: also show orientation/height (idx 3-6), pinned "
                        "to nominal under the hl_velocity_goals_only A1a default")
  ap.add_argument("--combined", action="store_true",
                   help="flight_base only, single CSV: stack velocity tracking + the "
                        "matching _hrl.csv's goal-space trace on a shared time axis")
  ap.add_argument("--run-labels", help="comma-separated legend labels, one per input CSV "
                                        "(default: filename stem)")
  ap.add_argument("--no-downsample", action="store_true",
                   help="disable the min/max-envelope decimation applied above "
                        f"{DOWNSAMPLE_THRESHOLD} rows/line")
  args = ap.parse_args()

  paths = [Path(c) for c in args.csvs]
  for p in paths:
    if not p.exists():
      raise SystemExit(f"not found: {p}")

  kinds = [detect_kind(read_header(p)) for p in paths]
  if len(set(kinds)) > 1:
    raise SystemExit(f"mixed log types across inputs: {list(zip(paths, kinds))}")
  kind = kinds[0]

  if args.combined:
    if len(paths) != 1:
      raise SystemExit("--combined supports exactly one base CSV, not multi-run overlay")
    if kind != "flight_base":
      raise SystemExit("--combined requires a flight-recorder base .csv (it locates the "
                        "matching _hrl.csv itself)")

  run_labels = args.run_labels.split(",") if args.run_labels else None
  runs = [load_run(p, kind) for p in paths]
  if run_labels is None:
    run_labels = [r.label for r in runs]
  elif len(run_labels) != len(paths):
    raise SystemExit("--run-labels count must match the number of input CSVs")

  out_dir = Path(args.out) if args.out else Path(f"{resolve_base(paths[0])}_plots")
  label_overrides = LabelOverrides(title=args.title, xlabel=args.xlabel, ylabel=args.ylabel)
  joint_filter = parse_joint_filter(args.joints)
  downsample = not args.no_downsample

  if args.combined:
    plot_combined(runs[0], out_dir, args.format, label_overrides, downsample)
    return

  views = args.views.split(",") if args.views else DEFAULT_VIEWS[kind]
  for v in views:
    fn = VIEW_FUNCS[kind].get(v)
    if fn is None:
      raise SystemExit(f"unknown view '{v}' for {kind}; choices: {list(VIEW_FUNCS[kind])}")
    fn(runs, run_labels, out_dir, args.format, label_overrides,
       joint_filter=joint_filter, all_goal_dims=args.all_goal_dims, downsample=downsample)

  print(f"[PLOT] {len(views)} view(s) written to {out_dir}")


if __name__ == "__main__":
  main()
