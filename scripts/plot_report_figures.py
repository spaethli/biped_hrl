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
  f_reward_total, f_reward_return, f_reward_terms_{walking,standing}
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
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plot_thesis_figures as ptf  # noqa: E402  (sets the pgf backend + the thesis rcParams)
import bench_flight_recorder as bfr  # noqa: E402

plt = ptf.plt
from matplotlib.lines import Line2D  # noqa: E402

# Report style (SCIFIG rubric, owner's calls 2026-09-29): sans-serif for readability, nothing below
# 8 pt, no grid lines or top/right box. `\sffamily` is what the pgf backend emits for sans-serif, so
# the including document's sans font is used (report_figures.tex loads Helvetica).
SANS = r"\usepackage[scaled=0.92]{helvet}"
plt.rcParams.update({
  "font.family": "sans-serif",
  "pgf.preamble": plt.rcParams["pgf.preamble"] + "\n" + SANS,
  "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
  "xtick.labelsize": 8, "ytick.labelsize": 8,
  "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})
NAMES = {"A0": "Flat", "A1a": "Hierarchical", "A1a_DR": "Hierarchical + DR"}
ptf.DISPLAY_NAME.update(NAMES)
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
  "act_legs_rad": "leg action rate (rad)",
  "jacc_legs": "leg joint accel. (rad/s\\textsuperscript{2})",
  "err_yaw": "yaw-rate error (rad/s)",
  "orient_dev": "torso tilt (-)",
  "omega_xy": "roll/pitch rate (rad/s)",
  "ub_pose_dev": "upper-body pose dev. (rad\\textsuperscript{2})",
  "ub_arm_vel": "upper-body joint speed (rad/s)",
  "err_vx": "sagittal velocity error (m/s)",
  "err_vy": "lateral velocity error (m/s)",
}
# The schedule-replay chain (2026-10-02): bench and bridge run ONE 253 s command schedule and go
# through ONE scorer (ptf.chain_seq_units). The simulation figures add the ground-truth velocity
# errors, which only the two simulated stages can measure; standing has no energy metrics.
CHAIN_TAGS = ("A0_s42", "A0_s123", "A1a_s42", "A1a_s123", "A1a_DR_s42", "A1a_DR_s123")
CHAIN_KEYS = [k for k, _ in ptf.CHAIN_METRICS]
SIM_KEYS = {"walking": CHAIN_KEYS + ["err_vx", "err_vy"],
            "standing": [k for k in CHAIN_KEYS if k not in ("cot", "mech_power_w")] + ["err_vx", "err_vy"]}
UPPER_VARIANTS = ("A0", "A1a", "A1a_DR")

REWARD_SEQ = "data/2026-10-02-chain-seq/env_reward_seq.json"   # score_env_reward.py: robot + schedule-replay bridge Runs
REWARD_TAGS = ("A0_s42", "A1a_s42", "A1a_DR_s42")
STAGE_KEYS = ("bench", "bridge", "hardware")
STAGE_LABEL = {"bench": "training bench", "bridge": "bridge sim", "hardware": "robot"}
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
      if s == "bench" and len(stages[s]) > 1:
        lo, hi = np.percentile(stages[s], [5, 95])
        ax.plot([lo, hi], [y, y], marker="|", markersize=4, linewidth=0.9, color=colour, zorder=2)
      out += [{"row": label, "stage": s, "value": round(float(v), 6),
               "n": len(stages[s]), "stage_mean": round(m, 6),
               f"loss_pct_bench_to_{vs}": "" if loss is None else round(loss, 2)} for v in stages[s]]
  ax.set_yticks(range(n))
  if show_labels:   # on shared y axes the labels are set ONCE: an empty list on a sibling wipes them all
    ax.set_yticklabels([ptf.tex(r[0]) for r in reversed(rows)])
  ax.tick_params(axis="y", length=0, labelleft=show_labels)
  from matplotlib.ticker import MaxNLocator
  ax.xaxis.set_major_locator(MaxNLocator(nticks))
  ax.set_ylim(-0.7, n - 0.3)
  ax.set_xlabel(xlabel)
  ax.margins(x=0.08)
  return out


def stage_handles(stages=STAGE_KEYS, runs=True, loss=True, vs="bridge"):
  h = [Line2D([], [], marker=STAGE_MARKER[s][0], markersize=5, linestyle="", color="0.3",
              markerfacecolor="0.3" if STAGE_MARKER[s][1] else "white", label=STAGE_LABEL[s])
       for s in stages]
  if runs:
    if "bench" in stages:
      h.append(Line2D([], [], marker="|", markersize=4, linewidth=0.9, color="0.3",
                      label="bench: 5th to 95th\npercentile of environments"))
  if loss and "bench" in stages and vs in stages:
    h.append(Line2D([], [], linestyle="", label="grey numbers: loss from bench\nto " + ("robot" if vs == "hardware" else "bridge sim") + ", % (+ = worse)"))
  return h


def grey_loss_note(fig):
  """Colour the loss entry of the figure's legend like the numbers it explains."""
  for leg in fig.legends + [a.get_legend() for a in fig.axes if a.get_legend()]:
    for t in leg.get_texts():
      if t.get_text().startswith("grey numbers"):
        t.set_color(LOSS_GREY)


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
        d = pd.read_csv(Path(tmp) / f"chain_{k}.csv")
        d = d[(d["stage"] == "hardware") & d["value"].notna()]
        hw[k] = {t: d[d.policy == t]["value"].astype(float).tolist() for t in CHAIN_TAGS}
  units, counts = {}, {}
  for t in CHAIN_TAGS:
    for st in ("bench", "bridge"):
      units[t, st], n, fell = ptf.chain_seq_units(root, t, st, regime)
      counts[t, st] = (n, fell)

  def rows(key, stages):
    return [(name(t, seed=True), ptf.policy_color(t),
             {st: hw.get(key, {}).get(t, []) if st == "hardware" else
                  [float(u[key]) for u in units[t, st] if u.get(key) is not None and np.isfinite(u[key])]
              for st in stages}) for t in CHAIN_TAGS]
  return rows, counts


def fig_chain_seq(rows, keys, stages, out, fname, ncol=3):
  vs = "hardware" if "hardware" in stages else "bridge"   # the stage the grey loss column compares with the bench
  nrow = -(-len(keys) // ncol)
  fig, axes = plt.subplots(nrow, ncol, figsize=(ptf.FIG_W, 2.05 * nrow), sharey=True)
  axes = list(np.atleast_1d(axes).flat)
  for i, (ax, key) in enumerate(zip(axes, keys)):
    write_rows(out, f"{fname}_{key}",
               dumbbell(ax, rows(key, stages), CHAIN_LABEL[key], show_labels=(i % ncol == 0),
                        nticks=3, better="lower", none_note="fell", vs=vs))
  spare = axes[len(keys):]
  for ax in spare:
    ax.set_axis_off()
  fig.subplots_adjust(hspace=0.55, wspace=0.5)   # room for the grey loss column
  if spare:     # the empty grid cell carries the legend instead of a band under the figure
    # anchored right of the cell's left edge: the previous panel's grey numbers hang into the cell's left margin
    spare[0].legend(handles=stage_handles(stages, vs=vs), loc="center left", bbox_to_anchor=(0.28, 0.5),
                    frameon=False, borderaxespad=0)
  else:
    ptf.bottom_legend(fig, stage_handles(stages, vs=vs), ncol=2, height=0.11)
  grey_loss_note(fig)
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


def stage_values(pts, stage, regime, terms):
  """The `terms` of every unit of one stage that has them in this regime."""
  out = []
  for r in pts[stage]:
    d = r.get(regime)
    if d is None or any(d.get(t) is None for t in terms):
      continue                                     # a unit without that regime's data is absent, not zero
    out.append({t: d[t] for t in terms})
  return out


def reward_rows(points, regime, key, terms, stages):
  return [(name(t), ptf.policy_color(t),
           {s: [key(v) for v in stage_values(p, s, regime, terms)] for s in stages})
          for t, p in points.items()]


def full17(pts, regime):
  """All 17 terms: live at the bench; 7 log terms + the 10 plant-replay terms at the bridge."""
  return {"bench": [sum(u[regime].values()) for u in pts["bench"] if regime in u],
          "bridge": [sum(b[regime][t] for t in TERMS_IN[regime]) + sum(b["replay"][regime].values())
                     for b in pts["bridge"] if "replay" in b]}


def fig_reward_total(points, out, stages=STAGE_KEYS, sfx=""):
  fig, axes = plt.subplots(1, 2, figsize=(ptf.FIG_W, 2.6), sharex=True, sharey=True)
  rows = []
  for i, (ax, regime) in enumerate(zip(axes, ("walking", "standing"))):
    r = dumbbell(ax, reward_rows(points, regime, lambda v: sum(v.values()), TERMS_IN[regime], stages),
                 "summed reward per step (-)", show_labels=(i == 0), better="higher")
    rows += [{**x, "regime": regime} for x in r]
    ax.set_title(regime.capitalize())
  write_rows(out, f"f_reward_total{sfx}_all", rows)
  ptf.bottom_legend(fig, stage_handles(stages), ncol=2, height=0.3)
  grey_loss_note(fig)
  ptf.save_pgf(fig, out, f"f_reward_total{sfx}")


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
      r = dumbbell(axes[i, j], [(name(t, seed=True), ptf.policy_color(t),
                                 {s: [u[key] for u in pts[t, regime][s] if key in u] for s in stages})
                                for t in LL_TAGS],
                   lab, show_labels=(j == 0), nticks=3, better="higher", none_note="fell")
      rows += [{**x, "regime": regime, "term": key} for x in r]
    axes[0, j].set_title(regime.capitalize())
  write_rows(out, f"f_ll_reward{sfx}_all", rows)
  fig.subplots_adjust(hspace=1.0, wspace=0.55)
  ptf.bottom_legend(fig, stage_handles(stages), ncol=2, height=0.16 if len(rows_def) < 4 else 0.11)
  grey_loss_note(fig)
  ptf.save_pgf(fig, out, f"f_ll_reward{sfx}")


def fig_reward_return(points, out, stages=STAGE_KEYS, sfx=""):
  """On W&B's scale: return = per-step reward x episode length. Two rows per policy: the 7 terms
  reproducible at all three stages, and all 17 terms (bench live, bridge by plant replay)."""
  fig, axes = plt.subplots(1, 2, figsize=(ptf.FIG_W, 3.0), sharex=True, sharey=True)
  rows = []
  for i, (ax, regime) in enumerate(zip(axes, ("walking", "standing"))):
    dr = []
    for t, p in points.items():
      seven = {s: [EPISODE_S * sum(v.values()) for v in stage_values(p, s, regime, TERMS_IN[regime])]
               for s in stages}
      full = {s: [EPISODE_S * v for v in vals] for s, vals in full17(p, regime).items()}
      dr += [(f"{name(t)}, 7 terms", ptf.policy_color(t), seven),
             (f"{name(t)}, all 17", ptf.policy_color(t), full)]
    r = dumbbell(ax, dr, f"episode return, {EPISODE_S:g} s (-)", show_labels=(i == 0), better="higher")
    rows += [{**x, "regime": regime} for x in r]
    ax.set_title(regime.capitalize())
  write_rows(out, f"f_reward_return{sfx}_all", rows)
  ptf.bottom_legend(fig, stage_handles(stages), ncol=2, height=0.26)
  grey_loss_note(fig)
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
                 show_labels=(i % ncol == 0), nticks=3, better="higher")
    rows += [{**x, "term": t} for x in r]
  spare = axes[len(terms):]
  for ax in spare:
    ax.set_axis_off()
  write_rows(out, f"f_reward_terms_{regime}{sfx}_all", rows)
  fig.subplots_adjust(hspace=0.7, wspace=0.5)   # room for the grey loss column
  if spare:     # the empty grid cell carries the legend instead of a band under the figure
    # anchored right of the cell's left edge: the previous panel's grey numbers hang into the cell's left margin
    spare[0].legend(handles=stage_handles(stages), loc="center left", bbox_to_anchor=(0.28, 0.5),
                    frameon=False, borderaxespad=0)
  else:
    ptf.bottom_legend(fig, stage_handles(stages), ncol=2, height=0.18)
  grey_loss_note(fig)
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
                     label="< 10 s in regime"),
              Patch(facecolor="0.92", label=f"beyond training range (> {PAYLOAD_TRAIN_MAX:g} kg)")]
  ptf.bottom_legend(fig, handles, ncol=2, height=0.21)
  ptf.save_pgf(fig, out, name_)
  return facts


# ----------------------------------------------------------------- standing share ----
# The crossed rel_standing_envs design (2026-09-21): Flat and Hierarchical trained at 5 / 12 / 20 % standing
# environments, each benched at zero command (the stand_pay0 cell, 64 envs x 600 steps x 2 rollout seeds).
# Read from the [BENCH] line of every .log (crossed_bench.json lacks the Hierarchical 5 % arms).
SHARE_DIR = "data/2026-09-21-rse-crossed"
SHARE_EVAL_S = 600 * 0.02    # eval_steps x step_dt of the stand_pay0 cell
SHARE_PANELS = (("act_legs_rad", "standing leg\naction rate (rad)"),
                ("td_per_foot", "touchdowns per foot\nin a 12 s stand (-)"))


def share_points(root):
  """One row per TRAINED checkpoint: re-benches of the same weights (`__rep2`, `__rep3`) are averaged
  into their checkpoint, an independent retrain (`_repro2`) is its own row."""
  import re
  per = {}
  for f in sorted((root / SHARE_DIR).glob("*__stand_pay0*.log")):
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
          ax.plot(x0 + dx, r[key], marker="o" if r["seed"] == "42" else "^", markersize=3.6, linestyle="",
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
    ax.set_xlabel("standing environments in training (%)")
    ax.set_ylabel(ylabel)
    ptf.write_panel_csv(out, name_, key, ["variant", "rse", "run", "seed", "n_bench", "x", "value"], panel)
  handles = [Line2D([], [], color=ptf.CHAIN_COLOR[v], marker="o", markersize=3.4, linewidth=0.8, label=NAMES[v])
             for v in ("A0", "A1a")]
  handles += [Line2D([], [], color="0.3", marker=m, linestyle="", markersize=3.4, label=f"seed {s}")
              for m, s in (("o", 42), ("^", 123))]
  ptf.bottom_legend(fig, handles, ncol=4, height=0.15)
  ptf.save_pgf(fig, out, name_)
  return facts


# --------------------------------------------------------------------------- lean ----
# Quiet-stand torso pitch on the robot (score_stand_attitude.score_entry, the ADR-0006 statistic) for every
# unloaded Run of 2026-09-16 and -23; joint_offset read from each Run's own _meta.json. The offset changes
# with a controller restart, not with the session, so the two sessions pool. Panel b: the simulated
# pitch-per-encoder-bias slopes from play.py --probe-lean (logs in data/2026-10-05-lean/).
LEAN_GLOBS = ("logs/robot_logs/2026_09_16-*/", "logs/robot_logs/2026_09_23-*/")
LEAN_PROBES = {"A0": "data/2026-10-05-lean/probe_A0_s42.log", "A1a": "data/2026-10-05-lean/probe_A1a_s42.log"}
LEAN_CATS = (("A0", True, "Flat\nwith offset"), ("A0", False, "Flat\nno offset"),
             ("A1a", False, "Hierarchical"), ("A1a_DR", False, "Hierarchical\n+ DR"))
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


def fig_lean(rows, slopes, out, name_="f_lean"):
  fig, (ax, bx) = plt.subplots(1, 2, figsize=(ptf.FIG_W, 2.6), gridspec_kw={"width_ratios": [3, 2]})
  ax.axhline(0, color="0.6", linewidth=0.6, zorder=0)
  panel, facts = [], {}
  for x, (v, on, _) in enumerate(LEAN_CATS):
    col = ptf.CHAIN_COLOR[v]
    for k, seed in enumerate(("42", "123")):
      cell = [r for r in rows if r["variant"] == v and r["offset"] == on and r["seed"] == seed
              and not np.isnan(r["pitch_deg"])]
      x0 = x + (k - 0.5) * 0.36
      jit = np.linspace(-0.08, 0.08, len(cell)) if len(cell) > 1 else [0.0]
      for dx, r in zip(jit, cell):
        ax.plot(x0 + dx, r["pitch_deg"], marker="o" if seed == "42" else "^", markersize=3.6, linestyle="",
                color=col, zorder=3)
        panel.append({**r, "x": round(x0 + dx, 3)})
      if cell:
        m = float(np.mean([r["pitch_deg"] for r in cell]))
        ax.plot([x0 - 0.14, x0 + 0.14], [m, m], color=col, linewidth=1.4, zorder=2)
        facts[v, on, seed] = (m, len(cell))
  ax.set_xticks(range(len(LEAN_CATS)))
  ax.set_xticklabels([c[2] for c in LEAN_CATS])
  ax.set_xlim(-0.6, len(LEAN_CATS) - 0.4)
  ax.set_ylabel("quiet-stand torso pitch (deg)\n(negative = leaning back)")
  ptf.write_panel_csv(out, name_, "pitch", ["variant", "seed", "run", "offset", "pitch_deg", "still_pct", "n_rows",
                                            "x"], panel)
  bx.axhline(0, color="0.6", linewidth=0.6, zorder=0)
  w = 0.38
  srows = []
  for k, v in enumerate(("A0", "A1a")):
    xs = np.arange(len(LEAN_GROUPS)) + (k - 0.5) * w
    ys = [slopes[v, g] for g, _ in LEAN_GROUPS]
    bx.bar(xs, ys, width=w * 0.92, color=ptf.CHAIN_COLOR[v], zorder=2)
    srows += [{"variant": v, "group": g, "slope_rad_per_rad": round(y, 4)} for (g, _), y in zip(LEAN_GROUPS, ys)]
  bx.set_xticks(range(len(LEAN_GROUPS)))
  bx.set_xticklabels([l for _, l in LEAN_GROUPS])
  bx.set_xlabel("biased encoder(s)")
  bx.set_ylabel("simulated pitch per\nencoder bias (rad/rad)")
  ptf.write_panel_csv(out, name_, "slope", ["variant", "group", "slope_rad_per_rad"], srows)
  handles = [Line2D([], [], color=ptf.CHAIN_COLOR[v], marker="s", linestyle="", markersize=4, label=NAMES[v])
             for v in ("A0", "A1a", "A1a_DR")]
  handles += [Line2D([], [], color="0.3", marker=m, linestyle="", markersize=3.4, label=f"seed {s}")
              for m, s in (("o", 42), ("^", 123))]
  ptf.bottom_legend(fig, handles, ncol=5, height=0.14)
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
  cax.set_ylabel("whole Run", rotation=0, ha="right", va="center")
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
\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[scaled=0.92]{helvet}   % the figures' sans-serif font (they are typeset in \sffamily)
\usepackage[textwidth=15cm]{geometry}
\usepackage{pgf}
\usepackage[font=small,labelfont=bf]{caption}
\begin{document}
\section*{Report figures}
Hardware Runs are from 2026-09-16, seed 42 and unloaded unless stated. Flat = the flat PPO policy (A0),
Hierarchical = the HIRO hierarchy (A1a), Hierarchical + DR = A1a trained with payload domain randomisation.
"""


def zoom_note(w):
  t0, t1, T = w
  return (f"Panels show {t1 - t0:.0f}\\,s of the {T:.0f}\\,s Run (t = {t0:.0f} to {t1:.0f}\\,s, the "
          "longest walking segment); the strip on top is the whole Run (sagittal command and ground truth, "
          "the window shaded). Error and saturation figures are over the whole Run.")


CHAIN_LOSS_NOTE = ("Grey numbers: loss from bench to bridge sim, (bridge $-$ bench) / |bench| of the two stage "
                   "means, in percent and signed so that positive is worse (all panels here are lower-is-better). "
                   "Where some bridge Runs fell the number covers the Runs that stayed upright; ``fell'' marks a "
                   "policy none of whose bridge Runs did.")
CHAIN_LOSS_NOTE_ROBOT = ("Grey numbers: loss from bench to robot, (robot $-$ bench) / |bench| of the two stage "
                         "means, in percent and signed so that positive is worse (all panels here are "
                         "lower-is-better). The robot ran operator commands, not the bench schedule, so this "
                         "number mixes the sim-to-real gap with the difference in commands; the bridge sim is "
                         "not part of it.")
REWARD_LOSS_NOTE = ("Grey numbers: loss from bench to bridge sim, (bench $-$ bridge) / |bench| of the two stage "
                    "means, in percent; positive means a lower reward at the bridge.")


def write_latex(out, facts):
  """report_figures.tex: every figure \\input with its caption. Caption numbers come from `facts`."""
  f = lambda x, n=2: f"{x:.{n}f}"
  figs = []
  for key in ("A1a", "A0", "A1a_DR"):
    r = facts[key]["rms"]
    figs.append((f"f_track_{key}", f"fig:track_{key}",
      f"Velocity tracking of the {NAMES[key]} policy on the robot (one Run). Operator command "
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
      f"The hierarchy's error decomposition for the {NAMES[key]} policy, same Run as "
      f"Fig.~\\ref{{fig:track_{key}}}: operator command (grey), high-level (HL) target (green), velocity "
      "delivered by the low level as read by the onboard estimate (ochre) and motion-capture ground truth "
      "(black). The sagittal and lateral HL target is the velocity CHANGE the HL requests over its next "
      "decision window, not a velocity: on the robot its velocity state is the IMU increment since its last "
      "decision, zero at each decision, so the green trace is read against zero, not against the command. "
      "The yaw-rate target is absolute. "
      f"Bottom: the HL's action $|g|$, the velocity change divided by the goal scale "
      f"fixed at training ({x['gs'][0]:.2f} m/s sagittal, {x['gs'][1]:.2f} m/s lateral); 1 is the largest "
      f"change it can request in one decision. It is at that limit in {x['sat']['vx']:.0f}\\,\\% (sagittal) "
      f"and {x['sat']['vy']:.0f}\\,\\% (lateral) of walking steps. "
      f"{zoom_note(x['window'])}"))
  def units(regime):
    """(bench envs per policy, bridge Runs per policy, 'who fell' sentence), from the scored units."""
    c = facts["chain"][regime]
    span = lambda v: str(v[0]) if len(set(v)) == 1 else f"{min(v)} to {max(v)}"
    runs = lambda v: span(v) + (" Run" if set(v) == {1} else " Runs")
    fell = [f"{name(t, seed=True)} at the {STAGE_LABEL[st]} ({k} of {n})" for (t, st), (n, k) in c.items() if k]
    return (span([c[t, "bench"][0] for t in CHAIN_TAGS]), runs([c[t, "bridge"][0] for t in CHAIN_TAGS]),
            "A unit that fell is excluded from the means: " + "; ".join(fell) + "." if fell
            else "No unit fell.")
  sim = ("Both stages run one identical 253\\,s command schedule (stand, sagittal 0.3, 0.5 and 1.0 m/s, "
         "backward, sideways, turning, walk-turn-stop) and are scored by the same code. Training bench: "
         "{nb} environments per policy in the training environment (circle: mean; whisker: 5th to 95th "
         "percentile over environments, which differ in randomised foot friction, encoder bias and torso "
         "centre of mass). Bridge sim: the exported policy in the robot's C++ controller with the onboard "
         "estimator, against the shipped MuJoCo plant ({nr} per policy; square: mean of the Runs, which "
         "repeat one plant; single Runs are not drawn, the largest range between the Runs of one policy is "
         "{bsp:.0f}\\,\\% of their mean on any panel here and every Run is in the panel data). {fell} "
         "Commanded joint targets are unclipped at the bench and clipped to the joint limits by the "
         "controller at the bridge.")
  for regime in ("walking", "standing"):
    nb, nr, fell = units(regime)
    figs.append((f"f_chain_sim_{regime}", f"fig:chain_sim_{regime}",
      f"{regime.capitalize()} metrics at the two simulated deploy stages. "
      + sim.format(nb=nb, nr=nr, fell=fell, bsp=facts["bridge_spread_pct"][regime])
      + " Velocity errors are against the simulator's true pelvis velocity, read in the IMU (torso) frame; "
      "yaw-rate error, tilt and roll/pitch rate are the IMU's. Leg joint acceleration is the second "
      "difference of measured joint position at the control rate."
      + (" Cost of transport is the undirected, diagnostic form." if regime == "walking" else "")
      + " " + CHAIN_LOSS_NOTE))
  nb, nr, fell = units("walking")
  figs.append(("f_chain_clean", "fig:chain",
    "Walking metrics at three deploy stages. Training bench and bridge sim as in "
    f"Fig.~\\ref{{fig:chain_sim_walking}} (one identical 253\\,s command schedule, one scorer; {nb} bench "
    f"environments and {nr} at the bridge per policy). {fell} Robot: Runs of 2026-09-16 under operator commands "
    "(diamond: mean of the Runs), so the robot is not command-matched to the simulated stages and a "
    "row shows a range, not a trend. One hierarchical s42 robot Run is omitted as an outlier (1.25 to 2.9 "
    "times its policy median on six of eight metrics). Cost of transport is the undirected, diagnostic form; "
    "its distance is the simulator's true velocity in simulation and motion capture or the onboard estimate "
    "on the robot. Leg joint acceleration is the second difference of measured joint position at the control "
    "rate at all three stages. " + CHAIN_LOSS_NOTE_ROBOT))
  n = facts["upper_n"]
  figs.append(("f_upper_3pol", "fig:upper",
    "Upper-body joint speed and pose deviation on the robot, standing and walking. Bars: mean over all "
    "unloaded Runs of the policy, both seeds, sessions of 2026-09-14 and -16 "
    f"(n = {n['A0']} / {n['A1a']} / {n['A1a_DR']} Runs); dots: single Runs; hollow: less than 10 s in "
    "the regime."))
  rn = facts["reward_n"]
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
      f"Training bench: true state, {lspan(ln['bench'])} environments, a separate rollout on the 253\\,s "
      "schedule of Fig.~\\ref{fig:chain_sim_walking}; bridge sim: the state the controller feeds the low level "
      f"(onboard estimate) and the logged and plant-replayed terms, {lspan(ln['bridge'])} Runs per policy that "
      "did not fall."
      + ("" if sfx else f" Robot: n = {lspan(ln['hardware'])} Runs under operator commands. The robot has no "
         "foot contact sensing, so this figure shows only $r^\\mathrm{track}$ and the contact-free part of "
         "$r^\\mathrm{shape}$ (stand-still, roll/pitch rate, action rate, posture, energy), the same part at "
         "all three stages; the full decomposition is Fig.~\\ref{fig:ll_reward_sim}.")
      + ("" if not facts["ll_partial"] else " Goal tracking only (the shaping terms could not be recomputed): "
         + "; ".join(f"{k} {STAGE_LABEL[s]} Run of {name(t, seed=True)}" for t, s, k in facts["ll_partial"]
                     if s != "bench") + ".")
      + " The yaw-rate goal state is the pelvis rate in training and at the bench but the torso gyro (which "
      "also carries the waist-yaw motion) in the controller. The last step of every high-level decision "
      "window is not scored in $r^\\mathrm{track}$ at any stage (on the robot the velocity state resets at the "
      "next decision). " + REWARD_LOSS_NOTE))
  seven = ("the seven reward terms that can be recomputed from a log (linear-velocity and yaw-rate tracking, "
           "torso tilt, roll/pitch rate, default pose, action rate, stand-still)")
  ru = facts["reward_units"]
  span = lambda v: str(v[0]) if len(set(v)) == 1 else " / ".join(map(str, v))
  simsrc = (f"the training bench (the live reward, {span(ru['bench'])} environments stepping the 253\\,s "
            f"command schedule) and the bridge sim (recomputed from the logs of {span(ru['bridge'])} "
            "Run(s) per policy of the same schedule)")
  for sfx, where, lab in (("_sim", simsrc + ".", "_sim"),
                          ("", simsrc + f", and the robot (recomputed from the logs; n = {rn[0]} / {rn[1]} / "
                               f"{rn[2]} Runs under operator commands).", "")):
    figs.append((f"f_reward_total{sfx}", f"fig:reward_total{lab}",
      f"Summed per-step reward of {seven}, at {where} " + ENV_OBJ + " Markers are means; whiskers as in "
      "Fig.~\\ref{fig:chain_sim_walking}. " + REWARD_LOSS_NOTE))
    figs.append((f"f_reward_return{sfx}", f"fig:reward_return{lab}",
      "Episode return (per-step reward times the 20 s episode, the scale of the flat policy's training "
      "return). " + ENV_OBJ + " Per "
      "policy: the seven reproducible terms, and all 17 terms at the training bench (live) and the bridge "
      "sim, where the ten contact and whole-body terms are recomputed by replaying the logged plant state "
      "in the bridge's own MuJoCo model (validated against the live reward on bench states)."
      + ("" if sfx else " The robot has no contact sensing, so it has no 17-term value.")
      + " The hierarchical policies' gait term is scored on the gait clock their high level commands, the "
      "one their low level trains on. " + REWARD_LOSS_NOTE))
    for regime in ("walking", "standing"):
      figs.append((f"f_reward_terms_{regime}{sfx}", f"fig:reward_terms_{regime}{lab}",
        f"The reward terms of Fig.~\\ref{{fig:reward_total{lab}}} one by one while {regime} (weight times raw "
        "value, per step; higher is better, penalties are negative). " + ENV_OBJ
        + ("" if sfx else " On the robot, linear-velocity tracking lacks the vertical-velocity part (not "
           "measured) and the roll/pitch rate is the body-frame gyro.") + " " + REWARD_LOSS_NOTE))
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
  hier = [m for (v, on, _), (m, _) in c.items() if v != "A0"]
  nl = lambda v, on: " / ".join(str(c.get((v, on, s), (0, 0))[1]) for s in ("42", "123"))
  figs.append(("f_lean", "fig:lean",
    "Standing upright needs a hand calibration for the flat policy and none for the hierarchy. Left: torso "
    "pitch of the robot standing still at zero command (IMU, rows where no leg joint moves faster than "
    "0.1\\,rad/s), every unloaded Run of 2026-09-16 and -23; bars are the mean per policy and seed. Without "
    f"the joint offset the flat policy leans back by {-c['A0', False, '42'][0]:.1f}\\,deg (seed 42) and "
    f"{-c['A0', False, '123'][0]:.1f}\\,deg (seed 123). The offset, one constant from a hand calibration "
    f"(the mean of five trials), brings seed 42 to {c['A0', True, '42'][0]:+.1f}\\,deg but leaves seed 123 at "
    f"{c['A0', True, '123'][0]:+.1f}\\,deg. Every hierarchical policy stands within {min(hier):+.1f} to "
    f"{max(hier):+.1f}\\,deg with no offset. Runs per seed (42 / 123): Flat with offset {nl('A0', True)}, "
    f"without {nl('A0', False)}, Hierarchical {nl('A1a', False)}, Hierarchical + DR {nl('A1a_DR', False)}. "
    "Right: in simulation, the change of standing pitch per unit of encoder bias placed on one sagittal leg "
    "joint, or split over all three (least-squares slope over biases from $-0.032$ to $+0.032$\\,rad, seed-42 "
    f"policies, 64 environments). The flat policy turns an encoder error into lean ({sl['A0', 'all']:+.2f}\\,rad/rad "
    f"for all three joints); the hierarchy barely does ({sl['A1a', 'all']:+.2f}, "
    f"{abs(sl['A0', 'all'] / sl['A1a', 'all']):.1f} times less, opposite sign)."))
  c, srow = facts["share"]["cell"], facts["share"]["rows"]
  lo, mid = min(r["rse"] for r in srow), sorted({r["rse"] for r in srow})[1]
  tds = [r["td_per_foot"] for r in srow if r["variant"] == "A1a" and r["rse"] == lo]
  flat = [c["act_legs_rad", "A0", s][0] for s in sorted({r["rse"] for r in srow})]
  rep = [r["act_legs_rad"] for r in srow if r["variant"] == "A0" and r["rse"] == mid]
  ncell = lambda v: " / ".join(str(c["act_legs_rad", v, s][1]) for s in sorted({r["rse"] for r in srow}))
  figs.append(("f_standing_share", "fig:standing_share",
    "Standing in simulation against the share of standing environments in training, a crossed design: both "
    "architectures trained at 5, 12 and 20\\,\\% standing environments, otherwise unchanged. Bench at zero "
    "command, 64 environments, 600 steps (12\\,s), two rollout seeds. With "
    f"{100 * lo:g}\\,\\% the hierarchical policy marches in place: standing leg action rate "
    f"{c['act_legs_rad', 'A1a', lo][0]:.3f} rad against the flat policy's {c['act_legs_rad', 'A0', lo][0]:.4f} "
    f"({c['act_legs_rad', 'A1a', lo][0] / c['act_legs_rad', 'A0', lo][0]:.0f} times), and "
    f"{min(tds):.0f} to {max(tds):.0f} touchdowns per foot where a still stand gives 1 (the floor). From "
    f"{100 * mid:g}\\,\\% on it stands like the flat policy ({c['act_legs_rad', 'A1a', mid][0]:.4f} rad, "
    f"{c['td_per_foot', 'A1a', mid][0]:.2f} touchdowns per foot). The flat policy never steps at any share and "
    f"moves within {min(flat):.4f} to {max(flat):.4f} rad, against {min(rep):.4f} to {max(rep):.4f} rad across "
    f"its {len(rep)} checkpoints at {100 * mid:g}\\,\\% alone. Touchdowns per foot = 12\\,s / the bench's stride period. "
    "Markers: one trained checkpoint each (circle seed 42, triangle seed 123; repeated benches of one "
    "checkpoint averaged, an independent retrain of seed 42 drawn separately), n = "
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
    "The robot carrying a backpack payload (Runs of 2026-09-23). Both policies run the same frozen low level "
    "and differ only in the seed-42 high level, trained without (Hierarchical) or with (Hierarchical + DR) "
    f"payload randomisation over 0 to {PAYLOAD_TRAIN_MAX:g}\\,kg. At {top:g}\\,kg, beyond that range, the high "
    f"level trained without randomisation loses the stand: standing leg action rate {a:.3f} against {b:.3f} rad "
    f"({a / b:.1f} times) and sagittal velocity error {e:.2f} against {g:.2f} m/s at zero sagittal command. It "
    "could not be commanded to walk (n/a), and both its Runs were stopped after less than 13\\,s to avoid a "
    "fall, so its values understate the failure. The operator log records that both Hierarchical + DR Runs "
    "at that load fell against the wall, after standing and walking under command. Markers: single Runs (n = "
    f"{ns('A1a')} Hierarchical and {ns('A1a_DR')} Hierarchical + DR at "
    + " / ".join(f"{kg:g}" for kg in kgs) + "\\,kg); bars: mean of the Runs with at least 10\\,s in the "
    "regime; hollow: shorter, drawn but not in the mean. Velocity is the onboard EKF estimate (no motion "
    f"capture in this session). Cost of transport is normalised by robot plus payload mass ({ROBOT_KG:g}\\,kg + "
    "payload); the operator's commands were not speed-matched across Runs."))
  var = []
  for kind, keys, ref in (("track", ("A1a", "A0", "A1a_DR"), "track"), ("hier", ("A1a", "A1a_DR"), "hier")):
    for key in keys:
      for tag, a, e, what in facts[key]["variants"]:
        var.append((f"f_{kind}_{key}_{tag}", f"fig:{kind}_{key}_{tag}",
          f"{NAMES[key]}, {'tracking' if kind == 'track' else 'hierarchy'}: t = {a:.0f} to {e:.0f}\\,s, {what}. "
          f"Otherwise as Fig.~\\ref{{fig:{ref}_{key}}}."))
  body = "".join(f"""
\\begin{{figure}}[p]
  \\centering
  \\input{{{f}.pgf}}
  \\caption{{{c}}}
  \\label{{{l}}}
\\end{{figure}}
\\clearpage
""" for f, l, c in figs)
  body += ("\n\\section*{Zoom variants}\nEvery walking segment and the start-up of the Runs in "
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
  print(f"[REPORT] chain: dropped {sorted(CHAIN_EXCLUDE)} and the cotcap Runs")
  rows_w, n_w = chain_points(kept, root, "walking")
  rows_s, n_s = chain_points(kept, root, "standing")
  fig_chain_seq(rows_w, CHAIN_KEYS, STAGE_KEYS, out, "f_chain_clean")
  fig_chain_seq(rows_w, SIM_KEYS["walking"], STAGE_KEYS[:2], out, "f_chain_sim_walking")
  fig_chain_seq(rows_s, SIM_KEYS["standing"], STAGE_KEYS[:2], out, "f_chain_sim_standing")
  facts["chain"] = {"walking": n_w, "standing": n_s}
  # Largest bridge Run-to-Run spread, (max - min) / mean over the Runs of one policy, over every
  # metric of the simulation figure: the number that justifies not drawing single bridge Runs.
  facts["bridge_spread_pct"] = {
    r: max(100 * (max(v) - min(v)) / abs(np.mean(v)) for k in SIM_KEYS[r]
           for _, _, st in rows(k, ("bridge",)) for v in [st["bridge"]] if len(v) > 1 and np.mean(v))
    for r, rows in (("walking", rows_w), ("standing", rows_s))}
  print(f"[REPORT] chain: largest bridge Run-to-Run spread {facts['bridge_spread_pct']} %")
  write_rows(out, "f_chain_units", [{"policy": t, "stage": st, "regime": r, "n_scored": n, "n_fell": k}
                                    for r, c in facts["chain"].items() for (t, st), (n, k) in c.items()])

  # Pool by VARIANT: `A1a_s42` / `A1a_s123` / the seedless 09-14 `A1a` are one policy, and
  # `policy_label` keeps loaded Runs apart by their `+kg` suffix, which the variant filter drops.
  pooled = []
  for b in both:
    variant, _seed = ptf._variant_seed(b.policy)
    if variant in UPPER_VARIANTS and b.regimes:
      pooled.append(dataclasses.replace(b, policy=variant))
  n = {v: sum(b.policy == v for b in pooled) for v in UPPER_VARIANTS}
  print(f"[REPORT] upper: Runs per policy {n}")
  facts["upper_n"] = n
  ptf.fig_upper(pooled, out, name="f_upper_3pol")

  points = load_reward_points(root)
  print("[REPORT] reward: hardware Runs per policy "
        + str({t: len(p["hardware"]) for t, p in points.items()}))
  for stages, sfx in ((STAGE_KEYS, ""), (STAGE_KEYS[:2], "_sim")):
    fig_reward_total(points, out, stages, sfx)
    fig_reward_return(points, out, stages, sfx)
    for regime in ("walking", "standing"):
      fig_reward_terms(points, out, regime, stages, sfx)
  facts["reward_units"] = {s: [len(p[s]) for p in points.values()] for s in ("bench", "bridge")}
  llp = ll_points(kept, root)
  for stages, sfx in ((STAGE_KEYS, ""), (STAGE_KEYS[:2], "_sim")):
    fig_ll_reward(llp, out, stages, sfx)
  facts["ll_n"] = {s: [len(llp[t, "walking"][s]) for t in LL_TAGS] for s in STAGE_KEYS}
  # units that have goal tracking but not the shaping terms (a refused replay pair, no velocity truth)
  facts["ll_partial"] = [(t, s, sum("r_track" in u and "r_shape_cf" not in u for u in llp[t, "walking"][s]))
                         for t in LL_TAGS for s in STAGE_KEYS]
  facts["ll_partial"] = [x for x in facts["ll_partial"] if x[2]]
  print(f"[REPORT] low-level goal reward units per policy {facts['ll_n']}")
  facts["reward_n"] = [len(p["hardware"]) for p in points.values()]

  trow = train_points(root)
  fig_training(trow, out)
  facts["train"] = trow
  lrow, lsl = lean_points(root), lean_slopes(root)
  facts["lean"] = {"cell": fig_lean(lrow, lsl, out), "slope": lsl, "n": len(lrow)}
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
