#!/usr/bin/env python3
"""Payload-DR comparison, thesis-figure-ready, from the 2026-09-23 hardware session.

A1a_s42 (no DR) vs A1a_DR_s42 (DR), same seed, at 0 / 7.5 / 15 kg backpack -- the loaded
hierarchical comparison ADR-0013 and f_seedband's own caption both flag as never having
flown ("no hierarchical DR arm was ever loaded on the robot"). This session is that data
point. The five A0 lean-offset Runs from the same session are OUT OF SCOPE here (a
calibration probe, not a payload Run) and excluded by the bundle glob, never by filtering
inside this script.

Six panels, standing over walking, one point per Run (never a bare mean, matching the rest
of this repo's figures): commanded cadence and arm-settling time (from
analyze_cadence_settling.py's cadence_settling.json) and standing act_legs_rad; walking cot,
mech_power_w and walking act_legs_rad (from bench_flight_recorder.py's *.standing/walking.json,
already scored per-bundle with --traj-dir pinned to each bundle so the pairing is the
labels.json one, not a fresh correlation search that would re-pick a stale file).

act_legs_rad, not ajit_legs_p95, is the standing channel: the tail statistic does not
re-measure across repeats on this project's own record (`quiet_policy_makes_a_tail_
statistic_noisy`), act_legs_rad does.

Reuses the OUTPUT convention from plot_thesis_figures.py (save_pgf, write_panel_csv, FIG_W,
rcParams, CHAIN_COLOR, load_bundles, load_cadence) rather than reinventing it -- same
precedent as figures/benchmark_blindness's separate script, because the data domain here
(one session, payload x DR, no mocap) differs from that module's mocap-session repertoire.

Usage:
  python scripts/plot_payload_dr.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_thesis_figures import (  # noqa: E402
  CHAIN_COLOR, FIG_W, PERIOD_RANGE, REPO, _variant_seed, bottom_legend, load_bundles,
  load_cadence, save_pgf, tex, write_panel_csv,
)

import matplotlib  # noqa: E402
matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

BUNDLE_GLOB = "logs/robot_logs/2026_09_23-_0[89]_*/"
CADENCE_JSON = "data/2026-09-23-hardware-session/cadence_settling.json"
PAYLOAD_KG = (0.0, 7.5, 15.0)
VARIANTS = ("A1a", "A1a_DR")


def fig_payload_dr(bundles, cad_rows, out_dir, name="f_payload_dr"):
  by_bundle = {r["bundle"]: r for r in cad_rows}
  fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 4.6))
  ax_of = dict(zip(("cad", "arm", "act_s", "cot", "power", "act_w"), axes.flat))
  panels = {k: [] for k in ax_of}
  cols = ["policy", "variant", "payload_kg", "run", "x", "value", "source"]

  def put(key, variant, kg, b, y, src, extra=None):
    x0 = PAYLOAD_KG.index(kg) + (0.12 if variant == "A1a_DR" else -0.12)
    st = _variant_seed(variant)  # ("A1a", "42") -- seed unused, colour only
    color = CHAIN_COLOR.get(variant, "0.4")
    ax_of[key].plot(x0, y, marker="o", markersize=3.6, linestyle="", color=color,
                    alpha=0.9, zorder=3)
    row = {"policy": b.policy, "variant": variant, "payload_kg": kg, "run": b.dir.name,
          "x": round(float(x0), 6), "value": round(float(y), 6), "source": src}
    if extra:
      row.update(extra)
    panels[key].append(row)

  n_skipped_torque = 0
  for b in bundles:
    tag = b.run["policy_tag"]
    variant, seed = _variant_seed(tag)
    if variant not in VARIANTS:
      continue
    kg = float(b.run.get("payload_kg") or 0.0)
    if kg not in PAYLOAD_KG:
      continue

    c = by_bundle.get(b.dir.name, {})
    cad = (c.get("cadence") or {}).get("standing") or {}
    if (c.get("cadence") or {}).get("available") and cad:
      put("cad", variant, kg, b, cad["mean"],
          "cadence_settling.json cadence.standing.mean")
    t01 = ((c.get("arm") or {}).get("t_settle_0.1") or {}).get("median")
    if t01 is not None:
      put("arm", variant, kg, b, t01,
          "cadence_settling.json arm.t_settle_0.1.median")

    S = b.regimes.get("standing", {})
    if "act_legs_rad" in S:
      put("act_s", variant, kg, b, S["act_legs_rad"], "standing.json act_legs_rad")

    W = b.regimes.get("walking", {})
    if W.get("cot") is not None and W.get("cot_distance_m") and W.get("duration_s"):
      put("cot", variant, kg, b, W["cot"], "walking.json cot",
          extra={"achieved_speed_m_s": round(W["cot_distance_m"] / W["duration_s"], 4)})
    if W.get("mech_power_w") is not None:
      put("power", variant, kg, b, W["mech_power_w"], "walking.json mech_power_w")
    elif "act_legs_rad" in W:
      n_skipped_torque += 1
    if "act_legs_rad" in W:
      put("act_w", variant, kg, b, W["act_legs_rad"], "walking.json act_legs_rad")

  titles = {"cad": "standing: commanded period (s)",
            "arm": "standing: arm settling, 0.1 rad/s (s)",
            "act_s": "standing: act\\_legs\\_rad",
            "cot": "walking: cost of transport",
            "power": "walking: mech power (W)",
            "act_w": "walking: act\\_legs\\_rad"}
  for key, ax in ax_of.items():
    ax.set_title(titles[key], fontsize=7)
    ax.tick_params(labelsize=6)
    ax.margins(y=0.15)
    ax.set_xticks(range(len(PAYLOAD_KG)))
    ax.set_xticklabels([f"{kg:g} kg" for kg in PAYLOAD_KG], fontsize=6)
    ax.set_xlim(-0.5, len(PAYLOAD_KG) - 0.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    write_panel_csv(out_dir, name, key, cols + (["achieved_speed_m_s"] if key == "cot" else []),
                    panels[key])
  ax_of["cad"].axhline(PERIOD_RANGE[1], color="0.35", linestyle="--", linewidth=0.7)
  ax_of["cad"].axhline(PERIOD_RANGE[0], color="0.35", linestyle="--", linewidth=0.7)

  from matplotlib.lines import Line2D
  handles = [Line2D([], [], marker="o", linestyle="", color=CHAIN_COLOR[v], markersize=3.5,
                    label=tex(v)) for v in VARIANTS]
  notes = [
    "One point per Run, same seed (s42) throughout; A0 lean-offset Runs from the same "
    "session are out of scope and excluded at the bundle-glob level. "
    f"{n_skipped_torque} Run(s) at 15 kg (no DR) dropped from cot/power: stopped early to "
    "avoid a crash, below the 20 s window bench\\_flight\\_recorder.py needs to accept a "
    "torque pairing -- act\\_legs\\_rad still shown (needs no telemetry).",
    "Commands were not speed-matched across Runs (operator-driven), so cot and mech power "
    "are not read at a fixed speed; achieved speed is recorded per point in "
    "f\\_payload\\_dr\\_cot.csv. Dashed lines on the cadence panel are the deploy yaml's "
    "cadence\\_period\\_range bounds.",
  ]
  import textwrap
  lines = [w for n in notes for w in textwrap.wrap(n, 118)]
  leg_in, note_in = 0.35, 0.0975 * len(lines)
  fig_h = 4.6 + leg_in + note_in - 0.3
  fig.set_size_inches(FIG_W, fig_h)
  bottom_legend(fig, handles, ncol=2, height=(leg_in + note_in + 0.2) / fig_h)
  for i, n in enumerate(lines):
    fig.text(0.012, (leg_in + note_in + 0.05) / fig_h - 0.0975 / fig_h * i, n, fontsize=5.0,
             color="0.3", ha="left", va="top")
  print("[PLOT] f_payload_dr: " + ", ".join(f"{k}={len(v)}" for k, v in panels.items()))
  return save_pgf(fig, out_dir, name)


def main():
  out_dir = REPO / "figures" / "2026-09-23-hardware-session"
  bundles = load_bundles(REPO, [BUNDLE_GLOB])
  print(f"[PLOT] {len(bundles)} bundles from {BUNDLE_GLOB}")
  cad_rows = load_cadence(REPO, override=CADENCE_JSON)
  fig_payload_dr(bundles, cad_rows, out_dir)
  return 0


if __name__ == "__main__":
  sys.exit(main())
