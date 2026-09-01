#!/usr/bin/env python3
"""WP1b figures: metric-vs-T curves, one panel per vx, one line per payload,
with +/-1 SEM bands (across-repeat). Marks the bootstrap T* where resolvable.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path("/home/iams/ramlab_ws/src/unitree_rl_mjlab")
OUT = REPO / "data/2026-08-28-wp1b-payload-repeat"
df = pd.read_csv(OUT / "wp1b_sweep.csv")
for k in (0.1, 0.3, 1.0):
  df[f"cot_copper_k{k}"] = df["cot"] + (k / 0.3) * (df["cot_copper"] - df["cot"])

VXS = sorted(df.vx_pinned.unique())
PAYLOADS = sorted(df.payload_kg.unique())
COLORS = plt.cm.viridis(np.linspace(0, 0.9, len(PAYLOADS)))

PANELS = [
  ("cot", "mechanical CoT"),
  ("cot_copper_k0.3", "copper-loss CoT (k=0.3)"),
  ("cot_copper_k1.0", "copper-loss CoT (k=1.0)"),
  ("err_vx", "err_vx"),
  ("ss_vx_var", "ss_vx_var"),
  ("mech_power_w", "mech power (W)"),
]

for metric, title in PANELS:
  fig, axes = plt.subplots(1, len(VXS), figsize=(5 * len(VXS), 4.2), squeeze=False)
  for j, vx in enumerate(VXS):
    ax = axes[0][j]
    for c, p in zip(COLORS, PAYLOADS):
      sub = df[(df.vx_pinned == vx) & (df.payload_kg == p)]
      g = sub.groupby("T_pinned")[metric]
      Ts = np.array(sorted(sub.T_pinned.unique()))
      mean = g.mean().reindex(Ts).to_numpy()
      sem = (g.std(ddof=1) / np.sqrt(g.size())).reindex(Ts).to_numpy()
      ax.plot(Ts, mean, "-o", color=c, ms=4, label=f"{p} kg")
      ax.fill_between(Ts, mean - sem, mean + sem, color=c, alpha=0.25)
    ax.set_title(f"vx = {vx} m/s")
    ax.set_xlabel("stride period T (s)")
    if j == 0:
      ax.set_ylabel(title)
    ax.grid(alpha=0.3)
  axes[0][0].legend(fontsize=8, title="payload")
  fig.suptitle(f"WP1b: {title} vs T  (bands = +/-1 SEM across 8 independent processes)")
  fig.tight_layout()
  safe = metric.replace(".", "p")
  fig.savefig(OUT / f"wp1b_{safe}_vs_T.png", dpi=110)
  plt.close(fig)
  print(f"wrote wp1b_{safe}_vs_T.png")
