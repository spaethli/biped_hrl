"""WP5 Phase 2 grill item (c): quantify the distribution shift between the histories
`phi` trained on (behavior = true-`e` HL) and the histories it would actually see once
its own estimate drives the HL (behavior = phi-driven HL) -- the standard RMA gap.

Per-column standardized mean difference, flagged at >2 sigma. Starts simple (no
discriminator) per doc/hrl/A1a_plan.md "WP5 Phase 2"; escalate only if this is ambiguous.

  python scripts/dist_shift.py --train-dirs data/2026-09-wp5-phase2/rollout_s42 \\
      data/2026-09-wp5-phase2/rollout_s123 \\
      --phi-driven-dirs data/2026-09-wp5-phase2/phidriven_s42 data/2026-09-wp5-phase2/phidriven_s123 \\
      --out data/2026-09-wp5-phase2/dist_shift.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

COLUMN_NAMES = (
  [f"ang_vel_{a}" for a in "xyz"] + [f"grav_{a}" for a in "xyz"] + ["phase_sin", "phase_cos"]
  + [f"joint_pos_{i}" for i in range(27)] + [f"joint_vel_{i}" for i in range(27)]
  + [f"action_{i}" for i in range(27)] + ["cmd_vx", "cmd_vy", "cmd_wz"]
)


def pooled_frames(dirs: list[Path]) -> np.ndarray:
  """All raw [T,92] frames across every env and every source dir, flattened to [N,92]."""
  parts = []
  for d in dirs:
    buf = np.load(d / "buf.npy", mmap_mode="r")
    parts.append(np.asarray(buf).reshape(-1, buf.shape[-1]))
  return np.concatenate(parts, axis=0)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--train-dirs", nargs="+", required=True)
  ap.add_argument("--phi-driven-dirs", nargs="+", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  a = pooled_frames([Path(d) for d in args.train_dirs])
  b = pooled_frames([Path(d) for d in args.phi_driven_dirs])
  assert a.shape[1] == b.shape[1] == 92 == len(COLUMN_NAMES)

  mean_a, std_a = a.mean(axis=0), a.std(axis=0)
  mean_b = b.mean(axis=0)
  smd = (mean_b - mean_a) / np.where(std_a > 1e-9, std_a, np.nan)

  report = {}
  flagged = []
  for i, name in enumerate(COLUMN_NAMES):
    report[name] = {
      "train_mean": float(mean_a[i]), "train_std": float(std_a[i]),
      "phi_driven_mean": float(mean_b[i]), "standardized_mean_diff": float(smd[i]),
    }
    if abs(smd[i]) > 2.0:
      flagged.append(name)

  out = {"n_flagged_gt_2sigma": len(flagged), "flagged_columns": flagged, "per_column": report}
  Path(args.out).write_text(json.dumps(out, indent=2))
  print(f"[dist_shift] {len(flagged)}/92 columns shift >2sigma: {flagged}")
  print(f"[dist_shift] wrote {args.out}")


if __name__ == "__main__":
  main()
