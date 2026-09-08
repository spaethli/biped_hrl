"""WP5 Phase 2: train `phi` (+ the command-confound ablation) on pooled rollout data.

Two models come out of this: the real, contract-compliant 92-dim `phi` (what
`export_adapt_encoder_onnx` ships), and a throwaway 89-dim ablation with the command
columns dropped (Python-only, never exported) -- the cheap A/B for grill item (d) agreed
in doc/hrl/A1a_plan.md "WP5 Phase 2" rather than reopening the WP5d deploy contract.

  python scripts/train_phi.py --dirs data/2026-09-wp5-phase2/rollout_s42 \\
      data/2026-09-wp5-phase2/rollout_s123 --out data/2026-09-wp5-phase2/models
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from export_dummy_phi import _CnnPhi
from phi_data import PooledRollouts, WindowDataset, normalization_stats, partition, \
  pooled_indices_from_dirs
from torch.utils.data import DataLoader

POLICY_DIM, COMMAND_DIM, HISTORY_LEN = 89, 3, 50


def train_one(train_ds, val_ds, d_in: int, device: str, tag: str,
              batch_size: int = 512, max_epochs: int = 100, patience: int = 10):
  model = _CnnPhi(n_z=5, d_in=d_in, h=HISTORY_LEN).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                         num_workers=4, drop_last=True, persistent_workers=True)
  val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=2,
                       persistent_workers=True)

  best_val, best_state, bad_epochs = float("inf"), None, 0
  for epoch in range(max_epochs):
    t0 = time.time()
    model.train()
    for x, y in train_dl:
      x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
      opt.zero_grad()
      loss = ((model(x) - y) ** 2).mean()
      loss.backward()
      opt.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
      for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        val_losses.append(((model(x) - y) ** 2).mean(dim=0).cpu())  # per-column
    val_loss = torch.stack(val_losses).mean(dim=0)
    val_scalar = val_loss.mean().item()
    print(f"[{tag}] epoch {epoch}: val_loss={val_scalar:.5f} "
          f"per-col={[round(v, 4) for v in val_loss.tolist()]} ({time.time() - t0:.1f}s)")

    if val_scalar < best_val - 1e-5:
      best_val, best_state, bad_epochs = val_scalar, {
        k: v.detach().clone() for k, v in model.state_dict().items()}, 0
    else:
      bad_epochs += 1
      if bad_epochs >= patience:
        print(f"[{tag}] early stop at epoch {epoch} (best val {best_val:.5f})")
        break

  model.load_state_dict(best_state)
  return model, best_val


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dirs", nargs="+", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--device", default=None)
  ap.add_argument("--max-epochs", type=int, default=100)
  args = ap.parse_args()

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  dirs = [Path(d) for d in args.dirs]

  pooled = PooledRollouts(dirs)
  indices = pooled_indices_from_dirs(dirs, pooled.offsets)
  parts = partition(pooled.e_true_pooled, seed=0)
  norm = normalization_stats(pooled.e_true_pooled, parts["train"])
  print(f"[train_phi] pooled envs={pooled.num_envs} "
        f"train={len(parts['train'])} val={len(parts['val'])} "
        f"test={len(parts['test'])} interp={len(parts['interp'])}")
  print(f"[train_phi] normalization: {norm}")

  train_set = set(parts["train"].tolist())
  val_set = set(parts["val"].tolist())
  train_idx = indices[np.isin(indices[:, 0], list(train_set))]
  val_idx = indices[np.isin(indices[:, 0], list(val_set))]
  print(f"[train_phi] windows: train={len(train_idx)} val={len(val_idx)}")

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  np.savez(out / "partition.npz", **parts)
  (out / "norm.json").write_text(json.dumps(norm, indent=2))
  (out / "meta.json").write_text(json.dumps({
    "dirs": [str(d) for d in dirs], "offsets": pooled.offsets,
  }, indent=2))

  for tag, drop_command, d_in in (("real", False, POLICY_DIM + COMMAND_DIM),
                                   ("ablation", True, POLICY_DIM)):
    train_ds = WindowDataset(pooled, train_idx, norm, drop_command=drop_command)
    val_ds = WindowDataset(pooled, val_idx, norm, drop_command=drop_command)
    model, best_val = train_one(train_ds, val_ds, d_in, device, tag,
                                 max_epochs=args.max_epochs)
    torch.save(model.state_dict(), out / f"phi_{tag}.pt")
    print(f"[train_phi] {tag}: best val loss {best_val:.5f} -> {out}/phi_{tag}.pt")


if __name__ == "__main__":
  main()
