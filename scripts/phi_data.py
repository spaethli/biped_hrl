"""WP5 Phase 2 shared data plumbing: pooling, partitioning, and window slicing.

Loads the ring-buffer rollouts `collect_phi_data.py` writes and turns their
`(env_id, t)` index lists into the train/val/test/interpolation partition agreed in
doc/hrl/A1a_plan.md "WP5 Phase 2". Windows are sliced from the memory-mapped buffers on
demand -- never materialized in bulk -- so this stays cheap regardless of dataset size.
Shared by `train_phi.py` and `eval_phi.py` so the two can never see a different split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from export_dummy_phi import _CnnPhi
from src.tasks.velocity.mdp.observations import E_NAMES

INTERP_BAND = (5.0, 7.0)  # withheld payload band -> split (ii)
TEST_FRAC = 0.20
VAL_FRAC = 0.10
# Pinned cold-start prior (WP5d contract): payload/CoM nominal (delta=0), friction at the
# DR midpoint (an ABSOLUTE coefficient -- a zero fill would be a frictionless floor).
Z_COLD = (0.0, 0.0, 0.0, 0.0, 0.95)


class PooledRollouts:
  """Memory-maps one or more `collect_phi_data.py` output dirs and assigns each env a
  global id `g = source_idx * num_envs_of_that_source + local_id`, so windows can be
  addressed uniformly regardless of which rollout they came from."""

  def __init__(self, dirs: list[Path]):
    self.bufs, self.e_true, self.meta, self.offsets = [], [], [], [0]
    for d in dirs:
      self.bufs.append(np.load(d / "buf.npy", mmap_mode="r"))
      self.e_true.append(np.load(d / "e_true.npy"))
      self.meta.append(json.loads((d / "meta.json").read_text()))
      self.offsets.append(self.offsets[-1] + self.meta[-1]["num_envs"])
    self.e_true_pooled = np.concatenate(self.e_true, axis=0)  # [G, 5]
    assert list(E_NAMES) == self.meta[0]["e_names"]

  @property
  def num_envs(self) -> int:
    return self.offsets[-1]

  def _locate(self, g: int) -> tuple[int, int]:
    src = next(i for i in range(len(self.offsets) - 1)
               if self.offsets[i] <= g < self.offsets[i + 1])
    return src, g - self.offsets[src]

  def window(self, g: int, t: int) -> np.ndarray:
    src, local = self._locate(g)
    return np.asarray(self.bufs[src][local, t - 49 : t + 1, :])  # [50, 92]


def pooled_indices_from_dirs(dirs: list[Path], offsets: list[int]) -> np.ndarray:
  rows = []
  for d, off in zip(dirs, offsets[:-1]):
    idx = np.load(d / "indices.npy")
    idx = idx.copy()
    idx[:, 0] += off
    rows.append(idx)
  return np.concatenate(rows, axis=0)


def partition(e_true_pooled: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
  """Env-level train/val/test/interpolation split (Topic 3/Topic 1 of the grilled spec).

  Held-out envs (test, split i) are drawn first from the FULL pool; the payload-band
  withholding (split ii) and the internal val carve-out (early stopping only) are then
  drawn from what remains, so a test env is never also an interpolation or val env.
  """
  n = e_true_pooled.shape[0]
  rng = np.random.default_rng(seed)
  perm = rng.permutation(n)
  n_test = int(round(TEST_FRAC * n))
  test_envs = perm[:n_test]
  remaining = perm[n_test:]

  payload = e_true_pooled[remaining, 0]
  band_mask = (payload >= INTERP_BAND[0]) & (payload < INTERP_BAND[1])
  interp_envs = remaining[band_mask]
  trainable = remaining[~band_mask]

  n_val = int(round(VAL_FRAC * len(trainable)))
  val_envs = trainable[:n_val]
  train_envs = trainable[n_val:]

  return {
    "train": train_envs, "val": val_envs, "test": test_envs, "interp": interp_envs,
  }


def normalization_stats(e_true_pooled: np.ndarray, train_envs: np.ndarray) -> dict:
  """z_center/z_scale = train-split mean/std; z_clip_lo/hi = train-split empirical
  min/max (NOT export_dummy_phi.py's `_latent_bounds()`, which reads `base_com`'s own
  event range and is stale under the ADR-0010 coupled DR), WIDENED to bracket `Z_COLD`:
  a continuous DR's finite sample almost never hits its true boundary exactly (payload's
  empirical min was 0.0011, not 0), so the pinned cold-start prior can fall just outside
  the raw empirical range even though it is physically nominal. The exporter refuses a
  cold prior outside [clip_lo, clip_hi] (by design -- WP5d), so the bound must be widened
  to include it rather than the prior narrowed to fit a sampling artifact."""
  e_train = e_true_pooled[train_envs]
  lo = np.minimum(e_train.min(axis=0), Z_COLD)
  hi = np.maximum(e_train.max(axis=0), Z_COLD)
  return {
    "center": e_train.mean(axis=0).tolist(),
    "scale": e_train.std(axis=0).tolist(),
    "clip_lo": lo.tolist(),
    "clip_hi": hi.tolist(),
    "z_cold": list(Z_COLD),
  }


def load_phi_for_inference(phi_dir: Path, device: str):
  """Loads `train_phi.py`'s real (92-dim) `phi` + its norm stats for SIM-SIDE inference
  (`play.py --hl-obs-e-source phi`, `collect_phi_data.py --phi-dir`). Not the deploy path
  -- that runs the exported ONNX; this runs the torch module directly."""
  norm = json.loads((Path(phi_dir) / "norm.json").read_text())
  if "z_cold" not in norm:
    norm["z_cold"] = list(Z_COLD)  # pre-Z_COLD norm.json written before this key existed
  model = _CnnPhi(n_z=5, d_in=89 + 3, h=50).to(device)
  model.load_state_dict(torch.load(Path(phi_dir) / "phi_real.pt", map_location=device))
  model.eval()
  return model, norm


def command_regime(window: np.ndarray) -> str:
  """Classify a [50,92] window by its command columns (89:92): 'standing' if constant
  (within float/heading-controller tolerance) and small, 'walking' if constant and
  large, else 'transition' (a resample happened inside the window)."""
  cmd = window[:, 89:92]
  spread = cmd.std(axis=0).max()
  level = np.abs(cmd).max()
  if spread > 0.02:  # a resample (or heading-hold drift beyond tolerance) occurred
    return "transition"
  return "standing" if level < 0.05 else ("walking" if level > 0.1 else "transition")


class WindowDataset(Dataset):
  """`indices[i] = (global_env_id, t)` -> (window [50,92 or 89], normalized target [5])."""

  def __init__(self, pooled: PooledRollouts, indices: np.ndarray, norm: dict,
               drop_command: bool = False):
    self.pooled = pooled
    self.indices = indices
    self.center = np.asarray(norm["center"], dtype=np.float32)
    self.scale = np.asarray(norm["scale"], dtype=np.float32)
    self.drop_command = drop_command

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(self, i: int):
    g, t = self.indices[i]
    w = self.pooled.window(int(g), int(t))
    if self.drop_command:
      w = w[:, :89]
    target = (self.pooled.e_true_pooled[int(g)] - self.center) / self.scale
    return torch.from_numpy(w.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
