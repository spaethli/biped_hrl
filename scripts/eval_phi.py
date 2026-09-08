"""WP5 Phase 2: evaluate `phi` (+ the ablation) against the pre-registered gate, the four
splits, and the two command-confound checks. Spec -> doc/hrl/A1a_plan.md "WP5 Phase 2".

  python scripts/eval_phi.py --model-dir data/2026-09-wp5-phase2/models \\
      --ood-mass-dirs data/2026-09-wp5-phase2/ood_mass_s42 data/2026-09-wp5-phase2/ood_mass_s123 \\
      --ood-lever-dirs data/2026-09-wp5-phase2/ood_lever_s42 data/2026-09-wp5-phase2/ood_lever_s123 \\
      --out data/2026-09-wp5-phase2/report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from export_dummy_phi import _CnnPhi
from phi_data import E_NAMES, PooledRollouts, command_regime, pooled_indices_from_dirs

POLICY_DIM, COMMAND_DIM = 89, 3


def load_model(path: Path, d_in: int, device: str) -> _CnnPhi:
  m = _CnnPhi(n_z=5, d_in=d_in, h=50).to(device)
  m.load_state_dict(torch.load(path, map_location=device))
  m.eval()
  return m


def decode(raw: np.ndarray, norm: dict) -> np.ndarray:
  return np.asarray(norm["center"]) + np.asarray(norm["scale"]) * raw


@torch.no_grad()
def predict_windows(model, pooled: PooledRollouts, idx: np.ndarray, norm: dict,
                     device: str, drop_command: bool, batch: int = 4096) -> np.ndarray:
  """Returns physical-unit z_hat, one row per idx entry, shape [len(idx), 5]."""
  out = []
  for s in range(0, len(idx), batch):
    chunk = idx[s : s + batch]
    ws = np.stack([pooled.window(int(g), int(t)) for g, t in chunk])
    if drop_command:
      ws = ws[:, :, :POLICY_DIM]
    x = torch.from_numpy(ws.astype(np.float32)).to(device)
    raw = model(x).cpu().numpy()
    out.append(decode(raw, norm))
  return np.concatenate(out, axis=0)


def per_env_mean(idx: np.ndarray, z_hat: np.ndarray, env_ids: np.ndarray) -> dict[int, np.ndarray]:
  d: dict[int, list] = {int(g): [] for g in env_ids}
  for (g, _t), z in zip(idx, z_hat):
    if int(g) in d:
      d[int(g)].append(z)
  return {g: np.mean(v, axis=0) for g, v in d.items() if v}


def partial_r(y_env: np.ndarray, x_env: np.ndarray, other_true: np.ndarray) -> float:
  """Residualize x and y on the other 4 true columns (linear reg + intercept), correlate
  the residuals. ADR-0011's mandated fix for Bar B's confounded-denominator failure."""
  A = np.concatenate([other_true, np.ones((len(other_true), 1))], axis=1)
  by, *_ = np.linalg.lstsq(A, y_env, rcond=None)
  bx, *_ = np.linalg.lstsq(A, x_env, rcond=None)
  ry = y_env - A @ by
  rx = x_env - A @ bx
  if ry.std() < 1e-9 or rx.std() < 1e-9:
    return float("nan")
  return float(np.corrcoef(rx, ry)[0, 1])


def column_metrics(z_hat_env: np.ndarray, e_true_env: np.ndarray) -> dict:
  """Per-column marginal R2/RMSE + partial-r, over one env-level array pair."""
  out = {}
  for i, name in enumerate(E_NAMES):
    y, x = e_true_env[:, i], z_hat_env[:, i]
    ss_res = np.sum((x - y) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    other = np.delete(e_true_env, i, axis=1)
    r_partial = partial_r(y, x, other)
    out[name] = {"r2": float(r2), "rmse": rmse, "r_partial": r_partial}
  return out


def eval_split(model, pooled: PooledRollouts, env_ids: np.ndarray, idx_all: np.ndarray,
               norm: dict, device: str, drop_command: bool) -> dict:
  """Per-regime + pooled-within-split metrics for one set of envs."""
  mask = np.isin(idx_all[:, 0], env_ids)
  idx = idx_all[mask]
  z_hat = predict_windows(model, pooled, idx, norm, device, drop_command)
  regimes = np.array([command_regime(pooled.window(int(g), int(t))) for g, t in idx])

  result = {}
  for regime in ("standing", "walking", "all"):
    sel = np.ones(len(idx), dtype=bool) if regime == "all" else (regimes == regime)
    if sel.sum() == 0:
      result[regime] = None
      continue
    per_env = per_env_mean(idx[sel], z_hat[sel], env_ids)
    envs = np.array(sorted(per_env))
    if len(envs) < 3:
      result[regime] = None
      continue
    z_env = np.stack([per_env[g] for g in envs])
    e_env = pooled.e_true_pooled[envs]
    result[regime] = {"n_envs": int(len(envs)), **column_metrics(z_env, e_env)}
  return result


def confound_check(model, pooled: PooledRollouts, env_ids: np.ndarray, idx_all: np.ndarray,
                    norm: dict, device: str, drop_command: bool) -> dict:
  """Grill item (d): within the SAME env (fixed true e), compare z_hat standing vs
  walking. Reported as a fraction of the between-env signal the gate relies on."""
  mask = np.isin(idx_all[:, 0], env_ids)
  idx = idx_all[mask]
  z_hat = predict_windows(model, pooled, idx, norm, device, drop_command)
  regimes = np.array([command_regime(pooled.window(int(g), int(t))) for g, t in idx])

  standing = per_env_mean(idx[regimes == "standing"], z_hat[regimes == "standing"], env_ids)
  walking = per_env_mean(idx[regimes == "walking"], z_hat[regimes == "walking"], env_ids)
  both = sorted(set(standing) & set(walking))
  if len(both) < 3:
    return {"n_envs_both_regimes": len(both), "note": "too few envs with both regimes"}

  shift = np.stack([walking[g] - standing[g] for g in both])  # [n, 5]
  between_env_std = np.stack([standing[g] for g in both]).std(axis=0)
  out = {"n_envs_both_regimes": len(both)}
  for i, name in enumerate(E_NAMES):
    within_shift_std = float(shift[:, i].std())
    out[name] = {
      "within_env_regime_shift_mean": float(shift[:, i].mean()),
      "within_env_regime_shift_std": within_shift_std,
      "between_env_signal_std": float(between_env_std[i]),
      "shift_over_signal_ratio": (within_shift_std / between_env_std[i]
                                   if between_env_std[i] > 1e-9 else float("nan")),
    }
  return out


def gate_verdict(metrics_i_all: dict) -> dict:
  return {
    name: bool(abs(metrics_i_all[name]["r_partial"]) >= 0.5 and metrics_i_all[name]["r2"] >= 0.3)
    for name in E_NAMES
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--ood-mass-dirs", nargs="*", default=[])
  ap.add_argument("--ood-lever-dirs", nargs="*", default=[])
  ap.add_argument("--out", required=True)
  ap.add_argument("--device", default=None)
  args = ap.parse_args()

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  model_dir = Path(args.model_dir)
  norm = json.loads((model_dir / "norm.json").read_text())
  parts_npz = np.load(model_dir / "partition.npz")
  parts = {k: parts_npz[k] for k in parts_npz}
  meta = json.loads((model_dir / "meta.json").read_text())
  dirs = [Path(d) for d in meta["dirs"]]

  pooled = PooledRollouts(dirs)
  idx_all = pooled_indices_from_dirs(dirs, pooled.offsets)

  report: dict = {"gate_columns": list(E_NAMES)}
  for tag, drop_command, d_in in (("real", False, POLICY_DIM + COMMAND_DIM),
                                   ("ablation", True, POLICY_DIM)):
    model = load_model(model_dir / f"phi_{tag}.pt", d_in, device)
    rep: dict = {}
    rep["split_i_held_out_envs"] = eval_split(
      model, pooled, parts["test"], idx_all, norm, device, drop_command)
    rep["split_ii_interp"] = eval_split(
      model, pooled, parts["interp"], idx_all, norm, device, drop_command)
    rep["gate"] = gate_verdict(rep["split_i_held_out_envs"]["all"])
    rep["confound_within_env"] = confound_check(
      model, pooled, parts["test"], idx_all, norm, device, drop_command)

    for split_name, ood_dirs in (("split_iii_ood_mass", args.ood_mass_dirs),
                                  ("split_iv_ood_lever", args.ood_lever_dirs)):
      if not ood_dirs:
        continue
      ood_pooled = PooledRollouts([Path(d) for d in ood_dirs])
      ood_idx = pooled_indices_from_dirs([Path(d) for d in ood_dirs], ood_pooled.offsets)
      ood_envs = np.arange(ood_pooled.num_envs)
      rep[split_name] = eval_split(
        model, ood_pooled, ood_envs, ood_idx, norm, device, drop_command)

    report[tag] = rep
    print(f"[eval_phi] {tag} gate: {rep['gate']}")

  Path(args.out).write_text(json.dumps(report, indent=2))
  print(f"[eval_phi] wrote {args.out}")


if __name__ == "__main__":
  main()
