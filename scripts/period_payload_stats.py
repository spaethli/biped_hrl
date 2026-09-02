"""WP5 Bar B statistic: commanded stride period vs the true environment latent.

Kept in its own module, deliberately: ``play.py`` imports the whole mjlab env stack at
module scope, so a test that imported it would pull MuJoCo in and break the suite's
CPU-only contract. This module imports nothing but torch. Same reason
``deploy_gate_analyzer``/``deploy_provenance`` are importable and ``play.py`` is not.

Second caller is coming: Phase 2 scores the same statistic on the ESTIMATED latent.
"""

import torch


def period_payload_stats(per_w: torch.Tensor, keep_w: torch.Tensor,
                         e: torch.Tensor) -> dict:
  """Per-env mean commanded stride period vs the per-env true latent.

  ``e`` is drawn by ``mode="startup"`` DR, which fires at construction and NOT on reset,
  so each env's payload/CoM/friction is constant for the whole process. **The independent
  unit is therefore the ENV, not the window** -- pooling windows would inflate n by the
  window count (~150x here) and shrink every interval derived from it while leaving the
  point estimate roughly right, i.e. fail silently.

  ``per_w``/``keep_w`` are [W, B] (commanded T at each HL fire, and whether that window
  was free of a reset); ``e`` is [B, 5] = (payload, com_dx, com_dy, com_dz, friction).
  Correlations return NaN -- never 0.0 -- when a regressor has no variance, which is the
  fixed-payload probe case.
  """
  kf = keep_w.float()
  n_ok = kf.sum(0).clamp(min=1.0)
  env_T = (per_w * kf).sum(0) / n_ok                                    # [B] per-env mean T
  var_w = ((per_w - env_T) ** 2 * kf).sum(0) / n_ok
  ok = keep_w.sum(0) > 1                                                # envs with data
  payload, com, fric = e[:, 0], e[:, 1:4], e[:, 4]

  def _r(x, y):
    """Pearson r over envs with data; NaN when either side has no variance."""
    x, y = x[ok].double(), y[ok].double()
    xc, yc = x - x.mean(), y - y.mean()
    d = (xc.norm() * yc.norm()).item()
    return float("nan") if d < 1e-12 else round((xc @ yc).item() / d, 4)

  def _slope(x, y):
    x, y = x[ok].double(), y[ok].double()
    xc = x - x.mean()
    v = (xc @ xc).item()
    return float("nan") if v < 1e-12 else round(((xc @ (y - y.mean())).item()) / v, 6)

  Te = env_T[ok]
  return {
    "n_envs": int(ok.sum()), "windows_per_env": int(keep_w.shape[0]),
    "period_mean": round(Te.mean().item(), 5),
    # sd of the PER-ENV MEAN T -- the residual the Bar B correlation must beat.
    "period_sd_env": round(Te.std(unbiased=True).item(), 5),
    # typical within-env wander of T across windows (averaged over envs).
    "period_sd_win": round(var_w[ok].mean().sqrt().item(), 5),
    "period_p5": round(torch.quantile(Te, 0.05).item(), 5),
    "period_p95": round(torch.quantile(Te, 0.95).item(), 5),
    "period_min": round(Te.min().item(), 5),
    "period_max": round(Te.max().item(), 5),
    "payload_mean": round(payload[ok].mean().item(), 4),
    "payload_sd": round(payload[ok].std(unbiased=True).item(), 4),
    # Bar B statistic + per-component reads (friction/CoM also vary per env, so they are
    # part of the residual for a payload-only correlation).
    "r_period_payload": _r(payload, env_T),
    "slope_period_payload": _slope(payload, env_T),
    "r_period_friction": _r(fric, env_T),
    "r_period_comdx": _r(com[:, 0], env_T),
    "r_period_comdy": _r(com[:, 1], env_T),
    "r_period_comdz": _r(com[:, 2], env_T),
    # Per-env columns for every latent component: a payload-only correlation buries the
    # other four in its residual, and the 2026-09-01 two-point probe found com_dx (-0.71)
    # dominating payload (+0.12), so the caller needs the raw columns to partial them out.
    "per_env": {"payload": [round(v, 4) for v in payload[ok].tolist()],
                "period": [round(v, 5) for v in Te.tolist()],
                "friction": [round(v, 4) for v in fric[ok].tolist()],
                "com_dx": [round(v, 5) for v in com[ok, 0].tolist()],
                "com_dy": [round(v, 5) for v in com[ok, 1].tolist()],
                "com_dz": [round(v, 5) for v in com[ok, 2].tolist()]},
  }
