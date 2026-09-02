"""WP5 (2026-08-31): the Bar B statistic itself must be right.

``_period_payload_stats`` turns per-(window, env) commanded stride periods into the
number the thesis reports: corr(per-env mean commanded T, per-env true payload). A
defect here does not crash -- it returns a plausible r, and Bar B is adjudicated on
it. That is the "silently wrong result" class the WP5 brief names as poisoning WP6/WP7.

Three seams, each independently wrong-able:
1. the keep mask must actually exclude reset-contaminated windows (a mask that is
   ignored quietly averages in garbage);
2. the aggregation unit must be the ENV, not the window -- startup DR fixes e per env
   for the whole run, so per-window pooling inflates n by ~150x and any CI with it;
3. the Pearson math must be right, and must return NaN (not 0.0, not a divide-by-zero)
   when payload has no variance -- the fixed-payload probe case.
"""

import math

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from period_payload_stats import period_payload_stats  # noqa: E402


def _e(payload, friction=None, com_dx=None):
  """Build an [B, 5] latent e = (payload, com_dx, com_dy, com_dz, friction)."""
  n = len(payload)
  e = torch.zeros(n, 5)
  e[:, 0] = torch.tensor(payload, dtype=torch.float32)
  if com_dx is not None:
    e[:, 1] = torch.tensor(com_dx, dtype=torch.float32)
  e[:, 4] = torch.tensor(friction if friction is not None else [1.0] * n, dtype=torch.float32)
  return e


def test_perfect_payload_response_gives_r_one():
  """T linear in payload with zero residual -> r = -1 for a negative slope."""
  payload = [0.0, 4.0, 8.0, 12.0]
  periods = [0.60, 0.56, 0.52, 0.48]                     # slope -0.01 s/kg exactly
  per_w = torch.tensor([periods, periods, periods])      # [3 windows, 4 envs], no noise
  keep_w = torch.ones(3, 4, dtype=torch.bool)

  s = period_payload_stats(per_w, keep_w, _e(payload))
  assert math.isclose(s["r_period_payload"], -1.0, abs_tol=1e-3)
  assert math.isclose(s["slope_period_payload"], -0.01, abs_tol=1e-5)
  assert math.isclose(s["period_sd_env"], torch.tensor(periods).std().item(), rel_tol=1e-3)


def test_no_payload_variance_gives_nan_not_zero():
  """The fixed-payload probe: r is UNDEFINED, and must not be reported as 0.0."""
  per_w = torch.tensor([[0.35, 0.36, 0.37, 0.38]] * 3)
  keep_w = torch.ones(3, 4, dtype=torch.bool)

  s = period_payload_stats(per_w, keep_w, _e([6.0, 6.0, 6.0, 6.0]))
  assert math.isnan(s["r_period_payload"]), "constant payload must give NaN, not a number"
  assert math.isnan(s["slope_period_payload"])
  assert s["payload_sd"] == 0.0


def test_keep_mask_excludes_contaminated_windows():
  """A masked-out window must not move the per-env mean at all."""
  good = [0.50, 0.50, 0.50, 0.50]
  garbage = [9.9, 9.9, 9.9, 9.9]                          # reset-contaminated
  per_w = torch.tensor([good, garbage, good])
  keep_w = torch.tensor([[True] * 4, [False] * 4, [True] * 4])

  s = period_payload_stats(per_w, keep_w, _e([0.0, 4.0, 8.0, 12.0]))
  assert math.isclose(s["period_mean"], 0.50, abs_tol=1e-6), \
      "masked windows leaked into the per-env mean"


def test_aggregation_unit_is_the_env_not_the_window():
  """period_sd_env is the spread of PER-ENV MEANS, so within-env noise must average out.

  Two envs, wildly noisy per window, but with identical means: the between-env sd is 0
  even though the pooled per-window sd is large. Pooling by window would report ~0.1.
  """
  per_w = torch.tensor([[0.40, 0.60], [0.60, 0.40], [0.50, 0.50]])
  keep_w = torch.ones(3, 2, dtype=torch.bool)

  s = period_payload_stats(per_w, keep_w, _e([0.0, 12.0]))
  assert math.isclose(s["period_sd_env"], 0.0, abs_tol=1e-6), \
      "period_sd_env must be the sd of per-env MEANS, not of pooled windows"
  assert s["period_sd_win"] > 0.05, "within-env wander should still be reported"


def test_per_env_com_columns_carry_the_right_latent_slots():
  """The CoM columns must map to e[:,1:4] in order, and honour the `ok` mask.

  WP5 2026-09-01: the two-point probe found com_dx (r=-0.71) dominating payload
  (r=+0.12), so Bar B has to partial the CoM out -- which is only possible if these
  columns are exported AND correctly ordered. Mis-slicing e (say com_dz reading the
  friction column) would leave a plausible-looking regression that silently attributes
  the HL's strongest response to the wrong physical cause.
  """
  n = 4
  e = torch.zeros(n, 5)
  e[:, 0] = torch.tensor([0.0, 4.0, 8.0, 12.0])   # payload
  e[:, 1] = torch.tensor([-0.05, -0.01, 0.02, 0.05])  # com_dx
  e[:, 2] = torch.tensor([0.011, 0.022, 0.033, 0.044])  # com_dy
  e[:, 3] = torch.tensor([0.101, 0.102, 0.103, 0.104])  # com_dz
  e[:, 4] = torch.tensor([0.3, 0.7, 1.1, 1.6])    # friction
  per_w = torch.tensor([[0.40, 0.41, 0.42, 0.43]] * 6)
  keep_w = torch.ones(6, n, dtype=torch.bool)
  # env 2 has no usable window -> must be dropped from every per-env column alike.
  keep_w[:, 2] = False

  out = period_payload_stats(per_w, keep_w, e)
  pe = out["per_env"]
  assert pe["com_dx"] == [-0.05, -0.01, 0.05]
  assert pe["com_dy"] == [0.011, 0.022, 0.044]
  assert pe["com_dz"] == [0.101, 0.102, 0.104]
  # every column is masked to the same envs, so a regression stays row-aligned
  assert len({len(v) for v in pe.values()}) == 1
  assert pe["friction"] == [0.3, 0.7, 1.6]
  assert pe["payload"] == [0.0, 4.0, 12.0]
