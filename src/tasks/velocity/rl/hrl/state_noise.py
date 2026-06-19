"""Estimator-noise model for the A1 goal/observation channel (#8b — train on a
deploy-realistic base-velocity estimate instead of sim ground-truth so the policy is
robust to the real onboard estimator, ``rt/sportmodestate``).

Corrupts ONLY the goal-space columns the estimator supplies — base linear velocity
(``vx, vy``), and optionally ``height``. Yaw-rate and orientation come from the
gyro/IMU and are left clean. The reward and the high level always see ground-truth
(privileged); only the low level's observed goal delta ``V*-s`` is noised.

Three composable, per-axis components (each independently switchable via its magnitude):
  * **bias**  — constant offset per episode, resampled on reset ~ U(-r, +r);
  * **drift** — slow OU random walk within an episode;
  * **lag**   — first-order low-pass (sensor/estimator delay).

Disabled (``enable=False`` or no corruptible columns) -> a transparent passthrough, so
runs without the flag are byte-identical to the ground-truth pipeline.
"""

from __future__ import annotations

import math

import torch


def _noisy_cols(goal_space, components: tuple[str, ...]) -> list[int]:
  """Goal-space column indices the estimator supplies. ``velocity`` -> ``vx, vy``
  only (yaw-rate, the 3rd velocity dim, is a clean gyro read); others -> all dims."""
  cols: list[int] = []
  i = 0
  for c in goal_space.components:
    if c.name in components:
      cols += [i, i + 1] if c.name == "velocity" else list(range(i, i + c.dim))
    i += c.dim
  return cols


class GoalStateNoise:
  """Stateful per-env estimator-noise applied to selected goal-space columns."""

  def __init__(self, cfg: dict, goal_space, num_envs: int, device: str) -> None:
    cfg = cfg or {}
    self.cols = _noisy_cols(goal_space, tuple(cfg.get("components", ("velocity",))))
    self.enable = bool(cfg.get("enable", False)) and len(self.cols) > 0
    if not self.enable:
      return
    self.device = device
    self.bias_range = float(cfg.get("bias_range", 0.0))
    self.drift_std = float(cfg.get("drift_std", 0.0))
    self.drift_decay = float(cfg.get("drift_decay", 0.99))
    lag_steps = float(cfg.get("lag_steps", 0.0))
    # 1st-order low-pass coefficient alpha = exp(-1/tau); 0 lag -> passthrough.
    self.lag_alpha = math.exp(-1.0 / lag_steps) if lag_steps > 0 else 0.0
    self.cols_t = torch.as_tensor(self.cols, device=device, dtype=torch.long)
    self.bias = self._sample_bias(num_envs)
    self.drift = torch.zeros(num_envs, len(self.cols), device=device)
    self.lag_state: torch.Tensor | None = None  # seeded lazily on first call

  def _sample_bias(self, n_rows: int) -> torch.Tensor:
    u = torch.rand(n_rows, len(self.cols), device=self.device)
    return (u * 2.0 - 1.0) * self.bias_range

  def __call__(self, state: torch.Tensor, dones: torch.Tensor | None) -> torch.Tensor:
    """Return ``state`` with the estimator-supplied columns corrupted. ``dones`` marks
    envs whose episode ended since the last call (resample bias / reset drift+lag)."""
    if not self.enable:
      return state
    sub = state.index_select(1, self.cols_t)  # [N, k] clean values on noisy columns
    if dones is not None and dones.any():
      d = dones.to(torch.bool)
      self.bias[d] = self._sample_bias(int(d.sum()))
      self.drift[d] = 0.0
      if self.lag_state is not None:
        self.lag_state[d] = sub[d]
    if self.lag_alpha > 0.0:
      if self.lag_state is None:
        self.lag_state = sub.clone()
      self.lag_state = self.lag_alpha * self.lag_state + (1.0 - self.lag_alpha) * sub
      filt = self.lag_state
    else:
      filt = sub
    if self.drift_std > 0.0:
      self.drift = self.drift_decay * self.drift + torch.randn_like(self.drift) * self.drift_std
    noisy = filt + self.bias + self.drift
    return state.clone().index_copy_(1, self.cols_t, noisy)
