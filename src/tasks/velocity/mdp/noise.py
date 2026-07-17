"""WL-D arm 9 (2026-07-17): correlated (low-pass) observation noise.

CorelLab's h12_rma reference low-passes its velocity-estimate noise (~2Hz cutoff,
drift-like) instead of drawing i.i.d. white noise every control step (WL-E parameter
audit, ADR-0005 Amendment 2, transfer-gap suspect iii). ``LowPassNoiseModel`` is a
first-order EMA filter over the same i.i.d. innovations the existing ``Unoise`` terms
already draw, with the innovation rescaled so the filter's STATIONARY std matches the
original (unfiltered) amplitude - same noise magnitude, slower (drift-like) temporal
structure, instead of independent-every-step jitter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.utils.noise.noise_cfg import NoiseModelCfg
from mjlab.utils.noise.noise_model import NoiseModel


class LowPassNoiseModel(NoiseModel):
  """First-order low-pass (EMA) filter over i.i.d. innovations, amplitude-matched."""

  def __init__(self, cfg: "LowPassNoiseModelCfg", num_envs: int, device: str):
    super().__init__(cfg, num_envs, device)
    alpha = 1.0 - math.exp(-2.0 * math.pi * cfg.cutoff_hz * cfg.dt)
    self._alpha = alpha
    # EMA x_t = (1-alpha) x_{t-1} + alpha w_t (w_t iid) has stationary
    # var_x = alpha^2 var_w / (1 - (1-alpha)^2); rescale w so var_x == var_w (the
    # raw noise_cfg's variance), i.e. same amplitude as the un-filtered term.
    self._innov_scale = math.sqrt(1.0 - (1.0 - alpha) ** 2) / alpha
    self._state: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._state is None:
      return
    indices = slice(None) if env_ids is None else env_ids
    self._state[indices] = 0.0

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    if self._state is None or self._state.shape != data.shape:
      self._state = torch.zeros_like(data)
    assert self._noise_model_cfg.noise_cfg is not None
    zeros = torch.zeros_like(data)
    innovation = self._noise_model_cfg.noise_cfg.apply(zeros) * self._innov_scale
    self._state = (1.0 - self._alpha) * self._state + self._alpha * innovation
    return data + self._state


@dataclass(kw_only=True)
class LowPassNoiseModelCfg(NoiseModelCfg, class_type=LowPassNoiseModel):
  """Config for LowPassNoiseModel. ``noise_cfg`` supplies the innovation distribution
  and amplitude (e.g. the same ``UniformNoiseCfg(n_min, n_max)`` the white-noise term
  used) - only the temporal structure changes, not the magnitude."""

  cutoff_hz: float = 2.0
  dt: float = 0.02
