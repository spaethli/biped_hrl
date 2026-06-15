"""Off-policy replay buffer for the high-level TD3 learner (M4/M5).

One HL transition is ``(s_t, g_t, R, s_{t+c}, done)`` where ``s`` is the deployable HL
observation vector (proprio ++ command, 92-dim), ``g`` is the raw bounded goal action
(``tanh`` output in ``[-1, 1]^goal_dim``), ``R`` is the task reward summed over the
``c``-step window, and ``done`` is the bootstrap mask (true terminals only; truncations
such as ``fell_over`` are *not* marked done so the TD target bootstraps them).

A flat GPU ring buffer over all envs: each iteration pushes ``num_envs * (num_steps_per_env
// c)`` transitions. Off-policy, so it persists across iterations (unlike the PPO HL's
``RolloutStorage``, which clears each iter). Relabeling (M5) will store the per-window
``(s_i, a_i)`` sequence alongside; for ``relabel=none`` only the 5-tuple is needed.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class HLBatch(NamedTuple):
  """A sampled minibatch of HL transitions (all tensors are [B, ...])."""

  states: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor  # [B]
  next_states: torch.Tensor
  dones: torch.Tensor  # [B], 1.0 = true terminal (no bootstrap)


class HLReplayBuffer:
  """Flat GPU ring buffer of HL transitions, added in batches of ``num_envs``."""

  def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str) -> None:
    self.capacity = capacity
    self.device = device
    self.states = torch.zeros(capacity, state_dim, device=device)
    self.actions = torch.zeros(capacity, action_dim, device=device)
    self.rewards = torch.zeros(capacity, device=device)
    self.next_states = torch.zeros(capacity, state_dim, device=device)
    self.dones = torch.zeros(capacity, device=device)
    self._ptr = 0
    self._size = 0

  def __len__(self) -> int:
    return self._size

  def add(
    self,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
  ) -> None:
    """Push a batch of ``N`` transitions (wraps around the ring)."""
    n = states.shape[0]
    idx = (torch.arange(n, device=self.device) + self._ptr) % self.capacity
    self.states[idx] = states
    self.actions[idx] = actions
    self.rewards[idx] = rewards
    self.next_states[idx] = next_states
    self.dones[idx] = dones
    self._ptr = (self._ptr + n) % self.capacity
    self._size = min(self._size + n, self.capacity)

  def sample(self, batch_size: int) -> HLBatch:
    i = torch.randint(0, self._size, (batch_size,), device=self.device)
    return HLBatch(
      self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i]
    )
