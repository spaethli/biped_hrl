"""Off-policy replay buffer for the high-level TD3 learner (M4/M5).

One HL transition is ``(s_t, g_t, R, s_{t+c}, done)`` where ``s`` is the deployable HL
observation vector (proprio ++ command, 92-dim), ``g`` is the raw bounded goal action
(``tanh`` output in ``[-1, 1]^goal_dim``), ``R`` is the task reward summed over the
``c``-step window, and ``done`` is the bootstrap mask (true terminals only; truncations
such as ``fell_over`` are *not* marked done so the TD target bootstraps them).

A flat GPU ring buffer over all envs: each iteration pushes ``num_envs * (num_steps_per_env
// c)`` transitions. Off-policy, so it persists across iterations (unlike the PPO HL's
``RolloutStorage``, which clears each iter).

Relabeling (M5, HIRO off-policy correction) needs the per-window LL trace alongside the
5-tuple: the proprio (``policy``) obs sequence, the goal-space state sequence, the LL
action sequence (all ``[c, ·]``), plus the per-window goal ``scale`` and the achieved
end-of-window goal state ``s_{t+c}``. These are stored only when ``relabel=True`` and the
sequence tensors are allocated lazily on the first ``add`` (so the buffer stays dim-free).
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class HLBatch(NamedTuple):
  """A sampled minibatch of HL transitions (all tensors are [B, ...]).

  The trailing fields are populated only for a relabeling buffer; otherwise ``None``."""

  states: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor  # [B]
  next_states: torch.Tensor
  dones: torch.Tensor  # [B], 1.0 = true terminal (no bootstrap)
  policy_seq: torch.Tensor | None = None  # [B, c, policy_dim]
  goal_state_seq: torch.Tensor | None = None  # [B, c, goal_dim]
  action_seq: torch.Tensor | None = None  # [B, c, action_dim]
  scale: torch.Tensor | None = None  # [B, goal_dim]
  next_goal_state: torch.Tensor | None = None  # [B, goal_dim] = s_{t+c}
  center: torch.Tensor | None = None  # [B, goal_dim] absolute-target center (absolute mode)


class HLReplayBuffer:
  """Flat GPU ring buffer of HL transitions, added in batches of ``num_envs``."""

  def __init__(
    self, capacity: int, state_dim: int, action_dim: int, device: str, relabel: bool = False
  ) -> None:
    self.capacity = capacity
    self.device = device
    self.relabel = relabel
    self.states = torch.zeros(capacity, state_dim, device=device)
    self.actions = torch.zeros(capacity, action_dim, device=device)
    self.rewards = torch.zeros(capacity, device=device)
    self.next_states = torch.zeros(capacity, state_dim, device=device)
    self.dones = torch.zeros(capacity, device=device)
    # Relabel-only sequence storage; allocated lazily on the first add (needs c + dims).
    self.policy_seq: torch.Tensor | None = None
    self.goal_state_seq: torch.Tensor | None = None
    self.action_seq: torch.Tensor | None = None
    self.scale: torch.Tensor | None = None
    self.next_goal_state: torch.Tensor | None = None
    self.center: torch.Tensor | None = None
    self._ptr = 0
    self._size = 0

  def __len__(self) -> int:
    return self._size

  def _alloc_relabel(self, policy_seq, goal_state_seq, action_seq, scale) -> None:
    c = policy_seq.shape[1]
    self.policy_seq = torch.zeros(self.capacity, c, policy_seq.shape[-1], device=self.device)
    self.goal_state_seq = torch.zeros(self.capacity, c, goal_state_seq.shape[-1], device=self.device)
    self.action_seq = torch.zeros(self.capacity, c, action_seq.shape[-1], device=self.device)
    self.scale = torch.zeros(self.capacity, scale.shape[-1], device=self.device)
    self.next_goal_state = torch.zeros(self.capacity, goal_state_seq.shape[-1], device=self.device)
    self.center = torch.zeros(self.capacity, scale.shape[-1], device=self.device)

  def add(
    self,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    policy_seq: torch.Tensor | None = None,
    goal_state_seq: torch.Tensor | None = None,
    action_seq: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    next_goal_state: torch.Tensor | None = None,
    center: torch.Tensor | None = None,
  ) -> None:
    """Push a batch of ``N`` transitions (wraps around the ring)."""
    n = states.shape[0]
    idx = (torch.arange(n, device=self.device) + self._ptr) % self.capacity
    self.states[idx] = states
    self.actions[idx] = actions
    self.rewards[idx] = rewards
    self.next_states[idx] = next_states
    self.dones[idx] = dones
    if self.relabel:
      if self.policy_seq is None:
        self._alloc_relabel(policy_seq, goal_state_seq, action_seq, scale)
      self.policy_seq[idx] = policy_seq
      self.goal_state_seq[idx] = goal_state_seq
      self.action_seq[idx] = action_seq
      self.scale[idx] = scale
      self.next_goal_state[idx] = next_goal_state
      self.center[idx] = center
    self._ptr = (self._ptr + n) % self.capacity
    self._size = min(self._size + n, self.capacity)

  def sample(self, batch_size: int) -> HLBatch:
    i = torch.randint(0, self._size, (batch_size,), device=self.device)
    if not self.relabel:
      return HLBatch(
        self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i]
      )
    return HLBatch(
      self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i],
      self.policy_seq[i], self.goal_state_seq[i], self.action_seq[i], self.scale[i],
      self.next_goal_state[i], self.center[i],
    )
