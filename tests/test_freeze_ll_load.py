"""WP5 (2026-08-31): freezing an LL must survive a WIDENED critic obs group.

WP2 added the privileged ``env_latent_e`` term unconditionally to the ``critic``
observation group, taking h1_2's critic obs 114 -> 119. Every A1 checkpoint on disk
predates that, so ``_load_frozen_ll``'s strict critic load crashed WP5 Phase 1 before
the first iteration -- the LL is frozen, so the critic it refuses to load is a tensor
the run never reads.

The seam is narrow and easy to over-fix: skipping the critic is correct, skipping the
ACTOR would silently freeze a randomly-initialised LL and every downstream number
(Bar B, Bar C) would be measured against noise while looking perfectly healthy. These
tests pin all three behaviours separately.
"""

from types import SimpleNamespace

import pytest
import torch

from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner


def _mlp(n_in: int, n_out: int = 1) -> torch.nn.Module:
  torch.manual_seed(0)
  return torch.nn.Sequential(torch.nn.Linear(n_in, 8), torch.nn.ELU(), torch.nn.Linear(8, n_out))


def _ckpt(tmp_path, actor: torch.nn.Module, critic: torch.nn.Module):
  p = tmp_path / "model.pt"
  torch.save({"actor_state_dict": actor.state_dict(),
              "critic_state_dict": critic.state_dict()}, p)
  return str(p)


def _fake_runner(actor: torch.nn.Module, critic: torch.nn.Module):
  return SimpleNamespace(device="cpu", alg=SimpleNamespace(actor=actor, critic=critic))


def test_frozen_ll_loads_actor_even_when_critic_obs_widened(tmp_path):
  """The WP5 Phase-1 case: 114-dim saved critic vs a 119-dim env critic must NOT block."""
  src_actor, src_critic = _mlp(96, 27), _mlp(114)
  dst_actor, dst_critic = _mlp(96, 27), _mlp(119)          # env critic widened by e (+5)
  with torch.no_grad():                                     # make the source distinguishable
    for p in src_actor.parameters():
      p.add_(1.0)

  HierarchicalRunner._load_frozen_ll(
    _fake_runner(dst_actor, dst_critic), _ckpt(tmp_path, src_actor, src_critic))

  # The actor is what the frozen LL actually runs: it must be the CHECKPOINT's weights.
  for got, want in zip(dst_actor.parameters(), src_actor.parameters()):
    assert torch.equal(got, want), "frozen LL actor was not loaded from the checkpoint"


def test_frozen_ll_still_raises_on_actor_mismatch(tmp_path):
  """The goal-space guard must stay strict -- a wrong-shape ACTOR is a real error."""
  src_actor, src_critic = _mlp(96, 27), _mlp(114)
  dst_actor, dst_critic = _mlp(103, 27), _mlp(114)          # wrong goal space -> wrong actor obs
  with pytest.raises(RuntimeError):
    HierarchicalRunner._load_frozen_ll(
      _fake_runner(dst_actor, dst_critic), _ckpt(tmp_path, src_actor, src_critic))


def test_frozen_ll_loads_critic_when_shapes_agree(tmp_path):
  """Don't over-fix: a COMPATIBLE critic must still be loaded, not blanket-skipped."""
  src_actor, src_critic = _mlp(96, 27), _mlp(114)
  dst_actor, dst_critic = _mlp(96, 27), _mlp(114)
  with torch.no_grad():
    for p in src_critic.parameters():
      p.add_(1.0)

  HierarchicalRunner._load_frozen_ll(
    _fake_runner(dst_actor, dst_critic), _ckpt(tmp_path, src_actor, src_critic))

  for got, want in zip(dst_critic.parameters(), src_critic.parameters()):
    assert torch.equal(got, want), "a shape-compatible LL critic should still be loaded"
