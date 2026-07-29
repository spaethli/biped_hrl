"""Warm-start / checkpoint contract (``src/tasks/velocity/rl/hrl/hrl_runner.py``).

A1's low-level actor drops A0's ``command`` observation and appends ``goal``. Because
``command`` sits MID-vector (actor cols 6:9), a plain leading-block copy would misalign
every proprio term after it — phase, joint_pos, joint_vel, actions — and the warm start
would silently degrade into noise rather than fail. The diagnostic that it worked is a
high iter-0 episode length, which is exactly the kind of signal that is easy to
misread, so the column mapping is pinned here instead.
"""

import pytest
import torch
from torch import nn

from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner

# A0 actor layout: 6 proprio cols, then the 3-wide command block, then 36 more proprio.
PRE, CMD_W, POST = 6, 3, 36
A0_IN = PRE + CMD_W + POST  # 45
GOAL_DIM = 7
A1_IN = PRE + POST + GOAL_DIM  # 49 — command dropped, goal appended


class _Normalizer(nn.Module):
  def __init__(self, dim: int, fill: float) -> None:
    super().__init__()
    for name in ("_mean", "_var", "_std"):
      self.register_buffer(name, torch.full((dim,), fill))


class _Actor(nn.Module):
  """Minimal stand-in with the state-dict keys ``_partial_load`` special-cases."""

  def __init__(self, in_dim: int, fill: float) -> None:
    super().__init__()
    self.mlp = nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU(), nn.Linear(8, 4))
    self.obs_normalizer = _Normalizer(in_dim, fill)
    with torch.no_grad():
      # Column j is marked with the value j, so a mis-mapped column is identifiable.
      self.mlp[0].weight.copy_(torch.arange(in_dim, dtype=torch.float).expand(8, in_dim))
      self.mlp[2].weight.fill_(fill)


def _source_cols():
  return HierarchicalRunner._source_cols(A0_IN, skip=(PRE, CMD_W))


# --- the column map ----------------------------------------------------------


def test_source_cols_drops_the_mid_vector_command_block():
  """The gap is interior, not trailing: cols before it are kept in place and everything
  after it shifts left by the command width."""
  cols = _source_cols()

  assert len(cols) == PRE + POST
  assert cols[:PRE] == list(range(PRE))
  assert cols[PRE] == PRE + CMD_W, "the column after the gap must be the first post-command one"
  assert cols[-1] == A0_IN - 1
  assert not set(cols) & set(range(PRE, PRE + CMD_W)), "command columns must not be copied"


def test_warm_start_maps_proprio_around_the_command_gap():
  """Every shared column lands where the A1 actor expects it. A naive contiguous copy
  would put A0's command weights into A1's phase/joint columns."""
  target = _Actor(A1_IN, fill=-1.0)
  source = _Actor(A0_IN, fill=99.0)

  HierarchicalRunner._partial_load(target, source.state_dict(), GOAL_DIM, _source_cols())

  got = target.mlp[0].weight[0]
  expected = torch.tensor(_source_cols(), dtype=torch.float)
  assert torch.equal(got[: PRE + POST], expected)
  # Concretely: A1 col 6 holds A0 col 9, not A0 col 6 (which was the command).
  assert got[PRE].item() == PRE + CMD_W


def test_warm_start_leaves_the_goal_columns_freshly_initialised():
  """The low level starts as the A0 walker that IGNORES the goal, then learns to use it.
  Goal columns must keep the target's init — not be zeroed, and not take A0 weights."""
  target = _Actor(A1_IN, fill=-1.0)
  fresh = target.mlp[0].weight.clone()

  HierarchicalRunner._partial_load(target, _Actor(A0_IN, fill=99.0).state_dict(),
                                   GOAL_DIM, _source_cols())

  assert torch.equal(target.mlp[0].weight[:, -GOAL_DIM:], fresh[:, -GOAL_DIM:])


def test_warm_start_maps_the_normalizer_stats_the_same_way():
  """Obs-normalizer stats are input-facing too — mapping the weights but not the
  normalizer would feed the copied policy mis-scaled observations."""
  target = _Actor(A1_IN, fill=-1.0)
  source = _Actor(A0_IN, fill=99.0)

  HierarchicalRunner._partial_load(target, source.state_dict(), GOAL_DIM, _source_cols())

  mean = target.obs_normalizer._mean
  assert torch.all(mean[: PRE + POST] == 99.0), "shared cols must come from A0"
  assert torch.all(mean[-GOAL_DIM:] == -1.0), "goal cols keep the fresh init"


def test_warm_start_copies_deeper_layers_verbatim():
  """Only the input face differs between A0 and A1; the rest of the walker transfers."""
  target = _Actor(A1_IN, fill=-1.0)
  source = _Actor(A0_IN, fill=99.0)

  HierarchicalRunner._partial_load(target, source.state_dict(), GOAL_DIM, _source_cols())

  assert torch.equal(target.mlp[2].weight, source.mlp[2].weight)


def test_warm_start_fails_loudly_on_a_column_width_mismatch():
  """A goal-space change alters the shared width. Better a raise at load than a policy
  quietly warm-started from misaligned columns."""
  target = _Actor(A1_IN, fill=-1.0)
  source = _Actor(A0_IN, fill=99.0)
  wrong = HierarchicalRunner._source_cols(A0_IN, skip=(PRE, CMD_W + 1))

  with pytest.raises(ValueError, match="Warm-start"):
    HierarchicalRunner._partial_load(target, source.state_dict(), GOAL_DIM, wrong)
