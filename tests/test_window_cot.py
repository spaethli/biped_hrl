"""Per-window CoT denominator for the HL reward (`hrl_runner.window_cot`, ADR-0004 S1c).

These pin the *incentive*, not the arithmetic. `CoT = energy / walked distance` is a ratio
whose denominator the HL controls, so without a cap the cheapest way to lower it is to walk
FURTHER than commanded. Measured at `hl_cot_coef=5`: signed overshoot +0.152 m/s at cmd 0.25
(61%), `hl_err_vx` 0.198 > `ll_err_vx` 0.117. `hl_cot_cap_commanded` closes that direction,
and the tests below are written so a mis-signed or mis-placed clamp fails them.
"""

import torch

from src.tasks.velocity.rl.hrl.hrl_runner import window_cot

D_FLOOR = 0.1 * 8 * 0.02  # c=8, step_dt=0.02 -> 0.016 m
MG = 75.0 * 9.81


def _t(x):
  return torch.tensor([x], dtype=torch.float32)


def test_cap_off_is_the_historical_formula():
  """Default must stay byte-identical: 40 runs and every published number assume it."""
  e, d, cmd = _t(30.0), _t(0.40), _t(0.25)
  got = window_cot(e, d, cmd, D_FLOOR, cap_commanded=False)
  assert torch.allclose(got, e / (d * MG))


def test_cap_off_lets_overshoot_pay():
  """The defect being fixed, pinned so the fix is demonstrably doing something."""
  e_on, e_over = _t(30.0), _t(33.0)          # overshooting costs MORE energy
  cmd = _t(0.25)
  on_cmd = window_cot(e_on, _t(0.25), cmd, D_FLOOR, cap_commanded=False)
  over = window_cot(e_over, _t(0.40), cmd, D_FLOOR, cap_commanded=False)
  assert over < on_cmd, "uncapped: walking further than asked lowers CoT"


def test_cap_on_makes_overshoot_strictly_worse():
  """With the cap, extra ground earns no credit, so the extra energy is pure cost."""
  e_on, e_over = _t(30.0), _t(33.0)
  cmd = _t(0.25)
  on_cmd = window_cot(e_on, _t(0.25), cmd, D_FLOOR, cap_commanded=True)
  over = window_cot(e_over, _t(0.40), cmd, D_FLOOR, cap_commanded=True)
  assert over > on_cmd, "capped: overshoot must cost, never pay"


def test_cap_on_credits_nothing_beyond_the_command():
  """Denominator saturates at d_commanded: same energy, more distance, same CoT."""
  e, cmd = _t(30.0), _t(0.25)
  a = window_cot(e, _t(0.25), cmd, D_FLOOR, cap_commanded=True)
  b = window_cot(e, _t(0.90), cmd, D_FLOOR, cap_commanded=True)
  assert torch.allclose(a, b)


def test_cap_does_not_touch_undershoot():
  """Going nowhere under command must stay expensive - that is what d_floor is for."""
  e, cmd = _t(30.0), _t(0.25)
  for d in (0.05, 0.12, 0.24):
    off = window_cot(e, _t(d), cmd, D_FLOOR, cap_commanded=False)
    on = window_cot(e, _t(d), cmd, D_FLOOR, cap_commanded=True)
    assert torch.allclose(off, on), f"cap changed an undershoot at d={d}"


def test_floor_still_bounds_a_stuck_robot_under_the_cap():
  """A stalled window is costly but FINITE, with the cap on and a tiny command."""
  got = window_cot(_t(30.0), _t(0.0), _t(0.0), D_FLOOR, cap_commanded=True)
  assert torch.isfinite(got).all()
  assert torch.allclose(got, _t(30.0) / (_t(D_FLOOR) * MG))


def test_standing_window_is_exactly_zero_either_way():
  """Speed-gated windows accumulate no energy -> the penalty must vanish, not divide-by-0."""
  for cap in (False, True):
    got = window_cot(_t(0.0), _t(0.0), _t(0.0), D_FLOOR, cap_commanded=cap)
    assert torch.allclose(got, _t(0.0))


def test_cap_is_elementwise_across_envs():
  """Batched: one env overshoots, one undershoots - the cap must apply per env."""
  e = torch.tensor([30.0, 30.0])
  d = torch.tensor([0.40, 0.10])   # over, under
  cmd = torch.tensor([0.25, 0.25])
  got = window_cot(e, d, cmd, D_FLOOR, cap_commanded=True)
  assert torch.allclose(got[0], e[0] / (cmd[0] * MG))   # capped at commanded
  assert torch.allclose(got[1], e[1] / (d[1] * MG))     # untouched
