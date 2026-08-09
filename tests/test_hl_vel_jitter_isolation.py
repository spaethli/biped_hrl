"""Isolation contract for the HL-only leg-odometry jitter (``HlVelJitter``,
``rl/hrl/state_noise.py``, 2026-08-05 probe).

The whole risk of this feature: ``HlVelJitter`` must corrupt ONLY ``obs["hl_vel"]``
(the HL's optional deployable base lin-vel input) and never touch ``state_n`` — the
value ``GoalStateNoise`` produces, which feeds the LL's observed goal delta
``V*-state_n``. A `delta`-mode LL cancels a *constant* corruption there (the existing
``noise_off`` carry-through), but NOT per-window jitter — leaking this into ``state_n``
would corrupt the one channel measured accurate on the bridge (vx 0.0139 / vy 0.0288)
and would silently bias the HL's TD3 replay buffer. Two failure shapes are checked:

1. A leak via shared/glue code: recomputing the LL's ``state_n``/``delta`` formula with
   an exercised (resampled, called) ``HlVelJitter`` sitting alongside it must be
   bit-identical to the same formula with the jitter absent.
2. A leak via in-place mutation: ``hrl_runner.py`` passes ``state_n[:, 0:2]`` — a VIEW,
   not a copy — into ``HlVelJitter.__call__``. An in-place op there would silently
   corrupt ``state_n`` itself. ``GoalStateNoise`` already avoids this
   (``state.clone().index_copy_``); ``HlVelJitter`` must too.
"""

import torch

from src.tasks.velocity.rl.hrl.goal_space import build_goal_space
from src.tasks.velocity.rl.hrl.state_noise import GoalStateNoise, HlVelJitter


def _run_ll_formula(jitter_cfg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Reproduces the exact per-step formula ``HierarchicalRunner.learn()`` uses to
  build the LL's ``state_n``/``delta`` and the HL's ``hl_vel`` obs at a fire step
  (hrl_runner.py ~543-559), using the REAL production classes. The HL jitter is fully
  exercised (resampled once, called at both the fire and post-step sites) so any
  cross-talk through shared/aliased state would show up in state_n/delta.
  """
  goal_space = build_goal_space(("velocity",), None)
  torch.manual_seed(0)
  state = torch.randn(4, goal_space.dim)
  target = torch.randn(4, goal_space.dim)

  state_noise = GoalStateNoise(
    {"enable": True, "components": ("velocity",), "bias_range": 0.1, "drift_std": 0.02},
    goal_space, num_envs=4, device="cpu",
  )
  jitter = HlVelJitter(jitter_cfg, num_envs=4, device="cpu")

  state_n = state_noise(state, None)
  noise_off = state_n - state
  target_obs = target + noise_off  # delta-mode's faithful-DR carry-through
  jitter.resample()  # fire-time resample
  hl_vel_fire = jitter(state_n[:, 0:2])
  delta = target_obs - state_n
  # Post-step refresh (hrl_runner.py ~779): same sample, no resample() in between.
  achieved = state + 0.01
  hl_vel_post = jitter((achieved + noise_off)[:, 0:2])

  return state_n, delta, torch.cat([hl_vel_fire, hl_vel_post], dim=0)


# --- 1. glue-level isolation --------------------------------------------------------


def test_ll_state_n_is_bit_identical_regardless_of_hl_vel_jitter():
  state_n_off, delta_off, hl_vel_off = _run_ll_formula({"enable": False})
  state_n_on, delta_on, hl_vel_on = _run_ll_formula(
    {"enable": True, "bias_vx": -0.038, "bias_vy": 0.0,
     "jitter_std_vx": 0.074, "jitter_std_vy": 0.049}
  )

  assert torch.equal(state_n_off, state_n_on), "HL jitter must never reach state_n"
  assert torch.equal(delta_off, delta_on), "HL jitter must never reach the LL's observed delta"
  assert not torch.equal(hl_vel_off, hl_vel_on), (
    "sanity: the two jitter configs must actually differ, or this test proves nothing"
  )


def test_hl_vel_jitter_disabled_is_a_transparent_passthrough():
  jitter = HlVelJitter({"enable": False}, num_envs=4, device="cpu")
  v = torch.randn(4, 2)
  jitter.resample()
  assert torch.equal(jitter(v), v)


def test_hl_vel_jitter_resamples_only_on_fire_not_every_call():
  jitter = HlVelJitter(
    {"enable": True, "jitter_std_vx": 1.0, "jitter_std_vy": 1.0}, num_envs=4, device="cpu"
  )
  v = torch.zeros(4, 2)
  jitter.resample()
  a = jitter(v)
  b = jitter(v)  # same window, no resample() in between
  assert torch.equal(a, b), "obs['hl_vel'] must be stable within one HL window"
  jitter.resample()
  c = jitter(v)
  assert not torch.equal(a, c), "resample() must actually draw a new sample"


def test_hl_vel_jitter_bias_is_constant_across_resamples():
  jitter = HlVelJitter(
    {"enable": True, "bias_vx": -0.038, "bias_vy": 0.0, "jitter_std_vx": 0.0,
     "jitter_std_vy": 0.0}, num_envs=4, device="cpu",
  )
  v = torch.zeros(4, 2)
  jitter.resample()
  a = jitter(v)
  jitter.resample()
  b = jitter(v)
  assert torch.equal(a, b), "bias is systematic, not resampled per window"
  assert torch.allclose(a[:, 0], torch.full((4,), -0.038))
  assert torch.allclose(a[:, 1], torch.zeros(4))


# --- 2. in-place-mutation isolation --------------------------------------------------


def test_hl_vel_jitter_does_not_mutate_its_input_view():
  """``hrl_runner.py`` passes ``state_n[:, 0:2]`` (a VIEW) into ``HlVelJitter``. An
  in-place op inside ``__call__`` would silently corrupt ``state_n`` -- and therefore
  the LL's delta -- through that shared storage."""
  state_n = torch.randn(4, 7)
  original = state_n.clone()
  jitter = HlVelJitter(
    {"enable": True, "bias_vx": -0.038, "jitter_std_vx": 0.074, "jitter_std_vy": 0.049},
    num_envs=4, device="cpu",
  )
  jitter.resample()

  hl_vel = jitter(state_n[:, 0:2])

  assert torch.equal(state_n, original), "HlVelJitter mutated its input view in place"
  assert not torch.equal(hl_vel, state_n[:, 0:2]), "sanity: jitter must have changed the value"
