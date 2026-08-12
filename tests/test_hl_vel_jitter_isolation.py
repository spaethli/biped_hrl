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

Since WL-F (2026-08-09) ``obs["hl_vel"]`` has a second possible source, the simulated
leg-odometry estimator (``hl_vel_source='leg_odom'``). The contract is IDENTICAL and is
checked for it too at the bottom of this file: it feeds ``obs["hl_vel"]`` only, and it must
hand out storage the caller cannot use to corrupt the accumulator (or vice versa). The
jitter checks are kept, not replaced — the jitter arm is the comparator the new estimator
is scored against, so it stays live.
"""

import torch

from src.tasks.velocity.rl.hrl.goal_space import build_goal_space
from src.tasks.velocity.rl.hrl.leg_odom import LegOdometry
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


# --- 3. the leg-odometry source obeys the same contract (WL-F) -----------------------


class _StubRunner:
  """Just enough of ``HierarchicalRunner`` for ``_hl_vel``. Calling the REAL unbound
  method (rather than restating its two lines here) is the point: a change to the source
  selection or to the clone has to break this file."""

  def __init__(self, source: str, jitter: HlVelJitter, residual: dict | None = None) -> None:
    self.hl_vel_source = source
    self.hl_vel_jitter = jitter
    self.hl_vel_residual = HlVelJitter(residual or {"enable": False}, 4, "cpu")


class _StubEnv:
  def __init__(self, odom: LegOdometry) -> None:
    self.leg_odom = odom


def _hl_vel(source, odom, v_est, fire, jitter_cfg=None):
  from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner

  runner = _StubRunner(source, HlVelJitter(jitter_cfg or {"enable": False}, 4, "cpu"))
  return HierarchicalRunner._hl_vel(runner, _StubEnv(odom), v_est, fire=fire)


def _primed_odom(n=4):
  """An accumulator with one closed window, so ``value`` is non-trivial."""
  odom = LegOdometry(n, "cpu")
  grav = torch.tensor([[0.0, 0.0, -1.0]] * n)
  q = torch.zeros(n, 12)
  for i in range(4):
    odom.step(q + i * 0.01, 0.005, torch.full((n, 3), 0.2), grav)
  odom.fire()
  return odom


def test_leg_odom_source_never_reads_or_writes_the_goal_state():
  """``hl_vel_source='leg_odom'`` reads the robot's legs, not ``state_n``. The runner still
  hands it ``state_n``-derived tensors at both call sites, so a leak either way -- reading
  the goal state, or writing into the view -- would be invisible without this."""
  odom = _primed_odom()
  state_n = torch.randn(4, 7)
  original = state_n.clone()

  out = _hl_vel("leg_odom", odom, state_n, fire=False)

  assert torch.equal(state_n, original), "the leg-odom path mutated the goal-state view"
  assert torch.equal(out, odom.value), "obs['hl_vel'] must be the latched window value"
  # And the value must NOT come from state_n, however plausible it looks.
  assert not torch.equal(out, state_n[:, 0:2])


def test_leg_odom_value_is_held_across_the_window_and_only_moves_on_a_fire():
  """Mirrors the C++: ``hl_vel_lo_`` is written only at the fire (State_RLHRL.cpp:444), so
  one window's obs writes share one realization -- the same contract HlVelJitter has, and
  what keeps the TD3 buffer's ``next_s`` the value the actor conditioned on."""
  odom = _primed_odom()
  v = torch.zeros(4, 7)
  grav = torch.tensor([[0.0, 0.0, -1.0]] * 4)

  a = _hl_vel("leg_odom", odom, v, fire=False)
  odom.step(torch.full((4, 12), 0.05), 0.005, torch.full((4, 3), 0.3), grav)
  b = _hl_vel("leg_odom", odom, v, fire=False)
  assert torch.equal(a, b), "obs['hl_vel'] must be stable within one HL window"

  c = _hl_vel("leg_odom", odom, v, fire=True)
  assert not torch.equal(a, c), "the fire must close the window and latch a new value"


def test_leg_odom_output_does_not_alias_the_accumulator():
  """``obs['hl_vel']`` is stored in the obs dict and read again later; handing out the
  accumulator's own storage would let a downstream in-place op corrupt the estimator (and
  a later fire silently rewrite an obs the HL already acted on)."""
  odom = _primed_odom()
  out = _hl_vel("leg_odom", odom, torch.zeros(4, 7), fire=False)
  before = odom.value.clone()

  out += 5.0

  assert torch.equal(odom.value, before), "obs['hl_vel'] aliases the accumulator's storage"


def test_state_source_is_unchanged_by_the_wl_f_refactor():
  """The historical path must still be exactly ``hl_vel_jitter(v_est[:, 0:2])`` -- every
  pre-2026-08-09 comparator (A0 anchor, blind HL, jitter keeper) depends on it."""
  cfg = {"enable": True, "bias_vx": -0.038, "jitter_std_vx": 0.074, "jitter_std_vy": 0.049}
  v_est = torch.randn(4, 7)
  torch.manual_seed(7)
  jitter = HlVelJitter(cfg, 4, "cpu")
  jitter.resample()
  want = jitter(v_est[:, 0:2])

  torch.manual_seed(7)
  runner = _StubRunner("state", HlVelJitter(cfg, 4, "cpu"))
  runner.hl_vel_jitter.resample()
  from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner
  got = HierarchicalRunner._hl_vel(runner, _StubEnv(_primed_odom()), v_est, fire=True)

  assert torch.equal(got, want)


def test_residual_noise_adds_to_the_estimator_rather_than_replacing_it():
  """WL-F task 4. ``hl_vel_residual`` covers what sim does not reproduce (encoder noise,
  contact geometry, 200 Hz vs ~991 Hz differencing). It must ADD to the estimator's own
  policy-coupled value -- replacing it would throw away the endogeneity the whole workline
  exists to install, and would read identically in every aggregate metric."""
  from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner

  odom = _primed_odom()
  clean = _hl_vel("leg_odom", odom, torch.zeros(4, 7), fire=False)

  runner = _StubRunner("leg_odom", HlVelJitter({"enable": False}, 4, "cpu"),
                       residual={"enable": True, "bias_vx": 0.25, "bias_vy": -0.1})
  runner.hl_vel_residual.resample()
  noisy = HierarchicalRunner._hl_vel(runner, _StubEnv(odom), torch.zeros(4, 7), fire=False)

  assert torch.allclose(noisy - clean, torch.tensor([[0.25, -0.1]]).expand(4, 2)), (
    "the residual must be an offset ON the estimator, not a replacement for it"
  )
  assert torch.equal(odom.value, clean), "applying the residual mutated the accumulator"
