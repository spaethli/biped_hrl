"""WP2 (2026-08-31): payload DR opt-in + privileged latent ``e`` plumbing.

Three seams, chosen because a defect there yields a confident WRONG RESULT (a
silently-mis-scaled latent, or a broken inertness guarantee) rather than a crash:

1. ``mdp.env_latent_e`` reads back the right per-env DELTA (payload/CoM) or absolute
   value (friction) from randomized sim state -- the single source of truth both the
   critic term and the HL channel depend on.
2. ``apply_payload_dr`` is opt-in: the base A0/A1 task registrations must NOT carry
   the ``base_mass`` event, or every existing checkpoint's RNG stream shifts (the
   WP2 acceptance gate).
3. ``HighLevelTd3``'s ``obs_e_dim`` wiring is byte-identical to the pre-WP2 shape
   when off, and actually appends ``obs["hl_e"]`` when on -- mirrors
   ``test_hl_vel_jitter_isolation.py``'s isolation style for ``obs_vel_dim``.
"""

from types import SimpleNamespace

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp.observations import env_latent_e


class _FakeIndexing:
  def __init__(self, body_ids: torch.Tensor, geom_ids: torch.Tensor) -> None:
    self.body_ids = body_ids
    self.geom_ids = geom_ids


class _FakeAsset:
  def __init__(self, indexing: _FakeIndexing) -> None:
    self.indexing = indexing


class _FakeSim:
  def __init__(self, model, defaults: dict) -> None:
    self.model = model
    self._defaults = defaults

  def get_default_field(self, field: str) -> torch.Tensor:
    return self._defaults[field]


def _make_env(body_mass, body_ipos, geom_friction, mass_default, ipos_default):
  # One robot entity, entity-local ids == global ids (identity map): 3 bodies, 2 geoms.
  indexing = _FakeIndexing(
    body_ids=torch.tensor([0, 1, 2]), geom_ids=torch.tensor([0, 1])
  )
  asset = _FakeAsset(indexing)
  model = SimpleNamespace(
    body_mass=body_mass, body_ipos=body_ipos, geom_friction=geom_friction
  )
  sim = _FakeSim(model, {"body_mass": mass_default, "body_ipos": ipos_default})
  return SimpleNamespace(scene={"robot": asset}, sim=sim)


def test_env_latent_e_reads_payload_and_com_as_deltas_and_friction_as_absolute():
  # torso = global body id 2; foot = global geom id 1. 3 envs: no DR, +4kg/+CoM shift,
  # +12kg/-z CoM shift. Friction is operation="abs" -- read straight, no default needed.
  mass_default = torch.tensor([10.0, 20.0, 5.0])
  body_mass = torch.tensor([[10.0, 20.0, 5.0], [10.0, 20.0, 9.0], [10.0, 20.0, 17.0]])
  ipos_default = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.02]])
  body_ipos = torch.tensor([
    [[0, 0, 0], [0, 0, 0], [0.01, 0.0, 0.02]],
    [[0, 0, 0], [0, 0, 0], [0.04, -0.01, 0.02]],
    [[0, 0, 0], [0, 0, 0], [0.01, 0.0, 0.00]],
  ], dtype=torch.float32)
  geom_friction = torch.tensor([
    [[0.3, 0, 0], [0.3, 0, 0]],
    [[0.9, 0, 0], [0.9, 0, 0]],
    [[1.6, 0, 0], [1.6, 0, 0]],
  ])
  env = _make_env(body_mass, body_ipos, geom_friction, mass_default, ipos_default)

  torso_cfg = SceneEntityCfg("robot")
  torso_cfg.body_ids = [2]
  foot_cfg = SceneEntityCfg("robot")
  foot_cfg.geom_ids = [1]

  e = env_latent_e(env, torso_cfg, foot_cfg)

  assert e.shape == (3, 5)
  expected = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.3],
    [4.0, 0.03, -0.01, 0.0, 0.9],
    [12.0, 0.0, 0.0, -0.02, 1.6],
  ])
  assert torch.allclose(e, expected, atol=1e-6)


def test_env_latent_e_is_zero_payload_and_zero_com_when_dr_is_disabled():
  """The no-DR case (payload DR off): body_mass/body_ipos equal their defaults, so e's
  first four columns must read exactly zero regardless of friction's (independent,
  always-on) DR."""
  mass_default = torch.tensor([10.0, 20.0, 5.0])
  body_mass = mass_default.unsqueeze(0).expand(2, -1).clone()
  ipos_default = torch.zeros(3, 3)
  body_ipos = ipos_default.unsqueeze(0).expand(2, -1, -1).clone()
  geom_friction = torch.zeros(2, 2, 3)
  geom_friction[:, 1, 0] = torch.tensor([0.5, 1.2])
  env = _make_env(body_mass, body_ipos, geom_friction, mass_default, ipos_default)

  torso_cfg = SceneEntityCfg("robot")
  torso_cfg.body_ids = [2]
  foot_cfg = SceneEntityCfg("robot")
  foot_cfg.geom_ids = [1]

  e = env_latent_e(env, torso_cfg, foot_cfg)
  assert torch.equal(e[:, :4], torch.zeros(2, 4))
  assert torch.allclose(e[:, 4], torch.tensor([0.5, 1.2]))


# --- apply_payload_dr: opt-in, base tasks untouched ----------------------------------


def test_apply_payload_dr_is_opt_in_not_on_the_base_tasks():
  """WP2 acceptance gate: the base A0/A1 task registrations must not carry the
  base_mass event, or dr.body_mass's RNG draw (even at a degenerate range) shifts the
  stream every later-registered event consumes, breaking the inertness proof."""
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  for task in ("Unitree-H1_2-Flat", "Unitree-H1_2-Flat-A1"):
    cfg = load_env_cfg(task)
    assert "base_mass" not in cfg.events, (
      f"{task}: base_mass event present on a task that never calls apply_payload_dr"
    )


def test_apply_payload_dr_adds_exactly_the_base_mass_event_at_the_given_range():
  from src.tasks.velocity.config.h1_2.env_cfgs import (
    apply_payload_dr,
    unitree_h1_2_flat_env_cfg,
  )

  cfg = unitree_h1_2_flat_env_cfg()
  assert "base_mass" not in cfg.events
  apply_payload_dr(cfg, ranges=(0.0, 12.0))
  assert cfg.events["base_mass"].params["ranges"] == (0.0, 12.0)
  assert cfg.events["base_mass"].params["asset_cfg"] is cfg.events["base_com"].params["asset_cfg"]


def test_payload_dr_task_variants_carry_the_event():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  for task in ("Unitree-H1_2-Flat-Payload", "Unitree-H1_2-Flat-A1-Payload"):
    cfg = load_env_cfg(task)
    assert "base_mass" in cfg.events
    assert cfg.events["base_mass"].params["ranges"] == (0.0, 12.0)


# --- HighLevelTd3 obs_e_dim wiring ----------------------------------------------------


def _td3_cfg() -> dict:
  net = {"hidden_dims": (8, 8), "activation": "elu"}
  return {
    "tau": 0.005, "policy_freq": 2, "expl_noise_std": 0.0, "target_noise_std": 0.2,
    "target_noise_clip": 0.5, "batch_size": 4, "n_grad_steps": 1,
    "learning_starts": 1, "warmup_transitions": 0, "buffer_capacity": 100,
    "actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3,
    "actor": net, "critic": net, "num_candidates": 2, "candidate_std": 0.1,
  }


def test_obs_e_dim_zero_is_byte_identical_to_the_pre_wp2_state_vec():
  from src.tasks.velocity.rl.hrl.goal_space import build_goal_space
  from src.tasks.velocity.rl.hrl.td3 import HighLevelTd3

  goal_space = build_goal_space(("velocity",), None)
  obs = {"policy": torch.zeros(4, 6), "command": torch.zeros(4, 3)}
  hl = HighLevelTd3(goal_space, obs, 4, goal_space.dim, 0.9, _td3_cfg(), "cpu")
  assert hl._state_dim == 9
  state = hl._state_vec(obs)
  assert torch.equal(state, torch.cat([obs["policy"], obs["command"]], dim=-1))


def test_obs_e_dim_appends_hl_e_and_changes_state_dim():
  from src.tasks.velocity.rl.hrl.goal_space import build_goal_space
  from src.tasks.velocity.rl.hrl.td3 import HighLevelTd3

  goal_space = build_goal_space(("velocity",), None)
  obs = {
    "policy": torch.zeros(4, 6), "command": torch.zeros(4, 3),
    "hl_e": torch.arange(20, dtype=torch.float32).reshape(4, 5),
  }
  hl = HighLevelTd3(
    goal_space, obs, 4, goal_space.dim, 0.9, _td3_cfg(), "cpu", obs_e_dim=5
  )
  assert hl._state_dim == 14
  state = hl._state_vec(obs)
  assert torch.equal(
    state, torch.cat([obs["policy"], obs["command"], obs["hl_e"]], dim=-1)
  )
