"""Rebuilding what a checkpoint was trained with (src/tasks/velocity/rl/hrl/checkpoint_structure.py).

Every replay script builds its runner from the task's CURRENT defaults, then loads weights trained
under older ones. These two steps bridge the gap: a key missed in the restore crashes the load or
builds the wrong high level (onnx_parity's hand copy lost `hl_obs_e` and could not load an H-adapt
checkpoint), and a goal scale derived from the replay env instead of the training ranges inerts
the whole goal channel (the 2026-07-15 phantom "HL hold degeneracy").
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import yaml

from src.tasks.velocity.rl.hrl.checkpoint_structure import pin_trained_goal_scale, restore_run_structure
from src.tasks.velocity.rl.hrl.goal_space import build_goal_space, goal_dim


@dataclass
class Td3Cfg:
  tau: float = 0.005
  actor_lr: float = 3e-4


@dataclass
class AgentCfg:
  """The structure fields the restore touches, at today's defaults."""
  c: int = 8
  goal_components: tuple = ("velocity", "orientation", "height")
  hl_algorithm: str = "td3"
  hl_obs_vel: bool = True
  hl_obs_e: bool = False
  hl_velocity_goals_only: bool = True
  hl_td3: Td3Cfg = field(default_factory=Td3Cfg)


def _env_cfg():
  return SimpleNamespace(observations={"goal": SimpleNamespace(terms={"goal": SimpleNamespace(params={"dim": -1})})})


def _ckpt(tmp_path, saved: dict | None):
  run = tmp_path / "run"
  (run / "params").mkdir(parents=True)
  if saved is not None:
    (run / "params" / "agent.yaml").write_text(yaml.dump(saved))
  return run / "model_100.pt"


def test_restores_every_structure_key_including_hl_obs_e(tmp_path):
  ckpt = _ckpt(tmp_path, {"c": 4, "goal_components": ["velocity"], "hl_algorithm": "ppo",
                          "hl_obs_vel": True, "hl_obs_e": True, "hl_velocity_goals_only": True})
  cfg, env_cfg = AgentCfg(), _env_cfg()
  restore_run_structure(cfg, env_cfg, ckpt)
  assert (cfg.c, cfg.hl_algorithm, cfg.hl_obs_e) == (4, "ppo", True)
  assert list(cfg.goal_components) == ["velocity"]
  assert env_cfg.observations["goal"].terms["goal"].params["dim"] == goal_dim(("velocity",))


def test_absent_flags_restore_false_where_the_default_flipped(tmp_path):
  """hl_velocity_goals_only and hl_obs_vel defaulted to True AFTER old checkpoints trained
  without them; absence must mean False. Every other absent key keeps the default."""
  ckpt = _ckpt(tmp_path, {"hl_algorithm": "td3"})
  cfg = AgentCfg()
  restore_run_structure(cfg, _env_cfg(), ckpt)
  assert cfg.hl_velocity_goals_only is False and cfg.hl_obs_vel is False
  assert cfg.c == 8 and cfg.hl_obs_e is False


def test_a_partial_nested_cfg_merges_over_its_defaults(tmp_path):
  """agent.yaml is dumped after construction pops keys, so a nested cfg can be partial."""
  ckpt = _ckpt(tmp_path, {"hl_td3": {"tau": 0.01}})
  cfg = AgentCfg()
  restore_run_structure(cfg, _env_cfg(), ckpt)
  assert cfg.hl_td3 == {"tau": 0.01, "actor_lr": 3e-4}


def test_no_params_file_leaves_the_cfg_alone(tmp_path):
  cfg, env_cfg = AgentCfg(), _env_cfg()
  restore_run_structure(cfg, env_cfg, _ckpt(tmp_path, None))
  assert cfg == AgentCfg() and env_cfg.observations["goal"].terms["goal"].params["dim"] == -1


def _train_cfg():
  """Non-play cfg: the base twist range plus a curriculum stage that widens vx to 1.5."""
  ranges = SimpleNamespace(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0))
  stage = SimpleNamespace(params={"velocity_stages": [{"step": 0}, {"lin_vel_x": (-1.5, 1.5)}]})
  return SimpleNamespace(commands={"twist": SimpleNamespace(ranges=ranges)}, curriculum={"vel": stage})


def test_goal_scale_shim_pins_the_trained_scale_and_restores_the_live_ranges(env):
  """Replay under --eval-cmd-vx 0.5 (twist collapsed to a point): the scale must come from
  the TRAINING ranges with the curriculum applied, and the live ranges must survive."""
  runner = SimpleNamespace(goal_space=build_goal_space())
  env.set_twist_ranges(lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
  assert pin_trained_goal_scale(runner, SimpleNamespace(unwrapped=env), _train_cfg) is True
  assert runner.goal_space.scale_frozen
  assert runner.goal_space.scale(env)[0].item() == pytest.approx(1.5)
  live = env.command_manager.get_term("twist").cfg.ranges
  assert (live.lin_vel_x, live.lin_vel_y, live.ang_vel_z) == ((0.5, 0.5), (0.0, 0.0), (0.0, 0.0))


def test_goal_scale_shim_is_a_noop_for_a_baked_scale_or_a_flat_runner(env):
  def _boom():
    raise AssertionError("the training cfg must not be loaded when the shim does not fire")
  runner = SimpleNamespace(goal_space=build_goal_space())
  runner.goal_space.freeze_scale(runner.goal_space.scale(env).clone())
  assert pin_trained_goal_scale(runner, SimpleNamespace(unwrapped=env), _boom) is False
  assert pin_trained_goal_scale(SimpleNamespace(), SimpleNamespace(unwrapped=env), _boom) is False
