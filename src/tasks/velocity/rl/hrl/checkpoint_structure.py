"""Rebuild what a checkpoint was TRAINED with, for any script that replays one outside training.

Two steps at two fixed points, because the second needs what the first lets you build:

  restore_run_structure(agent_cfg, env_cfg, ckpt)   # BEFORE building the env and runner
  ...build env, runner; runner.load(ckpt)...
  pin_trained_goal_scale(runner, env, lambda: load_env_cfg(task_id))   # AFTER runner.load

Callers: scripts/play.py, scripts/onnx_parity.py, scripts/collect_phi_data.py. A hand copy of
the first step drifted once (onnx_parity lost `hl_obs_e` and could not load an H-adapt
checkpoint), which is why there is one copy, here.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import yaml


def _deep_merge(default: dict, override: dict) -> dict:
  """Nested dict merge: ``override`` wins where present, ``default`` fills the rest.

  Needed because the dumped agent.yaml is written AFTER runner construction, and
  model construction pops keys from nested cfg dicts in place (e.g. MLPModel pops
  ``distribution_cfg["class_name"]``) — so saved cfgs can be missing keys that a
  rebuild requires. Older runs may also predate newly added cfg fields.
  """
  out = dict(default)
  for k, v in override.items():
    if isinstance(v, dict) and isinstance(out.get(k), dict):
      out[k] = _deep_merge(out[k], v)
    else:
      out[k] = v
  return out


def restore_run_structure(agent_cfg, env_cfg, resume_path: Path) -> None:
  """Rebuild ``agent_cfg`` (+ ``env_cfg``'s goal dim) with the structure the checkpoint at
  ``resume_path`` was trained with, not the task's current defaults: a checkpoint trained
  with a different goal space crashes the load (dim mismatch), and a different
  ``hl_algorithm`` silently builds the wrong high level (e.g. oracle instead of the
  trained ppo/td3, whose state is then ignored). ``train.py`` dumps the launch config to
  ``params/agent.yaml``; restore the structure-determining keys from there. Keys absent
  from the yaml (older runs) or not on the cfg (e.g. A0) keep the defaults.
  """
  params_yaml = Path(resume_path).parent / "params" / "agent.yaml"
  if not params_yaml.exists():
    return
  saved = yaml.full_load(params_yaml.read_text())  # dump_yaml writes python/tuple tags
  structure_keys = ("c", "goal_components", "goal_weights", "hl_algorithm",
                    "hl_ppo", "hl_td3", "relabeling", "gamma_hi", "hl_target_mode",
                    "hl_obs_vel", "hl_obs_e",
                    # A1a cadence channel: without hl_cadence restored, eval rebuilt the
                    # runner with the channel OFF -> fixed 0.6 clock, --eval-cadence-period
                    # silently inert (the 2026-07-02 "no entrainment" false verdicts).
                    "hl_cadence", "cadence_period_range", "ll_cadence_coef",
                    "hl_cot_coef", "hl_cadence_source", "cadence_swing_time",
                    "cadence_duty_range", "hl_velocity_goals_only")
  restored = {k: saved[k] for k in structure_keys
              if k in saved and hasattr(agent_cfg, k)}
  # hl_velocity_goals_only defaulted to True on 2026-07-09: checkpoints saved
  # before the key existed trained full-goal HLs, so ABSENCE must restore False
  # (the "keys absent keep the defaults" rule would silently rebuild a task_dim
  # HL and crash/mis-load every pre-change TD3 checkpoint).
  if "hl_velocity_goals_only" not in saved and hasattr(agent_cfg, "hl_velocity_goals_only"):
    restored["hl_velocity_goals_only"] = False
  # Same absence rule for hl_obs_vel (defaulted to True 2026-07-10): checkpoints saved
  # before the key existed trained HLs without the +2 vel obs -> absence must restore
  # False or every pre-velobs TD3 checkpoint mis-builds 94-dim nets vs its saved 92.
  if "hl_obs_vel" not in saved and hasattr(agent_cfg, "hl_obs_vel"):
    restored["hl_obs_vel"] = False
  for k, v in restored.items():
    cur = getattr(agent_cfg, k)
    if isinstance(v, dict) and cur is not None and not isinstance(cur, dict):
      v = _deep_merge(asdict(cur), v)  # cur is a cfg dataclass (hl_ppo / hl_td3)
    setattr(agent_cfg, k, v)
  # The env's goal obs dim was baked from the default runner cfg at task
  # registration; re-derive it from the restored component list.
  if "goal_components" in restored and "goal" in env_cfg.observations:
    from src.tasks.velocity.rl.hrl.goal_space import goal_dim

    env_cfg.observations["goal"].terms["goal"].params["dim"] = goal_dim(
      tuple(restored["goal_components"])
    )
  if restored:
    print(f"[INFO]: Restored run structure from {params_yaml.name}: "
          f"{ {k: v for k, v in restored.items() if k in ('goal_components', 'hl_algorithm')} }")


def pin_trained_goal_scale(runner, env, load_train_env_cfg: Callable) -> bool:
  """Absence shim: checkpoints saved before 2026-07-16 carry no baked `goal_scale`
  (HierarchicalRunner.save), so pin the scale the run TRAINED with. Deriving it live
  instead reads THIS env's command ranges — play mode narrows ang_vel_z (yaw scale 0.5
  vs 1.0 trained) and `--eval-cmd-vx` collapses them to a point, which drops the derived
  scale to its 1e-3 floor and pins V* = s + 1e-3*g ≈ s, inerting the whole goal channel
  (the 2026-07-15 "HL hold degeneracy" artifact; see the A1 findings ledger, WL-C row).
  Training ranges = the task's non-play cfg with the twist curriculum's stages applied.

  ``load_train_env_cfg``: zero-arg loader of that non-play cfg (``lambda: load_env_cfg(task)``),
  called only when the shim fires. Returns whether it fired. A no-op for a flat runner or a
  checkpoint with a baked scale (``runner.load`` already froze it).
  """
  gsp = getattr(runner, "goal_space", None)
  if gsp is None or gsp.scale_frozen:
    return False
  train_cfg = load_train_env_cfg()  # non-play => the ranges training ran with
  train_ranges = train_cfg.commands["twist"].ranges
  for term in train_cfg.curriculum.values():
    for stage in (term.params or {}).get("velocity_stages", []):
      for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if stage.get(axis) is not None:
          setattr(train_ranges, axis, stage[axis])
  live = env.unwrapped.command_manager.get_term("twist").cfg.ranges
  live_ranges = (live.lin_vel_x, live.lin_vel_y, live.ang_vel_z)
  # Reuse GoalSpace's own derivation against the training ranges, then restore.
  live.lin_vel_x, live.lin_vel_y, live.ang_vel_z = (
    train_ranges.lin_vel_x, train_ranges.lin_vel_y, train_ranges.ang_vel_z
  )
  gsp.freeze_scale(gsp.scale(env.unwrapped).clone())
  live.lin_vel_x, live.lin_vel_y, live.ang_vel_z = live_ranges
  print(f"[SHIM] no baked goal_scale in checkpoint -> pinned to the training value "
        f"{[round(x, 4) for x in gsp.scale(env.unwrapped).tolist()]}")
  return True
