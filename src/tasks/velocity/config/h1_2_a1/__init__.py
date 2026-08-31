from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import HierarchicalRunner

from .env_cfgs import unitree_h1_2_flat_a1_env_cfg
from .rl_cfg import unitree_h1_2_hrl_runner_cfg

# Runner cfg is the single source of truth for the goal space; the env derives its
# goal obs dim from the same component list so the two can't desync.
_rl_cfg = unitree_h1_2_hrl_runner_cfg()

register_mjlab_task(
  task_id="Unitree-H1_2-Flat-A1",
  env_cfg=unitree_h1_2_flat_a1_env_cfg(goal_components=_rl_cfg.goal_components),
  play_env_cfg=unitree_h1_2_flat_a1_env_cfg(
    play=True, goal_components=_rl_cfg.goal_components
  ),
  rl_cfg=_rl_cfg,
  runner_cls=HierarchicalRunner,
)

# Track F: lean-reward A1 (shaping stripped, same env reward as Unitree-H1_2-Flat-Lean).
# Warm-start from a *lean*-A0 checkpoint, not shaped-A0 (else shaped gait leaks in via
# the init). Fresh runner cfg instance to avoid shared mutable state with the A1 task.
# -> doc/hrl/hierarchy_benefit_roadmap.md Track F.
_rl_cfg_lean = unitree_h1_2_hrl_runner_cfg()
register_mjlab_task(
  task_id="Unitree-H1_2-Flat-A1-Lean",
  env_cfg=unitree_h1_2_flat_a1_env_cfg(
    goal_components=_rl_cfg_lean.goal_components, lean=True
  ),
  play_env_cfg=unitree_h1_2_flat_a1_env_cfg(
    play=True, goal_components=_rl_cfg_lean.goal_components, lean=True
  ),
  rl_cfg=_rl_cfg_lean,
  runner_cls=HierarchicalRunner,
)

# WP2/WP3 (2026-08-31): A1a + torso payload DR (0-12 kg), the H-mem baseline arm.
_rl_cfg_payload = unitree_h1_2_hrl_runner_cfg()
register_mjlab_task(
  task_id="Unitree-H1_2-Flat-A1-Payload",
  env_cfg=unitree_h1_2_flat_a1_env_cfg(
    goal_components=_rl_cfg_payload.goal_components, payload_dr=True
  ),
  play_env_cfg=unitree_h1_2_flat_a1_env_cfg(
    play=True, goal_components=_rl_cfg_payload.goal_components, payload_dr=True
  ),
  rl_cfg=_rl_cfg_payload,
  runner_cls=HierarchicalRunner,
)
