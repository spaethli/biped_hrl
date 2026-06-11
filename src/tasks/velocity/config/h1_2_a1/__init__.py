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
