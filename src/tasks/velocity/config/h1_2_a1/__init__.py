from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import HierarchicalRunner

from .env_cfgs import unitree_h1_2_flat_a1_env_cfg
from .rl_cfg import unitree_h1_2_hrl_runner_cfg

register_mjlab_task(
  task_id="Unitree-H1_2-Flat-A1",
  env_cfg=unitree_h1_2_flat_a1_env_cfg(),
  play_env_cfg=unitree_h1_2_flat_a1_env_cfg(play=True),
  rl_cfg=unitree_h1_2_hrl_runner_cfg(),
  runner_cls=HierarchicalRunner,
)
