from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_h1_2_flat_corr_noise_env_cfg,
  unitree_h1_2_flat_env_cfg,
  unitree_h1_2_flat_explicit_pd_env_cfg,
  unitree_h1_2_flat_lean_env_cfg,
  unitree_h1_2_flat_wide_dr_env_cfg,
  unitree_h1_2_rough_env_cfg,
)
from .rl_cfg import unitree_h1_2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-H1_2-Rough",
  env_cfg=unitree_h1_2_rough_env_cfg(),
  play_env_cfg=unitree_h1_2_rough_env_cfg(play=True),
  rl_cfg=unitree_h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H1_2-Flat",
  env_cfg=unitree_h1_2_flat_env_cfg(),
  play_env_cfg=unitree_h1_2_flat_env_cfg(play=True),
  rl_cfg=unitree_h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Track F: lean-reward A0 baseline (shaping stripped; same env/DR as A0). Pairs with
# Unitree-H1_2-Flat-A1-Lean -> doc/hrl/hierarchy_benefit_roadmap.md Track F.
register_mjlab_task(
  task_id="Unitree-H1_2-Flat-Lean",
  env_cfg=unitree_h1_2_flat_lean_env_cfg(),
  play_env_cfg=unitree_h1_2_flat_lean_env_cfg(play=True),
  rl_cfg=unitree_h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# WL-D arm 7 (2026-07-17, WL-E transfer audit): explicit discrete-torque PD actuation
# instead of the implicit MuJoCo <position> servo. See h1_2_constants.py.
register_mjlab_task(
  task_id="Unitree-H1_2-Flat-ExplicitPD",
  env_cfg=unitree_h1_2_flat_explicit_pd_env_cfg(),
  play_env_cfg=unitree_h1_2_flat_explicit_pd_env_cfg(play=True),
  rl_cfg=unitree_h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# WL-D arm 8 (2026-07-17, WL-E transfer audit): unitree_rl_gym-recipe perturbation DR
# bundle (push, base-mass, foot-friction floor). See apply_wide_dr in env_cfgs.py.
register_mjlab_task(
  task_id="Unitree-H1_2-Flat-WideDR",
  env_cfg=unitree_h1_2_flat_wide_dr_env_cfg(),
  play_env_cfg=unitree_h1_2_flat_wide_dr_env_cfg(play=True),
  rl_cfg=unitree_h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# WL-D arm 9 (2026-07-17, WL-E transfer audit): correlated (low-pass) obs noise on the
# actor's base_ang_vel/joint_vel channels. See noise.py / apply in env_cfgs.py.
register_mjlab_task(
  task_id="Unitree-H1_2-Flat-CorrNoise",
  env_cfg=unitree_h1_2_flat_corr_noise_env_cfg(),
  play_env_cfg=unitree_h1_2_flat_corr_noise_env_cfg(play=True),
  rl_cfg=unitree_h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
