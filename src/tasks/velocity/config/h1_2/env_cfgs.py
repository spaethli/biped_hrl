"""Unitree H1_2 velocity environment configurations."""

from src.assets.robots import (
  H1_2_ACTION_SCALE,
  get_h1_2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from src.tasks.velocity import mdp as project_mdp


def unitree_h1_2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1_2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_h1_2_robot_cfg()}

  # Set raycast sensor frame to H1_2 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # WL-D arm 6 formulation C (2026-07-24, position-based redesign 2026-07-29):
  # contact-sequence-based push-off reward. "pos" (global-frame contact point) is
  # required by the redesign: several sub-geoms are long capsules spanning the whole
  # foot, so a boolean "found" alone can't say which end (heel vs toe) is touching -
  # see heel_toe_rollover_contact's docstring. reduce="maxforce" keeps whichever end
  # currently bears more load, tracking the real center-of-pressure sweep.
  foot_subgeom_contact_cfg = ContactSensorCfg(
    name="foot_subgeom_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(left|right)_foot[1-7]_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "pos"),
    reduce="maxforce",
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    foot_subgeom_contact_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H1_2_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.55

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_yaw.*": 0.15,
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*torso.*": 0.15,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_yaw.*": 0.25,
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*torso.*": 0.25,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_h1_2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1_2 flat terrain velocity configuration."""
  cfg = unitree_h1_2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
    # Keep the bench command process at the pre-2026-07-14 training value: training moved
    # to (3.0, 20.0), but changing the eval distribution would break comparability of
    # every historical aggregate bench. Sustained-command behavior is scored by the
    # --eval-cmd-vx holds, which pin the command and never resample.
    twist_cmd.resampling_time_range = (3.0, 8.0)

  return cfg


# Track F (reward-shaping ablation): the shaping reward terms zeroed in the lean
# variant. Keep = task tracking (track_lin/ang_vel), torso orientation
# (body_orientation_l2), the posture/height anchor (pose = variable_posture; A0 has
# no standalone base-height term, so posture is what holds height), and a safety/
# training floor (is_terminated, joint_pos_limits, action_rate_l2, joint_acc_l2,
# self_collisions). Zeroed terms are still computed -> they keep logging as
# diagnostics, they just stop contributing to the gradient. To go "ultra-lean", add
# "pose" here too (warning: the humanoid may lose its upright anchor and never find a
# gait -> no signal). See doc/hrl/hierarchy_benefit_roadmap.md Track F.
_LEAN_ZERO_TERMS = (
  "foot_gait",
  "foot_clearance",
  "foot_slip",
  "soft_landing",
  "angular_momentum",
  "body_ang_vel",
  "stand_still",
  "action_rate_l2",
  "joint_acc_l2",
)


def apply_lean_reward(cfg: ManagerBasedRlEnvCfg) -> None:
  """Zero the shaping reward weights in place (Track F lean variant).

  Shared by lean-A0 and lean-A1 so the A0-vs-A1 comparison stays clean: the env
  reward is identical for both, the hierarchy is the only delta (RQ2 rule).
  """
  for name in _LEAN_ZERO_TERMS:
    cfg.rewards[name].weight = 0.0


def unitree_h1_2_flat_lean_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Lean-reward A0 flat env (Track F): A0 flat env, shaping terms stripped."""
  cfg = unitree_h1_2_flat_env_cfg(play=play)
  apply_lean_reward(cfg)
  return cfg


def unitree_h1_2_flat_explicit_pd_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """WL-D arm 7 (2026-07-17): A0 flat env with explicit discrete-torque PD actuation
  instead of MuJoCo's implicit <position> servo (same gains) - see h1_2_constants.py
  H1_2_ARTICULATION_EXPLICIT_PD. Everything else (reward/DR/terrain) is unchanged.
  """
  cfg = unitree_h1_2_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_h1_2_robot_cfg(explicit_pd=True)}
  return cfg


def apply_wide_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """WL-D arm 8 (2026-07-17): unitree_rl_gym-recipe perturbation DR, in place.

  Falsifies WL-E transfer-gap suspect (ii): push max linear speed 0.5->1.5 m/s, add
  base-mass DR (-1,+3) kg on torso_link, and lower the foot-friction floor 0.3->0.1.
  Bundled as one package (matching the reference recipe), not swept individually.
  """
  push = cfg.events["push_robot"].params["velocity_range"]
  push["x"] = (-1.5, 1.5)
  push["y"] = (-1.5, 1.5)
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": cfg.events["base_com"].params["asset_cfg"],
      "operation": "add",
      "ranges": (-1.0, 3.0),
    },
  )
  friction_ranges = cfg.events["foot_friction"].params["ranges"]
  cfg.events["foot_friction"].params["ranges"] = (0.1, friction_ranges[1])


def unitree_h1_2_flat_wide_dr_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """WL-D arm 8 (2026-07-17): A0 flat env with the wide-DR bundle applied."""
  cfg = unitree_h1_2_flat_env_cfg(play=play)
  if not play:
    apply_wide_dr(cfg)
  return cfg


def unitree_h1_2_flat_corr_noise_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """WL-D arm 9 (2026-07-17): A0 flat env with correlated (low-pass) obs noise.

  Falsifies WL-E transfer-gap suspect (iii): swaps the actor's base_ang_vel and
  joint_vel Unoise terms for an amplitude-matched ~2Hz EMA low-pass (same +-0.2 /
  +-1.5 ranges) instead of i.i.d. white noise each control step. Nothing else changes.
  """
  cfg = unitree_h1_2_flat_env_cfg(play=play)
  if not play:
    dt = cfg.decimation * cfg.sim.mujoco.timestep
    actor_terms = cfg.observations["actor"].terms
    actor_terms["base_ang_vel"].noise = project_mdp.LowPassNoiseModelCfg(
      noise_cfg=Unoise(n_min=-0.2, n_max=0.2), cutoff_hz=2.0, dt=dt
    )
    actor_terms["joint_vel"].noise = project_mdp.LowPassNoiseModelCfg(
      noise_cfg=Unoise(n_min=-1.5, n_max=1.5), cutoff_hz=2.0, dt=dt
    )
  return cfg
