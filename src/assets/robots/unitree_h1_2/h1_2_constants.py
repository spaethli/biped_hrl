"""Unitree H1_2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from src.assets.robots._utils import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

H1_2_XML: Path = SRC_PATH / "assets" / "robots" / "unitree_h1_2" / "xmls" / "h1_2.xml"
assert H1_2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, H1_2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H1_2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
# Model v2 (ADR-0005): upper-body gains = deploy hold gains (split deploy),
# frictionloss/viscous_damping nonzero on every joint (real joints have stiction;
# reference XMLs use 0.1/0.001). Stiffness/damping MUST stay in lockstep with the
# deploy YAML gain vectors (deploy/robots/h1_2/config/policy/*/params/*.yaml).
##

H1_2_ACTUATOR_M107_24_2 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw.*",
    ".*_hip_pitch.*",
    ".*_hip_roll.*",
  ),
  stiffness=200.0,
  damping=2.5,
  effort_limit=200.0,
  armature=0.025,
  frictionloss=0.1,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_TORSO = BuiltinPositionActuatorCfg(
  # M107_24_2 motor; hold gains 300/3 (torso leaves the hip group in v2).
  target_names_expr=("torso_joint",),
  stiffness=300.0,
  damping=3.0,
  effort_limit=200.0,
  armature=0.025,
  frictionloss=0.1,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_M107_24_1 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee.*",),
  stiffness=300.0,
  damping=4.0,
  effort_limit=300.0,
  armature=0.04,
  frictionloss=0.1,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_GO2HV_1 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch.*",
    ".*_ankle_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=40.0,
  armature=0.005,
  frictionloss=0.1,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_SHOULDER_PR = BuiltinPositionActuatorCfg(
  # GO2HV_1 motor; hold gains 120/2 (shoulders leave the ankle group in v2).
  target_names_expr=(
    ".*_shoulder_pitch.*",
    ".*_shoulder_roll.*",
  ),
  stiffness=120.0,
  damping=2.0,
  effort_limit=40.0,
  armature=0.005,
  frictionloss=0.1,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  # GO2HV_2 motor (effort 18, URDF); hold gains 120/2 like the other shoulder axes.
  target_names_expr=(".*_shoulder_yaw.*",),
  stiffness=120.0,
  damping=2.0,
  effort_limit=18.0,
  armature=0.002,
  frictionloss=0.1,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_GO2HV_2 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow.*",
    ".*_wrist_pitch.*",
    ".*_wrist_roll.*",
    ".*_wrist_yaw.*",
  ),
  stiffness=80.0,
  damping=1.0,
  effort_limit=18.0,
  armature=0.002,
  frictionloss=0.1,
  viscous_damping=0.001,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.02),
  joint_pos={
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_joint": 0.5,
    ".*_ankle_pitch_joint": -0.3,
    ".*_shoulder_pitch_joint": 0.28,
    ".*_elbow_joint": 0.52,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

H1_2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H1_2_ACTUATOR_M107_24_2,
    H1_2_ACTUATOR_M107_24_1,
    H1_2_ACTUATOR_GO2HV_1,
    H1_2_ACTUATOR_GO2HV_2,
    H1_2_ACTUATOR_SHOULDER_YAW,
    H1_2_ACTUATOR_SHOULDER_PR,
    H1_2_ACTUATOR_TORSO,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h1_2_robot_cfg() -> EntityCfg:
  """Get a fresh H1_2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H1_2_ARTICULATION,
  )


H1_2_ACTION_SCALE: dict[str, float] = {}
for a in H1_2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    H1_2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h1_2_robot_cfg())

  viewer.launch(robot.spec.compile())
