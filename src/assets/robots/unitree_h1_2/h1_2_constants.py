"""Unitree H1_2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
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
# Model v2 (ADR-0005): upper-body gains = deploy hold gains (split deploy).
# frictionloss is 0 in the training nominal BY CHOICE, not necessity (WL-E 2026-07-17,
# ADR-0005 Amendment 2): the old "0.1 stalls training" evidence was a kl-0.005 +
# torso/arm-hold confound; fric 0.1 at desired_kl 0.01 trains from scratch to baseline
# quality (a0_v2_optB_fric0p1_kl01_s42: err_vx 0.083, 0 falls) and stays a sanctioned
# option. Flipping it is a v2-style rebase (invalidates all checkpoints), Liam's call;
# until then joint-friction fidelity lives in the deploy-sim eval and A2 Wide DR.
# Stiffness/damping MUST stay in lockstep with the deploy YAML gain vectors
# (deploy/robots/h1_2/config/policy/*/params/*.yaml).
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
  frictionloss=0.0,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_TORSO = BuiltinPositionActuatorCfg(
  # M107_24_2 motor; hold gains 300/3 (option B, chosen 2026-07-09 for deploy fidelity:
  # torso held at its true deploy hold gain). REQUIRES desired_kl=0.01 — the torso+arm
  # hold combination stalls under 0.005 but converges under 0.01 (ADR-0005 amendment).
  target_names_expr=("torso_joint",),
  stiffness=300.0,
  damping=3.0,
  effort_limit=200.0,
  armature=0.025,
  frictionloss=0.0,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_M107_24_1 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee.*",),
  stiffness=300.0,
  damping=4.0,
  effort_limit=300.0,
  armature=0.04,
  frictionloss=0.0,
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
  frictionloss=0.0,
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
  frictionloss=0.0,
  viscous_damping=0.001,
)
H1_2_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  # GO2HV_2 motor (effort 18, URDF); hold gains 120/2 like the other shoulder axes.
  target_names_expr=(".*_shoulder_yaw.*",),
  stiffness=120.0,
  damping=2.0,
  effort_limit=18.0,
  armature=0.002,
  frictionloss=0.0,
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
  frictionloss=0.0,
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


##
# WL-D arm 7 (2026-07-17): explicit discrete-torque PD actuation, matching what the
# IsaacGym references train AND the bridge/firmware run (mjlab's BuiltinPositionActuator
# instead uses MuJoCo's native <position> actuator, an IMPLICIT servo baked into the
# solver). IdealPdActuator computes torque = kp*(q_des-q) + kd*(qdot_des-qdot) explicitly
# at sim dt via a <motor> actuator, clipped to effort_limit - same gains/armature/
# frictionloss/viscous_damping as the builtin groups above, transmission swapped only.
# Falsifies WL-E transfer-gap suspect (i): expect the vendor-plant latency sensitivity
# (stand qvel_rms 0.11->0.31 over 0-4ms) to flatten toward the stress-plant profile.
##


def _as_explicit_pd(cfg: BuiltinPositionActuatorCfg) -> IdealPdActuatorCfg:
  return IdealPdActuatorCfg(
    target_names_expr=cfg.target_names_expr,
    stiffness=cfg.stiffness,
    damping=cfg.damping,
    effort_limit=cfg.effort_limit if cfg.effort_limit is not None else float("inf"),
    armature=cfg.armature,
    frictionloss=cfg.frictionloss,
    viscous_damping=cfg.viscous_damping,
  )


H1_2_ARTICULATION_EXPLICIT_PD = EntityArticulationInfoCfg(
  actuators=tuple(_as_explicit_pd(a) for a in H1_2_ARTICULATION.actuators),  # type: ignore[arg-type]
  soft_joint_pos_limit_factor=H1_2_ARTICULATION.soft_joint_pos_limit_factor,
)


def get_h1_2_robot_cfg(explicit_pd: bool = False) -> EntityCfg:
  """Get a fresh H1_2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  ``explicit_pd=True`` (WL-D arm 7): swap the implicit MuJoCo <position> servo for an
  explicit discrete-torque PD actuator, same gains. See H1_2_ARTICULATION_EXPLICIT_PD.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H1_2_ARTICULATION_EXPLICIT_PD if explicit_pd else H1_2_ARTICULATION,
  )


# Derived 0.25*effort/kp per joint, NOT the references' flat 0.25: with hold-gain arms
# (kp 80-120) a flat 0.25 rad scale lets exploration noise hammer the torso and the
# policy never finds a gait (bisect 2026-07-08: flat stuck at 0.10 tracking @2500 vs
# derived 0.51 @2000, torso v1 in both). The small derived arm authority also matches
# split deploy, where arm commands are ignored anyway. Effort limits stay at the
# conservative URDF values as safety clamps.
H1_2_ACTION_SCALE: dict[str, float] = {}
for a in H1_2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    H1_2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h1_2_robot_cfg())

  viewer.launch(robot.spec.compile())
