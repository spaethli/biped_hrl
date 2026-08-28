from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def _gait_schedule(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        use_commanded_phase: bool = False,
        swing_time: float = 0.0,
        duty_max: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | float]:
    """Shared phase/duty schedule: per-leg phase in [0,1) + the scheduled-stance mask.

    Factored out of ``feet_gait`` (WL-D arm 6, 2026-07-17) so the roll-over/push-off
    terms below read the identical schedule - never a second hardcoded copy of the
    duty threshold (A1a_plan.md Arm 6 gate pin iii). Returns ``(leg_phase, is_stance,
    duty)``; ``duty`` is the threshold actually applied (float, or a per-env [B,1]
    tensor under the d(T) schedule) so callers can compute ``phi = leg_phase / duty``.
    """
    # A1a: when the HL commands cadence, key the schedule to the accumulated per-env phase
    # (env.hrl_phase in [0,1)) instead of the fixed-period clock. Default off -> A0 unchanged.
    if use_commanded_phase:
        global_phase = env.hrl_phase.unsqueeze(1)
        if swing_time > 0.0:
            # d(T) duty schedule (A1a): hold single-support ~= swing_time (the inverted-
            # pendulum constant, like humans) and let double stance absorb long periods:
            # d = 1 - swing_time/T, floored at the fixed threshold (fast band unchanged;
            # departs above T = swing_time/(1-threshold)) and capped at duty_max (0.70 =
            # human slow-walk duty; d <= 0.5 would be running). Per-env: T = hrl_period.
            threshold = (1.0 - swing_time / env.hrl_period).clamp(threshold, duty_max).unsqueeze(1)
    else:
        global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    return leg_phase, is_stance, threshold


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
        use_commanded_phase: bool = False,
        swing_time: float = 0.0,
        duty_max: float = 0.70,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    _, is_stance, _ = _gait_schedule(
        env, period, offset, threshold, use_commanded_phase, swing_time, duty_max
    )
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def _command_gate(env: ManagerBasedRlEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    """``1[|cmd| > threshold]`` gate shared by the Arm 6 terms below (same rule feet_gait
    and the other command-gated terms in this module already apply, factored out once
    there were 2 new call sites)."""
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    return (total_command > command_threshold).float()


def ankle_pushoff_pitchref(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        command_threshold: float,
        theta_hs: float,
        theta_to: float,
        k: float,
        sigma: float,
        period: float,
        offset: list[float],
        threshold: float,
        use_commanded_phase: bool = False,
        swing_time: float = 0.0,
        duty_max: float = 0.70,
) -> torch.Tensor:
    """WL-D arm 6, formulation A: heel-to-toe ankle roll-over phase-locking.

    Matches ``ankle_pitch`` to a raised-cosine reference interpolated between the
    measured heel-strike/toe-off angles over scheduled-stance progress ``phi``
    (Siekmann et al. arXiv:2011.01387's phase-indexed-reference generalization of
    ``feet_gait``). ``asset_cfg.joint_ids`` must list the ankle_pitch joints in the
    SAME left/right order as ``offset`` (see ``feet_gait``'s own offset convention).

    Gate pins (A1a_plan.md Arm 6, 2026-07-17 review): (i) command-gated - the phase
    clock keeps running at stand, so an ungated term rewards ankle-marching in place;
    (ii) gated on the SCHEDULED stance window (not actual contact) - ``phi`` is only
    well-defined on the schedule, and ``feet_gait`` already pushes contact to match it.
    """
    asset: Entity = env.scene[asset_cfg.name]
    leg_phase, is_stance, duty = _gait_schedule(
        env, period, offset, threshold, use_commanded_phase, swing_time, duty_max
    )
    phi = leg_phase / duty
    theta_ref = theta_hs + (theta_to - theta_hs) * (1 - torch.cos(math.pi * phi.pow(k))) / 2
    ankle_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    sq_err = (ankle_pos - theta_ref).square() * is_stance.float()
    reward = torch.exp(-sq_err.sum(dim=1) / sigma**2)
    return reward * _command_gate(env, command_name, command_threshold)


def ankle_pushoff_power(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        sensor_name: str,
        command_name: str,
        command_threshold: float,
        w: float,
        p_scale: float,
        period: float,
        offset: list[float],
        threshold: float,
        use_commanded_phase: bool = False,
        swing_time: float = 0.0,
        duty_max: float = 0.70,
) -> torch.Tensor:
    """WL-D arm 6, formulation B: ankle push-off power burst.

    Saturating bonus on signed ankle-pitch power (``tau * qd`` - NOT ``mech_power``,
    which is abs-summed over all joints) in the terminal-stance window, bounded above
    (per the same clamp-discipline that motivated ``cost_of_transport_penalty``'s
    clamp) instead of rewarding unbounded torque-cranking.

    **Direction-corrected 2026-07-20** (the user's visual replay at model_7400 caught it):
    the original formula gated only on ``ReLU(power)`` - "concentric, not eccentric" -
    which does NOT distinguish push-off (plantarflexion) from a toe-lift (dorsiflexion):
    both are concentric work, just in opposite directions, and the trained policy chose
    the toe-lift 100% of the time (measured on the buggy run's `model_10000`: every
    rewarded step had `qd < 0`). Root cause: the constants-probe's sign convention
    (2026-07-17) was ITSELF backwards - it inferred direction from a correlational signal
    smaller than its own noise. A forward-kinematics sweep of the XML (no policy, no
    dynamics - just "which way does the toe move as the joint angle changes") settles it:
    increasing ``ankle_pitch`` moves the toe DOWN (plantarflexion); decreasing moves it UP
    (dorsiflexion). So the fix gates on ``qd > 0`` - only power delivered WHILE actively
    plantarflexing counts, not any concentric power regardless of direction.

    Gate pins: SCHEDULE **and** actual contact (stricter than formulation A) - phase-only
    gating would let the LL harvest the bonus by driving the ankle in the air after an
    early liftoff; power without ground contact is thrash, not propulsion. Known watch
    item (monitor in replay, not a redesign): ReLU keeps the positive half of any ankle
    dither inside the window, so dithering nets some reward, capped at the same ceiling
    as genuine push-off.
    """
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    leg_phase, is_stance, duty = _gait_schedule(
        env, period, offset, threshold, use_commanded_phase, swing_time, duty_max
    )
    phi = leg_phase / duty
    gate = is_stance & (phi >= (1.0 - w)) & is_contact
    tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    power = tau * qd
    # Only count power delivered while actively plantarflexing (qd > 0, corrected
    # convention) - excludes concentric dorsiflexion (toe-lift), which ReLU alone let
    # through since it only checks torque-velocity sign agreement, not direction.
    directed_power = power * (qd > 0).float()
    bonus = 1.0 - torch.exp(-torch.relu(directed_power) / p_scale)
    reward = (gate.float() * bonus).sum(dim=1)
    return reward * _command_gate(env, command_name, command_threshold)


def heel_toe_rollover_contact(
        env: ManagerBasedRlEnv,
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
        geom_ids: list[list[int]],
        body_ids: list[int],
        heel_x_max: float,
        toe_x_min: float,
        command_name: str,
        command_threshold: float,
        w: float,
        period: float,
        offset: list[float],
        threshold: float,
        use_commanded_phase: bool = False,
        swing_time: float = 0.0,
        duty_max: float = 0.70,
) -> torch.Tensor:
    """WL-D arm 6, formulation C: heel-to-toe contact-sequence push-off (2026-07-24,
    position-based redesign 2026-07-29).

    Rewards the actual foot contact SEQUENCE (not joint angle): specifically the
    "heel lifts before toe" state (toe in contact AND heel not in contact) during
    terminal stance. Formulations A/B only shaped ``ankle_pitch`` joint angle; empirical
    replay showed whole feet lifted simultaneously despite correct angle rotation.

    **Redesigned 2026-07-29** - the original version grouped the 7 collision sub-geoms
    per foot into a fixed "heel" set (foot1/2) and "toe" set (foot5/6) by contact-onset
    TIMING rank on the untrained baseline. Reading ``h1_2.xml`` directly showed this was
    wrong: the sub-geoms are CAPSULES defined by ``fromto`` endpoints, and 5 of the 7
    (foot2-foot6) each individually span the foot's *entire* heel-to-toe length
    (local x from -0.08 to +0.14/+0.17) - only foot1/foot7 are short and toe-only
    (x in [0.045, 0.10]). A capsule's boolean ``found`` flag can't say WHICH end is
    touching, so the old foot1+2-vs-foot5+6 grouping mostly differed in Y (medial vs
    lateral), not X (heel vs toe) - it trained an ankle ROLL, exactly what the user's
    replay caught ("foot is rolling inward... not pitch... both feet roll").

    Fix: read each contact's actual world POSITION (``sensor.data.pos``, populated when
    ``reduce="maxforce"`` picks the highest-force contact per primary - during a rolling
    stance that point migrates from the heel end of a long capsule to its toe end,
    tracking the real center-of-pressure sweep) and transform it into the owning
    ``{side}_ankle_roll_link`` body's LOCAL frame (the same frame the ``fromto``
    coordinates are defined in, since these geoms are direct children of that body with
    no further nesting - confirmed in ``h1_2.xml``). A contact classifies as heel if its
    local x < ``heel_x_max`` and toe if x > ``toe_x_min``, leaving a neutral midfoot band
    between the two thresholds. This works even for the long, ambiguous capsules because
    it classifies by contact POSITION, not by geom identity - foot1/foot7 (already
    toe-only by construction) simply always land in the toe bucket when in contact.

    ``geom_ids``/``body_ids`` are PER-LEG (outer list index 0/1 = left/right, matching
    ``offset``'s [0.0, 0.5] convention) - resolved once at runner init time. Evaluated
    and summed over BOTH legs (like ``ankle_pushoff_power``).

    Gate: terminal-stance window + SCHEDULE only (gate pin ii, A1a_plan.md Arm 6).
    Requires ``hl_cadence`` and a contact sensor with per-subgeom ``.found``/``.pos`` data.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None and sensor.data.pos is not None
    found = sensor.data.found  # [B, N_geoms]
    pos_w = sensor.data.pos  # [B, N_geoms, 3]
    asset: Entity = env.scene[asset_cfg.name]
    n_legs = len(geom_ids)
    heel_contact = torch.zeros(found.shape[0], n_legs, dtype=torch.bool, device=found.device)
    toe_contact = torch.zeros(found.shape[0], n_legs, dtype=torch.bool, device=found.device)
    for leg in range(n_legs):
      body_pos_w = asset.data.body_link_pos_w[:, body_ids[leg]]
      body_quat_w = asset.data.body_link_quat_w[:, body_ids[leg]]
      for idx in geom_ids[leg]:
        is_found = found[:, idx] > 0
        local_x = quat_apply_inverse(body_quat_w, pos_w[:, idx] - body_pos_w)[:, 0]
        heel_contact[:, leg] |= is_found & (local_x < heel_x_max)
        toe_contact[:, leg] |= is_found & (local_x > toe_x_min)
    # Per-leg indicator: toe in contact AND heel NOT in contact (heel lifted, toe still down)
    rollover = toe_contact & ~heel_contact  # [B, n_legs]
    # Gate: terminal-stance window (phi in [1-w, 1)) on the SCHEDULE (not actual contact)
    leg_phase, is_stance, duty = _gait_schedule(
        env, period, offset, threshold, use_commanded_phase, swing_time, duty_max
    )
    phi = leg_phase / duty
    gate = is_stance & (phi >= (1.0 - w))  # [B, n_legs]
    # Reward: rollover indicator, gated, summed over both legs, command-gated
    reward = (gate.float() * rollover.float()).sum(dim=1)
    return reward * _command_gate(env, command_name, command_threshold)


class foot_step_symmetry:
  """WL-D arm 10, formulation B (primary training arm - A1a_plan.md "Arm 10"):
  step-time left/right symmetry index. Per-env last-touchdown-time buffer (mirrors
  ``feet_swing_height``'s per-env buffer pattern below) so each new touchdown can be
  compared against the OTHER foot's last one, giving the two alternating step-time
  intervals ``t_LR``/``t_RL`` (contact-based, no history buffer - contrast formulation
  A below). The reward is dense (available every step once both intervals have been
  observed at least once), not sparse-at-touchdown, so PPO gets a per-step gradient
  rather than one spike per stride.

  A double-tap (the observed defect: one foot touches down twice before the other
  lifts) drives one interval toward 0, so ``SI -> 1`` and the reward saturates toward
  0 - it fires on exactly the behaviour the arm targets.

  Plain-argument constructor (not the ``RewardTermCfg``-driven ``__init__`` above it):
  ``hrl_runner`` computes its LL intrinsic directly (no ``RewardManager`` in the
  loop), so it instantiates stateful helpers the way it already does for
  ``GoalStateNoise`` (``state_noise.py``) - construct once with plain args, call every
  step, reset explicitly from the runner's own ``dones``.
  """

  def __init__(self, num_envs: int, device: str) -> None:
    self.last_td_time = torch.full((num_envs, 2), -1.0, device=device)
    self.t_lr = torch.zeros(num_envs, device=device)
    self.t_rl = torch.zeros(num_envs, device=device)
    self.have_lr = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.have_rl = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def reset_envs(self, dones: torch.Tensor) -> None:
    """Called from ``hrl_runner`` with THIS step's ``dones`` before the touchdown read,
    so a stale pre-reset last-touchdown time never gets diffed against a fresh
    post-reset touchdown (the runner has no RewardManager driving its LL intrinsic,
    so it calls this directly instead of relying on ``RewardManager.reset``)."""
    if dones is None or not dones.any():
      return
    d = dones.bool()
    self.last_td_time[d] = -1.0
    self.have_lr[d] = False
    self.have_rl[d] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float,
    sigma_si: float,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    first_contact = sensor.compute_first_contact(dt=env.step_dt)  # [B, 2], 0=left, 1=right
    # episode_length_buf is already reset for envs whose episode just ended (auto-reset
    # vecenv semantics) - reusing it as the clock avoids a second manually-reset buffer.
    now = env.episode_length_buf.float() * env.step_dt  # [B]

    valid = self.last_td_time >= 0.0  # [B, 2]
    td_l = first_contact[:, 0] & valid[:, 1]  # left touchdown completes a t_RL interval
    self.t_rl = torch.where(td_l, now - self.last_td_time[:, 1], self.t_rl)
    self.have_rl = self.have_rl | td_l
    td_r = first_contact[:, 1] & valid[:, 0]  # right touchdown completes a t_LR interval
    self.t_lr = torch.where(td_r, now - self.last_td_time[:, 0], self.t_lr)
    self.have_lr = self.have_lr | td_r
    self.last_td_time = torch.where(first_contact, now.unsqueeze(1), self.last_td_time)

    si = (self.t_lr - self.t_rl).abs() / (self.t_lr + self.t_rl).clamp(min=1e-3)
    reward = torch.exp(-si.square() / sigma_si**2) * (self.have_lr & self.have_rl).float()
    return reward * _command_gate(env, command_name, command_threshold)


class phaseshift_joint_mirror:
  """WL-D arm 10, formulation A (implemented alongside B but left untrained pending
  B's read - the Arm 6 pattern): half-period phase-shifted joint mirror,
  ``r = exp(-mean_pairs(q_L(t) - m*q_R(t-tau))^2 / sigma^2)``, ``tau = T/2`` (``T`` =
  ``env.hrl_period``, the HL-commanded stride period). The CRITICAL adaptation vs. the
  dead-code reference ``joint_mirror`` (A1a_plan.md "Arm 10"): comparing same-instant
  ``q_L(t)`` to ``q_R(t)`` would force the antiphase legs INTO phase (hopping); this
  compares to the OTHER leg's state half a stride ago instead, the actual gait-symmetry
  statement.

  Buffers only the RIGHT leg's joint history (mirrors ``feet_swing_height``'s per-env
  buffer pattern below) - the formula only needs ``q_R(t-tau)``; ``q_L(t)`` is read
  live each step. ``tau`` varies per env with the commanded period, so the lookback is
  a per-env step count, not a fixed one; the ring buffer is sized off
  ``cadence_period_range``'s upper bound so the longest realistic ``tau`` always fits.

  Plain-argument constructor - see ``foot_step_symmetry`` above for why (no
  ``RewardManager`` drives ``hrl_runner``'s LL intrinsic).
  """

  def __init__(self, num_envs: int, n_joints: int, max_period: float, step_dt: float,
               device: str) -> None:
    self.step_dt = step_dt
    self.ring_len = max(2, int(math.ceil(max_period / 2.0 / step_dt)) + 1)
    self.buf = torch.zeros(self.ring_len, num_envs, n_joints, device=device)
    self.filled = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.ptr = 0
    self._env_idx = torch.arange(num_envs, device=device)

  def reset_envs(self, dones: torch.Tensor) -> None:
    """See ``foot_step_symmetry.reset_envs`` - same reason, same calling convention."""
    if dones is None or not dones.any():
      return
    self.filled[dones.bool()] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_asset_cfg: SceneEntityCfg,
    right_asset_cfg: SceneEntityCfg,
    sigma: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    asset: Entity = env.scene[left_asset_cfg.name]
    q_left = asset.data.joint_pos[:, left_asset_cfg.joint_ids]    # [B, J] live
    q_right = asset.data.joint_pos[:, right_asset_cfg.joint_ids]  # [B, J] pushed into history

    self.buf[self.ptr] = q_right
    delay_steps = (env.hrl_period / 2.0 / self.step_dt).round().long().clamp(1, self.ring_len - 1)
    idx = (self.ptr - delay_steps) % self.ring_len
    q_right_delayed = self.buf[idx, self._env_idx]  # [B, J]
    valid = self.filled > delay_steps

    self.ptr = (self.ptr + 1) % self.ring_len
    self.filled = torch.clamp(self.filled + 1, max=self.ring_len)

    err = (q_left - q_right_delayed).square().mean(dim=1)
    reward = torch.exp(-err / sigma**2) * valid.float()
    return reward * _command_gate(env, command_name, command_threshold)


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def mech_power(asset: Entity) -> torch.Tensor:
  """Total mechanical power draw (W): Σ_j |qfrc_actuator_j · joint_vel_j|.

  Shared by the benchmark (play.py), the A1a HL's window CoT (hrl_runner.py), and
  cost_of_transport_penalty below - keep the definition in exactly one place.
  """
  return (asset.data.qfrc_actuator * asset.data.joint_vel).abs().sum(dim=-1)


def cost_of_transport_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  vel_floor: float = 0.1,
  mass_g: float = 75.0 * 9.81,
  max_cot: float = 10.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-step command-gated cost-of-transport analog: P / (m·g·max(|v_cmd|, floor)).

  Same normalization the A1a HL's ``hl_cot_coef`` window reward uses (ADR-0004), so an
  A0 or A1-LL run shaped by this term is a fair architectural energy comparator
  (A1a_plan.md stage S4', "A0+energy"). Stateless/per-step - unlike the HL's
  window-integrated version, this uses COMMANDED speed as the distance-rate proxy
  (not achieved distance), so it needs no window state and can drop into either the
  A0 RewardManager or the A1 LL intrinsic unchanged.

  ``max_cot`` clamps the ratio (not just the denominator floor) - the same catastrophe-
  bound pattern the LL posture/action-rate penalties use (a torque-limited joint
  thrashing at near-zero net displacement can spike ``mech_power`` arbitrarily; healthy
  CoT sits at 0.5-1.5 across every measured run, so 10 is a physical-blowup bound, not
  a tuning knob). Deliberately NOT applied to the HL's own hl_cot_coef window
  computation (hrl_runner.py) - that path feeds a TD3 replay buffer, not an on-policy
  PPO rollout, so it lacks the single-sample GAE-corruption failure mode the clamp
  guards against (2026-07-17 WL-D discussion).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_speed = command[:, :2].norm(dim=-1)
  gated = (cmd_speed > command_threshold).float()
  cot = (mech_power(asset) / (cmd_speed.clamp(min=vel_floor) * mass_g)).clamp(max=max_cot)
  return cot * gated


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward

