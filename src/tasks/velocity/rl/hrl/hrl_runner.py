"""Hierarchical (A1) on-policy runner.

Drives a co-training loop: an on-policy PPO **low level** (reused from rsl_rl via
``MjlabOnPolicyRunner``) conditioned on a goal, plus a pluggable **high level**
(:class:`HighLevel`) that fires every ``c`` steps and emits the goal.

Milestone 1/2 ship the ``oracle`` high level (no learning). The learned high levels
(``ppo``, ``td3``) slot into ``_make_high_level`` without touching the loop.

Key wiring (HIRO delta encoding):
  * The high level fires every ``c`` steps and emits an absolute window target
    ``V*`` in goal space. Over the window the low level observes the *remaining*
    delta ``V* - s_i`` (written to ``env.hrl_goal``, read by the ``goal`` obs term)
    and learns on the intrinsic reward ``-Σ_c w_c ||V*_c - s_i_c||`` — not the env's
    task reward. The task reward is logged (A0-comparable) and accumulated for the HL.
  * The goal space (which state slice is commanded) is declarative and configurable;
    ``goal_dim`` is derived from it.
"""

from __future__ import annotations

import os
import re
import time

import torch
import wandb

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.utils.lab_api.string import resolve_matching_names_values

from ..runner import VelocityOnPolicyRunner
from .goal_space import build_goal_space, init_goal_buffer
from .high_level import HighLevel, HighLevelPpo, OracleHighLevel
from .state_noise import GoalStateNoise, HlVelJitter
from .td3 import HighLevelTd3
from ...mdp import rewards as mdp_rewards
from mjlab.envs.mdp.rewards import joint_acc_l2, joint_pos_limits


class HierarchicalRunner(VelocityOnPolicyRunner):
  """Two-level HRL runner (A1). Low level = PPO; high level = pluggable."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # HRL-specific config.
    self.c: int = train_cfg["c"]
    self.gamma_hi: float = train_cfg["gamma_hi"]
    self.relabeling: str = train_cfg["relabeling"]
    self.hl_algorithm: str = train_cfg["hl_algorithm"]
    self.hl_target_mode: str = train_cfg.get("hl_target_mode", "delta")
    self.hl_reward_mode: str = train_cfg.get("hl_reward_mode", "task")
    # Feed the HL the deployable base lin-vel estimate (vx,vy) as extra obs. Lets the
    # directional HL compute g = (command - v)/scale instead of guessing v. Off by default.
    self.hl_obs_vel: bool = train_cfg.get("hl_obs_vel", False)
    self._hl_vel_dim: int = 2 if self.hl_obs_vel else 0
    self.ll_task_reward_coef: float = train_cfg["ll_task_reward_coef"]
    # Upper-body deploy-hygiene penalties added to the LL intrinsic (ADR-0002): the env's
    # variable_posture / action_rate_l2 never reach the goal-only LL, so the A1 arms drift
    # behind the back and twist. Resolve the arms+waist joint ids once (0 coef -> no-op).
    self.ll_action_rate_coef: float = train_cfg.get("ll_action_rate_coef", 0.0)
    self.ll_posture_coef: float = train_cfg.get("ll_posture_coef", 0.0)
    # WL-D (2026-07-17): stand-still + A0-term-mirror LL-intrinsic levers, all 0 = off.
    self.ll_stand_still_coef: float = train_cfg.get("ll_stand_still_coef", 0.0)
    self.ll_angmom_coef: float = train_cfg.get("ll_angmom_coef", 0.0)
    self.ll_footslip_coef: float = train_cfg.get("ll_footslip_coef", 0.0)
    self.ll_footclear_coef: float = train_cfg.get("ll_footclear_coef", 0.0)
    self.ll_energy_coef: float = train_cfg.get("ll_energy_coef", 0.0)
    self.ll_joint_acc_coef: float = train_cfg.get("ll_joint_acc_coef", 0.0)
    self.ll_joint_limits_coef: float = train_cfg.get("ll_joint_limits_coef", 0.0)
    self.ll_soft_landing_coef: float = train_cfg.get("ll_soft_landing_coef", 0.0)
    self.ll_body_ang_vel_coef: float = train_cfg.get("ll_body_ang_vel_coef", 0.0)
    # WL-D arm 6 (2026-07-17): heel-to-toe roll-over (A) / push-off power burst (B) /
    # contact-sequence (C). Probe-derived defaults (D2 checkpoint constants probe, 2026-07-17);
    # sigma/k/w are free knobs (sigma anchored to env_cfgs.py's std_walking ankle_pitch tolerance).
    self.ll_pitchref_coef: float = train_cfg.get("ll_pitchref_coef", 0.0)
    self.ll_pitchref_theta_hs: float = train_cfg.get("ll_pitchref_theta_hs", -0.253)
    self.ll_pitchref_theta_to: float = train_cfg.get("ll_pitchref_theta_to", -0.275)
    self.ll_pitchref_sigma: float = train_cfg.get("ll_pitchref_sigma", 0.15)
    self.ll_pitchref_k: float = train_cfg.get("ll_pitchref_k", 2.0)
    self.ll_pushoff_coef: float = train_cfg.get("ll_pushoff_coef", 0.0)
    self.ll_pushoff_w: float = train_cfg.get("ll_pushoff_w", 0.175)
    self.ll_pushoff_p_scale: float = train_cfg.get("ll_pushoff_p_scale", 40.0)
    self.ll_rollover_coef: float = train_cfg.get("ll_rollover_coef", 0.0)
    self.ll_rollover_w: float = train_cfg.get("ll_rollover_w", 0.175)
    self.ll_rollover_heel_x_max: float = train_cfg.get("ll_rollover_heel_x_max", -0.03)
    self.ll_rollover_toe_x_min: float = train_cfg.get("ll_rollover_toe_x_min", 0.08)
    # WL-D arm 10 (2026-07-20): left/right gait symmetry. Formulation B (step-time
    # symmetry index) is the primary training arm; formulation A (phase-shifted joint
    # mirror) is implemented alongside but left untrained pending B's read (the arm 6
    # pattern). sigma_si anchored to the arm-10 probe (2026-07-20): healthy checkpoints
    # (D2/arm4d/old-fix0p8) measure SI 0.010-0.016, the one severely double-tapping
    # checkpoint measures 0.235 - 0.06 sits between, keeping healthy gaits near r~0.9-1.0
    # while strongly penalizing the observed defect magnitude.
    self.ll_symmetry_coef: float = train_cfg.get("ll_symmetry_coef", 0.0)
    self.ll_symmetry_sigma_si: float = train_cfg.get("ll_symmetry_sigma_si", 0.06)
    self.ll_mirror_coef: float = train_cfg.get("ll_mirror_coef", 0.0)
    self.ll_mirror_sigma: float = train_cfg.get("ll_mirror_sigma", 0.15)
    # Per-step alive bonus (from-scratch survival economics; see rl_cfg docstring).
    self.ll_alive_coef: float = train_cfg.get("ll_alive_coef", 0.0)
    self.ll_goal_kernel: str = train_cfg.get("ll_goal_kernel", "l2")
    # A1a (ADR-0004): HL gait-cadence channel + cost-of-transport HL objective.
    self.hl_cadence: bool = train_cfg.get("hl_cadence", False)
    if self.ll_mirror_coef != 0.0 and not self.hl_cadence:
      raise ValueError("ll_mirror_coef requires hl_cadence=True (reads env.hrl_period).")
    self.cadence_period_range = tuple(train_cfg.get("cadence_period_range", (0.5, 1.4)))
    self.ll_cadence_coef: float = train_cfg.get("ll_cadence_coef", 0.0)
    self.hl_cot_coef: float = train_cfg.get("hl_cot_coef", 0.0)
    # d(T) duty schedule: swing time (s) held constant across periods; 0 = fixed-floor duty.
    self.cadence_swing_time: float = train_cfg.get("cadence_swing_time", 0.0)
    self.cadence_duty_range = tuple(train_cfg.get("cadence_duty_range", (0.56, 0.70)))
    # S1c: period source. "random" = per-episode uniform (S2); "hl" = the TD3 HL's
    # +1 action dim (goal_dim+1 nets — old checkpoints stay loadable via the default).
    self.hl_cadence_source: str = train_cfg.get("hl_cadence_source", "random")
    if self.hl_cadence_source == "hl" and not (self.hl_cadence and self.hl_algorithm == "td3"):
      raise ValueError("hl_cadence_source='hl' requires hl_cadence=True and hl_algorithm='td3'.")
    # 2026-07-09 posture-sag fix (default True): the HL learns only the velocity goal
    # columns; orientation/height targets are pinned to nominal (the oracle path — see
    # GoalSpace.to_target task_only). Inert for the oracle (it already targets nominal
    # for non-task components); only ppo lacks the machinery.
    self.hl_velocity_goals_only: bool = train_cfg.get("hl_velocity_goals_only", False)
    if self.hl_velocity_goals_only and self.hl_algorithm == "ppo":
      raise ValueError("hl_velocity_goals_only is not implemented for hl_algorithm='ppo'.")
    # Eval-only: pin the commanded stride period (for the CoT(period) sweep). None ->
    # mid-range constant (oracle/random-trained LL) or, later, the learned HL's action (S1c).
    self.eval_cadence_period: float | None = None
    # Gait params for the LL-intrinsic feet_gait term (match A0's foot_gait; period is
    # ignored when keyed to the commanded phase via use_commanded_phase=True).
    self._gait_params = dict(period=0.6, offset=[0.0, 0.5],
                             threshold=self.cadence_duty_range[0],
                             duty_max=self.cadence_duty_range[1],
                             command_threshold=0.1, command_name="twist",
                             sensor_name="feet_ground_contact",
                             swing_time=self.cadence_swing_time)
    # WL-D arm 6 (2026-07-17): formulation A gates on the SCHEDULE only (gate pin ii,
    # A1a_plan.md Arm 6) - no sensor_name needed, so it's dropped from the shared dict.
    self._pitchref_gait_params = {
      k: v for k, v in self._gait_params.items() if k != "sensor_name"
    }
    # WL-D arm 6 formulation C (2026-07-24, position-based redesign 2026-07-29): resolve
    # per-leg geom indices (all 7 sub-geoms, not a fixed heel/toe split - see
    # heel_toe_rollover_contact's docstring for why the geometry rules that out) plus
    # each leg's owning ankle_roll_link body id, needed to transform each contact's
    # world position into that foot's local frame.
    robot = env.unwrapped.scene["robot"]
    # Find all geoms matching the foot sub-geom pattern. The sensor's primary_names
    # will have resolved these in order (left_foot1..7, right_foot1..7).
    geom_names = []
    for side in ("left", "right"):
      for i in range(1, 8):
        geom_names.append(f"{side}_foot{i}_collision")
    # Build the index mapping (assumption: sensor data matches pattern order)
    geom_name_to_idx = {name: i for i, name in enumerate(geom_names)}
    # PER-LEG (outer index 0/1 = left/right, matching offset=[0.0, 0.5]) - NOT a flat
    # combined list: heel_toe_rollover_contact evaluates+sums both legs separately.
    self._rollover_geom_ids = [[geom_name_to_idx[f"{s}_foot{i}_collision"] for i in range(1, 8)]
                               for s in ("left", "right")]
    body_ids, _ = robot.find_bodies(
      ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )
    self._rollover_body_ids = body_ids
    # Deploy diagnostics (2026-08-07): leg joint ids for metrics/act_rate_legs +
    # metrics/jacc_legs (see `learn`'s loss_dict) - coef-independent, unlike
    # `_ub_joint_ids` below which only gates a reward term. Same leg patterns as
    # ARDIAG's g_legs (scripts/play.py); resolve individually since find_joints raises
    # if ANY pattern in the list matches zero joints.
    leg_ids: set[int] = set()
    for p in (".*hip.*", ".*knee.*", ".*ankle.*"):
      try:
        pids, _ = robot.find_joints([p])
      except ValueError:
        continue
      leg_ids.update(pids)
    self._diag_leg_joint_ids = (
      torch.as_tensor(sorted(leg_ids), device=device) if leg_ids else None
    )
    # Arms+waist (ADR-0002) + hip yaw/roll (2026-07-07): the goal space is heading-
    # invariant, so nothing else anchors leg alignment — from-scratch LLs walked with a
    # ~20° hip twist. A0 pins the same two joints via its tightest variable_posture stds.
    anchor_patterns = [".*shoulder.*", ".*elbow.*", ".*wrist.*", ".*torso.*",
                       ".*hip_yaw.*", ".*hip_roll.*"]
    # WL-D arm 5 (2026-07-17): replay defect (ankles roll inward) - extend the anchor.
    if train_cfg.get("ll_posture_anchor_ankle_roll", False):
      anchor_patterns.append(".*ankle_roll.*")
    # WL-D: all-joints anchor supersedes both the subset and the ankle_roll flag above.
    if train_cfg.get("ll_posture_all_joints", False):
      anchor_patterns = [".*"]
    ub_ids, ub_names = env.unwrapped.scene["robot"].find_joints(anchor_patterns)
    # WL-D arm 4b/4c: resolved foot-site cfg for the feet_slip/feet_clearance mirrors
    # (same sites A0's own foot_slip/foot_clearance terms use).
    self._foot_asset_cfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    self._foot_asset_cfg.resolve(env.unwrapped.scene)
    # WL-D: resolved torso body cfg for the body_ang_vel mirror (mirrors A0's own
    # body_ang_vel asset_cfg, config/h1_2/env_cfgs.py).
    self._torso_asset_cfg = SceneEntityCfg("robot", body_names=("torso_link",))
    self._torso_asset_cfg.resolve(env.unwrapped.scene)
    # WL-D arm 6 (2026-07-17): ankle_pitch joints for the roll-over/push-off intrinsic
    # terms - left,right order matches _gait_params's offset order [0.0, 0.5].
    self._ankle_asset_cfg = SceneEntityCfg(
      "robot", joint_names=("left_ankle_pitch_joint", "right_ankle_pitch_joint"))
    self._ankle_asset_cfg.resolve(env.unwrapped.scene)
    # WL-D arm 10 (2026-07-20): stateful helpers for both formulations, constructed
    # unconditionally (cheap; mirrors _ankle_asset_cfg's unconditional resolve above) -
    # gated at the call site instead, like every other WL-D lever.
    self._symmetry = mdp_rewards.foot_step_symmetry(env.num_envs, device)
    self._mirror_left_cfg = SceneEntityCfg(
      "robot", joint_names=("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint"))
    self._mirror_right_cfg = SceneEntityCfg(
      "robot", joint_names=("right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"))
    self._mirror_left_cfg.resolve(env.unwrapped.scene)
    self._mirror_right_cfg.resolve(env.unwrapped.scene)
    self._mirror = mdp_rewards.phaseshift_joint_mirror(
      env.num_envs, len(self._mirror_left_cfg.joint_ids), self.cadence_period_range[1],
      env.unwrapped.step_dt, device)
    self._ub_joint_ids = torch.as_tensor(ub_ids, device=device)
    # Stage D (2026-07-14): per-joint posture multipliers (pattern -> weight; unmatched
    # joints stay 1.0). NOT renormalized, so all-ones = the uniform penalty and raising
    # the shoulders does not dilute the hip yaw/roll anchor.
    self._ub_weights = torch.ones(len(ub_names), device=device)
    pw = train_cfg.get("ll_posture_weights")
    if pw:
      # Drop patterns matching zero anchored joints (e.g. the ankle_roll entry when
      # ll_posture_anchor_ankle_roll=False) - resolve_matching_names_values requires
      # every key to match at least one name, so an inert/not-yet-anchored pattern
      # would otherwise hard-error instead of being a no-op.
      pw = {p: w for p, w in dict(pw).items()
            if any(re.search(p, n) for n in ub_names)}
    if pw:
      w_idx, w_names, w_vals = resolve_matching_names_values(pw, ub_names)
      self._ub_weights[torch.as_tensor(w_idx, device=device)] = torch.as_tensor(
        w_vals, dtype=torch.float32, device=device)
      print("[HRL] Posture weights: "
            + ", ".join(f"{n}={v:g}" for n, v in zip(w_names, w_vals)))
    self.warm_start_path: str | None = train_cfg.get("warm_start_path")
    self.freeze_ll_path: str | None = train_cfg.get("freeze_ll_path")
    self.freeze_ll: bool = self.freeze_ll_path is not None
    if self.warm_start_path and self.freeze_ll_path:
      raise ValueError("warm_start_path and freeze_ll_path are mutually exclusive.")

    # Declarative goal space -> goal_dim is derived (nothing hardcodes a dimension).
    self.goal_space = build_goal_space(
      tuple(train_cfg["goal_components"]), train_cfg.get("goal_weights")
    )
    self.goal_dim: int = self.goal_space.dim

    # Estimator-noise on the LL goal channel (#8b sim2real DR; off by default ->
    # transparent passthrough). Persists across rollouts so per-episode bias survives.
    self.state_noise = GoalStateNoise(
      train_cfg.get("goal_state_noise"), self.goal_space, env.num_envs, device
    )

    # Initialise the goal buffer BEFORE the base builds models, so the env's `goal`
    # observation group resolves to the right dimension at construction time.
    init_goal_buffer(env.unwrapped, self.goal_dim, device)
    # A1a: per-env commanded stride period + accumulated gait phase (read by mdp.phase /
    # feet_gait). Created only when hl_cadence -> current A1 keeps the fixed-period clock.
    if self.hl_cadence:
      lo, hi = self.cadence_period_range
      env.unwrapped.hrl_period = torch.empty(env.num_envs, device=device).uniform_(lo, hi)
      env.unwrapped.hrl_phase = torch.zeros(env.num_envs, device=device)

    # Base builds self.alg = low-level PPO (using top-level actor/critic/algorithm/
    # obs_groups, which describe the LL) plus self.logger.
    super().__init__(env, train_cfg, log_dir, device)

    if self.cfg["num_steps_per_env"] % self.c != 0:
      raise ValueError(
        f"num_steps_per_env ({self.cfg['num_steps_per_env']}) must be a multiple "
        f"of c ({self.c}) so rollouts contain whole HL windows."
      )

    self.hl: HighLevel = self._make_high_level()
    # Per-window absolute target V* in goal space (set when the HL fires).
    self._target = torch.zeros(env.num_envs, self.goal_dim, device=device)
    # LL-obs target base: clean V* in absolute, noisy-estimate V* in delta (#8b faithful DR).
    self._target_obs = self._target

    # Warm-start the LL from an A0 checkpoint (proprio columns + deeper layers).
    # A resume-load (if any) runs after __init__ and overrides this.
    if self.warm_start_path:
      self._warm_start_low_level(self.warm_start_path)

    # Freeze-LL mode: load a converged A1 LL in full and freeze it (only the HL learns).
    if self.freeze_ll_path:
      self._load_frozen_ll(self.freeze_ll_path)

  def _warm_start_low_level(self, path: str) -> None:
    """Initialise the LL PPO actor/critic from an A0 checkpoint.

    The A0 *actor* obs = the A1 ``policy`` group with the ``command`` term inserted at
    its original position (NOT at the end: command sits between projected_gravity and
    phase). The A1 LL actor obs = ``policy`` (= A0 actor minus command) ++ ``goal``.
    So warm-starting the actor requires dropping A0's command columns *in the middle*
    and shifting everything after them left — a plain leading-block copy would
    misalign phase/joint_pos/joint_vel/actions.

    The A0 *critic* obs = the A1 ``critic`` group unchanged (command is still present,
    same position); A1 only appends ``goal``. So the critic is a clean leading-block
    copy with no gap.

    For both: input-facing tensors (first-layer weight + obs normalizer stats) get the
    shared proprio/privileged columns mapped (with the actor's command gap removed),
    deeper layers / output head / action std are copied verbatim, and the trailing
    ``goal`` columns keep their fresh init. The LL therefore starts as the walking A0
    policy that ignores the goal, then learns to use it.
    """
    ck = torch.load(path, map_location=self.device, weights_only=False)
    actor = getattr(self.alg, "_raw_actor", self.alg.actor)
    critic = getattr(self.alg, "_raw_critic", self.alg.critic)

    # Locate the command block in the A0 actor layout. The A1 critic group keeps the
    # same leading term order as the A0 actor (critic_terms = {**actor_terms, ...}),
    # so command's column offset there equals its offset in the A0 actor.
    obs_mgr = self.env.unwrapped.observation_manager
    names = obs_mgr.active_terms["critic"]
    dims = [d[0] for d in obs_mgr.group_obs_term_dim["critic"]]
    cmd_idx = names.index("command")
    cmd_start = sum(dims[:cmd_idx])
    cmd_width = dims[cmd_idx]

    # Actor: source columns are A0 actor cols with the command gap removed.
    actor_src_cols = self._source_cols(ck["actor_state_dict"]["mlp.0.weight"].shape[-1],
                                       skip=(cmd_start, cmd_width))
    self._partial_load(actor, ck["actor_state_dict"], self.goal_dim, actor_src_cols)
    # Critic: command still present in both -> contiguous leading block, no gap.
    critic_in = ck["critic_state_dict"]["mlp.0.weight"].shape[-1]
    self._partial_load(critic, ck["critic_state_dict"], self.goal_dim,
                       list(range(critic_in)))
    print(
      f"[HRL] Warm-started LL from A0: {path} "
      f"(actor command gap cols [{cmd_start}:{cmd_start + cmd_width}] dropped)"
    )

  def _load_frozen_ll(self, path: str) -> None:
    """Load a converged A1 LL (actor+critic, full) and freeze it.

    Unlike ``_warm_start_low_level`` (A0->A1 gap-aware partial copy), the source is an
    A1 LL with the same goal space as this run, so it's a plain strict load — a shape
    mismatch (wrong goal space) raises here, loudly. The LL then acts as a fixed
    deterministic policy; only the HL learns (see :meth:`learn`)."""
    ck = torch.load(path, map_location=self.device, weights_only=False)
    actor = getattr(self.alg, "_raw_actor", self.alg.actor)
    critic = getattr(self.alg, "_raw_critic", self.alg.critic)
    actor.load_state_dict(ck["actor_state_dict"], strict=True)
    critic.load_state_dict(ck["critic_state_dict"], strict=True)
    print(f"[HRL] Loaded + FROZE LL from A1 checkpoint: {path} (LL will not learn).")

  @staticmethod
  def _source_cols(src_in: int, skip: tuple[int, int]) -> list[int]:
    """Source column indices [0, src_in) with the ``skip=(start, width)`` block removed."""
    start, width = skip
    return list(range(start)) + list(range(start + width, src_in))

  @staticmethod
  def _partial_load(
    module: torch.nn.Module, source_sd: dict, n_trailing: int, src_cols: list[int]
  ) -> None:
    """Load ``source_sd`` into ``module``.

    Input-facing tensors (first layer weight + normalizer stats) take their shared
    columns from ``source[..., src_cols]`` (which already excludes any gap such as the
    actor's command block); the trailing ``n_trailing`` (goal) columns keep the
    target's fresh init. Every other same-shape tensor is copied verbatim.
    """
    input_keys = {
      "mlp.0.weight",
      "obs_normalizer._mean",
      "obs_normalizer._var",
      "obs_normalizer._std",
    }
    target_sd = module.state_dict()
    idx = torch.as_tensor(src_cols)
    new_sd: dict[str, torch.Tensor] = {}
    for key, tgt in target_sd.items():
      src = source_sd.get(key)
      if src is None:
        new_sd[key] = tgt
        continue
      if key in input_keys:
        shared = tgt.shape[-1] - n_trailing  # proprio/privileged width
        if len(src_cols) != shared:
          raise ValueError(
            f"Warm-start '{key}': mapped {len(src_cols)} source cols but target has "
            f"{shared} shared cols (input {tgt.shape[-1]} - trailing {n_trailing})."
          )
        merged = tgt.clone()
        merged[..., :shared] = src.index_select(-1, idx.to(src.device)).to(
          tgt.device, tgt.dtype
        )
        new_sd[key] = merged
      elif src.shape == tgt.shape:
        new_sd[key] = src.to(tgt.device, tgt.dtype)
      else:
        new_sd[key] = tgt
    module.load_state_dict(new_sd)

  def _make_high_level(self) -> HighLevel:
    if self.hl_algorithm == "oracle":
      return OracleHighLevel(self.goal_space, command_name="twist")
    if self.hl_algorithm == "ppo":
      obs = self.env.get_observations().to(self.device)
      return HighLevelPpo(
        self.goal_space,
        obs,
        self.env.num_envs,
        self.cfg["num_steps_per_env"] // self.c,
        self.goal_dim,
        self.gamma_hi,
        self.cfg["hl_ppo"],
        self.device,
        target_mode=self.hl_target_mode,
      )
    if self.hl_algorithm == "td3":
      obs = self.env.get_observations().to(self.device)
      return HighLevelTd3(
        self.goal_space,
        obs,
        self.env.num_envs,
        self.goal_dim,
        self.gamma_hi,
        self.cfg["hl_td3"],
        self.device,
        relabel=self.relabeling,
        ll_actor=getattr(self.alg, "_raw_actor", self.alg.actor),
        target_mode=self.hl_target_mode,
        obs_vel_dim=self._hl_vel_dim,
        cadence_dim=1 if self.hl_cadence_source == "hl" else 0,
        cadence_period_range=self.cadence_period_range,
        task_only_goals=self.hl_velocity_goals_only,
      )
    raise NotImplementedError(f"hl_algorithm='{self.hl_algorithm}' is unknown.")

  def get_inference_policy(self, device: str | None = None):
    """Hierarchy-aware inference policy for play/eval.

    The base implementation returns the bare LL actor — but the LL's ``goal`` obs is
    only written by :meth:`learn`, so in play it would stay frozen at zero and the
    command could never reach the robot. This override mirrors the training loop's
    goal wiring: fire ``hl.act_inference`` (deterministic, side-effect free) every
    ``c``-th call to refresh ``V*``, write the remaining delta into ``env.hrl_goal``/
    ``obs["goal"]`` each step, then run the LL actor.

    Known approximation (matches training): the window clock is not reset on episode
    resets, so after a mid-window reset the stale target persists for < c steps.
    """
    self.alg.eval_mode()
    self.hl.eval_mode()
    ll_policy = self.alg.get_policy().to(device)
    uenv = self.env.unwrapped
    step = 0
    target = None

    def policy(obs):
      nonlocal step, target
      state = self.goal_space.extract(uenv)
      if step % self.c == 0:
        # Eval is noise-free: feed the HL the clean lin-vel (deploy substitutes the
        # sportmode estimate). No-op unless hl_obs_vel is on. hl_vel_jitter (#8b HL
        # probe) is deliberately NOT applied here either -- same precedent as
        # GoalStateNoise (also absent from this function): DR is a train-time-only
        # mechanism, so play.py's [BENCH]/[GOALDIAG] eval measures the trained
        # policy's competence against ground truth, matching how A0 and every prior
        # A1a comparator (blind-HL, bias-only probe) were scored.
        if self.hl_obs_vel:
          obs["hl_vel"] = state[:, 0:2]
        target = self.hl.act_inference(uenv, obs, state)
        # A1a: command the stride period for this window. source='hl': act_inference
        # already wrote the HL's period (an eval pin overrides it, for the CoT(period)
        # sweep on a cadence-HL checkpoint); random source: pin or mid-range constant.
        # The LL entrains via the phase clock fed to mdp.phase / feet_gait.
        if self.hl_cadence:
          if self.hl.cadence_dim:
            if self.eval_cadence_period is not None:
              uenv.hrl_period.fill_(self.eval_cadence_period)
          else:
            lo, hi = self.cadence_period_range
            p = self.eval_cadence_period if self.eval_cadence_period is not None else (lo + hi) / 2.0
            uenv.hrl_period.fill_(p)
      step += 1
      if self.hl_cadence:
        uenv.hrl_phase = (uenv.hrl_phase + uenv.step_dt / uenv.hrl_period) % 1.0
      delta = target - state
      uenv.hrl_goal = delta
      obs["goal"] = delta
      return ll_policy(obs)

    return policy

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    # Training always derives the goal scale live (it tracks the twist curriculum), so a
    # resumed run must not inherit the frozen value load() pinned for inference.
    self.goal_space.freeze_scale(None)
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    # Frozen LL: keep it in eval (fixed weights + obs normalizer); only the HL learns.
    self.alg.eval_mode() if self.freeze_ll else self.alg.train_mode()
    ll_policy = self.alg.get_policy() if self.freeze_ll else None
    if self.freeze_ll:
      # One stochastic forward to populate the (constant) action distribution, so the
      # per-iter `action_std` logging works — deterministic forwards don't set it.
      with torch.inference_mode():
        ll_policy(obs, stochastic_output=True)
    self.hl.train_mode()
    self.logger.init_logging_writer()

    # Tracking-only HL reward (penalty-free; the probe showed the full task reward's
    # penalties collapse the HL to g≈0). Std cached once — term cfgs are static.
    hl_track = self.hl_reward_mode == "tracking"
    if hl_track:
      rm = self.env.unwrapped.reward_manager
      std_lin = rm.get_term_cfg("track_linear_velocity").params["std"]
      std_ang = rm.get_term_cfg("track_angular_velocity").params["std"]

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    prev_dones = None  # for the estimator-noise model's per-episode reset (#8b)
    # A1a S1c: per-env window accumulators for the command-gated CoT penalty in the HL
    # reward (windows align with rollouts: num_steps_per_env % c == 0 is enforced).
    win_energy = torch.zeros(self.env.num_envs, device=self.device)
    win_dist = torch.zeros_like(win_energy)
    for it in range(start_it, total_it):
      start = time.time()
      intrinsic_sum = 0.0
      goal_sum = posture_sum = action_rate_sum = cadence_sum = cot_pen_sum = 0.0
      stand_still_sum = angmom_sum = footslip_sum = footclear_sum = energy_sum = 0.0
      joint_acc_sum = 0.0
      joint_limits_sum = soft_landing_sum = body_ang_vel_sum = 0.0
      pitchref_sum = pushoff_sum = rollover_sum = 0.0
      symmetry_sum = mirror_sum = 0.0
      cot_energy = cot_dist = 0.0  # cost-of-transport metric accumulators (see loss_dict)
      with torch.inference_mode():
        uenv = self.env.unwrapped
        for k in range(self.cfg["num_steps_per_env"]):
          state = self.goal_space.extract(uenv)  # ground-truth (reward + HL stay clean)
          # Deploy-realistic estimator reading for the LL goal channel only (#8b). When
          # noise is off this equals `state` (noise_off == 0), so the path is unchanged.
          state_n = self.state_noise(state, prev_dones)
          noise_off = state_n - state
          # High level fires at the start of each window -> new absolute target V*.
          # Built from the clean state so the absolute-mode reward target stays privileged.
          if k % self.c == 0:
            # Optional HL lin-vel input: the SAME noisy estimate the LL conditions on
            # (state_n vx,vy), so HL and LL see one consistent deployable reading, PLUS
            # the HL-only leg-odometry jitter (#8b HL probe) on top -- resampled here,
            # once per fire, and held for the rest of this window (below). Applied to
            # state_n (not `state`) so the two DR mechanisms compose the way deploy will:
            # the real HL never sees ground truth either.
            self.hl_vel_jitter.resample()
            if self.hl_obs_vel:
              obs["hl_vel"] = self.hl_vel_jitter(state_n[:, 0:2])
            self._target = self.hl.act(uenv, obs, state)
            # Faithful delta DR (#8b): the LL-obs target base uses the noisy estimate so a
            # constant bias cancels in V*-s, matching deploy (V* = v_est(t0) + g, obs = V*-v_est).
            # Absolute V* is state-independent -> noise_off contributes nothing -> identical path.
            self._target_obs = (
              self._target + noise_off if self.hl_target_mode == "delta" else self._target
            )
            self.hl.begin_window(uenv, obs, state)

          # Low level observes the remaining delta V* - s_i on the noisy estimate.
          delta = self._target_obs - state_n
          uenv.hrl_goal = delta
          obs["goal"] = delta

          # Frozen LL acts via its deterministic mean (the competent walker; its trained
          # action std ~1.3 is far too noisy to sample). No rollout storage. The learning
          # LL uses act(), which samples and records the transition.
          actions = ll_policy(obs) if self.freeze_ll else self.alg.act(obs)
          # Record the per-step LL trace (proprio, goal state, action) for HIRO relabeling
          # (no-op unless the HL is a relabeling TD3). Pre-step values = what the LL saw.
          self.hl.record_step(obs["policy"], state, actions)
          # A1a: advance the commanded gait phase one control step so env.step's phase obs
          # reflects the current cadence clock.
          if self.hl_cadence:
            uenv.hrl_phase = (uenv.hrl_phase + uenv.step_dt / uenv.hrl_period) % 1.0
          obs, task_rew, dones, extras = self.env.step(actions.to(self.env.device))
          obs, task_rew, dones = (
            obs.to(self.device),
            task_rew.to(self.device),
            dones.to(self.device),
          )

          # Intrinsic = HIRO goal distance + the ADR-0002 upper-body penalties. The three
          # pieces are logged separately (each as its signed contribution) so that
          # intrinsic_reward = goal_reward + posture_pen + action_rate_pen exactly.
          achieved = self.goal_space.extract(uenv)
          r_goal = self.goal_space.reward(self._target, achieved, kernel=self.ll_goal_kernel)
          r_lo = r_goal
          if self.ll_task_reward_coef != 0.0:
            r_lo = r_lo + self.ll_task_reward_coef * task_rew
          if self.ll_alive_coef != 0.0:
            r_lo = r_lo + self.ll_alive_coef
          goal_sum += r_goal.mean().item()
          # Upper-body deploy-hygiene penalties (ADR-0002): keep arms+waist near default
          # and low-jitter so the deployed LL stops drifting/twisting the arms.
          if self.ll_posture_coef != 0.0:
            rd = uenv.scene["robot"].data
            dev = (rd.joint_pos[:, self._ub_joint_ids]
                   - rd.default_joint_pos[:, self._ub_joint_ids]).square()
            # Clamp per-joint err^2 (pre-weighting): unbounded L2 otherwise lets a rare
            # sim blow-up (joints -> huge) spike this to ~-5e5 and detonate the LL PPO
            # update in one step (2026-06-30 crash @ it9624). 9.0 = a joint ~pi off
            # default, so this only bites physical blow-ups; identical to the old
            # post-mean clamp for healthy states, weight-independent under ll_posture_weights.
            dev = (dev.clamp(max=9.0) * self._ub_weights).mean(dim=1)
            pose_pen = self.ll_posture_coef * dev
            r_lo = r_lo - pose_pen
            posture_sum += -pose_pen.mean().item()  # signed reward contribution (<= 0)
          if self.ll_action_rate_coef != 0.0:
            # Same catastrophe bound as posture (healthy ar ~265; 4000 = deep headroom).
            ar = (uenv.action_manager.action
                  - uenv.action_manager.prev_action).square().sum(dim=1).clamp(max=4000.0)
            ar_pen = self.ll_action_rate_coef * ar
            r_lo = r_lo - ar_pen
            action_rate_sum += -ar_pen.mean().item()  # signed reward contribution (<= 0)
          # WL-D arm 3 (2026-07-17): stand-still gating - mirrors A0's stand_still term
          # (joint deviation from default, gated |cmd| < threshold) into the LL intrinsic.
          if self.ll_stand_still_coef != 0.0:
            ss_pen = self.ll_stand_still_coef * mdp_rewards.stand_still(
              uenv, command_name="twist", command_threshold=0.1)
            r_lo = r_lo - ss_pen
            stand_still_sum += -ss_pen.mean().item()
          # WL-D arm 4a: mirror A0's angular_momentum_penalty (natural arm counter-swing).
          if self.ll_angmom_coef != 0.0:
            am_pen = self.ll_angmom_coef * mdp_rewards.angular_momentum_penalty(
              uenv, sensor_name="robot/root_angmom")
            r_lo = r_lo - am_pen
            angmom_sum += -am_pen.mean().item()
          # WL-D arm 4b: mirror A0's feet_slip (contact-time foot xy velocity penalty).
          if self.ll_footslip_coef != 0.0:
            fs_pen = self.ll_footslip_coef * mdp_rewards.feet_slip(
              uenv, sensor_name="feet_ground_contact", command_name="twist",
              command_threshold=0.1, asset_cfg=self._foot_asset_cfg)
            r_lo = r_lo - fs_pen
            footslip_sum += -fs_pen.mean().item()
          # WL-D arm 4c: mirror A0's feet_clearance (0.10m swing-height target).
          if self.ll_footclear_coef != 0.0:
            fc_pen = self.ll_footclear_coef * mdp_rewards.feet_clearance(
              uenv, target_height=0.10, command_name="twist",
              command_threshold=0.1, asset_cfg=self._foot_asset_cfg)
            r_lo = r_lo - fc_pen
            footclear_sum += -fc_pen.mean().item()
          # WL-D arm 4d: mirror the new cost_of_transport_penalty (the direct CoT mirror).
          if self.ll_energy_coef != 0.0:
            en_pen = self.ll_energy_coef * mdp_rewards.cost_of_transport_penalty(
              uenv, command_name="twist", command_threshold=0.1)
            r_lo = r_lo - en_pen
            energy_sum += -en_pen.mean().item()
          # WL-D (2026-07-24): mirror A0's joint_acc_l2 (whole-body joint-accel L2) - the
          # missing half of A0's smoothness stack (action_rate + joint_acc).
          if self.ll_joint_acc_coef != 0.0:
            ja_pen = self.ll_joint_acc_coef * joint_acc_l2(uenv)
            r_lo = r_lo - ja_pen
            joint_acc_sum += -ja_pen.mean().item()
          # WL-D: mirror A0's joint_pos_limits (soft-limit crossing penalty) - the
          # largest measured A1-vs-A0 episode-reward gap (~515x, WL-D reward-gap audit).
          if self.ll_joint_limits_coef != 0.0:
            jl_pen = self.ll_joint_limits_coef * joint_pos_limits(uenv)
            r_lo = r_lo - jl_pen
            joint_limits_sum += -jl_pen.mean().item()
          # WL-D: mirror A0's soft_landing (first-contact impact-force penalty).
          if self.ll_soft_landing_coef != 0.0:
            sl_pen = self.ll_soft_landing_coef * mdp_rewards.soft_landing(
              uenv, sensor_name="feet_ground_contact", command_name="twist",
              command_threshold=0.1)
            r_lo = r_lo - sl_pen
            soft_landing_sum += -sl_pen.mean().item()
          # WL-D: mirror A0's body_angular_velocity_penalty (torso xy angular velocity).
          if self.ll_body_ang_vel_coef != 0.0:
            bav_pen = self.ll_body_ang_vel_coef * mdp_rewards.body_angular_velocity_penalty(
              uenv, asset_cfg=self._torso_asset_cfg)
            r_lo = r_lo - bav_pen
            body_ang_vel_sum += -bav_pen.mean().item()
          # A1a cadence entrainment (ADR-0004): reward the LL for matching the contact
          # schedule of the HL-commanded stride period. Positive feet_gait term.
          if self.hl_cadence and self.ll_cadence_coef != 0.0:
            cad = mdp_rewards.feet_gait(uenv, use_commanded_phase=True, **self._gait_params)
            r_lo = r_lo + self.ll_cadence_coef * cad
            cadence_sum += (self.ll_cadence_coef * cad).mean().item()
          # WL-D arm 6 formulation A (2026-07-17): heel-to-toe ankle roll-over phase-lock
          # to the probe-measured heel-strike/toe-off reference, gated command + SCHEDULED
          # stance only (gate pin ii: phi is only well-defined on the schedule).
          if self.hl_cadence and self.ll_pitchref_coef != 0.0:
            pr = mdp_rewards.ankle_pushoff_pitchref(
              uenv, asset_cfg=self._ankle_asset_cfg,
              theta_hs=self.ll_pitchref_theta_hs, theta_to=self.ll_pitchref_theta_to,
              k=self.ll_pitchref_k, sigma=self.ll_pitchref_sigma,
              use_commanded_phase=True, **self._pitchref_gait_params)
            r_lo = r_lo + self.ll_pitchref_coef * pr
            pitchref_sum += (self.ll_pitchref_coef * pr).mean().item()
          # WL-D arm 6 formulation B (2026-07-17): ankle push-off power burst, gated
          # command + SCHEDULE AND ACTUAL CONTACT (stricter than A). Implemented but left
          # untrained (coef 0) until formulation A's read (A1a_plan.md Arm 6 decision).
          if self.hl_cadence and self.ll_pushoff_coef != 0.0:
            po = mdp_rewards.ankle_pushoff_power(
              uenv, asset_cfg=self._ankle_asset_cfg,
              w=self.ll_pushoff_w, p_scale=self.ll_pushoff_p_scale,
              use_commanded_phase=True, **self._gait_params)
            r_lo = r_lo + self.ll_pushoff_coef * po
            pushoff_sum += (self.ll_pushoff_coef * po).mean().item()
          # WL-D arm 6 formulation C (2026-07-24, position-based redesign 2026-07-29):
          # contact-sequence heel-to-toe roll-over, gated command + SCHEDULED stance
          # only (gate pin ii: phi is only well-defined on the schedule).
          if self.hl_cadence and self.ll_rollover_coef != 0.0:
            ro = mdp_rewards.heel_toe_rollover_contact(
              uenv, sensor_name="foot_subgeom_contact", asset_cfg=SceneEntityCfg("robot"),
              geom_ids=self._rollover_geom_ids, body_ids=self._rollover_body_ids,
              heel_x_max=self.ll_rollover_heel_x_max, toe_x_min=self.ll_rollover_toe_x_min,
              w=self.ll_rollover_w, use_commanded_phase=True, **self._pitchref_gait_params)
            r_lo = r_lo + self.ll_rollover_coef * ro
            rollover_sum += (self.ll_rollover_coef * ro).mean().item()
          # WL-D arm 10 formulation B (2026-07-20): step-time left/right symmetry index,
          # the primary training arm. Reset the per-env touchdown-time buffer for envs
          # that reset THIS step before reading this step's touchdowns (no RewardManager
          # drives this loop, so the runner resets it explicitly - see foot_step_symmetry).
          if self.ll_symmetry_coef != 0.0:
            self._symmetry.reset_envs(dones)
            sym = self._symmetry(
              uenv, sensor_name="feet_ground_contact", command_name="twist",
              command_threshold=0.1, sigma_si=self.ll_symmetry_sigma_si)
            r_lo = r_lo + self.ll_symmetry_coef * sym
            symmetry_sum += (self.ll_symmetry_coef * sym).mean().item()
          # WL-D arm 10 formulation A: implemented alongside B but left untrained
          # pending B's read (the arm 6 pattern). Requires hl_cadence (env.hrl_period).
          if self.hl_cadence and self.ll_mirror_coef != 0.0:
            self._mirror.reset_envs(dones)
            mir = self._mirror(
              uenv, left_asset_cfg=self._mirror_left_cfg, right_asset_cfg=self._mirror_right_cfg,
              sigma=self.ll_mirror_sigma, command_name="twist", command_threshold=0.1)
            r_lo = r_lo + self.ll_mirror_coef * mir
            mirror_sum += (self.ll_mirror_coef * mir).mean().item()
          # A1a: cost-of-transport training metric (dimensionless; gated to commanded motion).
          # Power sub-formula deduped onto the shared helper (2026-07-17, WL-D) - the
          # window/floor/signed-distance logic below stays a deliberately separate
          # formula from cost_of_transport_penalty (achieved distance, not commanded
          # speed; window-integrated, not per-step), so only the power term is shared.
          rd_ = uenv.scene["robot"].data
          _pw = mdp_rewards.mech_power(uenv.scene["robot"])
          _eng = (uenv.command_manager.get_command("twist")[:, :2].norm(dim=-1) > 0.1).float()
          _e_step = _pw * _eng * uenv.step_dt
          _d_step = rd_.root_link_lin_vel_b[:, :2].norm(dim=-1) * _eng * uenv.step_dt
          cot_energy += _e_step.sum().item()
          cot_dist += _d_step.sum().item()
          if self.hl_cot_coef != 0.0:  # S1c: per-env window CoT for the HL reward
            win_energy += _e_step
            # HL-reward distance = SIGNED displacement along the commanded direction
            # (undirected |v| let the cot1 HL earn cheap meters walking sideways,
            # 2026-07-07; signed so fwd/bwd oscillation nets ~0 — the window's
            # d_floor clamp handles a negative total). Metric above stays undirected.
            _cmd = uenv.command_manager.get_command("twist")[:, :2]
            _d_par = ((rd_.root_link_lin_vel_b[:, :2] * _cmd).sum(dim=-1)
                      / _cmd.norm(dim=-1).clamp(min=1e-6)) * _eng * uenv.step_dt
            win_dist += _d_par
            #win_dist += _d_step
          # Refresh the goal in the post-step obs (remaining delta at the new state)
          # so the normalizer/next-act input is consistent. Carry the same per-step
          # estimator offset so the stored next-obs goal matches what the LL conditions on.
          post_delta = self._target_obs - (achieved + noise_off)
          uenv.hrl_goal = post_delta
          obs["goal"] = post_delta
          # Refresh the HL lin-vel input on the post-step obs (same noisy estimate as the
          # LL's post_delta) so end_window's bootstrap next-state is consistent. Same
          # HL-only jitter SAMPLE as this window's fire (no resample() here) -- this obs
          # becomes `next_s` in HighLevelTd3.end_window (the buffer's stored next-state,
          # read verbatim by end_window/_relabel via `self._state_vec(obs)`). HIRO
          # relabeling (td3.py `_relabel`) never reads `hl_vel`/`_state_vec` at all -- it
          # only re-scores which GOAL best explains the stored LL trace
          # (`goal_state_seq`/`policy_seq`/`action_seq`), so a relabeled transition
          # automatically inherits whatever jitter realization the window was collected
          # under. Drawing a FRESH sample here instead would be actively wrong: it would
          # let the critic bootstrap from a v_est the actor never actually conditioned
          # its action on, decorrelating the stored (state, action) pair from what
          # happened in the rollout.
          if self.hl_obs_vel:
            obs["hl_vel"] = self.hl_vel_jitter((achieved + noise_off)[:, 0:2])

          if not self.freeze_ll:
            self.alg.process_env_step(obs, r_lo, dones, extras)
          # HL reward: full task reward (penalty-dominated) or velocity-tracking only.
          if hl_track:
            hl_rew = (mdp_rewards.track_linear_velocity(uenv, std_lin, "twist")
                      + mdp_rewards.track_angular_velocity(uenv, std_ang, "twist"))
          else:
            hl_rew = task_rew
          self.hl.accumulate(hl_rew)
          if (k + 1) % self.c == 0:
            # A1a S1c (ADR-0004): command-gated window CoT penalty into the HL reward.
            # Denominator = actual walked distance (going nowhere under command is
            # expensive), clamped to the distance a threshold-speed (0.1 m/s) walk
            # covers in one window so a stuck robot is costly but finite; a fully
            # gated-off (standing-command) window has zero energy -> exactly 0.
            if self.hl_cot_coef != 0.0:
              d_floor = 0.1 * self.c * uenv.step_dt
              cot_w = win_energy / (win_dist.clamp(min=d_floor) * 75.0 * 9.81)
              self.hl.accumulate(-self.hl_cot_coef * cot_w)
              cot_pen_sum += -(self.hl_cot_coef * cot_w).mean().item()
              win_energy.zero_()
              win_dist.zero_()
            self.hl.end_window(uenv, obs, achieved, dones, extras)

          # Episode bookkeeping uses task reward (A0-comparable); intrinsic is the
          # LL's actual training signal, logged separately via the loss dict.
          intrinsic_sum += r_lo.mean().item()
          self.logger.process_env_step(task_rew, dones, extras)
          prev_dones = dones  # envs that reset -> resample estimator bias next step
          if self.hl_cadence:
            # Phase resets with the episode. Period: random source resamples per episode
            # (one held stride period per episode; per-window random (< 1 stride) is
            # unfollowable — v1). With source='hl' the period is the HL's action and
            # persists a reset for < c steps until the next fire (same stale-window
            # approximation as the goal target).
            d = dones.bool()
            uenv.hrl_phase = torch.where(d, torch.zeros_like(uenv.hrl_phase), uenv.hrl_phase)
            if self.hl.cadence_dim == 0 and d.any():
              lo, hi = self.cadence_period_range
              new_p = torch.empty_like(uenv.hrl_period).uniform_(lo, hi)
              uenv.hrl_period = torch.where(d, new_p, uenv.hrl_period)

        collect_time = time.time() - start
        start = time.time()
        if not self.freeze_ll:
          self.alg.compute_returns(obs)

      ll_losses = {} if self.freeze_ll else self.alg.update()
      hl_losses = self.hl.update()
      n_steps = self.cfg["num_steps_per_env"]
      loss_dict = {
        **ll_losses,
        "intrinsic_reward": intrinsic_sum / n_steps,
        "ll/goal_reward": goal_sum / n_steps,
        "ll/posture_pen": posture_sum / n_steps,
        "ll/action_rate_pen": action_rate_sum / n_steps,
        "ll/cadence_rew": cadence_sum / n_steps,
        "ll/stand_still_pen": stand_still_sum / n_steps,
        "ll/angmom_pen": angmom_sum / n_steps,
        "ll/footslip_pen": footslip_sum / n_steps,
        "ll/footclear_pen": footclear_sum / n_steps,
        "ll/energy_pen": energy_sum / n_steps,
        "ll/joint_acc_pen": joint_acc_sum / n_steps,
        "ll/joint_limits_pen": joint_limits_sum / n_steps,
        "ll/soft_landing_pen": soft_landing_sum / n_steps,
        "ll/body_ang_vel_pen": body_ang_vel_sum / n_steps,
        "ll/pitchref_rew": pitchref_sum / n_steps,
        "ll/pushoff_rew": pushoff_sum / n_steps,
        "ll/rollover_rew": rollover_sum / n_steps,
        "ll/symmetry_rew": symmetry_sum / n_steps,
        "ll/mirror_rew": mirror_sum / n_steps,
        "metrics/cot": cot_energy / (cot_dist * 75.0 * 9.81 + 1e-6),  # dimensionless E/(m g d)
        "hl/cot_pen": cot_pen_sum / (n_steps // self.c),  # per-window mean (0 when off)
        **{f"hl/{k}": v for k, v in hl_losses.items()},
      }
      if self.hl_cadence:
        loss_dict["hl/period_mean"] = uenv.hrl_period.mean().item()
      learn_time = time.time() - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    saved_dict["hl"] = self.hl.state_dict()
    # Bake the goal scale: it defines what the HL's g means (V* = ref + scale*g), so it
    # travels WITH the policy instead of being re-derived from whatever command ranges the
    # replay env happens to carry (GoalSpace.freeze_scale). Live here — save() runs inside
    # learn(), where the scale tracks the curriculum.
    saved_dict["goal_scale"] = self.goal_space.scale(self.env.unwrapped).detach().cpu()
    torch.save(saved_dict, path)
    self.export_hierarchy_to_onnx(path.split("model")[0])
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def export_hierarchy_to_onnx(self, policy_path: str) -> None:
    """Export the two deploy nets the C++ hierarchy loads:

    * ``low_level.onnx``  — LL actor, obs = ``policy ++ goal`` (the remaining delta).
    * ``high_level.onnx`` — HL actor, obs = ``policy ++ command``, output = raw goal ``g``
      (ppo: Gaussian mean; td3: ``tanh``-bounded — both baked in by ``HighLevel.as_onnx``,
      so C++ is algorithm-agnostic). Skipped for the oracle (no network).

    A1 has no ``actor`` obs group (split into policy/command/goal); ``get_base_metadata``
    reads ``active_terms["actor"]``, so we alias it to each net's deploy obs for the
    metadata, then remove it (``active_terms`` is the live dict ``compute`` iterates)."""
    om = self.env.unwrapped.observation_manager
    # Low level: obs = policy ++ goal.
    om.active_terms["actor"] = om.active_terms["policy"] + om.active_terms["goal"]
    try:
      self.export_policy_to_onnx(policy_path, "low_level.onnx")
      self._attach_hrl_metadata(os.path.join(policy_path, "low_level.onnx"))
    finally:
      del om.active_terms["actor"]
    # High level: obs = policy ++ command.
    self._export_high_level_onnx(policy_path)

  def _export_high_level_onnx(self, policy_path: str) -> None:
    module = self.hl.as_onnx(verbose=False)
    if module is None:
      print("[HRL] HL has no network (oracle) -> high_level.onnx not exported; deploy "
            "reconstructs V* from the command + hl_target_mode.")
      return
    module = module.to("cpu")
    module.eval()
    os.makedirs(policy_path, exist_ok=True)
    out = os.path.join(policy_path, "high_level.onnx")
    torch.onnx.export(
      module, module.get_dummy_inputs(), out, export_params=True, opset_version=18,
      verbose=False, input_names=module.input_names, output_names=module.output_names,
      dynamic_axes={}, dynamo=False,
    )
    om = self.env.unwrapped.observation_manager
    om.active_terms["actor"] = om.active_terms["policy"] + om.active_terms["command"]
    try:
      self._attach_hrl_metadata(out)
    finally:
      del om.active_terms["actor"]
    print(f"[HRL] Exported high_level.onnx (hl_algorithm={self.hl_algorithm}, "
          f"hl_target_mode={self.hl_target_mode}, goal_dim={self.goal_dim})")

  def _attach_hrl_metadata(self, onnx_path: str) -> None:
    """Base export metadata (joint names/gains/scale + observation_names from the current
    ``actor`` alias) plus the HRL structure fields the deploy reads."""
    # logger_type is absent when exporting outside training (e.g. play.py --export-onnx,
    # where the runner has no wandb logger) -> fall back to a local run name.
    logger_type = getattr(self.logger, "logger_type", None)
    run_name: str = (
      wandb.run.name if logger_type == "wandb" and wandb.run else "local"
    )
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    metadata["c"] = self.c
    metadata["hl_algorithm"] = self.hl_algorithm
    metadata["hl_target_mode"] = self.hl_target_mode
    metadata["goal_components"] = [c.name for c in self.goal_space.components]
    # The g -> V* decode (V* = ref + scale*g). Deploy reads `goal_scale` from here instead of
    # re-deriving it from deploy.yaml's command ranges: those ranges are an operator knob
    # (the joystick safety clamp) and any edit would silently rescale the HL's action.
    metadata["goal_scale"] = [round(x, 6) for x in self.goal_space.scale(self.env.unwrapped).tolist()]
    # `absolute`-mode center, for diagnostics/parity. NOTE: unlike the scale (every entry is a
    # *difference*, hence frame-independent) this holds ABSOLUTE values in the TRAINING frame
    # — its height column is pelvis z (1.02) whereas the C++ deploy measures height at the imu
    # site (nominal_root_height 1.3076, offset cancels in the delta). Deploy therefore must
    # NOT adopt this vector wholesale; see deploy/robots/h1_2/include/hrl/goal_space.h.
    metadata["goal_center"] = [
      round(x, 6) for x in self.goal_space.center(self.env.unwrapped)[0].tolist()
    ]
    # Deploy must know the HL output layout: velocity-only HL emits task_dim(+period)
    # values and C++ fills orientation/height targets with their nominals.
    metadata["hl_velocity_goals_only"] = self.hl_velocity_goals_only
    if self.hl_cadence:
      # S1c: deploy must map the HL's extra tanh dim -> stride period over this range
      # (source='hl'), or run the fixed/mid-range clock (source='random').
      metadata["hl_cadence_source"] = self.hl_cadence_source
      metadata["cadence_period_range"] = list(self.cadence_period_range)
    attach_metadata_to_onnx(onnx_path, metadata)

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    saved = torch.load(path, map_location=map_location, weights_only=False)
    hl_state = saved.get("hl")
    if hl_state:
      self.hl.load_state_dict(hl_state)
    # Pin the baked goal scale so inference decodes g exactly as trained; learn() restores
    # the live path for resumed training. Absent on pre-2026-07-16 checkpoints -> play.py's
    # absence shim supplies the training value (same pattern as the hl_obs_vel shim).
    goal_scale = saved.get("goal_scale")
    if goal_scale is not None:
      self.goal_space.freeze_scale(goal_scale.to(self.device))
    return infos
