"""RL configuration for Unitree H1_2 A1 (HIRO) task.

``HrlRunnerCfg`` extends the A0 on-policy runner cfg: the inherited
``actor``/``critic``/``algorithm`` describe the **low-level PPO** (identical to A0),
``obs_groups`` routes the hierarchical observation groups to each model, and the
extra fields configure the high level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.tasks.velocity.rl.hrl.goal_space import DEFAULT_GOAL_COMPONENTS


@dataclass
class HlPpoCfg:
  """High-level on-policy PPO config (used when ``hl_algorithm == "ppo"``).

  The HL actor sees ``policy`` (proprio, no command) ++ ``command`` and outputs the
  ``goal_dim`` directional goal; the critic sees the privileged ``critic`` group. The
  HL discount is ``HrlRunnerCfg.gamma_hi`` (not ``algorithm.gamma``, which is dropped).
  """

  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(256, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
        # std cap removed: with 4x tracking reward (A1 env cfg) the stronger HL task
        # gradient should self-regulate std (too-wide std misses the now-high-value
        # target). Watch for blowup; re-cap with std_range=(1e-3,1.0) if it recurs.
        # Cap the std so the HL goal can't blow up: at entropy_coef 0.02 the entropy
        # bonus swamped the weak HL task gradient and std ran away (goal_abs ~8.7 ->
        # impossible targets -> robot died). Capping at 1.0 makes blowup impossible.
        "std_range": (1e-3, 1.0),
      },
    )
  )
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(256, 256), activation="elu", obs_normalization=True
    )
  )
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,  # 0.005 collapsed HL to deterministic ("don't track"); 0.02
                          # blew std up (goals exploded, robot died). 0.01 + std cap.
                          # with 0.01 + no std_cap still blow up
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      lam=0.95,          # gamma is overridden by gamma_hi at build time
      desired_kl=0.01,
      max_grad_norm=1.0,
    )
  )


@dataclass
class Td3NetCfg:
  """Net spec for a TD3 actor/critic (self-contained MLP; no distribution/normalizer
  flags — TD3 owns a single shared input normalizer)."""

  hidden_dims: tuple[int, ...] = (256, 256)
  activation: str = "elu"


@dataclass
class HlTd3Cfg:
  """High-level off-policy TD3 config (used when ``hl_algorithm == "td3"``).

  The actor sees the deployable HL obs (``policy`` ++ ``command``) and outputs a
  ``tanh``-bounded goal ``g`` in ``[-1, 1]^goal_dim``; the window target is the HIRO
  delta ``V* = state + scale*g``. The twin critics take ``[norm(state), g]``. The HL
  discount is ``HrlRunnerCfg.gamma_hi``. A replay buffer (persisting across iterations)
  gives many more HL updates than the on-policy PPO HL — the fix for M3's failure.
  """

  actor: Td3NetCfg = field(default_factory=Td3NetCfg)
  critic: Td3NetCfg = field(default_factory=Td3NetCfg)
  actor_learning_rate: float = 3.0e-4
  """Run 1 used 1e-3 for both nets: the actor raced into the tanh bounds before the
  critic knew anything (|g| 0.88 by it250). The actor must move slower than the
  critic learns; 3e-4 is the standard TD3 value."""
  critic_learning_rate: float = 1.0e-3
  tau: float = 0.005
  """Soft target-update rate."""
  policy_freq: int = 2
  """Delayed actor update: update actor + targets every ``policy_freq`` critic steps."""
  expl_noise_std: float = 0.2
  """Gaussian exploration noise std added to the actor's ``[-1,1]`` goal at act time."""
  target_noise_std: float = 0.2
  """Target-policy smoothing noise std."""
  target_noise_clip: float = 0.5
  """Clip for the target-policy smoothing noise."""
  batch_size: int = 512
  n_grad_steps: int = 8
  """TD3 gradient steps per training iteration (sampled minibatches from the buffer)."""
  buffer_capacity: int = 500_000
  """Replay buffer capacity in HL transitions (~num_envs * num_steps_per_env//c per iter)."""
  learning_starts: int = 10_000
  """Don't update until the buffer holds at least this many transitions."""
  warmup_transitions: int = 100_000
  """While the buffer holds fewer transitions than this, act() emits uniform random
  goals g ~ U(-1,1) instead of the actor's output (TD3's start_timesteps). Run 1
  skipped this: exploration noise around the saturated actor mean left the buffer
  with zero interior-goal coverage, locking the saturation in (critic never learned
  that moderate goals are better). ~8 iters at 4096 envs."""
  num_candidates: int = 10
  """HIRO relabeling candidate count (used when ``relabeling == 'hiro'``): the stored
  goal, the empirical achieved delta, and (num_candidates - 2) Gaussian samples around
  it. The winner maximizes the current LL's log-likelihood of the stored action trace."""
  candidate_std: float = 0.5
  """Std (in raw [-1,1] goal units) of the Gaussian sampling for relabel candidates."""


@dataclass
class GoalStateNoiseCfg:
  """Estimator-noise model for the LL goal/observation channel (#8b — train on a
  deploy-realistic base-velocity estimate, ``rt/sportmodestate``, instead of sim
  ground-truth). Corrupts ONLY the goal-space columns the onboard estimator supplies
  (base linear velocity ``vx, vy``, optionally ``height``); yaw-rate/orientation come
  from the gyro/IMU and stay clean. The reward and HL always see ground-truth
  (privileged) — only the LL's observed goal delta ``V*-s`` is noised. Off by default
  (RQ2-safe). Meaningful in ``hl_target_mode=absolute`` only (a constant bias cancels in
  the ``delta`` map; see doc/hrl/hierarchy_benefit_roadmap.md #8b)."""

  enable: bool = False
  components: tuple[str, ...] = ("velocity",)
  """Goal components to corrupt. ``velocity`` noises ``vx, vy`` (not yaw-rate)."""
  bias_range: float = 0.10
  """Per-axis constant offset, resampled per episode ~ U(-bias_range, +bias_range) [m/s]."""
  drift_std: float = 0.0
  """Per-step OU innovation std [m/s]; 0 disables within-episode drift."""
  drift_decay: float = 0.99
  """OU mean-reversion factor (closer to 1 = slower-varying drift)."""
  lag_steps: float = 0.0
  """First-order low-pass time constant in control steps (sensor lag); 0 disables lag."""


@dataclass
class HlVelJitterCfg:
  """HL-only base-velocity jitter (leg-odometry probe, 2026-08-05). Corrupts ONLY
  ``obs["hl_vel"]`` (the TD3 HL's optional ``hl_obs_vel`` input) — never ``state_n``,
  so a `delta`-mode LL is provably unaffected (see
  ``tests/test_hl_vel_jitter_isolation.py``). A **sibling** of ``GoalStateNoiseCfg``,
  not a field on it: that class's noise lands on ``state_n`` (LL channel, per-step,
  cancels via ``noise_off`` only for constants); this one lands on the HL's velocity
  obs only (per-HL-fire, every ``c`` steps) and does NOT cancel for jitter (only a
  constant bias would). Mixing the two into one cfg would make it easy to wire a
  jitter term into the wrong channel. Off by default (RQ2-safe). Per-axis scalar
  fields (not a tuple) — the tyro CLI is finicky with tuple overrides (confirmed
  2026-07-03, ``cadence_period_range``)."""

  enable: bool = False
  bias_vx: float = 0.0
  """Fixed per-episode-INDEPENDENT vx offset — the odometry method's own systematic
  error (measured leg-odom bias: -0.038 m/s), added every call, never resampled."""
  bias_vy: float = 0.0
  jitter_std_vx: float = 0.0
  """Gaussian std (m/s) for the per-HL-window vx jitter, resampled once per HL fire
  (measured leg-odom RMS: 0.074 m/s)."""
  jitter_std_vy: float = 0.0
  """Measured leg-odom vy RMS: 0.049 m/s."""


@dataclass
class HrlRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Hierarchical (A1) runner config. LL = inherited PPO; HL = fields below."""

  class_name: str = "HierarchicalRunner"

  def __post_init__(self):
    # Horizon-match the HL discount to c. Unconditional (not `if None`): tyro carries the
    # factory-derived value forward when only `c` is overridden, so a conditional check
    # would keep the stale c=8 gamma. gamma_hi is horizon-matched by definition, so it is
    # always derived from c (not independently settable).
    parent_post = getattr(super(), "__post_init__", None)
    if parent_post is not None:
      parent_post()
    self.gamma_hi = 0.99**self.c

  # Observation routing: LL actor sees proprio+goal (no command); critic privileged+goal.
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": ("policy", "goal"),
      "critic": ("critic", "goal"),
    }
  )

  # High-level configuration.
  c: int = 8
  """High-level period in control steps (must divide num_steps_per_env)."""
  goal_components: tuple[str, ...] = DEFAULT_GOAL_COMPONENTS
  """Declarative goal space; goal_dim is derived from it. Default = velocity (3) +
  orientation (3) + height (1) = 7 dims."""
  goal_weights: dict[str, float] | None = None
  """Per-component reward weights (None -> all 1.0)."""
  hl_algorithm: Literal["oracle", "ppo", "td3"] = "td3"
  """High-level learner. **Default 'td3'** — the final A1 structure (TD3 + HIRO relabel).
  Historical milestones: M1/2 shipped 'oracle', M3 added 'ppo', M4 added 'td3'. Set
  'oracle' for the clean LL-isolation baseline (no learned HL, no relabeling)."""
  hl_ppo: HlPpoCfg = field(default_factory=HlPpoCfg)
  """High-level PPO config (used when hl_algorithm == 'ppo')."""
  hl_td3: HlTd3Cfg = field(default_factory=HlTd3Cfg)
  """High-level TD3 config (used when hl_algorithm == 'td3')."""
  goal_state_noise: GoalStateNoiseCfg = field(default_factory=GoalStateNoiseCfg)
  """Estimator-noise on the LL goal channel (#8b sim2real DR). Off by default."""
  hl_vel_jitter: HlVelJitterCfg = field(default_factory=HlVelJitterCfg)
  """HL-only leg-odometry-calibrated jitter on ``obs["hl_vel"]`` (2026-08-05 probe).
  Independent of ``goal_state_noise`` — see ``HlVelJitterCfg``. Off by default."""
  hl_vel_residual: HlVelJitterCfg = field(default_factory=HlVelJitterCfg)
  """Residual noise layered ON TOP of the simulated leg-odometry estimator (WL-F task 4).
  Only applies with ``hl_vel_source='leg_odom'``; off by default.

  Reuses ``HlVelJitterCfg`` but is NOT the same knob. ``hl_vel_jitter`` *replaces* the error
  model and is the exogenous comparator arm (mutually exclusive with ``leg_odom``); this one
  *adds* to a real, policy-coupled error, covering what sim does not reproduce — encoder
  noise, contact geometry, and differencing at 200 Hz where the robot gets ~991 Hz. Size it
  in quadrature against the hardware reference (``lo_vx`` std 0.316, ``lo_vy`` 0.173, bias
  -0.084) minus what the sim estimator already produces (``hl/vel_err_std_v*`` /
  ``hl/vel_err_bias_v*`` in the training log)."""
  hl_vel_source: Literal[
    "state", "leg_odom", "compl", "ekf", "ekf_grav", "ekf_rot", "jacobian", "ekf_att",
  ] = "state"
  """Where ``obs["hl_vel"]`` comes from (WL-F, 2026-08-09; six more arms added 2026-08-24).
  ``state`` (**default**, keeps every historical comparator reproducible): ground-truth base
  velocity, optionally corrupted by the exogenous ``hl_vel_jitter``. ``leg_odom``: the
  SIMULATED LEG-ODOMETRY ESTIMATOR (``rl/hrl/leg_odom.py``), computed from sim state at the
  physics rate and c-averaged per HL window exactly as the deployed C++ does. The other six
  names (``compl | ekf | ekf_grav | ekf_rot | jacobian | ekf_att``) select the matching arm
  from ``rl/hrl/base_estimators.py`` — the training-time port of the SAME seven arms
  ``deploy/robots/h1_2/include/hrl/base_estimators.h`` runs on hardware (``legodom`` there
  is this cfg's ``leg_odom``). Unlike ``leg_odom``, these are NOT bit-parity tested against
  the C++ — see that module's docstring for what IS checked and why fp32/approximate
  fidelity was judged sufficient for training.

  Why the option exists: on hardware the HL's velocity error is **not** exogenous noise.
  Leg odometry is computed from the legs, so its error is a function of the policy's own
  leg motion, and the loop closes (2026-08-09, 424 fires at zero command:
  ``corr(|lo_vx|_t, |tgt0|_t) = +0.703``, ``corr(|tgt0|_t, |lo_vx|_t+1) = +0.609``,
  escalating). Under jitter, thrashing the feet is free in sim. Any non-``state`` value is
  mutually exclusive with ``hl_vel_jitter`` — they are the two arms of that comparison, not
  a stack. Requires ``hl_obs_vel=True``. Does not change any network shape, so pre-2026-08-09
  checkpoints restore as ``state`` with no shim."""
  relabeling: Literal["none", "hiro"] = "hiro"
  """HIRO off-policy correction (td3 only; ignored otherwise). **Default 'hiro'** — part of
  the final A1 (TD3) structure. Only meaningful with ``hl_algorithm='td3'``."""
  hl_reward_mode: Literal["task", "tracking"] = "tracking"
  """What the learned HL accumulates as its per-window reward (ppo/td3 only). ``task``:
  the full env task reward summed over the window — but it is PENALTY-DOMINATED
  (joint/action penalties swamp the exp tracking term), so the probe (2026-06-16) showed the
  HL collapses to g≈0 ("ask for ~neutral velocity") and won't command forward. ``tracking``
  (**default**, the A1 structure lever since M4): velocity command-tracking only
  (``track_linear_velocity + track_angular_velocity``, the
  proven A0 exp terms) — the outcome the HL controls, with the LL-execution penalties removed
  (they are the LL's concern). The env reward is UNCHANGED (no RQ2 confound); only what the HL
  optimizes internally changes. Safe with ``hl_target_mode=absolute`` (V* is bounded to the
  command range, so a pure-positive tracking reward can't push unsafe targets)."""
  hl_target_mode: Literal["delta", "absolute"] = "delta"
  """How the learned HL maps its bounded goal g to the window target V* (ppo/td3 only;
  oracle ignores it — it always sets V*=command absolutely). ``delta`` (HIRO default):
  ``V*=state+scale*g`` — a STATE-DEPENDENT target. (NOTE: the old "delta makes the HL
  saturate |g|->1" claim was a *pre-tracking-reward* artifact — with ``hl_reward_mode=
  tracking`` delta is unsaturated and tracks well; ``tracking`` is the primary lever, NOT
  this flag. Correction 2026-06-24, see ``doc/hrl/A1_findings.md``.) ``absolute``:
  ``V*=center+scale*g`` with ``center``=command-range midpoint (velocity) / nominal
  (orient,height) — a STATIC command->g map; only *refines* delta+tracking (~0.14->0.098).
  The LL is unchanged either way (still observes V*-s_i)."""
  hl_obs_vel: bool = True
  """Feed the HL the deployable base lin-vel estimate (vx,vy) as extra obs (td3 only).
  Off -> HL input is ``policy ++ command`` (byte-identical; RQ2-safe). On (**default** since
  2026-07-10, the final A1 structure) ->
  ``policy ++ command ++ v_est``, where v_est is the SAME noisy estimate the LL conditions
  on (``state_n`` vx,vy under #8b noise, clean at eval). Motivated for ``hl_target_mode=delta``:
  the directional target ``V*=state+scale*g`` needs current velocity to pick g=(command-v)/scale,
  which a velocity-blind HL must otherwise guess (the residual directional HL wall). Changes the
  HL obs dim -> deploy ONNX/C++ must feed v_est in the same order (deferred)."""
  gamma_hi: float | None = None
  """High-level discount. None -> derived horizon-matched as ``0.99 ** c`` in
  ``__post_init__`` (so changing ``c`` rescales it automatically; the old hardcoded
  ``0.99**8`` silently mismatched any c != 8). Set explicitly to override."""
  ll_task_reward_coef: float = 0.0
  """Blend of task reward into the LL intrinsic reward. 0 = pure HIRO."""
  ll_action_rate_coef: float = 0.02
  """Weight on the whole-body action-rate penalty added to the LL intrinsic (ADR-0002); the
  env ``action_rate_l2`` never reaches the goal-only LL otherwise. **0.02, NOT A0's 0.05**:
  matching A0's 0.05 over-penalized A1's goal-only LL and spiked ``fell_over`` (~165); 0.02
  gives ``fell_over``≈0 (the keeper) — see ``doc/hrl/A1_findings.md`` (posture/ar row). The
  small divergence from A0's 0.05 is a deliberate A0-vs-A1 reward difference (note it in RQ2).
  0 disables (clean A1-baseline ablation)."""
  ll_posture_coef: float = 0.5
  """Weight on the deviation-from-default penalty added to the LL intrinsic: arms+waist
  (ADR-0002, deploy hygiene) **+ hip yaw/roll (2026-07-07)** — the goal space is heading-
  invariant, so without the hip anchor from-scratch LLs walk with a ~20° hip twist (A0
  pins the same joints via its tightest ``variable_posture`` stds). ~No-op for aligned
  warm-started policies (deviation ≈ 0). 0 disables."""
  ll_posture_weights: dict[str, float] | None = field(
    default_factory=lambda: {
      r".*shoulder.*": 16.0,
      r".*elbow.*": 4.0,
      r".*wrist.*": 4.0,
      r".*ankle_roll.*": 4.0,
    }
  )
  """Per-joint multipliers on the posture penalty (regex pattern -> weight), resolved
  against the anchored joint names at runner init; joints no pattern matches stay at 1.0.
  ``dev = (w * err^2).mean`` — NOT renormalized, so all-ones reproduces the uniform
  penalty and raising one group does not dilute the hip yaw/roll anchor. **Default since
  2026-07-17 (WL-D)**: the Stage D arm-calm lever (2026-07-14, D1/D2) — the uniform mean
  gives each of the 19 anchored joints ~coef/19 pull vs A0's per-joint-std pressure
  (~50-100x more on the shoulders) -> the A1a shoulder flail; shoulders 16 / elbow+wrist
  4 fixes it (pose_dev 10x down, validated x2). Promoted from a temp rl_cfg edit
  (structured tyro CLI overrides untrusted, the fix0p8 lesson) now that it dominates the
  keeper on holds+arms+tracking post-F2 (A1a_plan.md table f). Costs +10-28% CoT vs the
  old keeper — this is WL-D's new energy zero-point. Set ``None`` for the pre-D1 uniform
  baseline. RQ2: A1-only reward change, log next to the ar 0.02-vs-0.05 note.

  The ``ankle_roll`` entry (WL-D arm 5, 2026-07-17) is INERT unless
  ``ll_posture_anchor_ankle_roll=True`` (ankle roll is not in the anchored-joint set by
  default) — present here so the arm only needs the one flag flipped, no weight edit."""
  ll_posture_anchor_ankle_roll: bool = False
  """WL-D arm 5 (2026-07-17): add ``.*ankle_roll.*`` to the LL posture anchor's joint set
  (default anchor = arms+waist+hip yaw/roll; see ``HierarchicalRunner.__init__``).
  Targets a replay defect (ankles roll inward) with a small anchor pull; weight comes
  from ``ll_posture_weights``'s ``ankle_roll`` entry (4.0 default — the joint's range is
  only +-15 deg, far narrower than the shoulders, so it needs much less pull). Off by
  default (byte-identical to pre-arm-5 behavior)."""
  ll_posture_all_joints: bool = False
  """WL-D: extends the LL posture anchor from the arms+waist+hip-yaw/roll subset (plus
  ankle_roll if ``ll_posture_anchor_ankle_roll``) to EVERY joint - the closest available
  analog to A0's ``pose``/``variable_posture`` term (A0's single largest reward, +0.835,
  which pins ALL joints with per-joint speed-scheduled stds while A1 only anchors a
  subset). Motivated by the 2026-07-28 action-rate decomposition finding that A1's excess
  action rate is arms-heavy (arms share 0.304 vs A0's 0.115). When True this SUPERSEDES
  both the subset list and ``ll_posture_anchor_ankle_roll`` (all joints already includes
  ankle roll). ``ll_posture_weights`` still applies on top of whichever set is anchored;
  unmatched joints stay at 1.0. False by default (byte-identical to pre-existing
  behavior)."""
  ll_stand_still_coef: float = 0.0
  """WL-D arm 3 (2026-07-17): mirrors A0's ``stand_still`` term (joint deviation from
  default, gated ``|cmd| < command_threshold``) into the LL intrinsic - targets the
  "unsettled stepping" defect (the goal-only LL has no reason to fully stop stepping at
  a held near-zero command; the posture anchor above already pulls arms/waist/hips but
  is always-on, not stand-gated). Reuses ``mdp.stand_still`` unmodified. 0 disables."""
  ll_angmom_coef: float = 0.0
  """WL-D arm 4a (2026-07-17): mirrors A0's ``angular_momentum_penalty`` (whole-body
  angular momentum, encourages natural counter-swing arm motion) into the LL intrinsic.
  Targets the arm_vel energy gap (0.37-0.58 vs A0's 0.14) directly - the LL never sees
  this env term otherwise (``ll_task_reward_coef=0``). 0 disables."""
  ll_footslip_coef: float = 0.0
  """WL-D arm 4b (2026-07-17): mirrors A0's ``feet_slip`` (contact-time foot xy velocity
  penalty) into the LL intrinsic. Targets foot-quality/energy loss during stance.
  0 disables."""
  ll_footclear_coef: float = 0.0
  """WL-D arm 4c (2026-07-17): mirrors A0's ``feet_clearance`` (deviation from the 0.10m
  swing-height target, velocity-weighted) into the LL intrinsic. Targets the WL-E
  swing-clearance defect (A0-optB apex ~35% under target) at the LL level. 0 disables."""
  ll_energy_coef: float = 0.0
  """WL-D arm 4d (2026-07-17): mirrors the new ``cost_of_transport_penalty`` (see
  ``mdp/rewards.py``, same term the A0+energy S4' control uses) into the LL intrinsic -
  the most direct mirror candidate, since it targets the CoT gap itself rather than a
  proxy for it. 0 disables."""
  ll_joint_acc_coef: float = 0.0
  """WL-D (2026-07-24): mirror A0's ``joint_acc_l2`` (whole-body joint-acceleration L2, A0
  weight 2.5e-7) into the LL intrinsic - the one A0 smoothness term A1 never had (A0's
  2nd-largest penalty after ``action_rate_l2``). Targets the action-rate deploy gap
  (action-rate decomposition 2026-07-24: A1's twitch is a uniform arms-heavy floor +38% over
  A0, not just goal-stepping). The LL never sees this env term otherwise
  (``ll_task_reward_coef=0``). Start at A0's 2.5e-7; the LL intrinsic runs hotter than A0's
  task reward so it may read weak. 0 disables (byte-identical baseline)."""
  ll_joint_limits_coef: float = 0.0
  """WL-D: mirrors A0's ``joint_pos_limits`` (mjlab's soft-limit crossing penalty, A0
  weight -10.0) into the LL intrinsic - the largest measured A1-vs-A0 episode-reward gap
  in the whole reward set (A0 -0.00025 vs A1 -0.128, ~515x), consistent with the
  documented deploy failure where A1a rides its joint stops far more of a bridge session
  than A0. Uses the default asset_cfg (all joints), mirroring ``joint_acc_l2``'s call
  style. 0 disables (byte-identical baseline)."""
  ll_soft_landing_coef: float = 0.0
  """WL-D: mirrors A0's ``soft_landing`` (first-contact impact-force penalty, A0 weight
  -1e-3) into the LL intrinsic - targets the measured gap (A0 -0.024 vs A1 -0.085). Same
  params A0 uses: sensor_name="feet_ground_contact", command_name="twist",
  command_threshold=0.1. 0 disables."""
  ll_body_ang_vel_coef: float = 0.0
  """WL-D: mirrors A0's ``body_angular_velocity_penalty`` (torso xy angular velocity, A0
  weight -0.05) into the LL intrinsic - targets the measured gap (A0 -0.0085 vs A1
  -0.016). Resolved once against a torso_link asset_cfg (mirrors ``_foot_asset_cfg``'s
  idiom). 0 disables."""
  ll_pitchref_coef: float = 0.0
  """WL-D arm 6, formulation A (2026-07-17, approved): heel-to-toe ankle roll-over
  phase-locking - matches ``ankle_pitch`` to a raised-cosine reference interpolated
  between the measured heel-strike/toe-off angles over ``feet_gait``'s own scheduled-
  stance progress (never a second hardcoded duty threshold). Command + SCHEDULED-stance
  gated (not actual contact - phi is only well-defined on the schedule). Requires
  ``hl_cadence`` (reads ``env.hrl_phase``/``hrl_period``). See ``mdp.ankle_pushoff_pitchref``
  and ``A1a_plan.md`` Arm 6. 0 disables (byte-identical)."""
  ll_pitchref_theta_hs: float = -0.4913
  """Ankle_pitch angle (rad) at scheduled heel-strike.

  **Recalibrated 2026-07-22** (was -0.253). The original 2026-07-17 probe measured this
  off the D2 baseline and inferred its direction from a correlational signal smaller
  than its own noise, concluding "more negative = plantarflexion" - backwards. A
  forward-kinematics XML sweep (no policy/dynamics involved) proved the opposite:
  increasing ``ankle_pitch`` moves the toe DOWN (plantarflexion); decreasing moves it UP
  (dorsiflexion). Confirmed empirically 2026-07-22: training formulation A on the old
  (backwards) reference at coef 0.1/0.2 drove the LL toward dorsiflexion at toe-off even
  MORE consistently than the flawed reference itself asked for (93.7%/97% of bouts ended
  more dorsiflexed, not more plantarflexed) - the same class of bug formulation B had,
  confirmed rather than just suspected.

  Re-measured from the direction-corrected, genuinely-plantarflexing `pushoff0p5_pscale0p5`
  checkpoint (arm 6 formulation B, `P_scale=0.5`, the sweep's biggest correctly-signed
  ROM: +13.4 degrees) instead of the pre-arm-6 D2 baseline - this also fixes formulation
  A's second problem (the old reference's ~1.3 degree ROM was too small to ever be
  visible, even tracked perfectly). n=4969 scheduled heel-strike events, std 0.117."""
  ll_pitchref_theta_to: float = -0.2575
  """Ankle_pitch angle (rad) at scheduled toe-off.

  **Recalibrated 2026-07-22** (was -0.275, see `ll_pitchref_theta_hs` for the full
  direction-bug correction). Re-measured from the same direction-corrected
  `pushoff0p5_pscale0p5` checkpoint as `theta_hs` - `theta_to > theta_hs` now (ROM
  +0.234 rad, +13.4 degrees), i.e. the reference correctly trends toward plantarflexion
  (push-off) at toe-off, not dorsiflexion. n=5026 scheduled toe-off events, std 0.040.

  Caveat carried over from the source checkpoint (`A1a_plan.md` Arm 6): that checkpoint's
  push-off also had the worst ankle-roll (inversion) coupling of the whole sweep
  (roll/pitch ratio 1.78) - formulation A only reads/rewards `ankle_pitch`, so it cannot
  inherit that specific side effect mechanically, but the underlying whole-body policy
  tendency to compensate via roll when pitch is pushed hard may still show up here too -
  watch for it in formulation A's own roll-coupling check."""
  ll_pitchref_sigma: float = 0.15
  """Exp-kernel tolerance (rad) on the pitch-reference tracking error. Free knob (not
  probe-measured); anchored to ``env_cfgs.py``'s existing ``std_walking`` ankle_pitch
  tolerance (h1_2/env_cfgs.py, 0.15) as the natural in-codebase reference."""
  ll_pitchref_k: float = 2.0
  """Raised-cosine phase exponent - ``k>1`` concentrates the pitch rotation late in
  stance (matching the real ankle-angle curve: flat through midstance, rapid near
  push-off). Free knob (not probe-measured)."""
  ll_pushoff_coef: float = 0.0
  """WL-D arm 6, formulation B: ankle push-off power burst - a saturating bonus
  (``1 - exp(-ReLU(directed_P)/P_scale)``) on signed ankle-pitch power in the
  terminal-stance window, gated SCHEDULE **and** actual contact (stricter than formulation
  A - phase-only gating would let the LL harvest the bonus by driving the ankle in the
  air after an early liftoff) **and** the plantarflexion direction (``qd > 0`` -
  direction-corrected 2026-07-20, see ``mdp.ankle_pushoff_power`` docstring: the original
  formula rewarded a toe-lift dorsiflexion habit just as well as genuine push-off, caught
  via the user's visual replay of the first trained checkpoint). Requires ``hl_cadence``."""
  ll_pushoff_w: float = 0.175
  """Terminal-stance window width (``phi in [1-w, 1)``) - midpoint of the spec's
  0.15-0.2 range, matching the constants probe's own window."""
  ll_pushoff_p_scale: float = 3.0
  """Power scale (W) the saturating bonus is calibrated against.

  **Recalibrated 2026-07-20** (was 40.0): the original value was the D2 checkpoint's p90
  terminal-stance ReLU'd power WITHOUT the plantarflexion direction gate - re-measured
  WITH the corrected ``qd > 0`` gate, only 33.7% of terminal-window bouts show ANY
  plantarflexion-direction sample at all (D2 barely plantarflexes there), and that power
  is tiny: median 0.34W / p90 1.48W / max 17.96W (n=335 gated bouts) - the old 40W value
  would have starved the corrected reward of gradient (typical achievable power ~27x
  below the saturation point). 3.0W sits between the p90 and the observed max: high
  enough that the reward keeps differentiating past today's rare best cycles (P=18W is
  still not fully saturated, ``1-exp(-6)=0.9975`` - close but not flat), low enough that
  today's near-absent typical performance (median 0.34W) still gets a nonzero, legible
  gradient (``1-exp(-0.11)=0.10``) instead of ~0 everywhere.

  Original (now-superseded) reasoning, kept for provenance: the 40W value used p90 (not
  the median) of the ungated measurement so today's typical cycle wouldn't already sit
  near saturation and kill the improvement gradient - the same p90-over-median logic
  above, just applied to the wrong (ungated, direction-blind) power distribution."""
  ll_rollover_coef: float = 0.0
  """WL-D arm 6, formulation C: contact-sequence-based heel-to-toe push-off reward
  (2026-07-24, position-based redesign 2026-07-29). Rewards the actual foot contact
  sequence (toe in contact AND heel lifted) during terminal stance, directly measuring
  the "roll-over" defect rather than joint angle. Formulations A/B only shaped
  ``ankle_pitch`` and empirically produced simultaneous whole-foot liftoff despite
  correct angle rotation. Requires ``hl_cadence``.

  **Redesigned 2026-07-29**: the original per-geom heel(foot1/2)/toe(foot5/6) grouping
  was wrong - ``h1_2.xml`` shows those are capsules and 5 of the 7 sub-geoms per foot
  each span the *entire* foot length (heel to near-toe), varying mainly in the
  medial/lateral axis. That grouping's dominant learnable signal ended up being ankle
  ROLL, not pitch (confirmed by replay: the trained checkpoint rolled the foot inward
  rather than pitching it). Fixed by classifying each contact by its actual local
  x-POSITION along the capsule (see ``mdp.heel_toe_rollover_contact`` and
  ``ll_rollover_heel_x_max``/``ll_rollover_toe_x_min``) instead of by geom identity."""
  ll_rollover_w: float = 0.175
  """Terminal-stance window width (``phi in [1-w, 1)``) for formulation C, matching
  formulation B's window (0.175 = midpoint of the spec's 0.15-0.2 range)."""
  ll_rollover_heel_x_max: float = -0.03
  """Formulation C: a contact counts as "heel" if its position in the owning
  ``ankle_roll_link``'s local frame has x below this. The foot's collision capsules
  span local x in [-0.08 (heel), +0.17 (toe tip)] (``h1_2.xml``); -0.03 sits solidly in
  the rear third, leaving a neutral midfoot band before the toe threshold. Free knob,
  not yet probe-measured against real contact-position distributions."""
  ll_rollover_toe_x_min: float = 0.08
  """Formulation C: a contact counts as "toe" if its local x exceeds this - past the
  midfoot, into the region foot1/foot7 (the two short, toe-only capsules) always occupy
  when in contact. Free knob, not yet probe-measured."""
  ll_symmetry_coef: float = 0.0
  """WL-D arm 10, formulation B (2026-07-20, primary training arm): step-time
  left/right symmetry index penalty, ``r = exp(-SI^2/sigma_si^2)``, ``SI =
  |t_LR-t_RL|/(t_LR+t_RL)`` from the alternating left/right touchdown intervals.
  Targets the observed double-tap defect directly - a double-tap drives one interval
  toward 0, SI -> 1, reward -> 0. Dense (available every step once both intervals
  are known), contact-based, no history buffer. See ``mdp.foot_step_symmetry`` and
  ``A1a_plan.md`` "Arm 10". 0 disables (byte-identical)."""
  ll_symmetry_sigma_si: float = 0.06
  """SI tolerance - arm-10 probe (2026-07-20): healthy checkpoints (D2/arm4d/old-
  fix0p8) measure SI 0.010-0.016; the one severely double-tapping checkpoint measured
  0.235. 0.06 keeps healthy gaits near r~0.9-1.0 while strongly penalizing the
  observed defect magnitude. Free knob (not itself probe-measured)."""
  ll_mirror_coef: float = 0.0
  """WL-D arm 10, formulation A (2026-07-20): half-period phase-shifted joint mirror,
  ``r = exp(-mean_pairs(q_L(t) - q_R(t-tau))^2/sigma^2)``, ``tau = hrl_period/2``. The
  CRITICAL adaptation vs. the dead-code reference ``joint_mirror`` (A1a_plan.md "Arm
  10"): same-instant ``(q_L-q_R)^2`` would force the antiphase legs INTO phase
  (hopping); this compares to the OTHER leg's state half a stride ago instead.
  Implemented but left UNTRAINED (stays 0) pending formulation B's read - the arm 6
  pattern. Requires ``hl_cadence`` (reads ``env.hrl_period``). Sagittal pairs only
  (hip_pitch, knee, ankle_pitch) - the posture anchor already pins hip_yaw/hip_roll,
  so lateral pairs would partly duplicate existing machinery. See
  ``mdp.phaseshift_joint_mirror``."""
  ll_mirror_sigma: float = 0.15
  """Exp-kernel tolerance (rad) on the mirror-tracking error. Free knob (not probe-
  measured); anchored to ``ll_pitchref_sigma``'s own choice (same joint-angle
  tolerance family, ``env_cfgs.py``'s ``std_walking`` ankle_pitch tolerance)."""
  ll_goal_kernel: Literal["l2", "exp"] = "exp"
  """LL intrinsic reward kernel (``GoalSpace.reward``). ``l2`` = HIRO's negative goal
  distance (all warm-started baselines; requires ``fell_over=time_out``). ``exp`` =
  A0-parity positive-bounded kernel for from-scratch training (M3 of the no-warm-start
  track): velocity gets A0's two tracking exp kernels with V* in place of the command,
  orientation A0's quadratic penalty, height a weight-10 quadratic (no A0 analog).
  Pair with ``--env.terminations.fell-over.time-out False`` so death forfeits future
  positive value (with ``time_out`` the critic bootstraps a fantasy continuation at
  fallen states and dying stays ~free — the 2026-07-04/06 from-scratch basins:
  L2 collapse@13, alive-bonus plateau@25, task-blend plateau@65)."""
  ll_alive_coef: float = 0.0
  """Constant per-step alive bonus added to the LL intrinsic. The A1 analog of A0's
  survival economics (true terminal + ``is_terminated`` -200, both unusable here: a true
  terminal + negative intrinsic is the suicide attractor): with ``fell_over=time_out`` and
  a strictly negative intrinsic, a from-scratch LL converges to fall-at-spawn (2026-07-04
  scratch diagnostic, ep_len 13 after 10k it). Size it above the competent-policy per-step
  intrinsic magnitude (goal_rew ~-2.5) so walking is net positive while flailing (~-12)
  stays negative. 0 disables (all warm-started baselines)."""
  hl_cadence: bool = False
  """A1a (ADR-0004): give the HL a gait-cadence channel. The HL commands a stride ``period``
  (s) in ``cadence_period_range``; the LL observes the resulting phase clock (``mdp.phase``)
  and entrains via the ``feet_gait`` intrinsic term. A *gait reference*, not a goal-space
  component (no L2 achieved-state). Off -> byte-identical to current A1."""
  cadence_period_range: tuple[float, float] = (0.35, 1.0)
  """Stride-period bounds (s) the commanded cadence is sampled from, held **per episode**
  (random source) or spanned by the HL's period action (source='hl' — so this is also the S3
  HL command envelope; keep it inside the frozen LL's followable band). Progression: v1 wide
  (0.5..1.4) + per-*window* resample destroyed the warm-start; v2 narrow (0.5..0.7) entrained
  (the "no entrainment" reads were the structure_keys eval bug); v3 widened to (0.35..1.0);
  v5/v6 tried (0.35..1.3) for the d(T) slow band and both LOST entrainment (band is a
  curriculum variable — the unfollowable 1.0–1.3 slice diluted the signal); v7 stages back to
  (0.35..1.0). Old runs restore their own saved range via play.py structure_keys. (Set as the
  default: the tyro CLI tuple override is finicky — confirmed 2026-07-03, "Unrecognized
  options" under --agent.)"""
  ll_cadence_coef: float = 0.5
  """A1a (ADR-0004): weight on the ``feet_gait`` cadence-entrainment reward added to the LL
  intrinsic, keyed to the HL-commanded stride period (``hrl_phase``). Matches A0's foot_gait
  weight (0.5). Active only when ``hl_cadence``. 0 disables."""
  cadence_swing_time: float = 0.0
  """A1a d(T) duty schedule (slow-stride enabler, see A1a_plan "Gait geometry"): >0 makes the
  ``feet_gait`` stance threshold period-dependent, ``d(T) = clamp(1 - swing_time/T,
  *cadence_duty_range)`` — single-support time held ~= swing_time (s) like human walking,
  double stance absorbs long periods. 0.31 keeps the fast band byte-identical (the floor binds
  for T <= ~0.70 s) and lifts duty to the 0.70 cap by T ~= 1.06 s. Sim-only (reward schedule;
  the phase obs is unchanged, so deploy is untouched). 0 (default) = the fixed floor duty
  every pre-d(T) checkpoint trained with."""
  cadence_duty_range: tuple[float, float] = (0.56, 0.70)
  """Duty-factor clamp bounds for the d(T) schedule — the single source of truth for BOTH the
  ``feet_gait`` training reward and play.py's ``gait_match`` eval metric (restored via
  structure_keys so they can never diverge). The floor (0.56 = the trained A0/A1 stance
  threshold) doubles as the fixed duty when ``cadence_swing_time`` is 0; the cap 0.70 = human
  slow-walk duty. d <= 0.5 is the walk-run boundary (zero double support -> flight): lowering
  the floor below 0.5 demands RUNNING at short periods, which also needs ~2 m/s commands
  (Froude 0.5), a height-goal/termination tolerance review, and an LL retrain — not just this
  knob."""
  hl_cadence_source: Literal["random", "hl"] = "random"
  """A1a S1c (ADR-0004): where the commanded stride period comes from. ``random`` (default):
  held per episode, resampled uniformly from ``cadence_period_range`` on reset (the S2 setup,
  and what every pre-S1c checkpoint saved). ``hl``: the TD3 HL emits the period as one extra
  tanh action dim (HL action = goal_dim+1; the critics see it, so Q can rank periods for the
  CoT term), mapped affinely from [-1,1] to ``cadence_period_range`` and held for the window
  (phase integrates incrementally, so a period change never jumps the clock). Requires
  ``hl_cadence=True`` and ``hl_algorithm='td3'`` (loud error otherwise). Old checkpoints
  restore as ``random`` and keep their goal_dim-sized HL nets."""
  hl_cot_coef: float = 0.0
  """A1a (ADR-0004): weight on the (negative) dimensionless cost-of-transport penalty added
  to the HL *window* reward (window energy / (m g window walked-distance), each step gated by
  commanded linear speed > 0.1; a distance floor keeps stuck-under-command windows expensive
  but finite, and an all-gated-off window is exactly 0). HL-only; the LL stays a pure tracker
  (the decoupling claim). 0 disables (byte-identical HL reward). Set >0 in S3+."""
  hl_velocity_goals_only: bool = True
  """Posture-sag fix (2026-07-09): the TD3 HL learns only the velocity goal columns
  (action = task_dim(+period) instead of goal_dim(+period)); orientation/height targets are
  pinned to their nominal values at every fire — the oracle's target path. Rationale: a
  tracking-rewarded HL has no incentive to command nominal posture, and in delta mode a
  sagged height with g=0 is a rewarded steady state — the v2 R-round co-trains all walked
  bent-kneed (height_dev 0.18–0.29 vs the oracle's 0.005 with the same exp kernel). The LL
  is untouched (goal obs stays goal_dim; it simply always sees nominal O/H targets, which
  R4 shows it tracks to ~0.005). Requires ``hl_algorithm='td3'``. **Default True since
  2026-07-09** (user decision: nominal posture references are the intended design); set False for
  the legacy full-goal-authority HL. Checkpoints saved WITHOUT this key (pre-change) are
  restored as False by play.py's structure merge — their HL nets are goal_dim-sized."""
  warm_start_path: str | None = None
  """Path to an A0 checkpoint to warm-start the LL from. The shared proprio columns
  (and all deeper layers / output head / std) are copied; the goal input columns are
  left freshly initialised. None = train the LL from scratch. On-policy PPO at the LL
  is far less sample-efficient than HIRO's off-policy TD3, so warm-starting from the
  walking A0 policy skips the locomotion-discovery phase."""
  freeze_ll_path: str | None = None
  """Path to a converged A1 LL checkpoint to load IN FULL and FREEZE: the LL becomes a
  fixed obs->action function (deterministic actor mean) and does NOT learn; only the HL
  trains against it. The goal space must match the checkpoint's (e.g. velocity-only),
  else the load errors on shape mismatch. Mutually exclusive with warm_start_path.
  Removes the co-training spiral (the LL cannot degrade), isolating whether the HL can
  learn command->goal against a stationary, competent LL."""


def unitree_h1_2_hrl_runner_cfg() -> HrlRunnerCfg:
  """A1 runner cfg. Low level mirrors the A0 PPO agent exactly."""
  return HrlRunnerCfg(
    # Default goal space = velocity+orientation+height (DEFAULT_GOAL_COMPONENTS, 7-dim).
    # The velocity-only ablation tracked worse (esp. yaw) at equal survival, so orient/
    # height are kept as stabilizing regularizers. Velocity weighted 3x.
    goal_weights={"velocity": 3.0, "orientation": 1.0, "height": 1.0},
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,     # original was 0.01
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01, # 0.01 default (2026-07-09, exp-kernel scratch run validated it)
      max_grad_norm=1.0,
    ),
    experiment_name="h1_2_velocity_a1_v2",
    wandb_project="biped_hrl",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
