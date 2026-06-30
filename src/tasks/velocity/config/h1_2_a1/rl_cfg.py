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
  hl_algorithm: Literal["oracle", "ppo", "td3"] = "oracle"
  """High-level learner. M1/2 ship 'oracle'; M3 adds 'ppo'; M4 adds 'td3'."""
  hl_ppo: HlPpoCfg = field(default_factory=HlPpoCfg)
  """High-level PPO config (used when hl_algorithm == 'ppo')."""
  hl_td3: HlTd3Cfg = field(default_factory=HlTd3Cfg)
  """High-level TD3 config (used when hl_algorithm == 'td3')."""
  goal_state_noise: GoalStateNoiseCfg = field(default_factory=GoalStateNoiseCfg)
  """Estimator-noise on the LL goal channel (#8b sim2real DR). Off by default."""
  relabeling: Literal["none", "hiro"] = "none"
  """HIRO off-policy correction (td3 only; ignored otherwise)."""
  hl_reward_mode: Literal["task", "tracking"] = "task"
  """What the learned HL accumulates as its per-window reward (ppo/td3 only). ``task``
  (default): the full env task reward summed over the window — but it is PENALTY-DOMINATED
  (joint/action penalties swamp the exp tracking term), so the probe (2026-06-16) showed the
  HL collapses to g≈0 ("ask for ~neutral velocity") and won't command forward. ``tracking``:
  velocity command-tracking only (``track_linear_velocity + track_angular_velocity``, the
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
  hl_obs_vel: bool = False
  """Feed the HL the deployable base lin-vel estimate (vx,vy) as extra obs (td3 only).
  Off (default) -> HL input is ``policy ++ command`` (byte-identical; RQ2-safe). On ->
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
  ll_action_rate_coef: float = 0.05
  """Weight on the whole-body action-rate penalty added to the LL intrinsic (ADR-0002).
  Matches A0's ``action_rate_l2`` weight (0.05); the env term never reaches the goal-only
  LL otherwise. 0 disables (clean A1-baseline ablation)."""
  ll_posture_coef: float = 0.5
  """Weight on the arms+waist deviation-from-default penalty added to the LL intrinsic
  (ADR-0002), keeping the upper body deployable (no behind-the-back drift / wrist twist).
  No A0 analog (A0 uses the positive exp ``variable_posture``); 0.5 is a starting value to
  tune up until the arms settle. 0 disables."""
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
      desired_kl=0.005, #original kl was 0.01, best A0 was 0.005
      max_grad_norm=1.0,
    ),
    experiment_name="h1_2_velocity_a1",
    wandb_project="biped_hrl",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
