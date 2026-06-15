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


@dataclass
class HrlRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Hierarchical (A1) runner config. LL = inherited PPO; HL = fields below."""

  class_name: str = "HierarchicalRunner"

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
  relabeling: Literal["none", "hiro"] = "none"
  """HIRO off-policy correction (td3 only; ignored otherwise)."""
  gamma_hi: float = 0.99**8
  """High-level discount = gamma^c (horizon-matched)."""
  ll_task_reward_coef: float = 0.0
  """Blend of task reward into the LL intrinsic reward. 0 = pure HIRO."""
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
    # TEMPORARY (2026-06-15): default flipped to velocity-only (3-dim) for the frozen-LL
    # TD3 experiment (freeze the velocity-only oracle LL; the env goal obs dim must match
    # the frozen checkpoint, and the env derives it from this default at registration).
    # REVERT to 7-dim (drop this line) once that experiment concludes — the documented
    # default is velocity+orientation+height (orient/height are stabilizing regularizers
    # that track better; see plan doc §0). goal_dim stays derived from goal_components.
    goal_components=("velocity",),
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
