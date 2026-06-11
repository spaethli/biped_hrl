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
  """High-level learner. Milestone 1/2 ship 'oracle'; M3 adds 'ppo'."""
  hl_ppo: HlPpoCfg = field(default_factory=HlPpoCfg)
  """High-level PPO config (used when hl_algorithm == 'ppo')."""
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
