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
  """High-level learner. Milestone 1/2 ship 'oracle'."""
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
    # Survival components (orientation, height) outweigh velocity early so the LL
    # learns to stay upright before chasing velocity. Tunable per experiment.
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
      desired_kl=0.005, #original kl, not A0 0.005
      max_grad_norm=1.0,
    ),
    experiment_name="h1_2_velocity_a1",
    wandb_project="biped_hrl",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
