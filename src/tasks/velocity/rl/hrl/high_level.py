"""High-level policy strategies for the hierarchical (A1) runner.

The ``HighLevel`` interface decouples the high-level decision-maker from the
co-training loop, so the same ``HierarchicalRunner`` drives any of:

  * ``OracleHighLevel`` (M1/M2)  — no learning; emits an absolute window target
    (command velocity + nominal upright/height) to validate the LL pipeline.
  * ``HighLevelPpo`` (M3, later) — on-policy PPO over the goal delta (2-level PPO).
  * ``HighLevelTd3`` (M4/M5, later) — off-policy TD3 + HIRO relabeling.

``act`` returns the **absolute window target** ``V*`` in goal space; the runner turns
it into the per-step remaining-delta observation and the intrinsic reward. For a
learned HL, ``V* = state + scale * g`` (g = the network's bounded delta output); the
oracle returns ``goal_space.oracle_target`` directly.

Only the oracle is implemented in milestone 1/2. The learned strategies are stubbed
so the runner contract is fixed now and the learners slot in without touching the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from .goal_space import GoalSpace


class HighLevel(ABC):
  """Interface between the co-training loop and a high-level decision-maker."""

  # A1a S1c: extra action dims for the stride-period channel (TD3 sets 1 when it
  # commands the period; 0 = the runner's random per-episode source, the S2 setup).
  cadence_dim: int = 0

  def __init__(self, goal_space: GoalSpace) -> None:
    self.goal_space = goal_space

  @abstractmethod
  def act(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    """Fire the high level. Returns the absolute window target ``V*`` [N, goal_dim].

    ``obs`` is the current observation TensorDict (learned HLs read their input
    groups from it; the oracle ignores it). ``state`` is ``goal_space.extract(env)``
    at the fire step (learned HLs form ``V* = state + scale*g``; the oracle ignores it).
    """

  def act_inference(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    """Deployment-faithful fire: deterministic, no exploration noise, no transition
    storage, no normalizer updates. Learned HLs override; the oracle's ``act`` is
    already side-effect free, so the default delegates."""
    return self.act(env, obs, state)

  def begin_window(self, env, obs, state: torch.Tensor) -> None:
    """Called at HL fire, after :meth:`act`, to open a transition window."""

  def accumulate(self, task_reward: torch.Tensor) -> None:
    """Accumulate per-step task reward into the open HL transition."""

  def record_step(
    self, policy_obs: torch.Tensor, goal_state: torch.Tensor, action: torch.Tensor
  ) -> None:
    """Record the per-step LL trace (proprio obs, goal-space state, LL action) for the
    open window. Only the relabeling TD3 HL uses it (to score candidate goals by the LL
    action likelihood); no-op otherwise."""

  def end_window(self, env, obs, state: torch.Tensor, dones: torch.Tensor, extras=None) -> None:
    """Close the HL window and store/push the transition (no-op for oracle).

    ``extras`` carries the env step extras (e.g. ``time_outs``) for truncation
    bootstrapping; the oracle ignores it."""

  def update(self) -> dict[str, float]:
    """Run HL learning. Returns a loss/metric dict (empty for oracle)."""
    return {}

  def state_dict(self) -> dict:
    return {}

  def load_state_dict(self, state_dict: dict) -> None:
    del state_dict

  def train_mode(self) -> None:
    pass

  def eval_mode(self) -> None:
    pass

  def as_onnx(self, verbose: bool = False):
    """Return an ONNX-exportable deterministic module (flat obs vector -> goal ``g``),
    mirroring :meth:`act_inference`'s network math, or ``None`` if this HL has no
    network (the oracle: deploy reconstructs ``V*`` from the command + ``hl_target_mode``
    with no learned weights). The module's input is the HL obs (``policy ++ command``)
    concatenated in that order; output is the raw goal ``g`` (the runner/deploy then maps
    ``g -> V*`` via ``goal_space.to_target``)."""
    del verbose
    return None


class OracleHighLevel(HighLevel):
  """Non-learning HL: emit the absolute target = command velocity + nominal pose.

  Validates the entire LL pipeline (goal-space extraction, delta injection, intrinsic
  reward) and should reach ~A0 behaviour: the goal asks the body to track the
  commanded velocity while staying upright at nominal height.
  """

  def __init__(self, goal_space: GoalSpace, command_name: str = "twist") -> None:
    super().__init__(goal_space)
    self.command_name = command_name

  def act(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    del obs, state
    command = env.command_manager.get_command(self.command_name)
    return self.goal_space.oracle_target(env, command)


# Keys in an RslRlModelCfg / RslRlPpoAlgorithmCfg dict that MLPModel / PPO do not
# accept as kwargs (they are consumed by rsl_rl's construct_algorithm path, which we
# bypass to build the HL with a different action dim and obs routing).
_MLP_DROP = ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers", "class_name")
_PPO_DROP = ("class_name", "share_cnn_encoders", "gamma")


class HighLevelPpo(HighLevel):
  """On-policy PPO high level (M3): the naive 2-level-PPO hierarchy.

  A second standard ``rsl_rl.PPO`` running at the HL timescale (one transition per
  ``c`` LL steps). It *sees* the command (its task input) and emits a bounded,
  directional goal ``g``; the window target is the HIRO delta ``V* = state + scale*g``.
  Its reward is the env task reward summed over the window; its discount is ``gamma_hi``
  (= ``gamma**c``) so HL and LL horizons align.

  Obs routing: actor = ``policy`` (proprio, no command) ++ ``command``; critic =
  ``critic`` (privileged; already contains command). Storage holds
  ``num_steps_per_env // c`` transitions per env, mirroring the LL ``act ->
  accumulate -> end_window -> update`` cycle.
  """

  def __init__(
    self,
    goal_space: GoalSpace,
    obs: TensorDict,
    num_envs: int,
    num_transitions: int,
    goal_dim: int,
    gamma_hi: float,
    cfg: dict,
    device: str,
    target_mode: str = "delta",
  ) -> None:
    super().__init__(goal_space)
    self.device = device
    self.goal_dim = goal_dim
    self.target_mode = target_mode
    obs_groups = {"actor": ["policy", "command"], "critic": ["critic"]}

    actor_cfg = {k: v for k, v in cfg["actor"].items() if k not in _MLP_DROP}
    critic_cfg = {k: v for k, v in cfg["critic"].items() if k not in _MLP_DROP}
    actor = MLPModel(obs, obs_groups, "actor", goal_dim, **actor_cfg).to(device)
    critic = MLPModel(obs, obs_groups, "critic", 1, **critic_cfg).to(device)
    storage = RolloutStorage("rl", num_envs, num_transitions, obs, [goal_dim], device)
    alg_cfg = {k: v for k, v in cfg["algorithm"].items() if k not in _PPO_DROP}
    self.ppo = PPO(actor, critic, storage, device=device, gamma=gamma_hi, **alg_cfg)

    # Task reward accumulated over the open window, and the bootstrap obs for the
    # final window (the obs the next HL fire would see), set in :meth:`end_window`.
    self._window_reward = torch.zeros(num_envs, device=device)
    self._last_obs: TensorDict | None = None

  def act(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    # Sample the directional goal g and store the HL transition (obs/action/value/
    # logprob) in self.ppo.transition; finalized at window close.
    g = self.ppo.act(obs)
    self.ppo.actor.update_normalization(obs)
    self.ppo.critic.update_normalization(obs)
    return self.goal_space.to_target(env, state, g, self.target_mode)

  def act_inference(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    # Deterministic mean goal; no transition storage / normalizer updates.
    g = self.ppo.actor(obs)
    return self.goal_space.to_target(env, state, g, self.target_mode)

  def begin_window(self, env, obs, state: torch.Tensor) -> None:
    del env, obs, state
    self._window_reward.zero_()

  def accumulate(self, task_reward: torch.Tensor) -> None:
    self._window_reward += task_reward

  def end_window(self, env, obs, state, dones: torch.Tensor, extras=None) -> None:
    del env, state
    # Reward = Σ task reward over the window. Bootstrap truncations (e.g. fell_over,
    # which A1 marks time_out) by adding gamma_hi * V(s_t) so a fall is not treated as
    # a true terminal — mirrors rsl_rl's per-step time-out handling at the HL scale.
    rewards = self._window_reward.clone()
    if extras is not None and "time_outs" in extras:
      time_outs = extras["time_outs"].to(self.device)
      rewards = rewards + self.ppo.gamma * self.ppo.transition.values.squeeze(-1) * time_outs
    self.ppo.transition.rewards = rewards
    self.ppo.transition.dones = dones
    self.ppo.storage.add_transition(self.ppo.transition)
    self.ppo.transition.clear()
    self._last_obs = obs

  def update(self) -> dict[str, float]:
    goal_abs_mean = self.ppo.storage.actions.abs().mean().item()
    self.ppo.compute_returns(self._last_obs)
    losses = self.ppo.update()
    losses["goal_abs_mean"] = goal_abs_mean
    return losses

  def state_dict(self) -> dict:
    return self.ppo.save()

  def load_state_dict(self, state_dict: dict) -> None:
    self.ppo.load(state_dict, load_cfg=None, strict=True)

  def train_mode(self) -> None:
    self.ppo.train_mode()

  def eval_mode(self) -> None:
    self.ppo.eval_mode()

  def as_onnx(self, verbose: bool = False):
    # The HL actor is a standard MLPModel (same class as the LL): its ONNX wrapper bakes
    # in the obs normalizer + deterministic mean (= g). Input = policy ++ command.
    return self.ppo.actor.as_onnx(verbose)
