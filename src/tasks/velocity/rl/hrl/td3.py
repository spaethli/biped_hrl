"""Off-policy TD3 high level (M4) for the hierarchical (A1) runner.

Why off-policy here: the naive on-policy ``HighLevelPpo`` (M3) fails to learn the
command->goal map. It gets only ``num_steps_per_env // c`` (= 3) HL transitions per env
per iteration, and the small tracking term in the dense task reward is swamped by
penalties, so the HL mean drifts to "stay put" (credit-assignment + co-training
instability; see ``doc/A1_HIRO_implementation_plan.md`` §0 "M3 experiment results").

TD3 attacks both failure modes: a replay buffer gives orders-of-magnitude more HL
updates per environment step (transitions persist across iterations and are resampled),
and a *deterministic* actor + bounded exploration noise replaces entropy-driven std, so
there is no std-collapse / std-blowup basin to fall into.

Design (self-contained; small nets since the action is just a ``goal_dim`` goal):
  * Actor ``mu_hi``: MLP, ``tanh`` output -> raw goal ``g`` in ``[-1, 1]^goal_dim``.
    The window target is the HIRO delta ``V* = state + scale*g`` (same map as the PPO
    HL); ``scale`` (``GoalSpace.scale``) tracks the live twist curriculum.
  * Twin critics ``Q1, Q2``: MLP over ``[norm(state), g] -> 1``.
  * TD3 tricks: target nets (soft ``tau``), target-policy smoothing (clipped Gaussian on
    the next goal), clipped double-Q target, delayed actor update (``policy_freq``).
  * Input normalization is a single shared ``EmpiricalNormalization`` over the state,
    updated online in :meth:`act` and applied (outside the nets) to online and target
    forwards alike, so target deepcopies never hold stale normalizer stats.

Truncation handling matches the PPO HL: ``fell_over`` is a truncation in the A1 env, so
the buffer's ``done`` flag excludes ``time_outs`` -> the TD target bootstraps a fall
(``gamma_hi * Q``) instead of cutting it to 0, killing the suicide attractor.
"""

from __future__ import annotations

import copy

import torch
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization

from .goal_space import GoalSpace
from .high_level import HighLevel
from .storage import HLReplayBuffer


class HighLevelTd3(HighLevel):
  """Off-policy TD3 high level (M4). Slots into the existing ``HighLevel`` interface."""

  def __init__(
    self,
    goal_space: GoalSpace,
    obs: TensorDict,
    num_envs: int,
    goal_dim: int,
    gamma_hi: float,
    cfg: dict,
    device: str,
  ) -> None:
    super().__init__(goal_space)
    self.device = device
    self.goal_dim = goal_dim
    self.gamma_hi = gamma_hi
    # HL actor/critic state = deployable HL obs (proprio ++ command), same as the PPO HL.
    self._state_dim = obs["policy"].shape[-1] + obs["command"].shape[-1]

    self.tau: float = cfg["tau"]
    self.policy_freq: int = cfg["policy_freq"]
    self.expl_noise: float = cfg["expl_noise_std"]
    self.target_noise: float = cfg["target_noise_std"]
    self.target_noise_clip: float = cfg["target_noise_clip"]
    self.batch_size: int = cfg["batch_size"]
    self.n_grad_steps: int = cfg["n_grad_steps"]
    self.learning_starts: int = cfg["learning_starts"]
    self.warmup_transitions: int = cfg["warmup_transitions"]

    hidden = tuple(cfg["actor"]["hidden_dims"])
    act = cfg["actor"]["activation"]
    c_hidden = tuple(cfg["critic"]["hidden_dims"])
    c_act = cfg["critic"]["activation"]

    self.normalizer = EmpiricalNormalization(self._state_dim).to(device)
    self.actor = MLP(self._state_dim, goal_dim, hidden, act).to(device)
    self.q1 = MLP(self._state_dim + goal_dim, 1, c_hidden, c_act).to(device)
    self.q2 = MLP(self._state_dim + goal_dim, 1, c_hidden, c_act).to(device)
    self.actor_target = copy.deepcopy(self.actor)
    self.q1_target = copy.deepcopy(self.q1)
    self.q2_target = copy.deepcopy(self.q2)

    self.actor_opt = torch.optim.Adam(
      self.actor.parameters(), lr=cfg["actor_learning_rate"]
    )
    self.critic_opt = torch.optim.Adam(
      list(self.q1.parameters()) + list(self.q2.parameters()),
      lr=cfg["critic_learning_rate"],
    )

    self.buffer = HLReplayBuffer(cfg["buffer_capacity"], self._state_dim, goal_dim, device)
    self.training = True
    self._update_count = 0

    # Per-window scratch (set in act/begin/end_window).
    self._window_reward = torch.zeros(num_envs, device=device)
    self._win_state: torch.Tensor | None = None  # s_t at fire (buffer state)
    self._win_action: torch.Tensor | None = None  # g_t at fire (raw, [-1,1])
    self._last_act_abs = 0.0  # diagnostic: |g| mean of the last fire

  # --- helpers --------------------------------------------------------------

  def _state_vec(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs["policy"], obs["command"]], dim=-1)

  def _actor_forward(self, net: torch.nn.Module, state: torch.Tensor) -> torch.Tensor:
    return torch.tanh(net(self.normalizer(state)))

  def _q_forward(self, net: torch.nn.Module, state: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return net(torch.cat([self.normalizer(state), g], dim=-1))

  # --- HighLevel interface --------------------------------------------------

  def act(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    s_vec = self._state_vec(obs)
    if self.training:
      self.normalizer.update(s_vec)
    if self.training and len(self.buffer) < self.warmup_transitions:
      # Random warmup (TD3 start_timesteps): uniform goals give the critic coverage
      # of the whole goal box before the actor's output ever drives the env.
      g = torch.rand(s_vec.shape[0], self.goal_dim, device=self.device) * 2.0 - 1.0
    else:
      g = self._actor_forward(self.actor, s_vec)
      if self.training and self.expl_noise > 0.0:
        g = (g + torch.randn_like(g) * self.expl_noise).clamp_(-1.0, 1.0)
    self._win_state = s_vec
    self._win_action = g
    self._last_act_abs = g.abs().mean().item()
    scale = self.goal_space.scale(env)
    return state + scale * g

  def act_inference(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    # Deterministic actor; no exploration noise, no window scratch, no normalizer update.
    g = self._actor_forward(self.actor, self._state_vec(obs))
    return state + self.goal_space.scale(env) * g

  def begin_window(self, env, obs, state: torch.Tensor) -> None:
    del env, obs, state
    self._window_reward.zero_()

  def accumulate(self, task_reward: torch.Tensor) -> None:
    self._window_reward += task_reward

  def end_window(self, env, obs, state, dones: torch.Tensor, extras=None) -> None:
    del env, state
    next_s = self._state_vec(obs)
    # Bootstrap mask: only TRUE terminals cut the TD target. Truncations (time_outs,
    # incl. A1's fell_over) keep done=0 so y = R + gamma_hi * Q(s', a').
    done = dones.float()
    if extras is not None and "time_outs" in extras:
      done = done * (1.0 - extras["time_outs"].to(self.device).float())
    self.buffer.add(self._win_state, self._win_action, self._window_reward.clone(), next_s, done)

  def update(self) -> dict[str, float]:
    metrics = {"act_abs_mean": self._last_act_abs, "buffer_size": float(len(self.buffer))}
    if len(self.buffer) < self.learning_starts:
      return metrics

    q_loss_sum = 0.0
    actor_loss_sum = 0.0
    n_actor = 0
    for _ in range(self.n_grad_steps):
      batch = self.buffer.sample(self.batch_size)
      with torch.no_grad():
        noise = (torch.randn_like(batch.actions) * self.target_noise).clamp(
          -self.target_noise_clip, self.target_noise_clip
        )
        next_g = (self._actor_forward(self.actor_target, batch.next_states) + noise).clamp(-1.0, 1.0)
        q1t = self._q_forward(self.q1_target, batch.next_states, next_g).squeeze(-1)
        q2t = self._q_forward(self.q2_target, batch.next_states, next_g).squeeze(-1)
        target_q = torch.min(q1t, q2t)
        y = batch.rewards + self.gamma_hi * (1.0 - batch.dones) * target_q

      q1 = self._q_forward(self.q1, batch.states, batch.actions).squeeze(-1)
      q2 = self._q_forward(self.q2, batch.states, batch.actions).squeeze(-1)
      critic_loss = torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y)
      self.critic_opt.zero_grad()
      critic_loss.backward()
      self.critic_opt.step()
      q_loss_sum += critic_loss.item()

      self._update_count += 1
      if self._update_count % self.policy_freq == 0:
        actor_loss = -self._q_forward(self.q1, batch.states, self._actor_forward(self.actor, batch.states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        actor_loss_sum += actor_loss.item()
        n_actor += 1
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

    metrics["q_loss"] = q_loss_sum / self.n_grad_steps
    metrics["actor_loss"] = actor_loss_sum / max(n_actor, 1)
    metrics["q_value"] = y.mean().item()
    return metrics

  def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module) -> None:
    for p, tp in zip(online.parameters(), target.parameters()):
      tp.data.lerp_(p.data, self.tau)

  def state_dict(self) -> dict:
    return {
      "actor": self.actor.state_dict(),
      "q1": self.q1.state_dict(),
      "q2": self.q2.state_dict(),
      "actor_target": self.actor_target.state_dict(),
      "q1_target": self.q1_target.state_dict(),
      "q2_target": self.q2_target.state_dict(),
      "normalizer": self.normalizer.state_dict(),
      "actor_opt": self.actor_opt.state_dict(),
      "critic_opt": self.critic_opt.state_dict(),
    }

  def load_state_dict(self, state_dict: dict) -> None:
    self.actor.load_state_dict(state_dict["actor"])
    self.q1.load_state_dict(state_dict["q1"])
    self.q2.load_state_dict(state_dict["q2"])
    self.actor_target.load_state_dict(state_dict["actor_target"])
    self.q1_target.load_state_dict(state_dict["q1_target"])
    self.q2_target.load_state_dict(state_dict["q2_target"])
    self.normalizer.load_state_dict(state_dict["normalizer"])
    self.actor_opt.load_state_dict(state_dict["actor_opt"])
    self.critic_opt.load_state_dict(state_dict["critic_opt"])

  def train_mode(self) -> None:
    self.training = True
    self.normalizer.train()
    for m in (self.actor, self.q1, self.q2):
      m.train()

  def eval_mode(self) -> None:
    self.training = False
    self.normalizer.eval()
    for m in (self.actor, self.q1, self.q2):
      m.eval()
