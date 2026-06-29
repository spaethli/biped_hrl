"""Off-policy TD3 high level (M4) for the hierarchical (A1) runner.

Why off-policy here: the naive on-policy ``HighLevelPpo`` (M3) fails to learn the
command->goal map. It gets only ``num_steps_per_env // c`` (= 3) HL transitions per env
per iteration, and the small tracking term in the dense task reward is swamped by
penalties, so the HL mean drifts to "stay put" (credit-assignment + co-training
instability; see ``doc/hrl/A1_findings.md`` "M3").

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
    relabel: str = "none",
    ll_actor: torch.nn.Module | None = None,
    target_mode: str = "delta",
    obs_vel_dim: int = 0,
  ) -> None:
    super().__init__(goal_space)
    self.device = device
    self.goal_dim = goal_dim
    self.gamma_hi = gamma_hi
    self.target_mode = target_mode
    # Optional extra HL input: the deployable base lin-vel estimate (vx,vy) the runner
    # writes to obs["hl_vel"]. 0 -> byte-identical to the proprio++command HL.
    self.obs_vel_dim = obs_vel_dim
    # HIRO off-policy correction: relabel sampled goals to the goal the *current* LL is
    # most likely to have produced the stored action trace for. Needs a live LL actor.
    self.relabel = relabel != "none"
    self.ll_actor = ll_actor
    self.num_candidates: int = cfg.get("num_candidates", 10)
    self.candidate_std: float = cfg.get("candidate_std", 0.5)
    if self.relabel and ll_actor is None:
      raise ValueError("relabel HL needs a reference to the LL actor.")
    # HL actor/critic state = deployable HL obs (proprio ++ command [++ lin-vel est]).
    self._state_dim = obs["policy"].shape[-1] + obs["command"].shape[-1] + obs_vel_dim

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

    self.buffer = HLReplayBuffer(
      cfg["buffer_capacity"], self._state_dim, goal_dim, device, relabel=self.relabel
    )
    self.training = True
    self._update_count = 0
    self._last_relabel_frac = 0.0  # diagnostic: fraction of goals changed by relabeling

    # Per-window scratch (set in act/begin/end_window).
    self._window_reward = torch.zeros(num_envs, device=device)
    self._win_state: torch.Tensor | None = None  # s_t at fire (buffer state)
    self._win_action: torch.Tensor | None = None  # g_t at fire (raw, [-1,1])
    self._last_act_abs = 0.0  # diagnostic: |g| mean of the last fire
    # Relabel-only per-window LL trace (proprio obs / goal-space state / LL action).
    self._seq_policy: list[torch.Tensor] = []
    self._seq_gstate: list[torch.Tensor] = []
    self._seq_action: list[torch.Tensor] = []

  # --- helpers --------------------------------------------------------------

  def _state_vec(self, obs: TensorDict) -> torch.Tensor:
    parts = [obs["policy"], obs["command"]]
    if self.obs_vel_dim:
      parts.append(obs["hl_vel"])  # deployable base lin-vel est, written by the runner
    return torch.cat(parts, dim=-1)

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
    return self.goal_space.to_target(env, state, g, self.target_mode)

  def act_inference(self, env, obs, state: torch.Tensor) -> torch.Tensor:
    # Deterministic actor; no exploration noise, no window scratch, no normalizer update.
    g = self._actor_forward(self.actor, self._state_vec(obs))
    return self.goal_space.to_target(env, state, g, self.target_mode)

  def begin_window(self, env, obs, state: torch.Tensor) -> None:
    del env, obs, state
    self._window_reward.zero_()
    if self.relabel:
      self._seq_policy.clear()
      self._seq_gstate.clear()
      self._seq_action.clear()

  def accumulate(self, task_reward: torch.Tensor) -> None:
    self._window_reward += task_reward

  def record_step(self, policy_obs, goal_state, action) -> None:
    if not self.relabel:
      return
    # Clone: env obs tensors may be views into buffers overwritten on the next step.
    self._seq_policy.append(policy_obs.clone())
    self._seq_gstate.append(goal_state.clone())
    self._seq_action.append(action.clone())

  def end_window(self, env, obs, state, dones: torch.Tensor, extras=None) -> None:
    next_s = self._state_vec(obs)
    # Bootstrap mask: only TRUE terminals cut the TD target. Truncations (time_outs,
    # incl. A1's fell_over) keep done=0 so y = R + gamma_hi * Q(s', a').
    done = dones.float()
    if extras is not None and "time_outs" in extras:
      done = done * (1.0 - extras["time_outs"].to(self.device).float())
    if not self.relabel:
      del env, state
      self.buffer.add(self._win_state, self._win_action, self._window_reward.clone(), next_s, done)
      return
    # Stack the window trace [N, c, ·] and store the HIRO relabel inputs alongside the
    # 5-tuple. `state` is the achieved goal-space state at window end (s_{t+c}).
    policy_seq = torch.stack(self._seq_policy, dim=1)
    goal_state_seq = torch.stack(self._seq_gstate, dim=1)
    action_seq = torch.stack(self._seq_action, dim=1)
    scale = self.goal_space.scale(env).expand(next_s.shape[0], self.goal_dim)
    # Store the absolute-target center too (state-independent ref for `absolute` mode);
    # in `delta` mode it is unused at relabel time. Captured here so a curriculum range
    # change can't desync it from the stored window.
    center = self.goal_space.center(env)
    self.buffer.add(
      self._win_state, self._win_action, self._window_reward.clone(), next_s, done,
      policy_seq, goal_state_seq, action_seq, scale, state, center,
    )

  def update(self) -> dict[str, float]:
    metrics = {"act_abs_mean": self._last_act_abs, "buffer_size": float(len(self.buffer))}
    if len(self.buffer) < self.learning_starts:
      return metrics

    q_loss_sum = 0.0
    actor_loss_sum = 0.0
    n_actor = 0
    for _ in range(self.n_grad_steps):
      batch = self.buffer.sample(self.batch_size)
      if self.relabel:
        # HIRO off-policy correction: relabel the stored goal before the critic update.
        batch = batch._replace(actions=self._relabel(batch))
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
    if self.relabel:
      metrics["relabel_frac"] = self._last_relabel_frac
    return metrics

  @torch.no_grad()
  def _relabel(self, batch) -> torch.Tensor:
    """HIRO off-policy correction. For each sampled transition, pick the candidate goal
    that maximizes the *current* LL's log-likelihood of the stored action sequence:
    g~ = argmax_g Σ_i log π_lo(a_i | s_i, V*(g) - s_i). Candidates = the stored goal, the
    empirical achieved delta, and Gaussian samples around it (all raw, clamped to [-1,1]).
    """
    B, c, _ = batch.action_seq.shape
    k = self.num_candidates
    s_t = batch.goal_state_seq[:, 0]  # [B, goal_dim], goal state at window start
    scale = batch.scale  # [B, goal_dim]
    # Reference for the g<->V* map: window-start state (delta) or stored center (absolute).
    # V* = ref + scale*g, so g_emp recovering the achieved V* is (achieved - ref)/scale.
    ref = batch.center if self.target_mode == "absolute" else s_t
    g_emp = ((batch.next_goal_state - ref) / scale).clamp(-1.0, 1.0)  # achieved as raw g

    # Candidate goals [B, k, goal_dim]: stored g, empirical g_emp, +(k-2) sampled.
    cand = torch.empty(B, k, self.goal_dim, device=self.device)
    cand[:, 0] = batch.actions
    cand[:, 1] = g_emp
    noise = torch.randn(B, k - 2, self.goal_dim, device=self.device) * self.candidate_std
    cand[:, 2:] = (g_emp.unsqueeze(1) + noise).clamp(-1.0, 1.0)

    # Reconstruct the LL goal obs the candidate would have produced each step:
    # V*_cand = ref + scale*g; delta_i = V*_cand - s_i. Flatten [B,k,c] for one forward.
    v_star = ref.unsqueeze(1) + scale.unsqueeze(1) * cand  # [B, k, goal_dim]
    delta = v_star.unsqueeze(2) - batch.goal_state_seq.unsqueeze(1)  # [B, k, c, goal_dim]
    policy = batch.policy_seq.unsqueeze(1).expand(B, k, c, -1)  # [B, k, c, policy_dim]
    actions = batch.action_seq.unsqueeze(1).expand(B, k, c, -1)  # [B, k, c, action_dim]

    m = B * k * c
    flat_obs = TensorDict(
      {"policy": policy.reshape(m, -1), "goal": delta.reshape(m, -1)},
      batch_size=[m],
      device=self.device,
    )
    self.ll_actor(flat_obs, stochastic_output=True)  # populate the LL action distribution
    logp = self.ll_actor.get_output_log_prob(actions.reshape(m, -1))
    logp = logp.reshape(B, k, c, -1).sum(dim=(-1, -2))  # [B, k], Σ over window (+ any dim)
    best = logp.argmax(dim=1)  # [B]
    self._last_relabel_frac = (best != 0).float().mean().item()
    return cand[torch.arange(B, device=self.device), best]

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

  def as_onnx(self, verbose: bool = False):
    del verbose
    return _HlTd3OnnxModule(self)


class _HlTd3OnnxModule(torch.nn.Module):
  """ONNX-export wrapper for the TD3 HL actor: ``g = tanh(actor(normalizer(x)))``, where
  ``x`` is the flat HL obs (``policy ++ command``) the deploy feeds. Mirrors
  :meth:`HighLevelTd3.act_inference` (deterministic; no exploration noise / normalizer
  update). Exposes the same ``get_dummy_inputs`` / ``input_names`` / ``output_names``
  interface as rsl_rl's ``_OnnxMLPModel`` so the runner exports both HL types through one
  ``torch.onnx.export`` path. The copied normalizer is set to eval at export time so it
  applies frozen stats."""

  def __init__(self, hl: HighLevelTd3) -> None:
    super().__init__()
    self.normalizer = copy.deepcopy(hl.normalizer)
    self.actor = copy.deepcopy(hl.actor)
    self.input_size = hl._state_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.actor(self.normalizer(x)))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
