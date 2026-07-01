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
import time

import torch
import wandb

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from ..runner import VelocityOnPolicyRunner
from .goal_space import build_goal_space, init_goal_buffer
from .high_level import HighLevel, HighLevelPpo, OracleHighLevel
from .state_noise import GoalStateNoise
from .td3 import HighLevelTd3
from ...mdp import rewards as mdp_rewards


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
    # A1a (ADR-0004): HL gait-cadence channel + cost-of-transport HL objective.
    self.hl_cadence: bool = train_cfg.get("hl_cadence", False)
    self.cadence_period_range = tuple(train_cfg.get("cadence_period_range", (0.5, 1.4)))
    self.ll_cadence_coef: float = train_cfg.get("ll_cadence_coef", 0.0)
    self.hl_cot_coef: float = train_cfg.get("hl_cot_coef", 0.0)
    # Eval-only: pin the commanded stride period (for the CoT(period) sweep). None ->
    # mid-range constant (oracle/random-trained LL) or, later, the learned HL's action (S1c).
    self.eval_cadence_period: float | None = None
    # Gait params for the LL-intrinsic feet_gait term (match A0's foot_gait; period is
    # ignored when keyed to the commanded phase via use_commanded_phase=True).
    self._gait_params = dict(period=0.6, offset=[0.0, 0.5], threshold=0.56,
                             command_threshold=0.1, command_name="twist",
                             sensor_name="feet_ground_contact")
    ub_ids, _ = env.unwrapped.scene["robot"].find_joints(
      [".*shoulder.*", ".*elbow.*", ".*wrist.*", ".*torso.*"]
    )
    self._ub_joint_ids = torch.as_tensor(ub_ids, device=device)
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
        # sportmode estimate). No-op unless hl_obs_vel is on.
        if self.hl_obs_vel:
          obs["hl_vel"] = state[:, 0:2]
        target = self.hl.act_inference(uenv, obs, state)
        # A1a: command the stride period for this window (eval pins it for the CoT(period)
        # sweep; else the mid-range constant). The LL entrains via the phase clock fed to
        # mdp.phase / feet_gait.
        if self.hl_cadence:
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
    for it in range(start_it, total_it):
      start = time.time()
      intrinsic_sum = 0.0
      goal_sum = posture_sum = action_rate_sum = cadence_sum = 0.0
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
            # (state_n vx,vy), so HL and LL see one consistent deployable reading.
            if self.hl_obs_vel:
              obs["hl_vel"] = state_n[:, 0:2]
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
          r_goal = self.goal_space.reward(self._target, achieved)
          r_lo = r_goal
          if self.ll_task_reward_coef != 0.0:
            r_lo = r_lo + self.ll_task_reward_coef * task_rew
          goal_sum += r_goal.mean().item()
          # Upper-body deploy-hygiene penalties (ADR-0002): keep arms+waist near default
          # and low-jitter so the deployed LL stops drifting/twisting the arms.
          if self.ll_posture_coef != 0.0:
            rd = uenv.scene["robot"].data
            dev = (rd.joint_pos[:, self._ub_joint_ids]
                   - rd.default_joint_pos[:, self._ub_joint_ids]).square().mean(dim=1)
            # Clamp per-env deviation: unbounded L2 otherwise lets a rare sim blow-up
            # (joints -> huge) spike this to ~-5e5 and detonate the LL PPO update in one
            # step (2026-06-30 crash @ it9624). Healthy dev ~0.36 rad^2; 9.0 = all
            # upper-body joints ~pi off default, so this only bites physical blow-ups.
            dev = dev.clamp(max=9.0)
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
          # A1a cadence entrainment (ADR-0004): reward the LL for matching the contact
          # schedule of the HL-commanded stride period. Positive feet_gait term.
          if self.hl_cadence and self.ll_cadence_coef != 0.0:
            cad = mdp_rewards.feet_gait(uenv, use_commanded_phase=True, **self._gait_params)
            r_lo = r_lo + self.ll_cadence_coef * cad
            cadence_sum += (self.ll_cadence_coef * cad).mean().item()
          # A1a: cost-of-transport training metric (dimensionless; gated to commanded motion).
          rd_ = uenv.scene["robot"].data
          _pw = (rd_.qfrc_actuator * rd_.joint_vel).abs().sum(dim=1)
          _eng = (uenv.command_manager.get_command("twist")[:, :2].norm(dim=-1) > 0.1).float()
          cot_energy += (_pw * _eng).sum().item() * uenv.step_dt
          cot_dist += (rd_.root_link_lin_vel_b[:, :2].norm(dim=-1) * _eng).sum().item() * uenv.step_dt
          # Refresh the goal in the post-step obs (remaining delta at the new state)
          # so the normalizer/next-act input is consistent. Carry the same per-step
          # estimator offset so the stored next-obs goal matches what the LL conditions on.
          post_delta = self._target_obs - (achieved + noise_off)
          uenv.hrl_goal = post_delta
          obs["goal"] = post_delta
          # Refresh the HL lin-vel input on the post-step obs (same noisy estimate as the
          # LL's post_delta) so end_window's bootstrap next-state is consistent.
          if self.hl_obs_vel:
            obs["hl_vel"] = (achieved + noise_off)[:, 0:2]

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
            self.hl.end_window(uenv, obs, achieved, dones, extras)

          # Episode bookkeeping uses task reward (A0-comparable); intrinsic is the
          # LL's actual training signal, logged separately via the loss dict.
          intrinsic_sum += r_lo.mean().item()
          self.logger.process_env_step(task_rew, dones, extras)
          prev_dones = dones  # envs that reset -> resample estimator bias next step
          if self.hl_cadence:
            # Per-episode cadence: one held commanded stride period per episode (resampled on
            # reset, phase reset too). Per-window resampling (< 1 stride) is unfollowable.
            d = dones.bool()
            uenv.hrl_phase = torch.where(d, torch.zeros_like(uenv.hrl_phase), uenv.hrl_phase)
            if d.any():
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
        "metrics/cot": cot_energy / (cot_dist * 75.0 * 9.81 + 1e-6),  # dimensionless E/(m g d)
        **{f"hl/{k}": v for k, v in hl_losses.items()},
      }
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
    attach_metadata_to_onnx(onnx_path, metadata)

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    hl_state = torch.load(path, map_location=map_location, weights_only=False).get("hl")
    if hl_state:
      self.hl.load_state_dict(hl_state)
    return infos
