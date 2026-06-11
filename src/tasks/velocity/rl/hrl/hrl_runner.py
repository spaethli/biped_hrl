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

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from ..runner import VelocityOnPolicyRunner
from .goal_space import build_goal_space, init_goal_buffer
from .high_level import HighLevel, HighLevelPpo, OracleHighLevel


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
    self.ll_task_reward_coef: float = train_cfg["ll_task_reward_coef"]
    self.warm_start_path: str | None = train_cfg.get("warm_start_path")

    # Declarative goal space -> goal_dim is derived (nothing hardcodes a dimension).
    self.goal_space = build_goal_space(
      tuple(train_cfg["goal_components"]), train_cfg.get("goal_weights")
    )
    self.goal_dim: int = self.goal_space.dim

    # Initialise the goal buffer BEFORE the base builds models, so the env's `goal`
    # observation group resolves to the right dimension at construction time.
    init_goal_buffer(env.unwrapped, self.goal_dim, device)

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

    # Warm-start the LL from an A0 checkpoint (proprio columns + deeper layers).
    # A resume-load (if any) runs after __init__ and overrides this.
    if self.warm_start_path:
      self._warm_start_low_level(self.warm_start_path)

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
      )
    raise NotImplementedError(
      f"hl_algorithm='{self.hl_algorithm}' not implemented yet (milestone 1/2/3 ship "
      "'oracle' and 'ppo'; 'td3' arrives in later milestones)."
    )

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    self.hl.train_mode()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      intrinsic_sum = 0.0
      with torch.inference_mode():
        uenv = self.env.unwrapped
        for k in range(self.cfg["num_steps_per_env"]):
          state = self.goal_space.extract(uenv)
          # High level fires at the start of each window -> new absolute target V*.
          if k % self.c == 0:
            self._target = self.hl.act(uenv, obs, state)
            self.hl.begin_window(uenv, obs, state)

          # Low level observes the remaining delta V* - s_i.
          delta = self._target - state
          uenv.hrl_goal = delta
          obs["goal"] = delta

          actions = self.alg.act(obs)
          obs, task_rew, dones, extras = self.env.step(actions.to(self.env.device))
          obs, task_rew, dones = (
            obs.to(self.device),
            task_rew.to(self.device),
            dones.to(self.device),
          )

          # Intrinsic (goal-distance) reward is the LL's training signal.
          achieved = self.goal_space.extract(uenv)
          r_lo = self.goal_space.reward(self._target, achieved)
          if self.ll_task_reward_coef != 0.0:
            r_lo = r_lo + self.ll_task_reward_coef * task_rew
          # Refresh the goal in the post-step obs (remaining delta at the new state)
          # so the normalizer/next-act input is consistent.
          post_delta = self._target - achieved
          uenv.hrl_goal = post_delta
          obs["goal"] = post_delta

          self.alg.process_env_step(obs, r_lo, dones, extras)
          self.hl.accumulate(task_rew)
          if (k + 1) % self.c == 0:
            self.hl.end_window(uenv, obs, achieved, dones, extras)

          # Episode bookkeeping uses task reward (A0-comparable); intrinsic is the
          # LL's actual training signal, logged separately via the loss dict.
          intrinsic_sum += r_lo.mean().item()
          self.logger.process_env_step(task_rew, dones, extras)

        collect_time = time.time() - start
        start = time.time()
        self.alg.compute_returns(obs)

      ll_losses = self.alg.update()
      hl_losses = self.hl.update()
      loss_dict = {
        **ll_losses,
        "intrinsic_reward": intrinsic_sum / self.cfg["num_steps_per_env"],
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
    # A1 has no "actor" obs group (split into policy/command/goal); get_base_metadata
    # reads active_terms["actor"], so alias it to the LL deploy obs (proprio ++ goal)
    # for the export, then remove it — active_terms is the live dict compute() iterates.
    om = self.env.unwrapped.observation_manager
    om.active_terms["actor"] = om.active_terms["policy"] + om.active_terms["goal"]
    try:
      self._export_policy_onnx(path)
    finally:
      del om.active_terms["actor"]
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    hl_state = torch.load(path, map_location=map_location, weights_only=False).get("hl")
    if hl_state:
      self.hl.load_state_dict(hl_state)
    return infos
