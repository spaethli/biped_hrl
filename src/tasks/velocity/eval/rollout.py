"""The seed/reset/step skeleton the rollout probes share, and the hooks an observer gets.

Order is part of the measurement. Each seed: `torch.manual_seed(42 + seed_idx)`, then the
observers' `seed_begin` (before reset: a schedule pins its first command here), `env.reset()` in
its own inference-mode block, `reset` (outside inference mode, where per-seed state is set up),
then one inference-mode block holding `loop_begin` and the steps. Each step: the policy acts,
`before_step`, `env.step`, `after_step`. Observers run in list order at every hook; the probe goes
last, so add-ons it reads (the joint-limit range feeds the bench's ankle spans) are already done.
"""

import torch


class Observer:
  """No-op hooks; override the ones a probe or add-on needs."""

  def seed_begin(self, seed_idx: int, seed: int): ...
  def reset(self): ...
  def loop_begin(self, steps: int): ...
  def before_step(self, step_i: int, actions: torch.Tensor): ...
  def after_step(self, step_i: int, actions: torch.Tensor, dones: torch.Tensor): ...
  def seed_end(self, seed_idx: int, seed: int): ...
  def report(self): ...                # before the host probe's scorecard
  def finish(self, out: dict): ...     # after it, before the host's [TAG] line; may extend `out`


def rollout(ctx, steps: int, observers: list, pass_dones: bool = False):
  """`pass_dones`: call the policy as policy(obs, dones). Only the benchmark does; the
  diagnostic probes call policy(obs), which on a phi checkpoint never resets its history."""
  env, policy = ctx.env, ctx.policy
  for seed_idx in range(ctx.cfg.eval_seeds):
    seed = 42 + seed_idx
    torch.manual_seed(seed)
    for o in observers:
      o.seed_begin(seed_idx, seed)
    # reset() does in-place buffer updates; must be inside inference_mode or it errors
    # on seed >= 1 (the prior rollout marked those env buffers as inference tensors).
    with torch.inference_mode():
      obs, _ = env.reset()
    for o in observers:
      o.reset()
    with torch.inference_mode():
      # WP5 Phase 2 phi hook: force a full buffer reset at the start of THIS seed's
      # rollout (the closure's hist/fill persist across the seed loop, but env.reset()
      # just discarded the simulator state -- an all-True mask is a no-op on the very
      # first seed, where the buffer is already zeroed, and load-bearing on every seed
      # after it, else the phi path would start seed 1 on seed 0's stale tail).
      dones = torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.device)
      for o in observers:
        o.loop_begin(steps)
      for step_i in range(steps):
        actions = policy(obs, dones) if pass_dones else policy(obs)
        for o in observers:
          o.before_step(step_i, actions)
        obs, _, dones, _ = env.step(actions.to(env.device))
        for o in observers:
          o.after_step(step_i, actions, dones)
    for o in observers:
      o.seed_end(seed_idx, seed)


def mean_std(seed_results: list[dict], n_seeds: int):
  """Per-key mean and std over seeds (std NaN for one seed), as the [BENCH] line reports them."""
  keys = list(seed_results[0].keys())
  vals = {k: torch.tensor([r[k] for r in seed_results]) for k in keys}
  means = {k: vals[k].mean().item() for k in keys}
  stds = {k: vals[k].std().item() if n_seeds > 1 else float("nan") for k in keys}
  return keys, means, stds
