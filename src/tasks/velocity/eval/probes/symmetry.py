"""--diagnose-symmetry: WL-D arm 10 (2026-07-20, A1a_plan.md "Arm 10"), left/right gait symmetry.
feet_gait's own gait_match .mean(dim=1) dilutes one leg's systematic schedule violation against
the other's good match, and nothing in the aggregate bench compares the two legs to each other or
counts touchdowns per foot - this probe does, plus checks whether the bench's stride_period_s
(which divides eval time by raw touchdown count) is corrupted by a double-tap inflating that count."""

import json

import torch

from ..rollout import Observer, rollout


class Symmetry(Observer):

  def __init__(self, ctx):
    env, runner = ctx.env, ctx.runner
    uenv = self.uenv = env.unwrapped
    self.env, self.n_envs = env, uenv.num_envs
    self.contact_sensor = uenv.scene["feet_ground_contact"]
    self.step_dt = uenv.step_dt
    self.gait_offsets = torch.tensor([0.0, 0.5], device=env.device).view(1, -1)
    self.gait_swing = 0.0 if ctx.dummy else getattr(runner, "cadence_swing_time", 0.0)
    self.duty_lo, self.duty_hi = (0.56, 0.70) if ctx.dummy else getattr(
      runner, "cadence_duty_range", (0.56, 0.70))
    self.steps = ctx.cfg.diagnose_symmetry
    self.seed_results: list[dict] = []

  def reset(self):
    n_envs = self.n_envs
    self.elapsed = torch.zeros(n_envs, device=self.env.device)
    self.prev_contact: torch.Tensor | None = None
    # Per-env ordered touchdown event log: (time, foot) with foot 0=left, 1=right
    # (same left/right convention as gait_offsets / _gait_params's offset [0.0, 0.5]).
    self.td_events: list[list[tuple[float, int]]] = [[] for _ in range(n_envs)]
    self.gait_matches_foot: list[list[float]] = [[], []]
    self.both_contact_count = 0.0
    self.total_count = 0

  def after_step(self, step_i, actions, dones):
    uenv, n_envs, step_dt, td_events, elapsed = (
      self.uenv, self.n_envs, self.step_dt, self.td_events, self.elapsed)
    is_contact = self.contact_sensor.data.current_contact_time > 0  # [B, 2]
    if self.prev_contact is not None:
      first_td = is_contact & ~self.prev_contact  # [B, 2]
      for foot in range(2):
        for i in first_td[:, foot].nonzero(as_tuple=True)[0].tolist():
          td_events[i].append((elapsed[i].item(), foot))
    self.prev_contact = is_contact.clone()

    self.both_contact_count += is_contact.all(dim=1).float().sum().item()
    self.total_count += n_envs

    # Per-foot gait_match: same schedule feet_gait/eval_steps use, NOT averaged
    # over feet (arm 10's whole premise: the average hides a one-leg violation).
    hp = getattr(uenv, "hrl_phase", None)
    gait_thr = self.duty_lo
    if hp is None:
      hp = (uenv.episode_length_buf * step_dt) / 0.6
    elif self.gait_swing > 0.0:
      gait_thr = (1.0 - self.gait_swing / uenv.hrl_period).clamp(self.duty_lo, self.duty_hi).unsqueeze(1)
    leg_phase = (hp.unsqueeze(1) + self.gait_offsets) % 1.0
    matches = (leg_phase < gait_thr) == is_contact  # [B, 2]
    self.gait_matches_foot[0].append(matches[:, 0].float().mean().item())
    self.gait_matches_foot[1].append(matches[:, 1].float().mean().item())

    elapsed += step_dt
    if dones.any():
      d = dones.bool()
      elapsed[d] = 0.0
      # Drop in-progress event lists for reset envs so a reset never splices
      # into a fake short/long interval across the episode boundary.
      for i in d.nonzero(as_tuple=True)[0].tolist():
        td_events[i] = []

  def seed_end(self, seed_idx, seed):
    n_envs, step_dt = self.n_envs, self.step_dt
    # Post-hoc per-env event-sequence analysis: classify each consecutive touchdown
    # pair (any foot) as alternating (LR/RL -> the symmetry-index intervals) or a
    # same-foot repeat (LL/RR -> exactly the user's "touches down twice before the other
    # lifts" stutter, counted separately rather than folded into t_LR/t_RL).
    t_lr, t_rl = [], []
    repeat_n = {0: 0, 1: 0}
    td_counts = [0, 0]
    for events in self.td_events:
      events.sort(key=lambda e: e[0])
      td_counts[0] += sum(1 for _, f in events if f == 0)
      td_counts[1] += sum(1 for _, f in events if f == 1)
      for (t0, f0), (t1, f1) in zip(events, events[1:]):
        dt = t1 - t0
        if dt <= 0:
          continue
        if f0 == 0 and f1 == 1:
          t_lr.append(dt)
        elif f0 == 1 and f1 == 0:
          t_rl.append(dt)
        else:  # f0 == f1: same-foot repeat, the double-tap signature
          repeat_n[f1] += 1

    def _stats(xs):
      if not xs:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
      t = torch.tensor(xs)
      return {"n": len(xs), "mean": t.mean().item(),
              "std": t.std().item() if len(xs) > 1 else 0.0}

    lr_stats, rl_stats = _stats(t_lr), _stats(t_rl)
    si = (abs(lr_stats["mean"] - rl_stats["mean"]) / (lr_stats["mean"] + rl_stats["mean"])
          if lr_stats["n"] and rl_stats["n"] else float("nan"))
    # (c) stride corruption check: the bench's stride_period_s divides eval time by
    # (raw touchdown count / (n_envs*n_feet)) - recompute that SAME metric here so it
    # is directly comparable, in this run, to an alternation-based full-cycle estimate
    # (mean t_LR + mean t_RL) that is immune to double-taps (those land in repeat_n
    # instead of t_LR/t_RL, so they can't shrink the alternation-based stride).
    total_td = td_counts[0] + td_counts[1]
    stride_play_metric = ((self.steps * step_dt)
                          / max(total_td / max(n_envs * 2, 1), 1e-6))
    stride_alternation = (lr_stats["mean"] + rl_stats["mean"]
                          if lr_stats["n"] and rl_stats["n"] else float("nan"))

    self.seed_results.append({
      "td_count_left": float(td_counts[0]), "td_count_right": float(td_counts[1]),
      "t_lr_mean": lr_stats["mean"], "t_lr_std": lr_stats["std"], "t_lr_n": float(lr_stats["n"]),
      "t_rl_mean": rl_stats["mean"], "t_rl_std": rl_stats["std"], "t_rl_n": float(rl_stats["n"]),
      "si": si,
      "repeat_left_n": float(repeat_n[0]), "repeat_right_n": float(repeat_n[1]),
      "double_support_frac": self.both_contact_count / max(self.total_count, 1),
      "gait_match_left": float(torch.tensor(self.gait_matches_foot[0]).mean()),
      "gait_match_right": float(torch.tensor(self.gait_matches_foot[1]).mean()),
      "stride_play_metric": stride_play_metric,
      "stride_alternation": stride_alternation,
    })


def run(ctx) -> dict:
  cfg = ctx.cfg
  probe = Symmetry(ctx)
  rollout(ctx, cfg.diagnose_symmetry, [probe])
  seed_results, n_envs, duty_lo = probe.seed_results, probe.n_envs, probe.duty_lo

  keys = list(seed_results[0].keys())
  means = {k: float(torch.tensor([r[k] for r in seed_results]).mean()) for k in keys}

  print()
  print("=" * 70)
  print(f"  SYMMETRY PROBE (WL-D arm 10) | {cfg.diagnose_symmetry} steps x {n_envs} envs "
        f"x {cfg.eval_seeds} seed(s)")
  print("=" * 70)
  print(f"  Touchdowns   left={means['td_count_left']:.0f}  right={means['td_count_right']:.0f}"
        f"  ratio(R/L)={means['td_count_right'] / max(means['td_count_left'], 1e-6):.3f}")
  print(f"  t_LR  mean={means['t_lr_mean']:.4f}s std={means['t_lr_std']:.4f} (n={means['t_lr_n']:.0f})")
  print(f"  t_RL  mean={means['t_rl_mean']:.4f}s std={means['t_rl_std']:.4f} (n={means['t_rl_n']:.0f})")
  print(f"  SI (symmetry index) = {means['si']:.4f}")
  print(f"  Same-foot repeats (double-taps): left={means['repeat_left_n']:.0f}"
        f"  right={means['repeat_right_n']:.0f}")
  print(f"  Double-support fraction (realized) = {means['double_support_frac']:.4f}"
        f"  (scheduled ~0.12 at duty {duty_lo:.2f})")
  print(f"  gait_match  left={means['gait_match_left']:.4f}  right={means['gait_match_right']:.4f}")
  print(f"  stride_period_s (play.py:498 metric) = {means['stride_play_metric']:.4f}")
  print(f"  stride (alternation t_LR+t_RL)       = {means['stride_alternation']:.4f}")
  print("=" * 70)
  print()

  diag_out = {"label": ctx.label,
              **{k: round(v, 6) for k, v in means.items()},
              "eval_steps": cfg.diagnose_symmetry, "num_envs": n_envs,
              "eval_seeds": cfg.eval_seeds}
  print(f"[SYMDIAG] {json.dumps(diag_out)}")
  return diag_out
