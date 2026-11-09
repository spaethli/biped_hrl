"""--diagnose-cadence: cadence transmission (2026-08-31). Is the HL's commanded period stationary
over a stride? `get_inference_policy` rewrites env.hrl_period at every HL fire under
hl_cadence_source='hl' and leaves it fixed under 'random', so sampling it per step and
differencing per WINDOW measures exactly the reference the LL is asked to entrain to."""

import json

import torch

from ..rollout import Observer, rollout


class Cadence(Observer):

  def __init__(self, ctx):
    self.uenv = ctx.env.unwrapped
    self.c = getattr(ctx.runner, "c", 1)
    self.seed_results: list[dict] = []

  def reset(self):
    self.per_window: list[torch.Tensor] = []

  def before_step(self, step_i, actions):
    # Sample AFTER the policy call so a fire's new period is already written.
    if step_i % self.c == 0:
      self.per_window.append(self.uenv.hrl_period.detach().clone().flatten())

  def seed_end(self, seed_idx, seed):
    c, step_dt = self.c, self.uenv.step_dt
    P = torch.stack(self.per_window, dim=0)         # [n_windows, n_envs]
    d = (P[1:] - P[:-1]).abs()                      # window-to-window change
    stride = P.mean().item()
    self.seed_results.append({
      "period_mean":      stride,
      "period_std_win":   P.std(dim=0).mean().item(),   # per-env std ACROSS windows
      "period_cv":        (P.std(dim=0) / P.mean(dim=0).clamp(min=1e-6)).mean().item(),
      "dperiod_abs_mean": d.mean().item(),
      "dperiod_p95":      torch.quantile(d.flatten().float(), 0.95).item(),
      "period_spread_env": P.mean(dim=0).std().item(),  # across-env spread of the mean
      "windows_per_stride": stride / (c * step_dt),
      "n_windows":        int(P.shape[0]),
    })


def run(ctx) -> dict:
  cfg, runner = ctx.cfg, ctx.runner
  probe = Cadence(ctx)
  rollout(ctx, cfg.diagnose_cadence, [probe])
  n_envs, c, step_dt = probe.uenv.num_envs, probe.c, probe.uenv.step_dt
  seed_results = probe.seed_results

  keys = list(seed_results[0].keys())
  agg = {k: sum(r[k] for r in seed_results) / len(seed_results) for k in keys}
  print()
  print(f"  CADENCE TRANSMISSION PROBE | {cfg.diagnose_cadence} steps x {n_envs} envs "
        f"x {cfg.eval_seeds} seed(s) | c={c}, window={c * step_dt:.3f} s")
  print(f"    source                    : {getattr(runner, 'hl_cadence_source', '?')}")
  print(f"    commanded period (mean)   : {agg['period_mean']:.4f} s")
  print(f"    within-episode std        : {agg['period_std_win']:.4f} s  "
        f"(cv {agg['period_cv'] * 100:.1f}%)")
  print(f"    |change| per window       : {agg['dperiod_abs_mean']:.4f} s  "
        f"(p95 {agg['dperiod_p95']:.4f})")
  print(f"    windows per stride        : {agg['windows_per_stride']:.2f}")
  print(f"    across-env spread of mean : {agg['period_spread_env']:.4f} s")
  out = {**agg, 'label': str(cfg.checkpoint_file), 'num_envs': n_envs, 'eval_seeds': cfg.eval_seeds,
         'source': str(getattr(runner, 'hl_cadence_source', '?'))}
  print(f"  [CADDIAG] {json.dumps(out)}")
  return out
