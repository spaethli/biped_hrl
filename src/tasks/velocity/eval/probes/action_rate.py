"""--diagnose-action-rate: is A1's twitch concentrated at the HL fire step (goal obs jumps every c
steps) or spread uniformly (a smoothness-reward deficit)? Bins the whole-body action delta by
step % c and by joint group. A0 (no HL) is the flat control."""

import json

import torch

from ..rollout import Observer, rollout


class ActionRate(Observer):

  def __init__(self, ctx):
    env = ctx.env
    uenv = env.unwrapped
    self.n_envs = uenv.num_envs
    self.c = getattr(ctx.runner, "c", 8)  # A0 has no window; c=8 makes it the flat-profile control.

    def _find_group(pats):
      # find_joints([...]) raises if ANY pattern in the list matches zero joints (e.g.
      # this robot has a torso joint but no "waist" joint); resolve patterns one at a
      # time so a non-matching pattern is just skipped instead of failing the group.
      ids = set()
      for p in pats:
        try:
          pids, _ = uenv.scene["robot"].find_joints([p])
        except ValueError:
          continue
        ids.update(pids)
      return sorted(ids) if ids else None

    # Joint groups (regex vs the robot's joint names), resolved once.
    self.grp_ids = {}
    for name, pats in (("legs", [".*hip.*", ".*knee.*", ".*ankle.*"]),
                       ("arms", [".*shoulder.*", ".*elbow.*", ".*wrist.*"]),
                       ("waist", [".*waist.*", ".*torso.*"])):
      ids = _find_group(pats)
      self.grp_ids[name] = torch.as_tensor(ids, device=env.device) if ids is not None else None
    self.seed_results: list[dict] = []

  def reset(self):
    c = self.c
    self.phase_sum = [0.0] * c
    self.phase_n = [0] * c
    self.grp_sum = {k: 0.0 for k in self.grp_ids}
    self.grp_n = 0
    self.ar_sum = 0.0
    self.prev_actions: torch.Tensor | None = None

  def after_step(self, step_i, actions, dones):
    if self.prev_actions is not None:
      d = actions - self.prev_actions
      nrm = d.norm(dim=-1).mean().item()  # ||a_t - a_{t-1}|| over joints, env-mean
      b = step_i % self.c
      self.phase_sum[b] += nrm; self.phase_n[b] += 1
      self.ar_sum += nrm; self.grp_n += 1
      for k, ids in self.grp_ids.items():
        self.grp_sum[k] += (d[:, ids].norm(dim=-1).mean().item()
                            if ids is not None else float("nan"))
    self.prev_actions = actions.clone()

  def seed_end(self, seed_idx, seed):
    c, phase_sum, phase_n, grp_ids = self.c, self.phase_sum, self.phase_n, self.grp_ids
    phase = [phase_sum[i] / phase_n[i] if phase_n[i] else float("nan") for i in range(c)]
    rest = [phase[i] for i in range(1, c) if phase_n[i]]
    rest_mean = sum(rest) / len(rest) if rest else float("nan")
    grp_mean = {k: self.grp_sum[k] / max(self.grp_n, 1) for k in grp_ids}
    # Shares from summed squared group norms, so the three shares sum to 1.0.
    sq = {k: (grp_mean[k] ** 2 if grp_mean[k] == grp_mean[k] else 0.0) for k in grp_ids}
    tot_sq = sum(sq.values()) or 1.0
    res = {f"phase_{i}": phase[i] for i in range(c)}
    res.update({
      "fire_excess": phase[0] / rest_mean if rest_mean else float("nan"),
      "ar_mean": self.ar_sum / max(self.grp_n, 1),
      "g_legs": grp_mean["legs"], "g_arms": grp_mean["arms"], "g_waist": grp_mean["waist"],
      "share_legs": sq["legs"] / tot_sq, "share_arms": sq["arms"] / tot_sq,
      "share_waist": sq["waist"] / tot_sq,
    })
    self.seed_results.append(res)


def run(ctx) -> dict:
  cfg = ctx.cfg
  probe = ActionRate(ctx)
  rollout(ctx, cfg.diagnose_action_rate, [probe])
  seed_results, n_envs, c = probe.seed_results, probe.n_envs, probe.c

  keys = list(seed_results[0].keys())
  means = {k: float(torch.tensor([r[k] for r in seed_results]).mean()) for k in keys}

  print()
  print("=" * 70)
  print(f"  ACTION-RATE DECOMPOSITION | {cfg.diagnose_action_rate} steps x {n_envs} envs "
        f"x {cfg.eval_seeds} seed(s) | c={c}")
  print("=" * 70)
  prof = "  ".join(f"[{i}]{means[f'phase_{i}']:.3f}" + ("*" if i == 0 else "")
                   for i in range(c))
  print(f"  window profile (* = HL fire step): {prof}")
  print(f"  fire_excess (phase0 / mean rest) = {means['fire_excess']:.3f}")
  print(f"  ar_mean (all steps)              = {means['ar_mean']:.4f}")
  print(f"  legs  mean={means['g_legs']:.4f}  share={means['share_legs']:.3f}")
  print(f"  arms  mean={means['g_arms']:.4f}  share={means['share_arms']:.3f}")
  print(f"  waist mean={means['g_waist']:.4f}  share={means['share_waist']:.3f}")
  print("=" * 70)
  print()

  diag_out = {"label": ctx.label,
              **{k: round(v, 6) for k, v in means.items()},
              "eval_steps": cfg.diagnose_action_rate, "num_envs": n_envs,
              "eval_seeds": cfg.eval_seeds, "c": c}
  print(f"[ARDIAG] {json.dumps(diag_out)}")
  return diag_out
