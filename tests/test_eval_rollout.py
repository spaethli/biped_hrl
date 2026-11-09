"""The probe rollout skeleton (src/tasks/velocity/eval/rollout.py), through its interface.

Every thesis bench number comes out of this loop, and its ORDER is part of the measurement:
the seed is set before any hook runs (a schedule pins its first command before reset), per-seed
state is built outside inference mode and the steps run inside it, and only the benchmark hands
the policy the done mask (a phi checkpoint resets its history on it). A fake env stands in for
MuJoCo, so this runs on CPU.
"""

from types import SimpleNamespace

import math
import torch

from src.tasks.velocity.eval.rollout import Observer, mean_std, rollout

N_ENVS = 3


class FakeEnv:
  device = "cpu"

  def __init__(self, log):
    self.log = log
    self.unwrapped = SimpleNamespace(num_envs=N_ENVS)
    self.n_step = 0

  def reset(self):
    self.log.append(("env.reset", torch.is_inference_mode_enabled()))
    return {"t": torch.zeros(N_ENVS)}, {}

  def step(self, actions):
    self.n_step += 1
    self.log.append(("env.step",))
    dones = torch.tensor([self.n_step % 2 == 0, False, True])
    return {"t": torch.full((N_ENVS,), float(self.n_step))}, None, dones, {}


class Recorder(Observer):

  def __init__(self, log):
    self.log = log

  def seed_begin(self, seed_idx, seed):
    self.log.append(("seed_begin", seed_idx, seed, torch.initial_seed()))

  def reset(self):
    self.log.append(("reset", torch.is_inference_mode_enabled()))

  def loop_begin(self, steps):
    self.log.append(("loop_begin", steps, torch.is_inference_mode_enabled()))

  def before_step(self, step_i, actions):
    self.log.append(("before_step", step_i))

  def after_step(self, step_i, actions, dones):
    self.log.append(("after_step", step_i))

  def seed_end(self, seed_idx, seed):
    self.log.append(("seed_end", seed_idx, torch.is_inference_mode_enabled()))


def _run(steps=2, seeds=2, pass_dones=False):
  log, calls = [], []

  def policy(obs, *rest):
    calls.append(rest)
    log.append(("policy",))
    return torch.zeros(N_ENVS, 1)

  ctx = SimpleNamespace(env=FakeEnv(log), policy=policy, cfg=SimpleNamespace(eval_seeds=seeds))
  rollout(ctx, steps, [Recorder(log)], pass_dones=pass_dones)
  return log, calls


def test_rollout_seeds_each_seed_before_its_hooks_and_orders_every_step():
  log, _ = _run(steps=2, seeds=2)
  per_seed = [
    ("env.reset", True), ("reset", False), ("loop_begin", 2, True),
    ("policy",), ("before_step", 0), ("env.step",), ("after_step", 0),
    ("policy",), ("before_step", 1), ("env.step",), ("after_step", 1),
  ]
  assert log == [
    ("seed_begin", 0, 42, 42), *per_seed, ("seed_end", 0, False),
    ("seed_begin", 1, 43, 43), *per_seed, ("seed_end", 1, False),
  ]


def test_only_pass_dones_hands_the_policy_the_done_mask_starting_all_true():
  _, calls = _run(steps=3, seeds=1, pass_dones=True)
  assert len(calls) == 3 and all(len(c) == 1 for c in calls)
  # first call: the all-True reset mask, then whatever the previous env.step returned
  assert calls[0][0].tolist() == [True, True, True]
  assert calls[1][0].tolist() == [False, False, True]
  assert calls[2][0].tolist() == [True, False, True]
  _, calls = _run(steps=3, seeds=1, pass_dones=False)
  assert calls == [(), (), ()]


def test_mean_std_aggregates_over_seeds_in_key_order():
  rows = [{"b": 1.0, "a": 2.0}, {"b": 2.0, "a": 4.0}, {"b": 6.0, "a": 6.0}]
  keys, means, stds = mean_std(rows, 3)
  assert keys == ["b", "a"]
  assert means["b"] == 3.0 and means["a"] == 4.0          # mean, not median (2.0 / 4.0)
  assert abs(stds["a"] - 2.0) < 1e-6                       # unbiased (n-1), as [BENCH] reports
  _, means, stds = mean_std(rows[:1], 1)
  assert means == {"b": 1.0, "a": 2.0} and all(math.isnan(v) for v in stds.values())
