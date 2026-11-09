"""Checkpoint-agnostic ONNX-vs-PyTorch parity check for exported deploy policies.

Rebuilds the run structure exactly like ``scripts/play.py`` (the shared
``checkpoint_structure`` module: the structure-determining keys from ``params/agent.yaml``,
then the trained goal scale), builds the torch policy/
policies in eval mode, exports ONNX to a temp dir via the runner's own export
function (``export_hierarchy_to_onnx`` for A1, else ``export_policy_to_onnx``), then
feeds ``--num-samples`` random obs vectors through both the torch inference path and
onnxruntime for each exported net, comparing outputs. For A1 the HRL metadata baked
into the ONNX file (c, hl_algorithm, goal_components, ...) is also cross-checked
against the restored runner config.

Usage:
  conda activate unitree_mjlab_h1_2_rl
  python scripts/onnx_parity.py --task Unitree-H1_2-Flat-A1 \
      --checkpoint-file logs/rsl_rl/.../model_10000.pt
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import list_to_csv_str
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from src.tasks.velocity.rl.hrl.checkpoint_structure import (
  pin_trained_goal_scale, restore_run_structure)


def _mlp_forward(model, x: torch.Tensor) -> torch.Tensor:
  """Deterministic-mean forward for an rsl_rl MLPModel on a pre-concatenated flat obs
  vector, mirroring its own ``as_onnx()`` / ``_OnnxMLPModel`` path (obs_normalizer ->
  mlp -> deterministic_output). Computed on the live model, not a copy."""
  y = model.obs_normalizer(x)
  out = model.mlp(y)
  return model.distribution.deterministic_output(out) if model.distribution is not None else out


def _metadata_table(nets: dict, runner) -> bool:
  """Cross-check the HRL metadata baked into an exported net against the restored
  runner config. No-op (pass) for a non-hierarchical (A0) task."""
  meta_path = nets.get("low_level", nets.get("high_level"))
  if meta_path is None:
    print("\n[INFO] No HRL metadata to check (single-policy task).")
    return True
  actual = {p.key: p.value for p in onnx.load(meta_path[2]).metadata_props}
  expected = {
    "c": runner.c,
    "hl_algorithm": runner.hl_algorithm,
    "hl_target_mode": runner.hl_target_mode,
    "goal_components": [c.name for c in runner.goal_space.components],
    "hl_velocity_goals_only": runner.hl_velocity_goals_only,
  }
  if runner.hl_cadence:
    expected["hl_cadence_source"] = runner.hl_cadence_source
    expected["cadence_period_range"] = list(runner.cadence_period_range)

  print(f"\n{'field':<24}{'metadata':<26}{'runner':<26}status")
  ok = True
  for k, v in expected.items():
    exp_str = list_to_csv_str(v) if isinstance(v, (list, tuple)) else str(v)
    act_str = actual.get(k, "<MISSING>")
    good = act_str == exp_str
    ok = ok and good
    print(f"{k:<24}{act_str:<26}{exp_str:<26}{'OK' if good else 'MISMATCH'}")
  return ok


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint-file", required=True)
  ap.add_argument("--num-samples", type=int, default=256)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tol", type=float, default=1e-4)
  ap.add_argument("--device", default=None)
  ap.add_argument("--onnx-dir", default=None,
                  help="Compare the checkpoint's torch actor(s) against the ONNX files in "
                       "THIS directory (e.g. deploy/.../exported/) instead of against a "
                       "fresh temp export. Without it this script only ever validates the "
                       "export FUNCTION -- a correct function plus a stale DEPLOYED file "
                       "passes clean, which is exactly how a parked WL-D arm-6 export ran "
                       "unnoticed on 2026-08-03. Use this to prove that what is deployed "
                       "IS the named checkpoint, numerically, when md5 cannot say so.")
  args = ap.parse_args()

  import mjlab.tasks  # noqa: F401  (populates the task registry)
  import src.tasks  # noqa: F401

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  resume_path = Path(args.checkpoint_file).resolve()
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  restore_run_structure(agent_cfg, env_cfg, resume_path)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
  # Export what play.py --export-onnx exports: on a pre-2026-07-16 checkpoint, the trained scale.
  pin_trained_goal_scale(runner, env, lambda: load_env_cfg(args.task))
  runner.alg.eval_mode()
  if hasattr(runner, "hl"):
    runner.hl.eval_mode()

  tmp = tempfile.mkdtemp(prefix="onnx_parity_")
  # nets: name -> (torch_fn, torch_input_dim, onnx_path)
  nets: dict = {}
  # The numeric comparison always runs on CPU: onnxruntime's CPUExecutionProvider is
  # what the deploy binary links against, and a cuda-vs-cpu torch/onnxruntime mismatch
  # is not a genuine export bug -- it amplifies float32 rounding noise through the deep
  # MLP into diffs orders of magnitude above the tolerance for out-of-distribution
  # random inputs (verified: same env/runner built on cuda, ~1e-5 once both sides run
  # on cpu). Moving the live model to cpu here is safe: the ONNX files were already
  # written above, and nothing downstream reuses these modules on their training device.
  if hasattr(runner, "export_hierarchy_to_onnx"):
    runner.export_hierarchy_to_onnx(tmp + os.sep)
    ll_model = runner.alg.get_policy().to("cpu")
    nets["low_level"] = (lambda x, m=ll_model: _mlp_forward(m, x), ll_model.obs_dim,
                         os.path.join(tmp, "low_level.onnx"))
    hl_path = os.path.join(tmp, "high_level.onnx")
    if os.path.exists(hl_path):
      if runner.hl_algorithm == "td3":
        hl = runner.hl
        hl.actor.to("cpu")
        hl.normalizer.to("cpu")
        nets["high_level"] = (lambda x, hl=hl: torch.tanh(hl.actor(hl.normalizer(x))),
                              hl._state_dim, hl_path)
      elif runner.hl_algorithm == "ppo":
        hl_model = runner.hl.ppo.actor.to("cpu")
        nets["high_level"] = (lambda x, m=hl_model: _mlp_forward(m, x), hl_model.obs_dim,
                              hl_path)
      else:
        raise NotImplementedError(
          f"No torch-side reference wired for hl_algorithm={runner.hl_algorithm!r}"
        )
    else:
      print("[INFO] No high_level.onnx (oracle HL has no network).")
  else:
    runner.export_policy_to_onnx(tmp, filename="policy.onnx")
    policy_model = runner.alg.get_policy().to("cpu")
    nets["policy"] = (lambda x, m=policy_model: _mlp_forward(m, x), policy_model.obs_dim,
                      os.path.join(tmp, "policy.onnx"))

  if args.onnx_dir:
    # Point the comparison (and, via nets[...][2], the metadata cross-check) at the
    # DEPLOYED files. The torch side stays the reference, so a pass here means "the ONNX
    # sitting in that directory computes this checkpoint's policy". The temp export above
    # still ran and is left untouched -- it costs little and doubles as a control.
    target = Path(args.onnx_dir).resolve()
    missing = [f"{n}.onnx" for n in nets if not (target / f"{n}.onnx").exists()]
    if missing:
      raise SystemExit(f"--onnx-dir {target}: missing {', '.join(missing)}")
    nets = {n: (fn, dim, str(target / f"{n}.onnx")) for n, (fn, dim, _) in nets.items()}
    print(f"\n[PARITY] comparing against DEPLOYED ONNX in {target}")
    for n in nets:
      print(f"[PARITY]   {n}.onnx md5="
            f"{hashlib.md5((target / f'{n}.onnx').read_bytes()).hexdigest()}")

  torch.manual_seed(args.seed)
  results: dict = {}
  print(f"\n{'net':<12}{'dim':>6}{'max_abs_diff':>16}{'mean_abs_diff':>16}")
  for name, (torch_fn, torch_dim, onnx_path) in nets.items():
    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    onnx_dim = sess.get_inputs()[0].shape[-1]
    assert torch_dim == onnx_dim, (
      f"{name}: torch input dim {torch_dim} != onnx input dim {onnx_dim}"
    )
    x = torch.rand(args.num_samples, torch_dim) * 6.0 - 3.0  # U[-3, 3], cpu (see above)
    with torch.inference_mode():
      torch_out = torch_fn(x).numpy()
    x_np = x.numpy().astype(np.float32)
    onnx_out = np.concatenate(
      [sess.run(None, {in_name: x_np[i : i + 1]})[0] for i in range(args.num_samples)],
      axis=0,
    )
    diff = np.abs(torch_out - onnx_out)
    max_d, mean_d = float(diff.max()), float(diff.mean())
    results[name] = {"max_abs_diff": max_d, "mean_abs_diff": mean_d}
    print(f"{name:<12}{torch_dim:>6}{max_d:>16.6e}{mean_d:>16.6e}")

  metadata_ok = _metadata_table(nets, runner)

  pass_ = all(r["max_abs_diff"] < args.tol for r in results.values()) and metadata_ok
  out = {"nets": results, "metadata_ok": metadata_ok, "pass": pass_}
  print(f"\n[PARITY] {json.dumps(out)}")
  env.close()
  sys.exit(0 if pass_ else 1)


if __name__ == "__main__":
  main()
