"""WP5 Phase 2 rollout collection for the adaptation module `phi`.

Runs a FROZEN Phase-1 checkpoint (LL never retrained, HL frozen) under the current
(ADR-0010 coupled) payload DR and records `phi`'s pinned per-step input --
`obs["policy"] ++ obs["command"]`, 92 floats, command LAST -- into one ring buffer per
env. Ground truth `e = env_latent_e(...)` is read once per env (constant for the whole
run: `mode="startup"` DR does not resample on reset). Windows are NOT materialized here;
only a `(env_id, t)` index list is written alongside the raw buffer, so overlapping
50-frame windows are sliced at train/eval time instead of being duplicated on disk.

Spec, gate, and splits -> doc/hrl/A1a_plan.md "WP5 Phase 2".

  # main pooled corpus (one call per seed)
  python scripts/collect_phi_data.py --checkpoint-file <phase1 s42 model.pt> \\
      --out data/wp5-phase2/rollout_s42 --num-envs 4096 --steps 2000

  # OOD diagnostics (splits iii/iv), eval-only, never trained on
  python scripts/collect_phi_data.py --checkpoint-file <ckpt> \\
      --out data/wp5-phase2/ood_mass_s42 --split ood_mass --num-envs 1024 --steps 2000
  python scripts/collect_phi_data.py --checkpoint-file <ckpt> \\
      --out data/wp5-phase2/ood_lever_s42 --split ood_lever --num-envs 1024 --steps 2000
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from play import restore_run_structure
from src.tasks.velocity.mdp.observations import E_NAMES, env_latent_e
from src.tasks.velocity.mdp.payload_inertia import (
  MOUNT_D_MEAN,
  MOUNT_D_RANGES,
  PAYLOAD_MASS_RANGE,
  payload_mount,
)

TASK_ID = "Unitree-H1_2-Flat-A1-Payload"
POLICY_DIM, COMMAND_DIM = 89, 3  # the WP5d pinned contract: phi's per-step vector
SAMPLE_EVERY = 8  # HL-fire cadence (c=8) -- phi is only ever queried at this rate


def restore_twist_command_structure(env_cfg, ckpt_path: Path) -> None:
  """Reads the checkpoint's OWN ``params/env.yaml`` for ``commands.twist.
  resampling_time_range``/``rel_standing_envs`` and applies them to ``env_cfg`` in place.

  These are training-distribution properties of the specific HL being rolled out, not a
  Phase-2 constant -- hardcoding a literal here is exactly the "doesn't fit" bug class this
  function exists to prevent (WP5 Phase 1 moved from (3,8)/rse 0.15 to the keeper's
  (3,20)/rse 0.12 between the first and second time this pipeline ran). Absent file/keys
  (older runs) leave ``env_cfg``'s own defaults untouched.
  """
  params_yaml = ckpt_path.parent / "params" / "env.yaml"
  if not params_yaml.exists():
    return
  # env.yaml's viewer block can reference mjlab objects (e.g. OriginType) that have since
  # moved/been removed in the installed mjlab version -- irrelevant to the two scalars this
  # reads, so a lenient loader maps any unconstructible python/object tag to None instead of
  # raising, while still resolving the python/tuple tags the fields we need actually use.
  class _LenientLoader(yaml.UnsafeLoader):
    pass

  _ignore = lambda loader, suffix, node: None  # noqa: E731
  # Override the SAME (exact) prefixes UnsafeLoader already registers -- a shorter added
  # prefix loses to these more specific inherited ones in PyYAML's first-match-wins,
  # insertion-ordered lookup, so replacing them by exact key is what actually takes effect.
  for prefix in ("tag:yaml.org,2002:python/object/apply:",
                 "tag:yaml.org,2002:python/object/new:",
                 "tag:yaml.org,2002:python/object:",
                 "tag:yaml.org,2002:python/name:"):
    _LenientLoader.add_multi_constructor(prefix, _ignore)
  saved = yaml.load(params_yaml.read_text(), Loader=_LenientLoader)
  twist = saved.get("commands", {}).get("twist", {})
  live = env_cfg.commands["twist"]
  if "resampling_time_range" in twist:
    live.resampling_time_range = tuple(twist["resampling_time_range"])
  if "rel_standing_envs" in twist:
    live.rel_standing_envs = twist["rel_standing_envs"]
  print(f"[collect] restored twist command structure from {params_yaml.name}: "
        f"resampling_time_range={live.resampling_time_range} "
        f"rel_standing_envs={live.rel_standing_envs}")


def build_env(env_cfg, num_envs: int, mass_range=None, d_ranges=None, device: str = "cuda:0"):
  """Mutates the (already checkpoint-structure-restored) ``env_cfg`` in place and builds
  the wrapped env. Takes ``env_cfg`` rather than loading its own, so callers can restore
  the checkpoint's structure (goal dim, twist command params, etc.) first."""
  env_cfg.scene.num_envs = num_envs
  if mass_range is not None or d_ranges is not None:
    asset_cfg = env_cfg.events["payload_mount"].params["asset_cfg"]
    env_cfg.events["payload_mount"] = EventTermCfg(
      mode="startup",
      func=payload_mount,
      params={
        "asset_cfg": asset_cfg,
        "mass_range": mass_range or PAYLOAD_MASS_RANGE,
        "d_ranges": d_ranges or MOUNT_D_RANGES,
      },
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  return RslRlVecEnvWrapper(env, clip_actions=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint-file", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--steps", type=int, default=2000)
  ap.add_argument("--split", choices=["main", "ood_mass", "ood_lever"], default="main")
  ap.add_argument("--hl-obs-e-source", choices=["true", "phi"], default="true")
  ap.add_argument("--phi-dir", default=None,
                   help="train_phi.py output dir; required with --hl-obs-e-source phi "
                        "(grill item (c), distribution-shift comparison).")
  ap.add_argument("--device", default=None)
  args = ap.parse_args()

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  configure_torch_backends()

  mass_range = d_ranges = None
  if args.split == "ood_mass":
    # Split (iii): 15 kg OOD (Bar F's hardware block). Small spread around the point so
    # R2/RMSE are defined; d fixed at the mean deployment-ray lever arm.
    mass_range, d_ranges = (14.0, 16.0), tuple((v, v) for v in MOUNT_D_MEAN)
  elif args.split == "ood_lever":
    # Split (iv): OOD lever-arm (ADR-0010's own requested check). Single-axis shift, mass
    # and the other two axes left at their normal trained ranges.
    mass_range = PAYLOAD_MASS_RANGE
    d_ranges = ((0.22, 0.28), MOUNT_D_RANGES[1], MOUNT_D_RANGES[2])

  agent_cfg = load_rl_cfg(TASK_ID)
  # play=False: play mode never registers `payload_mount` (apply_payload_dr is a no-op
  # under play, matching apply_wide_dr's convention -- eval pins a point instead). Phase 2
  # needs per-env DIVERSITY across the full DR range, which only the training-mode cfg has.
  env_cfg = load_env_cfg(TASK_ID, play=False)
  # The Phase-1 checkpoint was trained with hl_obs_e=True (+ its own cadence/goal
  # structure), which the task's bare defaults don't carry -- restore it from the run's
  # own params/agent.yaml before building the env/runner, exactly like play.py.
  ckpt_path = Path(args.checkpoint_file)
  restore_run_structure(agent_cfg, env_cfg, ckpt_path)
  restore_twist_command_structure(env_cfg, ckpt_path)

  env = build_env(env_cfg, args.num_envs, mass_range, d_ranges, device)
  runner_cls = load_runner_cls(TASK_ID)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  if args.hl_obs_e_source == "phi":
    if args.phi_dir is None:
      raise SystemExit("--hl-obs-e-source phi requires --phi-dir")
    from phi_data import load_phi_for_inference
    phi_model, phi_norm = load_phi_for_inference(args.phi_dir, device)
    policy = runner.get_inference_policy(device=device, phi_model=phi_model, phi_norm=phi_norm)
  else:
    policy = runner.get_inference_policy(device=device)

  uenv = env.unwrapped
  torso_cfg, foot_cfg = runner._e_torso_cfg, runner._e_foot_cfg

  n, T, D = args.num_envs, args.steps, POLICY_DIM + COMMAND_DIM
  buf = np.empty((n, T, D), dtype=np.float32)

  with torch.inference_mode():
    obs, _ = env.reset()
    # Constant for the whole run (mode="startup" DR does not resample on reset) -- read once.
    e_true = env_latent_e(uenv, torso_cfg, foot_cfg).cpu().numpy()

    t0 = time.time()
    dones = torch.ones(n, dtype=torch.bool, device=device)  # forces phi's buffer cold at t=0
    for t in range(T):
      buf[:, t, :] = torch.cat([obs["policy"], obs["command"]], dim=-1).cpu().numpy()
      actions = policy(obs, dones)
      obs, _, dones, _extras = env.step(actions.to(device))
    print(f"[collect] {n} envs x {T} steps in {time.time() - t0:.1f}s "
          f"({args.split}, hl_obs_e_source={args.hl_obs_e_source}, {args.checkpoint_file})")

  # Index list, not materialized windows: sample every 8th step (HL-fire aligned) once the
  # 50-frame window is fully populated (t >= 49).
  idx_env, idx_t = np.meshgrid(
    np.arange(n), np.arange(49, T, SAMPLE_EVERY), indexing="ij"
  )
  indices = np.stack([idx_env.ravel(), idx_t.ravel()], axis=-1).astype(np.int32)

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  np.save(out / "buf.npy", buf)
  np.save(out / "e_true.npy", e_true)
  np.save(out / "indices.npy", indices)
  meta = {
    "checkpoint": str(args.checkpoint_file),
    "split": args.split,
    "num_envs": n,
    "steps": T,
    "policy_dim": POLICY_DIM,
    "command_dim": COMMAND_DIM,
    "e_names": list(E_NAMES),
    "sample_every": SAMPLE_EVERY,
    "mass_range": mass_range,
    "d_ranges": d_ranges,
    "hl_obs_e_source": args.hl_obs_e_source,
  }
  (out / "meta.json").write_text(json.dumps(meta, indent=2))
  print(f"[collect] wrote {out} ({indices.shape[0]} windows available)")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main()
