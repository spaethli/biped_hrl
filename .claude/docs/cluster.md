# Ehlers HPC cluster

Personal identifiers are intentionally omitted (public repo) — `$USER` below is the
cluster student account.

## Facts

- Login: `ehlers-login.informatik.uni-stuttgart.de`
- GPU: A100-SXM4-40GB ×4 per node, CUDA 12.8 (driver 570.86.15)
- `$WORK = /data/work/$USER` (500 GB); `$HOME` has only a 50 GB quota — keep
  everything (conda, code, caches, logs) under `$WORK`.
- Conda env: `$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl`
- Code: `$WORK/ramlab_ws/code/unitree_rl_mjlab/`
- mjlab (editable): `$WORK/ramlab_ws/code/mjlab/` — MUST be installed from source
  (PyPI mjlab==1.3.0 uses a broken warp API).
- SLURM logs: `$WORK/ramlab_ws/slurm_logs/`

## SLURM script

`train_h1_2.sh` at the repo root on the cluster.
Key params: `--gres=gpu:1`, `--cpus-per-task=16`, `--mem=64G`, `--time=04:00:00`.
Uses `WANDB_MODE=offline` + `wandb sync --sync-all` after training.
`LD_LIBRARY_PATH` manually points to the conda env's bundled nvidia CUDA libs.

## NVRTC compilation error — RESOLVED (2026-06-05)

Error was: `Warp NVRTC compilation error 6: NVRTC_ERROR_COMPILATION — Catastrophic
error: unable to obtain mapped memory`.
Fix: `WARP_CACHE_PATH=$WORK/.warp_cache` + `TMPDIR=$WORK/tmp` (the default TMPDIR /
warp cache hit size limits on compute nodes). Training works on both the A100
cluster and the lab RTX 5070.

## Pinned versions

torch 2.11.0+cu128, torchvision 0.26.0, warp-lang 1.13.0, mujoco-warp 3.8.1,
rsl-rl-lib 5.2.0, mjlab from source.
