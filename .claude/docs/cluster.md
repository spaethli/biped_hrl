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
Fix: `WARP_CACHE_PATH=$WORK/.warp_cache/$SLURM_JOB_ID` + `TMPDIR=$WORK/tmp` (the default
TMPDIR / warp cache hit size limits on compute nodes). Training works on both the A100
cluster and the lab RTX 5070.

## Warp cache concurrent-compile race — RESOLVED (2026-06-19)

Symptom: `Exception: Failed to load CUDA module '..._kernel...'` at the first sim step,
then the job hangs until SLURM kills it at the time limit. Cost a full day — all 4 lean
jobs (`train_h1_2_lean.sh`) co-scheduled on one node died identically; the noise runs,
which ran later/alone, were fine.
Cause: multiple jobs sharing **one** `WARP_CACHE_PATH=$WORK/.warp_cache` on NFS JIT-compile
the same cold kernels **simultaneously** → cache files collide → module load fails. The
NVRTC fix above set the cache *location*, not per-job isolation.
Fix: **per-job cache dir** `WARP_CACHE_PATH=$WORK/.warp_cache/$SLURM_JOB_ID` so concurrent
jobs never share cache files (applied in all `train_h1_2_*.sh`). For ad-hoc concurrent runs
(e.g. a play/benchmark beside training) use a unique suffix, e.g. `.../bench_$$`.

## Pinned versions

torch 2.11.0+cu128, torchvision 0.26.0, warp-lang 1.13.0, mujoco-warp 3.8.1,
rsl-rl-lib 5.2.0, mjlab from source.
