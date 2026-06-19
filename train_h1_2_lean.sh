#!/bin/bash
#SBATCH --job-name=h1_2_lean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/work/%u/ramlab_ws/slurm_logs/%j.out
#SBATCH --error=/data/work/%u/ramlab_ws/slurm_logs/%j.err
#
# Track F lean-reward experiment — one file, two modes (launcher + worker).
# doc/hrl/hierarchy_benefit_roadmap.md Track F.
#
#   SUBMIT (run on the login node):
#       ./train_h1_2_lean.sh submit
#     Fans out all 4 variants as separate 1-GPU jobs. a1_from_lean waits (afterok)
#     for a0_scratch because it warm-starts from a0_scratch's checkpoint; the other
#     three start immediately. With 16 GPUs all run concurrently.
#
#   WORKER (set automatically by submit; or run one by hand):
#       sbatch --export=ALL,VARIANT=a0_scratch train_h1_2_lean.sh
#
# Variants (all: 10001 iters, 4096 envs):
#   a0_scratch    lean A0, no warm-start (from scratch)
#   a0_polished   lean A0, warm-started (resume) from the polished shaped-A0
#   a1_from_lean  lean A1, warm-started from a0_scratch's lean-A0 checkpoint
#   a1_polished   lean A1, warm-started from the polished shaped-A0
#
# Knobs (env vars, optional): NUM_ENVS, MAX_ITER, POLISHED_A0 (path to the polished
# shaped-A0 model_*.pt).

set -euo pipefail

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-10001}
POLISHED_A0=${POLISHED_A0:-logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt}

# ============================ SUBMIT MODE ============================
# Entered when VARIANT is unset (i.e. invoked directly on the login node).
if [[ -z "${VARIANT:-}" ]]; then
  if [[ "${1:-}" != "submit" && "${1:-}" != "all" ]]; then
    echo "Usage: ./train_h1_2_lean.sh submit   # sbatch all 4 lean variants"
    exit 1
  fi
  SELF="$(realpath "$0")"
  J1=$(sbatch --parsable --job-name=lean_a0_scratch \
        --export=ALL,VARIANT=a0_scratch "$SELF")
  echo "submitted a0_scratch    -> $J1"
  J2=$(sbatch --parsable --job-name=lean_a0_polished \
        --export=ALL,VARIANT=a0_polished "$SELF")
  echo "submitted a0_polished   -> $J2"
  J3=$(sbatch --parsable --dependency=afterok:$J1 --job-name=lean_a1_from_lean \
        --export=ALL,VARIANT=a1_from_lean "$SELF")
  echo "submitted a1_from_lean  -> $J3  (waits for a0_scratch $J1)"
  J4=$(sbatch --parsable --job-name=lean_a1_polished \
        --export=ALL,VARIANT=a1_polished "$SELF")
  echo "submitted a1_polished   -> $J4"
  echo "All 4 lean runs submitted (NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER)."
  exit 0
fi

# ============================ WORKER MODE ============================
eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate unitree_mjlab_h1_2_rl
# fallback if wrong python is used
export PATH=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/bin:$PATH

# Use the conda env's CUDA 12.8 libraries (system only has cuda/12.3)
SITE_PKGS=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SITE_PKGS/nvidia/cuda_nvrtc/lib:$SITE_PKGS/nvidia/cuda_runtime/lib:$SITE_PKGS/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

ulimit -l unlimited

# Redirect warp cache and compiler temp to $WORK (compute node /tmp is too small)
export WARP_CACHE_PATH=$WORK/.warp_cache/$SLURM_JOB_ID
export TMPDIR=$WORK/tmp
mkdir -p $WORK/.warp_cache $WORK/tmp

export WANDB_MODE=offline

cd $WORK/ramlab_ws/code/unitree_rl_mjlab

COMMON="--env.scene.num-envs ${NUM_ENVS} --agent.max-iterations ${MAX_ITER}"
# The solved A1 HL config (absolute target + tracking HL reward, off-policy TD3 + HIRO).
A1_HL="--agent.hl-algorithm td3 --agent.relabeling hiro --agent.hl-target-mode absolute --agent.hl-reward-mode tracking"

echo "[lean] VARIANT=$VARIANT  NUM_ENVS=$NUM_ENVS  MAX_ITER=$MAX_ITER"

case "$VARIANT" in
  a0_scratch)
    python scripts/train.py Unitree-H1_2-Flat-Lean $COMMON \
      --agent.run-name lean_a0_scratch
    ;;

  a0_polished)
    # Resume (weights + obs-normalizer + optimizer) from the polished shaped-A0 run,
    # then fine-tune under the lean reward. Lean-A0 shares experiment_name
    # "h1_2_velocity" with shaped-A0, so --agent.load-run resolves in the same dir.
    # NOTE: resume continues the iteration counter from the polished checkpoint, so
    # MAX_ITER here are ADDITIONAL iters (checkpoints model_10000 -> model_20000).
    python scripts/train.py Unitree-H1_2-Flat-Lean $COMMON \
      --agent.resume True \
      --agent.load-run "$(basename "$(dirname "$POLISHED_A0")")" \
      --agent.load-checkpoint "$(basename "$POLISHED_A0")" \
      --agent.run-name lean_a0_from_polished
    ;;

  a1_from_lean)
    # Warm-start the lean-A1 LL from a0_scratch's lean-A0 checkpoint (last index =
    # MAX_ITER-1). Resolved here because this job runs after a0_scratch (afterok).
    LEAN_A0_CKPT=$(ls -dt logs/rsl_rl/h1_2_velocity/*_lean_a0_scratch/model_$((MAX_ITER-1)).pt 2>/dev/null | head -1 || true)
    if [[ -z "$LEAN_A0_CKPT" ]]; then
      echo "ERROR: no lean_a0_scratch checkpoint found under logs/rsl_rl/h1_2_velocity/*_lean_a0_scratch/"
      exit 1
    fi
    echo "[lean] warm-starting A1-lean from $LEAN_A0_CKPT"
    python scripts/train.py Unitree-H1_2-Flat-A1-Lean $COMMON $A1_HL \
      --agent.warm-start-path "$LEAN_A0_CKPT" \
      --agent.run-name lean_a1_from_lean_a0
    ;;

  a1_polished)
    python scripts/train.py Unitree-H1_2-Flat-A1-Lean $COMMON $A1_HL \
      --agent.warm-start-path "$POLISHED_A0" \
      --agent.run-name lean_a1_from_polished
    ;;

  *)
    echo "Unknown VARIANT: $VARIANT"; exit 1 ;;
esac

# Auto-sync after training
wandb sync --sync-all
