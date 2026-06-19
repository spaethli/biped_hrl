#!/bin/bash
#SBATCH --job-name=h1_2_noise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/work/%u/ramlab_ws/slurm_logs/%j.out
#SBATCH --error=/data/work/%u/ramlab_ws/slurm_logs/%j.err
#
# Track C #8b — estimator-noise DR on the A1 LL goal channel (absolute TD3 only).
# doc/hrl/hierarchy_benefit_roadmap.md Track C (#8b). Trains the LL on a deploy-realistic
# base-velocity estimate (rt/sportmodestate-like) instead of sim ground-truth, so it is
# robust to the real onboard estimator. Reward + HL stay on ground-truth (privileged);
# only the LL's observed goal V*-s sees the noise. Velocity (vx,vy) only; yaw/orient
# clean. Meaningful in absolute mode (a constant bias cancels in the delta map).
#
#   SUBMIT (run on the login node):
#       ./train_h1_2_noise.sh submit       # both variants concurrently (2 GPUs)
#       MODE=seq ./train_h1_2_noise.sh submit   # chained (afterok), reuse 1 GPU
#
#   WORKER (set automatically by submit; or run one by hand):
#       sbatch --export=ALL,VARIANT=abs_bias train_h1_2_noise.sh
#
# Variants (both: absolute TD3 + HIRO + tracking, 4096 envs, 10001 iters,
#           warm-started from the polished shaped-A0):
#   abs_bias   bias only            -- does the LL absorb a static estimator offset?
#   abs_full   bias + drift + lag   -- full estimator realism (noise + slow drift + lag)
#
# Knobs (env vars, optional): NUM_ENVS, MAX_ITER, POLISHED_A0, MODE (seq|"").
# Noise magnitudes (env vars, optional, override the cfg defaults):
#   BIAS_RANGE (0.10 m/s)  DRIFT_STD (0.01)  DRIFT_DECAY (0.99)  LAG_STEPS (3)

set -euo pipefail

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-10001}
POLISHED_A0=${POLISHED_A0:-logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt}
BIAS_RANGE=${BIAS_RANGE:-0.10}
DRIFT_STD=${DRIFT_STD:-0.01}
DRIFT_DECAY=${DRIFT_DECAY:-0.99}
LAG_STEPS=${LAG_STEPS:-3}

# ============================ SUBMIT MODE ============================
# Entered when VARIANT is unset (i.e. invoked directly on the login node).
if [[ -z "${VARIANT:-}" ]]; then
  if [[ "${1:-}" != "submit" && "${1:-}" != "all" ]]; then
    echo "Usage: ./train_h1_2_noise.sh submit         # both variants concurrently"
    echo "       MODE=seq ./train_h1_2_noise.sh submit # chained on one GPU"
    exit 1
  fi
  SELF="$(realpath "$0")"
  J1=$(sbatch --parsable --job-name=noise_abs_bias \
        --export=ALL,VARIANT=abs_bias "$SELF")
  echo "submitted abs_bias  -> $J1"
  DEP=""
  [[ "${MODE:-}" == "seq" ]] && DEP="--dependency=afterok:$J1"
  J2=$(sbatch --parsable $DEP --job-name=noise_abs_full \
        --export=ALL,VARIANT=abs_full "$SELF")
  echo "submitted abs_full  -> $J2 ${DEP:+(waits for abs_bias $J1)}"
  echo "Both noise runs submitted (NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER MODE=${MODE:-concurrent})."
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
# Estimator-noise on the LL goal channel; bias is shared by both variants.
NOISE_ON="--agent.goal-state-noise.enable True --agent.goal-state-noise.bias-range ${BIAS_RANGE}"
NOISE_FULL="${NOISE_ON} --agent.goal-state-noise.drift-std ${DRIFT_STD} --agent.goal-state-noise.drift-decay ${DRIFT_DECAY} --agent.goal-state-noise.lag-steps ${LAG_STEPS}"

echo "[noise] VARIANT=$VARIANT  NUM_ENVS=$NUM_ENVS  MAX_ITER=$MAX_ITER  warm-start=$POLISHED_A0"

case "$VARIANT" in
  abs_bias)
    # Bias only: drift_std/lag_steps left at their cfg defaults (0 -> off).
    python scripts/train.py Unitree-H1_2-Flat-A1 $COMMON $A1_HL $NOISE_ON \
      --agent.warm-start-path "$POLISHED_A0" \
      --agent.run-name noise_abs_bias
    ;;

  abs_full)
    # Bias + within-episode OU drift + first-order sensor lag.
    python scripts/train.py Unitree-H1_2-Flat-A1 $COMMON $A1_HL $NOISE_FULL \
      --agent.warm-start-path "$POLISHED_A0" \
      --agent.run-name noise_abs_full
    ;;

  *)
    echo "Unknown VARIANT: $VARIANT"; exit 1 ;;
esac

# Auto-sync after training
wandb sync --sync-all
