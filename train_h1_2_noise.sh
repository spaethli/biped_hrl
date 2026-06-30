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
# Variants (all: TD3 + HIRO + tracking, 4096 envs, 10001 iters, warm-started polished-A0).
# Two axes -- target map {abs,delta} x noise {bias,full}:
#   abs_bias    absolute target, bias only          -- does the LL absorb a static offset?
#   abs_full    absolute target, bias + drift + lag -- full estimator realism
#   delta_bias  directional target, bias only       -- bias CANCELS in V*-s (faithful DR,
#   delta_full  directional target, + drift + lag      hrl_runner _target_obs); residual = drift/lag
#
# Submit knobs (env vars): VARIANTS (default 'delta_bias delta_full'), SEEDS (default '42 123'),
#   NUM_ENVS, MAX_ITER, POLISHED_A0, MODE (seq chains all jobs on one GPU | "" concurrent),
#   HL_OBS_VEL (default True; feeds HL the base lin-vel estimate, tags run '_velobs').
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
HL_OBS_VEL=${HL_OBS_VEL:-True}   # feed HL the deployable base lin-vel (vx,vy) so the
                                 # directional HL computes g=(cmd-v)/scale; set False to ablate

# ============================ SUBMIT MODE ============================
# Entered when VARIANT is unset (i.e. invoked directly on the login node).
if [[ -z "${VARIANT:-}" ]]; then
  if [[ "${1:-}" != "submit" && "${1:-}" != "all" ]]; then
    echo "Usage: ./train_h1_2_noise.sh submit          # VARIANTS x SEEDS, concurrent"
    echo "       MODE=seq ./train_h1_2_noise.sh submit  # chain all jobs on one GPU"
    echo "       VARIANTS='abs_bias abs_full' SEEDS=42 ./train_h1_2_noise.sh submit"
    exit 1
  fi
  SELF="$(realpath "$0")"
  VARIANTS=${VARIANTS:-"delta_bias delta_full"}
  SEEDS=${SEEDS:-"42 123"}
  PREV=""
  for V in $VARIANTS; do
    for S in $SEEDS; do
      DEP=""
      [[ "${MODE:-}" == "seq" && -n "$PREV" ]] && DEP="--dependency=afterok:$PREV"
      JID=$(sbatch --parsable $DEP --job-name=noise_${V}_s${S} \
            --export=ALL,VARIANT=${V},SEED=${S} "$SELF")
      echo "submitted ${V} seed ${S} -> $JID ${DEP:+(after $PREV)}"
      PREV=$JID
    done
  done
  echo "Submitted VARIANTS='${VARIANTS}' SEEDS='${SEEDS}' (NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER MODE=${MODE:-concurrent})."
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
# Estimator-noise on the LL goal channel; bias is shared, full adds OU drift + sensor lag.
NOISE_ON="--agent.goal-state-noise.enable True --agent.goal-state-noise.bias-range ${BIAS_RANGE}"
NOISE_FULL="${NOISE_ON} --agent.goal-state-noise.drift-std ${DRIFT_STD} --agent.goal-state-noise.drift-decay ${DRIFT_DECAY} --agent.goal-state-noise.lag-steps ${LAG_STEPS}"

# Variant -> (target map, noise set). abs_* = absolute (V* = command-range center + g);
# delta_* = directional (V* = state + g; bias cancels in V*-s, faithful DR). *_full adds drift+lag.
case "$VARIANT" in
  abs_bias)   TGT=absolute; NOISE="$NOISE_ON"   ;;
  abs_full)   TGT=absolute; NOISE="$NOISE_FULL" ;;
  delta_bias) TGT=delta;    NOISE="$NOISE_ON"   ;;
  delta_full) TGT=delta;    NOISE="$NOISE_FULL" ;;
  *) echo "Unknown VARIANT: $VARIANT"; exit 1 ;;
esac
# Solved A1 HL config (off-policy TD3 + HIRO + tracking reward); target map per variant.
A1_HL="--agent.hl-algorithm td3 --agent.relabeling hiro --agent.hl-target-mode ${TGT} --agent.hl-reward-mode tracking"
SEED=${SEED:-42}
# Feed the directional HL the deployable base lin-vel (vx,vy); tag the run so velobs vs
# no-velobs deltas are distinguishable in the logs.
VEL_FLAG=""; VEL_TAG=""
if [[ "${HL_OBS_VEL}" == "True" ]]; then VEL_FLAG="--agent.hl-obs-vel True"; VEL_TAG="_velobs"; fi

echo "[noise] VARIANT=$VARIANT TGT=$TGT SEED=$SEED NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER warm-start=$POLISHED_A0"

python scripts/train.py Unitree-H1_2-Flat-A1 $COMMON $A1_HL $NOISE \
  --agent.warm-start-path "$POLISHED_A0" \
  --agent.seed ${SEED} \
  --agent.run-name noise_${VARIANT}${VEL_TAG}_s${SEED}

# Auto-sync after training
wandb sync --sync-all
