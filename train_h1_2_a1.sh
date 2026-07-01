#!/bin/bash
#SBATCH --job-name=h1_2_a1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/work/%u/ramlab_ws/slurm_logs/%j.out
#SBATCH --error=/data/work/%u/ramlab_ws/slurm_logs/%j.err
#
# A1 baseline (TD3 + HIRO + delta + tracking). Knobs via env vars:
#   SEED           (default 42)
#   HL_OBS_VEL     True|False (default True)  -> tags _velobs
#   LL_POSTURE     float (default 1.0; 0 = off)
#   LL_ACTION_RATE float (default 0.02; 0 = off)
#   NUM_ENVS, MAX_ITER, WARM_START
#
# Examples:
#   sbatch train_h1_2_a1.sh                                        # defaults
#   sbatch --export=ALL,SEED=42 train_h1_2_a1.sh
#   sbatch --export=ALL,HL_OBS_VEL=False,LL_POSTURE=0 train_h1_2_a1.sh

eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate unitree_mjlab_h1_2_rl
export PATH=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/bin:$PATH

SITE_PKGS=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SITE_PKGS/nvidia/cuda_nvrtc/lib:$SITE_PKGS/nvidia/cuda_runtime/lib:$SITE_PKGS/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

ulimit -l unlimited
export WARP_CACHE_PATH=$WORK/.warp_cache
export TMPDIR=$WORK/tmp
mkdir -p $WORK/.warp_cache $WORK/tmp
export WANDB_MODE=offline

cd $WORK/ramlab_ws/code/unitree_rl_mjlab

SEED=${SEED:-42}
HL_OBS_VEL=${HL_OBS_VEL:-True}
LL_POSTURE=${LL_POSTURE:-1.0}
LL_ACTION_RATE=${LL_ACTION_RATE:-0.02}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-10001}
WARM_START=${WARM_START:-logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt}

# Build optional flags and run-name tags from knobs
VEL_FLAG=""; VEL_TAG=""
[[ "$HL_OBS_VEL" == "True" ]] && VEL_FLAG="--agent.hl-obs-vel True" && VEL_TAG="_velobs"

POSE_TAG=""
(( $(echo "$LL_POSTURE > 0" | bc -l) )) && POSE_TAG="_pose$(echo $LL_POSTURE | tr '.' 'p')"

AR_TAG=""
(( $(echo "$LL_ACTION_RATE == 0" | bc -l) )) && AR_TAG="_noar"

RUN_NAME="a1_td3_delta_tracking${VEL_TAG}${POSE_TAG}${AR_TAG}_s${SEED}"

echo "[a1] RUN=$RUN_NAME  velobs=$HL_OBS_VEL  posture=$LL_POSTURE  action_rate=$LL_ACTION_RATE  seed=$SEED"

python scripts/train.py Unitree-H1_2-Flat-A1 \
    --env.scene.num-envs ${NUM_ENVS} \
    --agent.max-iterations ${MAX_ITER} \
    --agent.warm-start-path ${WARM_START} \
    --agent.hl-algorithm td3 --agent.relabeling hiro \
    --agent.hl-target-mode delta --agent.hl-reward-mode tracking \
    --agent.seed ${SEED} \
    --agent.ll-posture-coef ${LL_POSTURE} \
    --agent.ll-action-rate-coef ${LL_ACTION_RATE} \
    $VEL_FLAG \
    --agent.run-name ${RUN_NAME}

wandb sync --sync-all
