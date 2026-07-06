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
#   LL_POSTURE     float (default 0.5; 0 = off)
#   LL_ACTION_RATE float (default 0.02; 0 = off)
#   GOAL_KERNEL    l2|exp (default l2) -> tags _exp; exp AUTO-SETS fell_over to a true
#                  terminal (pairing rule: l2<->time_out, exp<->terminal — never mix)
#   WARM_START     A0 ckpt path, or "none" for from-scratch (tags _scratch)
#   HL_CADENCE     True|False (default False) -> tags _cad (random per-episode source)
#   CADENCE_SOURCE random|hl (default random; hl needs td3, tags _cadhl)
#   NUM_ENVS, MAX_ITER
#
# Examples:
#   sbatch train_h1_2_a1.sh                                        # defaults
#   sbatch --export=ALL,SEED=42 train_h1_2_a1.sh
#   sbatch --export=ALL,HL_OBS_VEL=False,LL_POSTURE=0 train_h1_2_a1.sh
#   # from-scratch exp-kernel TD3 arm:
#   sbatch --export=ALL,GOAL_KERNEL=exp,WARM_START=none,HL_CADENCE=True train_h1_2_a1.sh
#   # clean kernel ablation (kernel is the only LL reward):
#   sbatch --export=ALL,GOAL_KERNEL=exp,WARM_START=none,LL_POSTURE=0,LL_ACTION_RATE=0 train_h1_2_a1.sh

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
LL_POSTURE=${LL_POSTURE:-0.5}
LL_ACTION_RATE=${LL_ACTION_RATE:-0.02}
GOAL_KERNEL=${GOAL_KERNEL:-l2}
HL_CADENCE=${HL_CADENCE:-False}
CADENCE_SOURCE=${CADENCE_SOURCE:-random}
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

# Kernel: exp auto-pairs with a true-terminal fell_over (never mix — see CLAUDE.md gotchas)
KERNEL_FLAG=""; KERNEL_TAG=""
if [[ "$GOAL_KERNEL" == "exp" ]]; then
  KERNEL_FLAG="--agent.ll-goal-kernel exp --env.terminations.fell-over.time-out False"
  KERNEL_TAG="_exp"
fi

# Warm start: "none" = from-scratch (only sensible with GOAL_KERNEL=exp)
WS_FLAG="--agent.warm-start-path ${WARM_START}"; WS_TAG=""
[[ "$WARM_START" == "none" ]] && WS_FLAG="" && WS_TAG="_scratch"

CAD_FLAG=""; CAD_TAG=""
if [[ "$HL_CADENCE" == "True" ]]; then
  CAD_FLAG="--agent.hl-cadence True --agent.hl-cadence-source ${CADENCE_SOURCE}"
  CAD_TAG="_cad"; [[ "$CADENCE_SOURCE" == "hl" ]] && CAD_TAG="_cadhl"
fi

RUN_NAME="a1_td3_delta_tracking${VEL_TAG}${POSE_TAG}${AR_TAG}${KERNEL_TAG}${WS_TAG}${CAD_TAG}_s${SEED}"

echo "[a1] RUN=$RUN_NAME  velobs=$HL_OBS_VEL  posture=$LL_POSTURE  action_rate=$LL_ACTION_RATE  kernel=$GOAL_KERNEL  warm_start=$WARM_START  cadence=$HL_CADENCE/$CADENCE_SOURCE  seed=$SEED"

python scripts/train.py Unitree-H1_2-Flat-A1 \
    --env.scene.num-envs ${NUM_ENVS} \
    --agent.max-iterations ${MAX_ITER} \
    $WS_FLAG \
    --agent.hl-algorithm td3 --agent.relabeling hiro \
    --agent.hl-target-mode delta --agent.hl-reward-mode tracking \
    --agent.seed ${SEED} \
    --agent.ll-posture-coef ${LL_POSTURE} \
    --agent.ll-action-rate-coef ${LL_ACTION_RATE} \
    $VEL_FLAG $KERNEL_FLAG $CAD_FLAG \
    --agent.run-name ${RUN_NAME}

wandb sync --sync-all
