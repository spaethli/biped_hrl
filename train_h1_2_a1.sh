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
# A1 baseline. Run-name convention: DEFAULTS ARE MOSTLY IMPLICIT — only DEVIATIONS from
# the default state are tagged. EXCEPTIONS always shown (value/algo in the name, even at
# default, because they get swept for arm-posture tuning): HL algorithm, posture,
# action-rate. The default A1 (this script with no overrides) is: TD3 HL + HIRO relabel,
# delta target, tracking reward, exp goal kernel (true-terminal fell_over), from-scratch
# (no warm-start), velobs on, velocity-goals-only HL, posture 0.5, action-rate 0.02,
# desired_kl 0.01 -> name `a1_td3_pose0p5_ar0p02_s42`.
#
# Knobs via env vars:
#   SEED           (default 42)                     -> always _sSEED
#   HL_ALGO        td3|oracle (default td3)          -> ALWAYS tagged _td3 | _oracle
#   LL_POSTURE     float (default 0.5)               -> ALWAYS tagged _poseXpX
#   LL_ACTION_RATE float (default 0.02)              -> ALWAYS tagged _arXpXX
#   HL_OBS_VEL     True|False (default True)         -> False tags _noVelobs
#   HL_TARGET_MODE delta|abs  (default delta)        -> abs/absolute tags _abs
#   GOAL_KERNEL    exp|l2     (default exp)          -> l2 tags _l2 (AUTO-SETS fell_over
#                  time_out True; pairing rule l2<->time_out, exp<->true-terminal — never mix)
#   WARM_START     A0 ckpt path (default none=scratch) -> a path tags _warmstart
#   HL_CADENCE     True|False (default False)        -> True tags _cad (random source)
#   CADENCE_SOURCE random|hl (default random)        -> hl tags _cadhl (needs td3)
#   HL_COT         float (default 0.0)               -> >0 tags _cotXpX (needs CADENCE_SOURCE=hl)
#   HL_VELGOALS    True|False (default True)         -> False tags _fullgoals (legacy full-goal HL)
#   DESIRED_KL     float (default 0.01, LL PPO adaptive-LR target) -> != 0.01 tags _klXpXXX
#   NUM_ENVS, MAX_ITER
#
# Examples:
#   sbatch train_h1_2_a1.sh                                        # default A1 -> a1_td3_pose0p5_ar0p02_s42
#   sbatch --export=ALL,LL_POSTURE=1.0 train_h1_2_a1.sh            #           -> a1_td3_pose1p0_ar0p02_s42
#   sbatch --export=ALL,GOAL_KERNEL=l2,WARM_START=<a0.pt> train_h1_2_a1.sh   # -> a1_td3_l2_warmstart_pose0p5_ar0p02_s42
#   sbatch --export=ALL,HL_TARGET_MODE=abs,HL_OBS_VEL=False train_h1_2_a1.sh # -> a1_td3_noVelobs_abs_pose0p5_ar0p02_s42
#   # HL-steered cadence with CoT pressure:
#   sbatch --export=ALL,HL_CADENCE=True,CADENCE_SOURCE=hl,HL_COT=0.2 train_h1_2_a1.sh  # -> a1_td3_pose0p5_ar0p02_cadhl_cot0p2_s42

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
HL_TARGET_MODE=${HL_TARGET_MODE:-delta}
LL_POSTURE=${LL_POSTURE:-0.5}
LL_ACTION_RATE=${LL_ACTION_RATE:-0.02}
GOAL_KERNEL=${GOAL_KERNEL:-exp}
HL_CADENCE=${HL_CADENCE:-False}
CADENCE_SOURCE=${CADENCE_SOURCE:-random}
HL_ALGO=${HL_ALGO:-td3}
HL_COT=${HL_COT:-0.0}
HL_VELGOALS=${HL_VELGOALS:-True}
DESIRED_KL=${DESIRED_KL:-0.01}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-10001}
WARM_START=${WARM_START:-none}

# Build optional flags and run-name tags from knobs. DEFAULT state contributes NO tag;
# only deviations are tagged (see the convention note in the header).

# velobs on is the default (tagless); off -> _noVelobs. hl_obs_vel code-default is False,
# so the flag is passed explicitly in both directions.
if [[ "$HL_OBS_VEL" == "True" ]]; then VEL_FLAG="--agent.hl-obs-vel True"; VEL_TAG="";
else VEL_FLAG="--agent.hl-obs-vel False"; VEL_TAG="_noVelobs"; fi

# delta is the default (tagless); absolute -> _abs. Accept "abs" as an alias for "absolute".
TGT_MODE="$HL_TARGET_MODE"; TGT_TAG=""
[[ "$TGT_MODE" == "abs" ]] && TGT_MODE="absolute"
[[ "$TGT_MODE" == "absolute" ]] && TGT_TAG="_abs"
TGT_FLAG="--agent.hl-target-mode ${TGT_MODE}"

# posture + action-rate are ALWAYS tagged with their value (swept for arm-posture tuning).
POSE_TAG="_pose$(echo $LL_POSTURE | tr '.' 'p')"
AR_TAG="_ar$(echo $LL_ACTION_RATE | tr '.' 'p')"

# exp kernel + true-terminal fell_over is the default (tagless; env_cfgs already sets
# time_out False). l2 is the deviation -> _l2 and must restore the time_out truncation
# bootstrap (never mix — see CLAUDE.md gotchas).
KERNEL_FLAG=""; KERNEL_TAG=""
if [[ "$GOAL_KERNEL" == "l2" ]]; then
  KERNEL_FLAG="--agent.ll-goal-kernel l2 --env.terminations.fell-over.time-out True"
  KERNEL_TAG="_l2"
fi

# from-scratch is the default (tagless); a warm-start path -> _warmstart.
WS_FLAG=""; WS_TAG=""
if [[ "$WARM_START" != "none" ]]; then
  WS_FLAG="--agent.warm-start-path ${WARM_START}"
  WS_TAG="_warmstart"
fi

CAD_FLAG=""; CAD_TAG=""
if [[ "$HL_CADENCE" == "True" ]]; then
  CAD_FLAG="--agent.hl-cadence True --agent.hl-cadence-source ${CADENCE_SOURCE}"
  CAD_TAG="_cad"; [[ "$CADENCE_SOURCE" == "hl" ]] && CAD_TAG="_cadhl"
fi

# HL algorithm is ALWAYS tagged (_td3 default | _oracle drops the TD3/relabel flags,
# M2-style LL isolation).
HL_FLAG="--agent.hl-algorithm td3 --agent.relabeling hiro"; HL_TAG="_td3"
[[ "$HL_ALGO" == "oracle" ]] && HL_FLAG="--agent.hl-algorithm oracle" && HL_TAG="_oracle"

COT_FLAG=""; COT_TAG=""
awk "BEGIN{exit !($HL_COT > 0)}" && COT_FLAG="--agent.hl-cot-coef ${HL_COT}" \
  && COT_TAG="_cot$(echo $HL_COT | tr '.' 'p')"

# velocity-goals-only HL is the default (tagless); False -> _fullgoals (legacy full-goal HL).
VG_FLAG=""; VG_TAG=""
[[ "$HL_VELGOALS" == "False" ]] && VG_FLAG="--agent.hl-velocity-goals-only False" && VG_TAG="_fullgoals"

# desired_kl 0.01 is the default (tagless); any other value -> _klXpXXX.
KL_FLAG=""; KL_TAG=""
awk "BEGIN{exit !($DESIRED_KL != 0.01)}" && KL_FLAG="--agent.algorithm.desired-kl ${DESIRED_KL}" \
  && KL_TAG="_kl$(echo $DESIRED_KL | tr '.' 'p')"

RUN_NAME="a1${HL_TAG}${VEL_TAG}${TGT_TAG}${KERNEL_TAG}${WS_TAG}${POSE_TAG}${AR_TAG}${CAD_TAG}${COT_TAG}${VG_TAG}${KL_TAG}_s${SEED}"

echo "[a1] RUN=$RUN_NAME  velobs=$HL_OBS_VEL  target=$TGT_MODE  kernel=$GOAL_KERNEL  warm_start=$WARM_START  posture=$LL_POSTURE  action_rate=$LL_ACTION_RATE  cadence=$HL_CADENCE/$CADENCE_SOURCE  seed=$SEED"

python scripts/train.py Unitree-H1_2-Flat-A1 \
    --env.scene.num-envs ${NUM_ENVS} \
    --agent.max-iterations ${MAX_ITER} \
    $WS_FLAG \
    $HL_FLAG \
    $TGT_FLAG --agent.hl-reward-mode tracking \
    --agent.seed ${SEED} \
    --agent.ll-posture-coef ${LL_POSTURE} \
    --agent.ll-action-rate-coef ${LL_ACTION_RATE} \
    $VEL_FLAG $KERNEL_FLAG $CAD_FLAG $COT_FLAG $VG_FLAG $KL_FLAG \
    --agent.run-name ${RUN_NAME}

wandb sync --sync-all
