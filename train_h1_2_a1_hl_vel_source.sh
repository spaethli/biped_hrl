#!/bin/bash
#SBATCH --job-name=h1_2_a1_hlvel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/work/%u/ramlab_ws/slurm_logs/%j.out
#SBATCH --error=/data/work/%u/ramlab_ws/slurm_logs/%j.err
#
# WL-F redux (2026-08-24): does hl_vel_source cross ll_stand_still_coef fix the phantom
# goal? leg_odom alone was already tried and shelved 2026-08-10 (both retrains failed the
# policy bar AND stepped at cmd 0, see legodom_training_breaks_standing.md); the
# stand_still combination was never trained. Motivated by the 2026-08-13 hardware block:
# the HL asks for 0.61 m/s at zero stick regardless of WHICH deploy-side estimator arm is
# selected (all seven are logged; none of them changed the floor) -- so the floor is
# plausibly a training-side defect, not a live estimator artifact.
#
# ⚠️ Sanity check before reading results: `scripts/deploy_provenance.py --check` showed
# 2026-08-13 that the ONNX actually deployed on hardware is
# `2026-07-17_21-46-14_a1a_cot0p2_cad0p5_energy0p05_s42` -- BEFORE ll_stand_still_coef
# existed. The 0.61 m/s hardware number is not evidence against any hl_vel_source; it may
# just be the untreated baseline. Re-run the hardware block on a checkpoint from THIS
# script before drawing a WL-F verdict from it.
#
# ALL SEVEN deploy-side arms now have a training-side counterpart (2026-08-24,
# src/tasks/velocity/rl/hrl/base_estimators.py: legodom -> leg_odom, compl/ekf/ekf_grav/
# ekf_rot/jacobian/ekf_att keep their C++ names). Six of them (everything but leg_odom) are
# NOT bit-parity tested against the C++ -- see that module's docstring for what fp32
# approximation was judged sufficient here. The four ekf* arms are meaningfully more
# expensive (a batched Kalman filter, lazily constructed only when selected -- `state`/
# `leg_odom` runs pay nothing extra); compl/jacobian are cheap, same cost class as legodom.
#
# Base is the DEPLOYED HARDWARE BASELINE (a1a_cot0p2_cad0p5_energy0p05), not the a1.sh
# tagless default -- so `sbatch` with no overrides reproduces (by run-name convention,
# fresh timestamp dir) the run currently on the robot, and only the two new knobs below
# move you off it.
#
# Knobs via env vars:
#   SEED             (default 42)                -> always _sSEED
#   HL_VEL_SOURCE    state|leg_odom|compl|ekf|ekf_grav|ekf_rot|jacobian|ekf_att
#                    (default state)              -> non-state tags _<value>
#   LL_STAND_STILL   float (default 0.0)          -> >0 tags _standstillXpXX (the
#                    documented phantom-goal-at-cmd-0 fix, a1_stepping_zero_command.md)
#   LL_CADENCE_COEF  float (default 0.5)          -> != 0.5 tags _cadXpX (kept overridable
#                    for parity with train_h1_2_a1a_LL_rewards.sh; not swept by default)
#   HL_COT           float (default 0.2)          -> != 0.2 tags _cotXpXX
#   LL_ENERGY        float (default 0.05)         -> != 0.05 tags _energyXpXX
#   NUM_ENVS, MAX_ITER
#
# Examples:
#   # reproduce the hardware baseline exactly (state estimator, no stand-still):
#   sbatch train_h1_2_a1_hl_vel_source.sh
#   # the WL-F arm alone (already tried 2026-08-09/10, shelved -- rerun for a fresh seed):
#   sbatch --export=ALL,HL_VEL_SOURCE=leg_odom train_h1_2_a1_hl_vel_source.sh
#   # the untried leg_odom+stand_still combination motivating this script:
#   sbatch --export=ALL,HL_VEL_SOURCE=leg_odom,LL_STAND_STILL=1.0 train_h1_2_a1_hl_vel_source.sh
#   # train against ekf_rot (arm D), the arm the hardware bench found statistically tied
#   # with the cheaper compl (B) -- worth comparing which one trains a better policy:
#   sbatch --export=ALL,HL_VEL_SOURCE=ekf_rot train_h1_2_a1_hl_vel_source.sh
#   sbatch --export=ALL,HL_VEL_SOURCE=compl train_h1_2_a1_hl_vel_source.sh
#   # isolate whether stand_still alone (state estimator) already closes the gap:
#   sbatch --export=ALL,LL_STAND_STILL=1.0 train_h1_2_a1_hl_vel_source.sh
#   # sweep all seven non-state arms in one go, each its own job:
#   for arm in leg_odom compl ekf ekf_grav ekf_rot jacobian ekf_att; do
#     sbatch --export=ALL,HL_VEL_SOURCE=$arm train_h1_2_a1_hl_vel_source.sh
#   done

eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate unitree_mjlab_h1_2_rl
export PATH=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/bin:$PATH

SITE_PKGS=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SITE_PKGS/nvidia/cuda_nvrtc/lib:$SITE_PKGS/nvidia/cuda_runtime/lib:$SITE_PKGS/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

ulimit -l unlimited
export WARP_CACHE_PATH=$WORK/.warp_cache/$SLURM_JOB_ID
export TMPDIR=$WORK/tmp
mkdir -p $WARP_CACHE_PATH $WORK/tmp
export WANDB_MODE=offline

cd $WORK/ramlab_ws/code/unitree_rl_mjlab

SEED=${SEED:-42}
HL_VEL_SOURCE=${HL_VEL_SOURCE:-state}
LL_STAND_STILL=${LL_STAND_STILL:-0.0}
LL_CADENCE_COEF=${LL_CADENCE_COEF:-0.5}
HL_COT=${HL_COT:-0.2}
LL_ENERGY=${LL_ENERGY:-0.05}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-10001}

case "$HL_VEL_SOURCE" in
  state|leg_odom|compl|ekf|ekf_grav|ekf_rot|jacobian|ekf_att) ;;
  *)
    echo "[a1-hlvel] HL_VEL_SOURCE must be one of state|leg_odom|compl|ekf|ekf_grav|" \
         "ekf_rot|jacobian|ekf_att, got '$HL_VEL_SOURCE'" >&2
    exit 1
    ;;
esac

# state is the default (tagless); every other value tags with its own name (leg_odom ->
# _legodom for readability, matching kEstArmNames' spelling; the other five tag verbatim).
# hl_obs_vel stays at its rl_cfg default (True) in every case, which is what a non-state
# source requires -- no separate flag needed.
VS_FLAG="--agent.hl-vel-source ${HL_VEL_SOURCE}"; VS_TAG=""
if [[ "$HL_VEL_SOURCE" != "state" ]]; then
  TAG="$HL_VEL_SOURCE"; [[ "$TAG" == "leg_odom" ]] && TAG="legodom"
  VS_TAG="_${TAG}"
fi

# 0.0/off is tagless, matching train_h1_2_a1a_LL_rewards.sh's convention for this exact flag.
SS_FLAG=""; SS_TAG=""
awk "BEGIN{exit !($LL_STAND_STILL > 0)}" && SS_FLAG="--agent.ll-stand-still-coef ${LL_STAND_STILL}" \
  && SS_TAG="_standstill$(echo $LL_STAND_STILL | tr '.' 'p')"

# Cadence/CoT/energy: same construction as train_h1_2_a1a_LL_rewards.sh, but the DEFAULTS
# here are the hardware baseline's values (0.5 / 0.2 / 0.05), not that script's bare
# defaults, so a plain `sbatch` call with no overrides reproduces the deployed run.
CAD_FLAG="--agent.hl-cadence True --agent.hl-cadence-source hl --agent.hl-cot-coef ${HL_COT}"
CAD_TAG="_cot$(echo $HL_COT | tr '.' 'p')"
CADENCE_COEF_TAG="_cad$(echo $LL_CADENCE_COEF | tr '.' 'p')"

EN_FLAG=""; EN_TAG=""
awk "BEGIN{exit !($LL_ENERGY > 0)}" && EN_FLAG="--agent.ll-energy-coef ${LL_ENERGY}" \
  && EN_TAG="_energy$(echo $LL_ENERGY | tr '.' 'p')"

RUN_NAME="a1a${CAD_TAG}${CADENCE_COEF_TAG}${EN_TAG}${VS_TAG}${SS_TAG}_s${SEED}"

echo "[a1-hlvel] RUN=$RUN_NAME  hl_vel_source=$HL_VEL_SOURCE  ll_stand_still=$LL_STAND_STILL  cadence_coef=$LL_CADENCE_COEF  hl_cot=$HL_COT  energy=$LL_ENERGY  seed=$SEED"

python scripts/train.py Unitree-H1_2-Flat-A1 \
    --env.scene.num-envs ${NUM_ENVS} \
    --agent.max-iterations ${MAX_ITER} \
    --agent.seed ${SEED} \
    --agent.ll-cadence-coef ${LL_CADENCE_COEF} \
    $CAD_FLAG \
    $EN_FLAG \
    $VS_FLAG \
    $SS_FLAG \
    --agent.run-name ${RUN_NAME}

wandb sync --sync-all
