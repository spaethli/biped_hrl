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
LL_CADENCE_COEF=${LL_CADENCE_COEF:-0.5}
HL_COT=${HL_COT:-0.2}
FIX0P8=${FIX0P8:-False}
LL_STAND_STILL=${LL_STAND_STILL:-1.0}
LL_ANGMOM=${LL_ANGMOM:-0.025}
LL_FOOTSLIP=${LL_FOOTSLIP:-0.25}
LL_FOOTCLEAR=${LL_FOOTCLEAR:-1.0}
LL_JOINT_ACC=${LL_JOINT_ACC:-2.5e-7}
LL_JOINT_LIMITS=${LL_JOINT_LIMITS:-10.0}
LL_SOFT_LANDING=${LL_SOFT_LANDING:-1e-3}
LL_BODY_ANG_VEL=${LL_BODY_ANG_VEL:-0.05}
POSTURE_ALL_JOINTS=${POSTURE_ALL_JOINTS:-True}
LL_ACTION_RATE=${LL_ACTION_RATE:-0.05}
ANKLE_ANCHOR=${ANKLE_ANCHOR:-False}
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

if [[ "$FIX0P8" == "True" ]]; then
  CAD_FLAG="--agent.hl-cadence True --agent.hl-cadence-source random --agent.cadence-period-range 0.8,0.8 --agent.hl-cot-coef 0.0"
  CAD_TAG="_fix0p8"
else
  CAD_FLAG="--agent.hl-cadence True --agent.hl-cadence-source hl --agent.hl-cot-coef ${HL_COT}"
  CAD_TAG="_cot$(echo $HL_COT | tr '.' 'p')"
fi

# ll_cadence_coef is ALWAYS tagged with its value (arm 1's own lever).
CADENCE_COEF_TAG="_cad$(echo $LL_CADENCE_COEF | tr '.' 'p')"

# Arm 3/4 mirror levers: 0.0 = off/tagless (byte-identical to the batch baseline).
SS_FLAG=""; SS_TAG=""
awk "BEGIN{exit !($LL_STAND_STILL > 0)}" && SS_FLAG="--agent.ll-stand-still-coef ${LL_STAND_STILL}" \
  && SS_TAG="_standstill$(echo $LL_STAND_STILL | tr '.' 'p')"

AM_FLAG=""; AM_TAG=""
awk "BEGIN{exit !($LL_ANGMOM > 0)}" && AM_FLAG="--agent.ll-angmom-coef ${LL_ANGMOM}" \
  && AM_TAG="_angmom$(echo $LL_ANGMOM | tr '.' 'p')"

FS_FLAG=""; FS_TAG=""
awk "BEGIN{exit !($LL_FOOTSLIP > 0)}" && FS_FLAG="--agent.ll-footslip-coef ${LL_FOOTSLIP}" \
  && FS_TAG="_footslip$(echo $LL_FOOTSLIP | tr '.' 'p')"

FC_FLAG=""; FC_TAG=""
awk "BEGIN{exit !($LL_FOOTCLEAR > 0)}" && FC_FLAG="--agent.ll-footclear-coef ${LL_FOOTCLEAR}" \
  && FC_TAG="_footclear$(echo $LL_FOOTCLEAR | tr '.' 'p')"

EN_FLAG=""; EN_TAG=""
awk "BEGIN{exit !($LL_ENERGY > 0)}" && EN_FLAG="--agent.ll-energy-coef ${LL_ENERGY}" \
  && EN_TAG="_energy$(echo $LL_ENERGY | tr '.' 'p')"

# WL-D (2026-07-24): joint_acc_l2 mirror. Tagless at 0.0/off.
JA_FLAG=""; JA_TAG=""
awk "BEGIN{exit !($LL_JOINT_ACC > 0)}" && JA_FLAG="--agent.ll-joint-acc-coef ${LL_JOINT_ACC}" \
  && JA_TAG="_jacc$(echo $LL_JOINT_ACC | tr '.' 'p')"

# WL-D: joint_pos_limits mirror (the largest measured A1-vs-A0 reward-gap term).
JL_FLAG=""; JL_TAG=""
awk "BEGIN{exit !($LL_JOINT_LIMITS > 0)}" && JL_FLAG="--agent.ll-joint-limits-coef ${LL_JOINT_LIMITS}" \
  && JL_TAG="_jlim$(echo $LL_JOINT_LIMITS | tr '.' 'p')"

# WL-D: soft_landing mirror (first-contact impact-force penalty).
SL_FLAG=""; SL_TAG=""
awk "BEGIN{exit !($LL_SOFT_LANDING > 0)}" && SL_FLAG="--agent.ll-soft-landing-coef ${LL_SOFT_LANDING}" \
  && SL_TAG="_softland$(echo $LL_SOFT_LANDING | tr '.' 'p')"

# WL-D: body_angular_velocity_penalty mirror (torso xy angular velocity).
BAV_FLAG=""; BAV_TAG=""
awk "BEGIN{exit !($LL_BODY_ANG_VEL > 0)}" && BAV_FLAG="--agent.ll-body-ang-vel-coef ${LL_BODY_ANG_VEL}" \
  && BAV_TAG="_bodyangvel$(echo $LL_BODY_ANG_VEL | tr '.' 'p')"

# WL-D combo batch (2026-07-23): ll_action_rate_coef override (smoothness lever / A0
# action-rate mirror). Default 0.02 is the rl_cfg default, so only tag on divergence.
AR_FLAG=""; AR_TAG=""
awk "BEGIN{exit !($LL_ACTION_RATE != 0.02)}" && AR_FLAG="--agent.ll-action-rate-coef ${LL_ACTION_RATE}" \
  && AR_TAG="_ar$(echo $LL_ACTION_RATE | tr '.' 'p')"

# Arm 5: the ankle-roll posture anchor (weight fixed at 4.0 via the rl_cfg default).
ANKLE_FLAG=""; ANKLE_TAG=""
[[ "$ANKLE_ANCHOR" == "True" ]] && ANKLE_FLAG="--agent.ll-posture-anchor-ankle-roll True" \
  && ANKLE_TAG="_ankle"

# WL-D: extend the LL posture anchor to every joint (supersedes the subset + ankle_roll).
PAJ_FLAG=""; PAJ_TAG=""
[[ "$POSTURE_ALL_JOINTS" == "True" ]] && PAJ_FLAG="--agent.ll-posture-all-joints True" \
  && PAJ_TAG="_postureall"

RUN_NAME="a1a${CAD_TAG}${CADENCE_COEF_TAG}${SS_TAG}${AM_TAG}${FS_TAG}${FC_TAG}${EN_TAG}${JA_TAG}${JL_TAG}${SL_TAG}${BAV_TAG}${AR_TAG}${ANKLE_TAG}${PAJ_TAG}${VS_TAG}_s${SEED}"

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
    $AM_FLAG $FS_FLAG $FC_FLAG $JA_FLAG $JL_FLAG $SL_FLAG $BAV_FLAG $AR_FLAG $ANKLE_FLAG $PAJ_FLAG \
    --agent.run-name ${RUN_NAME}

wandb sync --sync-all
