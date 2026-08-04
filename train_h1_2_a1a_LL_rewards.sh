#!/bin/bash
#SBATCH --job-name=h1_2_a1a_ll
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/work/%u/ramlab_ws/slurm_logs/%j.out
#SBATCH --error=/data/work/%u/ramlab_ws/slurm_logs/%j.err
#
# WL-D (2026-07-17): A1a LL/HL reward-lever batch (arms 1,2,3,4a-d,5 + the fix0p8+
# weights control). A DEDICATED script rather than more train_h1_2_a1.sh knobs, because
# this batch's own baseline is NOT that script's tagless default: it always trains the
# full A1a cadence-HL + CoT line (hl_cadence=True, source=hl, hl_cot_coef=0.2) on the
# arm-calmed base (ll_posture_weights promoted to default 2026-07-17 - shoulders x16,
# elbow+wrist x4 - see rl_cfg.py). Every run from this script carries that base; only
# the swept lever gets a run-name tag.
#
# Knobs via env vars:
#   SEED             (default 42)                -> always _sSEED
#   LL_CADENCE_COEF   float (default 0.5)          -> arm 1; != 0.5 tags _cadXpX
#   HL_COT            float (default 0.2)          -> arm 2; tags _cotXpXX (ignored
#                     under FIX0P8)
#   FIX0P8            True|False (default False)   -> the fixed-clock control (arm 1/2's
#                     comparator): pins cadence_period_range=(0.8,0.8) via
#                     hl-cadence-source=random (NOT hl) and hl-cot-coef=0; tags _fix0p8
#                     in place of the cot tag
#   LL_STAND_STILL    float (default 0.0)          -> arm 3; >0 tags _standstillXpXX
#   LL_ANGMOM         float (default 0.0)          -> arm 4a; >0 tags _angmomXpXXX
#   LL_FOOTSLIP       float (default 0.0)          -> arm 4b; >0 tags _footslipXpXX
#   LL_FOOTCLEAR      float (default 0.0)          -> arm 4c; >0 tags _footclearXpX
#   LL_ENERGY         float (default 0.0)          -> arm 4d; >0 tags _energyXpXX
#   LL_JOINT_ACC      float (default 0.0)          -> WL-D 2026-07-24 joint-accel mirror;
#                     >0 tags _jacc<val>
#   LL_JOINT_LIMITS   float (default 0.0)          -> WL-D joint_pos_limits (soft-limit
#                     crossing) mirror, the largest measured A1-vs-A0 reward-gap term
#                     (~515x); >0 tags _jlimXpX
#   LL_SOFT_LANDING   float (default 0.0)          -> WL-D soft_landing (first-contact
#                     impact-force) mirror; >0 tags _softlandXpXXX
#   LL_BODY_ANG_VEL   float (default 0.0)          -> WL-D body_angular_velocity_penalty
#                     (torso xy ang-vel) mirror; >0 tags _bodyangvelXpXX
#   POSTURE_ALL_JOINTS True|False (default False)  -> extends the LL posture anchor to
#                     every joint instead of the arms+waist+hip-yaw/roll subset (closest
#                     analog to A0's variable_posture); True tags _postureall
#   LL_ACTION_RATE    float (default 0.02)          -> WL-D combo batch (2026-07-23)
#                     "smoothness" lever / A0-mirror action-rate weight; != 0.02 tags
#                     _arXpXX (A0's own action_rate_l2 weight is 0.05, vs A1's tuned 0.02
#                     default - see rl_cfg.py ll_action_rate_coef docstring)
#   ANKLE_ANCHOR      True|False (default False)   -> arm 5 (weight fixed at 4.0 via the
#                     rl_cfg default); True tags _ankle
#   LL_PITCHREF       float (default 0.0)          -> arm 6 formulation A (heel-to-toe
#                     roll-over); >0 tags _pitchrefXpXX. theta_hs/theta_to/sigma/k stay
#                     at the rl_cfg probe-derived defaults (not swept here). A's read
#                     (2026-07-17/19): tracking regresses, roll-over stays sub-visible
#                     (~1.3->2.4 degrees ROM) - weaker candidate than the other arms.
#   LL_PUSHOFF        float (default 0.0)          -> arm 6 formulation B (ankle
#                     push-off power burst); >0 tags _pushoffXpXX. w/P_scale stay at the
#                     rl_cfg defaults (not swept here). **Direction-corrected 2026-07-20**
#                     (see rewards.py/rl_cfg.py): the original formula rewarded a
#                     dorsiflexion toe-lift habit as readily as genuine plantarflexion
#                     push-off (caught via the user's visual replay); fixed with a qd>0 gate
#                     + P_scale recalibrated 40->3.0W for the corrected (much sparser)
#                     power distribution.
#   LL_SYMMETRY       float (default 0.0)          -> arm 10 formulation B (step-time
#                     left/right symmetry index, the primary training arm - see
#                     A1a_plan.md "Arm 10"); >0 tags _symXpXX. sigma_si stays at the
#                     rl_cfg probe-derived default (not swept here).
#   LL_MIRROR         float (default 0.0)          -> arm 10 formulation A (half-period
#                     phase-shifted joint mirror); >0 tags _mirrorXpXX. Implemented but
#                     left UNTRAINED pending formulation B's read (the arm 6 pattern) -
#                     no launch example below on purpose.
#   NUM_ENVS, MAX_ITER
#
# Examples:
#   # the new fix0p8+weights baseline (needed once, comparator for arms 1/2):
#   sbatch --export=ALL,FIX0P8=True train_h1_2_a1a_LL_rewards.sh
#   # arm 1 (two runs):
#   sbatch --export=ALL,LL_CADENCE_COEF=1.0 train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,LL_CADENCE_COEF=2.0 train_h1_2_a1a_LL_rewards.sh
#   # arm 2 (two runs):
#   sbatch --export=ALL,HL_COT=0.15 train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,HL_COT=0.25 train_h1_2_a1a_LL_rewards.sh
#   # arm 3:
#   sbatch --export=ALL,LL_STAND_STILL=1.0 train_h1_2_a1a_LL_rewards.sh
#   # arm 4a-4d (one term per run):
#   sbatch --export=ALL,LL_ANGMOM=0.025 train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,LL_FOOTSLIP=0.25 train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,LL_FOOTCLEAR=1.0 train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,LL_ENERGY=0.05 train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,LL_ENERGY=0.05,LL_JOINT_ACC=2.5e-7 train_h1_2_a1a_LL_rewards.sh
#   # WL-D combo batch (2026-07-23), e.g. combo1 arm4d+smoothness:
#   sbatch --export=ALL,LL_ENERGY=0.05,LL_ACTION_RATE=0.05 train_h1_2_a1a_LL_rewards.sh
#   # arm 5:
#   sbatch --export=ALL,ANKLE_ANCHOR=True train_h1_2_a1a_LL_rewards.sh
#   # arm 6 (formulation A only):
#   sbatch --export=ALL,LL_PITCHREF=0.5 train_h1_2_a1a_LL_rewards.sh
#   # arm 6 (formulation B, exploratory):
#   sbatch --export=ALL,LL_PUSHOFF=0.5 train_h1_2_a1a_LL_rewards.sh
#   # arm 10 (formulation B, the primary training arm):
#   sbatch --export=ALL,LL_SYMMETRY=0.25 train_h1_2_a1a_LL_rewards.sh
#   # all A0 rewards in the LL:
#   sbatch --export=ALL,LL_ENERGY=0.05,LL_ACTION_RATE=0.05,LL_ANGMOM=0.025,LL_FOOTSLIP=0.25,LL_FOOTCLEAR=1.0,LL_STAND_STILL=1.0,LL_BODY_ANG_VEL=0.05,LL_SOFT_LANDING=1e-3,LL_JOINT_LIMITS=10.0,POSTURE_ALL_JOINTS=True train_h1_2_a1a_LL_rewards.sh

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
LL_CADENCE_COEF=${LL_CADENCE_COEF:-0.5}
HL_COT=${HL_COT:-0.2}
FIX0P8=${FIX0P8:-False}
LL_STAND_STILL=${LL_STAND_STILL:-0.0}
LL_ANGMOM=${LL_ANGMOM:-0.0}
LL_FOOTSLIP=${LL_FOOTSLIP:-0.0}
LL_FOOTCLEAR=${LL_FOOTCLEAR:-0.0}
LL_ENERGY=${LL_ENERGY:-0.0}
LL_JOINT_ACC=${LL_JOINT_ACC:-0.0}
LL_JOINT_LIMITS=${LL_JOINT_LIMITS:-0.0}
LL_SOFT_LANDING=${LL_SOFT_LANDING:-0.0}
LL_BODY_ANG_VEL=${LL_BODY_ANG_VEL:-0.0}
POSTURE_ALL_JOINTS=${POSTURE_ALL_JOINTS:-False}
LL_ACTION_RATE=${LL_ACTION_RATE:-0.02}
ANKLE_ANCHOR=${ANKLE_ANCHOR:-False}
LL_PITCHREF=${LL_PITCHREF:-0.0}
LL_PUSHOFF=${LL_PUSHOFF:-0.0}
LL_SYMMETRY=${LL_SYMMETRY:-0.0}
LL_MIRROR=${LL_MIRROR:-0.0}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-10001}

# Cadence channel: learned (default, tags _cotXpXX) or the fixed-clock control (arm
# 1/2's comparator, tags _fix0p8). The tuple CLI override for cadence-period-range
# takes comma syntax (verified 2026-07-17 - the old "tyro tuple override broken" note
# was a space-separated-syntax mistake, not a real limitation).
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

# Arm 6 (formulation A - see A1a_plan.md Arm 6): tagless at 0.0/off.
PR_FLAG=""; PR_TAG=""
awk "BEGIN{exit !($LL_PITCHREF > 0)}" && PR_FLAG="--agent.ll-pitchref-coef ${LL_PITCHREF}" \
  && PR_TAG="_pitchref$(echo $LL_PITCHREF | tr '.' 'p')"

# Arm 6 (formulation B, exploratory - see A1a_plan.md Arm 6): tagless at 0.0/off.
PO_FLAG=""; PO_TAG=""
awk "BEGIN{exit !($LL_PUSHOFF > 0)}" && PO_FLAG="--agent.ll-pushoff-coef ${LL_PUSHOFF}" \
  && PO_TAG="_pushoff$(echo $LL_PUSHOFF | tr '.' 'p')"

# Arm 10 formulation B (the primary training arm - A1a_plan.md "Arm 10"): tagless at
# 0.0/off.
SYM_FLAG=""; SYM_TAG=""
awk "BEGIN{exit !($LL_SYMMETRY > 0)}" && SYM_FLAG="--agent.ll-symmetry-coef ${LL_SYMMETRY}" \
  && SYM_TAG="_sym$(echo $LL_SYMMETRY | tr '.' 'p')"

# Arm 10 formulation A: implemented, left UNTRAINED pending B's read (the arm 6
# pattern) - included here for the same reason PR_FLAG/PO_FLAG both exist above.
MIR_FLAG=""; MIR_TAG=""
awk "BEGIN{exit !($LL_MIRROR > 0)}" && MIR_FLAG="--agent.ll-mirror-coef ${LL_MIRROR}" \
  && MIR_TAG="_mirror$(echo $LL_MIRROR | tr '.' 'p')"

RUN_NAME="a1a${CAD_TAG}${CADENCE_COEF_TAG}${SS_TAG}${AM_TAG}${FS_TAG}${FC_TAG}${EN_TAG}${JA_TAG}${JL_TAG}${SL_TAG}${BAV_TAG}${AR_TAG}${ANKLE_TAG}${PAJ_TAG}${PR_TAG}${PO_TAG}${SYM_TAG}${MIR_TAG}_s${SEED}"

echo "[a1a-ll] RUN=$RUN_NAME  cadence=${CAD_TAG}  ll_cadence_coef=${LL_CADENCE_COEF}  stand_still=${LL_STAND_STILL}  angmom=${LL_ANGMOM}  footslip=${LL_FOOTSLIP}  footclear=${LL_FOOTCLEAR}  energy=${LL_ENERGY}  joint_acc=${LL_JOINT_ACC}  joint_limits=${LL_JOINT_LIMITS}  soft_landing=${LL_SOFT_LANDING}  body_ang_vel=${LL_BODY_ANG_VEL}  action_rate=${LL_ACTION_RATE}  ankle_anchor=${ANKLE_ANCHOR}  posture_all_joints=${POSTURE_ALL_JOINTS}  pitchref=${LL_PITCHREF}  pushoff=${LL_PUSHOFF}  symmetry=${LL_SYMMETRY}  mirror=${LL_MIRROR}  seed=${SEED}"

python scripts/train.py Unitree-H1_2-Flat-A1 \
    --env.scene.num-envs ${NUM_ENVS} \
    --agent.max-iterations ${MAX_ITER} \
    --agent.seed ${SEED} \
    --agent.ll-cadence-coef ${LL_CADENCE_COEF} \
    $CAD_FLAG \
    $SS_FLAG $AM_FLAG $FS_FLAG $FC_FLAG $EN_FLAG $JA_FLAG $JL_FLAG $SL_FLAG $BAV_FLAG $AR_FLAG $ANKLE_FLAG $PAJ_FLAG $PR_FLAG $PO_FLAG $SYM_FLAG $MIR_FLAG \
    --agent.run-name ${RUN_NAME}

wandb sync --sync-all
