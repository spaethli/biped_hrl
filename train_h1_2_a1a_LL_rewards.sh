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
#   STANDING_FRAC     float (default 0.05)         -> WL-S 2026-08-26 standing-gap lever:
#                     rel_standing_envs, the FRACTION of envs commanded to stand. An
#                     --env knob, not --agent (the twist command term); != 0.05 tags
#                     _standingNN as a PERCENT (_standing15), matching the 08-26 run.
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
# ============================================================================
# BASELINE (2026-08-25): full_jacc, folded in from scripts/launch_a1_reward_arms.sh
# ============================================================================
# The defaults below ARE `full_jacc`'s config -- the 2026-08-05 reward-mirror battery arm
# that leads on leg smoothness (act_legs 0.7014 vs A0's 0.5933) and the only one of three
# that cleared the 2026-08-24 bridge battery (GO/GO, alpha_max 0.0). So a bare `sbatch`
# reproduces it (fresh timestamp dir), the same way train_h1_2_a1_hl_vel_source.sh's bare
# call reproduces the deployed hardware baseline. Every knob below moves you off it, and
# the run name tags only the DEVIATION.
#
# ⚠ This CHANGED the script's baseline. Pre-2026-08-25 examples in git history assumed
# a bare base where every lever defaulted to 0.0; they do not mean the same thing now.
#
# --- the 7-arm batch (the point of the fold-in) -----------------------------
# Each arm is its own sbatch, so the cluster runs the batch in parallel. Paste all seven:
#
#   # repro: the control. 13 commits landed since the battery ran at 4739b1c (+1317 lines
#   # in the training path, all inert at hl_vel_source=state), so this re-establishes the
#   # 0.7014 reference the other six are scored against.
#   sbatch --job-name=a1a_repro    train_h1_2_a1a_LL_rewards.sh
#   # arm 1 -- joint-acc dose response, bracketing the 2.5e-7 baseline. Watch zero-command
#   # touchdowns: 5e-7 historically marched continuously (5325 vs a 128 floor).
#   sbatch --job-name=a1a_jacc1e-7 --export=ALL,LL_JOINT_ACC=1e-7 train_h1_2_a1a_LL_rewards.sh
#   sbatch --job-name=a1a_jacc5e-7 --export=ALL,LL_JOINT_ACC=5e-7 train_h1_2_a1a_LL_rewards.sh
#   # arm 2 -- action-rate dose. PRE-REGISTERED TO FAIL: the one dose step we have
#   # (0.02 -> 0.05) bought action_rate -8.7% but cost jacc_legs_p95 +31.1%. Stop rule: if
#   # jacc_legs_p95 > 70.55 (A0's), the lever is exhausted -- do NOT run 0.10, report closed.
#   sbatch --job-name=a1a_ar0p07   --export=ALL,LL_ACTION_RATE=0.07 train_h1_2_a1a_LL_rewards.sh
#   # WL-S 2026-08-26 -- the standing sweep. The gap A1-vs-A0 is 31-44x STANDING and only
#   # 1.3-1.6x walking, and 0.05 -> 0.15 took standing ajit to 0.75x A0 at the touchdown
#   # floor WITHOUT costing tracking (ss_err_vx 0.0528, best in batch). n=1 seed, and it
#   # diverges A1's command distribution from A0's 0.05 -- an RQ2 call before it is a
#   # default. Baseline arm is a plain sbatch (0.05); 15 is a re-run of the 08-26 dir.
#   sbatch --job-name=a1a_stand10 --export=ALL,LL_JOINT_ACC=1e-7,STANDING_FRAC=0.10 train_h1_2_a1a_LL_rewards.sh
#   sbatch --job-name=a1a_stand25 --export=ALL,LL_JOINT_ACC=1e-7,STANDING_FRAC=0.25 train_h1_2_a1a_LL_rewards.sh
#   sbatch --job-name=a1a_stand15b --export=ALL,LL_JOINT_ACC=1e-7,STANDING_FRAC=0.15,SEED=123 train_h1_2_a1a_LL_rewards.sh
#   # arm 4 -- cadence family. Bounding measurements, not expected fixes: regressing all 13
#   # battery arms' leg metrics on achieved stride period gives r=-0.33 (n=13, p~0.28), and
#   # even extrapolated to A0's 0.593 s stride the batch lands at act_legs 0.682 -- still
#   # +15% over A0, with jacc_legs still +68%.
#   #   nocad    : cadence off -> mdp.phase falls back to A0's fixed 0.6 s clock
#   #              (observations.py:66) and feet_gait self-disables. The HL loses its period
#   #              action dim, so the TD3 nets are task_dim, not task_dim+1.
#   #   pin0p625 : source=random drops the HL's period dim -> a TWO-variable arm. For a
#   #              one-variable pin, drop CADENCE_SOURCE: a degenerate range makes the affine
#   #              map return 0.625 for every action, so the dim goes inert while the nets
#   #              keep the baseline's shape.
#   #   floor0p5 : raise the floor only; the HL keeps commanding period.
#   sbatch --job-name=a1a_nocad    --export=ALL,HL_CADENCE=False,CADENCE_SOURCE=random train_h1_2_a1a_LL_rewards.sh
#   sbatch --job-name=a1a_pin0625  --export=ALL,CADENCE_SOURCE=random,CADENCE_RANGE=0.625,0.625 train_h1_2_a1a_LL_rewards.sh
#   sbatch --job-name=a1a_floor0p5 --export=ALL,CADENCE_RANGE=0.5,1.0 train_h1_2_a1a_LL_rewards.sh
#
# ⚠ CADENCE_RANGE contains a comma, and --export splits on commas. Quote it as
#   --export=ALL,"CADENCE_RANGE=0.5,1.0"  or  export it and use --export=ALL.
#
# --- other knobs ------------------------------------------------------------
#   # a second seed (always tagged _sNNN):
#   sbatch --export=ALL,SEED=123 train_h1_2_a1a_LL_rewards.sh
#   # the historical fixed-clock control (sets cadence source/range/cot together):
#   sbatch --export=ALL,FIX0P8=True train_h1_2_a1a_LL_rewards.sh
#   # arms 5/6/10, all off in the baseline:
#   sbatch --export=ALL,ANKLE_ANCHOR=True train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,LL_PITCHREF=0.5 train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,LL_PUSHOFF=0.5 train_h1_2_a1a_LL_rewards.sh
#   sbatch --export=ALL,LL_SYMMETRY=0.25 train_h1_2_a1a_LL_rewards.sh
#   # print the command without launching (works off-cluster):
#   DRY_RUN=1 LL_JOINT_ACC=1e-7 ./train_h1_2_a1a_LL_rewards.sh

# Cluster preamble, skipped under DRY_RUN so the command can be inspected off-cluster.
if [[ "${DRY_RUN:-0}" == "0" ]]; then
  eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
  conda activate unitree_mjlab_h1_2_rl
  export PATH=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/bin:$PATH

  SITE_PKGS=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/lib/python3.11/site-packages
  export LD_LIBRARY_PATH=$SITE_PKGS/nvidia/cuda_nvrtc/lib:$SITE_PKGS/nvidia/cuda_runtime/lib:$SITE_PKGS/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

  ulimit -l unlimited
  # Per-job cache dir: co-scheduled jobs sharing one warp cache on NFS collide (2026-06-19).
  export WARP_CACHE_PATH=$WORK/.warp_cache/$SLURM_JOB_ID
  export TMPDIR=$WORK/tmp
  mkdir -p $WARP_CACHE_PATH $WORK/tmp
  export WANDB_MODE=offline

  cd $WORK/ramlab_ws/code/unitree_rl_mjlab
fi

# ⚠ BASELINE CHANGED AGAIN 2026-08-26 -- Model v3 (ADR-0008) landed, and it is not a
# lever in this script: the commanded-target clip rides the ENV cfg
# (config/h1_2/env_cfgs.py:99, which the A1 task inherits verbatim), the +3.5 LL mirror is
# an rl_cfg default (ll_action_clip_coef), and the x1.20787 leg mass is in the XML. So a
# bare `sbatch` now trains v3 and logs to h1_2_velocity_a1_v3 -- it no longer reproduces
# `full_jacc`, and NO v2 run below is a valid comparator for anything launched now.
# Any new sweep must carry its own v3 baseline arm.
#
# ⚠ BASELINE CHANGED 2026-08-25. Everything below defaults to `full_jacc`'s config, not
# to the bare rl_cfg defaults, so a plain `sbatch` reproduces that run (the `repro` arm).
# Consequence for provenance: a WL-D-era example like `--export=ALL,LL_ENERGY=0.05` no
# longer means "arm 4d alone off a bare base" -- it now sets energy to a value it already
# has, on the full-mirror base. To re-run a pre-2026-08-25 arm, zero the other levers
# explicitly or use git to recover this file's older defaults.
SEED=${SEED:-42}
LL_CADENCE_COEF=${LL_CADENCE_COEF:-0.5}
HL_COT=${HL_COT:-0.2}
FIX0P8=${FIX0P8:-False}
LL_STAND_STILL=${LL_STAND_STILL:-1.0}
LL_ANGMOM=${LL_ANGMOM:-0.025}
LL_FOOTSLIP=${LL_FOOTSLIP:-0.25}
LL_FOOTCLEAR=${LL_FOOTCLEAR:-1.0}
LL_ENERGY=${LL_ENERGY:-0.05}
LL_JOINT_ACC=${LL_JOINT_ACC:-2.5e-7}
LL_JOINT_LIMITS=${LL_JOINT_LIMITS:-10.0}
LL_SOFT_LANDING=${LL_SOFT_LANDING:-1e-3}
LL_BODY_ANG_VEL=${LL_BODY_ANG_VEL:-0.05}
POSTURE_ALL_JOINTS=${POSTURE_ALL_JOINTS:-True}
LL_ACTION_RATE=${LL_ACTION_RATE:-0.05}
ANKLE_ANCHOR=${ANKLE_ANCHOR:-False}
LL_PITCHREF=${LL_PITCHREF:-0.0}
LL_PUSHOFF=${LL_PUSHOFF:-0.0}
LL_SYMMETRY=${LL_SYMMETRY:-0.0}
LL_MIRROR=${LL_MIRROR:-0.0}
STANDING_FRAC=${STANDING_FRAC:-0.05}
# Cadence family (new 2026-08-25): the three cadence arms need more than a scalar.
# CADENCE_RANGE empty = leave at the rl_cfg default (0.35,1.0), which is what full_jacc
# trained with. Tuple syntax: "lo,hi" or "(lo,hi)" -- both parse (mjlab sets tyro's
# UsePythonSyntaxForLiteralCollections); only the space-separated form is rejected.
HL_CADENCE=${HL_CADENCE:-True}
CADENCE_SOURCE=${CADENCE_SOURCE:-hl}
CADENCE_RANGE=${CADENCE_RANGE:-}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-10001}

# Every lever's flag is now ALWAYS passed: the baseline lives in the flags above, not in
# rl_cfg's defaults, so omitting one would silently drop it to 0 rather than hold it at
# the batch baseline. Only the run-name TAG is conditional -- it fires on DEVIATION from
# the baseline, per the run-naming convention. Called 15x below, hence a helper.
LEVER_FLAGS=(); LEVER_TAGS=""
lever () {  # lever <cli-flag> <value> <baseline> <tag-stem>
  LEVER_FLAGS+=("$1" "$2")
  awk "BEGIN{exit !($2 == $3)}" || LEVER_TAGS+="_$4$(echo "$2" | tr '.' 'p')"
}
flag () {   # flag <cli-flag> <True|False> <baseline> <tag>  -- booleans
  LEVER_FLAGS+=("$1" "$2")
  [[ "$2" == "$3" ]] || LEVER_TAGS+="_$4"
}

# Arm 3/4 A0-reward mirrors, then the WL-D additions. Baselines = full_jacc's values.
lever --agent.ll-stand-still-coef  "$LL_STAND_STILL"  1.0     standstill
lever --agent.ll-angmom-coef       "$LL_ANGMOM"       0.025   angmom
lever --agent.ll-footslip-coef     "$LL_FOOTSLIP"     0.25    footslip
lever --agent.ll-footclear-coef    "$LL_FOOTCLEAR"    1.0     footclear
lever --agent.ll-energy-coef       "$LL_ENERGY"       0.05    energy
lever --agent.ll-joint-acc-coef    "$LL_JOINT_ACC"    2.5e-7  jacc
lever --agent.ll-joint-limits-coef "$LL_JOINT_LIMITS" 10.0    jlim
lever --agent.ll-soft-landing-coef "$LL_SOFT_LANDING" 1e-3    softland
lever --agent.ll-body-ang-vel-coef "$LL_BODY_ANG_VEL" 0.05    bodyangvel
lever --agent.ll-action-rate-coef  "$LL_ACTION_RATE"  0.05    ar
lever --agent.ll-cadence-coef      "$LL_CADENCE_COEF" 0.5     cad

# Arms 5/6/10: off in the baseline, so these stay tagless unless swept. See A1a_plan.md.
lever --agent.ll-pitchref-coef     "$LL_PITCHREF"     0.0     pitchref
lever --agent.ll-pushoff-coef      "$LL_PUSHOFF"      0.0     pushoff
lever --agent.ll-symmetry-coef     "$LL_SYMMETRY"     0.0     sym
lever --agent.ll-mirror-coef       "$LL_MIRROR"       0.0     mirror

# rel_standing_envs lives on the env (twist command term), not the agent, and its tag is
# a percent -- so it gets its own two lines instead of a `lever` call. Last, so a sweep
# reproduces the 2026-08-26 `_jacc1e-7_standing15` name rather than reordering the tags.
LEVER_FLAGS+=(--env.commands.twist.rel-standing-envs "$STANDING_FRAC")
awk "BEGIN{exit !($STANDING_FRAC == 0.05)}" \
  || LEVER_TAGS+="_standing$(awk "BEGIN{printf \"%g\", $STANDING_FRAC*100}")"

flag --agent.ll-posture-all-joints        "$POSTURE_ALL_JOINTS" True  posturesubset
flag --agent.ll-posture-anchor-ankle-roll "$ANKLE_ANCHOR"       False ankle

# Cadence channel. FIX0P8 stays as the historical shorthand for the fixed-clock control;
# otherwise the three knobs are set independently, which is what the nocad / pin0p625 /
# floor0p5 arms need. CADENCE_RANGE empty = the rl_cfg default the baseline trained with.
if [[ "$FIX0P8" == "True" ]]; then
  HL_CADENCE=True; CADENCE_SOURCE=random; CADENCE_RANGE=0.8,0.8; HL_COT=0.0
fi
# hl_cadence_source='hl' requires hl_cadence=True (rl_cfg raises a loud error otherwise).
# Catch it here rather than after the job has been queued and allocated a GPU.
if [[ "$HL_CADENCE" != "True" && "$CADENCE_SOURCE" == "hl" ]]; then
  echo "ERROR: HL_CADENCE=False needs CADENCE_SOURCE=random (source='hl' requires cadence on)." >&2
  exit 2
fi
CAD_FLAG="--agent.hl-cadence ${HL_CADENCE} --agent.hl-cadence-source ${CADENCE_SOURCE} --agent.hl-cot-coef ${HL_COT}"
[[ -n "$CADENCE_RANGE" ]] && CAD_FLAG="$CAD_FLAG --agent.cadence-period-range ${CADENCE_RANGE}"

# Run-name tag for the cadence family. The first two cases NAME themselves (a fixed clock
# or no clock makes the other three knobs inert, so tagging them would be noise); below
# that the knobs are independent and their tags ACCUMULATE, so pin0p625's two-variable
# nature (source AND range) is visible in the directory listing, not just in the header.
CAD_TAG=""
if [[ "$FIX0P8" == "True" ]]; then
  CAD_TAG="_fix0p8"
elif [[ "$HL_CADENCE" != "True" ]]; then
  CAD_TAG="_nocad"
else
  [[ "$CADENCE_SOURCE" != "hl" ]] && CAD_TAG+="_cadrand"
  if [[ -n "$CADENCE_RANGE" ]]; then
    # Accept both accepted tuple syntaxes; "_per" not "_cad", which ll_cadence_coef owns.
    R=${CADENCE_RANGE//[()[:space:]]/}
    if [[ "${R%,*}" == "${R#*,}" ]]; then
      CAD_TAG+="_pin$(echo "${R%,*}" | tr '.' 'p')"
    else
      CAD_TAG+="_per$(echo "$R" | tr '.,' 'p-')"
    fi
  fi
  # Numeric compare, matching `lever`: a string test would tag 0.20 as a deviation.
  awk "BEGIN{exit !($HL_COT == 0.2)}" || CAD_TAG+="_cot$(echo "$HL_COT" | tr '.' 'p')"
fi

RUN_NAME="a1a_fullmirror${CAD_TAG}${LEVER_TAGS}_s${SEED}"

echo "[a1a-ll] RUN=$RUN_NAME  hl_cadence=${HL_CADENCE}  cadence_source=${CADENCE_SOURCE}  cadence_range=${CADENCE_RANGE:-<rl_cfg default>}  hl_cot=${HL_COT}  ll_cadence_coef=${LL_CADENCE_COEF}  stand_still=${LL_STAND_STILL}  angmom=${LL_ANGMOM}  footslip=${LL_FOOTSLIP}  footclear=${LL_FOOTCLEAR}  energy=${LL_ENERGY}  joint_acc=${LL_JOINT_ACC}  joint_limits=${LL_JOINT_LIMITS}  soft_landing=${LL_SOFT_LANDING}  body_ang_vel=${LL_BODY_ANG_VEL}  action_rate=${LL_ACTION_RATE}  ankle_anchor=${ANKLE_ANCHOR}  posture_all_joints=${POSTURE_ALL_JOINTS}  pitchref=${LL_PITCHREF}  pushoff=${LL_PUSHOFF}  symmetry=${LL_SYMMETRY}  mirror=${LL_MIRROR}  rel_standing_envs=${STANDING_FRAC}  seed=${SEED}"

# DRY_RUN=1 prints the command instead of running it -- check what a job will launch
# before it burns a slot. Works off-cluster too (nothing above needs $SLURM_JOB_ID).
if [[ "${DRY_RUN:-0}" != "0" ]]; then
  set -- python scripts/train.py Unitree-H1_2-Flat-A1 \
      --env.scene.num-envs "${NUM_ENVS}" --agent.max-iterations "${MAX_ITER}" \
      --agent.seed "${SEED}" $CAD_FLAG "${LEVER_FLAGS[@]}" --agent.run-name "${RUN_NAME}"
  printf '%q ' "$@"; echo
  exit 0
fi

python scripts/train.py Unitree-H1_2-Flat-A1 \
    --env.scene.num-envs ${NUM_ENVS} \
    --agent.max-iterations ${MAX_ITER} \
    --agent.seed ${SEED} \
    $CAD_FLAG \
    "${LEVER_FLAGS[@]}" \
    --agent.run-name ${RUN_NAME}

wandb sync --sync-all
