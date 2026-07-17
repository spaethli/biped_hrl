#!/bin/bash
#SBATCH --job-name=h1_2_a0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/work/%u/ramlab_ws/slurm_logs/%j.out
#SBATCH --error=/data/work/%u/ramlab_ws/slurm_logs/%j.err
#
# A0 baseline. Knobs via env vars (WL-D, 2026-07-17 - same tagged-deviation convention
# as train_h1_2_a1.sh: only non-default knobs get a run-name tag). These are all
# A0-scoped (no baseline mismatch vs the plain Unitree-H1_2-Flat task, unlike the
# A1a LL-reward batch - see train_h1_2_a1a_LL_rewards.sh for that one):
#   SEED         (default 42)                        -> always _sSEED
#   ENV_VARIANT  default|explicit_pd|wide_dr|corr_noise (default: default) -> tags
#                _pd (arm 7) | _widedr (arm 8) | _corrnoise (arm 9)
#   ENERGY_COEF  float (default 0.0)                  -> >0 tags _energyXpXX (WL-D
#                S4' "A0+energy" control: --env.rewards.cost-of-transport.weight)
#   NUM_ENVS, MAX_ITER
#
# Examples:
#   sbatch train_h1_2.sh                                          # a0_baseline_s42
#   sbatch --export=ALL,ENERGY_COEF=0.05 train_h1_2.sh             # a0_baseline_energy0p05_s42
#   sbatch --export=ALL,ENV_VARIANT=explicit_pd train_h1_2.sh      # a0_baseline_pd_s42
#   sbatch --export=ALL,ENV_VARIANT=wide_dr train_h1_2.sh          # a0_baseline_widedr_s42
#   sbatch --export=ALL,ENV_VARIANT=corr_noise train_h1_2.sh       # a0_baseline_corrnoise_s42

eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate unitree_mjlab_h1_2_rl
# fallback if wrong python is used
export PATH=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/bin:$PATH

# Use the conda env's CUDA 12.8 libraries (system only has cuda/12.3)
SITE_PKGS=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SITE_PKGS/nvidia/cuda_nvrtc/lib:$SITE_PKGS/nvidia/cuda_runtime/lib:$SITE_PKGS/nvidia/cublas/lib:$LD_LIBRARY_PATH

ulimit -l unlimited

# Redirect warp cache and compiler temp to $WORK (compute node /tmp is too small);
# per-job cache dir (2026-06-19 fix) so concurrent jobs never share cache files.
export WARP_CACHE_PATH=$WORK/.warp_cache/$SLURM_JOB_ID
export TMPDIR=$WORK/tmp
mkdir -p $WARP_CACHE_PATH $WORK/tmp

export WANDB_MODE=offline

cd $WORK/ramlab_ws/code/unitree_rl_mjlab

SEED=${SEED:-42}
ENV_VARIANT=${ENV_VARIANT:-default}
ENERGY_COEF=${ENERGY_COEF:-0.0}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-10001}

case "$ENV_VARIANT" in
  default)     TASK_ID="Unitree-H1_2-Flat";            VARIANT_TAG="" ;;
  explicit_pd) TASK_ID="Unitree-H1_2-Flat-ExplicitPD";  VARIANT_TAG="_pd" ;;
  wide_dr)     TASK_ID="Unitree-H1_2-Flat-WideDR";      VARIANT_TAG="_widedr" ;;
  corr_noise)  TASK_ID="Unitree-H1_2-Flat-CorrNoise";   VARIANT_TAG="_corrnoise" ;;
  *) echo "Unknown ENV_VARIANT=$ENV_VARIANT" >&2; exit 1 ;;
esac

ENERGY_FLAG=""; ENERGY_TAG=""
awk "BEGIN{exit !($ENERGY_COEF > 0)}" && \
  ENERGY_FLAG="--env.rewards.cost-of-transport.weight -${ENERGY_COEF}" \
  && ENERGY_TAG="_energy$(echo $ENERGY_COEF | tr '.' 'p')"

RUN_NAME="a0_baseline${VARIANT_TAG}${ENERGY_TAG}_s${SEED}"

echo "[a0] RUN=$RUN_NAME  task=$TASK_ID  energy_coef=$ENERGY_COEF  seed=$SEED"

# Create a gpu usage log
#nvidia-smi dmon -s u -d 1 > gpu_util.log &

python scripts/train.py ${TASK_ID} \
    --env.scene.num-envs ${NUM_ENVS} \
    --agent.max-iterations ${MAX_ITER} \
    --agent.seed ${SEED} \
    $ENERGY_FLAG \
    --agent.run-name ${RUN_NAME}

# Auto-sync after training
wandb sync --sync-all