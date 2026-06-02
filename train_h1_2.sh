#!/bin/bash
#SBATCH --job-name=h1_2_a0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/work/%u/ramlab_ws/slurm_logs/%j.out
#SBATCH --error=/data/work/%u/ramlab_ws/slurm_logs/%j.err

eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate unitree_mjlab_h1_2_rl
# fallback if wrong python is used
export PATH=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/bin:$PATH

# Use the conda env's CUDA 12.8 libraries (system only has cuda/12.3)
SITE_PKGS=$WORK/miniconda3/envs/unitree_mjlab_h1_2_rl/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SITE_PKGS/nvidia/cuda_nvrtc/lib:$SITE_PKGS/nvidia/cuda_runtime/lib:$SITE_PKGS/nvidia/cublas/lib:$LD_LIBRARY_PATH

ulimit -l unlimited

# Redirect warp cache and compiler temp to $WORK (compute node /tmp is too small)
export WARP_CACHE_PATH=$WORK/.warp_cache
export TMPDIR=$WORK/tmp
mkdir -p $WORK/.warp_cache $WORK/tmp

export WANDB_MODE=offline

cd $WORK/ramlab_ws/code/unitree_rl_mjlab

nvidia-smi dmon -s u -d 1 > gpu_util.log &

python scripts/train.py Unitree-H1_2-Flat \
    --env.scene.num-envs ${NUM_ENVS:-16384} \
    --agent.max-iterations ${MAX_ITER:-6001}

# Auto-sync after training
wandb sync --sync-all