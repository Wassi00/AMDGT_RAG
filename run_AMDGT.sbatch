#!/bin/bash
#SBATCH --job-name=AMDGT
#SBATCH --partition=student
#SBATCH --begin=now
#SBATCH --gres=gpu:0
#SBATCH --output=output/amdgt_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_gpu
export CUDA_VISIBLE_DEVICES=1
srun python train_DDA.py --dataset B-dataset --epochs 1000 > output/run_$(date +%Y%m%d_%H%M%S)_B.log