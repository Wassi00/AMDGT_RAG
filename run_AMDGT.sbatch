#!/bin/bash
#SBATCH --job-name=AMDGT
#SBATCH --partition=student
#SBATCH --begin=now
#SBATCH --gres=gpu:0
#SBATCH --output=output/amdgt_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_gpu
export CUDA_VISIBLE_DEVICES=0
srun python train_DDA.py --dataset F-dataset --retrieval_mode "full" --top_k "10" --epochs 1000 > output/run_$(date +%Y%m%d_%H%M%S)_F_full_10_cold_drug.log

# for dataset in B-dataset C-dataset F-dataset; do
#     for mode in full baseline retrieval; do
#         for top_k in 10 5 3; do
#             echo "Running: dataset=$dataset, mode=$mode, top_k=$top_k"
#             python train_DDA.py \
#                 --dataset "$dataset" \
#                 --retrieval_mode "$mode" \
#                 --top_k "$top_k" \
#                 --epochs 1000 \
#                 >> output/run_$(date +%Y%m%d_%H%M%S)_${dataset}_${mode}_${top_k}.log
#         done
#     done
# done

# Push results to GitHub
# git add .
# git commit -m "Results from AMDGT training runs - $(date +%Y%m%d_%H%M%S)"
# git push origin master