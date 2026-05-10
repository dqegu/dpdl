#!/bin/bash -l
#SBATCH -p l4,scavenger_l4,ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ao1g22@soton.ac.uk
#SBATCH --array=0-19          # 5 seeds x 4 conditions (dropping shd_max_former)
#SBATCH -o slurm_logs/shd_%A_%a.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spens-seq

export OMP_NUM_THREADS=8

export LD_LIBRARY_PATH=/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cublas/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cudnn/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cufft/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/curand/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cusolver/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cusparse/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# ---------- paths ----------
WORK_DIR=/iridisfs/home/ao1g22/comp6228/dpdl
DATA_PATH=/iridisfs/home/ao1g22/comp6228/dpdl/data/shd

cd "${WORK_DIR}"
mkdir -p slurm_logs

# ---------- parameter grid ----------
# 4 conditions x 5 seeds = 20 jobs (array 0-19)
CONDITIONS=(shd_snn_avg shd_snn_max shd_ann_avg shd_ann_max)
SEEDS=(42 123 456 789 1234)

N_CONDITIONS=${#CONDITIONS[@]}   # 4

COND_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

MODEL=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}

EXP_NAME="${MODEL}_T16_seed${SEED}"

echo "Job ${SLURM_ARRAY_TASK_ID}: model=${MODEL}  seed=${SEED}"
echo "Output: logs/${EXP_NAME}"

python train.py \
    -c shd.yaml \
    --model      "${MODEL}" \
    --data-path  "${DATA_PATH}" \
    --seed       "${SEED}" \
    --experiment "${EXP_NAME}" \
    --output-dir ./logs \
    --device     cuda:0

echo "Job ${SLURM_ARRAY_TASK_ID} complete."