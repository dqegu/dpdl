#!/bin/bash
# =============================================================================
# run_iridis.sh — SLURM array job for SHD MaxFormer experiments on Iridis
#
# Submits 5 seeds × 5 model conditions = 25 jobs in parallel.
# Each job takes ~2 hours on a single GPU for 96 epochs on SHD.
# Total wall-clock time with parallelism: ~2-3 hours.
#
# Usage:
#   sbatch run_iridis.sh
#
# After all jobs complete, run the analysis:
#   sbatch run_analysis.sh
# =============================================================================

#SBATCH --job-name=shd_maxformer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --array=0-24          # 5 seeds × 5 conditions = 25 jobs
#SBATCH --output=slurm_logs/shd_%A_%a.out
#SBATCH --error=slurm_logs/shd_%A_%a.err

# ---------- environment ----------
module load conda
conda activate maxformer       # adjust to your environment name

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

# ---------- parameter grid ----------
SEEDS=(42 123 456 789 1234)
CONDITIONS=(shd_snn_avg shd_snn_max shd_max_former shd_ann_avg shd_ann_max)

N_SEEDS=${#SEEDS[@]}
N_CONDITIONS=${#CONDITIONS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
COND_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))

SEED=${SEEDS[$SEED_IDX]}
MODEL=${CONDITIONS[$COND_IDX]}

# ---------- paths ----------
# Edit this to point at your SHD data directory on Iridis scratch
DATA_PATH="/scratch/${USER}/data/shd"

EXP_NAME="${MODEL}_T16_seed${SEED}"
OUTPUT_DIR="./logs"

echo "Job ${SLURM_ARRAY_TASK_ID}: model=${MODEL} seed=${SEED}"
echo "Output: ${OUTPUT_DIR}/${EXP_NAME}"

# ---------- training ----------
python train.py \
    -c shd.yaml \
    --model       "${MODEL}" \
    --data-path   "${DATA_PATH}" \
    --seed        "${SEED}" \
    --experiment  "${EXP_NAME}" \
    --output-dir  "${OUTPUT_DIR}" \
    --device      cuda:0

echo "Job ${SLURM_ARRAY_TASK_ID} complete."
