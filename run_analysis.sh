#!/bin/bash
# =============================================================================
# run_analysis.sh — run after all training jobs complete
# Runs the Fourier spectrum analysis using the best checkpoints from each seed,
# then aggregates accuracy results across seeds into a summary table.
# =============================================================================

#SBATCH --job-name=shd_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/analysis_%j.out

module load conda
conda activate maxformer
cd "${SLURM_SUBMIT_DIR}"

DATA_PATH="/scratch/${USER}/data/shd"
SEED=42   # use seed 42 for the spectrum analysis (representative single run)

# ---------- Fourier spectrum analysis ----------
python analyse_spectrum.py \
    --avg-ckpt      "logs/shd_snn_avg_T16_seed${SEED}/checkpoint_best.pth" \
    --max-ckpt      "logs/shd_snn_max_T16_seed${SEED}/checkpoint_best.pth" \
    --ann-avg-ckpt  "logs/shd_ann_avg_T16_seed${SEED}/checkpoint_best.pth" \
    --ann-max-ckpt  "logs/shd_ann_max_T16_seed${SEED}/checkpoint_best.pth" \
    --data-path     "${DATA_PATH}" \
    --T 16 \
    --n-batches 30 \
    --out-dir       ./figures

# ---------- aggregate accuracy results ----------
python aggregate_results.py --log-dir ./logs --out ./results_summary.csv

echo "Analysis complete."
