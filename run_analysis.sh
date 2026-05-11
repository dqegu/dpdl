#!/bin/bash -l
#SBATCH -p l4,scavenger_l4,ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ao1g22@soton.ac.uk
#SBATCH -o slurm_logs/analysis_%j.out

__conda_setup="$('/iridisfs/ixsoftware/conda/miniconda-py3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh" ]; then
        . "/iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh"
    else
        export PATH="/iridisfs/ixsoftware/conda/miniconda-py3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate spens-seq

export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cublas/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cudnn/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cufft/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/curand/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cusolver/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cusparse/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

WORK_DIR=/iridisfs/home/ao1g22/comp6258/dpdl
DATA_PATH=/iridisfs/home/ao1g22/comp6258/dpdl/data/shd
SEED=42

cd "${WORK_DIR}"
mkdir -p figures slurm_logs

# Run spectrum analysis with reduced batch count and smaller batch size
# to avoid OOM. 5 batches of 8 samples is plenty for stable spectrum estimates.
python analyse_spectrum.py \
    --avg-ckpt     "logs/shd_snn_avg_T16_seed${SEED}/checkpoint_best.pth" \
    --max-ckpt     "logs/shd_snn_max_T16_seed${SEED}/checkpoint_best.pth" \
    --ann-avg-ckpt "logs/shd_ann_avg_T16_seed${SEED}/checkpoint_best.pth" \
    --ann-max-ckpt "logs/shd_ann_max_T16_seed${SEED}/checkpoint_best.pth" \
    --data-path    "${DATA_PATH}" \
    --T 16 \
    --n-batches 5 \
    --batch-size 8 \
    --out-dir      ./figures

echo "Analysis complete."