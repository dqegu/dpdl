# Deep Learning on the SHD Dataset — MaxFormer Replication & Extension

This project investigates whether the MaxFormer paper's claims about high-frequency feature preservation transfer from visual neuromorphic data (CIFAR10-DVS) to an auditory neuromorphic dataset (Spiking Heidelberg Digits, SHD). The core idea is that LIF neurons in SNNs act as low-pass filters, attenuating high-frequency information, and that MaxPool-based token mixing can restore this. We test this on SHD rather than the image-domain datasets used in the original paper.

Four model conditions are compared across five random seeds:

| Model | Description |
|---|---|
| `shd_snn_avg` | SNN with AvgPool token mixing — low-pass baseline |
| `shd_snn_max` | SNN with MaxPool token mixing — high-frequency restoration |
| `shd_ann_avg` | ANN (ReLU) equivalent of SNN-Avg |
| `shd_ann_max` | ANN (ReLU) equivalent of SNN-Max |

The SHD dataset has 700 cochlear frequency channels encoded as spike events. We bin these into `T=16` time frames and treat each sample as a 1D sequence.

## Repository structure

```
train.py              — main training script
models.py             — SNN and ANN model definitions
dataset.py            — SHD data loading via tonic
embedding_hub_1d.py   — 1D patch embedding variants (Orig, Avg, Max, Max+)
mixer_hub_1d.py       — 1D token mixing blocks (Avg, Max, DWC, SSA, Identity)
shd.yaml              — default hyperparameters (96 epochs, AdamW, cosine schedule)
aggregate_results.py  — collects best Acc@1 from all runs into a summary table
analyse_spectrum.py   — Fourier spectrum analysis (cochlear and temporal axes)
```

## Dependencies

- PyTorch + CUDA
- [timm](https://github.com/huggingface/pytorch-image-models)
- [spikingjelly](https://github.com/fangwei123456/spikingjelly)
- [tonic](https://tonic.readthedocs.io/) (for SHD data loading)

```bash
pip install timm spikingjelly tonic
```

## Data

The SHD dataset is downloaded automatically by tonic on first run. Point `--data-path` at an empty directory and tonic will handle the rest.

## Training

### Local / single GPU

```bash
python train.py \
    -c shd.yaml \
    --model      shd_snn_max \
    --data-path  /path/to/data/shd \
    --seed       42 \
    --experiment shd_snn_max_T16_seed42 \
    --output-dir ./logs \
    --device     cuda:0
```

Available model names: `shd_snn_avg`, `shd_snn_max`, `shd_ann_avg`, `shd_ann_max`

To run all 4 conditions × 5 seeds (20 jobs) on a SLURM cluster, use a script along the lines of:

```bash
#!/bin/bash -l
#SBATCH -p <your-partition>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH -o slurm_logs/shd_%A_%a.out

# activate your conda environment
conda activate <your-env>

WORK_DIR=<path-to-this-repo>
DATA_PATH=<path-to-shd-data>

cd "${WORK_DIR}"
mkdir -p slurm_logs

CONDITIONS=(shd_snn_avg shd_snn_max shd_ann_avg shd_ann_max)
SEEDS=(42 123 456 789 1234)

COND_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

MODEL=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}
EXP_NAME="${MODEL}_T16_seed${SEED}"

python train.py \
    -c shd.yaml \
    --model      "${MODEL}" \
    --data-path  "${DATA_PATH}" \
    --seed       "${SEED}" \
    --experiment "${EXP_NAME}" \
    --output-dir ./logs \
    --device     cuda:0
```

Checkpoints and a `results.csv` are saved to `logs/<experiment-name>/`.

## Aggregating results

Once training runs are done, collect best Acc@1 across all seeds:

```bash
python aggregate_results.py --log-dir ./logs --out ./results_summary.csv
```

This prints a table of mean ± std accuracy per condition and checks whether the SNN-Max > SNN-Avg gap matches the paper's claim.

## Spectrum analysis

To run the Fourier analysis (cochlear-axis and temporal-axis) comparing how high-frequency information is preserved or attenuated across model stages:

```bash
#!/bin/bash -l
#SBATCH -p <your-partition>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/analysis_%j.out

conda activate <your-env>

WORK_DIR=<path-to-this-repo>
DATA_PATH=<path-to-shd-data>
SEED=42

cd "${WORK_DIR}"
mkdir -p figures slurm_logs

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
```

Figures are saved to `./figures/`. The analysis produces:
- `cochlear_spectrum.png` — FFT over the 700 cochlear channels at each stage
- `cochlear_hf_ratio.png` — high-frequency energy ratio across stages
- `temporal_spectrum.png` — FFT over the T=16 simulation timesteps at each stage
- `temporal_hf_ratio.png` — temporal high-frequency energy ratio across stages
