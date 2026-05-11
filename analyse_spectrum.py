"""
Fourier spectrum analysis for SHD models — two separate analyses.

ANALYSIS A — Cochlear-axis spectral analysis (FFT over L=700 channels)
    Tests whether high-frequency auditory channel structure is attenuated by
    spiking neurons and restored by MaxPool/DWC operators.
    Claim: architectural analogy to MaxFormer's spatial result.
    Figures: cochlear_spectrum.png, cochlear_hf_ratio.png

ANALYSIS B — Temporal-axis spectral analysis (FFT over T timesteps)
    Tests whether high-frequency temporal dynamics are attenuated across layers.
    This directly probes the LIF transfer function H(z) = (1-β)/(1-βz^{-1}),
    which is a discrete-time filter over the simulation timestep axis T.
    Claim: theoretical mechanism — LIF acts as a low-pass filter over time.
    Figures: temporal_spectrum.png, temporal_hf_ratio.png

Usage:
    python analyse_spectrum.py \
        --avg-ckpt     logs/shd_snn_avg_T16_seed42/checkpoint_best.pth \
        --max-ckpt     logs/shd_snn_max_T16_seed42/checkpoint_best.pth \
        --ann-avg-ckpt logs/shd_ann_avg_T16_seed42/checkpoint_best.pth \
        --ann-max-ckpt logs/shd_ann_max_T16_seed42/checkpoint_best.pth \
        --data-path ./data --T 16 --n-batches 20 --out-dir ./figures
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from timm.models import create_model

import models   # registers all shd_* model names
from dataset import get_shd_dataloaders
from spikingjelly.clock_driven import functional


# ---------------------------------------------------------------------------
# Shared style config
# ---------------------------------------------------------------------------

COLOURS = {
    'SNN-Avg': '#e74c3c',
    'SNN-Max': '#2ecc71',
    'ANN-Avg': '#3498db',
    'ANN-Max': '#9b59b6',
}
LINESTYLES = {
    'SNN-Avg': '-', 'SNN-Max': '-',
    'ANN-Avg': '--', 'ANN-Max': '--',
}
LAYER_SHORT  = ['Emb1', 'Stage1', 'Emb2', 'Stage2']
LAYER_LABELS = [
    'After Embed 1\n(Stage 1 input)',
    'After Stage 1\n(Token mix)',
    'After Embed 2\n(Stage 2 input)',
    'After Stage 2\n(Token mix)',
]
LAYER_NAMES = ['patch_embed1', 'stage1.0', 'patch_embed2', 'stage2.0']


# ---------------------------------------------------------------------------
# Hook-based activation recorder
# Stores activations in two shapes simultaneously:
#   cochlear : (T*B, C, L)  — for Analysis A FFT over L
#   temporal : (T, B*C*L)   — for Analysis B FFT over T
# ---------------------------------------------------------------------------

class ActivationRecorder:
    def __init__(self, model, layer_names):
        self.records_cochlear = {n: [] for n in layer_names}
        self.records_temporal = {n: [] for n in layer_names}
        self._handles = []
        for name, module in model.named_modules():
            if name in layer_names:
                h = module.register_forward_hook(self._make_hook(name))
                self._handles.append(h)

    def _make_hook(self, name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.detach().cpu().float()

            if out.dim() == 4:
                # (T, B, C, L) — we have the time axis explicitly
                T, B, C, L = out.shape
                cochlear = out.reshape(T * B, C, L)   # Analysis A
                temporal = out.reshape(T, B * C * L)  # Analysis B
            elif out.dim() == 3:
                # (T*B, C, L) — T/B merged, only cochlear is valid
                cochlear = out
                temporal = None
            else:
                cochlear = out.reshape(out.shape[0], -1, 1)
                temporal = None

            self.records_cochlear[name].append(cochlear)
            if temporal is not None:
                self.records_temporal[name].append(temporal)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


# ---------------------------------------------------------------------------
# Analysis A — cochlear-axis spectrum (FFT over L)
# ---------------------------------------------------------------------------

def compute_cochlear_spectrum(activations):
    """
    activations: list of (T*B, C, L) tensors.
    FFT over the L (cochlear channel) axis — structural analogue of the
    paper's spatial frequency analysis.
    Returns: freqs (L//2+1,), log_mag (L//2+1,)
    """
    cat = torch.cat(activations, dim=0)       # (N, C, L)
    fft = torch.fft.rfft(cat, dim=-1)         # FFT over L
    mag = fft.abs().mean(dim=(0, 1))          # mean over N, C
    log_mag = torch.log(mag + 1e-8)
    freqs = torch.fft.rfftfreq(cat.shape[-1])
    return freqs.numpy(), log_mag.numpy()


# ---------------------------------------------------------------------------
# Analysis B — temporal-axis spectrum (FFT over T)
# Tests H(z) = (1-β)/(1-βz^{-1}) from Equation 8 of the paper.
# High temporal frequency = fast transients across simulation timesteps.
# SNN-Avg should show steeper HF drop than ANN-Avg if the LIF claim holds.
# ---------------------------------------------------------------------------

def compute_temporal_spectrum(activations, T):
    """
    activations: list of (T, N) tensors where N = B*C*L.
    FFT over dim=1 (the T / simulation timestep axis).
    Returns: freqs (T//2+1,), log_mag (T//2+1,)
    """
    cat = torch.stack(activations, dim=0)     # (n_batches, T, N)
    fft = torch.fft.rfft(cat, dim=1)         # FFT over T axis
    mag = fft.abs().mean(dim=(0, 2))         # mean over batches and N
    log_mag = torch.log(mag + 1e-8)
    freqs = torch.fft.rfftfreq(T)
    return freqs.numpy(), log_mag.numpy()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_spectrum_grid(all_spectra, layer_names, title, xlabel, out_path,
                       hf_threshold=0.3):
    n = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, layer, label in zip(axes, layer_names, LAYER_LABELS[:n]):
        for mname, ldict in all_spectra.items():
            if layer not in ldict:
                continue
            freqs, log_mag = ldict[layer]
            log_mag = log_mag - log_mag[0]   # normalise to DC
            ax.plot(freqs, log_mag, label=mname,
                    color=COLOURS.get(mname, 'gray'),
                    linestyle=LINESTYLES.get(mname, '-'),
                    linewidth=1.8)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Relative Log Amplitude', fontsize=9)
        ax.axvline(hf_threshold, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def plot_hf_ratio(all_spectra, layer_names, title, out_path, hf_threshold=0.3):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(layer_names))

    for mname, ldict in all_spectra.items():
        ratios = []
        for layer in layer_names:
            if layer not in ldict:
                ratios.append(np.nan)
                continue
            freqs, log_mag = ldict[layer]
            mag = np.exp(log_mag)
            ratio = mag[freqs >= hf_threshold].sum() / (mag.sum() + 1e-8)
            ratios.append(ratio)
        ax.plot(x, ratios, marker='o', label=mname,
                color=COLOURS.get(mname, 'gray'), linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_SHORT[:len(layer_names)])
    ax.set_ylabel(f'HF Energy Ratio (freq > {hf_threshold})')
    ax.set_xlabel('Processing Stage')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def print_summary(label, all_spectra, final_layer, threshold):
    print(f'\nAnalysis {label} — HF ratio at {final_layer}:')
    ratios = {}
    for mname, ldict in all_spectra.items():
        if final_layer not in ldict:
            continue
        freqs, log_mag = ldict[final_layer]
        mag = np.exp(log_mag)
        r = mag[freqs >= threshold].sum() / (mag.sum() + 1e-8)
        ratios[mname] = r
        print(f'  {mname:<12}: {r:.4f}')
    if 'SNN-Avg' in ratios and 'SNN-Max' in ratios:
        d = ratios['SNN-Max'] - ratios['SNN-Avg']
        print(f'  SNN-Max > SNN-Avg: {d:+.4f} -> '
              f'{"SUPPORTED" if d > 0 else "NOT SUPPORTED"}')
    if 'SNN-Avg' in ratios and 'ANN-Avg' in ratios:
        d = ratios['ANN-Avg'] - ratios['SNN-Avg']
        print(f'  ANN-Avg > SNN-Avg: {d:+.4f} -> SNN-specific attenuation '
              f'{"CONFIRMED" if d > 0 else "NOT CONFIRMED"}')


# ---------------------------------------------------------------------------
# Model loading and activation collection
# ---------------------------------------------------------------------------

def load_model(model_name, ckpt_path, device, T):
    model = create_model(model_name, in_channels=1, num_classes=20,
                         embed_dims=256, mlp_ratio=1.0, T=T)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def collect_activations(model, loader, layer_names, device, n_batches, T):
    recorder = ActivationRecorder(model, layer_names)
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(images.to(device, non_blocking=True).float())
        functional.reset_net(model)
    recorder.remove()
    return recorder.records_cochlear, recorder.records_temporal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--avg-ckpt',     required=True)
    parser.add_argument('--max-ckpt',     required=True)
    parser.add_argument('--ann-avg-ckpt', default='')
    parser.add_argument('--ann-max-ckpt', default='')
    parser.add_argument('--data-path',    default='./data')
    parser.add_argument('--T',            default=16, type=int)
    parser.add_argument('--n-batches',    default=20, type=int)
    parser.add_argument('--batch-size',   default=8,  type=int)
    parser.add_argument('--hf-threshold', default=0.3, type=float)
    parser.add_argument('--out-dir',      default='./figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  T={args.T}  n_batches={args.n_batches}')

    _, test_loader = get_shd_dataloaders(
        args.data_path, T=args.T, batch_size=getattr(args, 'batch_size', 8), num_workers=2)

    to_analyse = {'SNN-Avg': ('shd_snn_avg', args.avg_ckpt),
                  'SNN-Max': ('shd_snn_max', args.max_ckpt)}
    if args.ann_avg_ckpt:
        to_analyse['ANN-Avg'] = ('shd_ann_avg', args.ann_avg_ckpt)
    if args.ann_max_ckpt:
        to_analyse['ANN-Max'] = ('shd_ann_max', args.ann_max_ckpt)

    all_cochlear, all_temporal = {}, {}

    for mname, (timm_name, ckpt_path) in to_analyse.items():
        print(f'\n{"="*50}\nAnalysing: {mname}')
        model = load_model(timm_name, ckpt_path, device, args.T)
        c_recs, t_recs = collect_activations(
            model, test_loader, LAYER_NAMES, device, args.n_batches, args.T)

        # Analysis A
        c_spectra = {}
        print('  [A] Cochlear (FFT over L):')
        for layer, acts in c_recs.items():
            if not acts: continue
            freqs, lmag = compute_cochlear_spectrum(acts)
            c_spectra[layer] = (freqs, lmag)
            mag = np.exp(lmag)
            hf = mag[freqs >= args.hf_threshold].sum() / (mag.sum() + 1e-8)
            print(f'    {layer:<20} L={acts[0].shape[-1]:>3}  HF={hf:.3f}')
        all_cochlear[mname] = c_spectra

        # Analysis B
        t_spectra = {}
        print(f'  [B] Temporal (FFT over T={args.T}):')
        for layer, acts in t_recs.items():
            if not acts: continue
            freqs_t, lmag_t = compute_temporal_spectrum(acts, args.T)
            t_spectra[layer] = (freqs_t, lmag_t)
            mag_t = np.exp(lmag_t)
            hf_t = mag_t[freqs_t >= args.hf_threshold].sum() / (mag_t.sum() + 1e-8)
            print(f'    {layer:<20} T={args.T}  HF={hf_t:.3f}')
        all_temporal[mname] = t_spectra

    # --- Analysis A plots ---
    print('\n--- Analysis A: cochlear-axis plots ---')
    plot_spectrum_grid(
        all_cochlear, LAYER_NAMES,
        title='Analysis A — Cochlear-Axis Fourier Spectrum (FFT over L)\n'
              'Do MaxPool operators preserve high auditory-channel frequency?',
        xlabel='Normalised Cochlear Frequency',
        out_path=os.path.join(args.out_dir, 'cochlear_spectrum.png'),
        hf_threshold=args.hf_threshold)
    plot_hf_ratio(
        all_cochlear, LAYER_NAMES,
        title='Analysis A — Cochlear HF Energy Ratio by Stage',
        out_path=os.path.join(args.out_dir, 'cochlear_hf_ratio.png'),
        hf_threshold=args.hf_threshold)

    # --- Analysis B plots ---
    print('\n--- Analysis B: temporal-axis plots ---')
    plot_spectrum_grid(
        all_temporal, LAYER_NAMES,
        title=f'Analysis B — Temporal-Axis Fourier Spectrum (FFT over T={args.T})\n'
              'LIF H(z)=(1-β)/(1-βz⁻¹): temporal low-pass attenuation across layers?',
        xlabel='Normalised Temporal Frequency',
        out_path=os.path.join(args.out_dir, 'temporal_spectrum.png'),
        hf_threshold=args.hf_threshold)
    plot_hf_ratio(
        all_temporal, LAYER_NAMES,
        title='Analysis B — Temporal HF Energy Ratio by Stage',
        out_path=os.path.join(args.out_dir, 'temporal_hf_ratio.png'),
        hf_threshold=args.hf_threshold)

    # --- summary ---
    print(f'\n{"="*50}\nSUMMARY')
    print_summary('A (cochlear)', all_cochlear, LAYER_NAMES[-1], args.hf_threshold)
    print_summary('B (temporal)', all_temporal, LAYER_NAMES[-1], args.hf_threshold)

    np.save(os.path.join(args.out_dir, 'cochlear_spectra.npy'),
            all_cochlear, allow_pickle=True)
    np.save(os.path.join(args.out_dir, 'temporal_spectra.npy'),
            all_temporal, allow_pickle=True)
    print('\nDone.')


if __name__ == '__main__':
    main()