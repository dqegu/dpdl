"""
aggregate_results.py — collect best Acc@1 from all training runs and
produce a summary table matching the format of Table 2 in the paper.

Usage:
    python aggregate_results.py --log-dir ./logs --out ./results_summary.csv
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path


CONDITIONS = [
    'shd_snn_avg',
    'shd_snn_max',
    'shd_max_former',
    'shd_ann_avg',
    'shd_ann_max',
]

CONDITION_LABELS = {
    'shd_snn_avg':     'SNN-Avg  (low-pass baseline)',
    'shd_snn_max':     'SNN-Max  (high-freq token mix)',
    'shd_max_former':  'MaxFormer (full, Embed-Max+/DWC/SSA)',
    'shd_ann_avg':     'ANN-Avg  (ReLU, low-pass)',
    'shd_ann_max':     'ANN-Max  (ReLU, high-freq)',
}


def read_best_acc(run_dir):
    """Read the maximum test_acc1 from a results.csv file."""
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return None
    best = 0.0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc = float(row.get('test_acc1', 0))
            if acc > best:
                best = acc
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='./logs')
    parser.add_argument('--out',     default='./results_summary.csv')
    args = parser.parse_args()

    results = {cond: [] for cond in CONDITIONS}

    # scan all run directories
    for run_dir in sorted(Path(args.log_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name   # e.g. shd_snn_avg_T16_seed42
        matched = False
        for cond in CONDITIONS:
            if name.startswith(cond):
                acc = read_best_acc(str(run_dir))
                if acc is not None:
                    results[cond].append(acc)
                    print(f'  {name}: {acc:.2f}%')
                matched = True
                break
        if not matched:
            print(f'  (skipping {name})')

    # print summary table
    print('\n' + '='*65)
    print(f'{"Model":<45}  {"Seeds":>5}  {"Mean":>6}  {"Std":>5}  {"Best":>6}')
    print('='*65)

    summary_rows = []
    for cond in CONDITIONS:
        accs = results[cond]
        if not accs:
            print(f'{CONDITION_LABELS[cond]:<45}  {"N/A":>5}')
            continue
        mean = np.mean(accs)
        std  = np.std(accs)
        best = np.max(accs)
        label = CONDITION_LABELS[cond]
        print(f'{label:<45}  {len(accs):>5}  {mean:>6.2f}  {std:>5.2f}  {best:>6.2f}')
        summary_rows.append({
            'condition': cond,
            'label': label,
            'n_seeds': len(accs),
            'mean_acc1': round(mean, 2),
            'std_acc1':  round(std,  2),
            'best_acc1': round(best, 2),
            'all_accs':  accs,
        })

    print('='*65)

    # save CSV
    with open(args.out, 'w', newline='') as f:
        fieldnames = ['condition', 'label', 'n_seeds',
                      'mean_acc1', 'std_acc1', 'best_acc1', 'all_accs']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f'\nSummary saved to {args.out}')

    # print the key comparison explicitly
    print('\nKey comparison (mirrors paper Figure 1):')
    avg_accs = results.get('shd_snn_avg', [])
    max_accs = results.get('shd_snn_max', [])
    if avg_accs and max_accs:
        delta = np.mean(max_accs) - np.mean(avg_accs)
        print(f'  SNN-Max - SNN-Avg = {delta:+.2f}% (paper reports +2.39% on CIFAR-100)')
        print(f'  -> {"Claim supported" if delta > 0 else "Claim NOT supported"} '
              f'in the temporal/auditory domain.')

    ann_avg = results.get('shd_ann_avg', [])
    ann_max = results.get('shd_ann_max', [])
    if ann_avg and ann_max and avg_accs and max_accs:
        snn_delta = np.mean(max_accs) - np.mean(avg_accs)
        ann_delta = np.mean(ann_max) - np.mean(ann_avg)
        print(f'\n  SNN delta (Max-Avg): {snn_delta:+.2f}%')
        print(f'  ANN delta (Max-Avg): {ann_delta:+.2f}%')
        print(f'  -> SNN benefits {"more" if snn_delta > ann_delta else "less"} '
              f'from max-pooling than ANN.')
        print(f'  -> {"Consistent with" if snn_delta > ann_delta else "Inconsistent with"} '
              f'the paper\'s claim that high-freq restoration is SNN-specific.')


if __name__ == '__main__':
    main()
