"""
SHD (Spiking Heidelberg Digits) dataset loading via tonic.

SHD sensor size: (700, 1, 1) — 700 cochlear frequency channels.
We bin events into T time frames using tonic.transforms.ToFrame,
producing tensors of shape (T, 700) which we reshape to (T, 1, 700)
(adding a channel dim) to match the (T, B, C, L) convention.

The dataloader collects these into (B, T, 1, 700).
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

try:
    import tonic
    import tonic.transforms as tonic_transforms
except ImportError:
    raise ImportError(
        "tonic is required for SHD loading. "
        "Install with: pip install tonic --break-system-packages"
    )


class SHDFrameDataset(torch.utils.data.Dataset):
    """
    Wraps the tonic SHD dataset, binning events into T frames
    and returning tensors of shape (T, 1, 700).
    """
    def __init__(self, data_path, train=True, T=16):
        sensor_size = tonic.datasets.SHD.sensor_size  # (700, 1, 1)

        frame_transform = tonic_transforms.Compose([
            tonic_transforms.ToFrame(
                sensor_size=sensor_size,
                n_time_bins=T
            )
        ])

        self.dataset = tonic.datasets.SHD(
            save_to=data_path,
            train=train,
            transform=frame_transform
        )
        self.T = T

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # frames: (T, 1, 700) — tonic produces (T, C, H, W) but H=W=1 for SHD
        frames, label = self.dataset[idx]

        # frames shape from tonic: (T, 1, 700) already; squeeze spatial dims
        # tonic SHD sensor_size is (700, 1, 1): x_size, y_size, pol_size
        # ToFrame returns (T, pol, y, x) = (T, 1, 1, 700)
        if frames.ndim == 4:
            frames = frames.squeeze(2)      # (T, 1, 700)  drop the y=1 dim
        
        frames = torch.tensor(frames, dtype=torch.float32)
        return frames, label


def get_shd_dataloaders(data_path, T=16, batch_size=16,
                         num_workers=4, seed=42):
    """
    Returns (train_loader, test_loader).
    Mirrors the split_to_train_test_set logic in event/train.py.
    """
    train_dataset = SHDFrameDataset(data_path, train=True,  T=T)
    test_dataset  = SHDFrameDataset(data_path, train=False, T=T)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    return train_loader, test_loader


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else './data'
    train_loader, test_loader = get_shd_dataloaders(path, T=16, batch_size=4)
    x, y = next(iter(train_loader))
    print(f'Batch shape: {x.shape}   Labels: {y}')
    # Expected: torch.Size([4, 16, 1, 700])
