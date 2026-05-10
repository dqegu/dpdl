"""
1D embedding modules for SHD (Spiking Heidelberg Digits).

These are direct analogues of the 2D embedding modules in event/embedding_hub.py,
with Conv2d -> Conv1d and MaxPool2d -> MaxPool1d. The input shape throughout is
(T, B, C, L) where:
    T = simulation timesteps
    B = batch size
    C = channel dimension
    L = cochlear/frequency channel dimension (700 for SHD)
"""

import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


class Embed1d(nn.Module):
    """
    Basic embedding block: LIF -> Conv1d -> BN.
    Analogue of event/embedding_hub.py Embed.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, shortcut=False):
        super().__init__()
        self.embed_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.embed_conv = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm1d(out_channels)
        self.shortcut = shortcut

    def forward(self, x):
        # x: (T, B, C, L)
        T, B, C, L = x.shape
        if not self.shortcut:
            x = self.embed_lif(x)
        x = self.embed_conv(x.flatten(0, 1))   # (T*B, C_out, L')
        x = self.embed_bn(x)
        return x


class MaxEmbed1d(nn.Module):
    """
    High-frequency embedding: LIF -> Conv1d -> BN -> MaxPool1d.
    Analogue of event/embedding_hub.py Max_Embed.
    This is the key high-frequency-preserving block.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, shortcut=False):
        super().__init__()
        self.embed_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.embed_conv = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm1d(out_channels)
        # MaxPool1d preserves high-frequency detail (local maxima) along L
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        # x: (T, B, C, L)
        T, B, C, L = x.shape
        if not self.shortcut:
            x = self.embed_lif(x)
        x_feat = x                              # save pre-pool for shortcut
        x = self.embed_conv(x.flatten(0, 1))   # (T*B, C_out, L)
        x = self.embed_bn(x)
        x = self.maxpool(x)                    # (T*B, C_out, L//2)
        return x, x_feat


class AvgEmbed1d(nn.Module):
    """
    Low-frequency (control) embedding: LIF -> Conv1d -> BN -> AvgPool1d.
    Analogue of event/embedding_hub.py Avg_Embed.
    Used as the low-pass baseline to compare against MaxEmbed1d.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, shortcut=False):
        super().__init__()
        self.embed_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.embed_conv = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm1d(out_channels)
        # AvgPool1d smooths over L, reinforcing the LIF low-pass bias
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        T, B, C, L = x.shape
        if not self.shortcut:
            x = self.embed_lif(x)
        x_feat = x
        x = self.embed_conv(x.flatten(0, 1))
        x = self.embed_bn(x)
        x = self.avgpool(x)
        return x, x_feat


# ---------------------------------------------------------------------------
# Compound patch embedding modules (matching the naming in embedding_hub.py)
# ---------------------------------------------------------------------------

class EmbedOrig1d(nn.Module):
    """
    Original (low-pass) patch embedding: both branches use plain Embed1d.
    Analogue of Embed_Orig. Used as the baseline condition.
    Input: (T, B, C_in, L)  ->  Output: (T, B, C_out, L)
    """
    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.proj_conv = nn.Conv1d(in_channels, embed_dims // 2,
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims // 2)
        self.embed1 = Embed1d(embed_dims // 2, embed_dims,
                              kernel_size=3, stride=1, padding=1)
        self.embed2 = Embed1d(embed_dims // 2, embed_dims,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        T, B, C, L = x.shape
        x = self.proj_conv(x.flatten(0, 1))        # (T*B, embed//2, L)
        x = self.proj_bn(x).reshape(T, B, -1, L)
        x_feat = x

        x = self.embed1(x)                          # (T*B, embed, L)
        x_feat = self.embed2(x_feat)                # shortcut
        x = (x + x_feat).reshape(T, B, -1, L)
        return x


class EmbedMax1d(nn.Module):
    """
    High-frequency patch embedding: main branch uses MaxEmbed1d.
    Analogue of Embed_Max. One of the key conditions being tested.
    Input: (T, B, C_in, L)  ->  Output: (T, B, C_out, L//2)
    """
    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.max_embed1 = MaxEmbed1d(in_channels, embed_dims,
                                     kernel_size=3, stride=1, padding=1)
        self.embed1 = Embed1d(embed_dims, embed_dims,
                              kernel_size=3, stride=1, padding=1)
        self.max_embed2 = MaxEmbed1d(in_channels, embed_dims,
                                     kernel_size=1, stride=1, padding=0,
                                     shortcut=True)

    def forward(self, x):
        T, B, C, L = x.shape
        x, x_feat = self.max_embed1(x)             # (T*B, embed, L//2)
        x = x.reshape(T, B, -1, L // 2)
        x = self.embed1(x)                          # (T*B, embed, L//2)
        x_feat, _ = self.max_embed2(x_feat)        # shortcut: (T*B, embed, L//2)
        x = (x + x_feat).reshape(T, B, -1, L // 2)
        return x


class EmbedAvg1d(nn.Module):
    """
    Low-frequency (control) patch embedding: main branch uses AvgEmbed1d.
    Analogue of a hypothetical Embed_Avg. Paired with EmbedMax1d for ablation.
    Input: (T, B, C_in, L)  ->  Output: (T, B, C_out, L//2)
    """
    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.avg_embed1 = AvgEmbed1d(in_channels, embed_dims,
                                     kernel_size=3, stride=1, padding=1)
        self.embed1 = Embed1d(embed_dims, embed_dims,
                              kernel_size=3, stride=1, padding=1)
        self.avg_embed2 = AvgEmbed1d(in_channels, embed_dims,
                                     kernel_size=1, stride=1, padding=0,
                                     shortcut=True)

    def forward(self, x):
        T, B, C, L = x.shape
        x, x_feat = self.avg_embed1(x)
        x = x.reshape(T, B, -1, L // 2)
        x = self.embed1(x)
        x_feat, _ = self.avg_embed2(x_feat)
        x = (x + x_feat).reshape(T, B, -1, L // 2)
        return x


class EmbedMaxPlus1d(nn.Module):
    """
    Deeper high-frequency embedding for longer inputs: three successive MaxEmbed1d
    stages, reducing L by 8x total. Analogue of Embed_Max_plus (designed for
    128x128 neuromorphic inputs). Use when L=700 needs significant downsampling.
    Input: (T, B, C_in, L)  ->  Output: (T, B, embed_dims, L//8)
    """
    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.proj_conv = nn.Conv1d(in_channels, embed_dims // 8,
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims // 8)

        self.max_embed1 = MaxEmbed1d(embed_dims // 8, embed_dims // 4,
                                     kernel_size=3, stride=1, padding=1)
        self.max_embed2 = MaxEmbed1d(embed_dims // 4, embed_dims // 2,
                                     kernel_size=3, stride=1, padding=1)
        self.max_embed3 = MaxEmbed1d(embed_dims // 2, embed_dims,
                                     kernel_size=3, stride=1, padding=1)
        self.embed1 = Embed1d(embed_dims // 4, embed_dims,
                              kernel_size=1, stride=4, padding=0, shortcut=True)

    def forward(self, x):
        T, B, C, L = x.shape
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, L)

        x, _ = self.max_embed1(x)
        x = x.reshape(T, B, -1, L // 2)

        x, x_feat = self.max_embed2(x)
        x = x.reshape(T, B, -1, L // 4)

        x, _ = self.max_embed3(x)                  # (T*B, embed, L//8)

        x_feat = self.embed1(x_feat)               # shortcut: (T*B, embed, L//8)
        x = (x + x_feat).reshape(T, B, -1, L // 8)
        return x