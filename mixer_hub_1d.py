"""
1D token mixing modules for SHD (Spiking Heidelberg Digits).

These are direct analogues of event/mixer_hub.py, adapted for (T, B, C, L) inputs.
Conv2d -> Conv1d, MaxPool2d -> MaxPool1d, and SSA operates over N=L tokens.

Three mixer families are provided:
    - AvgMixer1d / Block_Avg1d   : low-pass baseline (AvgPool1d token mixing)
    - MaxMixer1d / Block_Max1d   : high-frequency baseline (MaxPool1d token mixing)
    - DWC1d / Block_DWC1d        : high-frequency via depthwise Conv1d (main MaxFormer mixer)
    - SSA1d / Block_SSA1d        : Spiking Self-Attention over L tokens
    - S_MLP1d                    : spiking MLP block (channel mixer)
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


# ---------------------------------------------------------------------------
# Spiking MLP (channel mixer) — identical logic to event/mixer_hub.py S_MLP
# but operating over (T, B, C, L)
# ---------------------------------------------------------------------------

class S_MLP1d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.fc1_bn   = nn.BatchNorm1d(hidden_features)
        self.fc1_lif  = MultiStepLIFNode(detach_reset=True, backend='torch')

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1)
        self.fc2_bn   = nn.BatchNorm1d(out_features)
        self.fc2_lif  = MultiStepLIFNode(detach_reset=True, backend='torch')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        # x: (T, B, C, L)
        T, B, C, L = x.shape
        identity = x

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))          # (T*B, hidden, L)
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, L)
        if self.res:
            x = identity + x
            identity = x

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, L)
        x = x + identity
        return x


# ---------------------------------------------------------------------------
# AvgPool1d token mixer — low-pass baseline
# Mirrors Block_Avg in event/mixer_hub.py
# ---------------------------------------------------------------------------

class AvgMixer1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1,
                                  count_include_pad=False)

    def forward(self, x):
        T, B, C, L = x.shape
        x = self.pool(x.flatten(0, 1)).reshape(T, B, C, L)
        return x


class Block_Avg1d(nn.Module):
    def __init__(self, dim, mlp_ratio=1.0):
        super().__init__()
        self.mixer = AvgMixer1d(dim)
        self.mlp   = S_MLP1d(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = self.mixer(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# MaxPool1d token mixer — high-frequency baseline
# Mirrors Block_Max in event/mixer_hub.py
# This is the key comparison: AvgPool vs MaxPool on SHD
# ---------------------------------------------------------------------------

class MaxMixer1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        T, B, C, L = x.shape
        x = self.pool(x.flatten(0, 1)).reshape(T, B, C, L)
        return x


class Block_Max1d(nn.Module):
    def __init__(self, dim, mlp_ratio=1.0):
        super().__init__()
        self.mixer = MaxMixer1d(dim)
        self.mlp   = S_MLP1d(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = self.mixer(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Identity mixer — no token mixing (used in stage 1 of CIFAR config)
# ---------------------------------------------------------------------------

class IdentityMixer1d(nn.Module):
    def forward(self, x):
        return x


class Block_Identity1d(nn.Module):
    def __init__(self, dim, mlp_ratio=1.0):
        super().__init__()
        self.mixer = IdentityMixer1d()
        self.mlp   = S_MLP1d(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = self.mixer(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Depthwise Conv1d token mixer — high-frequency via local convolution
# Mirrors Mixer_DWC3/5/7 in event/mixer_hub.py
# ---------------------------------------------------------------------------

class DWCMixer1d(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv        = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                     padding=kernel_size // 2, groups=dim)
        self.conv_bn     = nn.BatchNorm1d(dim)
        self.conv_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, C, L = x.shape
        identity = x
        x = self.conv_neuron(x).reshape(T * B, C, L)
        x = self.conv(x)
        x = self.conv_bn(x).reshape(T, B, C, L)
        x = x + identity
        return x


class Block_DWC1d(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=1.0):
        super().__init__()
        self.mixer = DWCMixer1d(dim, kernel_size)
        self.mlp   = S_MLP1d(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = self.mixer(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Spiking Self-Attention (SSA) over L tokens
# Mirrors SSA in event/mixer_hub.py — operates over N=L positions
# ---------------------------------------------------------------------------

class SSA1d(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        self.scale     = 0.125

        self.x_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.q_bn   = nn.BatchNorm1d(dim)
        self.q_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.k_bn   = nn.BatchNorm1d(dim)
        self.k_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.v_bn   = nn.BatchNorm1d(dim)
        self.v_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5,
                                          detach_reset=True, backend='torch')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.proj_bn   = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x: (T, B, C, L)
        T, B, C, L = x.shape
        identity = x

        x = self.x_lif(x)                          # (T, B, C, L)
        x_flat = x.flatten(0, 1)                    # (T*B, C, L)

        q = self.q_lif(self.q_bn(self.q_conv(x_flat)).reshape(T, B, C, L))
        k = self.k_lif(self.k_bn(self.k_conv(x_flat)).reshape(T, B, C, L))
        v = self.v_lif(self.v_bn(self.v_conv(x_flat)).reshape(T, B, C, L))

        # reshape to multi-head: (T, B, heads, head_dim, L)
        q = q.reshape(T, B, self.num_heads, C // self.num_heads, L)
        k = k.reshape(T, B, self.num_heads, C // self.num_heads, L)
        v = v.reshape(T, B, self.num_heads, C // self.num_heads, L)

        # attention: k^T @ v then q @ (k^T v)
        x = k.transpose(-2, -1) @ v                # (T, B, heads, L, L)
        x = (q @ x) * self.scale                   # (T, B, heads, head_dim, L)

        x = x.reshape(T, B, C, L)
        x = self.attn_lif(x)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, L)
        x = x + identity
        return x


class Block_SSA1d(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=1.0):
        super().__init__()
        self.attn = SSA1d(dim, num_heads)
        self.mlp  = S_MLP1d(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x