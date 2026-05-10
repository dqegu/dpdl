"""
SHD MaxFormer models — three experimental conditions.

All three share the same 2-stage architecture matching the neuromorphic config
from Table 1 of the paper (Neuromorphic: 2 stages, 1/1 blocks).

Input shape: (B, T, C, L) from the dataloader, transposed to (T, B, C, L) internally.
    T = simulation timesteps (16 by default, matching cifar10dvs.yaml)
    B = batch size
    C = input channels (1 for SHD — each cochlear channel is a binary event count)
    L = 700 cochlear frequency channels

Conditions:
    SHDMaxFormerAvg  — Embed-Orig + AvgPool1d token mixing (low-pass baseline)
    SHDMaxFormerMax  — Embed-Max + MaxPool1d token mixing (high-frequency)
    SHDMaxFormer     — Full MaxFormer: Embed-Max+/Embed-Max + DWC+SSA (paper's design)

ANN baselines (ReLU replacing LIF) are handled by a flag, not a separate class,
to keep the architectural comparison clean.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from embedding_hub_1d import EmbedOrig1d, EmbedMax1d, EmbedAvg1d, EmbedMaxPlus1d
from mixer_hub_1d import (Block_Avg1d, Block_Max1d, Block_DWC1d,
                           Block_SSA1d, Block_Identity1d)

__all__ = ['shd_snn_avg', 'shd_snn_max', 'shd_max_former', 'shd_ann_avg', 'shd_ann_max']


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class SHDBase(nn.Module):
    """
    2-stage SHD transformer base. Subclasses set patch_embed1/2 and stage1/2.
    Following the neuromorphic config exactly:
        Stage 1: Embed-Max+ / DWC-3 token mixing
        Stage 2: Embed-Max  / SSA token mixing
    """
    def __init__(self, in_channels=1, num_classes=20,
                 embed_dims=256, mlp_ratio=1.0, T=16, **kwargs):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        # subclasses assign these
        self.patch_embed1 = None
        self.patch_embed2 = None
        self.stage1 = None
        self.stage2 = None

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.head = nn.Linear(embed_dims, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d,)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x: (T, B, C, L)
        x = self.patch_embed1(x)                    # (T, B, C', L')
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)                    # (T, B, C'', L'')
        for blk in self.stage2:
            x = blk(x)

        return x.mean(-1)                           # (T, B, C'') — pool over L

    def forward(self, x):
        # x from dataloader: (B, T, C, L)
        if x.dim() == 4:
            x = x.permute(1, 0, 2, 3).contiguous() # (T, B, C, L)
        elif x.dim() == 3:
            # (B, T, L) — no channel dim; add it
            x = x.permute(1, 0, 2).unsqueeze(2).contiguous()  # (T, B, 1, L)

        x = self.forward_features(x)               # (T, B, C)
        x = self.head_lif(x)
        x = self.head(x.mean(0))                   # average over T then classify
        return x


# ---------------------------------------------------------------------------
# Condition A-low: SNN with AvgPool1d token mixing (low-pass baseline)
# Mirrors the Avg-Pool Spiking Transformer in Figure 1 of the paper
# ---------------------------------------------------------------------------

class SHDMaxFormerAvg(SHDBase):
    def __init__(self, in_channels=1, num_classes=20,
                 embed_dims=256, mlp_ratio=1.0, T=16, **kwargs):
        super().__init__(in_channels, num_classes, embed_dims, mlp_ratio, T)

        # Stage 1: plain embedding, AvgPool token mixing
        self.patch_embed1 = EmbedOrig1d(in_channels, embed_dims // 2)
        self.stage1 = nn.ModuleList([
            Block_Avg1d(dim=embed_dims // 2, mlp_ratio=mlp_ratio)
        ])

        # Stage 2: plain embedding, AvgPool token mixing
        self.patch_embed2 = EmbedOrig1d(embed_dims // 2, embed_dims)
        self.stage2 = nn.ModuleList([
            Block_Avg1d(dim=embed_dims, mlp_ratio=mlp_ratio)
        ])


# ---------------------------------------------------------------------------
# Condition A-high: SNN with MaxPool1d token mixing (high-frequency)
# Mirrors the Max-Pool Spiking Transformer in Figure 1 of the paper
# ---------------------------------------------------------------------------

class SHDMaxFormerMax(SHDBase):
    def __init__(self, in_channels=1, num_classes=20,
                 embed_dims=256, mlp_ratio=1.0, T=16, **kwargs):
        super().__init__(in_channels, num_classes, embed_dims, mlp_ratio, T)

        # Stage 1: plain embedding, MaxPool token mixing
        self.patch_embed1 = EmbedOrig1d(in_channels, embed_dims // 2)
        self.stage1 = nn.ModuleList([
            Block_Max1d(dim=embed_dims // 2, mlp_ratio=mlp_ratio)
        ])

        # Stage 2: plain embedding, MaxPool token mixing
        self.patch_embed2 = EmbedOrig1d(embed_dims // 2, embed_dims)
        self.stage2 = nn.ModuleList([
            Block_Max1d(dim=embed_dims, mlp_ratio=mlp_ratio)
        ])


# ---------------------------------------------------------------------------
# Full MaxFormer for SHD: Embed-Max+ / Embed-Max + DWC + SSA
# Mirrors the neuromorphic config in Table 1 exactly
# ---------------------------------------------------------------------------

class SHDMaxFormer(SHDBase):
    def __init__(self, in_channels=1, num_classes=20,
                 embed_dims=256, mlp_ratio=1.0, T=16, **kwargs):
        super().__init__(in_channels, num_classes, embed_dims, mlp_ratio, T)

        # Stage 1: Embed-Max+ embedding, DWC-3 token mixing
        self.patch_embed1 = EmbedMaxPlus1d(in_channels, embed_dims // 2)
        self.stage1 = nn.ModuleList([
            Block_DWC1d(dim=embed_dims // 2, kernel_size=3, mlp_ratio=mlp_ratio)
        ])

        # Stage 2: Embed-Max embedding, SSA token mixing
        self.patch_embed2 = EmbedMax1d(embed_dims // 2, embed_dims)
        self.stage2 = nn.ModuleList([
            Block_SSA1d(dim=embed_dims, num_heads=16, mlp_ratio=mlp_ratio)
        ])


# ---------------------------------------------------------------------------
# ANN baselines — same architecture but ReLU replaces all LIF neurons
# Used to test whether the high-freq benefit is SNN-specific (per the rebuttal)
# ---------------------------------------------------------------------------

class ReLUWrapper(nn.Module):
    """Drop-in replacement for MultiStepLIFNode that applies ReLU."""
    def forward(self, x):
        return F.relu(x)


def _replace_lif_with_relu(module):
    """Recursively swap all MultiStepLIFNode with ReLUWrapper."""
    for name, child in module.named_children():
        if isinstance(child, MultiStepLIFNode):
            setattr(module, name, ReLUWrapper())
        else:
            _replace_lif_with_relu(child)
    return module


class SHDANNAvg(SHDMaxFormerAvg):
    """ANN version of the low-pass baseline."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _replace_lif_with_relu(self)


class SHDANNMax(SHDMaxFormerMax):
    """ANN version of the high-frequency baseline."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _replace_lif_with_relu(self)


# ---------------------------------------------------------------------------
# timm model registration
# ---------------------------------------------------------------------------

@register_model
def shd_snn_avg(pretrained=False, pretrained_cfg=None, **kwargs):
    model = SHDMaxFormerAvg(**kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def shd_snn_max(pretrained=False, pretrained_cfg=None, **kwargs):
    model = SHDMaxFormerMax(**kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def shd_max_former(pretrained=False, pretrained_cfg=None, **kwargs):
    model = SHDMaxFormer(**kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def shd_ann_avg(pretrained=False, pretrained_cfg=None, **kwargs):
    model = SHDANNAvg(**kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def shd_ann_max(pretrained=False, pretrained_cfg=None, **kwargs):
    model = SHDANNMax(**kwargs)
    model.default_cfg = _cfg()
    return model


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    T, B, L = 16, 4, 700
    x = torch.randn(B, T, 1, L).cuda()

    for name, cls in [('SNN-Avg', SHDMaxFormerAvg),
                       ('SNN-Max', SHDMaxFormerMax),
                       ('MaxFormer', SHDMaxFormer),
                       ('ANN-Avg', SHDANNAvg),
                       ('ANN-Max', SHDANNMax)]:
        model = cls(in_channels=1, num_classes=20,
                    embed_dims=256, mlp_ratio=1.0, T=T).cuda()
        out = model(x)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{name:12s}  params={n:,}  out={out.shape}')