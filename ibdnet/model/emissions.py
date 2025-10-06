"""Emission models mapping features to log-potentials."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class MLPEmissions(nn.Module):
    """Simple multi-layer perceptron for per-site emissions."""

    def __init__(self, in_dim: int, hidden: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        modules: List[nn.Module] = []
        prev_dim = in_dim
        for _ in range(max(layers - 1, 0)):
            modules.append(nn.Linear(prev_dim, hidden))
            modules.append(nn.GELU())
            modules.append(nn.Dropout(dropout))
            prev_dim = hidden
        modules.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-potentials shaped ``(batch, sites, 2)``."""

        batch, sites, feat_dim = x.shape
        flat = x.view(batch * sites, feat_dim)
        out = self.net(flat)
        return out.view(batch, sites, 2)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[..., :-self.conv.padding[0]] if self.conv.padding[0] > 0 else out
        return self.activation(self.norm(out))


class TCNEmissions(nn.Module):
    """A light-weight temporal convolutional network."""

    def __init__(self, in_dim: int, hidden: int, layers: int = 2):
        super().__init__()
        channels = [in_dim] + [hidden] * layers
        blocks: List[nn.Module] = []
        for i in range(layers):
            blocks.append(TemporalBlock(channels[i], channels[i + 1], kernel_size=3, dilation=2**i))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(channels[-1], 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sites, feat_dim = x.shape
        inp = x.transpose(1, 2)  # (batch, feat, sites)
        out = self.tcn(inp)
        out = self.head(out)
        out = out.transpose(1, 2)
        return out


class TinyTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        lin = self.linear(x)
        return self.norm2(x + lin)


class TinyTransformerEmissions(nn.Module):
    """Tiny Transformer encoder producing per-site logits."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [TinyTransformerLayer(hidden, heads=2, dropout=dropout) for _ in range(layers)]
        )
        self.out_proj = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask
        for layer in self.layers:
            h = layer(h, mask=attn_mask)
        return self.out_proj(h)


__all__ = ["MLPEmissions", "TCNEmissions", "TinyTransformerEmissions"]
