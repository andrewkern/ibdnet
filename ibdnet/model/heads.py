"""Output heads for aggregating CRF posterior statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PosteriorSummary:
    mean_ibd: Tensor
    min_ibd: Tensor


class PosteriorHead(nn.Module):
    """Compute simple posterior statistics from CRF outputs."""

    def forward(self, posterior: Tensor, mask: Tensor) -> PosteriorSummary:
        mask_f = mask.to(dtype=posterior.dtype)
        masked = posterior * mask_f.unsqueeze(-1)
        total = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        mean_ibd = masked[..., 1].sum(dim=1, keepdim=True) / total
        fill = torch.ones_like(posterior[..., 1])
        valid_values = torch.where(mask, posterior[..., 1], fill)
        min_ibd = valid_values.amin(dim=1, keepdim=True)
        return PosteriorSummary(mean_ibd=mean_ibd.squeeze(1), min_ibd=min_ibd.squeeze(1))


__all__ = ["PosteriorHead", "PosteriorSummary"]
