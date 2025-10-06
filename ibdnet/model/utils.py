"""Utility functions for CRF computations."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_logsumexp(tensor: Tensor, dim: int, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return torch.logsumexp(tensor, dim=dim)
    mask = mask.to(dtype=torch.bool)
    neg_inf = torch.finfo(tensor.dtype).min
    masked = torch.where(mask, tensor, neg_inf)
    return torch.logsumexp(masked, dim=dim)


__all__ = ["masked_logsumexp"]
