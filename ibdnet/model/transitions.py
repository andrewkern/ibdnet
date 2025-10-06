"""Transition potential helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import TransitionConfig


def compute_log_transitions(dcm: Tensor, cfg: TransitionConfig) -> Tensor:
    """Compute per-step log transition matrices."""

    if dcm.ndim == 1:
        dcm = dcm.unsqueeze(0)

    hazard_on = torch.clamp(1.0 - torch.exp(-cfg.base_lambda * dcm), cfg.min_non2ibd, 1 - 1e-6)
    hazard_off = torch.clamp(
        1.0 - torch.exp(-cfg.base_lambda * dcm * 1.5), cfg.min_non2ibd, cfg.max_ibd2non
    )

    stay_off = torch.log1p(-hazard_on)
    go_on = torch.log(hazard_on)
    go_off = torch.log(hazard_off)
    stay_on = torch.log1p(-hazard_off)

    mat = torch.zeros((*dcm.shape, 2, 2), dtype=dcm.dtype, device=dcm.device)
    mat[..., 0, 0] = stay_off
    mat[..., 0, 1] = go_on
    mat[..., 1, 0] = go_off
    mat[..., 1, 1] = stay_on
    return mat


__all__ = ["compute_log_transitions"]
