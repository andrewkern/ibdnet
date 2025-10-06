"""Two-state CRF for IBD detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from ..config import TransitionConfig
from .transitions import compute_log_transitions


@dataclass
class CRFOutput:
    log_posterior: Tensor
    posterior: Tensor
    log_likelihood: Tensor


class IBDCRF(nn.Module):
    """Neural CRF over binary IBD states."""

    def __init__(self, emissions: nn.Module, transition_cfg: TransitionConfig) -> None:
        super().__init__()
        self.emissions = emissions
        self.transition_cfg = transition_cfg

    def forward(
        self,
        feats: Tensor,
        dcm: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if mask is None:
            mask = torch.ones(feats.shape[:2], dtype=torch.bool, device=feats.device)

        try:
            node_potentials = self.emissions(feats, mask=mask)
        except TypeError:
            node_potentials = self.emissions(feats)

        batch, seq_len, states = node_potentials.shape
        if states != 2:
            raise ValueError("Emission models must output two-state potentials")

        mask = mask.to(torch.bool)
        lengths = mask.sum(dim=1).to(torch.long)

        if dcm.dim() == 1:
            dcm = dcm.unsqueeze(0).expand(batch, -1)
        step_dcm = dcm[..., 1:seq_len]
        log_transitions = compute_log_transitions(step_dcm, self.transition_cfg)

        neg_inf = torch.finfo(node_potentials.dtype).min
        neg_inf_tensor = torch.full((), neg_inf, device=feats.device, dtype=feats.dtype)

        log_alpha = torch.full_like(node_potentials, neg_inf)
        log_alpha[:, 0] = torch.where(mask[:, 0].unsqueeze(-1), node_potentials[:, 0], neg_inf_tensor)

        for t in range(1, seq_len):
            prev = log_alpha[:, t - 1].unsqueeze(-1) + log_transitions[:, t - 1]
            log_alpha[:, t] = torch.logsumexp(prev, dim=1) + node_potentials[:, t]
            log_alpha[:, t] = torch.where(mask[:, t].unsqueeze(-1), log_alpha[:, t], neg_inf_tensor)

        log_beta = torch.full_like(node_potentials, neg_inf)
        for b in range(batch):
            L = int(lengths[b].item())
            if L == 0:
                continue
            log_beta[b, L - 1] = 0.0
            for t in range(L - 2, -1, -1):
                trans = log_transitions[b, t]
                next_term = log_beta[b, t + 1] + node_potentials[b, t + 1]
                log_beta[b, t] = torch.logsumexp(trans + next_term.unsqueeze(0), dim=1)

        log_likelihood = []
        for b in range(batch):
            L = int(lengths[b].item())
            if L == 0:
                log_likelihood.append(neg_inf_tensor)
            else:
                log_likelihood.append(torch.logsumexp(log_alpha[b, L - 1], dim=-1))
        log_likelihood_tensor = torch.stack(log_likelihood)

        log_post = log_alpha + log_beta - log_likelihood_tensor.view(batch, 1, 1)
        log_post = torch.where(mask.unsqueeze(-1), log_post, neg_inf_tensor)

        posterior = torch.exp(log_post)

        return {
            "log_posterior": log_post,
            "posterior": posterior,
            "log_likelihood": log_likelihood_tensor,
        }


__all__ = ["IBDCRF", "CRFOutput"]
