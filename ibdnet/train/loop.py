"""Training loop utilities for IBDNet."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from ..config import EmissionConfig, ExperimentConfig, TransitionConfig
from ..model.crf import IBDCRF
from ..model.emissions import MLPEmissions, TCNEmissions, TinyTransformerEmissions


def build_emitter(cfg: EmissionConfig) -> nn.Module:
    if cfg.model == "mlp":
        return MLPEmissions(cfg.in_dim, cfg.hidden, cfg.layers, cfg.dropout)
    if cfg.model == "tcn":
        return TCNEmissions(cfg.in_dim, cfg.hidden, cfg.layers)
    if cfg.model == "tiny_transformer":
        return TinyTransformerEmissions(cfg.in_dim, cfg.hidden, cfg.layers, cfg.dropout)
    raise ValueError(f"Unknown emission model '{cfg.model}'")


def build_model(cfg: ExperimentConfig) -> IBDCRF:
    emitter = build_emitter(cfg.emissions)
    return IBDCRF(emitter, cfg.transitions)


def _compute_loss(log_post: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = mask.view(-1)
    logp = log_post.view(-1, log_post.shape[-1])[masked]
    y = labels.view(-1)[masked]
    return F.nll_loss(logp, y)


def train_epoch(
    model: IBDCRF,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        features = batch["features"].to(device)
        dcm = batch["dcm"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(features, dcm, mask)
        loss = _compute_loss(outputs["log_posterior"], labels, mask)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(dataloader), 1)


def evaluate(
    model: IBDCRF,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            dcm = batch["dcm"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(features, dcm, mask)
            loss = _compute_loss(outputs["log_posterior"], labels, mask)
            total_loss += float(loss.item())
    return total_loss / max(len(dataloader), 1)


__all__ = ["build_model", "train_epoch", "evaluate"]
