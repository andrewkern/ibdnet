"""Minimal PyTorch dataset wrappers for IBD training."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset[Dict[str, torch.Tensor]]):
    """Wrap feature tensors (pairs × sites × feat_dim) for training."""

    def __init__(
        self,
        features: np.ndarray,
        dcm: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if features.ndim != 3:
            raise ValueError("features must be 3D (pairs, sites, feat_dim)")
        self.features = torch.from_numpy(features.astype(np.float32))
        self.dcm = torch.from_numpy(dcm.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.bool_))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "dcm": self.dcm[idx],
            "mask": self.mask[idx],
            "labels": self.labels[idx],
        }


def create_dataloader(dataset: PairDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


__all__ = ["PairDataset", "create_dataloader"]
