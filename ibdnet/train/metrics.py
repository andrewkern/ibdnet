"""Training and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def segment_f1(pred_bed: pd.DataFrame, true_bed: pd.DataFrame, min_len_cm: float) -> float:
    """Compute segment-level F1 using segment identifiers."""

    pred = pred_bed[pred_bed["len_cm"] >= min_len_cm]
    truth = true_bed[true_bed["len_cm"] >= min_len_cm]

    pred_ids = set(pred["name"].astype(str))
    true_ids = set(truth["name"].astype(str))

    if not pred_ids and not true_ids:
        return 1.0

    intersection = pred_ids & true_ids
    if not intersection:
        return 0.0

    precision = len(intersection) / max(len(pred_ids), 1)
    recall = len(intersection) / max(len(true_ids), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calibration_ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error for binary probabilities."""

    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have identical shape")

    probs = probs.astype(np.float64).ravel()
    labels = labels.astype(np.float64).ravel()

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        bin_prob = probs[mask].mean()
        bin_acc = labels[mask].mean()
        weight = mask.mean()
        total += weight * abs(bin_acc - bin_prob)
    return float(total)


__all__ = ["segment_f1", "calibration_ece"]
