"""Posterior calibration utilities."""

from __future__ import annotations

import numpy as np


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.log(probs.clip(1e-6, 1 - 1e-6) / (1 - probs.clip(1e-6, 1 - 1e-6)))
    scaled = logits / temperature
    denom = 1 + np.exp(-scaled)
    return 1 / denom


__all__ = ["temperature_scale"]
