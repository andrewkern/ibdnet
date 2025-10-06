"""Small helpers for contiguous numpy arrays used within PBWT sweeps."""

from __future__ import annotations

import numpy as np


def ensure_int8(array: np.ndarray) -> np.ndarray:
    """Ensure that the haplotype array is ``int8`` and C-contiguous."""

    return np.ascontiguousarray(array, dtype=np.int8)


def ensure_int32(array: np.ndarray) -> np.ndarray:
    """Ensure index arrays are ``int32`` and C-contiguous."""

    return np.ascontiguousarray(array, dtype=np.int32)


__all__ = ["ensure_int8", "ensure_int32"]
