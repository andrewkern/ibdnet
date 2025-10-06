"""Minimal helpers for interacting with PLINK2 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency during tests
    import pgenlib
except ImportError:  # type: ignore[assignment]
    pgenlib = None


@dataclass(slots=True)
class PlinkDataset:
    """Thin wrapper around PLINK2 PGEN/PVAR/PSAM triples."""

    pgen_path: Path
    sample_path: Path
    variant_path: Path

    def open(self) -> "pgenlib.PgenReader":  # type: ignore[name-defined]
        if pgenlib is None:
            raise ImportError("pgenlib is required to read PGEN files")
        return pgenlib.PgenReader(str(self.pgen_path))


def load_haplotypes(ds: PlinkDataset, sample_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Load haplotypes as a numpy array shaped (variants, samples * 2)."""

    reader = ds.open()
    m = reader.get_variant_ct()
    n = reader.get_raw_sample_ct()
    hap_count = n * 2

    if sample_indices is not None:
        hap_count = len(sample_indices) * 2

    out = np.empty((m, hap_count), dtype=np.int8)

    scratch = np.empty(reader.get_raw_sample_ct(), dtype=np.int8)

    for i in range(m):
        reader.read_alleles(i, scratch)
        if sample_indices is None:
            out[i, 0::2] = scratch
            out[i, 1::2] = scratch
        else:
            for j, idx in enumerate(sample_indices):
                out[i, 2 * j] = scratch[idx]
                out[i, 2 * j + 1] = scratch[idx]
    return out


__all__ = ["PlinkDataset", "load_haplotypes"]
