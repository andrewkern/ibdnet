"""Genetic map IO utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(slots=True)
class GeneticMap:
    chrom: str
    positions_bp: np.ndarray
    positions_cm: np.ndarray

    def interpolate(self, bp: np.ndarray) -> np.ndarray:
        """Interpolate cM values for integer base-pair coordinates."""

        return np.interp(bp, self.positions_bp, self.positions_cm)


def read_map(path: str | Path) -> GeneticMap:
    """Read a PLINK-style genetic map file (chrom, pos_bp, cM)."""

    df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "pos_bp", "cm"])
    if df.empty:
        raise ValueError(f"Map file '{path}' is empty")
    chrom = str(df.iloc[0, 0])
    return GeneticMap(
        chrom=chrom,
        positions_bp=df["pos_bp"].to_numpy(dtype=float),
        positions_cm=df["cm"].to_numpy(dtype=float),
    )


__all__ = ["GeneticMap", "read_map"]
