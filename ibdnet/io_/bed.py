"""BED file helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


BED_COLUMNS = ["chrom", "start", "end", "name", "mean_post", "min_post", "len_bp", "len_cm"]


def write_bed(path: str | Path, df: pd.DataFrame) -> None:
    """Write a BED-like table with enforced columns."""

    missing = [col for col in BED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    df = df[BED_COLUMNS]
    df.to_csv(path, sep="\t", header=False, index=False)


__all__ = ["BED_COLUMNS", "write_bed"]
