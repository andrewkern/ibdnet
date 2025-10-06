"""Label generation for simulated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class LabelConfig:
    min_len_cm: float = 2.0


def random_labels(n_pairs: int, n_sites: int, cfg: LabelConfig) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    entries: List[dict[str, object]] = []
    for pair in range(n_pairs):
        start = rng.integers(0, max(n_sites - 10, 1))
        end = min(n_sites - 1, start + rng.integers(5, 15))
        entries.append(
            {
                "chrom": "chrUNK",
                "start": int(start),
                "end": int(end),
                "name": f"pair{pair}",
                "mean_post": rng.random(),
                "min_post": rng.random(),
                "len_bp": int(end - start),
                "len_cm": float(cfg.min_len_cm + rng.random()),
            }
        )
    return pd.DataFrame(entries)


__all__ = ["LabelConfig", "random_labels"]
