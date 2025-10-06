"""Feature construction over PBWT permutations."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from ..config import PBWTConfig
from .pbwt_core import build_pbwt


def _compute_ranks(prefix_arrays: Iterable[np.ndarray]) -> np.ndarray:
    orders = list(prefix_arrays)
    n_sites = len(orders)
    if n_sites == 0:
        return np.empty((0, 0), dtype=np.int32)
    n_haps = int(orders[0].shape[0])
    ranks = np.zeros((n_sites, n_haps), dtype=np.int32)
    for site, order in enumerate(orders):
        ranks[site, order] = np.arange(n_haps, dtype=np.int32)
    return ranks


def make_pair_features(
    haps: np.ndarray,
    cm_map: np.ndarray,
    pairs: np.ndarray,
    cfg: PBWTConfig,
) -> Dict[str, np.ndarray]:
    """Compute simple PBWT-inspired features for haplotype pairs.

    Parameters
    ----------
    haps:
        Array shaped (sites, haplotypes) with 0/1 alleles.
    cm_map:
        Genetic map positions in centiMorgans (shape ``(sites,)``).
    pairs:
        Array shaped (pairs, 2) listing hap indices.
    cfg:
        PBWT configuration controlling neighbour truncation.

    Returns
    -------
    dict
        Dictionary of feature name → array shaped (pairs, sites).
    """

    if haps.ndim != 2:
        raise ValueError("haps must be 2D (sites, haplotypes)")
    if cm_map.shape[0] != haps.shape[0]:
        raise ValueError("cm_map length must equal number of sites")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be of shape (n_pairs, 2)")

    n_sites, n_haps = haps.shape
    n_pairs = pairs.shape[0]

    prefix_arrays, _ = build_pbwt(haps)
    ranks = _compute_ranks(prefix_arrays)

    ibs = np.zeros((n_pairs, n_sites), dtype=np.float32)
    rank_dist = np.zeros_like(ibs)
    mismatch_run = np.zeros_like(ibs)

    cm_delta = np.gradient(cm_map.astype(np.float32))
    cm_delta[0] = 0.0

    for i, (a, b) in enumerate(pairs):
        if a >= n_haps or b >= n_haps:
            raise IndexError("pair indices exceed haplotype count")
        hap_a = haps[:, a]
        hap_b = haps[:, b]
        matches = hap_a == hap_b
        ibs[i] = matches.astype(np.float32)
        rank_dist[i] = np.minimum(
            np.abs(ranks[:, a] - ranks[:, b]).astype(np.float32), cfg.max_rank_dist
        )
        mism = (~matches).astype(np.int32)
        mismatch_run[i] = np.cumsum(mism).astype(np.float32)

    features = {
        "ibs": ibs,
        "rank_dist": rank_dist,
        "mismatch_run": mismatch_run,
        "cm_delta": np.broadcast_to(cm_delta, ibs.shape).astype(np.float32),
    }

    if cfg.normalize == "global":
        for key, value in features.items():
            mean = value.mean()
            std = value.std() or 1.0
            features[key] = (value - mean) / std

    return features


__all__ = ["make_pair_features"]
