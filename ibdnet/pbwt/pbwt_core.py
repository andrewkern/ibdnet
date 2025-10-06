"""Numba-accelerated PBWT core used for feature construction."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .fastarrays import ensure_int8, ensure_int32

try:  # pragma: no cover - numba is an optional runtime dependency in some envs
    from numba import njit
except ImportError:  # pragma: no cover

    def njit(*_args, **_kwargs):
        def decorate(func):
            return func

        return decorate


@njit(cache=True)
def _pbwt_sweep(haps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_sites, n_haps = haps.shape
    prefix = np.empty((n_sites, n_haps), dtype=np.int32)
    divergence = np.empty((n_sites, n_haps), dtype=np.int32)

    order = np.arange(n_haps, dtype=np.int32)
    div_state = np.zeros(n_haps, dtype=np.int32)

    zeros = np.empty(n_haps, dtype=np.int32)
    ones = np.empty(n_haps, dtype=np.int32)
    zeros_div = np.empty(n_haps, dtype=np.int32)
    ones_div = np.empty(n_haps, dtype=np.int32)
    diverge_row = np.empty(n_haps, dtype=np.int32)

    for site in range(n_sites):
        col = haps[site]
        n0 = 0
        n1 = 0
        su = site + 1
        sv = site + 1

        for i in range(n_haps):
            hap_idx = order[i]
            if col[hap_idx] == 0:
                zeros[n0] = hap_idx
                zeros_div[n0] = su
                su = div_state[i]
                n0 += 1
            else:
                ones[n1] = hap_idx
                ones_div[n1] = sv
                sv = div_state[i]
                n1 += 1

        for i in range(n0):
            order[i] = zeros[i]
            div_state[i] = zeros_div[i]
        for i in range(n1):
            order[n0 + i] = ones[i]
            div_state[n0 + i] = ones_div[i]

        prefix[site, :] = order

        if n_haps > 0:
            diverge_row[0] = 0
            for i in range(1, n_haps):
                val = div_state[i]
                if val > site:
                    val = site
                diverge_row[i] = val
            divergence[site, :] = diverge_row

    return prefix, divergence


def build_pbwt(haps: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build PBWT prefix/divergence arrays for haplotypes.

    Parameters
    ----------
    haps:
        Array shaped (variants, haplotypes) containing biallelic haplotypes encoded as 0/1.

    Returns
    -------
    prefix_arrays:
        List of orderings (numpy arrays of dtype int32) for each variant.
    divergence_arrays:
        List of divergence arrays aligned to the prefix ordering.
    """

    haps = ensure_int8(haps)
    if haps.ndim != 2:
        raise ValueError("haps must be a 2D array (sites, haplotypes)")

    n_sites, n_haps = haps.shape
    if n_sites == 0 or n_haps == 0:
        return [], []

    prefix_matrix, divergence_matrix = _pbwt_sweep(haps.astype(np.int8))

    prefix_arrays = [ensure_int32(prefix_matrix[i].copy()) for i in range(n_sites)]
    divergence_arrays = [ensure_int32(divergence_matrix[i].copy()) for i in range(n_sites)]
    return prefix_arrays, divergence_arrays


__all__ = ["build_pbwt"]
