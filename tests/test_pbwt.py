import numpy as np

from ibdnet.pbwt.pbwt_core import build_pbwt


def test_build_pbwt_basic_ordering():
    haps = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )

    prefix_arrays, divergence_arrays = build_pbwt(haps)

    assert len(prefix_arrays) == haps.shape[0]
    assert len(divergence_arrays) == haps.shape[0]

    for prefix, divergence in zip(prefix_arrays, divergence_arrays):
        assert sorted(prefix.tolist()) == list(range(haps.shape[1]))
        assert divergence[0] == 0
        assert np.all(divergence >= 0)
