import numpy as np

from ibdnet.config import PBWTConfig
from ibdnet.pbwt.features import make_pair_features


def test_make_pair_features_shapes():
    haps = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
        ],
        dtype=np.int8,
    )
    cm = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float32)
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)

    features = make_pair_features(haps, cm, pairs, PBWTConfig())

    assert set(features.keys()) == {"ibs", "rank_dist", "mismatch_run", "cm_delta"}
    for key, value in features.items():
        assert value.shape == (pairs.shape[0], haps.shape[0])
        assert value.dtype == np.float32
