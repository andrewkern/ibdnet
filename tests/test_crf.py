import numpy as np
import torch

from ibdnet.config import ExperimentConfig
from ibdnet.model.crf import IBDCRF
from ibdnet.train.loop import build_model


def test_crf_forward_shapes():
    cfg = ExperimentConfig()
    model = build_model(cfg)

    batch = 2
    seq = 5
    feats = torch.randn(batch, seq, cfg.emissions.in_dim)
    dcm = torch.linspace(0, 0.5, steps=seq).repeat(batch, 1)
    mask = torch.ones(batch, seq, dtype=torch.bool)

    out = model(feats, dcm, mask)
    assert set(out.keys()) == {"log_posterior", "posterior", "log_likelihood"}
    assert out["posterior"].shape == (batch, seq, 2)
    assert torch.allclose(out["posterior"].sum(dim=-1)[mask], torch.ones_like(mask, dtype=torch.float32)[mask])
