import tempfile

import numpy as np
import torch

from ibdnet.config import ExperimentConfig
from ibdnet.infer.runner import InferenceRunner
from ibdnet.train.loop import build_model


def test_inference_runner_end_to_end(tmp_path):
    cfg = ExperimentConfig()
    model = build_model(cfg)
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg}, checkpoint)

    runner = InferenceRunner(str(checkpoint))

    features = np.random.randn(1, 5, cfg.emissions.in_dim).astype(np.float32)
    dcm = np.linspace(0.0, 0.4, num=5, dtype=np.float32).reshape(1, -1)
    mask = np.ones((1, 5), dtype=bool)

    outputs = runner.run(features, dcm, mask)
    posterior = outputs["posterior"]
    assert posterior.shape == (1, 5, 2)
    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0, atol=1e-5)
