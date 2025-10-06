import torch

from ibdnet.config import TransitionConfig
from ibdnet.model.transitions import compute_log_transitions


def test_compute_log_transitions_shape_and_values():
    dcm = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32)
    cfg = TransitionConfig()
    mat = compute_log_transitions(dcm[:, 1:], cfg)
    assert mat.shape == (1, 2, 2, 2)
    probs = mat.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))
