"""Inference runner wrapping the CRF model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ..config import ExperimentConfig, InferConfig
from ..logging_ import get_logger
from ..model.crf import IBDCRF
from ..train.loop import build_model
from .postproc import segmentize

LOGGER = get_logger(__name__)


class InferenceRunner:
    def __init__(self, model_path: str, infer_cfg: InferConfig | None = None) -> None:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = state.get("config")
        if isinstance(cfg, ExperimentConfig):
            self.cfg = cfg
        else:
            self.cfg = ExperimentConfig()
        self.model: IBDCRF = build_model(self.cfg)
        self.model.load_state_dict(state.get("state_dict", {}))
        self.model.eval()
        self.infer_cfg = infer_cfg or self.cfg.infer

    def run(self, features: np.ndarray, dcm: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            feats_t = torch.from_numpy(features.astype(np.float32))
            dcm_t = torch.from_numpy(dcm.astype(np.float32))
            mask_t = torch.from_numpy(mask.astype(np.bool_))
            outputs = self.model(feats_t, dcm_t, mask_t)
            posterior = outputs["posterior"].cpu().numpy()
            return {
                "posterior": posterior,
            }

    def segmentize(self, posterior: np.ndarray, cm: np.ndarray) -> List[pd.DataFrame]:
        """Segmentise posterior probabilities into BED frames."""

        post_ibd = posterior[..., 1]
        results = []
        for row in post_ibd:
            df = segmentize(row, cm, self.infer_cfg)
            results.append(df)
        return results


__all__ = ["InferenceRunner"]
