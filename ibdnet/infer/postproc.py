"""Post-processing utilities for IBD inference."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import InferConfig


def segmentize(post: np.ndarray, cm: np.ndarray, cfg: InferConfig) -> pd.DataFrame:
    """Convert per-site posteriors into BED-like segments."""

    if post.shape != cm.shape:
        raise ValueError("post and cm arrays must align")

    mask = post >= cfg.min_post_mean
    segments: List[dict[str, float | str | int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx - 1
            segments.append(_summarise_segment(start, end, post, cm, cfg))
            start = None
    if start is not None:
        segments.append(_summarise_segment(start, len(post) - 1, post, cm, cfg))

    df = pd.DataFrame(segments, columns=[
        "chrom",
        "start",
        "end",
        "name",
        "mean_post",
        "min_post",
        "len_bp",
        "len_cm",
    ])
    if df.empty:
        return df
    return df[df["len_cm"] >= cfg.min_len_cm].reset_index(drop=True)


def _summarise_segment(start: int, end: int, post: np.ndarray, cm: np.ndarray, cfg: InferConfig) -> dict[str, float | str | int]:
    seg_post = post[start : end + 1]
    seg_cm = cm[start : end + 1]
    mean_post = float(seg_post.mean())
    min_post = float(seg_post.min())
    len_bp = int(end - start + 1)
    len_cm = float(seg_cm[-1] - seg_cm[0]) if seg_cm.size > 1 else 0.0
    return {
        "chrom": "chrUNK",
        "start": int(start),
        "end": int(end + 1),
        "name": f"segment_{start}_{end}",
        "mean_post": mean_post,
        "min_post": min_post,
        "len_bp": len_bp,
        "len_cm": len_cm,
    }


__all__ = ["segmentize"]
