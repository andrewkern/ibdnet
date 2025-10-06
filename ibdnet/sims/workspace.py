"""Helpers for writing simulation outputs into structured workspaces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import zarr

from ..config import ExperimentConfig, to_dict
from .msprime_sim import SimulationResult


def _choose_chunks(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    chunks = []
    for idx, dim in enumerate(shape):
        if dim <= 0:
            chunks.append(1)
        elif idx == 0:
            chunks.append(min(dim, 32))
        elif idx == 1:
            chunks.append(min(dim, 1024))
        else:
            chunks.append(dim)
    return tuple(max(1, c) for c in chunks)


def write_workspace(out_dir: Path, result: SimulationResult, cfg: ExperimentConfig) -> dict[str, Any]:
    """Persist simulation artefacts into a Zarr-backed workspace and manifest.

    Parameters
    ----------
    out_dir:
        Directory where the workspace and manifest should be written.
    result:
        Simulation artefacts returned by :func:`simulate_dataset`.
    cfg:
        Experiment configuration used to generate the dataset. Stored alongside
        the artefacts for provenance.

    Returns
    -------
    dict
        Manifest dictionary that was written to disk.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace_path = out_dir / "workspace.zarr"
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    store = zarr.open_group(str(workspace_path), mode="w")

    store.create_array(
        "features",
        data=result.features,
        chunks=_choose_chunks(result.features.shape),
        overwrite=True,
    )
    store.create_array(
        "dcm",
        data=result.dcm.astype(np.float32),
        chunks=_choose_chunks(result.dcm.shape),
        overwrite=True,
    )
    store.create_array(
        "mask",
        data=result.mask.astype(np.bool_),
        chunks=_choose_chunks(result.mask.shape),
        overwrite=True,
    )
    store.create_array(
        "labels",
        data=result.labels.astype(np.int8),
        chunks=_choose_chunks(result.labels.shape),
        overwrite=True,
    )
    store.create_array(
        "pairs",
        data=result.pairs.astype(np.int32),
        chunks=_choose_chunks(result.pairs.shape),
        overwrite=True,
    )
    store.create_array(
        "positions_bp",
        data=result.positions_bp.astype(np.float64),
        chunks=_choose_chunks(result.positions_bp.shape),
        overwrite=True,
    )
    store.create_array(
        "positions_cm",
        data=result.positions_cm.astype(np.float64),
        chunks=_choose_chunks(result.positions_cm.shape),
        overwrite=True,
    )

    store.attrs["feature_names"] = result.feature_names
    store.attrs["num_pairs"] = int(result.features.shape[0])
    store.attrs["num_sites"] = int(result.features.shape[1])

    segments_path = out_dir / "truth_segments.tsv"
    result.segments.to_csv(segments_path, sep="\t", index=False)

    manifest = {
        "workspace": {
            "path": workspace_path.name,
            "feature_names": result.feature_names,
        },
        "segments": {
            "path": segments_path.name,
            "count": int(len(result.segments)),
        },
        "config": to_dict(cfg),
        "simulation_metadata": result.metadata,
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


__all__ = ["write_workspace"]
