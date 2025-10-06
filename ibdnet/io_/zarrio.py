"""Helpers for storing dense arrays into Zarr workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr


def write_features(root: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write named feature arrays into a Zarr group."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(root_path), mode="a")
    for key, array in arrays.items():
        if key in group:
            del group[key]
        group.create_dataset(key, data=array, overwrite=True)


def read_features(root: str | Path) -> dict[str, np.ndarray]:
    """Load all arrays from a Zarr group."""

    group = zarr.open_group(str(root), mode="r")
    return {name: np.asarray(group[name]) for name in group.array_keys()}


__all__ = ["write_features", "read_features"]
