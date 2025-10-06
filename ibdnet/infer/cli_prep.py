"""Placeholder PBWT preparation CLI."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from ..config import PBWTConfig
from ..logging_ import get_logger
from ..pbwt.features import make_pair_features

LOGGER = get_logger(__name__)


@click.command()
@click.option("--vcf", "vcf_path", type=click.Path(exists=False), required=False)
@click.option("--map", "map_path", type=click.Path(exists=False), required=False)
@click.option("--pairs", "pairs_path", type=click.Path(exists=False), required=False)
@click.option("--workspace", "workspace", type=click.Path(), required=True)
def cli_prep(vcf_path: str | None, map_path: str | None, pairs_path: str | None, workspace: str) -> None:
    """Generate synthetic features as a placeholder."""

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    LOGGER.warning(
        "PBWT prep is not fully implemented; generating toy features at %s", workspace_path
    )
    haps = np.random.randint(0, 2, size=(100, 10), dtype=np.int8)
    cm = np.linspace(0.0, 1.0, num=100, dtype=np.float32)
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)
    features = make_pair_features(haps, cm, pairs, PBWTConfig())
    np.savez(workspace_path / "features.npz", **features)


if __name__ == "__main__":
    cli_prep()
