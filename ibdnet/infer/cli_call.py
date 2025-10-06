"""Placeholder inference CLI."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from ..config import InferConfig, load_yaml_config
from ..logging_ import get_logger
from .runner import InferenceRunner

LOGGER = get_logger(__name__)


@click.command()
@click.option("--model", "model_path", type=click.Path(exists=True), required=True)
@click.option("--config", "config_path", type=click.Path(exists=False), required=False)
@click.option("--features", "features_path", type=click.Path(exists=True), required=True)
@click.option("--out", "out_dir", type=click.Path(), required=True)
def cli_call(model_path: str, config_path: str | None, features_path: str, out_dir: str) -> None:
    infer_cfg = None
    if config_path:
        cfg = load_yaml_config(config_path)
        infer_cfg = cfg.infer
    runner = InferenceRunner(model_path, infer_cfg)

    data = np.load(features_path)
    feature_names = sorted(data.files)
    feature_stack = np.stack([data[name] for name in feature_names], axis=-1)
    n_pairs, n_sites, _ = feature_stack.shape

    dcm = np.tile(np.linspace(0.0, 1.0, num=n_sites, dtype=np.float32), (n_pairs, 1))
    mask = np.ones((n_pairs, n_sites), dtype=bool)

    outputs = runner.run(feature_stack, dcm, mask)
    posterior = outputs["posterior"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "posterior.npy", posterior)
    LOGGER.info("Posterior saved to %s", out_path / "posterior.npy")


if __name__ == "__main__":
    cli_call()
