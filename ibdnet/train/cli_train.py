"""Command line interface for training IBDNet models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import torch

from ..config import ExperimentConfig, load_yaml_config
from ..logging_ import get_logger
from .loop import build_model

LOGGER = get_logger(__name__)


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file")
@click.option("--out", "out_dir", type=click.Path(), required=True, help="Output directory")
@click.option("--data", "data_path", type=click.Path(), required=False, help="Training data directory")
def cli_train(config_path: Optional[str], out_dir: str, data_path: Optional[str]) -> None:
    """Train an IBDNet model (placeholder implementation)."""

    cfg = load_yaml_config(config_path) if config_path else ExperimentConfig()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    LOGGER.info("Initialised model on %s", device)
    if data_path:
        LOGGER.warning(
            "Training logic is not fully implemented. Provided data path: %s", data_path
        )
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out_path / "model.pt")
    LOGGER.info("Saved untrained model checkpoint to %s", out_path / "model.pt")


if __name__ == "__main__":
    cli_train()
