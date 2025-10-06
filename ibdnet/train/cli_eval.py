"""Placeholder evaluation CLI."""

from __future__ import annotations

from pathlib import Path

import click
import torch

from ..logging_ import get_logger

LOGGER = get_logger(__name__)


@click.command()
@click.option("--model", "model_path", type=click.Path(exists=True), required=True)
@click.option("--data", "data_path", type=click.Path(), required=False)
def cli_eval(model_path: str, data_path: str | None) -> None:
    state = torch.load(model_path, map_location="cpu")
    LOGGER.info("Loaded model checkpoint: %s", model_path)
    if data_path:
        LOGGER.warning("Evaluation pipeline not fully wired. Data path: %s", data_path)
    LOGGER.info("Available keys: %s", list(state.keys()))


if __name__ == "__main__":
    cli_eval()
