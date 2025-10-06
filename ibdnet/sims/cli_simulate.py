"""CLI for generating msprime-backed simulation datasets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import click

from ..config import ExperimentConfig, SimulationConfig, load_yaml_config
from ..logging_ import get_logger
from .msprime_sim import simulate_dataset
from .workspace import write_workspace

LOGGER = get_logger(__name__)


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--out", "out_dir", type=click.Path(), required=True)
def cli_simulate(config_path: str, out_dir: str) -> None:
    """Generate PBWT-ready simulation data and ground-truth labels."""

    cfg: ExperimentConfig = load_yaml_config(config_path)
    sim_cfg: SimulationConfig = cfg.simulation or SimulationConfig()

    LOGGER.info(
        "Simulating %s haplotypes across %s bp (pairs=%s)",
        sim_cfg.n_samples,
        sim_cfg.sequence_length,
        sim_cfg.n_pairs,
    )
    if sim_cfg.species:
        LOGGER.info(
            "stdpopsim species=%s model=%s map=%s chromosome=%s",
            sim_cfg.species,
            sim_cfg.demographic_model,
            sim_cfg.genetic_map,
            sim_cfg.chromosome,
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg.simulation = sim_cfg
    result = simulate_dataset(sim_cfg, cfg.pbwt)

    resolved_meta = result.metadata
    if resolved_meta.get("model") or resolved_meta.get("genetic_map"):
        LOGGER.info(
            "Resolved stdpopsim: species=%s model=%s map=%s",
            resolved_meta.get("species"),
            resolved_meta.get("model"),
            resolved_meta.get("genetic_map"),
        )
    cfg.simulation = replace(
        sim_cfg,
        species=resolved_meta.get("species", sim_cfg.species),
        demographic_model=resolved_meta.get("model", sim_cfg.demographic_model),
        genetic_map=resolved_meta.get("genetic_map", sim_cfg.genetic_map),
    )

    manifest = write_workspace(out_path, result, cfg)
    LOGGER.info(
        "Synthetic dataset written to %s/%s (features=%s×%s×%s)",
        out_path,
        manifest["workspace"]["path"],
        result.features.shape[0],
        result.features.shape[1],
        result.features.shape[2],
    )


if __name__ == "__main__":
    cli_simulate()
