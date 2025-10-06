import json
from dataclasses import replace

import numpy as np
import pytest
import zarr

from ibdnet.config import PBWTConfig, SimulationConfig
from ibdnet.config import ExperimentConfig
from ibdnet.sims import simulate_dataset, write_workspace


def test_simulate_dataset_shapes():
    sim_cfg = SimulationConfig(
        n_samples=6,
        sequence_length=100_000,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        seed=7,
        n_pairs=3,
        max_time=10_000.0,
        min_cm=0.0,
        ploidy=1,
    )
    result = simulate_dataset(sim_cfg, PBWTConfig())

    assert result.features.shape[0] == result.pairs.shape[0] == sim_cfg.n_pairs
    assert result.features.shape[1] == result.labels.shape[1]
    assert result.features.shape[:2] == result.mask.shape
    assert result.features.shape[2] == len(result.feature_names)
    assert result.dcm.shape == result.mask.shape
    assert result.segments.columns.tolist() == [
        "chrom",
        "start",
        "end",
        "name",
        "mean_post",
        "min_post",
        "len_bp",
        "len_cm",
    ]
    assert np.all(result.mask)


def test_simulate_dataset_stdpopsim():
    pytest.importorskip("stdpopsim")

    sim_cfg = SimulationConfig(
        n_samples=6,
        sequence_length=200_000,
        mutation_rate=1.25e-8,
        recombination_rate=1.2e-8,
        seed=11,
        n_pairs=2,
        min_cm=0.0,
        species="HomSap",
        demographic_model="human_out_of_africa",
        chromosome="chr22",
        population_samples={"YRI": 6},
    )

    result = simulate_dataset(sim_cfg, PBWTConfig())

    assert result.features.shape[0] == 2
    assert result.features.shape[1] == result.labels.shape[1]
    assert result.positions_bp.shape == result.positions_cm.shape


def test_simulate_dataset_with_masks(tmp_path):
    mask_file = tmp_path / "include.bed"
    mask_file.write_text("chrSIM\t0\t50000\nchrSIM\t80000\t120000\n", encoding="utf-8")

    sim_cfg = SimulationConfig(
        n_samples=6,
        sequence_length=150_000,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        seed=21,
        n_pairs=2,
        max_time=10_000.0,
        min_cm=0.0,
        ploidy=1,
        inclusion_mask=str(mask_file),
    )

    result = simulate_dataset(sim_cfg, PBWTConfig())

    site_mask = result.mask[0]
    assert not bool(site_mask.all())
    assert np.all(result.features[:, ~site_mask, :] == 0.0)
    assert np.all(result.labels[:, ~site_mask] == 0)
    if not result.segments.empty:
        assert (result.segments["start"] >= 0).all()
        assert (result.segments["end"] <= 120000).all()


def test_write_workspace(tmp_path):
    sim_cfg = SimulationConfig(
        n_samples=4,
        sequence_length=50_000,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        seed=17,
        n_pairs=1,
        min_cm=0.0,
    )

    result = simulate_dataset(sim_cfg, PBWTConfig())
    cfg = ExperimentConfig(simulation=sim_cfg)

    manifest = write_workspace(tmp_path, result, cfg)

    manifest_path = tmp_path / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["workspace"]["feature_names"] == result.feature_names
    assert data["simulation_metadata"]["species"] == "HomSap"

    group = zarr.open_group(str(tmp_path / manifest["workspace"]["path"]), mode="r")
    assert group["features"].shape == result.features.shape
    np.testing.assert_allclose(group["positions_bp"], result.positions_bp)


def test_genotype_error_rate_impacts_features():
    base_cfg = SimulationConfig(
        n_samples=4,
        sequence_length=80_000,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        seed=31,
        n_pairs=2,
        min_cm=0.0,
        genotype_error_rate=0.0,
    )

    noisy_cfg = replace(base_cfg, genotype_error_rate=0.2)

    clean = simulate_dataset(base_cfg, PBWTConfig())
    noisy = simulate_dataset(noisy_cfg, PBWTConfig())

    assert clean.features.shape == noisy.features.shape
    assert not np.array_equal(clean.features, noisy.features)


def test_human_model_alias_resolution():
    std = pytest.importorskip("stdpopsim")

    cfg = SimulationConfig(
        n_samples=4,
        sequence_length=120_000,
        recombination_rate=1.2e-8,
        mutation_rate=1.25e-8,
        seed=41,
        n_pairs=2,
        min_cm=0.0,
        demographic_model="human_ancient_europe",
        genetic_map="hapmap_grch38",
    )

    result = simulate_dataset(cfg, PBWTConfig())
    assert result.metadata["model"] == "AncientEurope_4A21"
    assert result.metadata["genetic_map"] == "HapMapII_GRCh38"
    assert result.metadata["species"] == "HomSap"
    assert result.features.shape[0] == cfg.n_pairs
