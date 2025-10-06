"""Configuration schemas and helpers for IBDNet."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import yaml


@dataclass(slots=True)
class PBWTConfig:
    """Configuration for PBWT feature generation."""

    k_neighbors: int = 8
    max_rank_dist: int = 32
    window_snps: int = 0
    normalize: Literal["chrom", "global"] = "chrom"


@dataclass(slots=True)
class EmissionConfig:
    """Configuration for emission networks."""

    model: Literal["mlp", "tcn", "tiny_transformer"] = "mlp"
    in_dim: int = 12
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1


@dataclass(slots=True)
class TransitionConfig:
    """Configuration for CRF transition potentials."""

    base_lambda: float = 2.0
    learn_lambda: bool = True
    min_non2ibd: float = 1e-6
    max_ibd2non: float = 0.5


@dataclass(slots=True)
class TrainConfig:
    """Training hyperparameters."""

    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_bp: int = 2_000_000
    grad_clip: float = 1.0
    epochs: int = 20
    amp: bool = True
    seed: int = 42


@dataclass(slots=True)
class InferConfig:
    """Inference and post-processing thresholds."""

    min_len_cm: float = 2.0
    min_post_mean: float = 0.6
    stitch_gap_bp: int = 20_000
    output_bigwig: bool = False


@dataclass(slots=True)
class SimulationConfig:
    """Simulation parameters for synthetic dataset generation."""

    n_samples: int = 200
    sequence_length: int = 5_000_000
    recombination_rate: float = 1.2e-8
    mutation_rate: float = 1.25e-8
    demographic_model: Optional[str] = None
    seed: int = 42
    n_pairs: int = 64
    max_time: float = 3_000.0
    min_cm: float = 1.0
    ploidy: int = 1
    species: Optional[str] = None
    genetic_map: Optional[str] = None
    chromosome: Optional[str] = None
    population_samples: Dict[str, int] = field(default_factory=dict)
    inclusion_mask: Optional[str] = None
    exclusion_mask: Optional[str] = None
    genotype_error_rate: float = 0.0


@dataclass(slots=True)
class DatasetConfig:
    """Data input specification for training."""

    path: str
    pairs: str
    map_path: Optional[str] = None
    chunk_size: int = 500_000


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level configuration container."""

    pbwt: PBWTConfig = field(default_factory=PBWTConfig)
    emissions: EmissionConfig = field(default_factory=EmissionConfig)
    transitions: TransitionConfig = field(default_factory=TransitionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    dataset: Optional[DatasetConfig] = None
    simulation: Optional[SimulationConfig] = None


T = TypeVar("T")


def _asdict_dataclass(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _asdict_dataclass(v) for k, v in asdict(obj).items()}
    return obj


def to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Convert a config object into a dict for logging/serialization."""

    return _asdict_dataclass(cfg)


def _merge_dict(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _merge_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _build_dataclass(cls: Type[T], payload: Mapping[str, Any]) -> T:
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in payload.items() if k in field_names}
    return cls(**kwargs)  # type: ignore[arg-type]


def load_yaml_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base = to_dict(ExperimentConfig())
    merged = _merge_dict(base, raw)

    return ExperimentConfig(
        pbwt=_build_dataclass(PBWTConfig, merged.get("pbwt", {})),
        emissions=_build_dataclass(EmissionConfig, merged.get("emissions", {})),
        transitions=_build_dataclass(TransitionConfig, merged.get("transitions", {})),
        train=_build_dataclass(TrainConfig, merged.get("train", {})),
        infer=_build_dataclass(InferConfig, merged.get("infer", {})),
        dataset=_build_dataclass(DatasetConfig, merged["dataset"]) if merged.get("dataset") else None,
        simulation=_build_dataclass(SimulationConfig, merged["simulation"]) if merged.get("simulation") else None,
    )


__all__ = [
    "PBWTConfig",
    "EmissionConfig",
    "TransitionConfig",
    "TrainConfig",
    "InferConfig",
    "SimulationConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "load_yaml_config",
    "to_dict",
]
