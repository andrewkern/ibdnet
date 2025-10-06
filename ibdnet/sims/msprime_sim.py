"""Simulation utilities built on :mod:`msprime`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import re

import msprime

from ..config import PBWTConfig, SimulationConfig
from ..pbwt.features import make_pair_features

try:  # pragma: no cover - optional dependency
    import stdpopsim

    _HAS_STDPOPSIM = True
except ImportError:  # pragma: no cover
    stdpopsim = None
    _HAS_STDPOPSIM = False

_HUMAN_MODEL_ALIAS_STATIC: Dict[str, str] = {
    "human_out_of_africa": "OutOfAfrica_3G09",
    "human_out_of_africa_extended": "OutOfAfricaExtendedNeandertalAdmixturePulse_3I21",
    "human_out_of_africa_archaic": "OutOfAfricaArchaicAdmixture_5R19",
    "human_out_of_africa_two_epoch": "OutOfAfrica_2T12",
    "human_out_of_africa_four_epoch": "OutOfAfrica_4J17",
    "human_africa": "Africa_1B08",
    "human_africa_two_epoch": "Africa_1T12",
    "human_american_admixture": "AmericanAdmixture_4B18",
    "human_zigzag": "Zigzag_1S14",
    "human_ancient_eurasia": "AncientEurasia_9K19",
    "human_ancient_europe": "AncientEurope_4A21",
    "human_papuans_out_of_africa": "PapuansOutOfAfrica_10J19",
    "human_ashkenazi_substructure": "AshkSub_7G19",
}

_HUMAN_MAP_ALIAS_STATIC: Dict[str, str] = {
    "hapmap_grch37": "HapMapII_GRCh37",
    "hapmap_grch38": "HapMapII_GRCh38",
    "decode_grch38": "DeCodeSexAveraged_GRCh38",
    "decode_grch36": "DeCodeSexAveraged_GRCh36",
}

_HUMAN_MODEL_CATALOG: Dict[str, tuple[str, str]] | None = None
_HUMAN_MAP_CATALOG: Dict[str, str] | None = None


def _snakecase(identifier: str) -> str:
    alias = identifier.replace("-", "_")
    alias = re.sub(r"(?<!^)(?=[A-Z])", "_", alias)
    alias = alias.replace("__", "_")
    return alias.lower()


def _ensure_human_catalog() -> None:
    if not _HAS_STDPOPSIM:
        return

    global _HUMAN_MODEL_CATALOG, _HUMAN_MAP_CATALOG
    if _HUMAN_MODEL_CATALOG is not None and _HUMAN_MAP_CATALOG is not None:
        return

    species = stdpopsim.get_species("HomSap")

    model_map: Dict[str, tuple[str, str]] = {}
    for model in species.demographic_models:
        canonical = model.id
        aliases = {
            canonical.lower(),
            _snakecase(canonical),
            f"human_{_snakecase(canonical)}",
        }
        for alias in aliases:
            model_map[alias] = ("HomSap", canonical)
    for alias, canonical in _HUMAN_MODEL_ALIAS_STATIC.items():
        model_map[alias.lower()] = ("HomSap", canonical)

    map_map: Dict[str, str] = {}
    for gmap in species.genetic_maps:
        canonical_map = gmap.id
        aliases = {
            canonical_map.lower(),
            _snakecase(canonical_map),
            f"human_{_snakecase(canonical_map)}",
        }
        for alias in aliases:
            map_map[alias] = canonical_map
    for alias, canonical in _HUMAN_MAP_ALIAS_STATIC.items():
        map_map[alias.lower()] = canonical

    _HUMAN_MODEL_CATALOG = model_map
    _HUMAN_MAP_CATALOG = map_map


def get_human_model_aliases() -> List[str]:  # pragma: no cover - helper for callers
    if not _HAS_STDPOPSIM:
        return sorted(_HUMAN_MODEL_ALIAS_STATIC.keys())
    _ensure_human_catalog()
    return sorted(_HUMAN_MODEL_CATALOG.keys()) if _HUMAN_MODEL_CATALOG else []


def get_human_map_aliases() -> List[str]:  # pragma: no cover - helper for callers
    if not _HAS_STDPOPSIM:
        return sorted(_HUMAN_MAP_ALIAS_STATIC.keys())
    _ensure_human_catalog()
    return sorted(_HUMAN_MAP_CATALOG.keys()) if _HUMAN_MAP_CATALOG else []


def _load_intervals(path: str) -> List[tuple[str | None, float, float]]:
    intervals: List[tuple[str | None, float, float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                chrom = None
                start, end = parts
            elif len(parts) >= 3:
                chrom, start, end = parts[:3]
            else:
                continue
            try:
                start_f = float(start)
                end_f = float(end)
            except ValueError:
                continue
            if end_f <= start_f:
                continue
            intervals.append((chrom, start_f, end_f))
    return intervals


def _apply_intervals(mask: np.ndarray, positions: np.ndarray, intervals: List[tuple[str | None, float, float]], chrom_label: str, value: bool) -> None:
    for chrom, start, end in intervals:
        if chrom is not None and chrom != chrom_label:
            continue
        left = int(np.searchsorted(positions, start, side="left"))
        right = int(np.searchsorted(positions, end, side="right"))
        if right > left:
            mask[left:right] = value


def _build_site_mask(positions_bp: np.ndarray, chrom_label: str, sim_cfg: SimulationConfig) -> np.ndarray:
    site_mask = np.ones_like(positions_bp, dtype=bool)

    if sim_cfg.inclusion_mask:
        intervals = _load_intervals(sim_cfg.inclusion_mask)
        site_mask[:] = False
        _apply_intervals(site_mask, positions_bp, intervals, chrom_label, True)

    if sim_cfg.exclusion_mask:
        intervals = _load_intervals(sim_cfg.exclusion_mask)
        _apply_intervals(site_mask, positions_bp, intervals, chrom_label, False)

    if not site_mask.any():
        raise ValueError("Masking removed all sites; adjust masks or simulation parameters")
    return site_mask


@dataclass(slots=True)
class SimulationResult:
    """Container for the synthetic dataset artefacts."""

    features: np.ndarray
    feature_names: List[str]
    dcm: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    pairs: np.ndarray
    positions_bp: np.ndarray
    positions_cm: np.ndarray
    segments: pd.DataFrame
    metadata: Dict[str, str] = field(default_factory=dict)


def _select_pairs(n_haps: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    combos = []
    for a in range(n_haps):
        for b in range(a + 1, n_haps):
            combos.append((a, b))
    if not combos:
        raise ValueError("need at least two haplotypes to form pairs")
    n_pairs = min(n_pairs, len(combos))
    indices = rng.choice(len(combos), size=n_pairs, replace=False)
    return np.array([combos[i] for i in indices], dtype=np.int32)


def _build_segments_and_labels(
    ts: msprime.TreeSequence,
    pairs: np.ndarray,
    positions_bp: np.ndarray,
    positions_cm: np.ndarray,
    sim_cfg: SimulationConfig,
    site_mask: np.ndarray | None,
    chrom_label: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    cm_per_bp = sim_cfg.recombination_rate * 100.0
    n_pairs = pairs.shape[0]
    n_sites = positions_bp.shape[0]
    labels = np.zeros((n_pairs, n_sites), dtype=np.int8)
    records: List[Dict[str, float | str]] = []

    samples = ts.samples()
    node_pairs = [(int(samples[a]), int(samples[b])) for a, b in map(tuple, pairs)]
    pair_lookup: Dict[Tuple[int, int], int] = {}
    for idx, (n1, n2) in enumerate(node_pairs):
        pair_lookup[(n1, n2)] = idx
        pair_lookup[(n2, n1)] = idx

    columns = [
        "chrom",
        "start",
        "end",
        "name",
        "mean_post",
        "min_post",
        "len_bp",
        "len_cm",
    ]

    segments_obj = ts.ibd_segments(
        max_time=sim_cfg.max_time,
        store_pairs=True,
        store_segments=True,
    )

    for (node1, node2), seg_list in segments_obj.items():
        pair_idx = pair_lookup.get((int(node1), int(node2)))
        if pair_idx is None:
            continue
        for seg in seg_list:
            len_cm = (seg.right - seg.left) * cm_per_bp
            if len_cm < sim_cfg.min_cm:
                continue
            start_bp = float(seg.left)
            end_bp = float(seg.right)

            start_idx = int(np.searchsorted(positions_bp, start_bp, side="left"))
            end_idx = int(np.searchsorted(positions_bp, end_bp, side="left") - 1)
            if start_idx >= n_sites or end_idx < start_idx:
                continue
            end_idx = min(end_idx, n_sites - 1)
            if site_mask is not None:
                orig_start = start_idx
                segment_mask = site_mask[orig_start : end_idx + 1]
                if not bool(segment_mask.any()):
                    continue
                first_rel = int(np.argmax(segment_mask))
                last_rel = int(len(segment_mask) - 1 - np.argmax(segment_mask[::-1]))
                start_idx = orig_start + first_rel
                end_idx = orig_start + last_rel

            labels[pair_idx, start_idx : end_idx + 1] = 1

            adj_start_bp = positions_bp[start_idx]
            adj_end_bp = positions_bp[end_idx]
            adj_start_cm = positions_cm[start_idx]
            adj_end_cm = positions_cm[end_idx]

            records.append(
                {
                    "chrom": chrom_label,
                    "start": int(adj_start_bp),
                    "end": int(adj_end_bp),
                    "name": f"pair_{pairs[pair_idx, 0]}_{pairs[pair_idx, 1]}",
                    "mean_post": 1.0,
                    "min_post": 1.0,
                    "len_bp": float(adj_end_bp - adj_start_bp),
                    "len_cm": float(adj_end_cm - adj_start_cm),
                }
            )

    segments = pd.DataFrame(records, columns=columns)
    return labels, segments


def _run_basic_msprime(sim_cfg: SimulationConfig) -> tuple[msprime.TreeSequence, object | None]:
    demography = msprime.Demography.isolated_model(initial_size=[10_000])
    ts = msprime.sim_ancestry(
        samples=sim_cfg.n_samples,
        sequence_length=sim_cfg.sequence_length,
        recombination_rate=sim_cfg.recombination_rate,
        random_seed=sim_cfg.seed,
        ploidy=sim_cfg.ploidy,
        demography=demography,
    )
    ts = msprime.sim_mutations(
        ts,
        rate=sim_cfg.mutation_rate,
        random_seed=sim_cfg.seed,
    )
    if ts.num_sites == 0:
        raise RuntimeError("Simulation produced no polymorphic sites; adjust parameters")
    return ts, None


def _run_stdpopsim(sim_cfg: SimulationConfig) -> tuple[msprime.TreeSequence, object, str, str, str | None, str]:
    if not _HAS_STDPOPSIM:
        raise ImportError("stdpopsim is not installed")

    _ensure_human_catalog()

    species_id = sim_cfg.species
    model_id = sim_cfg.demographic_model
    resolved_map = sim_cfg.genetic_map

    catalog = _HUMAN_MODEL_CATALOG or {}
    if model_id:
        alias_key = model_id.lower()
        if alias_key in catalog and species_id in (None, catalog[alias_key][0]):
            species_id, model_id = catalog[alias_key]
        elif species_id in (None, "HomSap") and alias_key not in catalog:
            # allow direct canonical IDs; handled below
            pass

    if species_id is None:
        species_id = "HomSap"

    if model_id is None:
        raise ValueError(
            "demographic_model must be provided when using stdpopsim simulations"
        )

    species = stdpopsim.get_species(species_id)
    available_models = {m.id for m in species.demographic_models}
    if model_id not in available_models:
        suggestions = sorted(available_models)
        if species_id == "HomSap" and _HUMAN_MODEL_CATALOG:
            suggestions = sorted(set(_HUMAN_MODEL_CATALOG.keys()) | set(suggestions))
        raise ValueError(
            f"Unknown demographic model '{model_id}' for species '{species_id}'. Available: {suggestions}"
        )
    model = species.get_demographic_model(model_id)

    default_chrom = species.genome.chromosomes[0].id
    chrom_id = sim_cfg.chromosome or default_chrom
    contig_kwargs: Dict[str, object] = {"chromosome": chrom_id}

    map_catalog = _HUMAN_MAP_CATALOG or {}
    if sim_cfg.genetic_map:
        map_key = sim_cfg.genetic_map.lower()
        if species_id == "HomSap" and map_key in map_catalog:
            resolved_map = map_catalog[map_key]
        elif map_key in map_catalog and species_id != "HomSap":
            raise ValueError(
                f"Genetic map alias '{sim_cfg.genetic_map}' requires species 'HomSap'"
            )
        else:
            resolved_map = sim_cfg.genetic_map
        contig_kwargs["genetic_map"] = resolved_map
    else:
        resolved_map = None

    available_maps = {g.id for g in species.genetic_maps}
    if resolved_map is not None and resolved_map not in available_maps:
        suggestions = sorted(available_maps)
        if species_id == "HomSap" and _HUMAN_MAP_CATALOG:
            suggestions = sorted(set(_HUMAN_MAP_CATALOG.keys()) | set(suggestions))
        raise ValueError(
            f"Unknown genetic map '{resolved_map}' for species '{species_id}'. Available: {suggestions}"
        )

    chrom = species.genome.get_chromosome(chrom_id)
    if sim_cfg.sequence_length and sim_cfg.sequence_length < chrom.length:
        contig_kwargs["right"] = float(sim_cfg.sequence_length)

    try:
        contig = species.get_contig(**contig_kwargs)
    except ValueError as err:
        if "All intervals are missing data" in str(err) and "right" in contig_kwargs:
            contig_kwargs.pop("right")
            contig = species.get_contig(**contig_kwargs)
        else:
            raise

    samples_dict: Dict[str, int] = {}
    if sim_cfg.population_samples:
        for pop_name, hap_count in sim_cfg.population_samples.items():
            if hap_count <= 0:
                continue
            if hap_count % sim_cfg.ploidy != 0:
                raise ValueError(
                    f"Population '{pop_name}' haplotype count {hap_count} is not divisible by ploidy {sim_cfg.ploidy}"
                )
            individuals = hap_count // sim_cfg.ploidy
            if individuals > 0:
                samples_dict[pop_name] = individuals

    if not samples_dict:
        default_pop = model.populations[0]
        pop_name = getattr(default_pop, "name", None) or str(default_pop.id)
        if sim_cfg.n_samples % sim_cfg.ploidy != 0:
            raise ValueError("n_samples must be divisible by ploidy for stdpopsim simulations")
        individuals = max(1, sim_cfg.n_samples // sim_cfg.ploidy)
        samples_dict[pop_name] = individuals

    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, contig, samples=samples_dict, seed=sim_cfg.seed)
    return ts.simplify(), contig, species_id, model_id, resolved_map, chrom_id


def simulate_dataset(sim_cfg: SimulationConfig, pbwt_cfg: PBWTConfig) -> SimulationResult:
    """Generate a synthetic dataset with PBWT-derived features and ground-truth labels."""

    rng = np.random.default_rng(sim_cfg.seed)
    resolved_species = sim_cfg.species
    resolved_model = sim_cfg.demographic_model
    resolved_map = sim_cfg.genetic_map

    use_stdpopsim = sim_cfg.species is not None or (
        sim_cfg.demographic_model is not None and _HAS_STDPOPSIM
    )

    resolved_chrom = None
    if use_stdpopsim and _HAS_STDPOPSIM:
        ts, contig, resolved_species, resolved_model, resolved_map, resolved_chrom = _run_stdpopsim(sim_cfg)
    else:
        ts, contig = _run_basic_msprime(sim_cfg)
        resolved_species = resolved_species or "HomSap"
        resolved_chrom = resolved_chrom or "chrSIM"

    haplotypes = ts.genotype_matrix().astype(np.int8)
    if not 0.0 <= sim_cfg.genotype_error_rate <= 1.0:
        raise ValueError("genotype_error_rate must be between 0 and 1")
    if sim_cfg.genotype_error_rate > 0:
        flip_mask = rng.random(haplotypes.shape) < sim_cfg.genotype_error_rate
        if flip_mask.any():
            haplotypes = np.where(flip_mask, 1 - haplotypes, haplotypes)
    positions_bp = np.asarray(ts.tables.sites.position, dtype=np.float64)
    if positions_bp.size == 0:
        raise RuntimeError("Simulation produced no variant sites; adjust parameters or masks")

    chrom_label = resolved_chrom or "chrSIM"
    if contig is not None and getattr(contig, "id", None):
        chrom_label = getattr(contig, "id") or chrom_label

    if contig is not None and getattr(contig, "genetic_map", None) is not None:
        rate_map = contig.genetic_map.get_chromosome_map(chrom_label)
        positions_cm = rate_map.get_cumulative_mass(positions_bp) * 100.0
    else:
        positions_cm = positions_bp * sim_cfg.recombination_rate * 100.0

    if sim_cfg.inclusion_mask or sim_cfg.exclusion_mask:
        site_mask = _build_site_mask(positions_bp, chrom_label, sim_cfg)
    else:
        site_mask = np.ones_like(positions_bp, dtype=bool)

    pairs = _select_pairs(haplotypes.shape[1], sim_cfg.n_pairs, rng)
    feature_dict = make_pair_features(haplotypes, positions_cm.astype(np.float32), pairs, pbwt_cfg)
    feature_names = sorted(feature_dict.keys())
    feature_stack = np.stack([feature_dict[name] for name in feature_names], axis=-1).astype(np.float32)

    feature_stack[:, ~site_mask, :] = 0.0

    dcm = np.diff(positions_cm, prepend=positions_cm[0])
    dcm = np.broadcast_to(dcm.astype(np.float32), feature_stack.shape[:2]).copy()
    mask = np.broadcast_to(site_mask, feature_stack.shape[:2]).copy()

    labels, segments = _build_segments_and_labels(
        ts,
        pairs,
        positions_bp,
        positions_cm,
        sim_cfg,
        site_mask,
        chrom_label,
    )
    labels[:, ~site_mask] = 0

    return SimulationResult(
        features=feature_stack,
        feature_names=feature_names,
        dcm=dcm,
        mask=mask,
        labels=labels,
        pairs=pairs,
        positions_bp=positions_bp,
        positions_cm=positions_cm,
        segments=segments,
        metadata={
            "species": resolved_species,
            "model": resolved_model,
            "genetic_map": resolved_map,
            "chromosome": chrom_label,
        },
    )


__all__ = ["SimulationResult", "simulate_dataset"]
