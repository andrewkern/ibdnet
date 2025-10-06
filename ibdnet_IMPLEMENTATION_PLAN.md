ibdnet_IMPLEMENTATION_PLAN.md
0) High-level spec

Goal: Infer per-site IBD posteriors and segment calls for haplotype pairs using PBWT-derived features fed into a neural emission model within a CRF/HMM whose transitions depend on genetic map distance.

Inputs: Phased VCF/BCF/PLINK + genetic map (cM). Optional genotype likelihoods/dosages.

Outputs:

Per-site posterior P(IBD) (NPZ or BigWig).

segments.bed.gz with scores.

JSON summaries + logs.

Benchmark reports vs baselines.

Core deps: Python 3.11, numpy, numba, polars, tskit, msprime, (py)PBWT or custom, PyTorch, torch-struct, zarr, cyvcf2, pybedtools (or pyranges).

1) Repository layout (to scaffold exactly)
ibdnet/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  Makefile
  ibdnet/
    __init__.py
    config.py
    logging_.py
    io_/
      __init__.py
      vcf.py
      plink.py
      maps.py
      zarrio.py
      bed.py
    pbwt/
      __init__.py
      pbwt_core.py
      features.py
      fastarrays.py
    model/
      __init__.py
      emissions.py
      transitions.py
      crf.py
      heads.py
      utils.py
    train/
      dataset.py
      loop.py
      metrics.py
      calibrate.py
      cli_train.py
      cli_eval.py
    infer/
      runner.py
      cli_prep.py
      cli_call.py
      postproc.py
    sims/
      msprime_sim.py
      slim_hooks.py
      labels.py
      cli_simulate.py
  tests/
    test_pbwt.py
    test_features.py
    test_transitions.py
    test_crf.py
    test_infer_end2end.py
  scripts/
    bench_hapibd.py
    bench_ilash.py
    bench_phasedibd.py
  configs/
    train_small.yaml
    infer_default.yaml
    sim_human_mix.yaml

2) Configs (single source of truth)

File: ibdnet/config.py

from dataclasses import dataclass
from typing import Literal

@dataclass
class PBWTConfig:
    k_neighbors: int = 8
    max_rank_dist: int = 32
    window_snps: int = 0                  # 0 => per-site features
    normalize: Literal["chrom","global"] = "chrom"

@dataclass
class EmissionConfig:
    model: Literal["mlp","tcn","tiny_transformer"] = "mlp"
    in_dim: int = 12                      # set by feature schema
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1

@dataclass
class TransitionConfig:
    base_lambda: float = 2.0              # per cM; softplus-learnable if learn_lambda
    learn_lambda: bool = True
    min_non2ibd: float = 1e-6
    max_ibd2non: float = 0.5

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_bp: int = 2_000_000            # tile length in bp/site index units
    grad_clip: float = 1.0
    epochs: int = 20
    amp: bool = True
    seed: int = 42

@dataclass
class InferConfig:
    min_len_cm: float = 2.0
    min_post_mean: float = 0.6
    stitch_gap_bp: int = 20000
    output_bigwig: bool = False

@dataclass
class IBDNetConfig:
    pbwt: PBWTConfig = PBWTConfig()
    emissions: EmissionConfig = EmissionConfig()
    transitions: TransitionConfig = TransitionConfig()
    train: TrainConfig = TrainConfig()
    infer: InferConfig = InferConfig()


Implement a YAML loader that maps 1:1 to these dataclasses.

3) PBWT + feature extraction
3.1 PBWT core

File: ibdnet/pbwt/pbwt_core.py

Implement forward PBWT (Durbin) over binary haplotypes {0,1,-1}.

Outputs per site:

order[i]: permutation of hap indices (np.int32).

div[i]: divergence array (np.int32) = last mismatch index for each hap at site i.

Numba-accelerate inner loops; memory-map large arrays.

Contracts:

import numpy as np
from typing import List, Tuple

def build_pbwt(hap_matrix: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    hap_matrix: [H, L] int8 in {0,1,-1} (-1=missing)
    Returns: (orders, divs), each a list of length L with np.ndarray[H] (int32)
    """

3.2 Pairwise PBWT features

File: ibdnet/pbwt/features.py

Per haplotype pair (ha, hb) at site i compute:

rank_a(i), rank_b(i) from inverse permutation of order[i]

Δrank(i) = |rank_a - rank_b|

div_a(i), div_b(i) from div[i]

match(i) (allele equality) and short run-length since last mismatch

neighbor features: min/mean Δrank among k PBWT neighbors of ha/hb

ΔcM(i) since previous site; cumulative cM

missingness flags; local MAF (within sliding window)

Persist feature stack to zarr with stable column order (write a schema JSON).

Contract:

from ..config import PBWTConfig
from typing import Dict

def make_pair_features(
    chrom: str,
    hap_a: int,
    hap_b: int,
    alleles: np.ndarray,               # [H, L] int8 {0,1,-1}
    orders: list[np.ndarray],
    divs: list[np.ndarray],
    cm: np.ndarray,                    # [L] float32 cM positions
    cfg: PBWTConfig
) -> Dict[str, np.ndarray]:
    """
    Returns dict of named feature arrays [L] (float32/int8 as appropriate).
    Required keys (stable ordering recorded in schema):
      ['drank','div_a','div_b','match','runlen_match','min_drank_k',
       'maf_local','het_local','missing_a','missing_b','dcm','cm']
    """

4) Simulation + labels

Files: ibdnet/sims/msprime_sim.py, ibdnet/sims/labels.py, ibdnet/sims/cli_simulate.py

Simulate human-like demographies (CEU, YRI, EAS, admixed) with msprime.

Export phased haplotypes, marker positions, genetic map positions.

Inject realism:

genotype error flips, missingness

phasing switch errors at a configurable rate per cM

thinning to array densities (e.g., 500k/1M)

Labels: from tree sequence, derive IBD tracts for each pair using msprime’s ibd_segments() or custom via breakpoints/TMRCA continuity.

Emit:

per-site binary IBD labels [L]

list of (start_idx, end_idx) tracts per pair

CLI:

python -m ibdnet.sims.cli_simulate \
  --config configs/sim_human_mix.yaml \
  --out data/sim1/

5) Model
5.1 Emissions

File: ibdnet/model/emissions.py

Default: MLP emissions (fast).

Alternatives: TCN (temporal conv) or TinyTransformer with 2 encoder layers, 4 heads.

Contract:

import torch
import torch.nn as nn

class MLPEmissions(nn.Module):
    def __init__(self, in_dim:int, hidden:int, layers:int, dropout:float): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F] float32
        returns logits: [B, L, 2] for states {non-IBD, IBD}
        """

class TCNEmissions(nn.Module): ...
class TinyTransformerEmissions(nn.Module): ...

5.2 Transitions (map-tied)

File: ibdnet/model/transitions.py

Per-site 2×2 transition matrices in log-space based on ΔcM.

Parameters:

p(IBD→non) = 1 − exp(−softplus(lambda) * ΔcM)

p(non→IBD) = clamp(alpha * ΔcM, min_non2ibd, 0.1)

lambda, alpha learnable (regularize with L2; clip to sane bounds).

Contract:

def compute_log_transitions(
    dcm: torch.Tensor,    # [B, L] float32
    cfg
) -> torch.Tensor:
    """
    Returns log_trans: [B, L, 2, 2] (log-prob transitions between sites)
    """

5.3 CRF wrapper

File: ibdnet/model/crf.py

Use torch_struct.LinearChainCRF with emissions [B,L,2] and log-transitions [B,L,2,2].

Support masks for missing sites.

Contract:

from torch_struct import LinearChainCRF

class IBDCRF(nn.Module):
    def __init__(self, emissions: nn.Module, trans_cfg): ...
    def forward(self, feats, dcm, mask) -> dict:
        """
        feats: [B,L,F], dcm: [B,L], mask: [B,L] boolean
        returns {'post': [B,L,2], 'logZ': [], 'vit': [B,L] int64}
        """

6) Training
6.1 Dataset/Loader

File: ibdnet/train/dataset.py

Zarr-backed feature/label store per pair per chromosome.

Tile long sequences to batch_bp windows with overlap (e.g., 50–100 kb).

Yield (features, ΔcM, mask, labels) tensors.

6.2 Loop

File: ibdnet/train/loop.py

AMP on, cosine LR with warmup, AdamW, grad clip=1.0.

Loss = CRF negative log-likelihood.

Optional boundary focal loss on dilated boundaries (±k sites).

Track:

per-base AUROC/AUPRC

segment F1 (min_len_cm in {1,2,3,5})

calibration (ECE, Brier)

CLI:

python -m ibdnet.train.cli_train \
  --config configs/train_small.yaml \
  --data data/sim1/ \
  --out runs/exp1/

6.3 Checkpoints

Save model.pt (state_dict), config.yaml, feature_schema.json.

Optional export_onnx.py for emissions.

7) Inference
7.1 Prep (PBWT + features)

File: ibdnet/infer/cli_prep.py

Inputs: --vcf, --map, --pairs pairs.tsv (pair_id, sample_a, hap_a(0/1), sample_b, hap_b(0/1)).

Build PBWT per chromosome once → cache under workspace/pbwt/chr*.npz.

Compute pairwise features → zarr under workspace/features/{pair}/{chrom}.zarr.

7.2 Call

File: ibdnet/infer/cli_call.py

Load model + config, stream tiles, output:

posteriors.npz (or BigWig if enabled)

segments via postproc.segmentize()

File: ibdnet/infer/postproc.py

import numpy as np
import pandas as pd
from ..config import InferConfig

def segmentize(post: np.ndarray, cm: np.ndarray, cfg: InferConfig) -> pd.DataFrame:
    """
    Hysteresis thresholding:
      start if post > 0.6, end if post < 0.4
      drop seg if len_cM < cfg.min_len_cm
      stitch gaps < cfg.stitch_gap_bp when inter-gap post > 0.5
    Return DataFrame with columns:
      ['chrom','start_bp','end_bp','pair_id','mean_post','min_post','len_bp','len_cM']
    """

7.3 Runner API

File: ibdnet/infer/runner.py

class IBDNet:
    def __init__(self, model_path:str, cfg): ...
    def call_pair(self, chrom:str, feats:np.ndarray, dcm:np.ndarray, mask:np.ndarray) -> dict:
        """Returns {'post': np.ndarray[L], 'segments': pd.DataFrame}"""

8) Evaluation & Benchmarks

File: ibdnet/train/metrics.py

Per-base: AUROC, AUPRC, Brier, ECE.

Segment-level: Precision/Recall/F1 at min_len_cm ∈ {1,2,3,5}; Jaccard overlap; boundary MAE (cM).

Length-binned FDR: bins (1–2, 2–3, 3–5, 5–10 cM).

File: ibdnet/train/cli_eval.py

Compare to truth (sims) or pedigree expectations.

Produce plots: reliability curve, PR curves, boundary residuals, runtime.

Baselines (optional wrappers under scripts/):

hap-ibd, iLASH, phasedibd—normalize to common BED schema for comparison.

9) Calibration

File: ibdnet/train/calibrate.py

Temperature scaling on validation (optimize scalar T to minimize NLL; apply at inference).

Isotonic regression fallback when non-monotonicity appears.

Store calibration params in checkpoint metadata and honor --calibrated flag at call time.

10) Performance & scaling

PBWT cache per chromosome (reused across all pairs).

Zarr chunking [site] with blosc:zstd level 5.

Tiling: 2–5 Mb windows, 50–100 kb overlap; stitch by weighted averaging in overlaps.

Batch multiple pairs’ tiles to saturate GPU; mixed precision on.

Optional candidate pruning (v2): prepass requires Δrank ≤ τ in ≥m of last w sites; only run CRF on flagged windows.

11) Tests (must pass)

File: tests/test_pbwt.py

PBWT invariants on tiny toy haplotypes; order/div arrays consistent with definitions.

File: tests/test_features.py

Δrank, divergence, ΔcM correctness; missingness handled; schema ordering fixed.

File: tests/test_transitions.py

Monotonicity: p(IBD→non) increases with ΔcM; probabilities normalize.

File: tests/test_crf.py

On synthetic sequences with known states, CRF posteriors near 1.0 on IBD stretches.

File: tests/test_infer_end2end.py

Sim tiny chromosome → PBWT → features → train quick → infer → segment F1@≥3cM ≥ 0.85.

Set up GitHub Actions CI to run pytest -q on Python 3.11.

12) CLI examples (to include in README)

Simulate:

python -m ibdnet.sims.cli_simulate \
  --config configs/sim_human_mix.yaml \
  --out data/sim1/


Train:

python -m ibdnet.train.cli_train \
  --config configs/train_small.yaml \
  --data data/sim1/ \
  --out runs/exp1/


Prep PBWT+features (real data):

python -m ibdnet.infer.cli_prep \
  --vcf cohort.vcf.bgz \
  --map chr22.map \
  --pairs pairs.tsv \
  --workspace workdir/


Call IBD:

python -m ibdnet.infer.cli_call \
  --model runs/exp1/model.pt \
  --config configs/infer_default.yaml \
  --features_zarr workdir/features/ \
  --out calls/

13) File formats

Pairs TSV: pair_id sample_a hap_a sample_b hap_b

Map TSV: chrom pos_bp cm

BED: chrom start end name(pair_id) mean_post min_post len_bp len_cM

Posteriors NPZ: key "{chrom}:{pair_id}" → float32 [L].

14) Coding conventions

Type hints everywhere; docstrings include tensor shapes.

Numba for PBWT hot loops; avoid Python branching in per-site loops.

Masks for missing sites; never drop sites silently.

Reproducibility: if seed set, fix PyTorch/NumPy RNGs; set deterministic cudnn where feasible.

15) Milestones

M1 — PBWT + features (week 1): implement PBWT core, features, schema, tests green.

M2 — Sims + labels (week 1–2): msprime pipeline; truth extraction; small dataset ready.

M3 — CRF prototype (week 2): MLP emissions + transitions + forward/backward; overfit tiny set.

M4 — Train + metrics (week 3): segment F1@≥3cM meets/exceeds hap-IBD on val sims.

M5 — Inference CLI (week 3–4): prep/call end-to-end; calibrated outputs; BED + NPZ.

M6 — Docs + CI (week 4): README, examples, CI passing; tag v0.1.0.

16) Edge cases & guardrails

Unphased or poor phasing: warn; proceed (v1 expects phased; v2 adds phase-uncertain mode).

Sparse/no genetic map: approximate via uniform cM/Mb; log warning in output.

Low-complexity/centromeres: support region masks/blacklists; zero out transitions/emissions where masked.

Short contigs: auto-shrink tile size to fit.

GPU OOM: auto-halve tile length and retry once; fail gracefully with actionable message.

17) Minimal code contracts (for Codex to implement verbatim)
# ibdnet/pbwt/pbwt_core.py
def build_pbwt(haps: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]: ...

# ibdnet/pbwt/features.py
def make_pair_features(..., cfg: PBWTConfig) -> dict[str, np.ndarray]: ...

# ibdnet/model/emissions.py
class MLPEmissions(nn.Module): ...
class TCNEmissions(nn.Module): ...
class TinyTransformerEmissions(nn.Module): ...

# ibdnet/model/transitions.py
def compute_log_transitions(dcm: torch.Tensor, cfg: TransitionConfig) -> torch.Tensor: ...

# ibdnet/model/crf.py
class IBDCRF(nn.Module):
    def forward(self, feats, dcm, mask) -> dict: ...

# ibdnet/infer/postproc.py
def segmentize(post: np.ndarray, cm: np.ndarray, cfg: InferConfig) -> "pd.DataFrame": ...

# ibdnet/train/metrics.py
def segment_f1(pred_bed, true_bed, min_len_cm: float) -> float: ...
def calibration_ece(probs: np.ndarray, labels: np.ndarray, bins:int=15) -> float: ...

18) Documentation

README.md: quickstart, data expectations, CLI examples, interpreting results, caveats.

/docs (optional): PBWT primer, model design, metrics, calibration guidance.

Notebooks (optional): examples/01_sim_train.ipynb, 02_infer_real.ipynb.

19) v2 Roadmap (post v0.1)

Multi-state coalescent HMM (TMRCA bins).

Semi-Markov durations to reduce over-fragmentation.

Candidate windowing via PBWT neighbor prefilter.

GL-aware emissions (consume entropy/dosage).

Query-vs-cohort joint calling → IBD graphs/clusters.

Done. Hand this file to Codex and ask it to:

scaffold the repo per layout,

generate the specified files with the declared contracts, and

implement PBWT core → features → CRF training/inference end-to-end, with tests and CLI.