# IBDNet

Inference of identity-by-descent (IBD) tracts using PBWT-derived features and a neural conditional random field.

## Project Goals

IBDNet implements a modular pipeline for preparing phased genotype data, extracting positional Burrows–Wheeler transform (PBWT) features, training neural emission models, and performing inference with a CRF whose transitions depend on genetic map distance.

## Key Components

- **PBWT + Features**: Efficient PBWT sweeps with neighbor rank caching and feature extraction for haplotype pairs.
- **Neural Emissions**: Configurable emission networks (MLP, temporal convolution, tiny transformer) implemented in PyTorch.
- **Distance-aware Transitions**: CRF transitions parameterised by map distances with optional learnable intensity.
- **Simulation + Training Pipelines**: msprime-based simulation CLI, training loop with metrics and calibration utilities.
- **Inference CLI**: Prepare PBWT workspaces, call IBD segments, and post-process outputs into BED/NPZ summaries.

## Environment Setup

```bash
uv venv .venv
UV_PROJECT_ENVIRONMENT=.venv uv pip install -e . --extra dev
```

## Quickstart

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run --extra dev python -m ibdnet.sims.cli_simulate --config configs/sim_human_mix.yaml --out data/sim1/
UV_PROJECT_ENVIRONMENT=.venv uv run --extra dev python -m ibdnet.train.cli_train --config configs/train_small.yaml --data data/sim1/ --out runs/exp1/
UV_PROJECT_ENVIRONMENT=.venv uv run --extra dev python -m ibdnet.infer.cli_prep --vcf cohort.vcf.gz --map chr22.map --pairs pairs.tsv --workspace workdir/
UV_PROJECT_ENVIRONMENT=.venv uv run --extra dev python -m ibdnet.infer.cli_call --model runs/exp1/model.pt --config configs/infer_default.yaml --features_zarr workdir/features/ --out calls/
```

See `ibdnet_IMPLEMENTATION_PLAN.md` for the detailed architecture agreed upon during design planning.
