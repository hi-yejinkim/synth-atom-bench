# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Package management (always uv, never pip)
uv sync                          # Install/sync dependencies
uv add <package>                 # Add a dependency
uv run <script>                  # Run any script

# Tests
uv run pytest tests/             # Run all tests
uv run pytest tests/test_models.py             # Single test file
uv run pytest tests/test_models.py::test_painn # Single test function

# Training (Hydra config — override via CLI args)
uv run experiments/train.py model=painn data=medium_small training.max_steps=50000
uv run experiments/train.py model=transformer data=chain_N20 model.num_layers=6

# Data generation
uv run data/generate.py --N 10 --eta 0.3 --radius 0.5 --num_samples 50000 --output outputs/data/N10_eta0.3/train.npz
uv run data/generate_chains.py --N 20 --num_samples 50000 --output outputs/data/chain_N20/train.npz
uv run data/generate_unified.py --rules 1,2,3 --N_backbone 10 --n_samples 50000 --output outputs/data/unified_R123_sp3_N10/train.npz

# Chinchilla scaling experiments (subcommand workflow: generate → run → collect → fit → plot)
uv run experiments/chinchilla.py generate --tasks sphere_easy --archs painn,transformer,pairformer
uv run experiments/chinchilla.py run --tasks sphere_easy --n_gpus 4
uv run experiments/chinchilla.py collect --tasks sphere_easy
uv run experiments/chinchilla.py fit --tasks sphere_easy
uv run experiments/chinchilla.py plot --tasks sphere_easy

# Evaluation
uv run experiments/evaluate.py --checkpoint outputs/checkpoints/painn/best.pt --arch painn --num_samples 10000
```

## High-Level Architecture

**SynthBench3D** studies scaling laws for 3D generative models using synthetic tasks with known ground truth. Three velocity network architectures (PaiNN, Transformer, Pairformer) share the same conditional flow matching framework — the only variable is the architecture. Same data, same sampler, same augmentation, same evaluation.

### Core Modules

- **`flow_matching/`** — Shared conditional flow matching: interpolation (x_t = (1-t)ε + t·x_0), training loss, ODE sampling (Euler), and post-hoc relaxation. All architectures plug into this identically.

- **`models/`** — Three velocity networks, each reimplemented faithfully from reference codebases (not imported as deps):
  - `painn.py` — Equivariant GNN (from SchNetPack). Local message passing with scalar+vector features.
  - `transformer.py` — Global self-attention with distance bias + AdaLN-Zero (from SimpleFold).
  - `pairformer.py` — Single + pair representations with triangular updates (from Boltz/AlphaFold3).
  - `common.py` — Shared timestep embedding, RBF features.

- **`data/`** — Task-specific generators and PyTorch datasets. Each task type has its own `generate_*.py` and `*_dataset.py`. Key generators:
  - `generate.py` — Hard sphere packing (MCMC Metropolis-Hastings)
  - `generate_chains.py` — Chain molecules with bond constraints
  - `generate_unified.py` — Unified 6-rule system (R1: VSEPR slots, R2: angles, R3: bonds, R4: torsions, R5: global pairs, R6: periodicity). Rules are independently toggleable for progressive difficulty.
  - `generate_nbody.py` — N-body energy distributions at various temperatures

- **`metrics/`** — Task-specific violation metrics. Each task has its own metrics module (`clash_rate.py`, `bond_violation.py`, `unified_metrics.py`, `wasserstein_distance.py`, etc.). Unified tasks track per-rule violation rates, not aggregated scores.

- **`experiments/`** — Training orchestration and scaling experiments:
  - `train.py` — Hydra-based training loop
  - `evaluate.py` — Sample generation + metric computation
  - `chinchilla.py` — CLI entry point for 5-subcommand Chinchilla pipeline
  - `chinchilla_lib/` — Modular Chinchilla pipeline (`config.py`, `generate.py`, `run.py`, `collect.py`, `fit.py`, `plot.py`, `helpers.py`)
  - `task_registry.py` — Task definitions with complexity levels 1-7
  - `model_registry.py` — Model size presets: `chinchilla_0` to `chinchilla_13` (1K–100M params)

- **`configs/`** — Hydra configuration. Data task configs in `configs/data/` (35+ tasks), model configs in `configs/model/`, training params in `configs/train.yaml`.

- **`viz/`** — Publication-quality plotting. All plots use `synthbench_style()` context manager from `viz/style.py`.

### Chinchilla Scaling Pipeline

The primary experiment workflow. Uses Approach 1 (IsoFLOP envelope) and Approach 3 (parametric fit L(N,D) = E + A/N^α + B/D^β):
- Model size presets: `chinchilla_0`–`chinchilla_9` (1K–11M), `chinchilla_11`/`chinchilla_13` (25M–100M)
- Data budgets D1–D4: 50K, 100K, 500K, 1M samples (exactly 1 epoch per budget)
- LR sweep: 1e-5, 1e-4, 1e-3 (best selected at collect time)
- FLOPs measured via `torch.utils.flop_counter` for fair compute matching

## Key Design Decisions

- All models share the same flow matching framework — only the velocity network differs
- Same ODE sampler (Euler, same steps) for all models at evaluation
- Random rotation augmentation applied to all models (even equivariant ones)
- FLOPs (not GPU-hours) is the x-axis for all scaling curves
- Reference implementations reimplemented faithfully rather than imported as dependencies
- Unified datasets are origin-centered; `train.py` skips box_size/2 shift for them
- Per-rule metrics for unified tasks (aggregation invalid for cross-rule comparison)

## Output Directory Convention

All generated artifacts go under `outputs/` (gitignored), never mixed with source code:

```
outputs/
├── data/                    # .npz datasets
├── checkpoints/{arch}/      # Model weights
├── logs/{arch}/             # Training logs
├── plots/                   # All visualizations
├── eval/{arch}/             # Evaluation results + samples
├── scaling/                 # Chinchilla sweep results (grid_meta.json, results.json, fits.json)
└── experiment_logs/         # Persistent records
```

Never write files to source directories. Always use `--output` flags pointing into `outputs/`.
