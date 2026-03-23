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

**Core idea**: Three velocity network architectures (PaiNN, Transformer, Pairformer) share the same conditional flow matching framework (`flow_matching/`). The only variable is the architecture — same data, same sampler, same augmentation, same evaluation.

**Task system**: Tasks are registered in `experiments/task_registry.py` with complexity levels 1-7. The unified 6-rule system (`data/generate_unified.py`) provides progressive difficulty via independently toggleable rules. Each task type has its own generator (`data/generate*.py`), dataset class (`data/dataset.py`), and metrics module (`metrics/`).

**Scaling orchestration**: `experiments/chinchilla.py` runs Chinchilla-style scaling law measurements with Approach 1 (IsoFLOP envelope) and Approach 3 (parametric fit L(N,D) = E + A/N^α + B/D^β). Model size presets (chinchilla_0-13, spanning 1K-100M params) are defined in `experiments/model_registry.py`.

**Config**: Hydra-based (`configs/`). Data configs in `configs/data/` (20+ task definitions), model configs in `configs/model/`, training config in `configs/train.yaml`.

**Visualization**: All plots use `synthbench_style()` context manager from `viz/style.py`.

---

## Project: SynthBench3D — Hard Sphere Packing Benchmark

### Big Picture

We want to discover **scaling laws for 3D generative models** — how does performance improve as you increase compute, data, and model size? The end goal is to guide architecture selection for 3D structure foundation models (molecules, proteins, materials).

Real molecular data is expensive and confounded — you can't isolate why one model beats another. So we build **synthetic tasks with known ground truth** where we can run controlled scaling experiments cheaply.

Hard sphere packing is the first task: the simplest possible 3D structure problem where the only challenge is avoiding atomic clashes. Future tasks will isolate other challenges (bond constraints, symmetry, multimodality, long-range dependencies). Together they form a diagnostic suite that decomposes what makes 3D structure prediction hard.

The key deliverable is: **for each architecture family, a scaling exponent that predicts how performance improves with compute.** If architecture A has a better scaling exponent than B on clash avoidance, that means A will increasingly dominate as foundation models scale up — even if B looks better at small scale. This is actionable information for anyone building 3D foundation models.

### Phase 1 Scope

Compare GNN, Transformer, and Pairformer on generating non-overlapping atom configurations. The only difficulty is the clash constraint.

## Problem

Sample from the uniform distribution over non-overlapping sphere configurations:

```
p(x_1, ..., x_N) ∝ ∏_{i<j} 𝟙[|x_i - x_j| > 2r]
```

N atoms with radius r in a cubic box of side L. Difficulty controlled by packing fraction η = N(4/3)πr³/L³.

## Data Generation

MCMC (Metropolis-Hastings) sampler:
1. Initialize by sequential random placement with rejection
2. Propose single-atom displacements, accept if no overlap
3. Collect samples after burn-in, thin to reduce autocorrelation
4. Save as .npz with positions (N×3), radius r, box size L

Generate 50k train / 5k val / 10k test samples for each setting:
- N=10, η=0.1 (easy)
- N=10, η=0.3 (medium)
- N=50, η=0.3 (medium-large)
- N=10, η=0.5 (hard)

## Generative Framework

Conditional flow matching (Lipman et al., 2023), shared across all architectures. Each architecture is a velocity network v_θ(x_t, t) → predicted velocity field.

Interpolation: x_t = (1 - t) ε + t x_0, where ε ~ N(0, I), t ∈ [0, 1]
Loss: ||v_θ(x_t, t) - (x_0 - ε)||²
Sampling: ODE integration from x_0 ~ N(0, I) to x_1 using Euler method with fixed number of steps (same for all models)

## Architectures

All architectures take atom positions x_t and timestep t as input, output predicted velocity of same shape (N×3).

**GNN (PaiNN)**
- Equivariant message passing with both scalar and vector features per atom
- Continuous-filter convolutions with radial basis functions on pairwise distances
- Vector features naturally map to velocity output (equivariant by construction)
- K message passing layers
- Local: each atom aggregates info from neighbors within cutoff
- Reference implementation: SchNetPack (https://github.com/atomistic-machine-learning/schnetpack)
  - Extract PaiNN representation from `schnetpack.representation` — reimplement faithfully based on their code
  - Reimplement as velocity network: add timestep embedding, read out velocity from vector features
  - Paper: "Equivariant message passing for the prediction of tensorial properties and molecular spectra" (Schütt et al., 2021)

**Transformer**
- Global self-attention over all atoms
- Pairwise distance features injected as attention bias
- No built-in equivariance, use random rotation augmentation
- Sinusoidal timestep embedding added to atom features via adaptive layer norm
- Reference implementation: SimpleFold (https://github.com/apple/ml-simplefold)
  - Uses standard transformer blocks with adaptive layers + flow matching — exactly our setup
  - Extract the FoldingDiT transformer blocks from `simplefold.model`
  - Already uses flow matching, so the integration is natural
  - Paper: "SimpleFold: Folding Proteins is Simpler than You Think" (Apple, 2025)

**Pairformer (AlphaFold2/Boltz-style)**
- Single representation (per-atom) + pair representation (per-atom-pair)
- Pair representation initialized from pairwise distance features
- Triangular multiplicative updates on pair representation
- Attention on single representation weighted by pair representation
- Reference implementation: Boltz (https://github.com/jwohlwend/boltz)
  - Extract PairformerStack from `boltz.model`
  - Well-tested against AlphaFold3 architecture
  - More complex codebase — extract only the Pairformer module, not the full pipeline
  - Paper: "Boltz-1: Democratizing Biomolecular Interaction Modeling" (Wohlwend et al., 2024)

## Metric

**Clash rate**: fraction of generated samples with any pairwise distance < 2r.

```python
def clash_rate(positions, radius):
    # positions: (batch, N, 3)
    dists = torch.cdist(positions, positions)  # (batch, N, N)
    mask = ~torch.eye(N, dtype=bool)  # exclude self
    min_dists = dists[:, mask].reshape(batch, -1).min(dim=1).values
    return (min_dists < 2 * radius).float().mean()
```

Generate 10k samples per model, report clash rate.

## Comparison: Compute-Matched Scaling

This is the core experiment. We want scaling curves: clash_rate vs. compute for each architecture.

For each total compute budget C (measured in total training FLOPs):
1. For each architecture, sweep model size (width, depth) and training steps
2. Constraint: FLOPs_per_step × num_steps ≤ C
3. Tune learning rate (2 trials: 1e-4, 1e-3)
4. Report best clash rate at each budget

Budgets (total training FLOPs): 1e15, 4e15, 1.6e16, 6.4e16, 2.56e17.

Fit scaling law per architecture:

```
clash_rate(C) = a × C^(-α) + floor      (C = total training FLOPs)
```

- **α** (scaling exponent): how fast performance improves with compute. Higher = better scaling. This is the main result.
- **floor**: irreducible clash rate. May differ by architecture — reveals fundamental limitations.
- **a** (prefactor): initial performance. Less important than α at scale.

### What to look for

- If α_pairformer > α_gnn > α_transformer: pair representations are the right inductive bias for geometric constraints, and this advantage compounds with scale.
- If α values are similar but floors differ: architectures scale similarly but have different fundamental limits.
- If rankings flip between small and large compute: the "best" architecture depends on your budget — critical for practitioners.
- If any architecture hits floor early: it has a fundamental bottleneck that more compute can't fix.

### Secondary scaling axes (run after main experiment)

- **Data scaling**: fix model size, vary training set size (1k, 5k, 10k, 50k). Which architecture is most data-efficient?
- **Problem scaling**: fix compute, vary N (10, 20, 50) and η (0.1, 0.3, 0.5). How does difficulty scaling interact with architecture choice?

## Project Structure

```
├── CLAUDE.md
├── configs/                    # Hydra configs
│   ├── config.yaml
│   ├── train.yaml
│   ├── sweep.yaml
│   ├── data/
│   ├── model/
│   │   ├── painn.yaml
│   │   ├── transformer.yaml
│   │   └── pairformer.yaml
│   └── logging/
├── data/
│   ├── generate.py             # MCMC hard sphere sampler
│   ├── dataset.py              # PyTorch dataset
│   └── validate.py             # Check g(r) of generated data
├── models/
│   ├── painn.py                # PaiNN velocity network from SchNetPack
│   ├── transformer.py          # Transformer velocity network from SimpleFold
│   ├── pairformer.py           # Pairformer velocity network from Boltz
│   └── common.py               # Shared: timestep embedding
├── flow_matching/
│   ├── interpolation.py
│   ├── training.py
│   └── sampling.py
├── metrics/
│   └── clash_rate.py
├── viz/
│   ├── style.py                # Global style: fonts, colors, save_figure
│   ├── structure.py            # 3D atom structure plots
│   ├── metrics.py              # g(r) and min distance histogram
│   ├── scaling.py              # Scaling curves and capability heatmap
│   ├── training.py             # Training loss/clash rate curves
│   └── examples/
│       └── generate_examples.py  # Visual QA script
├── experiments/
│   ├── train.py                # Hydra-based training loop
│   ├── evaluate.py             # Generate samples + compute clash rate
│   ├── scaling.py              # Compute-matched scaling sweep
│   ├── sweep_hparams.py        # Hyperparameter sweep orchestrator
│   ├── model_registry.py       # Shared model registry and size presets
│   ├── logger.py               # W&B logging wrapper
│   └── checkpointing.py        # Checkpoint management
├── scripts/
│   ├── run_scaling.sh
│   ├── run_sweep.sh
│   └── validate_painn.py
└── tests/
    ├── test_data.py
    ├── test_models.py
    ├── test_flow_matching.py
    └── test_metrics.py
```

## Implementation Order

1. `data/generate.py` — MCMC sampler, validate with pair correlation function
2. `data/dataset.py` — PyTorch dataset loading .npz files
3. `metrics/clash_rate.py` — GPU-accelerated clash rate computation
4. `flow_matching/` — shared interpolation, loss, ODE sampler
5. `models/gnn.py` — reimplement SchNetPack PaiNN as velocity network
6. `models/transformer.py` — reimplement SimpleFold transformer blocks as velocity network
7. `models/pairformer.py` — reimplement Boltz PairformerStack as velocity network
8. `experiments/train.py` — training loop with Hydra configs
9. `experiments/evaluate.py` — generate samples + compute clash rate
10. `experiments/scaling.py` — compute-matched sweep

## Tech Stack

- PyTorch
- Hydra for configs
- wandb for logging
- numpy for data generation
- Package manager: always use `uv` (never pip)
  - Install packages: `uv add <package>`
  - Run scripts: `uv run <script>`
  - Sync environment: `uv sync`

## W&B Setup

- W&B login token is sourced from `~/.zshrc` (environment variable)
- No need to run `wandb login` manually — token is available in the shell environment

## Output Directory Convention

All generated artifacts go under `outputs/`, never mixed with source code:

```
outputs/
├── data/{N}_{eta}/          # Generated .npz datasets (e.g. N10_eta0.1/)
├── plots/                   # All visualizations and figures
├── checkpoints/{arch}/      # Model weights (gnn/, transformer/, pairformer/)
├── logs/{arch}/             # Training logs
├── eval/{arch}/             # Evaluation results (generated samples + metrics)
├── scaling/                 # Scaling law sweep results
└── experiment_logs/         # Persistent records of completed experiments
```

Rules:
- **Never write files to source directories** (`data/`, `metrics/`, `models/`, etc.). All outputs (data, plots, checkpoints, logs) go under `outputs/`.
- **Always use `--output` flags** pointing into `outputs/` when running scripts. Example: `python data/generate.py --output outputs/data/N10_eta0.1/train.npz`
- **Clean up after test/debug runs.** If you generate temporary files for testing (e.g. small sample counts, scratch plots), delete them when done. Do not leave behind files named `test_*`, `tmp_*`, `debug_*`, or similar in `outputs/`.
- **No stale checkpoints.** When a training run is superseded or was a failed experiment, remove its checkpoint directory rather than leaving dead weights around.
- **Name files descriptively.** Use the pattern `{split}_{setting}.npz` for data (e.g. `train.npz`, `val.npz`, `test.npz`) and `{description}.png` for plots. Never use generic names like `output.npz` or `plot.png`.
- **The `outputs/` directory is gitignored.** It must never be committed. If a result needs to be preserved, export it to a report or wandb.

## Key Design Decisions

- All models share the same flow matching framework — the only variable is the velocity network architecture
- Same ODE sampler (Euler, same steps) for all models at evaluation
- Same training data, same augmentation (random rotations for all)
- FLOPs measured with torch profiler for fair compute matching — total training FLOPs (not GPU-hours) is the x-axis for all scaling curves
- Use established reference implementations (SchNetPack, SimpleFold, Boltz) — reimplement faithfully based on their code rather than importing as dependencies, adding only timestep embedding + output projection
- All visualization uses the `viz/` package with `synthbench_style()` context manager for consistent publication-quality plots
