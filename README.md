# SynthBench3D

**Scaling laws for 3D generative models via synthetic benchmarks.** Compare GNN, Transformer, and Pairformer architectures on controlled geometric tasks to predict which will scale best for 3D structure foundation models.

## Why synthetic benchmarks?

Real molecular data is expensive and confounded — you can't isolate *why* one model beats another. SynthBench3D builds synthetic tasks with known ground truth where the only challenge is a single geometric constraint. By measuring how each architecture's performance improves with compute, we extract **scaling exponents** that predict which architectures will dominate at foundation-model scale — actionable information you can't get from standard benchmarks.

<p align="center">
  <img src="docs/assets/hero_structures.png" width="95%" alt="Hard sphere and chain configurations at increasing atom counts">
</p>

## Tasks

Each task isolates a specific capability that 3D generative models need. Together they form a diagnostic suite that decomposes what makes 3D structure prediction hard.

### Hard Sphere Packing

Sample non-overlapping sphere configurations in a cubic box. The only constraint is **steric exclusion**: no two atoms can overlap (|x_i - x_j| > 2r). This tests a model's ability to learn **pairwise distance constraints** — the most fundamental geometric challenge. Difficulty is controlled by packing fraction (density) and atom count.

**Metric**: clash rate (fraction of generated samples with any overlap).

### Self-Avoiding Chains

Sample self-avoiding polymer chains: atoms connected by fixed-length bonds that must not self-intersect. This adds **sequential bonded constraints** on top of clash avoidance — the model must learn valid chain topology where consecutive atoms maintain bond lengths while non-bonded atoms avoid overlap. This isolates the challenge of generating structures with **connectivity and long-range self-avoidance**. Difficulty scales with chain length N, since longer chains are exponentially harder to fold without self-intersection.

**Metrics**: clash rate + bond length violation.

## Key Result: Compute Scaling Laws

For each architecture, we sweep model size and training steps under a fixed compute budget, then fit a power law: `clash_rate(C) = a × C^(-α) + floor`. The scaling exponent **α** tells you how fast performance improves with compute.

<p align="center">
  <img src="docs/assets/scaling_curves.png" width="65%" alt="Compute scaling curves">
</p>

## Architectures

| Architecture | Type | Equivariant? | Reference |
|---|---|---|---|
| **PaiNN** | Equivariant GNN | Yes | [Schütt et al., 2021](https://arxiv.org/abs/2102.03150) |
| **Transformer** | Global attention | No (augmentation) | [SimpleFold (Apple, 2025)](https://arxiv.org/abs/2503.11533) |
| **Pairformer** | Pair + triangle updates | No (augmentation) | [Boltz (Wohlwend et al., 2024)](https://arxiv.org/abs/2408.00778) |

All architectures share the same **conditional flow matching** framework — the only variable is the velocity network. Same training data, same ODE sampler, same evaluation.

## Quick Start

```bash
# Install dependencies
uv sync

# Generate training data (hard spheres: N=10, η=0.3)
uv run data/generate.py --N 10 --eta 0.3 --radius 0.5 \
    --num_samples 50000 --output outputs/data/N10_eta0.3/train.npz

# Generate training data (chains: N=20)
uv run data/generate_chains.py --N 20 --num_samples 50000 \
    --output outputs/data/chain_N20/train.npz

# Train a PaiNN model
uv run experiments/train.py model=painn data=default training.max_steps=50000

# Evaluate (generate samples + compute clash rate)
uv run experiments/evaluate.py --checkpoint outputs/checkpoints/painn/best.pt \
    --arch painn --num_samples 10000

# Run compute-matched scaling experiment
uv run experiments/scaling.py run --arch painn --budgets 1e15 4e15 1.6e16
```

> Regenerate README figures: `uv run docs/assets/generate_readme_figures.py`

## Project Structure

```
├── data/               # MCMC samplers + PyTorch dataset
├── models/             # PaiNN, Transformer, Pairformer velocity networks
├── flow_matching/      # Shared interpolation, training loss, ODE sampling
├── metrics/            # Clash rate, bond violation, g(r) distance
├── experiments/        # Training, evaluation, scaling sweeps
├── viz/                # Publication-quality plotting
├── configs/            # Hydra configuration files
└── outputs/            # All generated artifacts (gitignored)
```

## Documentation

- [Research big picture](docs/big_picture.md) — why synthetic benchmarks, the broader research program
- [Project description](docs/project_description.md) — detailed task specification and methodology

## Tech Stack

Python · PyTorch · Hydra · W&B · uv
