# CLAUDE.md

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

**GNN (SchNet-style)**
- Continuous-filter convolution with radial basis functions on pairwise distances
- K message passing layers
- Local: each atom aggregates info from neighbors within cutoff

**Transformer**
- Global self-attention over all atoms
- Pairwise distance features injected as attention bias
- No built-in equivariance, use random rotation augmentation
- Sinusoidal timestep embedding added to atom features

**Pairformer (AlphaFold2-style)**
- Single representation (per-atom) + pair representation (per-atom-pair)
- Pair representation initialized from pairwise distance features
- Triangular multiplicative updates on pair representation
- Attention on single representation weighted by pair representation

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

For each total compute budget C (measured in GPU-hours):
1. For each architecture, sweep model size (width, depth) and training steps
2. Constraint: FLOPs_per_step × num_steps ≤ C
3. Tune learning rate (5 trials, log-uniform in [1e-5, 1e-3])
4. Report best clash rate at each budget

Budgets: 1, 4, 16, 64, 256 GPU-hours.

Fit scaling law per architecture:

```
clash_rate(C) = a × C^(-α) + floor
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
synthbench3d/
├── CLAUDE.md
├── configs/                    # Hydra configs
│   ├── data/
│   ├── model/
│   │   ├── gnn.yaml
│   │   ├── transformer.yaml
│   │   └── pairformer.yaml
│   └── experiment/
├── data/
│   ├── generate.py             # MCMC hard sphere sampler
│   ├── dataset.py              # PyTorch dataset
│   └── validate.py             # Check g(r) of generated data
├── models/
│   ├── gnn.py
│   ├── transformer.py
│   ├── pairformer.py
│   └── common.py               # Shared layers
├── flow_matching/
│   ├── interpolation.py
│   ├── training.py
│   └── sampling.py
├── metrics/
│   └── clash_rate.py
├── experiments/
│   ├── train.py
│   ├── evaluate.py
│   └── scaling.py              # Run scaling law sweep
└── tests/
    ├── test_data.py
    └── test_models.py
```

## Implementation Order

1. `data/generate.py` — MCMC sampler, validate with pair correlation function
2. `data/dataset.py` — PyTorch dataset loading .npz files
3. `metrics/clash_rate.py` — GPU-accelerated clash rate computation
4. `flow_matching/` — shared interpolation, loss, ODE sampler
5. `models/gnn.py` — SchNet denoiser
6. `models/transformer.py` — transformer denoiser
7. `models/pairformer.py` — pairformer denoiser
8. `experiments/train.py` — training loop with Hydra configs
9. `experiments/evaluate.py` — generate samples + compute clash rate
10. `experiments/scaling.py` — compute-matched sweep

## Tech Stack

- PyTorch
- Hydra for configs
- wandb for logging
- numpy for data generation

## Key Design Decisions

- All models share the same flow matching framework — the only variable is the velocity network architecture
- Same ODE sampler (Euler, same steps) for all models at evaluation
- Same training data, same augmentation (random rotations for all)
- FLOPs measured with torch profiler for fair compute matching
