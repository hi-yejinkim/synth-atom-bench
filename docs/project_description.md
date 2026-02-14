# SynthBench3D: Scaling Laws for 3D Generative Models via Synthetic Benchmarks

## Where This Project Fits

This project is part of SPML's broader research theme: **Building Better Atomic AI Models Through Controlled Experiments**. The core idea is borrowed from NLP (Allen-Zhu's "Physics of Language Models") and CV (Fan et al.'s "Scaling Laws of Synthetic Images"): instead of training on expensive, noisy real-world data and hoping for the best, we design **cheap synthetic tasks with known ground truth** that let us systematically measure what works, what doesn't, and why.

The research theme proposes four strategies:
1. **Synthetic benchmarks** with known ground truth
2. **Consistent labels** to enable scaling analysis
3. **Simulation as a synthetic score function** for evaluation
4. **Transfer learning diagnostics** across domains

**SynthBench3D is our first project under Strategy 1.** It instantiates the simplest possible 3D structure generation task — hard sphere packing — and uses it to measure compute-matched scaling laws across three major architecture families.

## Motivation

When building a 3D foundation model (for molecules, proteins, or materials), you face basic design questions that currently require weeks of GPU time to answer:

- Should you use an equivariant GNN or a Transformer? At what scale does one overtake the other?
- How much does performance improve per 2x compute for each architecture?
- Does an architecture that looks better at small scale still win at large scale?

These questions are hard to answer on real data because real molecular datasets are noisy, expensive, and confounded — you can't isolate *why* one model beats another. SynthBench3D sidesteps this entirely: we generate synthetic data from a distribution with known ground truth, so all variation in model performance is attributable to the model itself.

This also matters for **agentic scientific workflows**. An AI agent iterating on model designs needs fast, clean evaluation signals. An agent that waits weeks per experiment or iterates on noisy benchmarks will make bad decisions. SynthBench3D provides evaluation that runs in minutes to hours, not weeks.

## The Task: Hard Sphere Packing

The task is to sample from the uniform distribution over non-overlapping sphere configurations in a cubic box:

$$p(x_1, \ldots, x_N) \propto \prod_{i<j} \mathbf{1}[|x_i - x_j| > 2r]$$

- **N** atoms of radius **r** in a cubic box of side **L**
- Difficulty is controlled by the **packing fraction** $\eta = N \frac{4}{3}\pi r^3 / L^3$
- Higher $\eta$ means atoms are more tightly packed and the constraint is harder to satisfy

This is the simplest possible 3D structure problem: the *only* challenge is avoiding atomic clashes. There are no bonds, no angles, no charges, no long-range interactions. This isolation is the point — it lets us measure how well each architecture handles geometric exclusion constraints in a clean setting.

**Why this task first?** Future SynthBench tasks will isolate other challenges (bond constraints, symmetry, multimodality, long-range dependencies). Together they form a diagnostic suite that decomposes what makes 3D structure prediction hard. Hard sphere packing is the baseline.

### Difficulty settings

| Setting | N | $\eta$ | Difficulty |
|---------|---|--------|------------|
| `easy_small` | 10 | 0.1 | Easy |
| `medium_small` | 10 | 0.3 | Medium |
| `medium_large` | 50 | 0.3 | Medium-large |
| `hard_small` | 10 | 0.5 | Hard |

Each setting has 50k train / 5k val / 10k test samples generated via MCMC (Metropolis-Hastings) with burn-in and thinning. All atoms have radius r = 0.5.

## Generative Framework: Conditional Flow Matching

All architectures share the **same** generative framework — conditional flow matching (Lipman et al., 2023). Each architecture serves as a velocity network $v_\theta(x_t, t)$ that predicts a velocity field.

- **Interpolation:** $x_t = (1 - t)\varepsilon + t\,x_0$, where $\varepsilon \sim \mathcal{N}(0, I)$
- **Training loss:** $\|v_\theta(x_t, t) - (x_0 - \varepsilon)\|^2$
- **Sampling:** Euler ODE integration from noise ($t=0$) to data ($t=1$)

The same ODE sampler (Euler, same number of steps), same training data, and same augmentation (random SO(3) rotations) are used for all models. **The only variable across experiments is the velocity network architecture.**

## Three Architectures

### 1. PaiNN (Equivariant GNN)

Based on SchNetPack's PaiNN (Schutt et al., 2021). Equivariant message passing with both scalar and vector features per atom.

- Continuous-filter convolutions with Gaussian radial basis functions on pairwise distances
- Each layer: PaiNN interaction (message passing) + PaiNN mixing (intra-atom update)
- Vector features naturally map to velocity output — **equivariant by construction**
- Local: each atom aggregates information from neighbors within a distance cutoff
- Implementation: `PaiNNVelocityNetwork(hidden_dim, n_layers, n_rbf, cutoff)`

### 2. Transformer (DiT-style)

Based on SimpleFold's FoldingDiT blocks (Apple, 2025). Global self-attention over all atoms with adaptive layer normalization.

- Pairwise distances encoded via Gaussian RBFs, projected to per-head attention bias
- Timestep conditioning via adaLN-Zero (zero-initialized so each block starts as identity)
- SwiGLU feed-forward layers, RMS normalization, QK normalization
- **No built-in equivariance** — relies on random rotation augmentation
- Implementation: `TransformerVelocityNetwork(hidden_dim, num_layers, num_heads, num_rbf, cutoff, mlp_ratio)`

### 3. Pairformer (AlphaFold2/Boltz-style)

Based on Boltz's PairformerStack (Wohlwend et al., 2024). Maintains both a single (per-atom) and pair (per-atom-pair) representation.

- Pair representation initialized from pairwise distance features
- **Triangular multiplicative updates** (outgoing + incoming) on pair representation
- Attention on single representation biased by pair representation
- Transition MLPs for both representations
- Implementation: `PairformerVelocityNetwork(hidden_dim, pair_dim, num_layers, num_heads, num_rbf, cutoff, expansion_factor)`

All architectures implement the same interface: `forward(positions: (B, N, 3), t: (B,)) -> (B, N, 3)`.

## The Core Experiment: Compute-Matched Scaling

This is the key deliverable. We measure **scaling curves: metric vs. total training FLOPs** for each architecture.

### Protocol

For each total compute budget C (measured in training FLOPs):
1. For each architecture, sweep model size (xs, small, medium, large, xl) and training steps
2. Constraint: FLOPs_per_step x num_steps <= C
3. Sweep learning rate (1e-4, 1e-3)
4. Report best metric at each budget

**Budgets:** 1e15, 4e15, 1.6e16, 6.4e16, 2.56e17 total training FLOPs.

### Scaling law fit

For each architecture, fit:

$$\text{clash\_rate}(C) = a \cdot C^{-\alpha} + \text{floor}$$

- **$\alpha$ (scaling exponent):** How fast performance improves with compute. Higher = better scaling. **This is the main result.**
- **floor:** Irreducible error. May differ by architecture — reveals fundamental limitations.
- **$a$ (prefactor):** Initial performance level. Less important than $\alpha$ at scale.

### What the results tell us

| Outcome | Interpretation |
|---------|---------------|
| $\alpha_\text{pairformer} > \alpha_\text{gnn} > \alpha_\text{transformer}$ | Pair representations are the right inductive bias for geometric constraints, and this advantage compounds with scale |
| Similar $\alpha$, different floors | Architectures scale similarly but have different fundamental limits |
| Rankings flip between small and large compute | The "best" architecture depends on your budget |
| Any architecture hits floor early | Fundamental bottleneck that more compute can't fix |

### Secondary scaling axes (planned)

- **Data scaling:** Fix model size, vary training set size (1k to 50k). Which architecture is most data-efficient?
- **Problem scaling:** Fix compute, vary N and $\eta$. How does difficulty scaling interact with architecture choice?

## Evaluation Metrics

1. **Clash rate** (primary): Fraction of generated samples with any pairwise distance < 2r. Binary, hard-constraint metric — a sample either has a clash or it doesn't.
2. **g(r) distance** (continuous): L1 distance between the pair correlation function of generated samples and ground truth. Measures distributional quality beyond just clash avoidance.

## Codebase Overview

```
synth-atom-bench/
├── data/               # MCMC sampler + PyTorch dataset
├── models/             # PaiNN, Transformer, Pairformer velocity networks
├── flow_matching/      # Shared interpolation, loss, ODE sampling
├── metrics/            # Clash rate (GPU-accelerated), g(r) distance
├── experiments/        # Training loop (Hydra), evaluation, scaling sweeps
├── viz/                # Publication-quality plots (scaling curves, structures, g(r))
├── configs/            # Hydra configs for models, data, training
├── tests/              # Unit tests for all components
└── scripts/            # Shell scripts for running experiments
```

### Key design principles

- **Fair comparison by construction:** Same flow matching, same sampler, same data, same augmentation. Only the velocity network varies.
- **Compute accounting:** FLOPs measured with torch profiler. Total training FLOPs (not GPU-hours) is the x-axis.
- **Faithful reimplementations:** PaiNN from SchNetPack, Transformer from SimpleFold, Pairformer from Boltz — reimplemented from source, not imported as dependencies.
- **Reproducibility:** Deterministic seeds, checkpointing with resume, Hydra configs for all hyperparameters, W&B logging.

### Running experiments

```bash
# Generate data
uv run data/generate.py --N 10 --eta 0.3 --num-samples 50000 --output outputs/data/N10_eta0.3/train.npz

# Train a model
uv run experiments/train.py model=painn data=medium_small

# Run scaling experiments
uv run experiments/scaling.py generate    # measure FLOPs, print commands
uv run experiments/scaling.py run         # execute the grid
uv run experiments/scaling.py collect     # gather results
uv run experiments/scaling.py fit         # fit scaling laws, generate plots

# Evaluate a checkpoint
uv run experiments/evaluate.py --checkpoint outputs/checkpoints/painn/best.pt --n-samples 10000
```

## Connection to the Broader Vision

SynthBench3D is a **proof of concept** for the synthetic benchmark strategy. If we can extract clean scaling laws from hard sphere packing, the methodology extends to progressively harder synthetic tasks:

| Task | What it isolates |
|------|-----------------|
| Hard sphere packing (this project) | Geometric exclusion / clash avoidance |
| Bond-constrained packing | Fixed-distance constraints |
| Symmetric structures | Discrete symmetry handling |
| Multi-modal distributions | Mode coverage and diversity |
| Long-range correlations | Information propagation across large systems |

Each task is cheap, deterministic, and tests one capability at a time. Together they form a diagnostic suite — a "Physics of 3D Generative Models" — that tells practitioners which architecture to use for which problem, and how much compute they need.

## Key References

- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023) — generative framework
- Schutt et al., "Equivariant message passing for the prediction of tensorial properties and molecular spectra" (ICML 2021) — PaiNN
- Apple, "SimpleFold: Folding Proteins is Simpler than You Think" (2025) — Transformer baseline
- Wohlwend et al., "Boltz-1: Democratizing Biomolecular Interaction Modeling" (2024) — Pairformer
- Allen-Zhu, "Physics of Language Models" series — methodological inspiration
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla) — scaling law template
- Fan et al., "Scaling Laws of Synthetic Images for Model Training" (CVPR 2024) — synthetic scaling precedent
