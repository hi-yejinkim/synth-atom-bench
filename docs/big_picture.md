# Research Theme: Building Better Atomic AI Models Through Controlled Experiments

## The Problem

We're building generative models for molecules, proteins, and materials, and every design decision — architecture, training data, loss function, noise schedule — is a guess. We try something, run an expensive experiment, check the benchmark, iterate. This is slow, wasteful, and teaches us very little about *why* something worked.

Questions that currently require weeks of GPU time per experiment:

- Transformer or equivariant GNN? At what scale does one overtake the other?
- 120M crystal structures from three databases. What mixing ratio? Does the order matter?
- Model fails on unseen ternary compounds. More data, bigger model, or different architecture?
- Does pre-training on bulk crystals help when fine-tuning on surface-adsorbate complexes?
- Flow matching or diffusion? At what system size does one become better?

The answers are specific to the exact setup and don't transfer. And this problem gets worse as AI agents enter scientific discovery workflows. An agent iterating on model designs needs fast, reliable evaluation signals — not two-week training runs and noisy benchmarks where variance masks the signal. Fast, clean, controlled evaluation is infrastructure for the agentic future of atomic AI.

## Precedent in NLP and CV

The NLP and CV communities have already converged on the same solution: **controlled synthetic data as a tool for understanding and improving models**.

**NLP.** Allen-Zhu's "Physics of Language Models" series designed synthetic tasks with known ground truth and discovered that specific architectural modifications (Canon layers) are necessary for certain reasoning capabilities — proved with small, cheap experiments that predict large-model behavior. Microsoft's Phi series (TinyStories, "Textbooks Are All You Need") showed that carefully designed training data lets 10M-parameter models match behaviors of models 100x larger.

**CV.** "Scaling Laws of Synthetic Images" (Fan et al., CVPR 2024) studied how synthetically generated images scale for training vision models, identifying factors (prompt design, guidance scale) that determine whether synthetic data helps or hurts. The sim-to-real community in robotics has long operated on the principle of pre-training on consistent synthetic data and fine-tuning on noisy real data.

The frontier is shifting from "train on everything and hope" to "understand what data teaches models, then train deliberately." For atomic and molecular AI, nobody has done this yet.

## The Proposal: Four Strategies

### Strategy 1: Synthetic Benchmarks with Known Ground Truth

Build synthetic atomic datasets where you control exactly what the model needs to learn. Vary architecture, data, and training recipe to find what works.

The practical outputs are engineering results — they tell you what to build and what not to waste time on:

- "Equivariant architectures learn bond angle constraints with 10x less data than Transformers" -- tells you when to use which
- "Compositional generalization to new elements requires at least N training elements with M examples each" -- tells you how much data you need
- "Flow matching decides global topology by t=0.3 and local geometry after t=0.7" -- tells you where to focus architectural capacity
- "This architecture fails at multi-hop geometric reasoning regardless of scale" -- tells you to fix the architecture, not add more data

These benchmarks are cheap (minutes to hours, not weeks), deterministic (no label noise confounding results), and modular (test one capability at a time). Ideal evaluation targets for agentic workflows: an agent proposes an architectural change, tests it against a battery of synthetic tasks, and gets a clean signal within a single GPU-hour.

**SynthBench3D (this project) is our first instantiation of Strategy 1.** See [project_description.md](project_description.md) for details.

### Strategy 2: Consistent Labels to Enable Scaling Analysis

We can't currently measure scaling laws for atomic AI because training data is too noisy. PDB labels mix wildly different resolution levels. Materials datasets use different DFT setups. This label inconsistency creates artificial plateaus that look like fundamental limits but are actually just noise.

The fix: create datasets where all labels come from one consistent source. For proteins, relax everything through one force field. For crystals, relax with MLIP or one DFT functional. The labels will be biased, but the bias is uniform — and that's what matters for measuring scaling. This is exactly the CV sim-to-real playbook: synthetic data with consistent labels for learning, real data for fine-tuning.

**Practical outputs:**

- Clean scaling curves: is the field data-limited or model-limited?
- Quantitative answers: how much does performance improve per 2x data / 2x parameters / 2x compute?
- A principled pre-training recipe: consistent labels for clean learning, noisy real data for fine-tuning
- Direct comparison of scaling exponents across architectures -- pick the one that scales best *before* you commit to an expensive training run

### Strategy 3: Simulation as a Synthetic Score Function

Strategies 1 and 2 address the training side. Evaluation has the same problem. Ground-truth assessment (DFT, wet-lab synthesis) costs hours to months per sample, fundamentally limiting iteration speed.

Build synthetic score functions: fast physics simulations (force field relaxation, short MD runs, MLIP energy evaluation) as cheap proxies for expensive ground truth. Does the generated MOF maintain its pore structure after relaxation? Does the catalyst surface stay stable? Does the protein-ligand complex hold together for 10ns of MD?

The key research question: how well does the cheap proxy track the expensive ground truth, and under what conditions does it break down?

**Practical outputs:**

- Standardized simulation-based score functions covering stability, strain, binding persistence, pore integrity
- Quantitative correlation between synthetic scores and expensive ground truth — showing when fast evaluation is trustworthy
- Identification of failure modes: which generated structures fool the fast proxy but fail under rigorous evaluation
- Plug-and-play evaluation modules for tight propose -> generate -> score -> iterate loops in seconds

### Strategy 4: Transfer Learning Diagnostics

When does pre-training on one domain actually help another? Bulk crystals -> surface-adsorbate complexes? Molecular data -> MOFs? Protein backbone data -> protein-ligand docking? Right now people just try it and see — an expensive guess each time.

Design controlled experiments that systematically measure transfer: train on domain A, fine-tune on domain B, measure how much A helped as a function of domain similarity, data size, and model capacity.

**Practical outputs:**

- A transfer map: "pre-training on X is worth Y equivalent labeled examples in domain Z"
- Identification of which features transfer (local geometry transfers; periodic boundary conditions don't) and which interfere
- A decision framework for allocating training budget between pre-training and fine-tuning data

## What This Is Not

This is **not interpretability**. Interpretability asks "what did this model learn?" and tries to reverse-engineer internal representations — results are often qualitative, hard to reproduce, and specific to one model.

We're asking **"what should we build and how should we train it?"** The outputs are quantitative recipes and design principles. The experiments are reproducible by design (synthetic data, controlled setup). And because we're probing fundamental properties of learning algorithms + data, the findings generalize across specific models and datasets.

Allen-Zhu didn't win attention by explaining GPT-2's internals. He showed that certain architectural choices are provably necessary for certain capabilities, and that you can predict this from small experiments. Microsoft's Phi team showed that the right synthetic data makes small models surprisingly capable. We want to bring this same methodology to atomic generative models.

## How SynthBench3D Fits In

SynthBench3D is a proof of concept for Strategy 1. It instantiates the simplest possible 3D structure generation task — hard sphere packing — and uses it to measure compute-matched scaling laws across three architecture families (PaiNN, Transformer, Pairformer).

If this works, the methodology extends to progressively harder synthetic tasks:

| Task | What it isolates |
|------|-----------------|
| Hard sphere packing (this project) | Geometric exclusion / clash avoidance |
| Chain/bond-constrained packing | Fixed-distance constraints |
| Symmetric structures | Discrete symmetry handling |
| Multi-modal distributions | Mode coverage and diversity |
| Long-range correlations | Information propagation across large systems |

Each task is cheap, deterministic, and tests one capability at a time. Together they form a diagnostic suite — a "Physics of 3D Generative Models" — that tells practitioners which architecture to use for which problem, and how much compute they need.

## References

- Allen-Zhu, "Physics of Language Models" series — physics.allen-zhu.com
- Eldan & Li, "TinyStories" / Gunasekar et al., "Textbooks Are All You Need" (Microsoft Phi series)
- Fan et al., "Scaling Laws of Synthetic Images for Model Training" (CVPR 2024)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla scaling laws)
- MLIP scaling studies (MatPES, OMat24) — consistent oracles enable clean scaling analysis in the atomic domain
