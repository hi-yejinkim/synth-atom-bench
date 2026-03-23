"""Chinchilla experiment grid constants and configuration."""

from __future__ import annotations

# ── Experiment grid constants ──────────────────────────────────────────────

# 10-size sweep for fine-grained Chinchilla (full study)
CHINCHILLA_SIZES = [f"chinchilla_{i}" for i in range(10)]

# 5-size sweep: log-spaced subset of chinchilla_0–9, calibrated for 50K-sample datasets.
# Spans ~1K–11M params (underfit → optimal → overfit) needed for L(N,D) fitting.
# Must stay in sync with model_registry.CHINCHILLA_5_SIZES.
CHINCHILLA_5_SIZES = ["chinchilla_1", "chinchilla_3", "chinchilla_5", "chinchilla_7", "chinchilla_9"]

# 7-size sweep: adds chinchilla_11 (~25-30M) and chinchilla_13 (~80-100M) to push
# past N* at D=100K, enabling proper Chinchilla overfit regime coverage.
# Use with --epochs 50 to ensure convergence at larger model sizes.
CHINCHILLA_7_SIZES = ["chinchilla_1", "chinchilla_3", "chinchilla_5", "chinchilla_7", "chinchilla_9",
                      "chinchilla_11", "chinchilla_13"]

# ── D budget grid ──────────────────────────────────────────────────────────
# Chinchilla requirement: exactly 1 epoch over D tokens.
# Total training steps = D_tokens / batch_size.
#
# D targets:   50K,   100K,   500K,   1M  total training samples
# Steps:        195,    390,  1953,  3906  (D // BATCH_SIZE)
# Actual D:   49920,  99840, 499968, 999936 (steps × BATCH_SIZE)
BATCH_SIZE = 256
D_TARGETS  = [50_000, 100_000, 500_000, 1_000_000]  # nominal data budgets
D_STEPS    = [d // BATCH_SIZE for d in D_TARGETS]   # [195, 390, 1953, 3906]
D_NAMES    = ["D1", "D2", "D3", "D4"]
D_VALUES   = [s * BATCH_SIZE for s in D_STEPS]      # actual samples per budget

# D1_steps is used as eval frequency so every D checkpoint is recorded
EVAL_EVERY = D_STEPS[0]  # 195 steps — captures D1, D2, D3, D4 milestones

# Three LRs swept per run; best selected at collect time.
# 1e-5 / 1e-4 / 1e-3: half-decade spacing across 4 decades of model size.
# 1e-5 is needed for chinchilla_11 (~30M) and chinchilla_13 (~100M params) —
# without it, both sweep values are 10-100x too large and large-model performance
# is systematically underestimated, biasing β downward.
LRS = [1e-5, 1e-4, 1e-3]
LR_NAMES = {1e-5: "lr1e-5", 1e-4: "lr1e-4", 1e-3: "lr1e-3"}

# Default tasks and archs
# Large-N tasks (N=50) are the primary Chinchilla targets:
#   N=50 → O(N²)=2500 attention pairs; FLOPs scale meaningfully with model size;
#   architectural biases (local/global/pair representation) start to matter.
# Small-N tasks (N≤20) are diagnostic comparisons only.
ALL_TASKS = [
    # ── Primary Chinchilla targets (large-N, meaningful FLOPs) ────────────
    "sphere_N50",       # N=50 hard sphere η=0.3  ← RECOMMENDED ENTRY POINT
    "chain_N50",        # N=50 chain, bonds + clash
    # ── Unified rule-ablation tasks (progressive difficulty) ──────────────
    "unified_R123_sp3_N10",     # Rules 1-3 (slots+angles+bonds)
    "unified_R1234_sp3_N10",    # Rules 1-4 (local VSEPR full)
    "unified_R5_sp3_N10",       # Rule 5 only (global pairs, no local VSEPR)
    "unified_R12345_sp3_N10",   # Rules 1-5 (local + global pairs)
    "unified_R123456_sp3_N10",  # Rules 1-6 (all rules)
    "unified_R123_sp3_N20",     # N=20 variants
    "unified_R1234_sp3_N20",
    "unified_R123456_sp3_N20",
    # ── Diagnostic / ablation (small-N, fast) ─────────────────────────────
    "sphere_easy", "sphere_medium", "sphere_hard",
    "chain_N10", "chain_N20",
    "vsepr_sp3",
    "sequence_linear",
]
ALL_ARCHS = ["painn", "transformer", "pairformer"]

# ── Gradient accumulation for OOM-prone (arch, size) combos ──────────────
#
# Large models on N=50 OOM at batch_size=256 on 24GB GPUs.  We use gradient
# accumulation to maintain effective_batch = 256 while reducing micro-batch.
# This preserves identical optimization dynamics (same gradient noise scale,
# same LR schedule) — no LR re-sweep needed.
#
# Strategy: probe actual GPU memory via a test forward pass.  If it OOMs,
# double grad_accum_steps (halve micro-batch) until it fits.
# Fallback table used when no GPU is available (e.g. generate on CPU node).

# Fallback table: (arch, min_size_index) → grad_accum_steps
# Derived from RTX 3090 24GB empirical OOM boundaries at N=50, batch=256.
_GA_FALLBACK_N50: dict[str, list[tuple[int, int]]] = {
    # PaiNN: chinchilla_7+ (idx>=3 in 7-size grid) OOMs
    "painn":      [(7, 4), (9, 8), (11, 16), (13, 16)],
    # Pairformer: O(N²) pair repr → OOMs at same boundary
    "pairformer": [(7, 4), (9, 8), (11, 16), (13, 16)],
    # Transformer: fits at all sizes on 24GB
    "transformer": [],
}
