"""Shared model registry, size presets, and default configs."""

from models.painn import PaiNNVelocityNetwork
from models.pairformer import PairformerVelocityNetwork
from models.transformer import TransformerVelocityNetwork

MODEL_REGISTRY = {
    "painn": PaiNNVelocityNetwork,
    "transformer": TransformerVelocityNetwork,
    "pairformer": PairformerVelocityNetwork,
}

SIZE_PRESETS = {
    "painn": {
        # --- Original presets ---
        "xs":     {"hidden_dim": 16,  "n_layers": 2},
        "small":  {"hidden_dim": 32,  "n_layers": 3},
        "medium": {"hidden_dim": 128, "n_layers": 5},
        "large":  {"hidden_dim": 256, "n_layers": 8},
        "xl":     {"hidden_dim": 512, "n_layers": 10},
        # --- Chinchilla sizes (log-spaced ~1k–2M params) ---
        "chinchilla_0": {"hidden_dim": 8,   "n_layers": 1},
        "chinchilla_1": {"hidden_dim": 12,  "n_layers": 1},
        "chinchilla_2": {"hidden_dim": 16,  "n_layers": 2},
        "chinchilla_3": {"hidden_dim": 24,  "n_layers": 2},
        "chinchilla_4": {"hidden_dim": 32,  "n_layers": 3},
        "chinchilla_5": {"hidden_dim": 64,  "n_layers": 4},
        "chinchilla_6": {"hidden_dim": 96,  "n_layers": 5},
        "chinchilla_7": {"hidden_dim": 128, "n_layers": 6},
        "chinchilla_8": {"hidden_dim": 192, "n_layers": 7},
        "chinchilla_9": {"hidden_dim": 256, "n_layers": 8},
        "chinchilla_11": {"hidden_dim": 512, "n_layers": 10},   # ~30M params
        "chinchilla_13": {"hidden_dim": 768, "n_layers": 14},   # ~100M params
    },
    "transformer": {
        # --- Original presets ---
        "xs":     {"hidden_dim": 32,  "num_layers": 2,  "num_heads": 2},
        "small":  {"hidden_dim": 64,  "num_layers": 3,  "num_heads": 4},
        "medium": {"hidden_dim": 128, "num_layers": 6,  "num_heads": 8},
        "large":  {"hidden_dim": 256, "num_layers": 8,  "num_heads": 8},
        "xl":     {"hidden_dim": 384, "num_layers": 10, "num_heads": 8},
        # --- Chinchilla sizes (log-spaced ~1k–2M params) ---
        # num_heads must divide hidden_dim
        "chinchilla_0": {"hidden_dim": 16,  "num_layers": 1, "num_heads": 2},
        "chinchilla_1": {"hidden_dim": 24,  "num_layers": 1, "num_heads": 4},
        "chinchilla_2": {"hidden_dim": 32,  "num_layers": 2, "num_heads": 4},
        "chinchilla_3": {"hidden_dim": 48,  "num_layers": 2, "num_heads": 4},
        "chinchilla_4": {"hidden_dim": 64,  "num_layers": 3, "num_heads": 4},
        "chinchilla_5": {"hidden_dim": 96,  "num_layers": 4, "num_heads": 8},
        "chinchilla_6": {"hidden_dim": 128, "num_layers": 6, "num_heads": 8},
        "chinchilla_7": {"hidden_dim": 192, "num_layers": 7, "num_heads": 8},
        "chinchilla_8": {"hidden_dim": 256, "num_layers": 8, "num_heads": 8},
        "chinchilla_9": {"hidden_dim": 320, "num_layers": 9, "num_heads": 8},
        "chinchilla_11": {"hidden_dim": 512, "num_layers": 10, "num_heads": 8},  # ~30M params
        "chinchilla_13": {"hidden_dim": 768, "num_layers": 12, "num_heads": 8},  # ~85M params
    },
    "pairformer": {
        # --- Original presets ---
        "xs":     {"hidden_dim": 32,  "pair_dim": 16,  "num_layers": 1, "num_heads": 2},
        "small":  {"hidden_dim": 64,  "pair_dim": 32,  "num_layers": 2, "num_heads": 4},
        "medium": {"hidden_dim": 128, "pair_dim": 64,  "num_layers": 4, "num_heads": 8},
        "large":  {"hidden_dim": 256, "pair_dim": 128, "num_layers": 6, "num_heads": 8},
        "xl":     {"hidden_dim": 384, "pair_dim": 192, "num_layers": 8, "num_heads": 8},
        # --- Chinchilla sizes (pair_dim = hidden_dim // 2, log-spaced ~1k–2M params) ---
        "chinchilla_0": {"hidden_dim": 16,  "pair_dim": 8,   "num_layers": 1, "num_heads": 2},
        "chinchilla_1": {"hidden_dim": 24,  "pair_dim": 12,  "num_layers": 1, "num_heads": 4},
        "chinchilla_2": {"hidden_dim": 32,  "pair_dim": 16,  "num_layers": 1, "num_heads": 4},
        "chinchilla_3": {"hidden_dim": 48,  "pair_dim": 24,  "num_layers": 2, "num_heads": 4},
        "chinchilla_4": {"hidden_dim": 64,  "pair_dim": 32,  "num_layers": 2, "num_heads": 4},
        "chinchilla_5": {"hidden_dim": 96,  "pair_dim": 48,  "num_layers": 3, "num_heads": 8},
        "chinchilla_6": {"hidden_dim": 128, "pair_dim": 64,  "num_layers": 4, "num_heads": 8},
        "chinchilla_7": {"hidden_dim": 192, "pair_dim": 96,  "num_layers": 5, "num_heads": 8},
        "chinchilla_8": {"hidden_dim": 256, "pair_dim": 128, "num_layers": 6, "num_heads": 8},
        "chinchilla_9": {"hidden_dim": 320, "pair_dim": 160, "num_layers": 7, "num_heads": 8},
        "chinchilla_11": {"hidden_dim": 384, "pair_dim": 192, "num_layers": 8, "num_heads": 8},   # ~24M params
        "chinchilla_13": {"hidden_dim": 512, "pair_dim": 256, "num_layers": 12, "num_heads": 8},  # ~80M params
    },
}

# Default configs for model kwargs not in SIZE_PRESETS
MODEL_DEFAULTS = {
    "painn": {"n_rbf": 20, "cutoff": 10.0},
    "transformer": {"num_rbf": 64, "cutoff": 10.0, "mlp_ratio": 4.0},
    "pairformer": {"num_rbf": 64, "cutoff": 10.0, "expansion_factor": 4.0},
}

# ── Chinchilla 5-size preset: log-spaced subset of chinchilla_0-9 ──────────
# Uses existing chinchilla sizes, which are calibrated for the original pipeline:
#   Dataset: 50K train samples  |  D budget: 50K–1M total samples seen (1–20 epochs)
#   Flow matching augmentation (random t per step) makes effective data >> 50K
#
# Rough param counts (arch-dependent, exact values computed at runtime):
#   chinchilla_1: ~2K–8K   params  (underfit region — reveals D* > N* regime)
#   chinchilla_3: ~10K–60K params
#   chinchilla_5: ~100K–450K params  ← expected sweet spot for 50K data
#   chinchilla_7: ~600K–3M  params
#   chinchilla_9: ~3M–11M   params  (may overfit at small D — reveals N* > D* regime)
#
# These 5 sizes span the full underfit→optimal→overfit spectrum needed for
# fitting L(N,D) = E + A/N^α + B/D^β and extracting optimal allocation exponents.
CHINCHILLA_5_SIZES = [
    "chinchilla_1",   # smallest: ~2K-8K params
    "chinchilla_3",   # small:    ~10K-60K params
    "chinchilla_5",   # medium:   ~100K-450K params
    "chinchilla_7",   # large:    ~600K-3M params
    "chinchilla_9",   # largest:  ~3M-11M params
]

# Extended 7-size preset including large models that push past N* at D=100K.
# chinchilla_11/13 require significantly more compute; use with --epochs 50
# to ensure convergence at the larger end.
CHINCHILLA_7_SIZES = [
    "chinchilla_1",   # ~2K-8K params
    "chinchilla_3",   # ~10K-60K params
    "chinchilla_5",   # ~100K-450K params
    "chinchilla_7",   # ~600K-3M params
    "chinchilla_9",   # ~3M-11M params
    "chinchilla_11",  # ~24M-30M params  (crosses N* at D≈100K)
    "chinchilla_13",  # ~80M-100M params (clearly beyond N* → overfit regime)
]

# ── Large-scale presets (for future datasets with 1M+ unique samples) ──────
# These would overfit on the current 50K-sample datasets.
# Use only when training data is scaled up proportionally.
LARGE_SCALE_SIZES = ["target_1m", "target_3m", "target_10m", "target_30m", "target_100m"]

for _arch, _sizes in [
    ("painn", {
        # hidden_dim, n_layers  → approx params (6*d^2*L + input/output heads)
        "target_1m":   {"hidden_dim": 112,  "n_layers": 5},   # ~378K×5 ≈ 400K+overhead ≈ 1M
        "target_3m":   {"hidden_dim": 192,  "n_layers": 6},   # ~1.1M×6 ≈ 3M
        "target_10m":  {"hidden_dim": 320,  "n_layers": 8},   # ~3.1M×8 ≈ 10M
        "target_30m":  {"hidden_dim": 512,  "n_layers": 10},  # ~7.9M×10 ≈ 30M
        "target_100m": {"hidden_dim": 768,  "n_layers": 14},  # ~100M
    }),
    ("transformer", {
        # hidden_dim, num_layers, num_heads  → approx 12*d^2*L
        "target_1m":   {"hidden_dim": 128,  "num_layers": 6,  "num_heads": 8},   # 12*128^2*6 ≈ 1.2M
        "target_3m":   {"hidden_dim": 256,  "num_layers": 4,  "num_heads": 8},   # 12*256^2*4 ≈ 3.1M
        "target_10m":  {"hidden_dim": 384,  "num_layers": 6,  "num_heads": 8},   # 12*384^2*6 ≈ 10.6M
        "target_30m":  {"hidden_dim": 512,  "num_layers": 8,  "num_heads": 8},   # 12*512^2*8 ≈ 25M
        "target_100m": {"hidden_dim": 768,  "num_layers": 12, "num_heads": 8},   # 12*768^2*12 ≈ 85M
    }),
    ("pairformer", {
        # hidden_dim, pair_dim, num_layers, num_heads
        "target_1m":   {"hidden_dim": 96,  "pair_dim": 48,  "num_layers": 3,  "num_heads": 8},
        "target_3m":   {"hidden_dim": 128, "pair_dim": 64,  "num_layers": 5,  "num_heads": 8},
        "target_10m":  {"hidden_dim": 192, "pair_dim": 96,  "num_layers": 8,  "num_heads": 8},
        "target_30m":  {"hidden_dim": 256, "pair_dim": 128, "num_layers": 10, "num_heads": 8},
        "target_100m": {"hidden_dim": 384, "pair_dim": 192, "num_layers": 14, "num_heads": 8},
    }),
]:
    SIZE_PRESETS[_arch].update(_sizes)
