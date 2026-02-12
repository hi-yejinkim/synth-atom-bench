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
        "xs": {"hidden_dim": 16, "n_layers": 2},
        "small": {"hidden_dim": 32, "n_layers": 3},
        "medium": {"hidden_dim": 128, "n_layers": 5},
        "large": {"hidden_dim": 256, "n_layers": 8},
        "xl": {"hidden_dim": 512, "n_layers": 10},
    },
    "transformer": {
        "xs": {"hidden_dim": 32, "num_layers": 2, "num_heads": 2},
        "small": {"hidden_dim": 64, "num_layers": 3, "num_heads": 4},
        "medium": {"hidden_dim": 128, "num_layers": 6, "num_heads": 8},
        "large": {"hidden_dim": 256, "num_layers": 8, "num_heads": 8},
        "xl": {"hidden_dim": 384, "num_layers": 10, "num_heads": 8},
    },
    "pairformer": {
        "xs": {"hidden_dim": 32, "pair_dim": 16, "num_layers": 1, "num_heads": 2},
        "small": {"hidden_dim": 64, "pair_dim": 32, "num_layers": 2, "num_heads": 4},
        "medium": {"hidden_dim": 128, "pair_dim": 64, "num_layers": 4, "num_heads": 8},
        "large": {"hidden_dim": 256, "pair_dim": 128, "num_layers": 6, "num_heads": 8},
        "xl": {"hidden_dim": 384, "pair_dim": 192, "num_layers": 8, "num_heads": 8},
    },
}

# Default configs for model kwargs not in SIZE_PRESETS
MODEL_DEFAULTS = {
    "painn": {"n_rbf": 20, "cutoff": 10.0},
    "transformer": {"num_rbf": 64, "cutoff": 10.0, "mlp_ratio": 4.0},
    "pairformer": {"num_rbf": 64, "cutoff": 10.0, "expansion_factor": 4.0},
}
