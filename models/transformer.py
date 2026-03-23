"""Transformer velocity network — inspired by SimpleFold FoldingDiT.

Reference: SimpleFold (Apple, 2025), "SimpleFold: Folding Proteins is Simpler
than You Think". Uses DiT-style adaptive layer norm zero (adaLN-Zero)
conditioning with pairwise distance attention bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.common import SinusoidalTimestepEmbedding


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions for distance expansion."""

    def __init__(self, n_rbf: int = 64, cutoff: float = 10.0):
        super().__init__()
        offsets = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("offsets", offsets)
        self.width = (offsets[1] - offsets[0]).item() if n_rbf > 1 else 1.0

    def forward(self, distances: Tensor) -> Tensor:
        return torch.exp(
            -0.5 * ((distances.unsqueeze(-1) - self.offsets) / self.width) ** 2
        )


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (rms + self.eps)


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK normalization and attention bias."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, bias: Tensor | None = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if bias is not None:
            attn = attn + bias
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    At initialization, the adaLN modulation is zero-initialized so each
    block acts as an identity, following the DiT design.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLUFeedForward(hidden_dim, int(hidden_dim * mlp_ratio))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.attn.apply(_basic_init)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor, bias: Tensor | None = None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), bias=bias)
        x = x + gate_msa.unsqueeze(1) * h
        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    """Output layer with adaLN modulation and zero-initialized projection."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class TransformerVelocityNetwork(nn.Module):
    """Transformer-based velocity network for flow matching.

    Uses DiT-style adaLN-Zero conditioning and pairwise distance attention
    bias with Gaussian RBF expansion, following the SimpleFold architecture.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        mlp_ratio: float = 4.0,
        num_atom_types: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Input: 3D coordinates → hidden_dim
        self.input_proj = nn.Linear(3, hidden_dim)

        # Per-atom type embedding (sp3=0, sp2=1, sp=2, sidechain=3)
        self.atom_type_embed = nn.Embedding(num_atom_types, hidden_dim)

        # Pairwise distance bias: distances → RBF → per-head bias
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.pair_proj = nn.Linear(num_rbf, num_heads, bias=False)

        # Timestep → conditioning vector (sinusoidal + MLP)
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.time_proj[0].weight, std=0.02)
        nn.init.normal_(self.time_proj[2].weight, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )

        # Output: adaLN + projection → 3D velocity
        self.final_layer = FinalLayer(hidden_dim, 3)

    def _compute_pair_bias(self, positions: Tensor) -> Tensor:
        """Compute pairwise distance attention bias.

        Args:
            positions: (batch, N, 3).

        Returns:
            Attention bias (batch, num_heads, N, N).
        """
        dists = torch.cdist(positions, positions)  # (batch, N, N)
        rbf_feats = self.rbf(dists)  # (batch, N, N, num_rbf)
        bias = self.pair_proj(rbf_feats)  # (batch, N, N, num_heads)
        return bias.permute(0, 3, 1, 2)  # (batch, num_heads, N, N)

    def forward(self, positions: Tensor, t: Tensor, atom_type_ids: Tensor | None = None) -> Tensor:
        """Predict velocity field.

        Args:
            positions: Atom positions (batch, N, 3).
            t: Timestep (batch,).
            atom_type_ids: Per-atom type ids (N,) int64, optional.

        Returns:
            Predicted velocity (batch, N, 3).
        """
        pair_bias = self._compute_pair_bias(positions)
        x = self.input_proj(positions)
        if atom_type_ids is not None:
            x = x + self.atom_type_embed(atom_type_ids).unsqueeze(0)
        c = self.time_proj(self.time_embed(t))

        for block in self.blocks:
            x = block(x, c, bias=pair_bias)

        return self.final_layer(x, c)
