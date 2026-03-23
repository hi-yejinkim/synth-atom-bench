"""Pairformer velocity network — reimplemented from Boltz PairformerStack.

Reference: Wohlwend et al., "Boltz-1: Democratizing Biomolecular Interaction
Modeling" (2024). Uses triangular multiplicative updates on pair representation
and pair-biased self-attention on single representation (PairformerNoSeqLayer
variant — no triangle attention, suitable for small N).
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


class TriangleMultiplication(nn.Module):
    """Triangular multiplicative update (outgoing or incoming).

    Outgoing: z_ij += sum_k a_ik * b_jk  (shared intermediate k)
    Incoming: z_ij += sum_k a_ki * b_kj  (shared intermediate k)
    """

    def __init__(self, pair_dim: int, mode: str = "outgoing"):
        super().__init__()
        assert mode in ("outgoing", "incoming")
        self.mode = mode

        self.norm_in = nn.LayerNorm(pair_dim)
        self.proj_a = nn.Linear(pair_dim, pair_dim)
        self.gate_a = nn.Linear(pair_dim, pair_dim)
        self.proj_b = nn.Linear(pair_dim, pair_dim)
        self.gate_b = nn.Linear(pair_dim, pair_dim)
        self.norm_out = nn.LayerNorm(pair_dim)
        self.out_proj = nn.Linear(pair_dim, pair_dim)
        self.out_gate = nn.Linear(pair_dim, pair_dim)

        self._init_weights()

    def _init_weights(self):
        for module in [self.proj_a, self.proj_b, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, z: Tensor) -> Tensor:
        """Triangular multiplicative update.

        Args:
            z: Pair representation (B, N, N, pair_dim).

        Returns:
            Update to pair representation (B, N, N, pair_dim).
        """
        z_ln = self.norm_in(z)
        a = torch.sigmoid(self.gate_a(z_ln)) * self.proj_a(z_ln)
        b = torch.sigmoid(self.gate_b(z_ln)) * self.proj_b(z_ln)

        if self.mode == "outgoing":
            # z_ij = sum_k a_ik * b_jk -> einsum bikd,bjkd->bijd
            out = torch.einsum("bikd,bjkd->bijd", a, b)
        else:
            # z_ij = sum_k a_ki * b_kj -> einsum bkid,bkjd->bijd
            out = torch.einsum("bkid,bkjd->bijd", a, b)

        out = self.norm_out(out)
        out = torch.sigmoid(self.out_gate(z_ln)) * self.out_proj(out)
        return out


class Transition(nn.Module):
    """SwiGLU MLP with pre-norm, reusable for single and pair representations."""

    def __init__(self, dim: int, expansion_factor: float = 4.0):
        super().__init__()
        hidden_dim = int(2 * dim * expansion_factor / 3)
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AttentionPairBias(nn.Module):
    """Multi-head self-attention on single repr with pair repr as bias."""

    def __init__(self, hidden_dim: int, pair_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_s = nn.LayerNorm(hidden_dim)
        self.norm_z = nn.LayerNorm(pair_dim)

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pair_bias_proj = nn.Linear(pair_dim, num_heads, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        """Attention with pair bias.

        Args:
            s: Single representation (B, N, hidden_dim).
            z: Pair representation (B, N, N, pair_dim).

        Returns:
            Update to single representation (B, N, hidden_dim).
        """
        B, N, C = s.shape
        s_ln = self.norm_s(s)
        z_ln = self.norm_z(z)

        # Q, K, V from single repr
        qkv = self.qkv(s_ln).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)

        # Sigmoid gate from single repr
        gate = torch.sigmoid(self.gate_proj(s_ln))  # (B, N, C)

        # Pair bias: (B, N, N, pair_dim) -> (B, N, N, num_heads) -> (B, H, N, N)
        pair_bias = self.pair_bias_proj(z_ln).permute(0, 3, 1, 2)

        # Standard scaled dot-product attention with bias
        attn = (q @ k.transpose(-2, -1)) * self.scale + pair_bias
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = gate * self.out_proj(out)
        return out


class PairformerBlock(nn.Module):
    """Single Pairformer block (PairformerNoSeqLayer variant).

    Updates pair repr with triangle multiplications + transition,
    then updates single repr with pair-biased attention + transition.
    """

    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int,
        num_heads: int,
        expansion_factor: float = 4.0,
    ):
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(pair_dim, mode="outgoing")
        self.tri_mul_in = TriangleMultiplication(pair_dim, mode="incoming")
        self.pair_transition = Transition(pair_dim, expansion_factor)
        self.attn_pair_bias = AttentionPairBias(hidden_dim, pair_dim, num_heads)
        self.single_transition = Transition(hidden_dim, expansion_factor)

    def forward(self, s: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            s: Single representation (B, N, hidden_dim).
            z: Pair representation (B, N, N, pair_dim).

        Returns:
            Updated (s, z).
        """
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.pair_transition(z)
        s = s + self.attn_pair_bias(s, z)
        s = s + self.single_transition(s)
        return s, z


class PairformerVelocityNetwork(nn.Module):
    """Pairformer-based velocity network for flow matching.

    Uses pair representation with triangular multiplicative updates and
    pair-biased self-attention, following the Boltz/AlphaFold2 design.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        pair_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        expansion_factor: float = 4.0,
        num_atom_types: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection: 3D coords -> single repr
        self.input_proj = nn.Linear(3, hidden_dim)

        # Per-atom type embedding (sp3=0, sp2=1, sp=2, sidechain=3)
        self.atom_type_embed = nn.Embedding(num_atom_types, hidden_dim)

        # Pair repr from pairwise distances: distances -> RBF -> pair_dim
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.pair_proj = nn.Linear(num_rbf, pair_dim)

        # Timestep conditioning: sinusoidal -> MLP -> additive to single repr
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.time_proj[0].weight, std=0.02)
        nn.init.normal_(self.time_proj[2].weight, std=0.02)

        # Pairformer blocks
        self.blocks = nn.ModuleList([
            PairformerBlock(hidden_dim, pair_dim, num_heads, expansion_factor)
            for _ in range(num_layers)
        ])

        # Output: LayerNorm -> zero-init Linear -> velocity (B, N, 3)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 3)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, positions: Tensor, t: Tensor, atom_type_ids: Tensor | None = None) -> Tensor:
        """Predict velocity field.

        Args:
            positions: Atom positions (batch, N, 3).
            t: Timestep (batch,).
            atom_type_ids: Per-atom type ids (N,) int64, optional.

        Returns:
            Predicted velocity (batch, N, 3).
        """
        # Single repr: input projection + timestep
        s = self.input_proj(positions)  # (B, N, hidden_dim)
        if atom_type_ids is not None:
            s = s + self.atom_type_embed(atom_type_ids).unsqueeze(0)
        t_emb = self.time_proj(self.time_embed(t))  # (B, hidden_dim)
        s = s + t_emb.unsqueeze(1)

        # Pair repr: pairwise distances -> RBF -> linear
        dists = torch.cdist(positions, positions)  # (B, N, N)
        z = self.pair_proj(self.rbf(dists))  # (B, N, N, pair_dim)

        # Pairformer blocks
        for block in self.blocks:
            s, z = block(s, z)

        # Output velocity
        return self.out_proj(self.out_norm(s))
