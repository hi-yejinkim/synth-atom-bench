"""PaiNN velocity network — faithful reimplementation from SchNetPack.

Reference: Schütt et al., "Equivariant message passing for the prediction of
tensorial properties and molecular spectra" (2021).

Includes pre-LayerNorm and 1/sqrt(N_neighbors) message scaling for numerical
stability at large hidden_dim / deep stacks (chinchilla_11+).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from models.common import SinusoidalTimestepEmbedding


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions with evenly spaced centers."""

    def __init__(self, n_rbf: int = 20, cutoff: float = 10.0):
        super().__init__()
        self.n_rbf = n_rbf
        offsets = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("offsets", offsets)
        self.width = (offsets[1] - offsets[0]).item() if n_rbf > 1 else 1.0

    def forward(self, distances: Tensor) -> Tensor:
        """Expand distances into Gaussian basis.

        Args:
            distances: (n_pairs,).

        Returns:
            RBF features, (n_pairs, n_rbf).
        """
        return torch.exp(-0.5 * ((distances[:, None] - self.offsets[None, :]) / self.width) ** 2)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""

    def __init__(self, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Apply cosine cutoff.

        Args:
            distances: (n_pairs,).

        Returns:
            Cutoff values in [0, 1], shape (n_pairs,).
        """
        return 0.5 * (1.0 + torch.cos(torch.pi * distances / self.cutoff)) * (distances < self.cutoff).float()


class PaiNNInteraction(nn.Module):
    """PaiNN message passing block."""

    def __init__(self, hidden_dim: int, n_rbf: int):
        super().__init__()
        # Pre-norm on scalar features
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        # Context net on neighbor scalar features -> 3H for (ds, dvs, dvv)
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        # Filter net on RBF -> 3H, modulated by cutoff
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        f_cut: Tensor,
        dir_ij: Tensor,
        inv_sqrt_neighbors: float,
    ) -> tuple[Tensor, Tensor]:
        """Message passing update.

        Args:
            s: Scalar features (n_atoms, H).
            v: Vector features (n_atoms, 3, H).
            idx_i: Receiver indices (n_edges,).
            idx_j: Sender indices (n_edges,).
            rbf: Radial basis values (n_edges, n_rbf).
            f_cut: Cutoff values (n_edges,).
            dir_ij: Unit direction vectors (n_edges, 3).
            inv_sqrt_neighbors: 1/sqrt(avg_neighbors) scaling factor.

        Returns:
            Updated (s, v).
        """
        H = s.shape[-1]

        # Pre-norm scalar features
        s_normed = self.scalar_norm(s)

        # Context from neighbor scalar features
        context = self.context_net(s_normed[idx_j])  # (n_edges, 3H)

        # Filter from radial basis, modulated by cutoff
        W = self.filter_net(rbf) * f_cut[:, None]  # (n_edges, 3H)

        # Element-wise product, scaled by 1/sqrt(neighbors)
        msg = context * W * inv_sqrt_neighbors  # (n_edges, 3H)

        # Split into scalar, vector-scalar, vector-vector contributions
        ds, dvs, dvv = msg[:, :H], msg[:, H:2*H], msg[:, 2*H:]

        # Scatter-add messages to receivers
        s_update = torch.zeros_like(s)
        s_update.scatter_add_(0, idx_i[:, None].expand_as(ds), ds)

        # Vector updates: dvs * dir_ij + dvv * v_j
        v_update = torch.zeros_like(v)
        # dvs contribution: (n_edges, H) -> (n_edges, 3, H) via dir_ij
        v_msg_s = dvs[:, None, :] * dir_ij[:, :, None]  # (n_edges, 3, H)
        # dvv contribution: scale neighbor vectors
        v_msg_v = dvv[:, None, :] * v[idx_j]  # (n_edges, 3, H)
        v_msg = v_msg_s + v_msg_v
        v_update.scatter_add_(0, idx_i[:, None, None].expand_as(v_msg), v_msg)

        s = s + s_update
        v = v + v_update
        return s, v


class PaiNNMixing(nn.Module):
    """PaiNN intra-atomic mixing block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Pre-norm on scalar features
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Intra-atomic refinement.

        Args:
            s: Scalar features (n_atoms, H).
            v: Vector features (n_atoms, 3, H).

        Returns:
            Updated (s, v).
        """
        H = s.shape[-1]

        # Linear transforms on vector feature channel dim
        # v: (n_atoms, 3, H) -> apply linear on last dim
        Uv = self.U(v)  # (n_atoms, 3, H)
        Vv = self.V(v)  # (n_atoms, 3, H)

        # Norm of Vv: (n_atoms, H)
        Vv_norm = torch.sqrt(torch.sum(Vv ** 2, dim=1) + 1e-8)

        # Pre-norm scalar features, then concatenate with |Vv|
        s_normed = self.scalar_norm(s)
        ctx_input = torch.cat([s_normed, Vv_norm], dim=-1)  # (n_atoms, 2H)
        ctx = self.context_net(ctx_input)  # (n_atoms, 3H)
        a_ss, a_sv, a_vv = ctx[:, :H], ctx[:, H:2*H], ctx[:, 2*H:]

        # Dot product of Uv and Vv: sum over spatial dim -> (n_atoms, H)
        dot_uv = torch.sum(Uv * Vv, dim=1)

        # Updates
        s = s + a_ss + a_sv * dot_uv
        v = v + a_vv[:, None, :] * Uv

        return s, v


class PaiNNVelocityNetwork(nn.Module):
    """PaiNN-based velocity network for flow matching.

    All atoms are identical (hard spheres), so we use a single learned
    embedding instead of per-element embeddings. Timestep conditioning
    is additive on scalar features.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 5,
        n_rbf: int = 20,
        cutoff: float = 10.0,
        num_atom_types: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cutoff = cutoff

        # Radial basis and cutoff
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.cosine_cutoff = CosineCutoff(cutoff)

        # Atom embedding: single learned embedding for identical atoms
        self.atom_embedding = nn.Parameter(torch.randn(1, hidden_dim) * (hidden_dim ** -0.5))

        # Per-atom type embedding (sp3=0, sp2=1, sp=2, sidechain=3)
        self.atom_type_embed = nn.Embedding(num_atom_types, hidden_dim)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        # Message passing layers
        self.interactions = nn.ModuleList([
            PaiNNInteraction(hidden_dim, n_rbf) for _ in range(n_layers)
        ])
        self.mixings = nn.ModuleList([
            PaiNNMixing(hidden_dim) for _ in range(n_layers)
        ])

        # Final norm before readout
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Velocity readout from vector features: (n_atoms, 3, H) -> (n_atoms, 3)
        self.velocity_readout = nn.Linear(hidden_dim, 1, bias=False)

    def _build_graph(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build all-pairs graph from batched positions.

        Args:
            positions: (batch, N, 3).

        Returns:
            idx_i, idx_j: Edge indices (n_edges,).
            rbf: Radial basis values (n_edges, n_rbf).
            f_cut: Cutoff values (n_edges,).
            dir_ij: Unit direction vectors (n_edges, 3).
        """
        batch_size, N, _ = positions.shape
        device = positions.device

        # Flatten to (batch*N, 3)
        pos_flat = positions.reshape(-1, 3)

        # Build all-pairs edges within each sample (excluding self-loops)
        # For each sample: N*(N-1) edges
        arange_N = torch.arange(N, device=device)
        # All pairs (i, j) where i != j within one sample
        src = arange_N.repeat_interleave(N - 1)  # [0,0,...,1,1,...,N-1,N-1,...]
        # For each i, all j != i
        dst_list = []
        for i in range(N):
            dst_list.append(torch.cat([arange_N[:i], arange_N[i+1:]]))
        dst = torch.cat(dst_list)  # (N*(N-1),)

        # Expand across batch with offsets
        batch_offsets = torch.arange(batch_size, device=device)[:, None] * N  # (batch, 1)
        idx_i = (src[None, :] + batch_offsets).reshape(-1)  # (batch * N*(N-1),)
        idx_j = (dst[None, :] + batch_offsets).reshape(-1)

        # Compute displacement vectors and distances
        diff = pos_flat[idx_j] - pos_flat[idx_i]  # (n_edges, 3)
        dist = torch.norm(diff, dim=-1, keepdim=False)  # (n_edges,)

        # Unit direction (with safe division)
        dir_ij = diff / (dist[:, None] + 1e-8)

        # Radial basis and cutoff
        rbf = self.rbf(dist)
        f_cut = self.cosine_cutoff(dist)

        return idx_i, idx_j, rbf, f_cut, dir_ij

    def forward(self, positions: Tensor, t: Tensor, atom_type_ids: Tensor | None = None) -> Tensor:
        """Predict velocity field.

        Args:
            positions: Atom positions (batch, N, 3).
            t: Timestep (batch,).
            atom_type_ids: Per-atom type ids (N,) int64, optional.

        Returns:
            Predicted velocity (batch, N, 3).
        """
        batch_size, N, _ = positions.shape

        # Build graph
        idx_i, idx_j, rbf, f_cut, dir_ij = self._build_graph(positions)

        # Message scaling: 1/sqrt(N-1) where N-1 is the number of neighbors
        inv_sqrt_neighbors = 1.0 / math.sqrt(max(N - 1, 1))

        # Initialize scalar features: learned embedding + timestep
        s = self.atom_embedding.expand(batch_size * N, -1).clone()  # (batch*N, H)
        if atom_type_ids is not None:
            # atom_type_ids: (N,) → expand to (batch*N, H)
            type_emb = self.atom_type_embed(atom_type_ids)  # (N, H)
            type_emb = type_emb.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * N, -1)
            s = s + type_emb
        t_emb = self.time_proj(self.time_embed(t))  # (batch, H)
        # Repeat timestep embedding for each atom in the sample
        t_emb = t_emb[:, None, :].expand(-1, N, -1).reshape(batch_size * N, -1)
        s = s + t_emb

        # Initialize vector features: zeros
        v = torch.zeros(batch_size * N, 3, self.hidden_dim, device=positions.device)

        # Message passing
        for interaction, mixing in zip(self.interactions, self.mixings):
            s, v = interaction(s, v, idx_i, idx_j, rbf, f_cut, dir_ij, inv_sqrt_neighbors)
            s, v = mixing(s, v)

        # Final norm on scalar (for symmetry with pre-norms; stabilises readout)
        s = self.final_norm(s)

        # Velocity readout: (batch*N, 3, H) -> (batch*N, 3, 1) -> (batch*N, 3)
        velocity = self.velocity_readout(v).squeeze(-1)

        return velocity.reshape(batch_size, N, 3)
