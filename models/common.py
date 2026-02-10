"""Shared components for velocity networks."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding for scalar timesteps."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: Timesteps, shape (batch,).

        Returns:
            Embeddings, shape (batch, embed_dim).
        """
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # If embed_dim is odd, pad with a zero
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class OutputMLP(nn.Module):
    """Simple 2-layer MLP: Linear -> SiLU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
