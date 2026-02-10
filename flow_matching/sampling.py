"""Euler ODE sampler for flow matching."""

import torch
from torch import Tensor
import torch.nn as nn


@torch.no_grad()
def sample(
    model: nn.Module,
    n_atoms: int,
    n_samples: int,
    n_steps: int = 100,
    device: str = "cpu",
) -> Tensor:
    """Generate samples via Euler ODE integration.

    Integrates from t=0 (noise) to t=1 (data).

    Args:
        model: Velocity network, callable as model(x, t) -> (batch, N, 3).
        n_atoms: Number of atoms per sample.
        n_samples: Number of samples to generate.
        n_steps: Number of Euler steps.
        device: Device for generation.

    Returns:
        Generated positions, shape (n_samples, n_atoms, 3).
    """
    model.eval()
    x = torch.randn(n_samples, n_atoms, 3, device=device)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=device)
        v = model(x, t)
        x = x + v * dt

    return x


@torch.no_grad()
def sample_batched(
    model: nn.Module,
    n_atoms: int,
    n_samples: int,
    n_steps: int = 100,
    batch_size: int = 256,
    device: str = "cpu",
) -> Tensor:
    """Generate samples in chunks to avoid OOM.

    Args:
        model: Velocity network.
        n_atoms: Number of atoms per sample.
        n_samples: Total number of samples to generate.
        n_steps: Number of Euler steps.
        batch_size: Samples per chunk.
        device: Device for generation.

    Returns:
        Generated positions, shape (n_samples, n_atoms, 3).
    """
    chunks = []
    remaining = n_samples
    while remaining > 0:
        chunk_size = min(batch_size, remaining)
        chunk = sample(model, n_atoms, chunk_size, n_steps, device)
        chunks.append(chunk.cpu())
        remaining -= chunk_size
    return torch.cat(chunks, dim=0)
