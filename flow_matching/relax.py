"""Post-generation energy minimization (relaxation) for n-body samples.

Runs gradient descent on the LJ pair potential to push apart clashing atoms.
Physically this is "cheating" but useful for evaluation: it separates
model quality (how close to valid) from hard-constraint satisfaction.

Usage:
    positions_relaxed = relax_lj(positions, sigma=1.0, box_size=3.5, n_steps=100)
"""

from __future__ import annotations

import torch
from torch import Tensor


def lj_energy_torch(positions: Tensor, sigma: float, epsilon: float = 1.0) -> Tensor:
    """Compute LJ pair energy for a batch of configurations.

    Args:
        positions: (batch, N, 3)
        sigma: LJ sigma parameter
        epsilon: LJ epsilon parameter

    Returns:
        energy: (batch,)
    """
    # Pairwise distances
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)
    dist2 = (diff ** 2).sum(-1)  # (B, N, N)

    # Mask diagonal
    N = positions.shape[1]
    mask = ~torch.eye(N, dtype=torch.bool, device=positions.device)
    dist2 = dist2[:, mask].reshape(-1, N, N - 1)

    # LJ: 4ε[(σ/r)^12 - (σ/r)^6], only repulsive part matters for relaxation
    inv_r2 = sigma ** 2 / dist2.clamp(min=1e-8)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    pair_energy = 4 * epsilon * (inv_r12 - inv_r6)

    return 0.5 * pair_energy.sum(dim=(-1, -2))  # 0.5 for double counting


def relax_lj(
    positions: Tensor,
    sigma: float,
    box_size: float,
    epsilon: float = 1.0,
    n_steps: int = 100,
    lr: float = 0.01,
    repulsive_only: bool = True,
) -> Tensor:
    """Gradient descent energy minimization on LJ potential.

    Args:
        positions: (batch, N, 3) generated positions
        sigma: LJ sigma
        box_size: box size (positions are clipped to [0, box_size])
        epsilon: LJ epsilon
        n_steps: number of gradient descent steps
        lr: step size
        repulsive_only: if True, only minimize repulsive (r < sigma) interactions

    Returns:
        Relaxed positions (batch, N, 3)
    """
    pos = positions.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 3)
        dist = (diff ** 2).sum(-1).clamp(min=1e-10).sqrt()  # (B, N, N)

        N = pos.shape[1]
        mask = ~torch.eye(N, dtype=torch.bool, device=pos.device)

        if repulsive_only:
            # Only penalize pairs closer than sigma
            clash_dist = dist[:, mask].reshape(-1, N, N - 1)
            penalty = torch.where(
                clash_dist < sigma,
                (sigma / clash_dist.clamp(min=1e-8)) ** 12,
                torch.zeros_like(clash_dist),
            )
            loss = penalty.sum(dim=(-1, -2)).mean()
        else:
            dist_masked = dist[:, mask].reshape(-1, N, N - 1)
            inv_r2 = sigma ** 2 / dist_masked.clamp(min=1e-8) ** 2
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 ** 2
            loss = (4 * epsilon * (inv_r12 - inv_r6)).sum(dim=(-1, -2)).mean()

        loss.backward()
        optimizer.step()

        # Clip to box
        with torch.no_grad():
            pos.clamp_(0, box_size)

    return pos.detach()


def relax_batched(
    positions: Tensor,
    sigma: float,
    box_size: float,
    batch_size: int = 256,
    **kwargs,
) -> Tensor:
    """Relax in batches to avoid OOM."""
    results = []
    for i in range(0, len(positions), batch_size):
        batch = positions[i : i + batch_size]
        relaxed = relax_lj(batch, sigma=sigma, box_size=box_size, **kwargs)
        results.append(relaxed)
    return torch.cat(results, dim=0)
