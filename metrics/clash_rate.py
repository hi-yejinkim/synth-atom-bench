"""GPU-accelerated clash rate computation."""

import torch


def has_clash(positions: torch.Tensor, radius: float) -> torch.Tensor:
    """Check which samples have at least one atomic clash.

    Args:
        positions: (batch, N, 3) atom positions.
        radius: atom radius. Clash if any pairwise distance < 2*radius.

    Returns:
        Boolean tensor of shape (batch,) — True if sample has a clash.
    """
    N = positions.shape[1]
    dists = torch.cdist(positions, positions)  # (batch, N, N)
    # Set diagonal (self-distances) to inf so they don't affect min
    eye = torch.eye(N, dtype=torch.bool, device=positions.device)
    dists = dists.masked_fill(eye.unsqueeze(0), float("inf"))
    # Min pairwise distance per sample
    min_dists = dists.reshape(positions.shape[0], -1).min(dim=1).values
    return min_dists < 2 * radius


def clash_rate(positions: torch.Tensor, radius: float) -> float:
    """Fraction of samples with at least one clash.

    Args:
        positions: (batch, N, 3) atom positions.
        radius: atom radius.

    Returns:
        Clash rate as a Python float in [0, 1].
    """
    return has_clash(positions, radius).float().mean().item()


def clash_rate_batched(positions: torch.Tensor, radius: float, chunk_size: int = 1000) -> float:
    """Clash rate for large batches, processed in chunks to avoid OOM.

    Args:
        positions: (batch, N, 3) atom positions.
        radius: atom radius.
        chunk_size: number of samples per chunk.

    Returns:
        Clash rate as a Python float in [0, 1].
    """
    total = 0
    clashing = 0
    for i in range(0, len(positions), chunk_size):
        chunk = positions[i : i + chunk_size]
        clashing += has_clash(chunk, radius).sum().item()
        total += len(chunk)
    return clashing / total
