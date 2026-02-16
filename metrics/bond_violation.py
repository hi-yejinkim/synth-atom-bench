"""Bond violation and non-bonded clash metrics for chain configurations."""

import torch


def bond_violation_rate(
    positions: torch.Tensor, bond_length: float, tolerance: float = 0.1,
) -> float:
    """Fraction of samples with any bond length violation.

    A bond violation occurs when any consecutive pair distance deviates from
    the target bond_length by more than tolerance.

    Args:
        positions: (batch, N, 3) atom positions forming a chain.
        bond_length: target distance between consecutive atoms.
        tolerance: maximum allowed deviation from bond_length.

    Returns:
        Violation rate as a Python float in [0, 1].
    """
    # Consecutive distances: (batch, N-1)
    diffs = positions[:, 1:] - positions[:, :-1]
    dists = torch.norm(diffs, dim=-1)
    deviations = torch.abs(dists - bond_length)
    # Sample has violation if any bond deviates beyond tolerance
    has_violation = deviations.max(dim=1).values > tolerance
    return has_violation.float().mean().item()


def nonbonded_clash_rate(positions: torch.Tensor, radius: float) -> float:
    """Fraction of samples with any non-bonded clash.

    Non-bonded pairs are all (i, j) where |i - j| > 1 (not consecutive).
    A clash occurs when any non-bonded pair distance < 2 * radius.

    Args:
        positions: (batch, N, 3) atom positions forming a chain.
        radius: atom radius.

    Returns:
        Clash rate as a Python float in [0, 1].
    """
    batch, N, _ = positions.shape
    dists = torch.cdist(positions, positions)  # (batch, N, N)

    # Mask: exclude self-pairs and bonded (consecutive) pairs
    idx = torch.arange(N, device=positions.device)
    mask = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) <= 1  # (N, N)
    dists = dists.masked_fill(mask.unsqueeze(0), float("inf"))

    min_dists = dists.reshape(batch, -1).min(dim=1).values
    has_clash = min_dists < 2 * radius
    return has_clash.float().mean().item()


def bond_violation_rate_batched(
    positions: torch.Tensor, bond_length: float, tolerance: float = 0.1,
    chunk_size: int = 1000,
) -> float:
    """Bond violation rate for large batches, processed in chunks."""
    total = 0
    violating = 0
    for i in range(0, len(positions), chunk_size):
        chunk = positions[i : i + chunk_size]
        diffs = chunk[:, 1:] - chunk[:, :-1]
        dists = torch.norm(diffs, dim=-1)
        deviations = torch.abs(dists - bond_length)
        violating += (deviations.max(dim=1).values > tolerance).sum().item()
        total += len(chunk)
    return violating / total


def nonbonded_clash_rate_batched(
    positions: torch.Tensor, radius: float, chunk_size: int = 1000,
) -> float:
    """Non-bonded clash rate for large batches, processed in chunks."""
    total = 0
    clashing = 0
    for i in range(0, len(positions), chunk_size):
        chunk = positions[i : i + chunk_size]
        N = chunk.shape[1]
        dists = torch.cdist(chunk, chunk)
        idx = torch.arange(N, device=chunk.device)
        mask = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) <= 1
        dists = dists.masked_fill(mask.unsqueeze(0), float("inf"))
        min_dists = dists.reshape(len(chunk), -1).min(dim=1).values
        clashing += (min_dists < 2 * radius).sum().item()
        total += len(chunk)
    return clashing / total
