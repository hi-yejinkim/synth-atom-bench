"""Metrics for the Sequence / Global Geometry task.

All metrics operate on batches of generated structures and return scalar floats.

Metrics:
  1. long_range_contact_recall    – fraction of designated contact pairs within contact_distance
  2. rdf_error                    – L2 / Wasserstein distance of g(r) vs ground truth
  3. radius_of_gyration_error     – mean absolute error in radius-of-gyration distribution
  4. bond_length_violation_rate   – fraction of samples with any bond length violation
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Long-Range Contact Recall
# ---------------------------------------------------------------------------

def long_range_contact_recall(
    positions: torch.Tensor,
    contact_pairs: np.ndarray,
    contact_distance: float = 5.0,
) -> float:
    """Fraction of designated contact pairs that are within contact_distance.

    Aggregated over all samples and all contact pairs.

    Args:
        positions: (batch, N, 3).
        contact_pairs: (n_contacts, 2) integer array of (i, j) index pairs.
        contact_distance: threshold distance in Angstroms.

    Returns:
        Recall in [0, 1].  Higher is better.
    """
    if len(contact_pairs) == 0:
        return 1.0

    batch = positions.shape[0]
    idx_i = torch.tensor(contact_pairs[:, 0], device=positions.device, dtype=torch.long)
    idx_j = torch.tensor(contact_pairs[:, 1], device=positions.device, dtype=torch.long)

    pos_i = positions[:, idx_i, :]  # (batch, n_contacts, 3)
    pos_j = positions[:, idx_j, :]  # (batch, n_contacts, 3)

    dists = torch.norm(pos_i - pos_j, dim=-1)  # (batch, n_contacts)
    satisfied = (dists <= contact_distance).float()
    return satisfied.mean().item()


def long_range_contact_recall_batched(
    positions: torch.Tensor,
    contact_pairs: np.ndarray,
    contact_distance: float = 5.0,
    chunk_size: int = 1000,
) -> float:
    """Chunked version for large batches."""
    if len(contact_pairs) == 0:
        return 1.0
    total, satisfied = 0.0, 0.0
    for i in range(0, len(positions), chunk_size):
        chunk = positions[i: i + chunk_size]
        r = long_range_contact_recall(chunk, contact_pairs, contact_distance)
        satisfied += r * len(chunk)
        total += len(chunk)
    return satisfied / total


# ---------------------------------------------------------------------------
# 2. RDF Error (L2 and Wasserstein)
# ---------------------------------------------------------------------------

def _compute_rdf(positions: np.ndarray, box_size: float, num_bins: int = 200
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Compute pair correlation function g(r) for a batch of structures.

    Reuses the same implementation used for g(r) distance in existing metrics.
    """
    from data.validate import pair_correlation
    return pair_correlation(positions, box_size, num_bins=num_bins)


def rdf_l2_error(
    positions: np.ndarray,
    gt_r: np.ndarray,
    gt_g_r: np.ndarray,
    box_size: float,
    num_bins: int = 200,
) -> float:
    """L2 distance between generated and ground-truth g(r).

    Args:
        positions: (batch, N, 3).
        gt_r: (num_bins,) bin centres from ground-truth.
        gt_g_r: (num_bins,) ground-truth g(r).
        box_size: cubic box side length.

    Returns:
        L2 distance (float).  Lower is better.
    """
    gen_r, gen_g_r = _compute_rdf(positions, box_size, num_bins)
    return float(np.sqrt(np.mean((gen_g_r - gt_g_r) ** 2)))


def rdf_wasserstein_error(
    positions: np.ndarray,
    gt_r: np.ndarray,
    gt_g_r: np.ndarray,
    box_size: float,
    num_bins: int = 200,
) -> float:
    """1-Wasserstein (Earth Mover's) distance between generated and GT g(r).

    Both distributions are normalised to sum to 1 before computing.

    Returns:
        Wasserstein distance (float).  Lower is better.
    """
    gen_r, gen_g_r = _compute_rdf(positions, box_size, num_bins)
    p = np.abs(gen_g_r) + 1e-12
    q = np.abs(gt_g_r) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    dr = gen_r[1] - gen_r[0] if len(gen_r) > 1 else 1.0
    return float(np.sum(np.abs(cdf_p - cdf_q)) * dr)


# ---------------------------------------------------------------------------
# 3. Radius of Gyration Error
# ---------------------------------------------------------------------------

def radius_of_gyration(positions: torch.Tensor) -> torch.Tensor:
    """Per-sample radius of gyration.

    Rg = sqrt(mean squared distance from centroid).

    Args:
        positions: (batch, N, 3).

    Returns:
        Rg: (batch,).
    """
    centroid = positions.mean(dim=1, keepdim=True)  # (batch, 1, 3)
    sq_dists = ((positions - centroid) ** 2).sum(dim=-1)  # (batch, N)
    return torch.sqrt(sq_dists.mean(dim=-1))  # (batch,)


def radius_of_gyration_error(
    positions: torch.Tensor,
    gt_rg_mean: float,
    gt_rg_std: float,
) -> float:
    """Mean absolute error of the Rg distribution relative to ground truth.

    Reports |mean(Rg_gen) - gt_rg_mean| + |std(Rg_gen) - gt_rg_std|
    normalised by gt_rg_mean.

    Args:
        positions: (batch, N, 3).
        gt_rg_mean: ground-truth mean Rg.
        gt_rg_std: ground-truth std Rg.

    Returns:
        Relative Rg error (float).  Lower is better.
    """
    rg = radius_of_gyration(positions)
    gen_mean = rg.mean().item()
    gen_std = rg.std().item()
    err = (abs(gen_mean - gt_rg_mean) + abs(gen_std - gt_rg_std))
    return err / (gt_rg_mean + 1e-8)


def radius_of_gyration_error_batched(
    positions: torch.Tensor,
    gt_rg_mean: float,
    gt_rg_std: float,
    chunk_size: int = 1000,
) -> float:
    """Chunked radius of gyration error."""
    all_rg = []
    for i in range(0, len(positions), chunk_size):
        all_rg.append(radius_of_gyration(positions[i: i + chunk_size]).cpu())
    rg = torch.cat(all_rg)
    gen_mean = rg.mean().item()
    gen_std = rg.std().item()
    return (abs(gen_mean - gt_rg_mean) + abs(gen_std - gt_rg_std)) / (gt_rg_mean + 1e-8)


def compute_gt_rg_stats(positions: np.ndarray) -> tuple[float, float]:
    """Compute ground-truth Rg mean and std from dataset positions.

    Args:
        positions: (dataset_size, N, 3).

    Returns:
        (rg_mean, rg_std).
    """
    pos_t = torch.from_numpy(positions.astype(np.float32))
    rg = radius_of_gyration(pos_t).numpy()
    return float(rg.mean()), float(rg.std())


# ---------------------------------------------------------------------------
# 4. Bond Length Violation Rate (polymer bonds)
# ---------------------------------------------------------------------------

def sequence_bond_violation_rate(
    positions: torch.Tensor,
    bond_list: np.ndarray,
    bond_length: float,
    tolerance: float = 0.1,
) -> float:
    """Fraction of samples with any bond length violation.

    Args:
        positions: (batch, N, 3).
        bond_list: (n_bonds, 2) int array of bonded atom index pairs.
        bond_length: target bond length.
        tolerance: max allowed deviation.

    Returns:
        Violation rate in [0, 1].  Lower is better.
    """
    if len(bond_list) == 0:
        return 0.0

    idx_i = torch.tensor(bond_list[:, 0], device=positions.device, dtype=torch.long)
    idx_j = torch.tensor(bond_list[:, 1], device=positions.device, dtype=torch.long)

    pos_i = positions[:, idx_i, :]  # (batch, n_bonds, 3)
    pos_j = positions[:, idx_j, :]
    dists = torch.norm(pos_i - pos_j, dim=-1)  # (batch, n_bonds)
    deviations = (dists - bond_length).abs()
    has_violation = deviations.max(dim=1).values > tolerance
    return has_violation.float().mean().item()


def sequence_bond_violation_rate_batched(
    positions: torch.Tensor,
    bond_list: np.ndarray,
    bond_length: float,
    tolerance: float = 0.1,
    chunk_size: int = 1000,
) -> float:
    """Chunked bond violation rate."""
    total, violating = 0, 0.0
    for i in range(0, len(positions), chunk_size):
        chunk = positions[i: i + chunk_size]
        violating += sequence_bond_violation_rate(chunk, bond_list, bond_length, tolerance) * len(chunk)
        total += len(chunk)
    return violating / total
