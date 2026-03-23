"""Metrics for VSEPR local geometry task.

All metrics operate on batches of generated structures (batch, N, 3) and
return scalar floats (lower = better unless noted).

Metrics:
  1. bond_length_in_peak_ratio   – fraction of bonds within allowed range ± tol
  2. angle_distribution_jsd      – Jensen-Shannon divergence of bond angle distributions
  3. torsional_out_of_bin_rate   – fraction of (sample, dihedral) outside allowed bins
  4. valence_overcoordination_rate – fraction of atoms exceeding max valence
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Bond Length In-Peak Ratio
# ---------------------------------------------------------------------------

def bond_length_in_peak_ratio(
    positions: torch.Tensor,
    bond_range: tuple[float, float],
    tolerance: float = 0.05,
) -> float:
    """Fraction of samples where ALL central–ligand bonds fall within bond_range ± tol.

    Central atom is assumed to be index 0; all other atoms are ligands.

    Args:
        positions: (batch, N, 3). positions[:, 0] = central atom.
        bond_range: (bond_min, bond_max) from dataset.
        tolerance: extra allowance beyond bond_range, in Angstroms.

    Returns:
        Fraction of samples (float in [0, 1]).  Higher is better.
    """
    central = positions[:, 0:1, :]          # (batch, 1, 3)
    ligands = positions[:, 1:, :]           # (batch, N-1, 3)
    bond_lengths = torch.norm(ligands - central, dim=-1)  # (batch, N-1)

    lo = bond_range[0] - tolerance
    hi = bond_range[1] + tolerance
    in_peak = (bond_lengths >= lo) & (bond_lengths <= hi)   # (batch, N-1)
    all_in_peak = in_peak.all(dim=1)                        # (batch,)
    return all_in_peak.float().mean().item()


def bond_length_in_peak_ratio_batched(
    positions: torch.Tensor,
    bond_range: tuple[float, float],
    tolerance: float = 0.05,
    chunk_size: int = 1000,
) -> float:
    """Chunked version for large batches."""
    total, hit = 0, 0
    for i in range(0, len(positions), chunk_size):
        chunk = positions[i: i + chunk_size]
        ratio = bond_length_in_peak_ratio(chunk, bond_range, tolerance)
        hit += ratio * len(chunk)
        total += len(chunk)
    return hit / total


# ---------------------------------------------------------------------------
# 2. Angle Distribution JSD
# ---------------------------------------------------------------------------

def _compute_bond_angles_deg(positions: np.ndarray) -> np.ndarray:
    """Compute all central–ligand-pair bond angles in degrees.

    Args:
        positions: (batch, N, 3)

    Returns:
        angles: (batch, n_pairs) where n_pairs = C(N-1, 2)
    """
    batch, N, _ = positions.shape
    central = positions[:, 0, :]   # (batch, 3)
    ligands = positions[:, 1:, :]  # (batch, N-1, 3)
    vecs = ligands - central[:, None, :]  # (batch, N-1, 3)
    # Normalize
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12
    vecs_unit = vecs / norms

    n_lig = N - 1
    angles = []
    for i in range(n_lig):
        for j in range(i + 1, n_lig):
            cos_a = np.clip(np.sum(vecs_unit[:, i] * vecs_unit[:, j], axis=-1), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_a)))  # (batch,)

    return np.stack(angles, axis=1)  # (batch, n_pairs)


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two normalized histograms."""
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def angle_distribution_jsd(
    positions: np.ndarray,
    target_angle_deg: float,
    angle_sigma_deg: float,
    n_bins: int = 90,
    angle_range: tuple[float, float] = (60.0, 180.0),
) -> float:
    """JSD between generated bond angle distribution and target Gaussian.

    Args:
        positions: (batch, N, 3) numpy array.
        target_angle_deg: mean of the target angle distribution.
        angle_sigma_deg: std of the target angle distribution.
        n_bins: histogram bins.
        angle_range: histogram range in degrees.

    Returns:
        JSD value (float in [0, log2]).  Lower is better (0 = perfect match).
    """
    angles = _compute_bond_angles_deg(positions).ravel()
    bins = np.linspace(angle_range[0], angle_range[1], n_bins + 1)
    gen_hist, _ = np.histogram(angles, bins=bins, density=False)

    # Target: discretised Gaussian
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    target_hist = np.exp(
        -0.5 * ((bin_centers - target_angle_deg) / angle_sigma_deg) ** 2
    )
    return _jsd(gen_hist.astype(float), target_hist)


# ---------------------------------------------------------------------------
# 3. Torsional Out-of-Bin Rate
# ---------------------------------------------------------------------------

def _compute_dihedrals_deg(positions: np.ndarray) -> np.ndarray | None:
    """Compute dihedral angles for consecutive ligand triplets.

    Dihedral: ligand[i] – central – ligand[i+1] – ligand[i+2] (or the
    pseudo-dihedral of any 4-atom set containing central as atom b).

    Returns (batch, n_dihedrals) array, or None if fewer than 3 ligands.
    """
    batch, N, _ = positions.shape
    n_lig = N - 1
    if n_lig < 3:
        return None

    central = positions[:, 0, :]  # (batch, 3)
    ligands = positions[:, 1:, :]  # (batch, N-1, 3)

    dihedrals = []
    for i in range(n_lig - 2):
        # Pseudo-dihedral: lig[i] - central - lig[i+1] - lig[i+2]
        a = ligands[:, i, :]
        b = central
        c = ligands[:, i + 1, :]
        d = ligands[:, i + 2, :]

        b1 = b - a
        b2 = c - b
        b3 = d - c

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-12
        n2_norm = np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-12
        n1_unit = n1 / n1_norm
        n2_unit = n2 / n2_norm

        cos_d = np.clip(np.sum(n1_unit * n2_unit, axis=-1), -1.0, 1.0)
        dih = np.degrees(np.arccos(cos_d))

        # Sign from triple product
        sign = np.sign(np.sum(np.cross(n1, n2) * b2, axis=-1))
        dih = np.where(sign < 0, -dih, dih)
        dihedrals.append(dih)

    return np.stack(dihedrals, axis=1)  # (batch, n_dihedrals)


def torsional_out_of_bin_rate(
    positions: np.ndarray,
    has_pi: bool,
    bin_centers_deg: tuple[float, ...] = (0.0, 180.0),
    bin_half_width_deg: float = 15.0,
) -> float:
    """Fraction of (sample, dihedral) pairs outside the allowed torsional bins.

    For has_pi=True: bins at 0° and 180° (planar constraint).
    For has_pi=False: all dihedrals are free → returns 0.0.

    Args:
        positions: (batch, N, 3) numpy array.
        has_pi: whether torsional bins apply.
        bin_centers_deg: allowed bin centres.
        bin_half_width_deg: half-width of each bin.

    Returns:
        Float in [0, 1].  Lower is better.
    """
    if not has_pi:
        return 0.0

    dihedrals = _compute_dihedrals_deg(positions)
    if dihedrals is None:
        return 0.0

    in_any_bin = np.zeros_like(dihedrals, dtype=bool)
    for center in bin_centers_deg:
        diff = np.abs(np.abs(dihedrals) - abs(center))  # handle ±180 wrap
        in_any_bin |= diff <= bin_half_width_deg

    out_of_bin_fraction = float((~in_any_bin).mean())
    return out_of_bin_fraction


# ---------------------------------------------------------------------------
# 4. Valence Overcoordination Rate
# ---------------------------------------------------------------------------

def valence_overcoordination_rate(
    positions: torch.Tensor,
    bond_range: tuple[float, float],
    max_valence: int | None = None,
    tolerance: float = 0.1,
) -> float:
    """Fraction of atoms (across all samples) that exceed their maximum valence.

    An atom is "bonded" to any other atom within [bond_range[0] - tol, bond_range[1] + tol].
    For the central atom (index 0) max_valence = N-1 (all ligands).
    For ligand atoms (indices 1..N-1) max_valence = 1 (only bonded to central).

    Args:
        positions: (batch, N, 3).
        bond_range: allowed bond length range from dataset.
        max_valence: override max valence for all atoms (None = use topology defaults).
        tolerance: extra allowance beyond bond_range.

    Returns:
        Float in [0, 1].  Lower is better.
    """
    batch, N, _ = positions.shape
    lo = bond_range[0] - tolerance
    hi = bond_range[1] + tolerance

    dists = torch.cdist(positions, positions)  # (batch, N, N)
    eye = torch.eye(N, dtype=torch.bool, device=positions.device)
    dists = dists.masked_fill(eye.unsqueeze(0), float("inf"))

    bonded = (dists >= lo) & (dists <= hi)  # (batch, N, N)
    degree = bonded.sum(dim=-1).float()     # (batch, N) — bonds per atom

    if max_valence is not None:
        overcoord = degree > max_valence
    else:
        # Central atom (idx 0): max valence = N-1; ligands: max valence = 1
        max_val = torch.ones(N, device=positions.device, dtype=torch.float)
        max_val[0] = float(N - 1)
        overcoord = degree > max_val.unsqueeze(0)

    return overcoord.float().mean().item()


def valence_overcoordination_rate_batched(
    positions: torch.Tensor,
    bond_range: tuple[float, float],
    max_valence: int | None = None,
    tolerance: float = 0.1,
    chunk_size: int = 1000,
) -> float:
    """Chunked version for large batches."""
    total, overcoord_count = 0, 0.0
    batch, N, _ = positions.shape
    for i in range(0, batch, chunk_size):
        chunk = positions[i: i + chunk_size]
        rate = valence_overcoordination_rate(chunk, bond_range, max_valence, tolerance)
        overcoord_count += rate * chunk.shape[0] * N
        total += chunk.shape[0] * N
    return overcoord_count / total
