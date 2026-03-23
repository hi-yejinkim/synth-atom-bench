"""Metrics for VSEPR+ task (backbone chain + sidechains + optional global contacts).

All metrics operate on batches of generated structures (batch, N_total, 3) and
return scalar floats in [0, 1] (lower = better unless noted).

Constraint hierarchy:
  1. clash_violation_rate          – any non-bonded pair closer than 2*radius
  2. backbone_bond_length_violation_rate  – backbone bond outside [bond_min, bond_max]
  3. sidechain_bond_length_violation_rate – sidechain bond outside its bond_range
  4. bond_angle_violation_rate     – VSEPR angle outside target ± tol at each backbone atom
  5. pi_planarity_violation_rate   – 4-atom planarity for pi-bond sites
  6. contact_recall                – fraction of designated contact pairs within contact_distance
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# 1. Clash violation rate
# ---------------------------------------------------------------------------

def clash_violation_rate(
    positions: torch.Tensor,
    bond_list: torch.Tensor,
    radius: float,
) -> float:
    """Fraction of samples with ANY non-bonded clash (distance < 2*radius).

    Args:
        positions: (batch, N, 3) float tensor.
        bond_list: (n_bonds, 2) int64 tensor of bonded atom-index pairs.
        radius: hard-sphere radius (Angstroms).

    Returns:
        Float in [0, 1].  Lower is better.
    """
    batch, N, _ = positions.shape
    device = positions.device

    # All-pairs distances: (batch, N, N)
    dists = torch.cdist(positions, positions)

    # Build bonded mask (N, N) — True where pair is bonded (exclude from clash)
    bonded_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    if bond_list.numel() > 0:
        bl = bond_list.long()
        bonded_mask[bl[:, 0], bl[:, 1]] = True
        bonded_mask[bl[:, 1], bl[:, 0]] = True

    # Self-pairs also excluded
    eye_mask = torch.eye(N, dtype=torch.bool, device=device)
    exclude = bonded_mask | eye_mask  # (N, N)

    # Set excluded pairs to large distance so they don't trigger clash
    dists = dists.masked_fill(exclude.unsqueeze(0), float("inf"))

    # For each sample, find minimum non-bonded distance
    min_dists = dists.reshape(batch, -1).min(dim=1).values  # (batch,)

    # Sample has a clash if any pair is below 2*radius
    has_clash = min_dists < (2.0 * radius)
    return has_clash.float().mean().item()


# ---------------------------------------------------------------------------
# 2. Backbone bond length violation rate
# ---------------------------------------------------------------------------

def backbone_bond_length_violation_rate(
    positions: torch.Tensor,
    N_backbone: int,
    backbone_bond_ranges: torch.Tensor,
) -> float:
    """Fraction of (sample, backbone_bond) pairs where bond length is outside [bond_min, bond_max].

    Args:
        positions: (batch, N_total, 3) float tensor.
        N_backbone: number of backbone atoms (B).
        backbone_bond_ranges: (B-1, 2) float tensor with [bond_min, bond_max] per backbone bond.

    Returns:
        Float in [0, 1].  Lower is better.
    """
    if N_backbone < 2:
        return 0.0

    batch = positions.shape[0]
    bb_pos = positions[:, :N_backbone, :]  # (batch, B, 3)

    # Backbone bond vectors: atom i to atom i+1 for i in 0..B-2
    p0 = bb_pos[:, :-1, :]   # (batch, B-1, 3)
    p1 = bb_pos[:, 1:, :]    # (batch, B-1, 3)
    bond_lengths = torch.norm(p1 - p0, dim=-1)  # (batch, B-1)

    bond_min = backbone_bond_ranges[:, 0]  # (B-1,)
    bond_max = backbone_bond_ranges[:, 1]  # (B-1,)

    violation = (bond_lengths < bond_min.unsqueeze(0)) | (bond_lengths > bond_max.unsqueeze(0))
    # violation: (batch, B-1)
    return violation.float().mean().item()


# ---------------------------------------------------------------------------
# 3. Sidechain bond length violation rate
# ---------------------------------------------------------------------------

def sidechain_bond_length_violation_rate(
    positions: torch.Tensor,
    N_backbone: int,
    sidechain_parent: torch.Tensor,
    sidechain_counts: torch.Tensor,
    sidechain_bond_ranges: torch.Tensor,
) -> float:
    """Fraction of (sample, sidechain_bond) pairs where bond length is outside bond_range.

    Args:
        positions: (batch, N_total, 3) float tensor.
        N_backbone: number of backbone atoms (B).
        sidechain_parent: (N_sc,) int64 tensor — backbone index for each sidechain atom.
        sidechain_counts: (B,) int64 tensor — number of sidechains per backbone atom.
        sidechain_bond_ranges: (N_sc, 2) float tensor — [bond_min, bond_max] per sidechain atom.

    Returns:
        Float in [0, 1].  Lower is better.  Returns 0.0 if no sidechain atoms.
    """
    N_sc = sidechain_parent.shape[0]
    if N_sc == 0:
        return 0.0

    batch = positions.shape[0]

    # Sidechain absolute indices: B .. B+N_sc-1
    sc_indices = torch.arange(N_backbone, N_backbone + N_sc, device=positions.device)

    # Sidechain atom positions: (batch, N_sc, 3)
    sc_pos = positions[:, sc_indices, :]

    # Parent backbone atom positions: (batch, N_sc, 3)
    parent_idx = sidechain_parent.long().to(positions.device)
    parent_pos = positions[:, parent_idx, :]  # (batch, N_sc, 3)

    # Bond lengths: (batch, N_sc)
    bond_lengths = torch.norm(sc_pos - parent_pos, dim=-1)

    bond_min = sidechain_bond_ranges[:, 0].to(positions.device)  # (N_sc,)
    bond_max = sidechain_bond_ranges[:, 1].to(positions.device)  # (N_sc,)

    violation = (bond_lengths < bond_min.unsqueeze(0)) | (bond_lengths > bond_max.unsqueeze(0))
    # violation: (batch, N_sc)
    return violation.float().mean().item()


# ---------------------------------------------------------------------------
# 4. Bond angle violation rate
# ---------------------------------------------------------------------------

def bond_angle_violation_rate(
    positions: torch.Tensor,
    N_backbone: int,
    sidechain_parent: torch.Tensor,
    sidechain_counts: torch.Tensor,
    target_angles: torch.Tensor,
    angle_tols: torch.Tensor,
) -> float:
    """Fraction of (sample, backbone_atom, angle_pair) triplets where angle is outside tolerance.

    For each backbone atom, computes all C(n_neighbors, 2) pairwise angles among its bonded
    neighbors.  n_neighbors = (backbone bonds) + sidechain_counts[i].

    Args:
        positions: (batch, N_total, 3) float tensor.
        N_backbone: number of backbone atoms (B).
        sidechain_parent: (N_sc,) int64 tensor — backbone index per sidechain atom.
        sidechain_counts: (B,) int64 tensor — sidechains per backbone atom.
        target_angles: (B,) float tensor — VSEPR target angle in degrees per backbone atom.
        angle_tols: (B,) float tensor — tolerance in degrees per backbone atom.

    Returns:
        Float in [0, 1].  Lower is better.  Returns 0.0 if no angle pairs exist.
    """
    batch = positions.shape[0]
    device = positions.device

    sc_counts = sidechain_counts.long().to(device)
    sc_parent = sidechain_parent.long().to(device)

    # Precompute cumulative sidechain offsets
    sc_offsets = torch.zeros(N_backbone, dtype=torch.long, device=device)
    if N_backbone > 1:
        sc_offsets[1:] = torch.cumsum(sc_counts[:-1], dim=0)

    total_pairs = 0
    total_violations = 0.0

    for i in range(N_backbone):
        # Collect neighbor indices for backbone atom i
        neighbors = []

        # Backbone neighbors: i-1 (if exists) and i+1 (if exists)
        if i > 0:
            neighbors.append(i - 1)
        if i < N_backbone - 1:
            neighbors.append(i + 1)

        # Sidechain neighbors
        n_sc = int(sc_counts[i].item())
        if n_sc > 0:
            sc_start = N_backbone + int(sc_offsets[i].item())
            for j in range(n_sc):
                neighbors.append(sc_start + j)

        n_nb = len(neighbors)
        if n_nb < 2:
            continue

        # Compute angle pairs: C(n_nb, 2)
        center_pos = positions[:, i, :]  # (batch, 3)
        nb_tensor = torch.tensor(neighbors, dtype=torch.long, device=device)
        nb_pos = positions[:, nb_tensor, :]  # (batch, n_nb, 3)

        # Vectors from center to each neighbor: (batch, n_nb, 3)
        vecs = nb_pos - center_pos.unsqueeze(1)
        norms = torch.norm(vecs, dim=-1, keepdim=True).clamp(min=1e-10)
        vecs_unit = vecs / norms  # (batch, n_nb, 3)

        # All pairs (i_pair, j_pair) with i_pair < j_pair
        target_deg = target_angles[i].item()
        tol_deg = angle_tols[i].item()
        target_rad = math.radians(target_deg)
        tol_rad = math.radians(tol_deg)

        for a in range(n_nb):
            for b in range(a + 1, n_nb):
                cos_angle = (vecs_unit[:, a, :] * vecs_unit[:, b, :]).sum(dim=-1)  # (batch,)
                cos_angle = cos_angle.clamp(-1.0, 1.0)
                angles = torch.acos(cos_angle)  # (batch,) in radians

                violation = (angles - target_rad).abs() > tol_rad
                total_violations += violation.float().sum().item()
                total_pairs += batch

    if total_pairs == 0:
        return 0.0

    return total_violations / total_pairs


# ---------------------------------------------------------------------------
# 5. Pi planarity violation rate
# ---------------------------------------------------------------------------

def pi_planarity_violation_rate(
    positions: torch.Tensor,
    N_backbone: int,
    has_pi_arr: torch.Tensor,
    plane_tol: float = 0.2,
) -> float:
    """Fraction of pi bond sites where 4-atom planarity is violated.

    For backbone bond i->i+1 where has_pi[i]=True: checks planarity of
    backbone[i-1, i, i+1, i+2].  Only checkable when i >= 1 and i+2 <= B-1.

    Args:
        positions: (batch, N_total, 3) float tensor.
        N_backbone: number of backbone atoms (B).
        has_pi_arr: (B,) bool tensor — whether atom i has a pi bond to i+1.
        plane_tol: maximum allowed out-of-plane deviation (Angstroms).

    Returns:
        Float in [0, 1].  Lower is better.  Returns 0.0 if no checkable pi bonds.
    """
    batch = positions.shape[0]
    device = positions.device

    has_pi = has_pi_arr.bool().to(device)
    bb_pos = positions[:, :N_backbone, :]  # (batch, B, 3)

    total_checks = 0
    total_violations = 0.0

    for i in range(N_backbone - 1):
        if not has_pi[i].item():
            continue
        # Need atoms i-1, i, i+1, i+2
        if i < 1 or i + 2 > N_backbone - 1:
            continue

        p1 = bb_pos[:, i - 1, :]  # (batch, 3)
        p2 = bb_pos[:, i, :]
        p3 = bb_pos[:, i + 1, :]
        p4 = bb_pos[:, i + 2, :]

        # Compute max deviation of 4 points from best-fit plane
        pts = torch.stack([p1, p2, p3, p4], dim=1)  # (batch, 4, 3)
        centroid = pts.mean(dim=1, keepdim=True)     # (batch, 1, 3)
        centered = pts - centroid                    # (batch, 4, 3)

        # SVD: smallest singular vector is the plane normal
        # torch.linalg.svd returns U, S, Vh; shape: centered (batch, 4, 3)
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        except Exception:
            # Fallback: skip this site
            continue

        normal = Vh[:, -1, :]  # (batch, 3) — smallest right-singular vector

        # Deviation of each point from the plane
        deviations = (centered * normal.unsqueeze(1)).sum(dim=-1).abs()  # (batch, 4)
        max_dev = deviations.max(dim=1).values  # (batch,)

        violation = max_dev > plane_tol
        total_violations += violation.float().sum().item()
        total_checks += batch

    if total_checks == 0:
        return 0.0

    return total_violations / total_checks


# ---------------------------------------------------------------------------
# 6. Contact recall
# ---------------------------------------------------------------------------

def contact_recall(
    positions: torch.Tensor,
    contact_pairs: torch.Tensor,
    contact_distance: float,
) -> float:
    """Fraction of designated contact pairs within contact_distance (higher=better).

    Args:
        positions: (batch, N_total, 3) float tensor.
        contact_pairs: (n_contacts, 2) int64 tensor of backbone atom-index pairs.
        contact_distance: maximum distance to count as a contact.

    Returns:
        Float in [0, 1].  Higher is better.  Returns 1.0 if no contacts.
    """
    if contact_pairs.numel() == 0:
        return 1.0

    batch = positions.shape[0]
    device = positions.device

    cp = contact_pairs.long().to(device)
    idx_a = cp[:, 0]  # (n_contacts,)
    idx_b = cp[:, 1]  # (n_contacts,)

    pos_a = positions[:, idx_a, :]  # (batch, n_contacts, 3)
    pos_b = positions[:, idx_b, :]  # (batch, n_contacts, 3)

    dists = torch.norm(pos_a - pos_b, dim=-1)  # (batch, n_contacts)
    satisfied = dists <= contact_distance       # (batch, n_contacts)

    return satisfied.float().mean().item()


# ---------------------------------------------------------------------------
# 7. Unified vsepr_plus_violation_rate
# ---------------------------------------------------------------------------

def vsepr_plus_violation_rate(
    positions: torch.Tensor,
    npz_meta: dict,
) -> dict:
    """Compute all active constraint metrics given NPZ metadata.

    Args:
        positions: (batch, N_total, 3) float tensor.
        npz_meta: dict with keys from VSEPRPlusDataset.npz_meta.

    Returns:
        dict with metric keys for all active constraints plus unified "violation_rate".
        All values are floats in [0, 1].  Lower is better except contact_recall.
    """
    N_backbone: int = int(npz_meta["N_backbone"])
    bond_list: torch.Tensor = npz_meta["bond_list"]
    radius: float = float(npz_meta["radius"])
    backbone_bond_ranges: torch.Tensor = npz_meta["backbone_bond_ranges"]
    sidechain_parent: torch.Tensor = npz_meta["sidechain_parent"]
    sidechain_counts: torch.Tensor = npz_meta["sidechain_counts"]
    sidechain_bond_ranges: torch.Tensor = npz_meta["sidechain_bond_ranges"]
    target_angles: torch.Tensor = npz_meta["target_angles"]
    angle_tols: torch.Tensor = npz_meta["angle_tols"]
    has_pi_arr: torch.Tensor = npz_meta["has_pi_arr"]
    contact_pairs: torch.Tensor = npz_meta["contact_pairs"]
    contact_distance: float = float(npz_meta["contact_distance"])

    use_clash: bool = bool(npz_meta["use_clash"])
    use_bond_lengths: bool = bool(npz_meta["use_bond_lengths"])
    use_bond_angles: bool = bool(npz_meta["use_bond_angles"])
    use_torsions: bool = bool(npz_meta["use_torsions"])
    use_global_contacts: bool = bool(npz_meta["use_global_contacts"])

    results: dict = {}
    violation_components: list[float] = []

    if use_clash:
        rate = clash_violation_rate(positions, bond_list, radius)
        results["clash_violation_rate"] = rate
        violation_components.append(rate)

    if use_bond_lengths:
        bb_rate = backbone_bond_length_violation_rate(
            positions, N_backbone, backbone_bond_ranges
        )
        results["bb_bond_length_violation_rate"] = bb_rate
        violation_components.append(bb_rate)

        sc_rate = sidechain_bond_length_violation_rate(
            positions, N_backbone, sidechain_parent, sidechain_counts, sidechain_bond_ranges
        )
        results["sc_bond_length_violation_rate"] = sc_rate
        violation_components.append(sc_rate)

    if use_bond_angles:
        ang_rate = bond_angle_violation_rate(
            positions, N_backbone, sidechain_parent, sidechain_counts,
            target_angles, angle_tols
        )
        results["bond_angle_violation_rate"] = ang_rate
        violation_components.append(ang_rate)

    if use_torsions and has_pi_arr.any():
        pi_rate = pi_planarity_violation_rate(positions, N_backbone, has_pi_arr)
        results["pi_planarity_violation_rate"] = pi_rate
        violation_components.append(pi_rate)

    if use_global_contacts:
        recall = contact_recall(positions, contact_pairs, contact_distance)
        results["contact_recall"] = recall
        # Contact recall is "higher is better", so violation component = 1 - recall
        violation_components.append(1.0 - recall)

    # Unified violation rate: max over all active constraint violations
    if violation_components:
        results["violation_rate"] = float(max(violation_components))
    else:
        results["violation_rate"] = 0.0

    return results
