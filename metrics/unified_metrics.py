"""Metrics for unified 6-rule structured tasks.

Extends vsepr_plus_metrics with Rule 1 (slot violation) and Rule 6 (periodicity).

All metrics operate on batches of generated structures (batch, N_total, 3) and
return scalar floats in [0, 1] (lower = better unless noted).

Rule metric mapping:
  1. slot_violation_rate             – coordination mismatch with orbital type
  2. bond_angle_violation_rate       – VSEPR angle outside target ± tol
  3. backbone/sidechain_bond_length  – bond outside [bond_min, bond_max]
  4. pi_planarity_violation_rate     – 4-atom planarity for pi-bond sites
  5. contact_recall + repulsive_recall – designated pairs within/beyond distance
  6. periodicity_violation_rate      – deviation from periodic backbone arrangement
  (always) clash_violation_rate      – any non-bonded pair closer than 2*radius
"""

from __future__ import annotations

import torch

from metrics.vsepr_plus_metrics import (
    backbone_bond_length_violation_rate,
    bond_angle_violation_rate,
    clash_violation_rate,
    contact_recall,
    pi_planarity_violation_rate,
    sidechain_bond_length_violation_rate,
)


# ---------------------------------------------------------------------------
# Rule 1: Slot violation rate
# ---------------------------------------------------------------------------

ORBITAL_N_SLOTS: dict = {"sp3": 4, "sp2": 3, "sp": 2}


def slot_violation_rate(
    N_backbone: int,
    orbital_types: list[str],
    sidechain_counts: torch.Tensor,
    n_lonepairs: torch.Tensor,
) -> float:
    """Fraction of backbone atoms where coordination != expected slots.

    This is a static property of the structure specification, not positions.
    It checks that (n_backbone_bonds + n_lonepairs + n_sidechain) == n_slots
    for each backbone atom.

    Args:
        N_backbone: number of backbone atoms (B).
        orbital_types: list of orbital type strings per backbone atom.
        sidechain_counts: (B,) int64 tensor.
        n_lonepairs: (B,) int64 tensor.

    Returns:
        Float in [0, 1].  Lower is better.  0 if all slots are correct.
    """
    B = N_backbone
    if B == 0:
        return 0.0

    violations = 0
    sc = sidechain_counts.long()
    nlp = n_lonepairs.long()

    for i in range(B):
        if B == 1:
            n_bb_bonds = 0
        elif i == 0 or i == B - 1:
            n_bb_bonds = 1
        else:
            n_bb_bonds = 2

        expected = ORBITAL_N_SLOTS.get(orbital_types[i], 4)
        total = n_bb_bonds + int(nlp[i].item()) + int(sc[i].item())
        if total != expected:
            violations += 1

    return violations / B


# ---------------------------------------------------------------------------
# Rule 5b: Repulsive pair recall
# ---------------------------------------------------------------------------

def repulsive_recall(
    positions: torch.Tensor,
    repulsive_pairs: torch.Tensor,
    repulsive_distances: torch.Tensor,
) -> float:
    """Fraction of repulsive pairs where distance >= min_dist threshold.

    Args:
        positions: (batch, N_total, 3) float tensor.
        repulsive_pairs: (n_rep, 2) int64 tensor.
        repulsive_distances: (n_rep,) float32 tensor, min distance per pair.

    Returns:
        Float in [0, 1].  Higher is better.  1.0 if all repulsive constraints met.
        Returns 1.0 if no repulsive pairs.
    """
    if repulsive_pairs.numel() == 0:
        return 1.0

    device = positions.device
    pos_a = positions[:, repulsive_pairs[:, 0].to(device), :]  # (batch, n_rep, 3)
    pos_b = positions[:, repulsive_pairs[:, 1].to(device), :]
    dists = torch.norm(pos_a - pos_b, dim=-1)  # (batch, n_rep)
    thresholds = repulsive_distances.to(device)  # (n_rep,)
    satisfied = (dists >= thresholds.unsqueeze(0)).float()
    return satisfied.mean().item()


# ---------------------------------------------------------------------------
# Rule 6: Periodicity violation rate
# ---------------------------------------------------------------------------

def periodicity_violation_rate(
    positions: torch.Tensor,
    N_backbone: int,
    period_length: int,
    period_tol: float = 0.5,
) -> float:
    """Fraction of periodic checks with deviation > period_tol.

    Matches the MCMC energy function _periodicity_energy:
      1. Anchor alignment: pos[k*P] should be at pos[0] + k*(pos[P]-pos[0]).
      2. Intra-period shape: relative geometry within each period should repeat.
         For atom j in period k: (pos[k*P+j] - pos[k*P]) should match
         (pos[j] - pos[0]).

    Args:
        positions: (batch, N_total, 3) float tensor.
        N_backbone: number of backbone atoms (B).
        period_length: periodicity period P (in backbone atoms).
        period_tol: maximum allowed deviation (Angstroms).

    Returns:
        Float in [0, 1].  Lower is better.  Returns 0.0 if period_length <= 0
        or fewer than 2 periods.
    """
    if period_length <= 0 or N_backbone <= period_length:
        return 0.0

    batch = positions.shape[0]
    P = period_length
    n_periods = N_backbone // P
    if n_periods < 2:
        return 0.0

    pos0 = positions[:, 0, :]   # (batch, 3)
    posP = positions[:, P, :]   # (batch, 3)
    step_vec = posP - pos0      # (batch, 3)

    total_checks = 0
    total_violations = 0.0

    for k in range(1, n_periods):
        # 1. Anchor alignment
        expected_anchor = pos0 + k * step_vec           # (batch, 3)
        actual_anchor = positions[:, k * P, :]          # (batch, 3)
        dev = torch.norm(actual_anchor - expected_anchor, dim=-1)
        total_violations += (dev > period_tol).float().sum().item()
        total_checks += batch

        # 2. Intra-period shape
        for j in range(1, P):
            idx = k * P + j
            if idx >= N_backbone:
                break
            expected_rel = positions[:, j, :] - pos0           # (batch, 3)
            actual_rel = positions[:, idx, :] - actual_anchor  # (batch, 3)
            dev = torch.norm(actual_rel - expected_rel, dim=-1)
            total_violations += (dev > period_tol).float().sum().item()
            total_checks += batch

    if total_checks == 0:
        return 0.0
    return total_violations / total_checks


# ---------------------------------------------------------------------------
# Unified violation rate (all 6 rules)
# ---------------------------------------------------------------------------

def unified_violation_rate(
    positions: torch.Tensor,
    npz_meta: dict,
) -> dict:
    """Compute all active constraint metrics given npz_meta from UnifiedDataset.

    Args:
        positions: (batch, N_total, 3) float tensor.
        npz_meta: dict from UnifiedDataset.npz_meta.

    Returns:
        dict with metric keys for all active constraints plus unified
        "violation_rate" (max over all active violations).
        All values are floats in [0, 1].  Lower is better except contact_recall.
    """
    N_backbone: int = int(npz_meta["N_backbone"])
    bond_list: torch.Tensor = npz_meta["bond_list"]
    radius: float = float(npz_meta["radius"])

    use_vsepr_slots: bool = bool(npz_meta["use_vsepr_slots"])
    use_vsepr_angles: bool = bool(npz_meta["use_vsepr_angles"])
    use_vsepr_bond_lengths: bool = bool(npz_meta["use_vsepr_bond_lengths"])
    use_vsepr_torsions: bool = bool(npz_meta["use_vsepr_torsions"])
    use_global_pairs: bool = bool(npz_meta["use_global_pairs"])
    use_global_periodicity: bool = bool(npz_meta["use_global_periodicity"])

    results: dict = {}
    violation_components: list[float] = []

    # Always: clash check
    cr = clash_violation_rate(positions, bond_list, radius)
    results["clash_violation_rate"] = cr
    violation_components.append(cr)

    # Rule 1: slot check (static — does not depend on positions)
    if use_vsepr_slots:
        svr = slot_violation_rate(
            N_backbone,
            npz_meta["orbital_types"],
            npz_meta["sidechain_counts"],
            npz_meta.get("n_lonepairs", torch.zeros(N_backbone, dtype=torch.long)),
        )
        results["slot_violation_rate"] = svr
        violation_components.append(svr)

    # Rule 2: bond angles (apply angle_tol_factor if global rules were active)
    if use_vsepr_angles:
        atf = float(npz_meta.get("angle_tol_factor", 1.0))
        angle_tols = npz_meta["angle_tols"]
        if atf != 1.0:
            angle_tols = angle_tols * atf
        avr = bond_angle_violation_rate(
            positions, N_backbone,
            npz_meta["sidechain_parent"],
            npz_meta["sidechain_counts"],
            npz_meta["target_angles"],
            angle_tols,
        )
        results["bond_angle_violation_rate"] = avr
        violation_components.append(avr)

    # Rule 3: bond lengths
    if use_vsepr_bond_lengths:
        bb_rate = backbone_bond_length_violation_rate(
            positions, N_backbone, npz_meta["backbone_bond_ranges"]
        )
        results["bb_bond_length_violation_rate"] = bb_rate
        violation_components.append(bb_rate)

        sc_rate = sidechain_bond_length_violation_rate(
            positions, N_backbone,
            npz_meta["sidechain_parent"],
            npz_meta["sidechain_counts"],
            npz_meta["sidechain_bond_ranges"],
        )
        results["sc_bond_length_violation_rate"] = sc_rate
        violation_components.append(sc_rate)

    # Rule 4: pi planarity / torsions
    if use_vsepr_torsions:
        has_pi = npz_meta["has_pi_arr"]
        if has_pi.any():
            pvr = pi_planarity_violation_rate(positions, N_backbone, has_pi)
            results["pi_planarity_violation_rate"] = pvr
            violation_components.append(pvr)

    # Rule 5: global pairs (attractive + repulsive)
    if use_global_pairs:
        # 5a: attractive contact pairs
        cp = npz_meta["contact_pairs"]
        if cp.numel() > 0:
            cd = npz_meta.get("contact_distances", None)
            if cd is not None and cd.numel() > 0:
                device = positions.device
                pos_a = positions[:, cp[:, 0].to(device), :]
                pos_b = positions[:, cp[:, 1].to(device), :]
                dists = torch.norm(pos_a - pos_b, dim=-1)
                thresholds = cd.to(device)
                satisfied = (dists <= thresholds.unsqueeze(0)).float()
                recall_val = satisfied.mean().item()
            else:
                recall_val = 1.0
            results["contact_recall"] = recall_val
            violation_components.append(1.0 - recall_val)

        # 5b: repulsive pairs
        rp = npz_meta.get("repulsive_pairs", torch.zeros((0, 2), dtype=torch.long))
        rd = npz_meta.get("repulsive_distances", torch.zeros((0,), dtype=torch.float32))
        if rp.numel() > 0:
            rr = repulsive_recall(positions, rp, rd)
            results["repulsive_recall"] = rr
            violation_components.append(1.0 - rr)

    # Rule 6: periodicity (use saved period_tol)
    if use_global_periodicity:
        period_length = int(npz_meta.get("period_length", 0))
        ptol = float(npz_meta.get("period_tol", 0.5))
        if period_length > 0:
            pvr = periodicity_violation_rate(
                positions, N_backbone, period_length, period_tol=ptol
            )
            results["periodicity_violation_rate"] = pvr
            violation_components.append(pvr)

    # Unified violation rate: max over all active constraint violations
    if violation_components:
        results["violation_rate"] = float(max(violation_components))
    else:
        results["violation_rate"] = 0.0

    return results
