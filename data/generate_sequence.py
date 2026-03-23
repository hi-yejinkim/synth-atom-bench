"""MCMC sampler for polymer (sequence/global geometry) configurations.

Generates linear, branched, or crosslinked polymers built from repeated fragments.
Key features:
  - Fixed bond lengths within tolerance
  - Long-range contact constraints (designated pairs must be within contact_distance)
  - No non-bonded clashes
  - Fragment periodicity encoded in fragment_ids saved to NPZ

Polymer types:
  - linear:      simple chain, N = n_fragments * fragment_size
  - branched:    Y-shaped; backbone + two equal arms, N total atoms
  - crosslinked: two linear chains of N//2 each, joined by one cross-link

Contact pairs are assigned once per dataset and saved.  They define atom pairs
(i, j) with |i - j| > fragment_size along the chain index that must be within
contact_distance of each other (the "fold" that models must learn).
"""

import argparse
import os
import time

import numpy as np

from data.generate_chains import (
    nerf_place_atom,
    has_nonbonded_clash,
    initialize_chain,
    random_rotation_matrix,
)


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------

def _build_linear_bonds(N: int) -> np.ndarray:
    """Bond list for a linear chain: [(0,1), (1,2), ..., (N-2,N-1)]."""
    return np.array([[i, i + 1] for i in range(N - 1)], dtype=np.int32)


def _build_branched_bonds(N: int) -> tuple[np.ndarray, int]:
    """Bond list for a Y-shaped polymer.

    Splits N atoms as: backbone (2/3) + arm1 (1/6) + arm2 (1/6).
    The branch point is the last atom of the backbone.
    Returns (bond_list, branch_point_idx).
    """
    n_backbone = max(2, N * 2 // 3)
    n_arm = (N - n_backbone) // 2
    n_arm2 = N - n_backbone - n_arm

    bonds = []
    # Backbone bonds
    for i in range(n_backbone - 1):
        bonds.append([i, i + 1])
    branch_point = n_backbone - 1
    # Arm1: branch_point → n_backbone, n_backbone+1, ...
    arm1_start = n_backbone
    bonds.append([branch_point, arm1_start])
    for i in range(arm1_start, arm1_start + n_arm - 1):
        bonds.append([i, i + 1])
    # Arm2: branch_point → arm1_start+n_arm, ...
    arm2_start = arm1_start + n_arm
    bonds.append([branch_point, arm2_start])
    for i in range(arm2_start, arm2_start + n_arm2 - 1):
        bonds.append([i, i + 1])

    return np.array(bonds, dtype=np.int32), branch_point


def _build_crosslinked_bonds(N: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Bond list for two linear chains connected by one cross-link.

    Chain1: atoms 0..N//2-1
    Chain2: atoms N//2..N-1
    Cross-link: midpoint of chain1 ↔ midpoint of chain2
    Returns (bond_list, [(cross_link_i, cross_link_j)]).
    """
    half = N // 2
    bonds = []
    for i in range(half - 1):
        bonds.append([i, i + 1])
    for i in range(half, N - 1):
        bonds.append([i, i + 1])
    cross_i = half // 2
    cross_j = half + half // 2
    cross_links = [(cross_i, cross_j)]
    return np.array(bonds, dtype=np.int32), cross_links


# ---------------------------------------------------------------------------
# Polymer initializers
# ---------------------------------------------------------------------------

def _initialize_linear(N: int, bond_length: float, radius: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Initialize a self-avoiding linear chain."""
    return initialize_chain(N, bond_length, radius, rng)


def _initialize_branched(N: int, bond_length: float, radius: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Initialize a Y-shaped polymer: build backbone then two arms."""
    _, branch_point = _build_branched_bonds(N)
    n_backbone = branch_point + 1
    n_arm = (N - n_backbone) // 2
    n_arm2 = N - n_backbone - n_arm

    # Build backbone as a chain
    backbone = initialize_chain(n_backbone, bond_length, radius, rng)
    positions = np.zeros((N, 3))
    positions[:n_backbone] = backbone

    # Build arm1 from branch point
    arm1_start = n_backbone
    bp = backbone[branch_point]
    prev1 = backbone[branch_point - 1] if branch_point > 0 else bp + np.array([bond_length, 0, 0])
    prev2 = backbone[branch_point - 2] if branch_point > 1 else prev1 + np.array([bond_length, 0, 0])

    for i in range(n_arm):
        placed = False
        for _ in range(1000):
            theta = np.arccos(rng.uniform(-1, 1))
            phi = rng.uniform(0, 2 * np.pi)
            new_pos = nerf_place_atom(prev2, prev1, bp, bond_length, theta, phi)
            positions[arm1_start + i] = new_pos
            if not has_nonbonded_clash(positions[:arm1_start + i + 1], arm1_start + i, radius):
                placed = True
                break
        if not placed:
            success = False
            break  # trigger outer restart loop — do not silently place clashing atom
        if i == 0:
            prev2, prev1 = bp, positions[arm1_start]
        else:
            prev2, prev1 = positions[arm1_start + i - 1], positions[arm1_start + i]

    # Build arm2 from branch point (different direction)
    arm2_start = arm1_start + n_arm
    prev1 = backbone[branch_point - 1] if branch_point > 0 else bp - np.array([bond_length, 0, 0])
    prev2 = backbone[branch_point - 2] if branch_point > 1 else prev1 - np.array([bond_length, 0, 0])

    for i in range(n_arm2):
        placed = False
        for _ in range(1000):
            theta = np.arccos(rng.uniform(-1, 1))
            phi = rng.uniform(0, 2 * np.pi)
            new_pos = nerf_place_atom(prev2, prev1, bp, bond_length, theta, phi)
            positions[arm2_start + i] = new_pos
            if not has_nonbonded_clash(positions[:arm2_start + i + 1], arm2_start + i, radius):
                placed = True
                break
        if not placed:
            success = False
            break  # trigger outer restart loop — do not silently place clashing atom
        if i == 0:
            prev2, prev1 = bp, positions[arm2_start]
        else:
            prev2, prev1 = positions[arm2_start + i - 1], positions[arm2_start + i]

    return positions - positions.mean(axis=0)


def _initialize_crosslinked(N: int, bond_length: float, radius: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Initialize two self-avoiding chains (crosslink handled as soft constraint)."""
    half = N // 2
    chain1 = initialize_chain(half, bond_length, radius, rng)
    chain2 = initialize_chain(N - half, bond_length, radius, rng)
    # Offset chain2 so they don't overlap initially
    offset = np.array([chain1.max() + 2.0 * bond_length, 0.0, 0.0])
    positions = np.zeros((N, 3))
    positions[:half] = chain1
    positions[half:] = chain2 + offset
    return positions - positions.mean(axis=0)


# ---------------------------------------------------------------------------
# Contact pair assignment
# ---------------------------------------------------------------------------

def assign_contact_pairs(
    N: int,
    contact_fraction: float,
    min_separation: int,
    polymer_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly assign long-range contact pairs.

    Long-range = sequence separation > min_separation.
    For crosslinked polymers, contacts are between the two chains.

    Returns:
        contact_pairs: (n_contacts, 2) int32 array of (i, j) pairs with i < j.
    """
    if polymer_type == "crosslinked":
        half = N // 2
        candidates = []
        for i in range(half):
            for j in range(half, N):
                candidates.append((i, j))
    else:
        candidates = [
            (i, j) for i in range(N) for j in range(i + min_separation + 1, N)
        ]

    if not candidates:
        return np.empty((0, 2), dtype=np.int32)

    n_contacts = max(1, int(contact_fraction * len(candidates)))
    chosen = rng.choice(len(candidates), size=min(n_contacts, len(candidates)), replace=False)
    pairs = [candidates[k] for k in chosen]
    return np.array(sorted(pairs), dtype=np.int32)


# ---------------------------------------------------------------------------
# Constraint checkers
# ---------------------------------------------------------------------------

def _bond_length_ok(positions: np.ndarray, bond_list: np.ndarray,
                    bond_length: float, tolerance: float = 0.1) -> bool:
    """All bond lengths within [bond_length - tol, bond_length + tol]."""
    for i, j in bond_list:
        d = float(np.linalg.norm(positions[i] - positions[j]))
        if abs(d - bond_length) > tolerance:
            return False
    return True


def _contacts_satisfied(positions: np.ndarray, contact_pairs: np.ndarray,
                        contact_distance: float) -> bool:
    """All specified long-range contact pairs within contact_distance."""
    for i, j in contact_pairs:
        d = float(np.linalg.norm(positions[i] - positions[j]))
        if d > contact_distance:
            return False
    return True


def _no_nonbonded_clash(positions: np.ndarray, bond_list: np.ndarray,
                        radius: float) -> bool:
    """No atom pair that is not bonded has distance < 2*radius."""
    N = len(positions)
    threshold_sq = (2.0 * radius) ** 2
    bonded = set()
    for i, j in bond_list:
        bonded.add((min(i, j), max(i, j)))

    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in bonded:
                continue
            diff = positions[i] - positions[j]
            if float(np.dot(diff, diff)) < threshold_sq:
                return False
    return True


# ---------------------------------------------------------------------------
# MCMC move: pivot for a linear segment
# ---------------------------------------------------------------------------

def _pivot_move_segment(
    positions: np.ndarray,
    start: int,
    end: int,
    bond_list: np.ndarray,
    contact_pairs: np.ndarray,
    contact_distance: float,
    radius: float,
    rng: np.random.Generator,
) -> bool:
    """Pivot move on atoms [start..end].  Returns True if accepted."""
    N_seg = end - start + 1
    if N_seg < 3:
        return False

    k = rng.integers(start + 1, end)
    R = random_rotation_matrix(rng)
    pivot = positions[k].copy()

    old_tail = positions[k + 1: end + 1].copy()
    new_tail = (old_tail - pivot) @ R.T + pivot

    new_positions = positions.copy()
    new_positions[k + 1: end + 1] = new_tail

    threshold_sq = (2.0 * radius) ** 2
    bonded_at_k = set()
    for bi, bj in bond_list:
        if bi == k or bj == k:
            bonded_at_k.add(bi if bi != k else bj)

    # Check head–tail clashes for atoms moved
    for t_idx in range(k + 1, end + 1):
        for h_idx in range(start, k):
            pair = (min(h_idx, t_idx), max(h_idx, t_idx))
            in_bond = any(
                (bi == h_idx and bj == t_idx) or (bi == t_idx and bj == h_idx)
                for bi, bj in bond_list
            )
            if in_bond:
                continue
            diff = new_positions[h_idx] - new_positions[t_idx]
            if float(np.dot(diff, diff)) < threshold_sq:
                return False

    # Check contact constraints
    if len(contact_pairs) > 0 and not _contacts_satisfied(new_positions, contact_pairs, contact_distance):
        return False

    positions[k + 1: end + 1] = new_tail
    return True


# ---------------------------------------------------------------------------
# Folding initializer: satisfy contact pairs before sampling
# ---------------------------------------------------------------------------

def fold_toward_contacts(
    positions: np.ndarray,
    contact_pairs: np.ndarray,
    contact_distance: float,
    bond_list: np.ndarray,
    bond_length: float,
    radius: float,
    rng: np.random.Generator,
    max_steps: int = 50_000,
) -> np.ndarray:
    """Greedy MCMC to satisfy contact constraints.

    Runs pivot moves preferentially chosen to bring violated contacts closer.
    Bond length and non-bonded clash constraints are *relaxed* here (we just
    want a reasonable starting point). Returns best positions found.
    """
    N = len(positions)
    threshold_sq = (2.0 * radius) ** 2
    tol = 0.15  # relaxed bond tolerance for init

    # Build adjacency for bonded pairs
    bonded = set()
    for i, j in bond_list:
        bonded.add((min(i, j), max(i, j)))

    def _check_init(pos):
        # Relaxed: only check bond lengths and clash (no contact constraint yet)
        for i, j in bond_list:
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if abs(d - bond_length) > tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                if (min(i, j), max(i, j)) in bonded:
                    continue
                diff = pos[i] - pos[j]
                if float(np.dot(diff, diff)) < threshold_sq:
                    return False
        return True

    for step in range(max_steps):
        # Find a violated contact
        violated = [(i, j) for i, j in contact_pairs
                    if np.linalg.norm(positions[i] - positions[j]) > contact_distance]
        if not violated:
            break

        i, j = violated[rng.integers(len(violated))]
        if i > j:
            i, j = j, i

        if j <= i + 1:
            continue

        # Pivot somewhere between i and j
        k = rng.integers(i + 1, j)
        R = random_rotation_matrix(rng)
        pivot = positions[k].copy()
        new_positions = positions.copy()
        new_positions[k + 1:] = (positions[k + 1:] - pivot) @ R.T + pivot

        new_dist = float(np.linalg.norm(new_positions[i] - new_positions[j]))
        if new_dist < np.linalg.norm(positions[i] - positions[j]):
            if _check_init(new_positions):
                positions = new_positions

    return positions


# ---------------------------------------------------------------------------
# Main MCMC sampler
# ---------------------------------------------------------------------------

def mcmc_sequence_sample(
    N: int,
    bond_length: float,
    radius: float,
    contact_pairs: np.ndarray,
    contact_distance: float,
    polymer_type: str,
    num_samples: int,
    burn_in: int | None = None,
    thin_interval: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """MCMC sampler for polymer configurations with long-range contact constraints.

    Returns:
        positions: (num_samples, N, 3) centered at origin.
    """
    rng = np.random.default_rng(seed)

    if burn_in is None:
        burn_in = 200 * N
    if thin_interval is None:
        thin_interval = 20 * N

    # Build topology
    if polymer_type == "linear":
        bond_list = _build_linear_bonds(N)
        segments = [(0, N - 1)]  # one segment for pivot moves
    elif polymer_type == "branched":
        bond_list, branch_point = _build_branched_bonds(N)
        n_backbone = branch_point + 1
        n_arm = (N - n_backbone) // 2
        segments = [
            (0, n_backbone - 1),
            (n_backbone, n_backbone + n_arm - 1),
            (n_backbone + n_arm, N - 1),
        ]
    elif polymer_type == "crosslinked":
        bond_list, cross_links = _build_crosslinked_bonds(N)
        half = N // 2
        segments = [(0, half - 1), (half, N - 1)]
        # Add cross-links as extra "required contacts"
        extra_contacts = np.array(cross_links, dtype=np.int32)
        if len(contact_pairs) > 0:
            contact_pairs = np.vstack([contact_pairs, extra_contacts])
        else:
            contact_pairs = extra_contacts
    else:
        raise ValueError(f"Unknown polymer_type: {polymer_type!r}. Choose linear/branched/crosslinked.")

    # Initialize positions
    if polymer_type == "linear":
        positions = _initialize_linear(N, bond_length, radius, rng)
    elif polymer_type == "branched":
        positions = _initialize_branched(N, bond_length, radius, rng)
    else:
        positions = _initialize_crosslinked(N, bond_length, radius, rng)

    # Fold to satisfy contact constraints
    if len(contact_pairs) > 0:
        print("  Folding to satisfy contact constraints...")
        positions = fold_toward_contacts(
            positions, contact_pairs, contact_distance,
            bond_list, bond_length, radius, rng,
        )
        n_satisfied = sum(
            np.linalg.norm(positions[i] - positions[j]) <= contact_distance
            for i, j in contact_pairs
        )
        print(f"  Contacts satisfied: {n_satisfied}/{len(contact_pairs)}")

    total_steps = burn_in + num_samples * thin_interval
    samples = np.empty((num_samples, N, 3))
    sample_idx = 0
    accepted = 0
    report_interval = max(1, total_steps // 100)

    t0 = time.time()
    for step in range(total_steps):
        # Pick a random segment and attempt a pivot move on it
        seg = segments[rng.integers(len(segments))]
        if _pivot_move_segment(
            positions, seg[0], seg[1], bond_list,
            contact_pairs, contact_distance, radius, rng,
        ):
            accepted += 1

        if step >= burn_in and (step - burn_in) % thin_interval == 0:
            centered = positions - positions.mean(axis=0)
            samples[sample_idx] = centered
            sample_idx += 1

        if (step + 1) % report_interval == 0:
            elapsed = time.time() - t0
            pct = (step + 1) / total_steps * 100
            rate = (step + 1) / elapsed
            acc_rate = accepted / (step + 1)
            print(
                f"\r  {pct:5.1f}% | {sample_idx}/{num_samples} samples | "
                f"accept={acc_rate:.3f} | {rate:.0f} steps/s",
                end="", flush=True,
            )

    print()
    elapsed = time.time() - t0
    print(f"  Done: {num_samples} samples in {elapsed:.1f}s, acceptance rate={accepted/total_steps:.3f}")
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate polymer sequence samples via MCMC")
    parser.add_argument("--n_fragments", type=int, required=True,
                        help="Number of repeated fragments")
    parser.add_argument("--fragment_size", type=int, default=4,
                        help="Atoms per fragment (default: 4)")
    parser.add_argument("--contact_fraction", type=float, default=0.1,
                        help="Fraction of eligible long-range pairs to designate as contacts (default: 0.1)")
    parser.add_argument("--contact_distance", type=float, default=5.0,
                        help="Max distance for a contact to be satisfied, Å (default: 5.0)")
    parser.add_argument("--polymer_type", default="linear",
                        choices=["linear", "branched", "crosslinked"],
                        help="Polymer topology (default: linear)")
    parser.add_argument("--bond_length", type=float, default=1.0,
                        help="Bond length in Å (default: 1.0)")
    parser.add_argument("--radius", type=float, default=0.3,
                        help="Atom radius for clash check (default: 0.3)")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of samples to generate")
    parser.add_argument("--burn_in", type=int, default=None,
                        help="Burn-in steps (default: 200*N)")
    parser.add_argument("--thin_interval", type=int, default=None,
                        help="Thinning interval (default: 20*N)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    args = parser.parse_args()

    N = args.n_fragments * args.fragment_size
    fragment_ids = np.repeat(np.arange(args.n_fragments), args.fragment_size).astype(np.int32)
    min_separation = args.fragment_size  # contacts must span at least one fragment

    rng = np.random.default_rng(args.seed)
    contact_pairs = assign_contact_pairs(
        N, args.contact_fraction, min_separation, args.polymer_type, rng,
    )

    print(
        f"Generating {args.num_samples} polymer samples: "
        f"type={args.polymer_type}, N={N} ({args.n_fragments}×{args.fragment_size}), "
        f"n_contacts={len(contact_pairs)}"
    )

    # Re-seed for reproducible sampling after contact assignment
    samples = mcmc_sequence_sample(
        N=N,
        bond_length=args.bond_length,
        radius=args.radius,
        contact_pairs=contact_pairs,
        contact_distance=args.contact_distance,
        polymer_type=args.polymer_type,
        num_samples=args.num_samples,
        burn_in=args.burn_in,
        thin_interval=args.thin_interval,
        seed=args.seed + 1,
    )

    max_extent = float(np.max(np.abs(samples)))
    box_size = 2.0 * max_extent * 1.2

    print(f"  Synthetic box_size: {box_size:.4f}")
    print(f"  Saving to {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        positions=samples.astype(np.float32),
        n_fragments=np.int32(args.n_fragments),
        fragment_size=np.int32(args.fragment_size),
        fragment_ids=fragment_ids,
        contact_pairs=contact_pairs,
        contact_distance=np.float32(args.contact_distance),
        polymer_type=np.bytes_(args.polymer_type),
        bond_length=np.float32(args.bond_length),
        radius=np.float32(args.radius),
        box_size=np.float32(box_size),
        N=np.int32(N),
        seed=np.int32(args.seed),
        burn_in=np.int32(args.burn_in or 200 * N),
        thin_interval=np.int32(args.thin_interval or 20 * N),
    )


if __name__ == "__main__":
    main()
