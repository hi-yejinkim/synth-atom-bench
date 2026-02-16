"""MCMC sampler for self-avoiding chain configurations using pivot moves."""

import argparse
import sys
import time

import numpy as np


def nerf_place_atom(
    prev3: np.ndarray, prev2: np.ndarray, prev1: np.ndarray,
    d: float, theta: float, phi: float,
) -> np.ndarray:
    """Place atom using Natural Extension Reference Frame (NeRF) algorithm.

    Given three previous atoms and internal coordinates (bond length d,
    bond angle theta, dihedral angle phi), compute the position of the next atom.

    Args:
        prev3: position of atom i-3 (or any reference for building frame)
        prev2: position of atom i-2
        prev1: position of atom i-1 (bonded to new atom)
        d: bond length to new atom
        theta: bond angle (prev2-prev1-new) in radians
        phi: dihedral angle (prev3-prev2-prev1-new) in radians

    Returns:
        Position of the new atom (3,).
    """
    # Build local coordinate frame from last three atoms
    bc = prev1 - prev2
    bc = bc / np.linalg.norm(bc)

    ab = prev2 - prev3
    n = np.cross(ab, bc)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Degenerate case: collinear atoms, pick arbitrary perpendicular
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(bc, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        n = np.cross(bc, perp)
    n = n / np.linalg.norm(n)

    m = np.cross(n, bc)

    # New atom in local frame
    dx = d * np.cos(np.pi - theta)
    dy = d * np.sin(np.pi - theta) * np.cos(phi)
    dz = d * np.sin(np.pi - theta) * np.sin(phi)

    return prev1 + dx * bc + dy * m + dz * n


def has_nonbonded_clash(positions: np.ndarray, new_idx: int, radius: float) -> bool:
    """Check if atom new_idx clashes with any non-bonded predecessors.

    Bonded neighbors (new_idx-1) are excluded from clash check.
    """
    if new_idx <= 1:
        return False
    threshold_sq = (2.0 * radius) ** 2
    # Check against atoms 0..new_idx-2 (skip new_idx-1 which is bonded)
    diff = positions[:new_idx - 1] - positions[new_idx]
    dist_sq = np.sum(diff**2, axis=1)
    return bool(np.any(dist_sq < threshold_sq))


def initialize_chain(
    N: int, bond_length: float, radius: float, rng: np.random.Generator,
    max_attempts_per_atom: int = 1000, max_restarts: int = 100,
) -> np.ndarray:
    """Build initial chain by sequential NeRF placement with rejection.

    Args:
        N: number of atoms
        bond_length: distance between consecutive atoms
        radius: atom radius for non-bonded clash check
        rng: numpy random generator
        max_attempts_per_atom: max angle resamples per atom
        max_restarts: max full chain restarts

    Returns:
        positions: (N, 3) centered at origin
    """
    for _ in range(max_restarts):
        positions = np.zeros((N, 3))
        positions[1] = np.array([bond_length, 0.0, 0.0])

        if N <= 2:
            positions -= positions.mean(axis=0)
            return positions

        # Place a virtual atom behind atom 0 for NeRF frame
        virtual = np.array([-bond_length, 0.0, 0.0])
        success = True

        for i in range(2, N):
            placed = False
            for _ in range(max_attempts_per_atom):
                # Random bond angle: cos(theta) uniform on [-1, 1]
                cos_theta = rng.uniform(-1, 1)
                theta = np.arccos(cos_theta)
                phi = rng.uniform(0, 2 * np.pi)

                if i == 2:
                    p3 = virtual
                else:
                    p3 = positions[i - 3]

                positions[i] = nerf_place_atom(
                    p3, positions[i - 2], positions[i - 1],
                    bond_length, theta, phi,
                )

                if not has_nonbonded_clash(positions, i, radius):
                    placed = True
                    break

            if not placed:
                success = False
                break

        if success:
            positions -= positions.mean(axis=0)
            return positions

    raise RuntimeError(
        f"Failed to initialize chain of length {N} after {max_restarts} restarts."
    )


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate uniform random SO(3) rotation via QR decomposition."""
    z = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(z)
    # Ensure proper rotation (det = +1)
    d = np.diag(np.sign(np.diag(r)))
    q = q @ d
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def pivot_move(
    positions: np.ndarray, N: int, radius: float, rng: np.random.Generator,
) -> bool:
    """Attempt a single pivot move. Returns True if accepted.

    Picks a random pivot atom k (1..N-2), applies a random rotation to the
    tail (k+1..N-1) around atom k. Accepts if no new non-bonded clashes
    between tail and head atoms.
    """
    if N <= 2:
        return False

    # Pick pivot: k in [1, N-2] (need at least 1 atom on each side)
    k = rng.integers(1, N - 1)
    R = random_rotation_matrix(rng)

    # Rotate tail around pivot
    pivot = positions[k]
    tail_old = positions[k + 1:].copy()
    tail_new = (tail_old - pivot) @ R.T + pivot

    # Check non-bonded clashes between tail and head
    # Head: atoms 0..k-1 (atom k is pivot, bonded to k+1)
    # For tail atom k+1, it's bonded to k, so skip k. Check against 0..k-1.
    # For tail atoms k+2..N-1, check against 0..k (all head + pivot, but k+1 is bonded to k+2, etc.)
    # More precisely: non-bonded pairs are (i, j) where |i - j| > 1
    threshold_sq = (2.0 * radius) ** 2
    head = positions[:k]  # atoms 0..k-1
    n_head = len(head)
    n_tail = len(tail_new)

    if n_head == 0:
        # Nothing to check
        positions[k + 1:] = tail_new
        return True

    # Vectorized: compute all head-tail distances
    # head: (n_head, 3), tail_new: (n_tail, 3)
    diff = head[:, None, :] - tail_new[None, :, :]  # (n_head, n_tail, 3)
    dist_sq = np.sum(diff**2, axis=-1)  # (n_head, n_tail)

    # Mask out bonded pair: atom k-1 (last head) is bonded to k, which is bonded to k+1 (first tail)
    # So k-1 and k+1 are distance-2 along chain — NOT bonded, should be checked.
    # Only bonded pairs across head-tail boundary: none (k is pivot, k+1 is first tail,
    # k-1 is last head; k-1 and k+1 are not bonded).
    # So we check all pairs.
    if np.any(dist_sq < threshold_sq):
        return False

    positions[k + 1:] = tail_new
    return True


def mcmc_chain_sample(
    N: int,
    bond_length: float,
    radius: float,
    num_samples: int,
    burn_in: int | None = None,
    thin_interval: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Run MCMC chain sampler with pivot moves.

    Returns:
        positions: (num_samples, N, 3) centered at origin.
    """
    rng = np.random.default_rng(seed)

    if burn_in is None:
        burn_in = 100 * N
    if thin_interval is None:
        thin_interval = 10 * N

    positions = initialize_chain(N, bond_length, radius, rng)

    total_steps = burn_in + num_samples * thin_interval
    samples = np.empty((num_samples, N, 3))
    sample_idx = 0
    accepted = 0
    report_interval = max(1, total_steps // 100)

    t0 = time.time()
    for step in range(total_steps):
        if pivot_move(positions, N, radius, rng):
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
    acc_rate = accepted / total_steps
    elapsed = time.time() - t0
    print(f"  Done: {num_samples} samples in {elapsed:.1f}s, acceptance rate={acc_rate:.3f}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate self-avoiding chain samples via pivot MCMC")
    parser.add_argument("--N", type=int, required=True, help="Number of atoms in chain")
    parser.add_argument("--bond_length", type=float, default=1.0, help="Bond length (default: 1.0)")
    parser.add_argument("--radius", type=float, default=0.3, help="Atom radius (default: 0.3)")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--burn_in", type=int, default=None, help="Burn-in steps (default: 100*N)")
    parser.add_argument("--thin_interval", type=int, default=None, help="Thinning interval (default: 10*N)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} chain samples: N={args.N}, bond_length={args.bond_length}, radius={args.radius}")

    samples = mcmc_chain_sample(
        N=args.N,
        bond_length=args.bond_length,
        radius=args.radius,
        num_samples=args.num_samples,
        burn_in=args.burn_in,
        thin_interval=args.thin_interval,
        seed=args.seed,
    )

    # Compute synthetic box_size for model cutoff compatibility
    max_extent = np.max(np.abs(samples))
    box_size = 2.0 * max_extent * 1.2

    print(f"  Synthetic box_size: {box_size:.4f}")
    print(f"  Saving to {args.output}")

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez(
        args.output,
        positions=samples.astype(np.float32),
        bond_length=np.float32(args.bond_length),
        radius=np.float32(args.radius),
        box_size=np.float32(box_size),
        N=args.N,
        seed=args.seed,
        burn_in=args.burn_in or 100 * args.N,
        thin_interval=args.thin_interval or 10 * args.N,
    )


if __name__ == "__main__":
    main()
