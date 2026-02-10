"""MCMC hard sphere sampler using Metropolis-Hastings."""

import argparse
import sys
import time

import numpy as np


def compute_box_size(N: int, radius: float, eta: float) -> float:
    """Compute box side length from packing fraction: eta = N * (4/3) * pi * r^3 / L^3."""
    return (N * (4.0 / 3.0) * np.pi * radius**3 / eta) ** (1.0 / 3.0)


def has_overlap(positions: np.ndarray, idx: int, threshold_sq: float) -> bool:
    """Check if atom idx overlaps with any other atom. Uses squared distances."""
    diff = positions - positions[idx]
    dist_sq = np.sum(diff**2, axis=1)
    dist_sq[idx] = np.inf  # exclude self
    return np.any(dist_sq < threshold_sq)


def initialize_positions(
    N: int, box_size: float, threshold_sq: float, rng: np.random.Generator,
    max_attempts: int = 100_000, max_restarts: int = 100,
) -> np.ndarray:
    """Sequential random placement with rejection."""
    for restart in range(max_restarts):
        positions = np.empty((N, 3))
        success = True
        for i in range(N):
            placed = False
            for _ in range(max_attempts):
                positions[i] = rng.uniform(0, box_size, size=3)
                if i == 0 or not has_overlap(positions[: i + 1], i, threshold_sq):
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            return positions
    raise RuntimeError(
        f"Failed to initialize {N} atoms after {max_restarts} restarts. "
        f"Packing fraction may be too high."
    )


def mcmc_sample(
    N: int,
    radius: float,
    eta: float,
    num_samples: int,
    burn_in: int = 10_000,
    thin_interval: int | None = None,
    step_size: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Run MCMC hard sphere sampler.

    Returns (positions array of shape (num_samples, N, 3), box_size).
    """
    rng = np.random.default_rng(seed)
    box_size = compute_box_size(N, radius, eta)
    threshold_sq = (2.0 * radius) ** 2
    delta = step_size * box_size

    if thin_interval is None:
        thin_interval = N * 100

    positions = initialize_positions(N, box_size, threshold_sq, rng)

    total_steps = burn_in + num_samples * thin_interval
    samples = np.empty((num_samples, N, 3))
    sample_idx = 0
    accepted = 0
    total_proposals = 0
    report_interval = max(1, total_steps // 100)

    t0 = time.time()
    for step in range(total_steps):
        # Pick random atom
        atom = rng.integers(N)
        old_pos = positions[atom].copy()

        # Propose displacement
        positions[atom] += rng.uniform(-delta, delta, size=3)
        total_proposals += 1

        # Reject if outside box or overlapping
        if (
            np.any(positions[atom] < 0)
            or np.any(positions[atom] >= box_size)
            or has_overlap(positions, atom, threshold_sq)
        ):
            positions[atom] = old_pos
        else:
            accepted += 1

        # Collect sample after burn-in
        if step >= burn_in and (step - burn_in) % thin_interval == 0:
            samples[sample_idx] = positions.copy()
            sample_idx += 1

        if (step + 1) % report_interval == 0:
            elapsed = time.time() - t0
            pct = (step + 1) / total_steps * 100
            rate = (step + 1) / elapsed
            acc_rate = accepted / total_proposals if total_proposals > 0 else 0
            print(
                f"\r  {pct:5.1f}% | {sample_idx}/{num_samples} samples | "
                f"accept={acc_rate:.3f} | {rate:.0f} steps/s",
                end="", flush=True,
            )

    print()  # newline after progress
    acc_rate = accepted / total_proposals
    elapsed = time.time() - t0
    print(f"  Done: {num_samples} samples in {elapsed:.1f}s, acceptance rate={acc_rate:.3f}")
    return samples, box_size


def main():
    parser = argparse.ArgumentParser(description="Generate hard sphere packing samples via MCMC")
    parser.add_argument("--N", type=int, required=True, help="Number of atoms")
    parser.add_argument("--eta", type=float, required=True, help="Packing fraction")
    parser.add_argument("--radius", type=float, default=0.5, help="Atom radius (default: 0.5)")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--burn_in", type=int, default=10_000, help="Burn-in steps")
    parser.add_argument("--thin_interval", type=int, default=None, help="Thinning interval (default: N*100)")
    parser.add_argument("--step_size", type=float, default=0.1, help="Step size as fraction of box size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples: N={args.N}, eta={args.eta}, radius={args.radius}")

    samples, box_size = mcmc_sample(
        N=args.N,
        radius=args.radius,
        eta=args.eta,
        num_samples=args.num_samples,
        burn_in=args.burn_in,
        thin_interval=args.thin_interval,
        step_size=args.step_size,
        seed=args.seed,
    )

    print(f"  Box size: {box_size:.4f}")
    print(f"  Saving to {args.output}")

    np.savez(
        args.output,
        positions=samples.astype(np.float32),
        radius=np.float32(args.radius),
        box_size=np.float32(box_size),
        N=args.N,
        eta=args.eta,
        seed=args.seed,
        burn_in=args.burn_in,
        thin_interval=args.thin_interval or args.N * 100,
        step_size=args.step_size,
    )


if __name__ == "__main__":
    main()
