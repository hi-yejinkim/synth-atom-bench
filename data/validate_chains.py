"""Validation script for self-avoiding chain samples."""

import argparse

import numpy as np


def check_bond_lengths(positions: np.ndarray, bond_length: float, tol: float = 1e-6) -> int:
    """Count samples with any bond length violation (tight tolerance).

    Args:
        positions: (num_samples, N, 3)
        bond_length: expected distance between consecutive atoms
        tol: absolute tolerance

    Returns:
        Number of samples with at least one violated bond.
    """
    diffs = positions[:, 1:] - positions[:, :-1]  # (num_samples, N-1, 3)
    dists = np.sqrt(np.sum(diffs**2, axis=-1))  # (num_samples, N-1)
    deviations = np.abs(dists - bond_length)
    return int(np.sum(deviations.max(axis=1) > tol))


def check_nonbonded_clashes(positions: np.ndarray, radius: float) -> int:
    """Count samples with any non-bonded clash.

    Args:
        positions: (num_samples, N, 3)
        radius: atom radius

    Returns:
        Number of samples with at least one non-bonded clash.
    """
    threshold_sq = (2.0 * radius) ** 2
    N = positions.shape[1]
    num_clashing = 0

    # Build non-bonded mask (|i - j| > 1)
    idx = np.arange(N)
    nonbonded = np.abs(idx[:, None] - idx[None, :]) > 1

    for i in range(len(positions)):
        diff = positions[i, :, None, :] - positions[i, None, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff**2, axis=-1)  # (N, N)
        if np.any(dist_sq[nonbonded] < threshold_sq):
            num_clashing += 1
    return num_clashing


def end_to_end_distance(positions: np.ndarray) -> np.ndarray:
    """Compute end-to-end distance for each sample.

    Args:
        positions: (num_samples, N, 3)

    Returns:
        (num_samples,) array of end-to-end distances.
    """
    return np.sqrt(np.sum((positions[:, -1] - positions[:, 0])**2, axis=-1))


def radius_of_gyration(positions: np.ndarray) -> np.ndarray:
    """Compute radius of gyration for each sample.

    Args:
        positions: (num_samples, N, 3)

    Returns:
        (num_samples,) array of Rg values.
    """
    com = positions.mean(axis=1, keepdims=True)  # (num_samples, 1, 3)
    displacements = positions - com
    return np.sqrt(np.mean(np.sum(displacements**2, axis=-1), axis=1))


def plot_chain_diagnostics(
    positions: np.ndarray, bond_length: float, radius: float, output_path: str,
):
    """Plot bond length histogram, end-to-end distance, and Rg distributions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Bond length histogram
    diffs = positions[:, 1:] - positions[:, :-1]
    dists = np.sqrt(np.sum(diffs**2, axis=-1)).ravel()
    center = dists.mean()
    half_width = max(dists.max() - dists.min(), 0.01) * 1.5
    bin_edges = np.linspace(center - half_width, center + half_width, 51)
    axes[0].hist(dists, bins=bin_edges)
    axes[0].axvline(bond_length, color="r", linestyle="--", label=f"target={bond_length}")
    axes[0].set_xlabel("Bond length")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Bond Length Distribution")
    axes[0].legend()

    # End-to-end distance
    ete = end_to_end_distance(positions)
    N = positions.shape[1]
    # Freely-jointed chain: <R²> = (N-1) * d² => <R> ≈ sqrt(N-1) * d
    expected_rms = bond_length * np.sqrt(N - 1)
    axes[1].hist(ete, bins=50, density=True)
    axes[1].axvline(expected_rms, color="r", linestyle="--", label=f"FJC RMS={expected_rms:.2f}")
    axes[1].set_xlabel("End-to-end distance")
    axes[1].set_ylabel("Density")
    axes[1].set_title("End-to-End Distance")
    axes[1].legend()

    # Radius of gyration
    rg = radius_of_gyration(positions)
    axes[2].hist(rg, bins=50, density=True)
    axes[2].set_xlabel("Radius of gyration")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Radius of Gyration")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved diagnostics plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate self-avoiding chain samples")
    parser.add_argument("--input", type=str, required=True, help="Input .npz file")
    parser.add_argument("--output_plot", type=str, default=None, help="Output diagnostics plot path")
    args = parser.parse_args()

    data = np.load(args.input)
    positions = data["positions"]
    bond_length = float(data["bond_length"])
    radius = float(data["radius"])
    N = positions.shape[1]
    num_samples = positions.shape[0]

    print(f"Loaded {num_samples} chain samples: N={N}, bond_length={bond_length:.4f}, radius={radius:.4f}")

    # Bond length check
    print("Checking bond lengths...")
    num_violated = check_bond_lengths(positions, bond_length)
    print(f"  {num_violated}/{num_samples} samples have bond violations ({num_violated/num_samples*100:.2f}%)")

    # Non-bonded clash check
    print("Checking non-bonded clashes...")
    num_clashing = check_nonbonded_clashes(positions, radius)
    print(f"  {num_clashing}/{num_samples} samples have clashes ({num_clashing/num_samples*100:.2f}%)")

    # Chain statistics
    ete = end_to_end_distance(positions)
    rg = radius_of_gyration(positions)
    expected_rms = bond_length * np.sqrt(N - 1)
    print(f"  End-to-end distance: mean={ete.mean():.3f}, std={ete.std():.3f} (FJC RMS={expected_rms:.3f})")
    print(f"  Radius of gyration: mean={rg.mean():.3f}, std={rg.std():.3f}")

    if args.output_plot:
        import os
        os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
        plot_chain_diagnostics(positions, bond_length, radius, args.output_plot)


if __name__ == "__main__":
    main()
