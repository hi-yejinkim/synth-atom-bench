"""Validation script: clash check + pair correlation function g(r)."""

import argparse

import numpy as np


def check_clashes(positions: np.ndarray, radius: float) -> int:
    """Count samples with clashes (pure numpy, no torch dependency).

    Args:
        positions: (num_samples, N, 3)
        radius: atom radius

    Returns:
        Number of samples with at least one clash.
    """
    threshold_sq = (2.0 * radius) ** 2
    num_clashing = 0
    for i in range(len(positions)):
        # Pairwise squared distances
        diff = positions[i, :, None, :] - positions[i, None, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff**2, axis=-1)  # (N, N)
        np.fill_diagonal(dist_sq, np.inf)
        if np.any(dist_sq < threshold_sq):
            num_clashing += 1
    return num_clashing


def pair_correlation(
    positions: np.ndarray, box_size: float, num_bins: int = 200, r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pair correlation function g(r).

    Args:
        positions: (num_samples, N, 3)
        box_size: cubic box side length
        num_bins: number of histogram bins
        r_max: maximum r (default: box_size / 2)

    Returns:
        (r_centers, g_r) arrays.
    """
    if r_max is None:
        r_max = box_size / 2.0

    num_samples, N, _ = positions.shape
    bin_edges = np.linspace(0, r_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist = np.zeros(num_bins)

    for s in range(num_samples):
        diff = positions[s, :, None, :] - positions[s, None, :, :]  # (N, N, 3)
        dists = np.sqrt(np.sum(diff**2, axis=-1))  # (N, N)
        # Upper triangle only (avoid double counting)
        triu_idx = np.triu_indices(N, k=1)
        pair_dists = dists[triu_idx]
        h, _ = np.histogram(pair_dists, bins=bin_edges)
        hist += h

    # Normalize: g(r) = hist / (num_pairs * num_samples * shell_volume * number_density)
    num_pairs = N * (N - 1) / 2
    number_density = N / box_size**3
    shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    g_r = hist / (num_samples * num_pairs * number_density * shell_volumes)

    return bin_centers, g_r


def plot_gr(r: np.ndarray, g_r: np.ndarray, radius: float, output_path: str):
    """Plot g(r) with reference lines."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, g_r, "b-", linewidth=1.5)
    ax.axvline(2 * radius, color="r", linestyle="--", label=f"r = 2*radius = {2*radius:.2f}")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="ideal gas (g=1)")
    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("Pair Correlation Function")
    ax.legend()
    ax.set_xlim(0, r[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved g(r) plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate hard sphere samples")
    parser.add_argument("--input", type=str, required=True, help="Input .npz file")
    parser.add_argument("--output_plot", type=str, default=None, help="Output g(r) plot path")
    parser.add_argument("--num_bins", type=int, default=200, help="Number of bins for g(r)")
    args = parser.parse_args()

    data = np.load(args.input)
    positions = data["positions"]
    radius = float(data["radius"])
    box_size = float(data["box_size"])
    N = positions.shape[1]
    num_samples = positions.shape[0]

    print(f"Loaded {num_samples} samples: N={N}, radius={radius:.4f}, box_size={box_size:.4f}")
    eta = N * (4.0 / 3.0) * np.pi * radius**3 / box_size**3
    print(f"  Packing fraction eta={eta:.4f}")

    # Clash check
    print("Checking for clashes...")
    num_clashing = check_clashes(positions, radius)
    print(f"  {num_clashing}/{num_samples} samples have clashes ({num_clashing/num_samples*100:.2f}%)")

    # Pair correlation
    print("Computing g(r)...")
    r, g_r = pair_correlation(positions, box_size, num_bins=args.num_bins)

    # Check g(r) = 0 below 2*radius
    below_cutoff = r < 2 * radius
    if np.any(below_cutoff):
        max_gr_below = np.max(g_r[below_cutoff])
        print(f"  Max g(r) for r < 2*radius: {max_gr_below:.6f} (should be ~0)")

    if args.output_plot:
        plot_gr(r, g_r, radius, args.output_plot)


if __name__ == "__main__":
    main()
