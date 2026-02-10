"""Visualize hard sphere packing samples."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d


def draw_sphere(ax, center, radius, color="steelblue", alpha=0.6):
    """Draw a sphere on a 3D axis."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def draw_box(ax, box_size):
    """Draw wireframe cube."""
    L = box_size
    for s, e in [
        ([0, 0, 0], [L, 0, 0]), ([0, 0, 0], [0, L, 0]), ([0, 0, 0], [0, 0, L]),
        ([L, L, L], [0, L, L]), ([L, L, L], [L, 0, L]), ([L, L, L], [L, L, 0]),
        ([L, 0, 0], [L, L, 0]), ([L, 0, 0], [L, 0, L]),
        ([0, L, 0], [L, L, 0]), ([0, L, 0], [0, L, L]),
        ([0, 0, L], [L, 0, L]), ([0, 0, L], [0, L, L]),
    ]:
        ax.plot3D(*zip(s, e), color="gray", linewidth=0.5, alpha=0.4)


def plot_samples_3d(positions, radius, box_size, sample_indices, output_path):
    """Plot 3D sphere configurations for selected samples."""
    n = len(sample_indices)
    fig = plt.figure(figsize=(5 * n, 5))

    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        draw_box(ax, box_size)
        for atom in positions[idx]:
            draw_sphere(ax, atom, radius)
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_zlim(0, box_size)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Sample {idx}")
        ax.set_box_aspect([1, 1, 1])

    fig.suptitle(f"Hard Sphere Configurations (N={positions.shape[1]}, r={radius:.2f})", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 3D plot to {output_path}")


def plot_min_distance_hist(positions, radius, output_path):
    """Histogram of minimum pairwise distances across samples."""
    num_samples, N, _ = positions.shape
    min_dists = []
    for s in range(num_samples):
        diff = positions[s, :, None, :] - positions[s, None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        min_dists.append(dist.min())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(min_dists, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(2 * radius, color="red", linestyle="--", linewidth=1.5, label=f"2r = {2*radius:.2f}")
    ax.set_xlabel("Minimum pairwise distance")
    ax.set_ylabel("Count")
    ax.set_title(f"Min Distance Distribution ({num_samples} samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved min-distance histogram to {output_path}")


def plot_pairwise_distance_hist(positions, radius, output_path):
    """Histogram of all pairwise distances pooled across samples (subsample if large)."""
    num_samples, N, _ = positions.shape
    max_samples = min(num_samples, 500)
    all_dists = []
    for s in range(max_samples):
        diff = positions[s, :, None, :] - positions[s, None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        triu = dist[np.triu_indices(N, k=1)]
        all_dists.append(triu)
    all_dists = np.concatenate(all_dists)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_dists, bins=100, color="steelblue", edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(2 * radius, color="red", linestyle="--", linewidth=1.5, label=f"2r = {2*radius:.2f}")
    ax.set_xlabel("Pairwise distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Pairwise Distance Distribution (N={N}, {max_samples} samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved pairwise distance histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize hard sphere samples")
    parser.add_argument("--input", type=str, required=True, help="Input .npz file")
    parser.add_argument("--output_prefix", type=str, default="data/vis", help="Output file prefix")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2],
                        help="Sample indices to plot in 3D")
    args = parser.parse_args()

    data = np.load(args.input)
    positions = data["positions"]
    radius = float(data["radius"])
    box_size = float(data["box_size"])
    N = positions.shape[1]
    eta = N * (4.0 / 3.0) * np.pi * radius**3 / box_size**3

    print(f"Loaded {len(positions)} samples: N={N}, r={radius:.3f}, L={box_size:.3f}, eta={eta:.3f}")

    indices = [i for i in args.samples if i < len(positions)]
    plot_samples_3d(positions, radius, box_size, indices, f"{args.output_prefix}_3d.png")
    plot_min_distance_hist(positions, radius, f"{args.output_prefix}_min_dist.png")
    plot_pairwise_distance_hist(positions, radius, f"{args.output_prefix}_pairwise_dist.png")


if __name__ == "__main__":
    main()
