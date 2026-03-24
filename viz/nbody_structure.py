"""Visualize n-body MCMC 3D structures across temperatures.

Renders atom configurations as 3D sphere plots with clean axes.
Designed for comparing multiple temperature datasets side-by-side.

Usage
-----
    # Single dataset
    uv run viz/nbody_structure.py outputs/data/nbody_n15_b2_T1.0/train.npz

    # Compare temperatures
    uv run viz/nbody_structure.py \
        outputs/data/nbody_n15_b2_T0.5/train.npz \
        outputs/data/nbody_n15_b2_T1.0/train.npz \
        outputs/data/nbody_n15_b2_T2.0/train.npz
        
    uv run viz/nbody_structure.py outputs/data/nbody_n15_b2_T*.0/train.npz

    # Custom output directory and sample count
    uv run viz/nbody_structure.py *.npz --outdir outputs/plots/struct --n_samples 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz.style import save_figure, synthbench_style

# Soft blue for all atoms (no clash coloring for LJ potential systems)
_ATOM_COLOR = "#6BAED6"
_ATOM_ALPHA = 0.5
_EDGE_ALPHA = 0.9


def _draw_sphere(ax, center, radius, color=_ATOM_COLOR,
                 alpha=_ATOM_ALPHA, edge_alpha=_EDGE_ALPHA):
    """Draw a translucent sphere on a 3D axes."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha,
                    edgecolor=color, linewidth=0.3 * edge_alpha, shade=True)


def _draw_box(ax, box_size: float):
    """Draw wireframe cube."""
    L = box_size
    corners = np.array([
        [0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0],
        [0, 0, L], [L, 0, L], [L, L, L], [0, L, L],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]),
                  color="gray", linewidth=0.4, linestyle="--", alpha=0.4)


def _clean_3d_axes(ax, box_size: float):
    """Remove clutter from 3D axes for clean appearance."""
    # Light gray panes
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor((0.97, 0.97, 0.97, 0.2))
        pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))

    # Set limits with small margin
    margin = 0.1
    ax.set_xlim(-margin, box_size + margin)
    ax.set_ylim(-margin, box_size + margin)
    ax.set_zlim(-margin, box_size + margin)

    # Minimal tick labels
    ticks = [0, box_size / 2, box_size]
    tick_labels = ["0", f"{box_size / 2:.1f}", f"{box_size:.1f}"]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_zticklabels(tick_labels, fontsize=6)

    # Hide axis labels (redundant with ticks)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)


def plot_single(ax, positions: np.ndarray, display_radius: float,
                box_size: float, title: str | None = None):
    """Plot one 3D configuration on given axes."""
    _draw_box(ax, box_size)
    for pos in positions:
        _draw_sphere(ax, pos, display_radius)
    _clean_3d_axes(ax, box_size)
    if title:
        ax.set_title(title, fontsize=11, pad=2)


def plot_temperature_comparison(
    npz_paths: list[str],
    n_samples: int = 4,
    outdir: str = "outputs/plots/nbody_structure",
    seed: int = 42,
) -> None:
    """Generate comparison grid: rows=temperatures, cols=random samples."""
    outdir = Path(outdir)
    rng = np.random.default_rng(seed)

    # Load all datasets
    datasets = []
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        info = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
        datasets.append(info)

    n_temps = len(datasets)

    # Use display_radius = sigma * 0.35 so spheres don't visually overlap
    sigma = float(datasets[0].get("sigma", 1.0))
    display_radius = sigma * 0.35

    with synthbench_style():
        fig = plt.figure(figsize=(4 * n_samples, 4 * n_temps))

        for row, d in enumerate(datasets):
            pos_all = d["positions"]
            box_size = float(d.get("box_size", 3.5))
            T = float(d.get("T", 1.0))
            n_atoms = int(d.get("n", pos_all.shape[1]))

            # Pick random samples (same seed per temperature for consistency)
            idx = rng.choice(len(pos_all), n_samples, replace=False)

            for col, si in enumerate(idx):
                ax = fig.add_subplot(
                    n_temps, n_samples, row * n_samples + col + 1,
                    projection="3d",
                )
                title = f"T={T}" if col == 0 else ""
                plot_single(ax, pos_all[si], display_radius, box_size, title=title)

                # Column header on first row
                if row == 0:
                    ax.text2D(0.5, 1.05, f"Sample {col + 1}",
                              transform=ax.transAxes, ha="center", fontsize=10,
                              color="gray")

        fig.suptitle(
            f"N={n_atoms}, {int(datasets[0].get('body', 2))}-body LJ  |  "
            f"display r = {display_radius:.2f}σ  (no clash)",
            fontsize=13, y=1.02,
        )
        save_figure(fig, outdir / "structure_comparison")
        print(f"Saved to {outdir}/structure_comparison.{{pdf,png}}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize n-body 3D structures across temperatures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help=".npz dataset files")
    parser.add_argument("--outdir", default="outputs/plots/nbody_structure")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of random samples per temperature")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plot_temperature_comparison(
        args.files,
        n_samples=args.n_samples,
        outdir=args.outdir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
