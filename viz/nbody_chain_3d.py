"""3D chain conformation visualizer for N-body chain datasets.

Plots a 2x3 grid of randomly sampled 15-atom chain configurations from a
given .npz dataset.  Atoms are colored by chain index (viridis), consecutive
atoms are connected by bond lines, and the total energy is shown in each
subplot title.

Usage
-----
    # Default: b=4, T=1.0
    uv run viz/nbody_chain_3d.py

    # Specific file
    uv run viz/nbody_chain_3d.py outputs/data/nbody_chain_N15_b3_T1.0/train.npz

    # Custom number of samples and output dir
    uv run viz/nbody_chain_3d.py outputs/data/nbody_chain_N15_b4_T2.0/train.npz \
        --n_samples 6 --outdir outputs/plots/my_dir --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from viz.style import save_figure  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(path: str | Path) -> dict:
    """Load .npz and convert 0-d arrays to Python scalars."""
    raw = np.load(path, allow_pickle=True)
    d = {}
    for k in raw.files:
        v = raw[k]
        d[k] = v.item() if (isinstance(v, np.ndarray) and v.ndim == 0) else v
    return d


def _dataset_label(d: dict, path: Path) -> str:
    parts = []
    if "body" in d:
        parts.append(f"b={d['body']}")
    if "N" in d:
        parts.append(f"N={d['N']}")
    if "T" in d:
        parts.append(f"T={d['T']}")
    return ",  ".join(parts) if parts else path.stem


# ---------------------------------------------------------------------------
# Core plot
# ---------------------------------------------------------------------------

def plot_chain_3d(
    npz_path: str | Path,
    n_samples: int = 6,
    seed: int = 0,
    outdir: str | Path | None = None,
    out_stem: str | None = None,
) -> None:
    """Draw *n_samples* 3D chain conformations in a grid and save as PNG+PDF.

    Parameters
    ----------
    npz_path:
        Path to a .npz produced by data/generate_nbody_chain.py.
    n_samples:
        Number of conformations to draw (must be a perfect rectangle, e.g. 6).
    seed:
        RNG seed for reproducible sample selection.
    outdir:
        Directory to write figures into; defaults to outputs/plots/.
    out_stem:
        Filename stem (no extension).  Defaults to ``nbody_chain_3d_<dataset_label>``.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    ds = load_npz(npz_path)
    positions: np.ndarray = ds["positions"]   # (N_total, N_atoms, 3)
    energies: np.ndarray  = ds["energies"]    # (N_total,)
    n_atoms = positions.shape[1]

    # Pick layout
    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(positions), size=n_samples, replace=False)
    sel_pos = positions[idx]        # (n_samples, n_atoms, 3)
    sel_eng = energies[idx]         # (n_samples,)

    dataset_label = _dataset_label(ds, npz_path)

    # Output path
    if outdir is None:
        outdir = REPO_ROOT / "outputs" / "plots"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if out_stem is None:
        safe = dataset_label.replace(",", "").replace(" ", "_").replace("=", "")
        out_stem = f"nbody_chain_3d_{safe}"

    # -----------------------------------------------------------------------
    # Figure
    # Note: constrained_layout conflicts with 3D axes — use manual subplots_adjust.
    # -----------------------------------------------------------------------
    with mpl.rc_context({
        "font.family":   "sans-serif",
        "font.size":     9,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.constrained_layout.use": False,
    }):
        fig = plt.figure(figsize=(14, 5 * n_rows))
        fig.suptitle(
            f"3D chain conformations  ({dataset_label})",
            fontsize=13,
            y=0.99,
        )

        for i in range(n_samples):
            ax: Axes3D = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")

            pos = sel_pos[i]                        # (n_atoms, 3)
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            atom_idx = np.arange(n_atoms)

            # Bond lines (consecutive pairs)
            for j in range(n_atoms - 1):
                ax.plot(
                    [x[j], x[j + 1]],
                    [y[j], y[j + 1]],
                    [z[j], z[j + 1]],
                    color="gray",
                    linewidth=1.4,
                    alpha=0.65,
                    zorder=1,
                )

            # Atom scatter (color = chain index)
            sc = ax.scatter(
                x, y, z,
                c=atom_idx,
                cmap="viridis",
                vmin=0,
                vmax=n_atoms - 1,
                s=60,
                edgecolors="k",
                linewidths=0.4,
                zorder=2,
            )

            energy = float(sel_eng[i])
            ax.set_title(f"E = {energy:.1f} ε", fontsize=9, pad=4)

            # Axis cosmetics
            ax.set_xlabel("x", fontsize=7, labelpad=1)
            ax.set_ylabel("y", fontsize=7, labelpad=1)
            ax.set_zlabel("z", fontsize=7, labelpad=1)
            ax.tick_params(axis="both", which="major", labelsize=5, pad=0)
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("lightgray")
            ax.yaxis.pane.set_edgecolor("lightgray")
            ax.zaxis.pane.set_edgecolor("lightgray")
            try:
                ax.set_box_aspect([1, 1, 1])
            except AttributeError:
                pass  # matplotlib < 3.3

        # Shared colorbar (atom index 0 → n_atoms-1)
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=mpl.colors.Normalize(vmin=0, vmax=n_atoms - 1),
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=fig.axes,
            orientation="vertical",
            fraction=0.015,
            pad=0.02,
            shrink=0.55,
        )
        cbar.set_label("Atom index", fontsize=9)
        mid = (n_atoms - 1) // 2
        cbar.set_ticks([0, mid, n_atoms - 1])
        cbar.set_ticklabels(["0", str(mid), str(n_atoms - 1)])

        fig.subplots_adjust(
            left=0.04, right=0.88,
            top=0.94, bottom=0.04,
            wspace=0.12, hspace=0.28,
        )

        out_path = outdir / out_stem
        save_figure(fig, out_path)
        print(f"Saved  {out_path}.png")
        print(f"Saved  {out_path}.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    default_npz = REPO_ROOT / "outputs" / "data" / "nbody_chain_N15_b4_T1.0" / "train.npz"

    parser = argparse.ArgumentParser(
        description="Visualize 3D chain conformations from an N-body chain dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "npz",
        nargs="?",
        default=str(default_npz),
        help=f".npz file (default: {default_npz.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--n_samples", type=int, default=6,
        help="Number of conformations to plot (default: 6)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for sample selection (default: 0)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: outputs/plots/)",
    )
    parser.add_argument(
        "--out_stem", type=str, default=None,
        help="Output filename stem without extension (default: auto)",
    )
    args = parser.parse_args()

    plot_chain_3d(
        npz_path=args.npz,
        n_samples=args.n_samples,
        seed=args.seed,
        outdir=args.outdir,
        out_stem=args.out_stem,
    )


if __name__ == "__main__":
    main()
