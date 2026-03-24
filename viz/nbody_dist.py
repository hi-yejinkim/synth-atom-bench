"""Visualize n-body MCMC sample distributions.

Reads one or more .npz files produced by data/generate_nbody.py and generates:
  1. Energy distribution (histogram + KDE) — overlaid per file
  2. Per-body energy decomposition (V2, V3, V4)
  3. Pairwise distance distribution g(r)
  4. Energy trace plot (mixing diagnostic)
  5. Min pairwise distance histogram
  6. Autocorrelation function

Usage
-----
    # Single file
    uv run viz/nbody_dist.py outputs/data/nbody_n15_b2_T1.0/train.npz

    # Multiple files (T comparison)
    uv run viz/nbody_dist.py outputs/data/nbody_n15_b2_T0.5/train.npz \
                             outputs/data/nbody_n15_b2_T1.0/train.npz \
                             outputs/data/nbody_n15_b2_T2.0/train.npz

    # Save to specific directory
    uv run viz/nbody_dist.py *.npz --outdir outputs/plots/nbody_dist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz.style import save_figure, synthbench_style

# Distinct colors for multi-file comparison
_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B3", "#CCB974",
           "#64B5CD", "#E07B54", "#8C8C8C"]


def load_npz(path: str) -> dict:
    """Load .npz and return dict with arrays + scalar metadata."""
    data = np.load(path, allow_pickle=True)
    d = {k: data[k] for k in data.files}
    # Convert 0-d arrays to scalars
    for k in list(d.keys()):
        if isinstance(d[k], np.ndarray) and d[k].ndim == 0:
            d[k] = d[k].item()
    return d


def _make_label(d: dict, path: str) -> str:
    """Auto-generate a concise label from metadata."""
    parts = []
    if "T" in d:
        parts.append(f"T={d['T']}")
    if "n" in d:
        parts.append(f"N={d['n']}")
    if "body" in d:
        parts.append(f"b={d['body']}")
    return ", ".join(parts) if parts else Path(path).stem


def _autocorrelation(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Compute normalized autocorrelation function up to max_lag."""
    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag)
    n = len(x)
    max_lag = min(max_lag, n // 2)
    ac = np.array([np.mean(x[:n - k] * x[k:]) / var for k in range(max_lag)])
    return ac


def plot_energy_distributions(datasets: list[dict], labels: list[str],
                              outdir: Path) -> None:
    """1. Overlaid energy histograms."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (d, label) in enumerate(zip(datasets, labels)):
            E = d["energies"]
            c = _COLORS[i % len(_COLORS)]
            ax.hist(E, bins=60, density=True, alpha=0.35, color=c, label=label)
            ax.hist(E, bins=60, density=True, histtype="step", linewidth=1.5, color=c)
        ax.set_xlabel("Total Energy")
        ax.set_ylabel("Density")
        ax.set_title("Energy Distribution")
        ax.legend()
        save_figure(fig, outdir / "energy_distribution")


def plot_body_decomposition(datasets: list[dict], labels: list[str],
                            outdir: Path) -> None:
    """2. Per-body energy decomposition."""
    with synthbench_style():
        n_files = len(datasets)
        fig, axes = plt.subplots(1, n_files, figsize=(4 * n_files, 4),
                                 squeeze=False)
        for i, (d, label) in enumerate(zip(datasets, labels)):
            ax = axes[0, i]
            body = int(d.get("body", 2))
            components = [("V2", d["energies_2body"], "#4C72B0")]
            if body >= 3:
                components.append(("V3", d["energies_3body"], "#C44E52"))
            if body >= 4:
                components.append(("V4", d["energies_4body"], "#55A868"))

            for name, vals, c in components:
                ax.hist(vals, bins=50, density=True, alpha=0.4, color=c, label=name)
                ax.hist(vals, bins=50, density=True, histtype="step",
                        linewidth=1.5, color=c)
            ax.set_xlabel("Energy")
            ax.set_ylabel("Density")
            ax.set_title(label)
            ax.legend(fontsize=9)

            # Annotate ratios
            v2_abs = np.mean(np.abs(d["energies_2body"]))
            if body >= 3 and v2_abs > 0:
                v3_abs = np.mean(np.abs(d["energies_3body"]))
                ratio = v3_abs / v2_abs * 100
                ax.text(0.97, 0.97, f"|V3|/|V2|={ratio:.1f}%",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        save_figure(fig, outdir / "body_decomposition")


def plot_pairwise_distance(datasets: list[dict], labels: list[str],
                           outdir: Path) -> None:
    """3. Pairwise distance distribution g(r)."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (d, label) in enumerate(zip(datasets, labels)):
            pos = d["positions"]
            sigma = float(d.get("sigma", 1.0))
            # Sample up to 500 configs for speed
            idx = np.random.default_rng(0).choice(len(pos), min(500, len(pos)),
                                                   replace=False)
            all_dists = []
            for j in idx:
                diff = pos[j][:, None, :] - pos[j][None, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=-1))
                triu = dist[np.triu_indices(len(pos[j]), k=1)]
                all_dists.append(triu)
            all_dists = np.concatenate(all_dists)
            c = _COLORS[i % len(_COLORS)]
            ax.hist(all_dists, bins=100, density=True, alpha=0.35, color=c,
                    label=label, range=(0, float(d.get("box_size", 5.0)) / 2))
            ax.hist(all_dists, bins=100, density=True, histtype="step",
                    linewidth=1.5, color=c,
                    range=(0, float(d.get("box_size", 5.0)) / 2))
        ax.axvline(sigma, color="gray", ls="--", lw=1, label=f"σ={sigma}")
        ax.set_xlabel("Pairwise Distance r")
        ax.set_ylabel("Density")
        ax.set_title("Pairwise Distance Distribution")
        ax.legend()
        save_figure(fig, outdir / "pairwise_distance")


def plot_energy_trace(datasets: list[dict], labels: list[str],
                      outdir: Path) -> None:
    """4. Energy trace plot (mixing diagnostic)."""
    with synthbench_style():
        n_files = len(datasets)
        fig, axes = plt.subplots(n_files, 1, figsize=(8, 2.5 * n_files),
                                 squeeze=False)
        for i, (d, label) in enumerate(zip(datasets, labels)):
            ax = axes[i, 0]
            E = d["energies"]
            c = _COLORS[i % len(_COLORS)]
            ax.plot(E, color=c, lw=0.5, alpha=0.8)
            ax.axhline(np.mean(E), color=c, ls="--", lw=1, alpha=0.6)
            ax.set_ylabel("Energy")
            ax.set_title(f"{label} (mean={np.mean(E):.2f}, std={np.std(E):.2f})")
        axes[-1, 0].set_xlabel("Sample Index")
        fig.suptitle("Energy Trace", fontsize=14, y=1.01)
        save_figure(fig, outdir / "energy_trace")


def plot_min_distance(datasets: list[dict], labels: list[str],
                      outdir: Path) -> None:
    """5. Min pairwise distance per sample."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (d, label) in enumerate(zip(datasets, labels)):
            pos = d["positions"]
            sigma = float(d.get("sigma", 1.0))
            n_check = min(len(pos), 1000)
            min_dists = np.empty(n_check)
            for j in range(n_check):
                diff = pos[j][:, None, :] - pos[j][None, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=-1))
                np.fill_diagonal(dist, 999.0)
                min_dists[j] = dist.min()
            c = _COLORS[i % len(_COLORS)]
            ax.hist(min_dists, bins=50, density=True, alpha=0.35, color=c,
                    label=label)
            ax.hist(min_dists, bins=50, density=True, histtype="step",
                    linewidth=1.5, color=c)
        ax.axvline(sigma, color="gray", ls="--", lw=1, label=f"σ={sigma}")
        ax.set_xlabel("Min Pairwise Distance")
        ax.set_ylabel("Density")
        ax.set_title("Minimum Distance Distribution")
        ax.legend()
        save_figure(fig, outdir / "min_distance")


def plot_autocorrelation(datasets: list[dict], labels: list[str],
                         outdir: Path) -> None:
    """6. Energy autocorrelation function."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (d, label) in enumerate(zip(datasets, labels)):
            E = d["energies"]
            max_lag = min(200, len(E) // 4)
            if max_lag < 5:
                continue
            ac = _autocorrelation(E, max_lag)
            c = _COLORS[i % len(_COLORS)]
            ax.plot(ac, color=c, lw=1.5, label=label)
        ax.axhline(0, color="gray", ls="-", lw=0.5)
        ax.axhline(0.05, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax.axhline(-0.05, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Energy Autocorrelation")
        ax.legend()
        save_figure(fig, outdir / "autocorrelation")


def plot_summary_stats(datasets: list[dict], labels: list[str],
                       outdir: Path) -> None:
    """Print summary table to console."""
    header = (f"{'Label':>20} {'N':>3} {'B':>1} {'T':>4} "
              f"{'E_mean':>8} {'E_std':>6} {'acc%':>5} {'min_d':>5}")
    print(f"\n{'='*70}")
    print(header)
    print(f"{'-'*70}")
    for d, label in zip(datasets, labels):
        E = d["energies"]
        pos = d["positions"]
        acc = float(d.get("acceptance_rate_production",
                          d.get("acceptance_rate", 0)))
        # min dist from first 100 samples
        md = 999.0
        for j in range(min(100, len(pos))):
            diff = pos[j][:, None, :] - pos[j][None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dist, 999.0)
            md = min(md, dist.min())
        print(f"{label:>20} {int(d.get('n', 0)):>3} {int(d.get('body', 0)):>1} "
              f"{float(d.get('T', 0)):>4.1f} "
              f"{np.mean(E):>8.2f} {np.std(E):>6.2f} "
              f"{acc*100:>5.1f} {md:>5.3f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize n-body MCMC distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help=".npz files from generate_nbody.py")
    parser.add_argument("--outdir", type=str, default="outputs/plots/nbody_dist",
                        help="Output directory for plots (default: outputs/plots/nbody_dist)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    datasets = []
    labels = []
    for f in args.files:
        d = load_npz(f)
        datasets.append(d)
        labels.append(_make_label(d, f))

    print(f"Loaded {len(datasets)} dataset(s)")
    plot_summary_stats(datasets, labels, outdir)

    print("Generating plots...")
    plot_energy_distributions(datasets, labels, outdir)
    print("  [1/6] energy_distribution")
    plot_body_decomposition(datasets, labels, outdir)
    print("  [2/6] body_decomposition")
    plot_pairwise_distance(datasets, labels, outdir)
    print("  [3/6] pairwise_distance")
    plot_energy_trace(datasets, labels, outdir)
    print("  [4/6] energy_trace")
    plot_min_distance(datasets, labels, outdir)
    print("  [5/6] min_distance")
    plot_autocorrelation(datasets, labels, outdir)
    print("  [6/6] autocorrelation")

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
