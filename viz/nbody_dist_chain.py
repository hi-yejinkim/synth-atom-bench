"""Visualize chain n-body MCMC sample distributions.

Reads one or more .npz files produced by data/generate_nbody_chain.py and generates:
  1. Energy distribution (histogram) — overlaid per file
  2. Component energy decomposition (bond, LJ, angle, dihedral)
  3. Bond length distribution
  4. Bond angle distribution
  5. Dihedral angle distribution
  6. Pairwise distance distribution g(r)
  7. Energy trace plot (mixing diagnostic)
  8. Min pairwise distance histogram
  9. Autocorrelation function

Usage
-----
    # Single file
    uv run viz/nbody_dist_chain.py outputs/data/nbody_chain_N15_b4_T1.0/train.npz

    # Multiple files (T comparison)
    uv run viz/nbody_dist_chain.py outputs/data/nbody_chain_N15_b4_T0.5/train.npz \
                                   outputs/data/nbody_chain_N15_b4_T1.0/train.npz \
                                   outputs/data/nbody_chain_N15_b4_T2.0/train.npz

    # Save to specific directory
    uv run viz/nbody_dist_chain.py *.npz --outdir outputs/plots/nbody_dist_chain
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

_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B3", "#CCB974",
           "#64B5CD", "#E07B54", "#8C8C8C"]

_COMP_COLORS = {
    "Bond": "#4C72B0",
    "LJ": "#C44E52",
    "Angle": "#55A868",
    "Dihedral": "#8172B3",
}


def load_npz(path: str) -> dict:
    """Load .npz and return dict with arrays + scalar metadata."""
    data = np.load(path, allow_pickle=True)
    d = {k: data[k] for k in data.files}
    for k in list(d.keys()):
        if isinstance(d[k], np.ndarray) and d[k].ndim == 0:
            d[k] = d[k].item()
    return d


def _make_label(d: dict, path: str) -> str:
    parts = []
    if "T" in d:
        parts.append(f"T={d['T']}")
    if "N" in d:
        parts.append(f"N={d['N']}")
    if "body" in d:
        parts.append(f"b={d['body']}")
    return ", ".join(parts) if parts else Path(path).stem


def _autocorrelation(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag)
    n = len(x)
    max_lag = min(max_lag, n // 2)
    ac = np.array([np.mean(x[:n - k] * x[k:]) / var for k in range(max_lag)])
    return ac


def _compute_bond_lengths(pos: np.ndarray) -> np.ndarray:
    """Compute bond lengths for consecutive atoms. pos: (n_samples, N, 3)."""
    diff = pos[:, 1:, :] - pos[:, :-1, :]  # (n_samples, N-1, 3)
    return np.sqrt(np.sum(diff ** 2, axis=-1))  # (n_samples, N-1)


def _compute_bond_angles(pos: np.ndarray) -> np.ndarray:
    """Compute bond angles (in degrees) for triplets i-1, i, i+1."""
    b1 = pos[:, :-2, :] - pos[:, 1:-1, :]  # i-1 <- i
    b2 = pos[:, 2:, :] - pos[:, 1:-1, :]   # i+1 <- i
    cos_theta = np.sum(b1 * b2, axis=-1) / (
        np.sqrt(np.sum(b1 ** 2, axis=-1)) * np.sqrt(np.sum(b2 ** 2, axis=-1)) + 1e-12
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _compute_dihedrals(pos: np.ndarray) -> np.ndarray:
    """Compute dihedral angles (in degrees) for quadruplets i, i+1, i+2, i+3."""
    b1 = pos[:, 1:-2, :] - pos[:, :-3, :]   # j - i
    b2 = pos[:, 2:-1, :] - pos[:, 1:-2, :]   # k - j
    b3 = pos[:, 3:, :] - pos[:, 2:-1, :]     # l - k
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.sqrt(np.sum(b2 ** 2, axis=-1, keepdims=True)) + 1e-12))
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    return np.degrees(np.arctan2(y, x))


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_energy_distributions(datasets, labels, outdir):
    """1. Overlaid total energy histograms."""
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


def plot_component_decomposition(datasets, labels, outdir):
    """2. Per-component energy decomposition."""
    with synthbench_style():
        n_files = len(datasets)
        fig, axes = plt.subplots(1, n_files, figsize=(5 * n_files, 4), squeeze=False)
        for i, (d, label) in enumerate(zip(datasets, labels)):
            ax = axes[0, i]
            body = int(d.get("body", 2))
            components = [("Bond", d["energies_bond"]), ("LJ", d["energies_lj"])]
            if body >= 3:
                components.append(("Angle", d["energies_angle"]))
            if body >= 4:
                components.append(("Dihedral", d["energies_dihedral"]))

            for name, vals in components:
                c = _COMP_COLORS[name]
                ax.hist(vals, bins=50, density=True, alpha=0.4, color=c, label=name)
                ax.hist(vals, bins=50, density=True, histtype="step", linewidth=1.5, color=c)
            ax.set_xlabel("Energy")
            ax.set_ylabel("Density")
            ax.set_title(label)
            ax.legend(fontsize=9)

            # Annotate mean values
            text_lines = []
            for name, vals in components:
                text_lines.append(f"⟨{name}⟩={np.mean(vals):.2f}")
            ax.text(0.97, 0.97, "\n".join(text_lines),
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        save_figure(fig, outdir / "component_decomposition")


def plot_bond_lengths(datasets, labels, outdir):
    """3. Bond length distribution."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (d, label) in enumerate(zip(datasets, labels)):
            bl = _compute_bond_lengths(d["positions"]).ravel()
            c = _COLORS[i % len(_COLORS)]
            ax.hist(bl, bins=80, density=True, alpha=0.35, color=c, label=label)
            ax.hist(bl, bins=80, density=True, histtype="step", linewidth=1.5, color=c)
        r0 = float(datasets[0].get("r0", 1.5))
        ax.axvline(r0, color="gray", ls="--", lw=1, label=f"r₀={r0}")
        ax.set_xlabel("Bond Length")
        ax.set_ylabel("Density")
        ax.set_title("Bond Length Distribution")
        ax.legend()
        save_figure(fig, outdir / "bond_length")


def plot_bond_angles(datasets, labels, outdir):
    """4. Bond angle distribution."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (d, label) in enumerate(zip(datasets, labels)):
            angles = _compute_bond_angles(d["positions"]).ravel()
            c = _COLORS[i % len(_COLORS)]
            ax.hist(angles, bins=80, density=True, alpha=0.35, color=c, label=label)
            ax.hist(angles, bins=80, density=True, histtype="step", linewidth=1.5, color=c)
        theta0 = float(datasets[0].get("theta0", 1.911))
        ax.axvline(np.degrees(theta0), color="gray", ls="--", lw=1,
                   label=f"θ₀={np.degrees(theta0):.1f}°")
        ax.set_xlabel("Bond Angle (degrees)")
        ax.set_ylabel("Density")
        ax.set_title("Bond Angle Distribution")
        ax.legend()
        save_figure(fig, outdir / "bond_angle")


def plot_dihedrals(datasets, labels, outdir):
    """5. Dihedral angle distribution."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (d, label) in enumerate(zip(datasets, labels)):
            if int(d.get("body", 2)) < 4:
                continue
            dihedrals = _compute_dihedrals(d["positions"]).ravel()
            c = _COLORS[i % len(_COLORS)]
            ax.hist(dihedrals, bins=90, density=True, alpha=0.35, color=c,
                    label=label, range=(-180, 180))
            ax.hist(dihedrals, bins=90, density=True, histtype="step",
                    linewidth=1.5, color=c, range=(-180, 180))
        ax.axvline(180, color="gray", ls="--", lw=0.8, alpha=0.5, label="trans")
        ax.axvline(-180, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(60, color="orange", ls=":", lw=0.8, alpha=0.5, label="gauche±")
        ax.axvline(-60, color="orange", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel("Dihedral Angle (degrees)")
        ax.set_ylabel("Density")
        ax.set_title("Dihedral Angle Distribution")
        ax.set_xlim(-180, 180)
        ax.legend()
        save_figure(fig, outdir / "dihedral_angle")


def plot_pairwise_distance(datasets, labels, outdir):
    """6. Pairwise distance distribution g(r)."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (d, label) in enumerate(zip(datasets, labels)):
            pos = d["positions"]
            sigma = float(d.get("sigma", 1.0))
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
            rmax = np.percentile(all_dists, 99)
            ax.hist(all_dists, bins=100, density=True, alpha=0.35, color=c,
                    label=label, range=(0, rmax))
            ax.hist(all_dists, bins=100, density=True, histtype="step",
                    linewidth=1.5, color=c, range=(0, rmax))
        ax.axvline(sigma, color="gray", ls="--", lw=1, label=f"σ={sigma}")
        ax.set_xlabel("Pairwise Distance r")
        ax.set_ylabel("Density")
        ax.set_title("Pairwise Distance Distribution")
        ax.legend()
        save_figure(fig, outdir / "pairwise_distance")


def plot_energy_trace(datasets, labels, outdir):
    """7. Energy trace plot (mixing diagnostic)."""
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


def plot_min_distance(datasets, labels, outdir):
    """8. Min pairwise distance per sample."""
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


def plot_autocorrelation(datasets, labels, outdir):
    """9. Energy autocorrelation function."""
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


def plot_summary_stats(datasets, labels, outdir):
    """Print summary table to console."""
    header = (f"{'Label':>25} {'N':>3} {'B':>1} {'T':>4} "
              f"{'E_mean':>8} {'E_std':>6} {'acc%':>5} "
              f"{'<bond>':>6} {'<angle>':>7} {'<dih>':>7}")
    print(f"\n{'='*85}")
    print(header)
    print(f"{'-'*85}")
    for d, label in zip(datasets, labels):
        E = d["energies"]
        acc = float(d.get("acceptance_rate", 0))
        bl_mean = np.mean(_compute_bond_lengths(d["positions"]))
        body = int(d.get("body", 2))
        ang_mean = np.mean(_compute_bond_angles(d["positions"])) if body >= 3 else 0
        dih_str = ""
        if body >= 4:
            dih_mean = np.mean(np.abs(_compute_dihedrals(d["positions"])))
            dih_str = f"{dih_mean:>7.1f}"
        else:
            dih_str = f"{'—':>7}"
        print(f"{label:>25} {int(d.get('N', 0)):>3} {int(d.get('body', 0)):>1} "
              f"{float(d.get('T', 0)):>4.1f} "
              f"{np.mean(E):>8.2f} {np.std(E):>6.2f} "
              f"{acc*100:>5.1f} "
              f"{bl_mean:>6.3f} {ang_mean:>7.1f} {dih_str}")
    print(f"{'='*85}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize chain n-body MCMC distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help=".npz files from generate_nbody_chain.py")
    parser.add_argument("--outdir", type=str, default="outputs/plots/nbody_dist_chain",
                        help="Output directory for plots")
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

    has_dihedral = any(int(d.get("body", 2)) >= 4 for d in datasets)
    has_angle = any(int(d.get("body", 2)) >= 3 for d in datasets)

    plots = [
        ("energy_distribution", plot_energy_distributions),
        ("component_decomposition", plot_component_decomposition),
        ("bond_length", plot_bond_lengths),
    ]
    if has_angle:
        plots.append(("bond_angle", plot_bond_angles))
    if has_dihedral:
        plots.append(("dihedral_angle", plot_dihedrals))
    plots += [
        ("pairwise_distance", plot_pairwise_distance),
        ("energy_trace", plot_energy_trace),
        ("min_distance", plot_min_distance),
        ("autocorrelation", plot_autocorrelation),
    ]

    print("Generating plots...")
    for idx, (name, fn) in enumerate(plots, 1):
        fn(datasets, labels, outdir)
        print(f"  [{idx}/{len(plots)}] {name}")

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
