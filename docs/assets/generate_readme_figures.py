"""Generate figures for README.md."""

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt

from data.generate import mcmc_sample
from data.generate_chains import mcmc_chain_sample
from viz.scaling import plot_scaling_curves
from viz.structure import plot_structure
from viz.style import save_figure, synthbench_style

OUT_DIR = _project_root / "docs" / "assets"


def _load_or_generate_spheres(N, eta, radius=0.5, seed=42):
    """Load hard sphere data from outputs, or generate on the fly."""
    data_path = _project_root / f"outputs/data/N{N}_eta{eta}/train.npz"
    if data_path.exists():
        data = np.load(data_path)
        return data["positions"][0], float(data["radius"]), float(data["box_size"])

    print(f"  Generating hard spheres N={N}, eta={eta} (no cached data)...")
    samples, box_size = mcmc_sample(N=N, radius=radius, eta=eta, num_samples=5, seed=seed)
    return samples[0], radius, box_size


def _load_or_generate_chain(N, bond_length=1.0, radius=0.3, seed=42):
    """Load chain data from outputs, or generate on the fly."""
    data_path = _project_root / f"outputs/data/chain_N{N}/train.npz"
    if data_path.exists():
        data = np.load(data_path)
        return data["positions"][0], float(data["radius"])

    print(f"  Generating chain N={N} (no cached data)...")
    samples = mcmc_chain_sample(
        N=N, bond_length=bond_length, radius=radius, num_samples=5, seed=seed,
    )
    return samples[0], radius


def hero_structures():
    """2x3 grid: hard spheres (top) and chains (bottom) at N=10, 20, 50."""
    Ns = [10, 20, 50]

    with synthbench_style():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                                 subplot_kw={"projection": "3d"},
                                 layout="none")

        # Top row: hard spheres
        for col, N in enumerate(Ns):
            positions, radius, box_size = _load_or_generate_spheres(N, eta=0.3)
            ax = axes[0, col]
            plot_structure(positions, radius, box_size, ax=ax, title=f"N = {N}")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")

        # Bottom row: chains
        for col, N in enumerate(Ns):
            positions, radius = _load_or_generate_chain(N)
            bonds = [(i, i + 1) for i in range(N - 1)]
            ax = axes[1, col]
            plot_structure(
                positions, radius, box_size=None, ax=ax, title=f"N = {N}",
                bonds=bonds, draw_box=False,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")

        # Row labels
        fig.text(0.02, 0.72, "Hard Sphere\nPacking", va="center", ha="center",
                 fontsize=14, fontweight="bold", rotation=90)
        fig.text(0.02, 0.28, "Self-Avoiding\nChains", va="center", ha="center",
                 fontsize=14, fontweight="bold", rotation=90)

        fig.subplots_adjust(left=0.07, wspace=0.05, hspace=0.12)
        save_figure(fig, OUT_DIR / "hero_structures")
    print("  hero_structures")


def scaling_curves():
    """Scaling curves from real experiment results."""
    results_path = _project_root / "outputs" / "scaling" / "results.json"
    if not results_path.exists():
        print("  scaling_curves: SKIPPED (no results.json)")
        return

    with open(results_path) as f:
        raw = json.load(f)

    best = raw["best_per_budget"]

    # Group by architecture
    arch_name_map = {"painn": "PaiNN", "transformer": "Transformer", "pairformer": "Pairformer"}
    arch_data = {}
    for entry in best.values():
        arch = entry["arch"]
        display_name = arch_name_map.get(arch, arch)
        if display_name not in arch_data:
            arch_data[display_name] = {"flops": [], "clash_rate": []}
        arch_data[display_name]["flops"].append(entry["total_flops"])
        arch_data[display_name]["clash_rate"].append(entry["best_clash_rate"])

    # Sort by flops and convert to arrays
    results = {}
    for name, data in arch_data.items():
        order = np.argsort(data["flops"])
        results[name] = {
            "flops": np.array(data["flops"])[order],
            "clash_rate": np.array(data["clash_rate"])[order],
        }

    with synthbench_style():
        fig = plot_scaling_curves(results, fit_curves=True)
        save_figure(fig, OUT_DIR / "scaling_curves")
    print("  scaling_curves")


def main():
    print("Generating README figures...")
    hero_structures()
    scaling_curves()
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
