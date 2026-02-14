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
from data.validate import pair_correlation
from viz.metrics import plot_gr
from viz.scaling import plot_scaling_curves
from viz.structure import plot_structure
from viz.style import save_figure, synthbench_style

OUT_DIR = _project_root / "docs" / "assets"


def hero_structures():
    """1x3 grid of 3D structures at different packing fractions."""
    settings = [
        ("outputs/data/N10_eta0.1/test.npz", "η = 0.1 (easy)"),
        ("outputs/data/N10_eta0.3/train.npz", "η = 0.3 (medium)"),
        ("outputs/data/N10_eta0.5/train.npz", "η = 0.5 (hard)"),
    ]

    with synthbench_style():
        fig = plt.figure(figsize=(15, 5))
        for i, (rel_path, title) in enumerate(settings):
            data_path = _project_root / rel_path
            if data_path.exists():
                data = np.load(data_path)
                positions = data["positions"][0]
                radius = float(data["radius"])
                box_size = float(data["box_size"])
            else:
                # Generate on the fly for missing data
                eta = float(title.split("=")[1].split()[0])
                print(f"  Generating samples for {title} (file not found)...")
                samples, box_size = mcmc_sample(
                    N=10, radius=0.5, eta=eta, num_samples=10, seed=42,
                )
                positions = samples[0]
                radius = 0.5

            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            plot_structure(positions, radius, box_size, ax=ax, title=title)

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


def pair_correlation_plot():
    """Pair correlation g(r) from training data."""
    data_path = _project_root / "outputs" / "data" / "N10_eta0.3" / "train.npz"
    if not data_path.exists():
        print("  pair_correlation: SKIPPED (no data)")
        return

    data = np.load(data_path)
    positions = data["positions"][:200]
    box_size = float(data["box_size"])
    radius = float(data["radius"])

    r, g_r = pair_correlation(positions, box_size)

    with synthbench_style():
        fig = plot_gr(r, g_r, radius, label="MCMC samples")
        save_figure(fig, OUT_DIR / "pair_correlation")
    print("  pair_correlation")


def main():
    print("Generating README figures...")
    hero_structures()
    scaling_curves()
    pair_correlation_plot()
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
