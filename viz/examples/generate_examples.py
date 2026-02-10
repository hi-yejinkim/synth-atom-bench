"""Generate example plots for visual QA."""

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.validate import pair_correlation
from viz.style import ARCH_COLORS, synthbench_style, save_figure
from viz.structure import plot_structures_grid
from viz.metrics import plot_gr, plot_min_distance_hist
from viz.scaling import plot_scaling_curves, plot_capability_heatmap

OUT_DIR = _project_root / "outputs" / "plots" / "examples"


def _load_test_data():
    path = _project_root / "outputs" / "data" / "N10_eta0.1" / "test_easy.npz"
    data = np.load(path)
    return data["positions"], float(data["radius"]), float(data["box_size"])


def example_structure_grid():
    positions, radius, box_size = _load_test_data()
    with synthbench_style():
        fig = plot_structures_grid(
            [positions[i] for i in range(4)],
            radius, box_size, ncols=4,
        )
        save_figure(fig, OUT_DIR / "structure_grid")
    print("  structure_grid")


def example_pair_correlation():
    positions, radius, box_size = _load_test_data()
    r, g_r = pair_correlation(positions[:200], box_size)
    with synthbench_style():
        fig = plot_gr(r, g_r, radius)
        save_figure(fig, OUT_DIR / "pair_correlation")
    print("  pair_correlation")


def example_min_distance_hist():
    positions, radius, _ = _load_test_data()
    with synthbench_style():
        fig = plot_min_distance_hist(positions[:500], radius)
        save_figure(fig, OUT_DIR / "min_distance_hist")
    print("  min_distance_hist")


def example_scaling_curves():
    """Generate dummy scaling curves for 4 architectures."""
    rng = np.random.default_rng(42)
    results = {}
    flops_vals = np.array([1e15, 4e15, 1.6e16, 6.4e16, 2.56e17])
    for i, arch in enumerate(["PaiNN", "Transformer", "Pairformer"]):
        alpha = 0.3 + 0.15 * i
        a = 2.0 - 0.2 * i
        floor = 0.01 + 0.005 * i
        cr = a * flops_vals ** (-alpha) + floor
        cr += rng.normal(0, 0.002, size=cr.shape)
        cr = np.clip(cr, 1e-4, 1.0)
        results[arch] = {
            "flops": flops_vals,
            "clash_rate": cr,
            "clash_rate_std": np.abs(rng.normal(0, 0.005, size=cr.shape)),
        }

    with synthbench_style():
        fig = plot_scaling_curves(results)
        save_figure(fig, OUT_DIR / "scaling_curves_dummy")
    print("  scaling_curves_dummy")


def example_capability_heatmap():
    """Generate dummy capability heatmap."""
    rng = np.random.default_rng(123)
    archs = ["PaiNN", "Transformer", "Pairformer"]
    datasets = ["N10 \u03b7=0.1", "N10 \u03b7=0.3", "N50 \u03b7=0.3", "N10 \u03b7=0.5"]
    data = rng.uniform(0.01, 0.6, size=(3, 4))
    # Make harder datasets have higher clash rates
    data[:, 2:] += 0.15
    data = np.clip(data, 0, 1)

    with synthbench_style():
        fig = plot_capability_heatmap(data, architectures=archs, datasets=datasets)
        save_figure(fig, OUT_DIR / "capability_heatmap_dummy")
    print("  capability_heatmap_dummy")


def main():
    print("Generating example plots...")
    example_structure_grid()
    example_pair_correlation()
    example_min_distance_hist()
    example_scaling_curves()
    example_capability_heatmap()
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
