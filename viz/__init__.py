"""SynthBench3D visualization utilities."""

from viz.style import (
    ARCH_COLORS,
    ARCH_MARKERS,
    DOUBLE_COL,
    SINGLE_COL,
    save_figure,
    synthbench_style,
)
from viz.metrics import plot_gr, plot_min_distance_hist
from viz.scaling import fit_scaling_law, plot_capability_heatmap, plot_scaling_curves
from viz.structure import plot_structure, plot_structures_grid
from viz.training import plot_training_curves

__all__ = [
    "ARCH_COLORS",
    "ARCH_MARKERS",
    "DOUBLE_COL",
    "SINGLE_COL",
    "save_figure",
    "synthbench_style",
    "plot_gr",
    "plot_min_distance_hist",
    "fit_scaling_law",
    "plot_capability_heatmap",
    "plot_scaling_curves",
    "plot_structure",
    "plot_structures_grid",
    "plot_training_curves",
]
