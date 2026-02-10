"""Pair correlation g(r) and minimum distance histogram plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_gr(
    r: np.ndarray,
    g_r: np.ndarray,
    radius: float,
    gt_r: np.ndarray | None = None,
    gt_g_r: np.ndarray | None = None,
    color: str = "#4C72B0",
    label: str = "Generated",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot pair correlation function g(r).

    Args:
        r: bin centers.
        g_r: g(r) values.
        radius: atom radius (exclusion at 2*radius).
        gt_r: optional ground truth bin centers.
        gt_g_r: optional ground truth g(r).
        color: line color for generated curve.
        label: legend label for generated curve.
        ax: optional existing axes.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    else:
        fig = ax.get_figure()

    sigma = 2.0 * radius

    # Normalize x-axis by sigma
    r_norm = r / sigma
    ax.plot(r_norm, g_r, color=color, linewidth=2.0, label=label)

    if gt_r is not None and gt_g_r is not None:
        gt_r_norm = gt_r / sigma
        ax.plot(gt_r_norm, gt_g_r, color="black", linestyle="--", linewidth=1.5,
                label="Ground truth")
        # Shading between generated and GT (interpolate to common grid)
        r_common = np.intersect1d(
            np.round(r_norm, 6), np.round(gt_r_norm, 6),
        )
        if len(r_common) == 0:
            # Fall back to simpler shading on the generated grid
            ax.fill_between(r_norm, g_r, 1.0, alpha=0.15, color=color)
        else:
            g_interp = np.interp(r_common, r_norm, g_r)
            gt_interp = np.interp(r_common, gt_r_norm, gt_g_r)
            ax.fill_between(r_common, g_interp, gt_interp, alpha=0.15, color=color)

    # Exclusion radius line
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
               label="exclusion radius")

    ax.set_xlabel(r"$r\,/\,\sigma$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_xlim(0, r_norm[-1])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title("Pair Correlation Function")

    return fig


def plot_min_distance_hist(
    positions: np.ndarray,
    radius: float,
    ax: plt.Axes | None = None,
    color: str = "#4C72B0",
    label: str = "Min distance",
    num_bins: int = 60,
) -> plt.Figure:
    """Histogram of per-sample minimum pairwise distances.

    Args:
        positions: (num_samples, N, 3) atom positions.
        radius: atom radius.
        ax: optional existing axes.
        color: histogram color.
        label: legend label.
        num_bins: number of histogram bins.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    else:
        fig = ax.get_figure()

    # Compute per-sample minimum pairwise distance
    num_samples, N, _ = positions.shape
    min_dists = np.empty(num_samples)
    for i in range(num_samples):
        diff = positions[i, :, None, :] - positions[i, None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        min_dists[i] = np.sqrt(np.min(dist_sq))

    threshold = 2.0 * radius

    # Clash region shading
    ax.axvspan(0, threshold, alpha=0.08, color="red", zorder=0)

    # Histogram
    ax.hist(min_dists, bins=num_bins, color=color, alpha=0.5, edgecolor=color,
            linewidth=0.8, label=label, density=True)

    # Clash threshold line
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"clash threshold (2r = {threshold:.3f})")

    ax.set_xlabel("Minimum pairwise distance")
    ax.set_ylabel("Density")
    ax.set_title("Minimum Distance Distribution")
    ax.legend(frameon=False, fontsize=9)

    return fig
