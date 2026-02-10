"""3D atom structure visualization."""

import numpy as np
import matplotlib.pyplot as plt


def _find_clashing_atoms(positions: np.ndarray, radius: float) -> np.ndarray:
    """Return boolean mask (N,) of atoms involved in at least one clash.

    Pure numpy — no torch dependency.
    """
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff**2, axis=-1)  # (N, N)
    np.fill_diagonal(dist_sq, np.inf)
    threshold_sq = (2.0 * radius) ** 2
    return np.any(dist_sq < threshold_sq, axis=1)  # (N,)


def _draw_sphere(ax, center, radius, color, alpha=0.4, edge_alpha=0.8):
    """Draw a translucent sphere with edge outline on a 3D axes."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor=color,
                    linewidth=0.3 * edge_alpha, shade=True)


def _draw_box(ax, box_size: float):
    """Draw a thin dashed gray wireframe cube from origin to box_size."""
    L = box_size
    corners = np.array([
        [0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0],
        [0, 0, L], [L, 0, L], [L, L, L], [0, L, L],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # pillars
    ]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color="gray", linewidth=0.5,
                  linestyle="--", alpha=0.5)


def plot_structure(
    positions: np.ndarray,
    radius: float,
    box_size: float,
    ax=None,
    title: str | None = None,
) -> plt.Figure:
    """Plot a single 3D atom configuration.

    Args:
        positions: (N, 3) atom positions.
        radius: atom radius.
        box_size: cubic box side length.
        ax: optional existing 3D axes.
        title: optional subplot title.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    clashing = _find_clashing_atoms(positions, radius)
    _draw_box(ax, box_size)

    color_ok = "#6BAED6"
    color_clash = "#E07A5F"

    for i, pos in enumerate(positions):
        if clashing[i]:
            _draw_sphere(ax, pos, radius, color_clash, alpha=0.5, edge_alpha=1.0)
        else:
            _draw_sphere(ax, pos, radius, color_ok, alpha=0.4, edge_alpha=0.8)

    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)

    # Light gray panes
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor((0.95, 0.95, 0.95, 0.3))
        pane.set_edgecolor("gray")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title, fontsize=12)

    return fig


def plot_structures_grid(
    positions_list: list[np.ndarray],
    radius: float,
    box_size: float,
    ncols: int = 4,
    labels: list[str] | None = None,
) -> plt.Figure:
    """Grid of 3D structure plots.

    Args:
        positions_list: list of (N, 3) arrays.
        radius: atom radius.
        box_size: cubic box side length.
        ncols: columns in grid.
        labels: optional per-subplot labels (default: "Sample 1", ...).

    Returns:
        The matplotlib Figure.
    """
    n = len(positions_list)
    if labels is None:
        labels = [f"Sample {i + 1}" for i in range(n)]
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, (pos, label) in enumerate(zip(positions_list, labels)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        plot_structure(pos, radius, box_size, ax=ax, title=label)
    return fig
