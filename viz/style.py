"""Global plot style foundation for SynthBench3D."""

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Architecture visual identity
ARCH_COLORS = {
    "PaiNN": "#4C72B0",
    "EquiformerV2": "#55A868",
    "Transformer": "#C44E52",
    "Pairformer": "#8172B3",
}
ARCH_MARKERS = {
    "PaiNN": "o",
    "EquiformerV2": "s",
    "Transformer": "^",
    "Pairformer": "D",
}

# Standard figure sizes (inches) — single and double column
SINGLE_COL = (3.5, 3.5 * 0.75)
DOUBLE_COL = (7.0, 7.0 * 0.75)


def _resolve_font(candidates: list[str] = ("Inter", "Helvetica Neue", "DejaVu Sans")) -> str:
    """Return the first available font family from *candidates*."""
    for name in candidates:
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if path and "LastResort" not in path:
                return name
        except ValueError:
            continue
    return "DejaVu Sans"


@contextmanager
def synthbench_style():
    """Context manager that applies publication-quality plot style."""
    font = _resolve_font()
    rc = {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": [font],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Lines
        "lines.linewidth": 2.0,
        "grid.linewidth": 1.0,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "gray",
        # Layout
        "figure.constrained_layout.use": True,
        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
    with mpl.rc_context(rc):
        yield


def save_figure(fig: plt.Figure, path: str | Path, close: bool = True) -> None:
    """Save *fig* as both PDF and PNG (300 dpi). Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    if close:
        plt.close(fig)
