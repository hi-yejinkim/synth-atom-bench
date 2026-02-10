"""Training curve visualization."""

import numpy as np
import matplotlib.pyplot as plt


def _ema_smooth(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average smoothing."""
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def plot_training_curves(
    steps: np.ndarray,
    loss: np.ndarray,
    clash_rate_steps: np.ndarray | None = None,
    clash_rate: np.ndarray | None = None,
    color: str = "#4C72B0",
    ema_alpha: float = 0.1,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot training loss (with EMA) and optional clash rate on twin y-axis.

    Args:
        steps: training step indices for loss.
        loss: per-step loss values.
        clash_rate_steps: step indices for clash rate evaluations.
        clash_rate: clash rate values.
        color: color for loss curves.
        ema_alpha: EMA smoothing factor.
        ax: optional existing axes (left y-axis).

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    # Raw loss (faint)
    ax.plot(steps, loss, color=color, alpha=0.2, linewidth=0.8)
    # EMA-smoothed loss (bold)
    smoothed = _ema_smooth(loss, alpha=ema_alpha)
    ax.plot(steps, smoothed, color=color, linewidth=2.0, label="Loss (EMA)")

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss", color=color)
    ax.tick_params(axis="y", labelcolor=color)

    # Clash rate on right y-axis
    if clash_rate_steps is not None and clash_rate is not None:
        ax2 = ax.twinx()
        ax2.plot(clash_rate_steps, clash_rate, color="gray", marker="o",
                 markersize=4, linewidth=1.0, label="Clash rate")
        ax2.set_ylabel("Clash rate", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    else:
        ax.legend(frameon=False)

    ax.set_title("Training Progress")

    return fig
