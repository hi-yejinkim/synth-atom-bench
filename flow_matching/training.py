"""Flow matching training loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow_matching.interpolation import interpolate


def _pairwise_distances(x: Tensor) -> Tensor:
    """Compute pairwise distance matrix. Returns (batch, N, N)."""
    return torch.cdist(x, x)


def _snr_weights(t: Tensor, eps: float = 1e-6) -> Tensor:
    """SNR weighting: (t/(1-t))² — upweights t→1 (data end).

    At t→1 the signal-to-noise ratio is high and small errors in velocity
    prediction matter most for final sample quality.
    """
    w = (t / (1 - t + eps)) ** 2
    # Normalize so mean weight = 1 (keeps loss scale comparable)
    return w / (w.mean() + eps)


def flow_matching_loss(
    model: nn.Module,
    x_0: Tensor,
    atom_type_ids: Tensor | None = None,
    snr_weight: bool = False,
    aux_dist_weight: float = 0.0,
) -> Tensor:
    """Compute flow matching MSE loss.

    Args:
        model: Velocity network, callable as model(x_t, t) -> (batch, N, 3).
        x_0: Clean data positions, shape (batch, N, 3).
        atom_type_ids: Per-atom type ids (N,) int64, optional conditioning.
        snr_weight: If True, apply SNR weighting (t/(1-t))² per sample.
        aux_dist_weight: Weight for auxiliary pairwise distance loss.
            0.0 disables it. Recommended: 0.3 for n-body tasks.

    Returns:
        Scalar loss.
    """
    batch_size = x_0.shape[0]
    t = torch.rand(batch_size, device=x_0.device)
    x_t, noise, velocity_target = interpolate(x_0, t)
    v_pred = model(x_t, t, atom_type_ids=atom_type_ids)

    # Per-sample MSE: (batch,)
    per_sample_mse = (v_pred - velocity_target).pow(2).mean(dim=(1, 2))

    if snr_weight:
        weights = _snr_weights(t)
        loss_fm = (weights * per_sample_mse).mean()
    else:
        loss_fm = per_sample_mse.mean()

    # Auxiliary pairwise distance loss
    if aux_dist_weight > 0.0:
        # Predicted x_1 from the ODE: x_1 = x_t + (1-t) * v_pred
        t_expand = t[:, None, None]
        x1_pred = x_t + (1 - t_expand) * v_pred
        dist_pred = _pairwise_distances(x1_pred)
        dist_true = _pairwise_distances(x_0)
        loss_aux = F.mse_loss(dist_pred, dist_true)
        return loss_fm + aux_dist_weight * loss_aux

    return loss_fm
