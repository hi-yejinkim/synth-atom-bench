"""Flow matching training loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow_matching.interpolation import interpolate


def flow_matching_loss(
    model: nn.Module,
    x_0: Tensor,
    atom_type_ids: Tensor | None = None,
) -> Tensor:
    """Compute flow matching MSE loss.

    Args:
        model: Velocity network, callable as model(x_t, t) -> (batch, N, 3).
        x_0: Clean data positions, shape (batch, N, 3).
        atom_type_ids: Per-atom type ids (N,) int64, optional conditioning.

    Returns:
        Scalar MSE loss.
    """
    batch_size = x_0.shape[0]
    t = torch.rand(batch_size, device=x_0.device)
    x_t, noise, velocity_target = interpolate(x_0, t)
    v_pred = model(x_t, t, atom_type_ids=atom_type_ids)
    return F.mse_loss(v_pred, velocity_target)
