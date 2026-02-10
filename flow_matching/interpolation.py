"""Linear interpolation for conditional flow matching."""

import torch
from torch import Tensor


def interpolate(x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compute interpolated positions, noise, and velocity target.

    Args:
        x_0: Clean data positions, shape (batch, N, 3).
        t: Timesteps in [0, 1], shape (batch,).

    Returns:
        x_t: Interpolated positions (batch, N, 3).
        noise: Sampled noise (batch, N, 3).
        velocity_target: Target velocity x_0 - noise (batch, N, 3).
    """
    noise = torch.randn_like(x_0)
    t = t[:, None, None]  # (batch, 1, 1) for broadcasting
    x_t = (1 - t) * noise + t * x_0
    velocity_target = x_0 - noise
    return x_t, noise, velocity_target
