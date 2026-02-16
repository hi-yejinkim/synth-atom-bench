"""g(r) L1 distance metric: continuous measure of distributional quality."""

import numpy as np

from data.validate import pair_correlation


def gr_distance(
    generated: np.ndarray,
    gt_r: np.ndarray,
    gt_g_r: np.ndarray,
    box_size: float,
    num_bins: int = 200,
) -> float:
    """L1 distance between generated and ground-truth pair correlation functions.

    Args:
        generated: (num_samples, N, 3) generated atom positions.
        gt_r: (num_bins,) bin centers from ground-truth g(r).
        gt_g_r: (num_bins,) ground-truth g(r) values.
        box_size: cubic box side length.
        num_bins: number of bins for g(r) computation (must match gt arrays).

    Returns:
        Mean absolute difference |g_gen(r) - g_gt(r)|. Lower is better (0 = perfect).
    """
    gen_r, gen_g_r = pair_correlation(generated, box_size, num_bins=num_bins)
    return float(np.mean(np.abs(gen_g_r - gt_g_r)))
