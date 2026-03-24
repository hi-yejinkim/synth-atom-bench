"""Wasserstein distances on energy distributions for n-body MCMC evaluation.

Compares the 1D energy distribution of generated samples against a reference
dataset using Wasserstein-1 (W1) and Wasserstein-2 (W2) distances.

For 1D distributions, both have closed forms via quantile functions:
    W1  = ∫₀¹ |F⁻¹_gen(q) - F⁻¹_ref(q)| dq     (mean abs diff)
    W2² = ∫₀¹ (F⁻¹_gen(q) - F⁻¹_ref(q))² dq     (RMS diff)

W1 is more robust to outliers (linear penalty), W2 is more sensitive to
tail differences (quadratic penalty). Both are computed from sorted arrays.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _align_quantiles(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort and interpolate two 1D arrays to a common quantile grid."""
    a = np.sort(a.ravel().astype(np.float64))
    b = np.sort(b.ravel().astype(np.float64))
    if len(a) != len(b):
        n = max(len(a), len(b))
        q = np.linspace(0, 1, n)
        a = np.interp(q, np.linspace(0, 1, len(a)), a)
        b = np.interp(q, np.linspace(0, 1, len(b)), b)
    return a, b


def _w1_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 distance between two 1D empirical distributions.

    Uses quantile coupling: sort both, interpolate to common size, L1.
    More robust to outliers than W2 (linear penalty).
    """
    a, b = _align_quantiles(a, b)
    return float(np.mean(np.abs(a - b)))


def _w2_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-2 distance between two 1D empirical distributions.

    Uses quantile coupling: sort both, interpolate to common size, L2.
    More sensitive to tail differences than W1 (quadratic penalty).
    """
    a, b = _align_quantiles(a, b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def energy_w2(
    generated_energies: np.ndarray | torch.Tensor,
    reference_energies: np.ndarray | torch.Tensor,
) -> float:
    """Wasserstein-2 distance between two 1D energy distributions.

    Args:
        generated_energies: (M,) energies from generated/model samples.
        reference_energies: (N,) energies from reference dataset.

    Returns:
        W2 distance (lower is better, 0 = identical distributions).
    """
    return _w2_1d(_to_numpy(generated_energies), _to_numpy(reference_energies))


def energy_w2_decomposed(
    gen_energies_2body: np.ndarray | torch.Tensor,
    gen_energies_3body: np.ndarray | torch.Tensor,
    gen_energies_4body: np.ndarray | torch.Tensor,
    ref_energies_2body: np.ndarray | torch.Tensor,
    ref_energies_3body: np.ndarray | torch.Tensor,
    ref_energies_4body: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Per-body-order Wasserstein-2 distances.

    Returns:
        Dict with keys: w2_2body, w2_3body, w2_4body (0.0 if arrays are all-zero).
    """
    result = {}
    for label, gen, ref in [
        ("w2_2body", gen_energies_2body, ref_energies_2body),
        ("w2_3body", gen_energies_3body, ref_energies_3body),
        ("w2_4body", gen_energies_4body, ref_energies_4body),
    ]:
        gen = _to_numpy(gen).ravel().astype(np.float64)
        ref = _to_numpy(ref).ravel().astype(np.float64)
        if np.all(gen == 0) and np.all(ref == 0):
            result[label] = 0.0
        else:
            result[label] = _w2_1d(gen, ref)
    return result


def energy_w2_from_positions(
    generated_positions: np.ndarray,
    reference_energies: np.ndarray,
    body: int,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    nu: float = 1.0,
    mu: float = 0.2,
    box_size: float = 3.5,
    bc: str = "pbc",
) -> dict[str, float]:
    """Compute energies from generated positions, then W2 against reference.

    Main entry point for model evaluation: model produces positions,
    we compute their energies under the n-body potential, then W2.

    Args:
        generated_positions: (M, N, 3) from the generative model.
        reference_energies: (K,) total energies from MCMC reference.
        body, sigma, epsilon, nu, mu, box_size, bc: potential parameters.

    Returns:
        Dict with 'w2_total'.
    """
    from data.generate_nbody import PotentialParams, total_energy

    params = PotentialParams(
        body=body, sigma=sigma, epsilon=epsilon,
        nu=nu, mu=mu, box_size=box_size, bc=bc,
    )

    gen_total = np.empty(len(generated_positions))
    for i in range(len(generated_positions)):
        et, _, _, _ = total_energy(generated_positions[i], params)
        gen_total[i] = et

    ref = reference_energies.ravel()
    return {
        "w1_total": _w1_1d(gen_total, ref),
        "w2_total": _w2_1d(gen_total, ref),
    }


def energy_w2_batched(
    generated_positions: np.ndarray,
    reference_energies: np.ndarray,
    body: int,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    nu: float = 1.0,
    mu: float = 0.2,
    box_size: float = 3.5,
    bc: str = "pbc",
) -> float:
    """W2 from positions vs reference energies. Returns w2_total only."""
    return energy_w2_from_positions(
        generated_positions, reference_energies,
        body=body, sigma=sigma, epsilon=epsilon,
        nu=nu, mu=mu, box_size=box_size, bc=bc,
    )["w2_total"]


def energy_w1_batched(
    generated_positions: np.ndarray,
    reference_energies: np.ndarray,
    body: int,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    nu: float = 1.0,
    mu: float = 0.2,
    box_size: float = 3.5,
    bc: str = "pbc",
) -> float:
    """W1 from positions vs reference energies. Returns w1_total only."""
    return energy_w2_from_positions(
        generated_positions, reference_energies,
        body=body, sigma=sigma, epsilon=epsilon,
        nu=nu, mu=mu, box_size=box_size, bc=bc,
    )["w1_total"]
