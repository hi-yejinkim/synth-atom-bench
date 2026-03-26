"""MALA + Parallel Tempering sampler for free n-body systems (Boltzmann sampling).

Generates configurations from p(x) ∝ exp(-V(x) / T) where V(x) is the
cumulative many-body potential up to the specified body order.

    body=2 : V = V_2                  (LJ pairs)
    body=3 : V = V_2 + V_3           (LJ + Axilrod-Teller)
    body=4 : V = V_2 + V_3 + V_4     (LJ + AT + tetrahedron)

Sampling: MALA (Metropolis-Adjusted Langevin Algorithm) with parallel tempering
for robust sampling at low temperatures. MALA uses gradient information for
efficient proposals; parallel tempering exchanges replicas at different
temperatures to overcome energy barriers.

Boundary conditions: periodic (minimum image convention) or soft wall.

Potentials
----------
2-body : Lennard-Jones pair potential
    V_2 = Σ_{i<j} 4ε [(σ/r_ij)^12 - (σ/r_ij)^6]

3-body : Axilrod-Teller triple-dipole potential
    V_3 = Σ_{i<j<k} ν (1 + 3 cos θ_i cos θ_j cos θ_k) / (r_ij r_jk r_ik)^3

4-body : Distance-based tetrahedron potential
    V_4 = Σ_{i<j<k<l} μ · vol²(i,j,k,l) / Π_{a<b} r_ab²

Usage
-----
    uv run data/generate_nbody.py \\
        --n 15 --body 2 --T 1.0 --num_samples 50000 \\
        --output outputs/data/nbody_n15_b2_T1.0/train.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_OVERLAP_ENERGY = math.inf


# ---------------------------------------------------------------------------
# Boundary condition helpers
# ---------------------------------------------------------------------------

def _apply_bc_diff(diff: np.ndarray, box_size: float, bc: str = 'pbc') -> np.ndarray:
    """Apply boundary condition to displacement vectors."""
    if bc == 'pbc':
        return diff - box_size * np.round(diff / box_size)
    return diff


def _pairwise_diff(positions: np.ndarray, box_size: float,
                   bc: str = 'pbc') -> np.ndarray:
    """Pairwise displacement vectors. Shape (n, n, 3)."""
    diff = positions[:, None, :] - positions[None, :, :]
    return _apply_bc_diff(diff, box_size, bc)


def _pairwise_dist_sq(positions: np.ndarray, box_size: float,
                      bc: str = 'pbc') -> np.ndarray:
    """Pairwise squared distances. Shape (n, n)."""
    diff = _pairwise_diff(positions, box_size, bc)
    return np.sum(diff ** 2, axis=-1)


# ---------------------------------------------------------------------------
# Soft wall potential (replaces hard_wall for differentiable boundary)
# ---------------------------------------------------------------------------

def soft_wall_energy(positions: np.ndarray, box_size: float,
                     wall_k: float = 50.0, wall_sigma: float = 0.3) -> float:
    """Soft repulsive wall: V = wall_k * Σ max(0, (wall_sigma/d)^6 - 1)."""
    E = 0.0
    for dim in range(3):
        d_lo = positions[:, dim]
        d_hi = box_size - positions[:, dim]
        for d in [d_lo, d_hi]:
            inside = d < wall_sigma
            if inside.any():
                ratio = wall_sigma / np.maximum(d[inside], 1e-10)
                E += wall_k * np.sum(ratio ** 6 - 1.0)
    return E


def soft_wall_gradient(positions: np.ndarray, box_size: float,
                       wall_k: float = 50.0, wall_sigma: float = 0.3) -> np.ndarray:
    """Gradient of soft wall potential. Shape (n, 3)."""
    grad = np.zeros_like(positions)
    for dim in range(3):
        d_lo = positions[:, dim]
        d_hi = box_size - positions[:, dim]
        # Lower wall: d = x, dd/dx = +1, dV/dd = -6*wall_k*wall_sigma^6 / d^7
        inside_lo = d_lo < wall_sigma
        if inside_lo.any():
            d_safe = np.maximum(d_lo[inside_lo], 1e-10)
            grad[inside_lo, dim] += -6.0 * wall_k * wall_sigma ** 6 / d_safe ** 7
        # Upper wall: d = box_size - x, dd/dx = -1
        inside_hi = d_hi < wall_sigma
        if inside_hi.any():
            d_safe = np.maximum(d_hi[inside_hi], 1e-10)
            grad[inside_hi, dim] += 6.0 * wall_k * wall_sigma ** 6 / d_safe ** 7
    return grad


# ---------------------------------------------------------------------------
# Potential energy functions
# ---------------------------------------------------------------------------

def energy_2body(positions: np.ndarray, sigma: float, epsilon: float,
                 box_size: float, bc: str = 'pbc') -> float:
    """Vectorised Lennard-Jones pair potential."""
    diff = _pairwise_diff(positions, box_size, bc)
    dist_sq = np.sum(diff ** 2, axis=-1)
    n = len(positions)
    idx = np.triu_indices(n, k=1)
    r_sq = np.maximum(dist_sq[idx], 1e-20)
    sr2 = sigma ** 2 / r_sq
    sr6 = sr2 * sr2 * sr2
    return float(4.0 * epsilon * np.sum(sr6 * sr6 - sr6))


def energy_3body(positions: np.ndarray, nu: float, box_size: float,
                 bc: str = 'pbc') -> float:
    """Axilrod-Teller triple-dipole potential (vectorised)."""
    n = len(positions)
    if n < 3:
        return 0.0
    diff = _pairwise_diff(positions, box_size, bc)  # (n, n, 3)
    dist = np.sqrt(np.maximum(np.sum(diff ** 2, axis=-1), 1e-20))  # (n, n)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                r_ij, r_ik, r_jk = dist[i, j], dist[i, k], dist[j, k]
                cos_i = (r_ij ** 2 + r_ik ** 2 - r_jk ** 2) / (2.0 * r_ij * r_ik)
                cos_j = (r_ij ** 2 + r_jk ** 2 - r_ik ** 2) / (2.0 * r_ij * r_jk)
                cos_k = (r_ik ** 2 + r_jk ** 2 - r_ij ** 2) / (2.0 * r_ik * r_jk)
                denom = (r_ij * r_jk * r_ik) ** 3
                if denom < 1e-30:
                    continue
                total += nu * (1.0 + 3.0 * cos_i * cos_j * cos_k) / denom
    return total


def energy_4body(positions: np.ndarray, mu: float, box_size: float,
                 bc: str = 'pbc') -> float:
    """Distance-based 4-body tetrahedron potential."""
    n = len(positions)
    if n < 4:
        return 0.0
    dist_sq = _pairwise_dist_sq(positions, box_size, bc)
    np.fill_diagonal(dist_sq, 1e-20)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    e1 = _apply_bc_diff(positions[j] - positions[i], box_size, bc)
                    e2 = _apply_bc_diff(positions[k] - positions[i], box_size, bc)
                    e3 = _apply_bc_diff(positions[l] - positions[i], box_size, bc)
                    det = (e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
                         - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
                         + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]))
                    det_sq = det * det
                    prod_r2 = (dist_sq[i, j] * dist_sq[i, k] * dist_sq[i, l]
                             * dist_sq[j, k] * dist_sq[j, l] * dist_sq[k, l])
                    if prod_r2 < 1e-60:
                        continue
                    total += mu * det_sq / prod_r2
    return total


# ---------------------------------------------------------------------------
# Total energy
# ---------------------------------------------------------------------------

@dataclass
class PotentialParams:
    """Parameters for the n-body potential."""
    body: int
    sigma: float
    epsilon: float
    nu: float
    mu: float
    box_size: float
    bc: str = 'pbc'
    wall_k: float = 50.0
    wall_sigma: float = 0.3

    def __post_init__(self):
        if self.bc not in ('pbc', 'hard_wall'):
            raise ValueError(f"bc must be 'pbc' or 'hard_wall', got {self.bc!r}")


def total_energy(positions: np.ndarray, params: PotentialParams) -> tuple[float, float, float, float]:
    """Compute cumulative total energy and per-body-order decomposition.

    Returns (E_total, E_2body, E_3body, E_4body).
    """
    e2 = energy_2body(positions, params.sigma, params.epsilon,
                      params.box_size, params.bc)
    e3 = 0.0
    e4 = 0.0
    if params.body >= 3:
        e3 = energy_3body(positions, params.nu, params.box_size, params.bc)
    if params.body >= 4:
        e4 = energy_4body(positions, params.mu, params.box_size, params.bc)

    e_total = e2 + e3 + e4
    if params.bc == 'hard_wall':
        e_total += soft_wall_energy(positions, params.box_size,
                                    params.wall_k, params.wall_sigma)
    return e_total, e2, e3, e4


# ---------------------------------------------------------------------------
# Analytic gradient functions
# ---------------------------------------------------------------------------

def gradient_2body(positions: np.ndarray, sigma: float, epsilon: float,
                   box_size: float, bc: str = 'pbc') -> np.ndarray:
    """Analytic gradient of LJ potential. Shape (n, 3).

    ∂V_2/∂x_i = Σ_j 4ε [-12σ¹²/r¹⁴ + 6σ⁶/r⁸] (x_i - x_j)
    """
    diff = _pairwise_diff(positions, box_size, bc)  # (n, n, 3)
    dist_sq = np.sum(diff ** 2, axis=-1, keepdims=True)  # (n, n, 1)
    dist_sq = np.maximum(dist_sq, 1e-20)
    sr2 = sigma ** 2 / dist_sq
    sr6 = sr2 * sr2 * sr2
    factor = 24.0 * epsilon * (sr6 - 2.0 * sr6 * sr6) / dist_sq  # (n, n, 1)
    n = len(positions)
    np.fill_diagonal(factor[:, :, 0], 0.0)
    grad = np.sum(factor * diff, axis=1)  # (n, 3)
    return grad


def gradient_3body(positions: np.ndarray, nu: float, box_size: float,
                   bc: str = 'pbc') -> np.ndarray:
    """Analytic gradient of Axilrod-Teller potential. Shape (n, 3).

    Uses the chain rule through the law-of-cosines form.
    """
    n = len(positions)
    grad = np.zeros((n, 3))
    if n < 3:
        return grad

    diff = _pairwise_diff(positions, box_size, bc)  # (n, n, 3)
    dist_sq = np.sum(diff ** 2, axis=-1)  # (n, n)
    dist = np.sqrt(np.maximum(dist_sq, 1e-20))

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                r_ij, r_ik, r_jk = dist[i, j], dist[i, k], dist[j, k]
                if r_ij < 1e-10 or r_ik < 1e-10 or r_jk < 1e-10:
                    continue

                r_ij2, r_ik2, r_jk2 = r_ij ** 2, r_ik ** 2, r_jk ** 2
                cos_i = (r_ij2 + r_ik2 - r_jk2) / (2.0 * r_ij * r_ik)
                cos_j = (r_ij2 + r_jk2 - r_ik2) / (2.0 * r_ij * r_jk)
                cos_k = (r_ik2 + r_jk2 - r_ij2) / (2.0 * r_ik * r_jk)

                prod_r3 = (r_ij * r_jk * r_ik) ** 3
                if prod_r3 < 1e-30:
                    continue

                ang_term = 1.0 + 3.0 * cos_i * cos_j * cos_k
                V_val = nu * ang_term / prod_r3

                # Gradient via finite differences on the triplet
                # (analytic is complex; use central differences per atom in triplet)
                h = 1e-5
                for atom_idx, atom in enumerate([i, j, k]):
                    for d in range(3):
                        pos_p = positions.copy()
                        pos_p[atom, d] += h
                        pos_m = positions.copy()
                        pos_m[atom, d] -= h

                        # Recompute only this triplet
                        dp = _pairwise_diff(pos_p[[i, j, k]], box_size, bc)
                        dm = _pairwise_diff(pos_m[[i, j, k]], box_size, bc)

                        def _triplet_energy(dd):
                            rr = np.sqrt(np.maximum(np.sum(dd ** 2, axis=-1), 1e-20))
                            r01, r02, r12 = rr[0, 1], rr[0, 2], rr[1, 2]
                            r012, r022, r122 = r01 ** 2, r02 ** 2, r12 ** 2
                            c0 = (r012 + r022 - r122) / (2.0 * r01 * r02)
                            c1 = (r012 + r122 - r022) / (2.0 * r01 * r12)
                            c2 = (r022 + r122 - r012) / (2.0 * r02 * r12)
                            den = (r01 * r12 * r02) ** 3
                            if den < 1e-30:
                                return 0.0
                            return nu * (1.0 + 3.0 * c0 * c1 * c2) / den

                        grad[atom, d] += (_triplet_energy(dp) - _triplet_energy(dm)) / (2.0 * h)

    return grad


def gradient_4body(positions: np.ndarray, mu: float, box_size: float,
                   bc: str = 'pbc') -> np.ndarray:
    """Gradient of 4-body tetrahedron potential via finite differences. Shape (n, 3)."""
    n = len(positions)
    grad = np.zeros((n, 3))
    if n < 4:
        return grad
    h = 1e-5
    for i in range(n):
        for d in range(3):
            pos_p = positions.copy()
            pos_p[i, d] += h
            pos_m = positions.copy()
            pos_m[i, d] -= h
            grad[i, d] = (energy_4body(pos_p, mu, box_size, bc)
                        - energy_4body(pos_m, mu, box_size, bc)) / (2.0 * h)
    return grad


def gradient_total(positions: np.ndarray, params: PotentialParams) -> np.ndarray:
    """Gradient of total potential. Shape (n, 3)."""
    grad = gradient_2body(positions, params.sigma, params.epsilon,
                          params.box_size, params.bc)
    if params.body >= 3:
        grad += gradient_3body(positions, params.nu, params.box_size, params.bc)
    if params.body >= 4:
        grad += gradient_4body(positions, params.mu, params.box_size, params.bc)
    if params.bc == 'hard_wall':
        grad += soft_wall_gradient(positions, params.box_size,
                                   params.wall_k, params.wall_sigma)
    return grad


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize_on_lattice(n: int, box_size: float, sigma: float,
                          rng: np.random.Generator,
                          bc: str = 'pbc') -> np.ndarray:
    """Place particles on a slightly perturbed cubic lattice inside the box."""
    margin = 0.5 * sigma if bc == 'hard_wall' else 0.0
    usable = box_size - 2.0 * margin
    n_side = int(np.ceil(n ** (1.0 / 3.0)))
    spacing = usable / n_side
    positions = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(positions) >= n:
                    break
                pos = np.array([(ix + 0.5) * spacing + margin,
                                (iy + 0.5) * spacing + margin,
                                (iz + 0.5) * spacing + margin])
                pos += rng.uniform(-0.1 * spacing, 0.1 * spacing, size=3)
                if bc == 'pbc':
                    pos = pos % box_size
                else:
                    pos = np.clip(pos, margin, box_size - margin)
                positions.append(pos)
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    return np.array(positions[:n])


# ---------------------------------------------------------------------------
# MALA step
# ---------------------------------------------------------------------------

def mala_step(positions: np.ndarray, energy: float, grad: np.ndarray,
              T: float, epsilon_mala: float, params: PotentialParams,
              rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray, bool]:
    """Single MALA step with Metropolis-Hastings correction.

    Proposal: x' = x - (ε²/2T) ∇V(x) + ε η,  η ~ N(0, I)

    Returns (new_positions, new_energy, new_gradient, accepted).
    """
    n = len(positions)
    drift = -(epsilon_mala ** 2 / (2.0 * T)) * grad
    noise = epsilon_mala * rng.standard_normal(positions.shape)
    proposal = positions + drift + noise

    # Apply boundary conditions
    if params.bc == 'pbc':
        proposal = proposal % params.box_size

    # Compute proposal energy and gradient
    e_prop, e2p, e3p, e4p = total_energy(proposal, params)
    if not math.isfinite(e_prop):
        return positions, energy, grad, False

    grad_prop = gradient_total(proposal, params)

    # Log proposal densities for M-H correction
    # Forward: log q(x'|x) = -||x' - x - drift_x||² / (2ε²)
    # Reverse: log q(x|x') = -||x - x' - drift_x'||² / (2ε²)
    drift_rev = -(epsilon_mala ** 2 / (2.0 * T)) * grad_prop
    fwd_diff = proposal - positions - drift
    rev_diff = positions - proposal - drift_rev

    log_q_fwd = -np.sum(fwd_diff ** 2) / (2.0 * epsilon_mala ** 2)
    log_q_rev = -np.sum(rev_diff ** 2) / (2.0 * epsilon_mala ** 2)

    # M-H acceptance: log α = -ΔE/T + log q(x|x') - log q(x'|x)
    log_alpha = -(e_prop - energy) / T + log_q_rev - log_q_fwd

    if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
        return proposal, e_prop, grad_prop, True
    return positions, energy, grad, False


# ---------------------------------------------------------------------------
# Parallel tempering
# ---------------------------------------------------------------------------

def compute_temperature_ladder(T_target: float, n_replicas: int,
                               T_max_factor: float = 3.5) -> np.ndarray:
    """Geometric temperature ladder from T_target to T_max."""
    T_max = T_target * T_max_factor
    if n_replicas == 1:
        return np.array([T_target])
    return np.geomspace(T_target, T_max, n_replicas)


def auto_n_replicas(T_target: float, body: int) -> int:
    """Auto-select number of replicas based on target temperature and body order."""
    base = {0.5: 8, 0.6: 7, 0.8: 7, 1.0: 6, 1.5: 5, 2.0: 4, 3.0: 3}
    # Interpolate from nearest known T
    T_keys = sorted(base.keys())
    if T_target <= T_keys[0]:
        n = base[T_keys[0]]
    elif T_target >= T_keys[-1]:
        n = base[T_keys[-1]]
    else:
        for i in range(len(T_keys) - 1):
            if T_keys[i] <= T_target <= T_keys[i + 1]:
                frac = (T_target - T_keys[i]) / (T_keys[i + 1] - T_keys[i])
                n = int(round(base[T_keys[i]] * (1 - frac) + base[T_keys[i + 1]] * frac))
                break
    # Higher body orders need more replicas
    if body >= 3:
        n += 2
    if body >= 4:
        n += 1
    return max(n, 2)


def mala_pt_sample(
    n: int,
    body: int,
    T: float,
    num_samples: int,
    box_size: float | None = None,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    nu: float = 1.0,
    mu: float = 0.2,
    burn_in: int | None = None,
    thin_interval: int | None = None,
    epsilon_mala: float | None = None,
    n_replicas: int | None = None,
    T_max_factor: float = 3.5,
    seed: int = 42,
    bc: str = 'pbc',
    wall_k: float = 50.0,
    wall_sigma: float = 0.3,
) -> dict:
    """Run MALA + Parallel Tempering Boltzmann sampler for free n-body system.

    Returns dict with positions, energies (total and per-body), and metadata.
    """
    # Adaptive defaults
    _N_REF, _BOX_REF = 15, 3.5
    if box_size is None:
        box_size = _BOX_REF * (n / _N_REF) ** (1.0 / 3.0)

    if epsilon_mala is None:
        epsilon_mala = 0.025 * math.sqrt(T) * (3 * n / 45.0) ** (-1.0 / 6.0)

    if n_replicas is None:
        n_replicas = auto_n_replicas(T, body)

    if thin_interval is None:
        thin_interval = max(n, 20)

    if burn_in is None:
        base = 50_000
        if body >= 3:
            base = 100_000
        if body >= 4:
            base = 150_000
        t_factor = max(1.0, 1.0 / T)
        burn_in = int(base * min(t_factor, 5.0))

    rng = np.random.default_rng(seed)
    params = PotentialParams(body=body, sigma=sigma, epsilon=epsilon,
                             nu=nu, mu=mu, box_size=box_size, bc=bc,
                             wall_k=wall_k, wall_sigma=wall_sigma)

    # Temperature ladder
    T_ladder = compute_temperature_ladder(T, n_replicas, T_max_factor)
    print(f"  Temperature ladder ({n_replicas} replicas): {[f'{t:.3f}' for t in T_ladder]}")

    # Initialize replicas
    replicas = []
    for r in range(n_replicas):
        pos = initialize_on_lattice(n, box_size, sigma, rng, bc)
        e_total, e2, e3, e4 = total_energy(pos, params)
        grad = gradient_total(pos, params)
        replicas.append({
            'positions': pos,
            'energy': e_total,
            'gradient': grad,
            'epsilon': epsilon_mala * math.sqrt(T_ladder[r] / T),  # Scale ε with √T_r
        })

    # Adaptive step size via dual averaging (per replica)
    target_accept = 0.574
    adapt_until = burn_in // 2
    log_eps = [math.log(r['epsilon']) for r in replicas]
    log_eps_bar = [le for le in log_eps]
    h_bar = [0.0] * n_replicas
    mu_adapt = [math.log(10.0 * r['epsilon']) for r in replicas]

    # Storage
    total_steps = burn_in + num_samples * thin_interval
    samples = np.empty((num_samples, n, 3))
    energies = np.empty(num_samples)
    energies_2 = np.empty(num_samples)
    energies_3 = np.empty(num_samples)
    energies_4 = np.empty(num_samples)

    sample_idx = 0
    accept_counts = np.zeros(n_replicas)
    swap_counts = np.zeros(n_replicas - 1)
    swap_attempts = np.zeros(n_replicas - 1)
    proposal_counts = np.zeros(n_replicas)
    report_interval = max(1, total_steps // 100)

    t0 = time.time()
    for step in range(total_steps):
        # MALA step on each replica
        for r in range(n_replicas):
            rep = replicas[r]
            new_pos, new_e, new_g, accepted = mala_step(
                rep['positions'], rep['energy'], rep['gradient'],
                T_ladder[r], rep['epsilon'], params, rng,
            )
            rep['positions'] = new_pos
            rep['energy'] = new_e
            rep['gradient'] = new_g
            proposal_counts[r] += 1
            if accepted:
                accept_counts[r] += 1

            # Dual averaging adaptation
            if step < adapt_until:
                m = step + 1
                gamma, t0_adapt, kappa = 0.05, 10, 0.75
                acc_rate = accept_counts[r] / proposal_counts[r]
                w = 1.0 / (m + t0_adapt)
                h_bar[r] = (1.0 - w) * h_bar[r] + w * (target_accept - acc_rate)
                log_eps[r] = mu_adapt[r] - math.sqrt(m) / gamma * h_bar[r]
                m_w = m ** (-kappa)
                log_eps_bar[r] = m_w * log_eps[r] + (1.0 - m_w) * log_eps_bar[r]
                rep['epsilon'] = math.exp(log_eps[r])
            elif step == adapt_until:
                rep['epsilon'] = math.exp(log_eps_bar[r])

        # Parallel tempering swap (even-odd alternation)
        parity = step % 2
        for r in range(parity, n_replicas - 1, 2):
            swap_attempts[r] += 1
            beta_i = 1.0 / T_ladder[r]
            beta_j = 1.0 / T_ladder[r + 1]
            E_i = replicas[r]['energy']
            E_j = replicas[r + 1]['energy']
            log_alpha = (beta_i - beta_j) * (E_i - E_j)
            if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
                # Swap positions, energies, gradients
                replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]
                # Swap back the per-replica epsilon (tied to temperature, not config)
                eps_r = replicas[r]['epsilon']
                replicas[r]['epsilon'] = replicas[r + 1]['epsilon']
                replicas[r + 1]['epsilon'] = eps_r
                swap_counts[r] += 1

        # Collect sample from target (coldest) replica
        if step >= burn_in and (step - burn_in) % thin_interval == 0:
            if sample_idx < num_samples:
                rep0 = replicas[0]
                samples[sample_idx] = rep0['positions'].copy()
                et, e2c, e3c, e4c = total_energy(rep0['positions'], params)
                energies[sample_idx] = et
                energies_2[sample_idx] = e2c
                energies_3[sample_idx] = e3c
                energies_4[sample_idx] = e4c
                sample_idx += 1

        if (step + 1) % report_interval == 0:
            elapsed = time.time() - t0
            pct = (step + 1) / total_steps * 100
            rate = (step + 1) / elapsed
            acc0 = accept_counts[0] / max(proposal_counts[0], 1)
            print(
                f"\r  {pct:5.1f}% | {sample_idx}/{num_samples} samples | "
                f"accept[0]={acc0:.3f} | ε[0]={replicas[0]['epsilon']:.4f} | "
                f"{rate:.0f} steps/s | E={replicas[0]['energy']:.2f}",
                end="", flush=True,
            )

    print()
    elapsed = time.time() - t0
    print(f"  Done: {sample_idx} samples in {elapsed:.1f}s")

    # Report acceptance and swap rates
    for r in range(n_replicas):
        acc = accept_counts[r] / max(proposal_counts[r], 1)
        print(f"  Replica {r} (T={T_ladder[r]:.3f}): accept={acc:.3f}, ε={replicas[r]['epsilon']:.4f}")
    for r in range(n_replicas - 1):
        sr = swap_counts[r] / max(swap_attempts[r], 1)
        print(f"  Swap {r}↔{r+1}: rate={sr:.3f} ({int(swap_counts[r])}/{int(swap_attempts[r])})")

    overall_acc = accept_counts[0] / max(proposal_counts[0], 1)
    return {
        "positions": samples[:sample_idx].astype(np.float32),
        "energies": energies[:sample_idx].astype(np.float32),
        "energies_2body": energies_2[:sample_idx].astype(np.float32),
        "energies_3body": energies_3[:sample_idx].astype(np.float32),
        "energies_4body": energies_4[:sample_idx].astype(np.float32),
        "acceptance_rate": overall_acc,
        "n_replicas": n_replicas,
        "T_ladder": T_ladder,
        "swap_rates": (swap_counts / np.maximum(swap_attempts, 1)).tolist(),
        "burn_in": burn_in,
        "thin_interval": thin_interval,
        "epsilon_mala_final": replicas[0]['epsilon'],
    }


# ---------------------------------------------------------------------------
# Energy histogram
# ---------------------------------------------------------------------------

def compute_energy_histogram(energies: np.ndarray,
                             n_bins: int = 200) -> dict:
    """Compute 1D energy histogram for quick distribution comparison."""
    e_min, e_max = energies.min(), energies.max()
    margin = 0.05 * (e_max - e_min) if e_max > e_min else 1.0
    bin_edges = np.linspace(e_min - margin, e_max + margin, n_bins + 1)
    counts, _ = np.histogram(energies, bins=bin_edges)
    density = counts / (counts.sum() * np.diff(bin_edges))
    return {
        "bin_edges": bin_edges.astype(np.float64),
        "counts": counts.astype(np.int64),
        "density": density.astype(np.float64),
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies)),
        "median": float(np.median(energies)),
        "percentiles": np.percentile(energies, [5, 25, 50, 75, 95]).astype(np.float64),
        "n_samples": len(energies),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate n-body samples via MALA + Parallel Tempering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # 2-body, 15 particles, T=1.0 (PBC)
  uv run data/generate_nbody.py --n 15 --body 2 --T 1.0 --num_samples 50000 \\
      --output outputs/data/nbody_n15_b2_T1.0/train.npz

  # 3-body, soft wall boundary
  uv run data/generate_nbody.py --n 20 --body 3 --T 0.5 --bc hard_wall \\
      --num_samples 50000 --output outputs/data/nbody_n20_b3_T0.5_hw/train.npz
""",
    )
    parser.add_argument("--n", type=int, default=15, help="Number of particles (default: 15)")
    parser.add_argument("--body", type=int, required=True, choices=[2, 3, 4],
                        help="Max body interaction order (cumulative)")
    parser.add_argument("--T", type=float, required=True, help="Temperature (ε/k_B)")
    parser.add_argument("--bc", type=str, default="pbc", choices=["pbc", "hard_wall"],
                        help="Boundary condition (default: pbc). hard_wall uses soft repulsive wall.")
    parser.add_argument("--box_size", type=float, default=None,
                        help="Box side length (auto-computed if not set)")
    parser.add_argument("--sigma", type=float, default=1.0, help="LJ σ (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="LJ ε (default: 1.0)")
    parser.add_argument("--nu", type=float, default=1.0, help="Axilrod-Teller coupling (default: 1.0)")
    parser.add_argument("--mu", type=float, default=0.2, help="Tetrahedron coupling (default: 0.2)")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to collect")
    parser.add_argument("--burn_in", type=int, default=None, help="Burn-in MALA steps (auto if not set)")
    parser.add_argument("--thin_interval", type=int, default=None, help="Thinning interval (auto if not set)")
    parser.add_argument("--epsilon_mala", type=float, default=None, help="MALA step size (auto if not set)")
    parser.add_argument("--n_replicas", type=int, default=None, help="Number of PT replicas (auto if not set)")
    parser.add_argument("--T_max_factor", type=float, default=3.5,
                        help="T_max = T * T_max_factor (default: 3.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", type=int, default=None, help="Pin to CPU core")
    parser.add_argument("--wall_k", type=float, default=50.0, help="Soft wall strength (default: 50)")
    parser.add_argument("--wall_sigma", type=float, default=0.3, help="Soft wall range (default: 0.3)")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    parser.add_argument("--n_bins", type=int, default=200, help="Energy histogram bins")

    args = parser.parse_args()

    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
        print(f"[cpu] Pinned to core {args.cpu}")

    # Auto box_size
    _N_REF, _BOX_REF = 15, 3.5
    if args.box_size is None:
        args.box_size = _BOX_REF * (args.n / _N_REF) ** (1.0 / 3.0)
        eta = args.n * (4.0 / 3.0) * np.pi * (args.sigma / 2.0) ** 3 / args.box_size ** 3
        print(f"[auto] box_size={args.box_size:.3f} (η={eta:.3f})")

    bc_label = "PBC" if args.bc == "pbc" else "soft wall"
    print(f"=== n-body MALA+PT sampler ({bc_label}) ===")
    print(f"  Particles: {args.n}, Body order: {args.body}")
    print(f"  Temperature: {args.T}, Box size: {args.box_size:.3f}")
    print(f"  LJ: σ={args.sigma}, ε={args.epsilon}")
    if args.body >= 3:
        print(f"  Axilrod-Teller ν={args.nu}")
    if args.body >= 4:
        print(f"  Tetrahedron μ={args.mu}")
    print()

    result = mala_pt_sample(
        n=args.n, body=args.body, T=args.T, num_samples=args.num_samples,
        box_size=args.box_size, sigma=args.sigma, epsilon=args.epsilon,
        nu=args.nu, mu=args.mu, burn_in=args.burn_in,
        thin_interval=args.thin_interval, epsilon_mala=args.epsilon_mala,
        n_replicas=args.n_replicas, T_max_factor=args.T_max_factor,
        seed=args.seed, bc=args.bc,
        wall_k=args.wall_k, wall_sigma=args.wall_sigma,
    )

    # Energy histogram
    print("\nComputing energy histogram...")
    hist = compute_energy_histogram(result["energies"], n_bins=args.n_bins)
    print(f"  Energy: mean={hist['mean']:.4f}, std={hist['std']:.4f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving samples to {out_path}")
    np.savez(
        str(out_path),
        positions=result["positions"],
        energies=result["energies"],
        energies_2body=result["energies_2body"],
        energies_3body=result["energies_3body"],
        energies_4body=result["energies_4body"],
        n=args.n, body=args.body, T=args.T, box_size=args.box_size,
        sigma=args.sigma, epsilon=args.epsilon, nu=args.nu, mu=args.mu,
        seed=args.seed, burn_in=result["burn_in"],
        thin_interval=result["thin_interval"],
        epsilon_mala=result["epsilon_mala_final"],
        acceptance_rate=result["acceptance_rate"],
        n_replicas=result["n_replicas"],
        T_ladder=result["T_ladder"],
        swap_rates=np.array(result["swap_rates"]),
        boundary=args.bc,
    )

    # Energy histogram
    hist_path = out_path.with_name(out_path.stem + "_energy_hist.npz")
    print(f"Saving energy histogram to {hist_path}")
    np.savez(
        str(hist_path),
        bin_edges=hist["bin_edges"], counts=hist["counts"],
        density=hist["density"], mean=hist["mean"], std=hist["std"],
        median=hist["median"], percentiles=hist["percentiles"],
        n_samples=hist["n_samples"], n=args.n, body=args.body, T=args.T,
    )

    # Metadata JSON
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    print(f"Saving metadata to {meta_path}")
    meta = {
        "n_particles": args.n, "body_order": args.body,
        "temperature": args.T, "box_size": args.box_size, "boundary": args.bc,
        "potential_params": {
            "sigma": args.sigma, "epsilon": args.epsilon,
            "nu": args.nu if args.body >= 3 else None,
            "mu": args.mu if args.body >= 4 else None,
        },
        "mcmc": {
            "method": "MALA + Parallel Tempering",
            "num_samples": len(result["energies"]),
            "burn_in": result["burn_in"],
            "thin_interval": result["thin_interval"],
            "epsilon_mala": result["epsilon_mala_final"],
            "n_replicas": result["n_replicas"],
            "T_ladder": result["T_ladder"].tolist(),
            "swap_rates": result["swap_rates"],
            "acceptance_rate": float(result["acceptance_rate"]),
            "seed": args.seed,
        },
        "energy_stats": {
            "mean": hist["mean"], "std": hist["std"],
            "median": hist["median"],
            "min": float(result["energies"].min()),
            "max": float(result["energies"].max()),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
