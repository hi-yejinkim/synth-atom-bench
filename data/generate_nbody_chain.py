"""MALA + Parallel Tempering sampler for chain (polymer) n-body systems.

Generates configurations from p(x) ∝ exp(-V(x) / T) where V(x) combines
bonded and non-bonded interactions along a linear chain topology 1-2-3-...-N.

Potentials (cumulative by body order)
--------------------------------------
body=2 : V = V_bond + V_LJ_nb
    - Bonded (1-2): harmonic bond V_bond = Σ_{i} k2 (r_{i,i+1} - r0)²
    - Non-bonded LJ (exclude 1-2 and 1-3; 1-4 scaled by 0.5, AMBER/OPLS convention):
        V_LJ = Σ_{i<j, |i-j|≥3} s_ij · 4ε [(σ/r_ij)^12 - (σ/r_ij)^6]
        where s_ij = 0.5 if |i-j|=3 (1-4 pair), else 1.0

body=3 : V = V_bond + V_LJ_nb + V_angle
    - Harmonic angle: V_angle = Σ_{i} k3 (θ_{i-1,i,i+1} - θ0)²

body=4 : V = V_bond + V_LJ_nb + V_angle + V_dihedral
    - Cosine dihedral (OPLS-style):
        V_dih = Σ_{i} [c1(1+cos φ) + c2(1-cos 2φ)]
      Creates trans (global min) and gauche (local min) states.

Sampling: MALA with parallel tempering for robust low-T sampling.
The chain topology is encoded in the data (.npz) so the model can use
index-based chain embeddings.

Usage
-----
    uv run data/generate_nbody_chain.py \\
        --N 20 --body 2 --T 1.0 --num_samples 50000 \\
        --output outputs/data/nbody_chain_N20_b2_T1.0/train.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Chain topology helpers
# ---------------------------------------------------------------------------

def build_nonbonded_pairs(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Build non-bonded pair indices with 1-4 scaling flags.

    Excludes 1-2 and 1-3 pairs. 1-4 pairs get 0.5 scaling (AMBER/OPLS convention),
    all other pairs get 1.0 scaling.

    Returns:
        pairs: (n_pairs, 2) upper-triangular pair indices
        scale: (n_pairs,) scaling factor per pair (0.5 for 1-4, 1.0 otherwise)
    """
    idx_i, idx_j = np.triu_indices(N, k=1)
    sep = idx_j - idx_i  # separation along chain
    # Exclude 1-2 (sep=1) and 1-3 (sep=2)
    valid = sep >= 3
    pairs = np.stack([idx_i[valid], idx_j[valid]], axis=1)
    # 1-4 pairs (sep=3) get 0.5 scaling
    scale = np.where(sep[valid] == 3, 0.5, 1.0)
    return pairs, scale


# ---------------------------------------------------------------------------
# Bond potential (harmonic, consecutive pairs)
# ---------------------------------------------------------------------------

def bond_energy(positions: np.ndarray, k2: float, r0: float) -> float:
    """Harmonic bond energy: V = Σ k2 (r_{i,i+1} - r0)². O(N)."""
    diffs = positions[1:] - positions[:-1]  # (N-1, 3)
    dists = np.linalg.norm(diffs, axis=-1)  # (N-1,)
    return float(k2 * np.sum((dists - r0) ** 2))


def bond_gradient(positions: np.ndarray, k2: float, r0: float) -> np.ndarray:
    """Analytic gradient of harmonic bond energy. Shape (N, 3)."""
    diffs = positions[1:] - positions[:-1]  # (N-1, 3)
    dists = np.linalg.norm(diffs, axis=-1, keepdims=True)  # (N-1, 1)
    dists_safe = np.maximum(dists, 1e-10)
    factor = 2.0 * k2 * (dists_safe - r0) / dists_safe  # (N-1, 1)
    bond_force = factor * diffs  # (N-1, 3)
    grad = np.zeros_like(positions)
    grad[1:] += bond_force
    grad[:-1] -= bond_force
    return grad


# ---------------------------------------------------------------------------
# Non-bonded LJ (exclude 1-2 and 1-3)
# ---------------------------------------------------------------------------

def nonbonded_lj_energy(positions: np.ndarray, sigma: float, epsilon_lj: float,
                        nb_pairs: np.ndarray, nb_scale: np.ndarray) -> float:
    """LJ energy for non-bonded pairs with per-pair scaling. O(N²) but sparse."""
    if len(nb_pairs) == 0:
        return 0.0
    d = positions[nb_pairs[:, 0]] - positions[nb_pairs[:, 1]]  # (n_pairs, 3)
    r_sq = np.maximum(np.sum(d ** 2, axis=-1), 1e-20)  # (n_pairs,)
    sr2 = sigma ** 2 / r_sq
    sr6 = sr2 * sr2 * sr2
    return float(4.0 * epsilon_lj * np.sum(nb_scale * (sr6 * sr6 - sr6)))


def nonbonded_lj_gradient(positions: np.ndarray, sigma: float, epsilon_lj: float,
                          nb_pairs: np.ndarray, nb_scale: np.ndarray) -> np.ndarray:
    """Analytic gradient of non-bonded LJ with per-pair scaling. Shape (N, 3)."""
    grad = np.zeros_like(positions)
    if len(nb_pairs) == 0:
        return grad
    idx_i, idx_j = nb_pairs[:, 0], nb_pairs[:, 1]
    d = positions[idx_i] - positions[idx_j]  # (n_pairs, 3)
    r_sq = np.maximum(np.sum(d ** 2, axis=-1, keepdims=True), 1e-20)  # (n_pairs, 1)
    sr2 = sigma ** 2 / r_sq
    sr6 = sr2 * sr2 * sr2
    # Factor: 24ε/r² (sr6 - 2·sr12), with per-pair scaling
    scale = nb_scale[:, None]  # (n_pairs, 1)
    factor = 24.0 * epsilon_lj * scale * (sr6 - 2.0 * sr6 * sr6) / r_sq  # (n_pairs, 1)
    pair_grad = factor * d  # dV/dx_i for this pair
    # Accumulate: grad[i] += dV/dx_i, grad[j] -= dV/dx_i
    np.add.at(grad, idx_i, pair_grad)
    np.subtract.at(grad, idx_j, pair_grad)
    return grad


# ---------------------------------------------------------------------------
# Angle potential (harmonic, consecutive triplets)
# ---------------------------------------------------------------------------

def angle_energy(positions: np.ndarray, k3: float, theta0: float) -> float:
    """Harmonic angle energy: V = Σ k3 (θ_{i-1,i,i+1} - θ0)². O(N)."""
    N = len(positions)
    if N < 3:
        return 0.0
    v1 = positions[:-2] - positions[1:-1]  # (N-2, 3)
    v2 = positions[2:] - positions[1:-1]   # (N-2, 3)
    cos_theta = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-10
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(k3 * np.sum((theta - theta0) ** 2))


def angle_gradient(positions: np.ndarray, k3: float, theta0: float) -> np.ndarray:
    """Analytic gradient of harmonic angle energy. Shape (N, 3)."""
    N = len(positions)
    grad = np.zeros_like(positions)
    if N < 3:
        return grad
    u = positions[:-2] - positions[1:-1]  # r_{i-1} - r_i
    v = positions[2:] - positions[1:-1]   # r_{i+1} - r_i
    u_norm = np.linalg.norm(u, axis=-1, keepdims=True) + 1e-10
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10
    cos_theta = np.sum(u * v, axis=-1, keepdims=True) / (u_norm * v_norm)
    cos_theta = np.clip(cos_theta, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    sin_theta = np.maximum(sin_theta, 1e-10)
    prefactor = 2.0 * k3 * (theta - theta0) / (-sin_theta)
    grad_i = prefactor * (v / (u_norm * v_norm) - cos_theta * u / u_norm ** 2)
    grad_k = prefactor * (u / (u_norm * v_norm) - cos_theta * v / v_norm ** 2)
    grad_j = -(grad_i + grad_k)
    grad[:-2] += grad_i
    grad[1:-1] += grad_j
    grad[2:] += grad_k
    return grad


# ---------------------------------------------------------------------------
# Dihedral potential (cosine OPLS-style, consecutive quadruplets)
# ---------------------------------------------------------------------------

def _dihedral_angles(positions: np.ndarray) -> np.ndarray:
    """Compute dihedral angles for consecutive quadruplets. Shape (N-3,)."""
    b1 = positions[1:-2] - positions[:-3]   # (N-3, 3)
    b2 = positions[2:-1] - positions[1:-2]  # (N-3, 3)
    b3 = positions[3:] - positions[2:-1]    # (N-3, 3)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-10
    n2_norm = np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-10
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    b2_norm = np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-10
    b2_hat = b2 / b2_norm
    m1 = np.cross(n1, b2_hat)
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    return np.arctan2(y, x)


def dihedral_energy(positions: np.ndarray, c1: float, c2: float) -> float:
    """OPLS-style cosine dihedral: V = Σ [c1(1+cos φ) + c2(1-cos 2φ)].

    Creates trans (φ=π, global min) and gauche (φ≈±60°, local min) states.
    """
    N = len(positions)
    if N < 4:
        return 0.0
    phi = _dihedral_angles(positions)
    return float(np.sum(c1 * (1.0 + np.cos(phi)) + c2 * (1.0 - np.cos(2.0 * phi))))


def dihedral_gradient(positions: np.ndarray, c1: float, c2: float) -> np.ndarray:
    """Analytic gradient of OPLS cosine dihedral. Shape (N, 3).

    Rewrites V = c1(1+cos φ) + c2(1-cos 2φ) in terms of x = cos φ:
        V = (c1+2c2) + c1·x - 2c2·x²,  dV/dx = c1 - 4c2·x
    then differentiates x = n1·n2/(|n1||n2|) analytically w.r.t. positions
    via the chain rule through bond vectors b1, b2, b3.
    """
    N = len(positions)
    grad = np.zeros_like(positions)
    if N < 4:
        return grad

    # Bond vectors: b1 = r_{i+1}-r_i, b2 = r_{i+2}-r_{i+1}, b3 = r_{i+3}-r_{i+2}
    b1 = positions[1:-2] - positions[:-3]   # (N-3, 3)
    b2 = positions[2:-1] - positions[1:-2]  # (N-3, 3)
    b3 = positions[3:] - positions[2:-1]    # (N-3, 3)

    # Normal vectors
    n1 = np.cross(b1, b2)  # (N-3, 3)
    n2 = np.cross(b2, b3)  # (N-3, 3)

    B = np.linalg.norm(n1, axis=-1, keepdims=True)  # |n1|, (N-3, 1)
    C = np.linalg.norm(n2, axis=-1, keepdims=True)  # |n2|, (N-3, 1)
    B = np.maximum(B, 1e-10)
    C = np.maximum(C, 1e-10)
    BC = B * C  # (N-3, 1)

    n1_hat = n1 / B  # (N-3, 3)
    n2_hat = n2 / C  # (N-3, 3)

    # cos φ = n1·n2 / (|n1|·|n2|)
    A = np.sum(n1 * n2, axis=-1, keepdims=True)  # n1·n2, (N-3, 1)
    x = np.clip(A / BC, -1.0, 1.0)  # cos φ, (N-3, 1)

    # dV/dx where V = c1(1+x) + c2(2-2x²)
    dVdx = c1 - 4.0 * c2 * x  # (N-3, 1)

    # Derivatives of A = n1·n2, B = |n1|, C = |n2| w.r.t. bond vectors
    # Using scalar triple product identities:
    #   d(n1·n2)/db1 = b2 × n2,     d|n1|/db1 = b2 × n1_hat
    #   d(n1·n2)/db2 = n2×b1 + b3×n1, d|n1|/db2 = n1_hat×b1, d|n2|/db2 = b3×n2_hat
    #   d(n1·n2)/db3 = n1 × b2,     d|n2|/db3 = n2_hat × b2

    B_sq = B ** 2  # (N-3, 1)
    C_sq = C ** 2  # (N-3, 1)

    # --- dx/db1 = dA_db1/(B·C) - A·dB_db1/(B²·C) ---
    dA_db1 = np.cross(b2, n2)           # (N-3, 3)
    dB_db1 = np.cross(b2, n1_hat)       # (N-3, 3)
    dx_db1 = dA_db1 / BC - A * dB_db1 / (B_sq * C)

    # --- dx/db3 = dA_db3/(B·C) - A·dC_db3/(B·C²) ---
    dA_db3 = np.cross(n1, b2)           # (N-3, 3)
    dC_db3 = np.cross(n2_hat, b2)       # (N-3, 3)
    dx_db3 = dA_db3 / BC - A * dC_db3 / (B * C_sq)

    # --- dx/db2 = dA_db2/(B·C) - A·dB_db2/(B²·C) - A·dC_db2/(B·C²) ---
    dA_db2 = np.cross(n2, b1) + np.cross(b3, n1)  # (N-3, 3)
    dB_db2 = np.cross(n1_hat, b1)       # (N-3, 3)
    dC_db2 = np.cross(b3, n2_hat)       # (N-3, 3)
    dx_db2 = dA_db2 / BC - A * dB_db2 / (B_sq * C) - A * dC_db2 / (B * C_sq)

    # Chain rule: dV/dr = dV/dx · dx/dr
    # dr_i → db1/dr_i = -I  →  dV/dr_i = dV/dx · (-dx/db1)
    # dr_j → db1/dr_j = +I, db2/dr_j = -I  →  dV/dr_j = dV/dx · (dx/db1 - dx/db2)
    # dr_k → db2/dr_k = +I, db3/dr_k = -I  →  dV/dr_k = dV/dx · (dx/db2 - dx/db3)
    # dr_l → db3/dr_l = +I  →  dV/dr_l = dV/dx · dx/db3
    grad[:-3]  += dVdx * (-dx_db1)              # atom i
    grad[1:-2] += dVdx * (dx_db1 - dx_db2)      # atom j
    grad[2:-1] += dVdx * (dx_db2 - dx_db3)      # atom k
    grad[3:]   += dVdx * dx_db3                  # atom l

    return grad


# ---------------------------------------------------------------------------
# Total energy and gradient
# ---------------------------------------------------------------------------

@dataclass
class ChainParams:
    """Parameters for chain potential."""
    body: int           # max body order: 2, 3, or 4
    N: int              # number of atoms in chain
    # Bond (2-body bonded)
    k2: float = 100.0   # bond spring constant (stiff to maintain chain)
    r0: float = 1.5     # equilibrium bond length
    # Non-bonded LJ (2-body non-bonded)
    sigma: float = 1.0
    epsilon_lj: float = 1.0
    # Angle (3-body bonded)
    k3: float = 20.0    # angle spring constant
    theta0: float = 1.911  # ~109.5° (tetrahedral), in radians
    # Dihedral (4-body bonded, OPLS cosine)
    c1: float = 1.0     # V = c1(1+cos φ) + c2(1-cos 2φ)
    c2: float = 0.5
    # Box (for non-bonded LJ cutoff and centering)
    box_size: float = 0.0  # computed from chain length
    # Precomputed topology
    nb_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    nb_scale: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self):
        if len(self.nb_pairs) == 0:
            self.nb_pairs, self.nb_scale = build_nonbonded_pairs(self.N)
        if self.box_size <= 0:
            # Box large enough to contain extended chain + buffer
            self.box_size = self.N * self.r0 * 1.5


def total_energy(positions: np.ndarray, params: ChainParams
                 ) -> tuple[float, float, float, float, float]:
    """Compute total energy with decomposition.

    Returns (E_total, E_bond, E_lj_nb, E_angle, E_dihedral).
    """
    e_bond = bond_energy(positions, params.k2, params.r0)
    e_lj = nonbonded_lj_energy(positions, params.sigma, params.epsilon_lj,
                                params.nb_pairs, params.nb_scale)
    e_angle = 0.0
    e_dih = 0.0
    if params.body >= 3:
        e_angle = angle_energy(positions, params.k3, params.theta0)
    if params.body >= 4:
        e_dih = dihedral_energy(positions, params.c1, params.c2)
    return e_bond + e_lj + e_angle + e_dih, e_bond, e_lj, e_angle, e_dih


def gradient_total(positions: np.ndarray, params: ChainParams) -> np.ndarray:
    """Total gradient. Shape (N, 3)."""
    grad = bond_gradient(positions, params.k2, params.r0)
    grad += nonbonded_lj_gradient(positions, params.sigma, params.epsilon_lj,
                                   params.nb_pairs, params.nb_scale)
    if params.body >= 3:
        grad += angle_gradient(positions, params.k3, params.theta0)
    if params.body >= 4:
        grad += dihedral_gradient(positions, params.c1, params.c2)
    return grad


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize_chain(N: int, r0: float, rng: np.random.Generator) -> np.ndarray:
    """Initialize chain along x-axis with small perturbation. Shape (N, 3)."""
    positions = np.zeros((N, 3))
    for i in range(N):
        positions[i, 0] = i * r0
    # Small random perturbation (5% of r0)
    positions += 0.05 * r0 * rng.standard_normal((N, 3))
    # Center at origin
    positions -= positions.mean(axis=0)
    return positions


# ---------------------------------------------------------------------------
# MALA step
# ---------------------------------------------------------------------------

def mala_step(positions: np.ndarray, energy: float, grad: np.ndarray,
              T: float, eps: float, params: ChainParams,
              rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray, bool]:
    """Single MALA step with M-H correction.

    Proposal: x' = x - (ε²/2T) ∇V(x) + ε η
    """
    drift = -(eps ** 2 / (2.0 * T)) * grad
    noise = eps * rng.standard_normal(positions.shape)
    proposal = positions + drift + noise

    # Center at origin (remove translational drift)
    proposal -= proposal.mean(axis=0)

    e_prop = total_energy(proposal, params)[0]
    if not math.isfinite(e_prop):
        return positions, energy, grad, False

    grad_prop = gradient_total(proposal, params)

    # M-H correction (log proposal densities)
    drift_rev = -(eps ** 2 / (2.0 * T)) * grad_prop
    fwd_diff = proposal - positions - drift
    rev_diff = positions - proposal - drift_rev
    log_q_fwd = -np.sum(fwd_diff ** 2) / (2.0 * eps ** 2)
    log_q_rev = -np.sum(rev_diff ** 2) / (2.0 * eps ** 2)

    log_alpha = -(e_prop - energy) / T + log_q_rev - log_q_fwd
    if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
        return proposal, e_prop, grad_prop, True
    return positions, energy, grad, False


# ---------------------------------------------------------------------------
# Parallel tempering
# ---------------------------------------------------------------------------

def compute_temperature_ladder(T_target: float, n_replicas: int,
                               T_max_factor: float = 4.0) -> np.ndarray:
    """Geometric temperature ladder."""
    T_max = T_target * T_max_factor
    if n_replicas == 1:
        return np.array([T_target])
    return np.geomspace(T_target, T_max, n_replicas)


def auto_n_replicas(T_target: float, body: int) -> int:
    """Auto-select replicas (chain needs more due to topology constraints)."""
    base = {0.5: 10, 0.8: 8, 1.0: 7, 1.5: 6, 2.0: 5, 3.0: 4}
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
    if body >= 4:
        n += 1
    return max(n, 2)


def mala_pt_sample(
    N: int,
    body: int,
    T: float,
    num_samples: int,
    k2: float = 100.0,
    r0: float = 1.5,
    sigma: float = 1.0,
    epsilon_lj: float = 1.0,
    k3: float = 20.0,
    theta0: float = 1.911,
    c1: float = 1.0,
    c2: float = 0.5,
    burn_in: int | None = None,
    thin_interval: int | None = None,
    epsilon_mala: float | None = None,
    n_replicas: int | None = None,
    T_max_factor: float = 4.0,
    seed: int = 42,
) -> dict:
    """Run MALA + Parallel Tempering for chain n-body system.

    Returns dict with positions, energies, and metadata.
    """
    if epsilon_mala is None:
        # More conservative for chain (bonded stiffness)
        epsilon_mala = 0.015 * math.sqrt(T) * (3 * N / 45.0) ** (-1.0 / 6.0)

    if n_replicas is None:
        n_replicas = auto_n_replicas(T, body)

    if thin_interval is None:
        thin_interval = 2 * N

    if burn_in is None:
        base = 100_000
        if body >= 4:
            base = 150_000
        t_factor = max(1.0, 1.0 / T)
        burn_in = int(base * min(t_factor, 5.0))

    rng = np.random.default_rng(seed)
    params = ChainParams(body=body, N=N, k2=k2, r0=r0, sigma=sigma,
                         epsilon_lj=epsilon_lj, k3=k3, theta0=theta0,
                         c1=c1, c2=c2)

    T_ladder = compute_temperature_ladder(T, n_replicas, T_max_factor)
    print(f"  Temperature ladder ({n_replicas} replicas): "
          f"{[f'{t:.3f}' for t in T_ladder]}")

    # Initialize replicas
    replicas = []
    for r in range(n_replicas):
        pos = initialize_chain(N, r0, rng)
        e_total = total_energy(pos, params)[0]
        grad = gradient_total(pos, params)
        replicas.append({
            'positions': pos,
            'energy': e_total,
            'gradient': grad,
            'epsilon': epsilon_mala * math.sqrt(T_ladder[r] / T),
        })

    # Dual averaging adaptation
    target_accept = 0.574
    adapt_until = burn_in // 2
    log_eps = [math.log(r['epsilon']) for r in replicas]
    log_eps_bar = [le for le in log_eps]
    h_bar = [0.0] * n_replicas
    mu_adapt = [math.log(10.0 * r['epsilon']) for r in replicas]

    # Storage
    total_steps = burn_in + num_samples * thin_interval
    samples = np.empty((num_samples, N, 3))
    energies = np.empty(num_samples)
    energies_bond = np.empty(num_samples)
    energies_lj = np.empty(num_samples)
    energies_angle = np.empty(num_samples)
    energies_dih = np.empty(num_samples)

    sample_idx = 0
    accept_counts = np.zeros(n_replicas)
    swap_counts = np.zeros(max(n_replicas - 1, 1))
    swap_attempts = np.zeros(max(n_replicas - 1, 1))
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

            # Dual averaging
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
        if n_replicas > 1:
            parity = step % 2
            for r in range(parity, n_replicas - 1, 2):
                swap_attempts[r] += 1
                beta_i = 1.0 / T_ladder[r]
                beta_j = 1.0 / T_ladder[r + 1]
                E_i = replicas[r]['energy']
                E_j = replicas[r + 1]['energy']
                log_alpha = (beta_i - beta_j) * (E_i - E_j)
                if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
                    replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]
                    eps_r = replicas[r]['epsilon']
                    replicas[r]['epsilon'] = replicas[r + 1]['epsilon']
                    replicas[r + 1]['epsilon'] = eps_r
                    swap_counts[r] += 1

        # Collect from target (coldest) replica
        if step >= burn_in and (step - burn_in) % thin_interval == 0:
            if sample_idx < num_samples:
                rep0 = replicas[0]
                pos = rep0['positions'].copy()
                # Center at origin
                pos -= pos.mean(axis=0)
                samples[sample_idx] = pos
                et, eb, elj, ea, ed = total_energy(pos, params)
                energies[sample_idx] = et
                energies_bond[sample_idx] = eb
                energies_lj[sample_idx] = elj
                energies_angle[sample_idx] = ea
                energies_dih[sample_idx] = ed
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

    for r in range(n_replicas):
        acc = accept_counts[r] / max(proposal_counts[r], 1)
        print(f"  Replica {r} (T={T_ladder[r]:.3f}): accept={acc:.3f}, "
              f"ε={replicas[r]['epsilon']:.4f}")
    for r in range(n_replicas - 1):
        sr = swap_counts[r] / max(swap_attempts[r], 1)
        print(f"  Swap {r}↔{r+1}: rate={sr:.3f}")

    overall_acc = accept_counts[0] / max(proposal_counts[0], 1)
    return {
        "positions": samples[:sample_idx].astype(np.float32),
        "energies": energies[:sample_idx].astype(np.float32),
        "energies_bond": energies_bond[:sample_idx].astype(np.float32),
        "energies_lj": energies_lj[:sample_idx].astype(np.float32),
        "energies_angle": energies_angle[:sample_idx].astype(np.float32),
        "energies_dihedral": energies_dih[:sample_idx].astype(np.float32),
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
    """Compute 1D energy histogram."""
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
        description="Generate chain n-body samples via MALA + Parallel Tempering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # 2-body chain (bonds + non-bonded LJ), N=20, T=1.0
  uv run data/generate_nbody_chain.py --N 20 --body 2 --T 1.0 \\
      --num_samples 50000 --output outputs/data/nbody_chain_N20_b2_T1.0/train.npz

  # 4-body chain (full: bonds + LJ + angles + dihedrals)
  uv run data/generate_nbody_chain.py --N 15 --body 4 --T 0.5 \\
      --num_samples 50000 --output outputs/data/nbody_chain_N15_b4_T0.5/train.npz
""",
    )
    parser.add_argument("--N", type=int, default=20, help="Chain length (default: 20)")
    parser.add_argument("--body", type=int, required=True, choices=[2, 3, 4],
                        help="Max body order (cumulative)")
    parser.add_argument("--T", type=float, required=True, help="Temperature")

    # Bond parameters
    parser.add_argument("--k2", type=float, default=100.0,
                        help="Bond spring constant (default: 100)")
    parser.add_argument("--r0", type=float, default=1.5,
                        help="Equilibrium bond length (default: 1.5)")

    # Non-bonded LJ
    parser.add_argument("--sigma", type=float, default=1.0, help="LJ σ (default: 1.0)")
    parser.add_argument("--epsilon_lj", type=float, default=1.0, help="LJ ε (default: 1.0)")

    # Angle parameters
    parser.add_argument("--k3", type=float, default=20.0,
                        help="Angle spring constant (default: 20)")
    parser.add_argument("--theta0", type=float, default=1.911,
                        help="Equilibrium angle in radians (default: 1.911 ≈ 109.5°)")

    # Dihedral parameters (OPLS cosine)
    parser.add_argument("--c1", type=float, default=1.0,
                        help="Dihedral cosine coefficient c1 (default: 1.0)")
    parser.add_argument("--c2", type=float, default=0.5,
                        help="Dihedral cosine coefficient c2 (default: 0.5)")

    # MCMC parameters
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--burn_in", type=int, default=None, help="Burn-in steps (auto)")
    parser.add_argument("--thin_interval", type=int, default=None, help="Thin interval (auto)")
    parser.add_argument("--epsilon_mala", type=float, default=None, help="MALA step size (auto)")
    parser.add_argument("--n_replicas", type=int, default=None, help="PT replicas (auto)")
    parser.add_argument("--T_max_factor", type=float, default=4.0,
                        help="T_max = T * factor (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", type=int, default=None, help="Pin to CPU core")

    parser.add_argument("--output", type=str, required=True, help="Output .npz file")
    parser.add_argument("--n_bins", type=int, default=200, help="Histogram bins")

    args = parser.parse_args()

    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
        print(f"[cpu] Pinned to core {args.cpu}")

    body_desc = {
        2: "bonds + non-bonded LJ",
        3: "bonds + LJ + angles",
        4: "bonds + LJ + angles + cosine dihedrals",
    }
    print(f"=== Chain MALA+PT sampler ===")
    print(f"  Chain length: {args.N}, Body order: {args.body} ({body_desc[args.body]})")
    print(f"  Temperature: {args.T}")
    print(f"  Bond: k2={args.k2}, r0={args.r0}")
    print(f"  Non-bonded LJ: σ={args.sigma}, ε={args.epsilon_lj}")
    if args.body >= 3:
        print(f"  Angle: k3={args.k3}, θ0={args.theta0:.3f} rad ({math.degrees(args.theta0):.1f}°)")
    if args.body >= 4:
        print(f"  Dihedral (OPLS cosine): c1={args.c1}, c2={args.c2}")
    print()

    result = mala_pt_sample(
        N=args.N, body=args.body, T=args.T, num_samples=args.num_samples,
        k2=args.k2, r0=args.r0, sigma=args.sigma, epsilon_lj=args.epsilon_lj,
        k3=args.k3, theta0=args.theta0, c1=args.c1, c2=args.c2,
        burn_in=args.burn_in, thin_interval=args.thin_interval,
        epsilon_mala=args.epsilon_mala, n_replicas=args.n_replicas,
        T_max_factor=args.T_max_factor, seed=args.seed,
    )

    print("\nComputing energy histogram...")
    hist = compute_energy_histogram(result["energies"], n_bins=args.n_bins)
    print(f"  Energy: mean={hist['mean']:.4f}, std={hist['std']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving samples to {out_path}")

    # Compute box_size from samples for dataset compatibility
    max_extent = np.abs(result["positions"]).max()
    box_size = 2.0 * max_extent * 1.2

    np.savez(
        str(out_path),
        positions=result["positions"],
        energies=result["energies"],
        energies_bond=result["energies_bond"],
        energies_lj=result["energies_lj"],
        energies_angle=result["energies_angle"],
        energies_dihedral=result["energies_dihedral"],
        # Metadata
        N=args.N, body=args.body, T=args.T,
        box_size=np.float32(box_size),
        topology="chain",  # signals chain topology to dataset/model
        k2=np.float32(args.k2), r0=np.float32(args.r0),
        sigma=np.float32(args.sigma), epsilon_lj=np.float32(args.epsilon_lj),
        k3=np.float32(args.k3), theta0=np.float32(args.theta0),
        c1=np.float32(args.c1), c2=np.float32(args.c2),
        seed=args.seed, burn_in=result["burn_in"],
        thin_interval=result["thin_interval"],
        epsilon_mala=result["epsilon_mala_final"],
        acceptance_rate=result["acceptance_rate"],
        n_replicas=result["n_replicas"],
        T_ladder=result["T_ladder"],
        swap_rates=np.array(result["swap_rates"]),
    )

    # Energy histogram
    hist_path = out_path.with_name(out_path.stem + "_energy_hist.npz")
    print(f"Saving energy histogram to {hist_path}")
    np.savez(
        str(hist_path),
        bin_edges=hist["bin_edges"], counts=hist["counts"],
        density=hist["density"], mean=hist["mean"], std=hist["std"],
        median=hist["median"], percentiles=hist["percentiles"],
        n_samples=hist["n_samples"], N=args.N, body=args.body, T=args.T,
    )

    # Metadata JSON
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    print(f"Saving metadata to {meta_path}")
    meta = {
        "topology": "chain",
        "n_atoms": args.N, "body_order": args.body,
        "temperature": args.T, "box_size": float(box_size),
        "potential_params": {
            "bond": {"k2": args.k2, "r0": args.r0},
            "nonbonded_lj": {"sigma": args.sigma, "epsilon": args.epsilon_lj,
                             "exclusions": "1-2 and 1-3",
                             "1-4_scaling": 0.5},
            "angle": {"k3": args.k3, "theta0": args.theta0} if args.body >= 3 else None,
            "dihedral": {"type": "cosine_opls", "c1": args.c1, "c2": args.c2}
                       if args.body >= 4 else None,
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
            "min": float(result["energies"].min()),
            "max": float(result["energies"].max()),
        },
        "design": (
            "Chain polymer with bonded (harmonic) + non-bonded (LJ, excl 1-2/1-3) "
            "interactions. Cosine OPLS dihedral creates trans/gauche metastability. "
            "Sampled via MALA + parallel tempering."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
