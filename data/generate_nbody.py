"""MCMC sampler for n-body energy distributions (Boltzmann sampling).

Generates configurations from p(x) ∝ exp(-V(x) / T) where V(x) is the
cumulative many-body potential up to the specified body order.

    body=2 : V = V_2                  (LJ pairs)
    body=3 : V = V_2 + V_3           (LJ + Axilrod-Teller)
    body=4 : V = V_2 + V_3 + V_4     (LJ + AT + tetrahedron)

This is the standard many-body expansion used in molecular simulation
(LAMMPS, GULP, etc.).  Higher body orders add correction terms on top of
the LJ pair potential — the LJ repulsive core prevents particle collapse
at all body orders.

Boundary conditions: periodic (minimum image convention) or hard wall.

Potentials
----------
2-body : Lennard-Jones pair potential
    V_2 = Σ_{i<j} 4ε [(σ/r_ij)^12 - (σ/r_ij)^6]

3-body : Axilrod-Teller triple-dipole potential
    V_3 = Σ_{i<j<k} ν (1 + 3 cos θ_i cos θ_j cos θ_k) / (r_ij r_jk r_ik)^3
    where θ_i is the angle at vertex i of triangle (i,j,k).

4-body : Distance-based tetrahedron potential
    V_4 = Σ_{i<j<k<l} μ · vol²(i,j,k,l) / Π_{a<b} r_ab²
    where vol² is the squared volume of the tetrahedron from the Cayley-Menger
    determinant.  This is an irreducible 4-body invariant expressible purely
    through pairwise distances.

MCMC
----
Metropolis-Hastings with single-particle displacements.  Acceptance criterion
uses the full cumulative Hamiltonian:
    α = min(1, exp(-(V_new - V_old) / T))

Outputs
-------
Saved .npz contains:
    positions       (num_samples, n, 3)  — sampled configurations
    energies        (num_samples,)       — total energy V per sample
    energies_2body  (num_samples,)       — V_2 contribution
    energies_3body  (num_samples,)       — V_3 contribution (body >= 3)
    energies_4body  (num_samples,)       — V_4 contribution (body >= 4)
    + metadata (n, body, T, box_size, sigma, epsilon, ...)

A companion *_energy_hist.npz stores the 1D energy histogram for quick
comparison.  For full Wasserstein-2 in configuration space, use the positions
array in the main .npz directly.

Usage
-----
    uv run data/generate_nbody.py \\
        --n 25 --body 2 --T 1.0 --num_samples 50000 \\
        --output outputs/data/nbody_n25_b2_T1.0/train.npz
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Sentinel energy returned on hard overlap (r < 1e-10).
# Using math.inf guarantees rejection regardless of other energy terms.
_OVERLAP_ENERGY = math.inf


# ---------------------------------------------------------------------------
# Minimum image convention helper
# ---------------------------------------------------------------------------

def _apply_bc_diff(diff: np.ndarray, box_size: float, bc: str = 'pbc') -> np.ndarray:
    """Apply boundary condition to displacement vectors.

    bc='pbc': minimum image convention (periodic).
    bc='hard_wall': no wrapping (all particles inside box).
    """
    if bc == 'pbc':
        return diff - box_size * np.round(diff / box_size)
    return diff


def _pairwise_dist_sq(positions: np.ndarray, box_size: float,
                      bc: str = 'pbc') -> np.ndarray:
    """Pairwise squared distances with boundary condition. Shape (n, n)."""
    diff = positions[:, None, :] - positions[None, :, :]
    diff = _apply_bc_diff(diff, box_size, bc)
    return np.sum(diff ** 2, axis=-1)


def _pairwise_dist(positions: np.ndarray, box_size: float,
                   bc: str = 'pbc') -> np.ndarray:
    """Pairwise distances with boundary condition. Shape (n, n)."""
    return np.sqrt(_pairwise_dist_sq(positions, box_size, bc))


# ---------------------------------------------------------------------------
# Potential energy functions
# ---------------------------------------------------------------------------

def energy_2body_fast(positions: np.ndarray, sigma: float, epsilon: float,
                      box_size: float, bc: str = 'pbc') -> float:
    """Vectorised Lennard-Jones pair potential."""
    diff = positions[:, None, :] - positions[None, :, :]
    diff = _apply_bc_diff(diff, box_size, bc)
    dist_sq = np.sum(diff ** 2, axis=-1)
    n = len(positions)
    idx = np.triu_indices(n, k=1)
    r_sq = dist_sq[idx]
    r_sq = np.maximum(r_sq, 1e-20)
    sr2 = sigma ** 2 / r_sq
    sr6 = sr2 * sr2 * sr2
    return float(4.0 * epsilon * np.sum(sr6 * sr6 - sr6))


def energy_3body(positions: np.ndarray, nu: float, box_size: float,
                 bc: str = 'pbc') -> float:
    """Axilrod-Teller triple-dipole potential.

    V_3 = Σ_{i<j<k} ν (1 + 3 cos θ_i cos θ_j cos θ_k) / (r_ij r_jk r_ik)^3
    """
    n = len(positions)
    if n < 3:
        return 0.0

    dist = _pairwise_dist(positions, box_size, bc)
    np.fill_diagonal(dist, 1e-10)

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                r_ij = dist[i, j]
                r_ik = dist[i, k]
                r_jk = dist[j, k]

                cos_i = (r_ij ** 2 + r_ik ** 2 - r_jk ** 2) / (2.0 * r_ij * r_ik)
                cos_j = (r_ij ** 2 + r_jk ** 2 - r_ik ** 2) / (2.0 * r_ij * r_jk)
                cos_k = (r_ik ** 2 + r_jk ** 2 - r_ij ** 2) / (2.0 * r_ik * r_jk)

                denom = (r_ij * r_jk * r_ik) ** 3
                if denom < 1e-30:
                    continue
                total += nu * (1.0 + 3.0 * cos_i * cos_j * cos_k) / denom

    return total


def _tetrahedron_det_sq(pi: np.ndarray, pj: np.ndarray,
                        pk: np.ndarray, pl: np.ndarray,
                        box_size: float, bc: str = 'pbc') -> float:
    """Squared determinant of edge matrix for tetrahedron (i,j,k,l).

    det(M) where M = [pj-pi, pk-pi, pl-pi].
    Returns det(M)² = (6·V)² = 36·V².  The factor 36 is absorbed into
    the coupling constant μ so callers use this directly.
    """
    e1 = _apply_bc_diff(pj - pi, box_size, bc)
    e2 = _apply_bc_diff(pk - pi, box_size, bc)
    e3 = _apply_bc_diff(pl - pi, box_size, bc)
    det = (e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
         - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
         + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]))
    return det * det


def energy_4body(positions: np.ndarray, mu: float, box_size: float,
                 bc: str = 'pbc') -> float:
    """Distance-based 4-body tetrahedron potential.

    V_4 = Σ_{i<j<k<l} μ · det(M)² / Π_{a<b} r_ab²

    where M = [pj-pi, pk-pi, pl-pi] and det(M)² = 36·V² (factor absorbed
    into μ).  This is an irreducible 4-body invariant.  The denominator
    provides distance decay so widely separated quadruplets contribute
    negligibly.

    Complexity: O(n⁴) with n⁴/24 quadruplets.
    """
    n = len(positions)
    if n < 4:
        return 0.0

    dist_sq = _pairwise_dist_sq(positions, box_size, bc)
    np.fill_diagonal(dist_sq, 1e-20)  # avoid division by zero

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    det_sq = _tetrahedron_det_sq(
                        positions[i], positions[j],
                        positions[k], positions[l],
                        box_size, bc,
                    )
                    # Product of 6 squared pairwise distances
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
    body: int           # max body order: 2, 3, or 4
    sigma: float        # LJ length scale
    epsilon: float      # LJ energy scale
    nu: float           # Axilrod-Teller coupling (3-body)
    mu: float           # tetrahedron coupling (4-body)
    box_size: float     # cubic box side length
    bc: str = 'pbc'     # boundary condition: 'pbc' or 'hard_wall'

    def __post_init__(self):
        if self.bc not in ('pbc', 'hard_wall'):
            raise ValueError(f"bc must be 'pbc' or 'hard_wall', got {self.bc!r}")


def total_energy(positions: np.ndarray, params: PotentialParams) -> tuple[float, float, float, float]:
    """Compute cumulative total energy and per-body-order decomposition.

    The total potential is V = V_2 + V_3 + V_4 (up to params.body).
    This is the standard many-body expansion — higher-order terms are
    corrections on top of the LJ pair potential, not replacements.

    Returns (E_total, E_2body, E_3body, E_4body).
    """
    e2 = energy_2body_fast(positions, params.sigma, params.epsilon,
                           params.box_size, params.bc)
    e3 = 0.0
    e4 = 0.0

    if params.body >= 3:
        e3 = energy_3body(positions, params.nu, params.box_size, params.bc)
    if params.body >= 4:
        e4 = energy_4body(positions, params.mu, params.box_size, params.bc)

    return e2 + e3 + e4, e2, e3, e4


# ---------------------------------------------------------------------------
# Gradient functions (for MALA)
# ---------------------------------------------------------------------------

def gradient_2body(positions: np.ndarray, sigma: float, epsilon: float,
                   box_size: float, bc: str = 'pbc') -> np.ndarray:
    """Vectorised gradient of LJ potential w.r.t. all positions. Shape (n, 3).

    ∂V_2/∂x_i = Σ_j 4ε [-12σ¹²/r¹⁴ + 6σ⁶/r⁸] (x_i - x_j)
    """
    diff = positions[:, None, :] - positions[None, :, :]   # (n, n, 3)
    diff = _apply_bc_diff(diff, box_size, bc)
    dist_sq = np.sum(diff ** 2, axis=-1, keepdims=True)    # (n, n, 1)
    dist_sq = np.maximum(dist_sq, 1e-20)

    sr2 = sigma ** 2 / dist_sq                              # (n, n, 1)
    sr6 = sr2 * sr2 * sr2
    # Force factor: 4ε[-12 sr12/r² + 6 sr6/r²] = 4ε/r² [-12sr12 + 6sr6]
    # = 24ε/r² [sr6 - 2sr12]
    factor = 24.0 * epsilon * (sr6 - 2.0 * sr6 * sr6) / dist_sq  # (n, n, 1)

    # Zero out self-interactions
    n = len(positions)
    np.fill_diagonal(factor[:, :, 0], 0.0)

    # grad_i = -Σ_j factor_ij * diff_ij  (diff = x_i - x_j)
    grad = -np.sum(factor * diff, axis=1)                   # (n, 3)
    return grad


def gradient_3body(positions: np.ndarray, nu: float, box_size: float,
                   bc: str = 'pbc') -> np.ndarray:
    """Gradient of Axilrod-Teller potential. Shape (n, 3).

    Computed via finite differences for robustness.  O(N³ * N) total,
    but only called once per MALA step.
    """
    n = len(positions)
    grad = np.zeros((n, 3))
    h = 1e-5
    for i in range(n):
        for d in range(3):
            pos_plus = positions.copy()
            pos_plus[i, d] += h
            pos_minus = positions.copy()
            pos_minus[i, d] -= h
            e_plus = energy_3body(pos_plus, nu, box_size, bc)
            e_minus = energy_3body(pos_minus, nu, box_size, bc)
            grad[i, d] = (e_plus - e_minus) / (2.0 * h)
    return grad


def gradient_total(positions: np.ndarray, params: PotentialParams) -> np.ndarray:
    """Gradient of total cumulative potential. Shape (n, 3)."""
    grad = gradient_2body(positions, params.sigma, params.epsilon,
                          params.box_size, params.bc)
    if params.body >= 3:
        grad += gradient_3body(positions, params.nu, params.box_size, params.bc)
    # 4-body gradient omitted — use RW for body>=4
    return grad


# ---------------------------------------------------------------------------
# Single-particle energy differences (efficient MCMC updates)
# ---------------------------------------------------------------------------

def delta_energy_2body(positions: np.ndarray, idx: int, old_pos: np.ndarray,
                       new_pos: np.ndarray, sigma: float, epsilon: float,
                       box_size: float, bc: str = 'pbc') -> float:
    """O(N) change in 2-body energy when moving particle idx."""
    n = len(positions)
    delta = 0.0
    for j in range(n):
        if j == idx:
            continue
        pj = positions[j]

        # Old contribution
        d_old = _apply_bc_diff(old_pos - pj, box_size, bc)
        r_old = np.linalg.norm(d_old)
        if r_old < 1e-10:
            r_old = 1e-10
        sr6_old = (sigma / r_old) ** 6
        e_old = 4.0 * epsilon * (sr6_old ** 2 - sr6_old)

        # New contribution
        d_new = _apply_bc_diff(new_pos - pj, box_size, bc)
        r_new = np.linalg.norm(d_new)
        if r_new < 1e-10:
            return _OVERLAP_ENERGY
        sr6_new = (sigma / r_new) ** 6
        e_new = 4.0 * epsilon * (sr6_new ** 2 - sr6_new)

        delta += e_new - e_old
    return delta


def delta_energy_3body(positions: np.ndarray, idx: int, old_pos: np.ndarray,
                       new_pos: np.ndarray, nu: float, box_size: float,
                       bc: str = 'pbc') -> float:
    """O(N²) change in 3-body energy when moving particle idx.

    Only triplets containing idx contribute to the delta.

    Precondition: this function reads only positions[j] for j != idx.
    The value of positions[idx] is ignored — old_pos and new_pos are
    used explicitly.  Safe to call before or after mutating positions[idx].
    """
    n = len(positions)
    if n < 3:
        return 0.0

    def _at_triplet(pi: np.ndarray, pj: np.ndarray, pk: np.ndarray) -> float:
        """Axilrod-Teller energy for a single triplet (i, j, k)."""
        dij = _apply_bc_diff(pi - pj, box_size, bc)
        dik = _apply_bc_diff(pi - pk, box_size, bc)
        djk = _apply_bc_diff(pj - pk, box_size, bc)
        r_ij = np.linalg.norm(dij)
        r_ik = np.linalg.norm(dik)
        r_jk = np.linalg.norm(djk)
        if r_ij < 1e-10 or r_ik < 1e-10 or r_jk < 1e-10:
            return 0.0
        cos_i = (r_ij**2 + r_ik**2 - r_jk**2) / (2.0 * r_ij * r_ik)
        cos_j = (r_ij**2 + r_jk**2 - r_ik**2) / (2.0 * r_ij * r_jk)
        cos_k = (r_ik**2 + r_jk**2 - r_ij**2) / (2.0 * r_ik * r_jk)
        denom = (r_ij * r_jk * r_ik) ** 3
        if denom < 1e-30:
            return 0.0
        return nu * (1.0 + 3.0 * cos_i * cos_j * cos_k) / denom

    delta = 0.0
    for j in range(n):
        if j == idx:
            continue
        for k in range(j + 1, n):
            if k == idx:
                continue
            # This triplet contains idx — compute old and new
            delta -= _at_triplet(old_pos, positions[j], positions[k])
            delta += _at_triplet(new_pos, positions[j], positions[k])

    return delta


def delta_energy_4body(positions: np.ndarray, idx: int, old_pos: np.ndarray,
                        new_pos: np.ndarray, mu: float, box_size: float,
                        bc: str = 'pbc') -> float:
    """O(N³) change in 4-body energy when moving particle idx.

    Only quadruplets containing idx contribute to the delta.
    """
    n = len(positions)
    if n < 4:
        return 0.0

    def _quad_energy(pi: np.ndarray, pj: np.ndarray,
                     pk: np.ndarray, pl: np.ndarray) -> float:
        """Tetrahedron potential for a single quadruplet.

        Computes edge vectors once, reuses for both det² and distances.
        """
        # 3 edges from pi, 3 edges among (pj, pk, pl)
        e_ij = _apply_bc_diff(pj - pi, box_size, bc)
        e_ik = _apply_bc_diff(pk - pi, box_size, bc)
        e_il = _apply_bc_diff(pl - pi, box_size, bc)
        e_jk = _apply_bc_diff(pk - pj, box_size, bc)
        e_jl = _apply_bc_diff(pl - pj, box_size, bc)
        e_kl = _apply_bc_diff(pl - pk, box_size, bc)
        # det(M) where M = [e_ij, e_ik, e_il]
        det = (e_ij[0] * (e_ik[1] * e_il[2] - e_ik[2] * e_il[1])
             - e_ij[1] * (e_ik[0] * e_il[2] - e_ik[2] * e_il[0])
             + e_ij[2] * (e_ik[0] * e_il[1] - e_ik[1] * e_il[0]))
        det_sq = det * det
        prod_r2 = (float(np.dot(e_ij, e_ij)) * float(np.dot(e_ik, e_ik))
                  * float(np.dot(e_il, e_il)) * float(np.dot(e_jk, e_jk))
                  * float(np.dot(e_jl, e_jl)) * float(np.dot(e_kl, e_kl)))
        if prod_r2 < 1e-60:
            return 0.0
        return mu * det_sq / prod_r2

    delta = 0.0
    others = [i for i in range(n) if i != idx]
    m = len(others)
    for ai in range(m):
        for bi in range(ai + 1, m):
            for ci in range(bi + 1, m):
                j, k, l = others[ai], others[bi], others[ci]
                delta -= _quad_energy(old_pos, positions[j],
                                      positions[k], positions[l])
                delta += _quad_energy(new_pos, positions[j],
                                      positions[k], positions[l])
    return delta


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

    positions = np.array(positions[:n])

    # Check minimum separation
    if n > 1:
        dist_sq = _pairwise_dist_sq(positions, box_size, bc)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist = np.sqrt(dist_sq.min())
        if min_dist < 0.5 * sigma:
            print(f"  WARNING: Initial min distance {min_dist:.3f} < 0.5σ={0.5*sigma:.3f}. "
                  f"Consider increasing box_size or reducing n.")

    return positions


# ---------------------------------------------------------------------------
# MCMC sampler
# ---------------------------------------------------------------------------

def mcmc_sample(
    n: int,
    body: int,
    T: float,
    num_samples: int,
    box_size: float = 10.0,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    nu: float = 0.1,
    mu: float = 0.01,
    burn_in: int | None = None,
    thin_interval: int | None = None,
    step_size: float = 5.0,
    seed: int = 42,
    bc: str = 'pbc',
) -> dict:
    """Run RW-MH Boltzmann sampler for n-body system.

    Sampling Hamiltonian is the full cumulative potential V = V_2 [+ V_3 [+ V_4]]
    depending on the body order.  The Metropolis acceptance uses ΔV of the full
    Hamiltonian.

    bc: 'pbc' (periodic) or 'hard_wall' (reject out-of-bounds proposals).

    Returns dict with positions, energies (total and per-body), and metadata.
    """
    rng = np.random.default_rng(seed)
    params = PotentialParams(body=body, sigma=sigma, epsilon=epsilon,
                             nu=nu, mu=mu, box_size=box_size, bc=bc)

    if thin_interval is None:
        thin_interval = max(n * 10, 100)

    # Adaptive burn-in: scale with system size and inverse temperature
    if burn_in is None:
        base_burn = max(50_000, n * thin_interval * 10)
        # Low temperature needs more burn-in (slower mixing)
        t_factor = max(1.0, 1.0 / T) if T > 0 else 10.0
        burn_in = int(base_burn * min(t_factor, 10.0))

    effective_burnin = burn_in // thin_interval
    if effective_burnin < 500:
        print(f"  WARNING: burn-in covers only ~{effective_burnin} thinned samples. "
              f"Consider increasing --burn_in for low-T or large-N runs.")

    disp = step_size * sigma  # displacement scale ~ σ

    # Initialise on lattice
    positions = initialize_on_lattice(n, box_size, sigma, rng, bc)
    e_total, e2, e3, e4 = total_energy(positions, params)

    total_steps = burn_in + num_samples * thin_interval
    samples = np.empty((num_samples, n, 3))
    energies = np.empty(num_samples)
    energies_2 = np.empty(num_samples)
    energies_3 = np.empty(num_samples)
    energies_4 = np.empty(num_samples)

    sample_idx = 0
    accepted_burn = 0
    proposals_burn = 0
    accepted_prod = 0
    proposals_prod = 0
    report_interval = max(1, total_steps // 100)

    t0 = time.time()
    for step in range(total_steps):
        atom = rng.integers(n)
        old_pos = positions[atom].copy()

        # Propose displacement
        new_pos = old_pos + rng.uniform(-disp, disp, size=3)
        if bc == 'pbc':
            new_pos = new_pos % box_size

        # Hard wall: out of bounds → guaranteed rejection (de = inf)
        oob = (bc == 'hard_wall'
               and (np.any(new_pos < 0) or np.any(new_pos >= box_size)))

        if oob:
            de = math.inf
        else:
            # Local delta updates: O(N) for 2-body, O(N²) for 3-body,
            # O(N³) for 4-body.
            de = delta_energy_2body(positions, atom, old_pos, new_pos,
                                    sigma, epsilon, box_size, bc)
            if not math.isfinite(de):
                pass  # hard overlap — guaranteed reject
            else:
                if body >= 3:
                    de += delta_energy_3body(positions, atom, old_pos, new_pos,
                                             nu, box_size, bc)
                if math.isfinite(de) and body >= 4:
                    de += delta_energy_4body(positions, atom, old_pos, new_pos,
                                             mu, box_size, bc)

        # Metropolis criterion
        if math.isfinite(de) and (de <= 0.0 or rng.random() < np.exp(-de / T)):
            positions[atom] = new_pos
            e_total += de
            if step < burn_in:
                accepted_burn += 1
            else:
                accepted_prod += 1

        if step < burn_in:
            proposals_burn += 1
        else:
            proposals_prod += 1

        # Collect sample after burn-in
        if step >= burn_in and (step - burn_in) % thin_interval == 0:
            samples[sample_idx] = positions.copy()

            # Decompose energy explicitly for correct per-body attribution
            et, e2c, e3c, e4c = total_energy(positions, params)
            energies[sample_idx] = et
            energies_2[sample_idx] = e2c
            energies_3[sample_idx] = e3c
            energies_4[sample_idx] = e4c

            sample_idx += 1

        if (step + 1) % report_interval == 0:
            elapsed = time.time() - t0
            pct = (step + 1) / total_steps * 100
            rate = (step + 1) / elapsed
            total_acc = (accepted_burn + accepted_prod)
            total_prop = (proposals_burn + proposals_prod)
            acc_rate = total_acc / total_prop if total_prop > 0 else 0
            print(
                f"\r  {pct:5.1f}% | {sample_idx}/{num_samples} samples | "
                f"accept={acc_rate:.3f} | {rate:.0f} steps/s | "
                f"E={e_total:.2f}",
                end="", flush=True,
            )

    print()
    burn_acc = accepted_burn / proposals_burn if proposals_burn > 0 else 0
    prod_acc = accepted_prod / proposals_prod if proposals_prod > 0 else 0
    overall_acc = (accepted_burn + accepted_prod) / (proposals_burn + proposals_prod)
    elapsed = time.time() - t0
    print(f"  Done: {sample_idx} samples in {elapsed:.1f}s")
    print(f"  Acceptance rate: burn-in={burn_acc:.3f}, production={prod_acc:.3f}")
    print(f"  Energy range: [{energies[:sample_idx].min():.2f}, {energies[:sample_idx].max():.2f}]")

    return {
        "positions": samples[:sample_idx].astype(np.float32),
        "energies": energies[:sample_idx].astype(np.float32),
        "energies_2body": energies_2[:sample_idx].astype(np.float32),
        "energies_3body": energies_3[:sample_idx].astype(np.float32),
        "energies_4body": energies_4[:sample_idx].astype(np.float32),
        "acceptance_rate": overall_acc,
        "acceptance_rate_burnin": burn_acc,
        "acceptance_rate_production": prod_acc,
        "burn_in": burn_in,
        "thin_interval": thin_interval,
    }



# NOTE: MALA sampler removed — RW-MH is used exclusively.
# Hard wall BC makes MALA problematic (acceptance rate drops when any
# particle's gradient-directed proposal lands outside the box).
# RW-MH with single-particle moves is simpler, correct for all BC
# types, and sufficient for the system sizes in this benchmark.


# ---------------------------------------------------------------------------
# Energy histogram (1D marginal — NOT config-space W2 reference)
# ---------------------------------------------------------------------------

def compute_energy_histogram(energies: np.ndarray,
                             n_bins: int = 200) -> dict:
    """Compute 1D energy histogram for quick distribution comparison.

    NOTE: This is the energy *marginal*.  For Wasserstein-2 in configuration
    space (R^{N×3}), use the full positions array from the main .npz file.
    W2 on the 1D energy is a lower bound on W2 in configuration space.
    """
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
        description="Generate n-body energy distribution samples via RW-MH MCMC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # 2-body, 25 particles, T=1.0 (PBC)
  uv run data/generate_nbody.py --n 25 --body 2 --T 1.0 --num_samples 50000 \\
      --output outputs/data/nbody_n25_b2_T1.0/train.npz

  # 3-body, hard wall boundary
  uv run data/generate_nbody.py --n 25 --body 3 --T 0.5 --bc hard_wall \\
      --num_samples 50000 --output outputs/data/nbody_n25_b3_T0.5_hw/train.npz

  # 4-body, 15 particles (O(n^4) tetrahedron potential)
  uv run data/generate_nbody.py --n 15 --body 4 --T 1.0 --num_samples 10000 \\
      --output outputs/data/nbody_n15_b4_T1.0/train.npz
""",
    )

    # System parameters
    parser.add_argument("--n", type=int, default=25,
                        help="Number of particles (default: 25)")
    parser.add_argument("--body", type=int, required=True, choices=[2, 3, 4],
                        help="Max body interaction order (2, 3, or 4). "
                             "Cumulative: body=3 → V_2+V_3.")
    parser.add_argument("--T", type=float, required=True,
                        help="Temperature in units of ε/k_B (controls distribution sharpness)")
    parser.add_argument("--bc", type=str, default="pbc", choices=["pbc", "hard_wall"],
                        help="Boundary condition: 'pbc' (periodic) or 'hard_wall' (default: pbc)")

    # Potential parameters
    parser.add_argument("--box_size", type=float, default=10.0,
                        help="Cubic box side length (default: 10.0)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="LJ length scale σ (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="LJ energy scale ε (default: 1.0)")
    parser.add_argument("--nu", type=float, default=0.1,
                        help="Axilrod-Teller coupling for 3-body (default: 0.1). "
                             "Should satisfy |V_3| << |V_2| for many-body hierarchy.")
    parser.add_argument("--mu", type=float, default=0.01,
                        help="Tetrahedron coupling for 4-body (default: 0.01). "
                             "Should satisfy |V_4| << |V_3| for many-body hierarchy.")

    # MCMC parameters
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of MCMC samples to collect")
    parser.add_argument("--burn_in", type=int, default=None,
                        help="Burn-in steps (default: adaptive based on n, T)")
    parser.add_argument("--thin_interval", type=int, default=None,
                        help="Thinning interval (default: max(n*10, 100))")
    parser.add_argument("--step_size", type=float, default=0.3,
                        help="Displacement step size in units of σ (default: 0.3). "
                             "For hard_wall BC, consider step_size <= 1.0 to "
                             "avoid high OOB rejection rates near walls.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz file path")
    parser.add_argument("--n_bins", type=int, default=200,
                        help="Number of bins for energy histogram (default: 200)")

    args = parser.parse_args()

    # Compute actual burn-in/thin for display
    thin = args.thin_interval or max(args.n * 10, 100)
    if args.burn_in is not None:
        burn = args.burn_in
    else:
        base_burn = max(50_000, args.n * thin * 10)
        t_factor = max(1.0, 1.0 / args.T) if args.T > 0 else 10.0
        burn = int(base_burn * min(t_factor, 10.0))

    bc_label = "PBC" if args.bc == "pbc" else "hard wall"
    print(f"=== n-body RW-MH sampler ({bc_label}) ===")
    print(f"  Particles: {args.n}")
    print(f"  Body order: {args.body}-body (cumulative: V = V_2{' + V_3' if args.body >= 3 else ''}{' + V_4' if args.body >= 4 else ''})")
    print(f"  Temperature: {args.T} (ε/k_B)")
    print(f"  Box size: {args.box_size}, BC: {bc_label}")
    print(f"  LJ params: σ={args.sigma}, ε={args.epsilon}")
    if args.body >= 3:
        print(f"  Axilrod-Teller ν={args.nu}")
    if args.body >= 4:
        print(f"  Tetrahedron μ={args.mu}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Burn-in: {burn}, thin: {thin}")
    print()

    # Run MCMC
    result = mcmc_sample(
        n=args.n,
        body=args.body,
        T=args.T,
        num_samples=args.num_samples,
        box_size=args.box_size,
        sigma=args.sigma,
        epsilon=args.epsilon,
        nu=args.nu,
        mu=args.mu,
        burn_in=args.burn_in,
        thin_interval=args.thin_interval,
        step_size=args.step_size,
        seed=args.seed,
        bc=args.bc,
    )

    # Compute energy histogram
    print("\nComputing energy histogram...")
    hist = compute_energy_histogram(result["energies"], n_bins=args.n_bins)
    print(f"  Energy: mean={hist['mean']:.4f}, std={hist['std']:.4f}")
    print(f"  Percentiles [5,25,50,75,95]: {hist['percentiles']}")

    # Save main data
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
        # Metadata
        n=args.n,
        body=args.body,
        T=args.T,
        box_size=args.box_size,
        sigma=args.sigma,
        epsilon=args.epsilon,
        nu=args.nu,
        mu=args.mu,
        seed=args.seed,
        burn_in=result["burn_in"],
        thin_interval=result["thin_interval"],
        step_size=args.step_size,
        acceptance_rate=result["acceptance_rate"],
        acceptance_rate_burnin=result["acceptance_rate_burnin"],
        acceptance_rate_production=result["acceptance_rate_production"],
        boundary=args.bc,
    )

    # Save energy histogram alongside
    hist_path = out_path.with_name(out_path.stem + "_energy_hist.npz")
    print(f"Saving energy histogram to {hist_path}")
    np.savez(
        str(hist_path),
        bin_edges=hist["bin_edges"],
        counts=hist["counts"],
        density=hist["density"],
        mean=hist["mean"],
        std=hist["std"],
        median=hist["median"],
        percentiles=hist["percentiles"],
        n_samples=hist["n_samples"],
        n=args.n,
        body=args.body,
        T=args.T,
    )

    # Save human-readable metadata as JSON
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    print(f"Saving metadata to {meta_path}")
    meta = {
        "n_particles": args.n,
        "body_order": args.body,
        "temperature": args.T,
        "box_size": args.box_size,
        "boundary": args.bc,
        "potential_params": {
            "sigma": args.sigma,
            "epsilon": args.epsilon,
            "nu": args.nu if args.body >= 3 else None,
            "mu": args.mu if args.body >= 4 else None,
        },
        "mcmc": {
            "num_samples": len(result["energies"]),
            "burn_in": result["burn_in"],
            "thin_interval": result["thin_interval"],
            "step_size": args.step_size,
            "seed": args.seed,
            "acceptance_rate": float(result["acceptance_rate"]),
            "acceptance_rate_burnin": float(result["acceptance_rate_burnin"]),
            "acceptance_rate_production": float(result["acceptance_rate_production"]),
        },
        "energy_stats": {
            "mean": hist["mean"],
            "std": hist["std"],
            "median": hist["median"],
            "min": float(result["energies"].min()),
            "max": float(result["energies"].max()),
        },
        "design": "cumulative many-body expansion — body=k samples from "
                  "exp(-(V_2+...+V_k)/T) using RW-MH with single-particle moves.",
        "note": "For W2 in configuration space, use positions array directly. "
                "The energy_hist.npz is a 1D marginal only.",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
