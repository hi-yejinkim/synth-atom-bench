"""MCMC sampler for VSEPR local geometry configurations.

Generates molecules with one central atom and 2–4 ligand atoms.
Constraints:
  - Bond length: each ligand-central distance within [bond_min, bond_max]
  - Bond angle: each ligand-pair angle within target ± tol (lone pairs distort target)
  - Planarity: if has_pi, all atoms must be coplanar (pi bond → sp2-like constraint)
  - No clash: ligand-ligand distances > 2 * radius
"""

import argparse
import os
import time

import numpy as np

# -------------------------------------------------------------------------
# Ideal unit-direction vectors for each orbital type
# -------------------------------------------------------------------------
# Tetrahedral: pairwise angle = arccos(-1/3) ≈ 109.47°
_TETRA_DIRS = np.array([
    [0.0, 0.0, 1.0],
    [2 * np.sqrt(2) / 3, 0.0, -1.0 / 3],
    [-np.sqrt(2) / 3, np.sqrt(2.0 / 3), -1.0 / 3],
    [-np.sqrt(2) / 3, -np.sqrt(2.0 / 3), -1.0 / 3],
])

# Trigonal planar: 120° apart in xy-plane
_TRIG_DIRS = np.array([
    [1.0, 0.0, 0.0],
    [-0.5, np.sqrt(3) / 2, 0.0],
    [-0.5, -np.sqrt(3) / 2, 0.0],
])

# Linear: 180° apart along z-axis
_LINEAR_DIRS = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
])

ORBITAL_PARAMS: dict = {
    "sp3": {
        "base_angle_deg": 109.5,
        "bond_range": (1.4, 1.6),
        "ideal_dirs": _TETRA_DIRS,
        "n_ligands_max": 4,
    },
    "sp2": {
        "base_angle_deg": 120.0,
        "bond_range": (1.2, 1.4),
        "ideal_dirs": _TRIG_DIRS,
        "n_ligands_max": 3,
    },
    "sp": {
        "base_angle_deg": 180.0,
        "bond_range": (1.0, 1.2),
        "ideal_dirs": _LINEAR_DIRS,
        "n_ligands_max": 2,
    },
}

# Each lone pair reduces the bond angle by ~2.5° (VSEPR rule)
_LONE_PAIR_ANGLE_REDUCTION_DEG = 2.5

# Angle acceptance window: ±N sigma from target
_ANGLE_ACCEPT_SIGMA = 3.0

# Planarity tolerance for has_pi constraint (Angstroms)
_PLANE_TOL = 0.2


def get_target_angle_deg(orbital_type: str, n_lonepairs: int) -> float:
    """Bond angle target with lone pair distortion."""
    base = ORBITAL_PARAMS[orbital_type]["base_angle_deg"]
    return base - _LONE_PAIR_ANGLE_REDUCTION_DEG * n_lonepairs


def get_angle_sigma_deg(n_lonepairs: int) -> float:
    """Acceptance window half-width: tighter without lone pairs."""
    return 5.0 if n_lonepairs > 0 else 2.0


def _bond_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Bond angle in degrees between vectors v1 and v2 from the origin."""
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _plane_deviation(positions: np.ndarray) -> float:
    """Max distance of any atom from the best-fit plane through all atoms.

    Uses PCA: normal to the plane is the smallest singular vector of the
    centered positions matrix.
    """
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    if centered.shape[0] < 3:
        return 0.0
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    return float(np.abs(centered @ normal).max())


def check_constraints(
    positions: np.ndarray,
    orbital_type: str,
    n_lonepairs: int,
    has_pi: bool,
    bond_range: tuple[float, float],
    radius: float,
) -> bool:
    """Check all VSEPR constraints.

    positions[0] = central atom (fixed at origin in MCMC, then centered).
    positions[1:] = ligand atoms.
    """
    bond_min, bond_max = bond_range
    central = positions[0]
    ligands = positions[1:]
    n_ligands = len(ligands)

    target_angle = get_target_angle_deg(orbital_type, n_lonepairs)
    angle_sigma = get_angle_sigma_deg(n_lonepairs)
    angle_tol = _ANGLE_ACCEPT_SIGMA * angle_sigma

    # 1. Bond lengths
    for lig in ligands:
        bl = float(np.linalg.norm(lig - central))
        if not (bond_min <= bl <= bond_max):
            return False

    # 2. Bond angles (every ligand pair through central atom)
    for i in range(n_ligands):
        for j in range(i + 1, n_ligands):
            v1 = ligands[i] - central
            v2 = ligands[j] - central
            angle = _bond_angle_deg(v1, v2)
            if abs(angle - target_angle) > angle_tol:
                return False

    # 3. Planarity constraint (has_pi → coplanar molecule)
    if has_pi and n_ligands >= 3:
        if _plane_deviation(positions) > _PLANE_TOL:
            return False

    # 4. Ligand–ligand clash avoidance
    threshold_sq = (2.0 * radius) ** 2
    for i in range(n_ligands):
        for j in range(i + 1, n_ligands):
            diff = ligands[i] - ligands[j]
            if float(np.dot(diff, diff)) < threshold_sq:
                return False

    return True


def initialize_vsepr(
    orbital_type: str,
    n_lonepairs: int,
    has_pi: bool,
    bond_range: tuple[float, float],
    radius: float,
    rng: np.random.Generator,
    max_attempts: int = 20_000,
) -> np.ndarray:
    """Initialize from ideal geometry plus small random perturbation.

    Returns positions (1 + n_ligands, 3) with central atom at index 0.
    """
    params = ORBITAL_PARAMS[orbital_type]
    n_ligands = max(2, params["n_ligands_max"] - n_lonepairs)
    ideal_dirs = params["ideal_dirs"][:n_ligands]
    bond_target = 0.5 * (bond_range[0] + bond_range[1])
    bond_half = 0.5 * (bond_range[1] - bond_range[0])
    angle_sigma_rad = np.deg2rad(get_angle_sigma_deg(n_lonepairs))

    for _ in range(max_attempts):
        positions = np.zeros((n_ligands + 1, 3))
        # Central atom at origin; ligands perturbed from ideal directions
        for i in range(n_ligands):
            direction = ideal_dirs[i].copy()
            direction += rng.normal(0.0, angle_sigma_rad * 0.5, 3)
            direction /= np.linalg.norm(direction)
            bl = bond_target + rng.uniform(-bond_half * 0.5, bond_half * 0.5)
            positions[i + 1] = direction * bl

        if check_constraints(positions, orbital_type, n_lonepairs, has_pi, bond_range, radius):
            return positions

    raise RuntimeError(
        f"Failed to initialize VSEPR ({orbital_type}, lp={n_lonepairs}, pi={has_pi}) "
        f"after {max_attempts} attempts."
    )


def mcmc_vsepr_sample(
    orbital_type: str,
    n_lonepairs: int,
    has_pi: bool,
    bond_range: tuple[float, float],
    radius: float,
    num_samples: int,
    burn_in: int = 5_000,
    thin_interval: int = 500,
    step_size: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """MCMC sampler for VSEPR configurations using Metropolis-Hastings.

    Central atom is fixed at the origin; only ligand atoms are displaced.
    Samples are centered at the molecular centroid before saving.

    Returns:
        positions: (num_samples, n_atoms, 3) float32.
    """
    rng = np.random.default_rng(seed)
    positions = initialize_vsepr(orbital_type, n_lonepairs, has_pi, bond_range, radius, rng)
    n_atoms = len(positions)

    total_steps = burn_in + num_samples * thin_interval
    samples = np.empty((num_samples, n_atoms, 3))
    sample_idx = 0
    accepted = 0
    report_interval = max(1, total_steps // 100)

    t0 = time.time()
    for step in range(total_steps):
        # Pick a random ligand (index 1..n_atoms-1; central atom fixed)
        atom = rng.integers(1, n_atoms)
        old_pos = positions[atom].copy()

        positions[atom] += rng.uniform(-step_size, step_size, size=3)

        if not check_constraints(positions, orbital_type, n_lonepairs, has_pi, bond_range, radius):
            positions[atom] = old_pos
        else:
            accepted += 1

        if step >= burn_in and (step - burn_in) % thin_interval == 0:
            centered = positions - positions.mean(axis=0)
            samples[sample_idx] = centered
            sample_idx += 1

        if (step + 1) % report_interval == 0:
            elapsed = time.time() - t0
            pct = (step + 1) / total_steps * 100
            rate = (step + 1) / elapsed
            acc_rate = accepted / (step + 1)
            print(
                f"\r  {pct:5.1f}% | {sample_idx}/{num_samples} samples | "
                f"accept={acc_rate:.3f} | {rate:.0f} steps/s",
                end="", flush=True,
            )

    print()
    elapsed = time.time() - t0
    print(f"  Done: {num_samples} samples in {elapsed:.1f}s, acceptance rate={accepted/total_steps:.3f}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate VSEPR local geometry samples via MCMC")
    parser.add_argument("--orbital_type", default="sp3", choices=["sp", "sp2", "sp3"],
                        help="Orbital hybridization type (default: sp3)")
    parser.add_argument("--n_lonepairs", type=int, default=0,
                        help="Number of lone pairs — distorts bond angles (default: 0)")
    parser.add_argument("--has_pi", action="store_true",
                        help="Enforce planarity constraint (pi bond)")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of samples to generate")
    parser.add_argument("--burn_in", type=int, default=5_000,
                        help="Burn-in steps (default: 5000)")
    parser.add_argument("--thin_interval", type=int, default=500,
                        help="Thinning interval (default: 500)")
    parser.add_argument("--step_size", type=float, default=0.05,
                        help="MCMC step size in Angstroms (default: 0.05)")
    parser.add_argument("--radius", type=float, default=0.5,
                        help="Atom radius for clash check (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    args = parser.parse_args()

    params = ORBITAL_PARAMS[args.orbital_type]
    bond_range = params["bond_range"]
    n_ligands = max(2, params["n_ligands_max"] - args.n_lonepairs)
    n_atoms = n_ligands + 1
    target_angle = get_target_angle_deg(args.orbital_type, args.n_lonepairs)

    print(
        f"Generating {args.num_samples} VSEPR samples: "
        f"orbital={args.orbital_type}, n_lonepairs={args.n_lonepairs}, "
        f"has_pi={args.has_pi}, n_atoms={n_atoms}"
    )
    print(f"  Target bond angle: {target_angle:.1f}°")
    print(f"  Bond range: {bond_range[0]:.2f}–{bond_range[1]:.2f} Å")

    samples = mcmc_vsepr_sample(
        orbital_type=args.orbital_type,
        n_lonepairs=args.n_lonepairs,
        has_pi=args.has_pi,
        bond_range=bond_range,
        radius=args.radius,
        num_samples=args.num_samples,
        burn_in=args.burn_in,
        thin_interval=args.thin_interval,
        step_size=args.step_size,
        seed=args.seed,
    )

    max_extent = float(np.max(np.abs(samples)))
    box_size = 2.0 * max_extent * 1.5

    print(f"  Synthetic box_size: {box_size:.4f}")
    print(f"  Saving to {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        positions=samples.astype(np.float32),
        orbital_type=np.bytes_(args.orbital_type),
        n_lonepairs=np.int32(args.n_lonepairs),
        has_pi=np.bool_(args.has_pi),
        bond_range=np.array(bond_range, dtype=np.float32),
        target_angle=np.float32(target_angle),
        radius=np.float32(args.radius),
        box_size=np.float32(box_size),
        N=np.int32(n_atoms),
        seed=np.int32(args.seed),
        burn_in=np.int32(args.burn_in),
        thin_interval=np.int32(args.thin_interval),
    )


if __name__ == "__main__":
    main()
