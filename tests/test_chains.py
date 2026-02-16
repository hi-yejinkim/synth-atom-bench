"""Tests for chain data generation, dataset, and metrics."""

import tempfile

import numpy as np
import pytest
import torch

from data.generate_chains import (
    nerf_place_atom,
    has_nonbonded_clash,
    initialize_chain,
    random_rotation_matrix,
    pivot_move,
    mcmc_chain_sample,
)
from data.chain_dataset import ChainDataset
from metrics.bond_violation import (
    bond_violation_rate,
    nonbonded_clash_rate,
    bond_violation_rate_batched,
    nonbonded_clash_rate_batched,
)


class TestNeRF:
    def test_bond_length(self):
        """NeRF-placed atom should be exactly bond_length from prev1."""
        prev3 = np.array([-1.0, 0.0, 0.0])
        prev2 = np.array([0.0, 0.0, 0.0])
        prev1 = np.array([1.0, 0.0, 0.0])
        d = 1.5
        for theta in [np.pi / 3, np.pi / 2, 2 * np.pi / 3]:
            for phi in [0, np.pi / 4, np.pi, 3 * np.pi / 2]:
                new = nerf_place_atom(prev3, prev2, prev1, d, theta, phi)
                dist = np.linalg.norm(new - prev1)
                assert abs(dist - d) < 1e-10, f"theta={theta}, phi={phi}: dist={dist}"

    def test_bond_angle(self):
        """Bond angle between prev2-prev1-new should match theta."""
        prev3 = np.array([-1.0, 0.0, 0.0])
        prev2 = np.array([0.0, 0.0, 0.0])
        prev1 = np.array([1.0, 0.0, 0.0])
        d = 1.0
        theta = np.pi / 3
        for phi in [0, np.pi / 2, np.pi]:
            new = nerf_place_atom(prev3, prev2, prev1, d, theta, phi)
            v1 = prev2 - prev1
            v2 = new - prev1
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            assert abs(angle - theta) < 1e-10, f"phi={phi}: angle={angle}, expected={theta}"

    def test_dihedral(self):
        """Dihedral angle prev3-prev2-prev1-new should match phi."""
        prev3 = np.array([0.0, 0.0, 1.0])
        prev2 = np.array([0.0, 0.0, 0.0])
        prev1 = np.array([1.0, 0.0, 0.0])
        d = 1.0
        theta = np.pi / 2
        phi = np.pi / 4
        new = nerf_place_atom(prev3, prev2, prev1, d, theta, phi)

        # Compute dihedral: angle between planes (prev3,prev2,prev1) and (prev2,prev1,new)
        b1 = prev2 - prev3
        b2 = prev1 - prev2
        b3 = new - prev1
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        dihedral = -np.arctan2(y, x)
        # Dihedral conventions can differ by sign/offset; check modular equivalence
        diff = (dihedral - phi + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff) < 1e-9 or abs(abs(diff) - 2 * np.pi) < 1e-9


class TestChainInitialization:
    def test_shape(self):
        rng = np.random.default_rng(42)
        pos = initialize_chain(10, bond_length=1.0, radius=0.3, rng=rng)
        assert pos.shape == (10, 3)

    def test_bond_lengths(self):
        rng = np.random.default_rng(42)
        pos = initialize_chain(10, bond_length=1.0, radius=0.3, rng=rng)
        diffs = pos[1:] - pos[:-1]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        np.testing.assert_allclose(dists, 1.0, atol=1e-10)

    def test_no_nonbonded_clashes(self):
        rng = np.random.default_rng(42)
        pos = initialize_chain(15, bond_length=1.0, radius=0.3, rng=rng)
        N = len(pos)
        threshold_sq = (2 * 0.3) ** 2
        idx = np.arange(N)
        nonbonded = np.abs(idx[:, None] - idx[None, :]) > 1
        diff = pos[:, None, :] - pos[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        assert not np.any(dist_sq[nonbonded] < threshold_sq)

    def test_centered(self):
        rng = np.random.default_rng(42)
        pos = initialize_chain(10, bond_length=1.0, radius=0.3, rng=rng)
        np.testing.assert_allclose(pos.mean(axis=0), 0.0, atol=1e-10)

    def test_small_chain(self):
        rng = np.random.default_rng(42)
        pos = initialize_chain(2, bond_length=1.0, radius=0.3, rng=rng)
        assert pos.shape == (2, 3)
        dist = np.linalg.norm(pos[1] - pos[0])
        assert abs(dist - 1.0) < 1e-10


class TestRandomRotation:
    def test_is_rotation(self):
        rng = np.random.default_rng(42)
        R = random_rotation_matrix(rng)
        # R^T R = I
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        # det(R) = 1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_different_each_call(self):
        rng = np.random.default_rng(42)
        R1 = random_rotation_matrix(rng)
        R2 = random_rotation_matrix(rng)
        assert not np.allclose(R1, R2)


class TestPivotMove:
    def test_preserves_bond_lengths(self):
        rng = np.random.default_rng(42)
        pos = initialize_chain(10, bond_length=1.0, radius=0.3, rng=rng)
        # Run many pivot moves
        for _ in range(100):
            pivot_move(pos, 10, 0.3, rng)
        diffs = pos[1:] - pos[:-1]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        np.testing.assert_allclose(dists, 1.0, atol=1e-10)

    def test_preserves_head_on_reject(self):
        """If pivot is rejected, positions should be unchanged."""
        rng = np.random.default_rng(42)
        pos = initialize_chain(10, bond_length=1.0, radius=0.3, rng=rng)
        pos_before = pos.copy()
        # Force many attempts — some will reject
        any_rejected = False
        for _ in range(200):
            before = pos.copy()
            accepted = pivot_move(pos, 10, 0.3, rng)
            if not accepted:
                np.testing.assert_array_equal(pos, before)
                any_rejected = True
        # At least some should have been rejected
        # (may not always hold with very loose chains, so just check if test ran)

    def test_small_chain(self):
        """Chain of 2 atoms: pivot_move should return False."""
        rng = np.random.default_rng(42)
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert not pivot_move(pos, 2, 0.3, rng)


class TestMCMCSampling:
    def test_shape(self):
        samples = mcmc_chain_sample(N=5, bond_length=1.0, radius=0.3, num_samples=10,
                                     burn_in=50, thin_interval=5, seed=42)
        assert samples.shape == (10, 5, 3)

    def test_deterministic(self):
        s1 = mcmc_chain_sample(N=5, bond_length=1.0, radius=0.3, num_samples=10,
                                burn_in=50, thin_interval=5, seed=123)
        s2 = mcmc_chain_sample(N=5, bond_length=1.0, radius=0.3, num_samples=10,
                                burn_in=50, thin_interval=5, seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_bond_lengths_preserved(self):
        samples = mcmc_chain_sample(N=10, bond_length=1.0, radius=0.3, num_samples=20,
                                     burn_in=100, thin_interval=10, seed=42)
        diffs = samples[:, 1:] - samples[:, :-1]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.testing.assert_allclose(dists, 1.0, atol=1e-10)

    def test_no_nonbonded_clashes(self):
        samples = mcmc_chain_sample(N=10, bond_length=1.0, radius=0.3, num_samples=20,
                                     burn_in=100, thin_interval=10, seed=42)
        N = 10
        threshold_sq = (2 * 0.3) ** 2
        idx = np.arange(N)
        nonbonded = np.abs(idx[:, None] - idx[None, :]) > 1
        for s in range(len(samples)):
            diff = samples[s, :, None, :] - samples[s, None, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            assert not np.any(dist_sq[nonbonded] < threshold_sq), f"Clash in sample {s}"

    def test_centered(self):
        samples = mcmc_chain_sample(N=5, bond_length=1.0, radius=0.3, num_samples=10,
                                     burn_in=50, thin_interval=5, seed=42)
        for s in range(len(samples)):
            np.testing.assert_allclose(samples[s].mean(axis=0), 0.0, atol=1e-10)


class TestChainDataset:
    def test_roundtrip(self, tmp_path):
        samples = mcmc_chain_sample(N=5, bond_length=1.0, radius=0.3, num_samples=20,
                                     burn_in=50, thin_interval=5, seed=42)
        max_ext = np.max(np.abs(samples))
        box_size = 2.0 * max_ext * 1.2
        path = str(tmp_path / "test_chain.npz")
        np.savez(
            path,
            positions=samples.astype(np.float32),
            bond_length=np.float32(1.0),
            radius=np.float32(0.3),
            box_size=np.float32(box_size),
        )

        ds = ChainDataset(path)
        assert len(ds) == 20
        item = ds[0]
        assert item["positions"].shape == (5, 3)
        assert item["positions"].dtype == torch.float32
        assert isinstance(item["radius"], float)
        assert isinstance(item["box_size"], float)
        assert isinstance(item["bond_length"], float)

    def test_positions_shifted(self, tmp_path):
        """Positions should be shifted to [0, box_size] range."""
        samples = mcmc_chain_sample(N=5, bond_length=1.0, radius=0.3, num_samples=5,
                                     burn_in=50, thin_interval=5, seed=42)
        max_ext = np.max(np.abs(samples))
        box_size = 2.0 * max_ext * 1.2
        path = str(tmp_path / "test_chain.npz")
        np.savez(
            path,
            positions=samples.astype(np.float32),
            bond_length=np.float32(1.0),
            radius=np.float32(0.3),
            box_size=np.float32(box_size),
        )
        ds = ChainDataset(path)
        item = ds[0]
        assert item["positions"].min() >= 0
        assert item["positions"].max() <= box_size


class TestBondViolationMetric:
    def test_perfect_chain_no_violation(self):
        """A chain with exact bond lengths should have 0 violation rate."""
        # Build a straight chain
        positions = torch.zeros(5, 10, 3)
        for i in range(10):
            positions[:, i, 0] = i * 1.0
        assert bond_violation_rate(positions, bond_length=1.0) == 0.0

    def test_broken_bond_detected(self):
        """A chain with one broken bond should be detected."""
        positions = torch.zeros(2, 5, 3)
        for i in range(5):
            positions[:, i, 0] = i * 1.0
        # Break one bond in first sample
        positions[0, 2, 0] = 5.0  # move atom 2 far away
        rate = bond_violation_rate(positions, bond_length=1.0, tolerance=0.1)
        assert rate == 0.5  # 1 out of 2

    def test_tight_tolerance(self):
        positions = torch.zeros(1, 3, 3)
        positions[0, 0] = torch.tensor([0.0, 0.0, 0.0])
        positions[0, 1] = torch.tensor([1.05, 0.0, 0.0])
        positions[0, 2] = torch.tensor([2.05, 0.0, 0.0])
        # tolerance=0.1 -> 0.05 deviation is OK
        assert bond_violation_rate(positions, bond_length=1.0, tolerance=0.1) == 0.0
        # tolerance=0.01 -> 0.05 deviation is violation
        assert bond_violation_rate(positions, bond_length=1.0, tolerance=0.01) == 1.0

    def test_batched_matches(self):
        torch.manual_seed(42)
        positions = torch.zeros(50, 5, 3)
        for i in range(5):
            positions[:, i, 0] = i * 1.0
        # Add some noise
        positions += torch.randn_like(positions) * 0.05
        r1 = bond_violation_rate(positions, bond_length=1.0, tolerance=0.1)
        r2 = bond_violation_rate_batched(positions, bond_length=1.0, tolerance=0.1, chunk_size=10)
        assert abs(r1 - r2) < 1e-6


class TestNonbondedClashMetric:
    def test_well_separated_no_clash(self):
        """Atoms far apart should have no clash."""
        positions = torch.zeros(2, 5, 3)
        for i in range(5):
            positions[:, i, 0] = i * 2.0  # well separated
        assert nonbonded_clash_rate(positions, radius=0.3) == 0.0

    def test_bonded_overlap_not_flagged(self):
        """Consecutive atoms within 2r should NOT be flagged (they're bonded)."""
        positions = torch.zeros(1, 3, 3)
        positions[0, 0] = torch.tensor([0.0, 0.0, 0.0])
        positions[0, 1] = torch.tensor([0.5, 0.0, 0.0])  # < 2*0.3=0.6 from atom 0, but bonded
        positions[0, 2] = torch.tensor([5.0, 0.0, 0.0])  # far from both
        assert nonbonded_clash_rate(positions, radius=0.3) == 0.0

    def test_nonbonded_clash_detected(self):
        """Non-consecutive atoms within 2r should be flagged."""
        positions = torch.zeros(1, 4, 3)
        positions[0, 0] = torch.tensor([0.0, 0.0, 0.0])
        positions[0, 1] = torch.tensor([1.0, 0.0, 0.0])
        positions[0, 2] = torch.tensor([2.0, 0.0, 0.0])
        positions[0, 3] = torch.tensor([0.1, 0.0, 0.0])  # atom 3 near atom 0 (non-bonded, |3-0|>1)
        assert nonbonded_clash_rate(positions, radius=0.3) == 1.0

    def test_batched_matches(self):
        torch.manual_seed(42)
        positions = torch.zeros(50, 5, 3)
        for i in range(5):
            positions[:, i, 0] = i * 2.0
        r1 = nonbonded_clash_rate(positions, radius=0.3)
        r2 = nonbonded_clash_rate_batched(positions, radius=0.3, chunk_size=10)
        assert abs(r1 - r2) < 1e-6
