"""Tests for data generation and dataset loading."""

import tempfile
import os

import numpy as np
import pytest
import torch

from data.generate import compute_box_size, initialize_positions, mcmc_sample
from data.dataset import HardSphereDataset


class TestBoxSize:
    def test_known_value(self):
        # eta = N * (4/3) * pi * r^3 / L^3 => L = (N * (4/3) * pi * r^3 / eta)^(1/3)
        L = compute_box_size(N=10, radius=0.5, eta=0.1)
        eta_check = 10 * (4 / 3) * np.pi * 0.5**3 / L**3
        assert abs(eta_check - 0.1) < 1e-10

    def test_higher_eta_smaller_box(self):
        L1 = compute_box_size(N=10, radius=0.5, eta=0.1)
        L2 = compute_box_size(N=10, radius=0.5, eta=0.3)
        assert L2 < L1


class TestInitialization:
    def test_no_overlaps(self):
        rng = np.random.default_rng(42)
        N, radius = 10, 0.5
        box_size = compute_box_size(N, radius, eta=0.1)
        threshold_sq = (2 * radius) ** 2
        positions = initialize_positions(N, box_size, threshold_sq, rng)

        # Check no overlaps
        diff = positions[:, None, :] - positions[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        assert np.all(dist_sq >= threshold_sq)

    def test_within_box(self):
        rng = np.random.default_rng(42)
        N, radius = 10, 0.5
        box_size = compute_box_size(N, radius, eta=0.1)
        threshold_sq = (2 * radius) ** 2
        positions = initialize_positions(N, box_size, threshold_sq, rng)
        assert np.all(positions >= 0)
        assert np.all(positions < box_size)


class TestMCMC:
    def test_generates_correct_shape(self):
        samples, box_size = mcmc_sample(N=5, radius=0.5, eta=0.1, num_samples=10, burn_in=100, thin_interval=10, seed=0)
        assert samples.shape == (10, 5, 3)
        assert box_size > 0

    def test_no_clashes_easy(self):
        samples, box_size = mcmc_sample(N=5, radius=0.5, eta=0.1, num_samples=50, burn_in=500, thin_interval=50, seed=42)
        threshold_sq = (2 * 0.5) ** 2
        for s in range(len(samples)):
            diff = samples[s, :, None, :] - samples[s, None, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            assert np.all(dist_sq >= threshold_sq), f"Clash in sample {s}"

    def test_within_box(self):
        samples, box_size = mcmc_sample(N=5, radius=0.5, eta=0.1, num_samples=20, burn_in=100, thin_interval=10, seed=0)
        assert np.all(samples >= 0)
        assert np.all(samples < box_size)

    def test_deterministic(self):
        s1, _ = mcmc_sample(N=5, radius=0.5, eta=0.1, num_samples=10, burn_in=100, thin_interval=10, seed=123)
        s2, _ = mcmc_sample(N=5, radius=0.5, eta=0.1, num_samples=10, burn_in=100, thin_interval=10, seed=123)
        np.testing.assert_array_equal(s1, s2)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        samples, box_size = mcmc_sample(N=5, radius=0.5, eta=0.1, num_samples=20, burn_in=100, thin_interval=10, seed=0)
        path = str(tmp_path / "test.npz")
        np.savez(path, positions=samples.astype(np.float32), radius=np.float32(0.5), box_size=np.float32(box_size))

        ds = HardSphereDataset(path)
        assert len(ds) == 20
        item = ds[0]
        assert item["positions"].shape == (5, 3)
        assert item["positions"].dtype == torch.float32
        assert isinstance(item["radius"], float)
        assert isinstance(item["box_size"], float)
