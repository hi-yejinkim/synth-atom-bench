"""Tests for clash rate and g(r) distance metrics."""

import numpy as np
import torch
import pytest

from metrics.clash_rate import has_clash, clash_rate, clash_rate_batched
from metrics.gr_distance import gr_distance
from data.validate import pair_correlation


class TestHasClash:
    def test_no_clash(self):
        # Two atoms far apart
        positions = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
        assert not has_clash(positions, radius=0.5).item()

    def test_has_clash(self):
        # Two atoms overlapping
        positions = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
        assert has_clash(positions, radius=0.5).item()

    def test_exactly_touching(self):
        # Distance = 2*radius exactly — not a clash (< not <=)
        positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        assert not has_clash(positions, radius=0.5).item()

    def test_batch(self):
        positions = torch.tensor([
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],  # no clash
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # clash
        ])
        result = has_clash(positions, radius=0.5)
        assert result.shape == (2,)
        assert not result[0].item()
        assert result[1].item()


class TestClashRate:
    def test_all_clean(self):
        positions = torch.tensor([
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        ])
        assert clash_rate(positions, radius=0.5) == 0.0

    def test_all_clashing(self):
        positions = torch.tensor([
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
        ])
        assert clash_rate(positions, radius=0.5) == 1.0

    def test_half_clashing(self):
        positions = torch.tensor([
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],  # clean
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],  # clash
        ])
        assert clash_rate(positions, radius=0.5) == 0.5


class TestClashRateBatched:
    def test_matches_unbatched(self):
        torch.manual_seed(42)
        positions = torch.randn(100, 3, 3) * 5
        r1 = clash_rate(positions, radius=0.5)
        r2 = clash_rate_batched(positions, radius=0.5, chunk_size=10)
        assert abs(r1 - r2) < 1e-6


class TestGrDistance:
    def _random_uniform_positions(self, n_samples, n_atoms, box_size, seed=42):
        rng = np.random.default_rng(seed)
        return rng.uniform(0, box_size, size=(n_samples, n_atoms, 3))

    def test_identical_distributions(self):
        """Same data for generated and ground truth should give distance ~0."""
        box_size = 5.0
        positions = self._random_uniform_positions(200, 10, box_size)
        gt_r, gt_g_r = pair_correlation(positions, box_size)
        dist = gr_distance(positions, gt_r, gt_g_r, box_size)
        assert dist < 0.05, f"Expected ~0 distance for identical data, got {dist}"

    def test_different_distributions(self):
        """Clustered positions should have different g(r) than uniform."""
        box_size = 5.0
        # Ground truth: uniform
        gt_positions = self._random_uniform_positions(200, 10, box_size, seed=0)
        gt_r, gt_g_r = pair_correlation(gt_positions, box_size)
        # Generated: clustered (all atoms near origin)
        rng = np.random.default_rng(99)
        gen_positions = rng.normal(loc=box_size / 2, scale=0.3, size=(200, 10, 3))
        gen_positions = np.clip(gen_positions, 0, box_size)
        dist = gr_distance(gen_positions, gt_r, gt_g_r, box_size)
        assert dist > 0.1, f"Expected large distance for different distributions, got {dist}"

    def test_lower_is_better(self):
        """More similar distributions should have lower distance."""
        box_size = 5.0
        gt_positions = self._random_uniform_positions(500, 10, box_size, seed=0)
        gt_r, gt_g_r = pair_correlation(gt_positions, box_size)
        # Similar: another uniform sample
        similar = self._random_uniform_positions(500, 10, box_size, seed=1)
        dist_similar = gr_distance(similar, gt_r, gt_g_r, box_size)
        # Dissimilar: clustered
        rng = np.random.default_rng(99)
        dissimilar = rng.normal(loc=box_size / 2, scale=0.3, size=(500, 10, 3))
        dissimilar = np.clip(dissimilar, 0, box_size)
        dist_dissimilar = gr_distance(dissimilar, gt_r, gt_g_r, box_size)
        assert dist_similar < dist_dissimilar
