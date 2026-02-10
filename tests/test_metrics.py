"""Tests for clash rate metrics."""

import torch
import pytest

from metrics.clash_rate import has_clash, clash_rate, clash_rate_batched


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
