"""Tests for flow matching framework."""

import torch
import pytest

from flow_matching.interpolation import interpolate
from flow_matching.training import flow_matching_loss
from flow_matching.sampling import sample


class TestInterpolation:
    def test_output_shapes(self):
        x_0 = torch.randn(8, 10, 3)
        t = torch.rand(8)
        x_t, noise, velocity_target = interpolate(x_0, t)
        assert x_t.shape == (8, 10, 3)
        assert noise.shape == (8, 10, 3)
        assert velocity_target.shape == (8, 10, 3)

    def test_t_zero_gives_noise(self):
        torch.manual_seed(0)
        x_0 = torch.randn(4, 5, 3)
        t = torch.zeros(4)
        x_t, noise, _ = interpolate(x_0, t)
        # At t=0: x_t = (1-0)*noise + 0*x_0 = noise
        torch.testing.assert_close(x_t, noise)

    def test_t_one_gives_data(self):
        torch.manual_seed(0)
        x_0 = torch.randn(4, 5, 3)
        t = torch.ones(4)
        x_t, _, _ = interpolate(x_0, t)
        # At t=1: x_t = 0*noise + 1*x_0 = x_0
        torch.testing.assert_close(x_t, x_0)

    def test_velocity_target_is_x0_minus_noise(self):
        torch.manual_seed(0)
        x_0 = torch.randn(4, 5, 3)
        t = torch.rand(4)
        _, noise, velocity_target = interpolate(x_0, t)
        torch.testing.assert_close(velocity_target, x_0 - noise)

    def test_interpolation_formula(self):
        torch.manual_seed(0)
        x_0 = torch.randn(4, 5, 3)
        t = torch.tensor([0.3, 0.5, 0.7, 0.9])
        x_t, noise, _ = interpolate(x_0, t)
        expected = (1 - t[:, None, None]) * noise + t[:, None, None] * x_0
        torch.testing.assert_close(x_t, expected)


class TestFlowMatchingLoss:
    def test_returns_scalar(self):
        # Simple dummy model
        model = torch.nn.Linear(3, 3)
        # Wrap to handle (batch, N, 3) + t
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
            def forward(self, x, t):
                return self.linear(x)

        model = DummyModel()
        x_0 = torch.randn(8, 5, 3)
        loss = flow_matching_loss(model, x_0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_is_differentiable(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
            def forward(self, x, t):
                return self.linear(x)

        model = DummyModel()
        x_0 = torch.randn(8, 5, 3)
        loss = flow_matching_loss(model, x_0)
        loss.backward()
        assert model.linear.weight.grad is not None


class TestSampling:
    def test_output_shape(self):
        class DummyModel(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        model = DummyModel()
        result = sample(model, n_atoms=5, n_samples=4, n_steps=10)
        assert result.shape == (4, 5, 3)

    def test_zero_velocity_returns_noise(self):
        """With zero velocity, samples should stay at initial noise."""
        class ZeroModel(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        torch.manual_seed(42)
        model = ZeroModel()
        result = sample(model, n_atoms=5, n_samples=4, n_steps=10)
        # Result should be the initial noise (since v=0 throughout)
        # Can't check exact values since noise is sampled inside sample(),
        # but we can check it's not all zeros (i.e., noise was generated)
        assert result.abs().sum() > 0
