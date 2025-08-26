import math

import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from d3rlpy.models.torch.distributions import (
    GaussianDistribution,
    SquashedGaussianDistribution,
)


@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n", [100])
def test_gaussian_distribution(
    action_size: int, batch_size: int, n: int
) -> None:
    mean = torch.rand(batch_size, action_size)
    std = torch.rand(batch_size, action_size).exp()
    dist = GaussianDistribution(torch.tanh(mean), std, raw_loc=mean)
    ref_dist = Normal(torch.tanh(mean), std)

    y = dist.sample()
    assert y.shape == (batch_size, action_size)

    y, log_prob = dist.sample_with_log_prob()
    ref_log_prob = ref_dist.log_prob(y).sum(dim=1, keepdims=True)
    assert y.shape == (batch_size, action_size)
    assert log_prob.shape == (batch_size, 1)
    assert torch.allclose(log_prob, ref_log_prob)

    y = dist.sample_without_squash()
    assert y.shape == (batch_size, action_size)

    y = dist.sample_n(n)
    assert y.shape == (batch_size, n, action_size)

    y = dist.onnx_safe_sample_n(n)
    assert y.shape == (batch_size, n, action_size)

    y, log_prob = dist.sample_n_with_log_prob(n)
    ref_log_prob = ref_dist.log_prob(y.transpose(0, 1)).sum(
        dim=2, keepdims=True
    )
    assert y.shape == (batch_size, n, action_size)
    assert log_prob.shape == (batch_size, n, 1)
    assert torch.allclose(log_prob, ref_log_prob.transpose(0, 1))

    y = dist.sample_n_without_squash(n)
    assert y.shape == (batch_size, n, action_size)

    assert torch.all(dist.mean == torch.tanh(mean))
    assert torch.all(dist.std == std)


def _ref_squashed_log_prob(dist: Normal, y: torch.Tensor) -> torch.Tensor:
    clipped_y = y.clamp(-0.9999999, 0.9999999)
    raw_y = torch.atanh(clipped_y)
    jacob = 2 * (math.log(2) - raw_y - F.softplus(-2 * raw_y))
    return (dist.log_prob(raw_y) - jacob).sum(dim=-1, keepdims=True)


@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n", [10])
def test_squashed_gaussian_distribution(
    action_size: int, batch_size: int, n: int
) -> None:
    mean = torch.rand(batch_size, action_size)
    std = torch.rand(batch_size, action_size).exp()
    dist = SquashedGaussianDistribution(mean, std)
    ref_dist = Normal(mean, std)

    y = dist.sample()
    assert y.shape == (batch_size, action_size)

    y, log_prob = dist.sample_with_log_prob()
    ref_log_prob = _ref_squashed_log_prob(ref_dist, y)
    assert y.shape == (batch_size, action_size)
    assert log_prob.shape == (batch_size, 1)
    assert torch.allclose(log_prob, ref_log_prob, atol=0.5)

    y = dist.sample_without_squash()
    assert y.shape == (batch_size, action_size)

    y = dist.sample_n(n)
    assert y.shape == (batch_size, n, action_size)

    y = dist.onnx_safe_sample_n(n)
    assert y.shape == (batch_size, n, action_size)

    y, log_prob = dist.sample_n_with_log_prob(n)
    ref_log_prob = _ref_squashed_log_prob(ref_dist, y.transpose(0, 1))
    assert y.shape == (batch_size, n, action_size)
    assert log_prob.shape == (batch_size, n, 1)
    assert torch.allclose(log_prob, ref_log_prob.transpose(0, 1), atol=0.5)

    y = dist.sample_n_without_squash(n)
    assert y.shape == (batch_size, n, action_size)

    assert torch.all(dist.mean == torch.tanh(mean))
    assert torch.all(dist.std == std)
