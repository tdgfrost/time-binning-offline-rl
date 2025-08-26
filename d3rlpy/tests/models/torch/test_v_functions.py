import pytest
import torch
import torch.nn.functional as F

from d3rlpy.models.torch.v_functions import (
    ValueFunction,
    compute_v_function_error,
)
from d3rlpy.types import Shape

from ...testing_utils import create_torch_observations
from .model_test import DummyEncoder, check_parameter_updates


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("batch_size", [32])
def test_value_function(observation_shape: Shape, batch_size: int) -> None:
    encoder = DummyEncoder(observation_shape)
    v_func = ValueFunction(encoder, encoder.get_feature_size())

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    y = v_func(x)
    assert y.shape == (batch_size, 1)

    # check compute_error
    returns = torch.rand(batch_size, 1)
    loss = compute_v_function_error(v_func, x, returns)
    assert torch.allclose(loss, F.mse_loss(y, returns))

    # check layer connections
    check_parameter_updates(
        v_func,
        (x,),
    )
