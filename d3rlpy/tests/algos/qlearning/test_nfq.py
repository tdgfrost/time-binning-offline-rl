from typing import Optional

import pytest

from d3rlpy.algos.qlearning.nfq import NFQConfig
from d3rlpy.models import (
    MeanQFunctionFactory,
    QFunctionFactory,
    QRQFunctionFactory,
)
from d3rlpy.types import Shape

from ...models.torch.model_test import DummyEncoderFactory
from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_nfq(
    observation_shape: Shape,
    n_critics: int,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, _, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = NFQConfig(
        encoder_factory=DummyEncoderFactory(),
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    nfq = config.create()
    algo_tester(
        nfq,  # type: ignore
        observation_shape,
        test_policy_copy=False,
        test_policy_optim_copy=False,
    )
