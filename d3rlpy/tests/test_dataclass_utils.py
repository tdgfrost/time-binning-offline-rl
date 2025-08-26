import dataclasses

import torch

from d3rlpy.dataclass_utils import asdict_as_float, asdict_without_copy


@dataclasses.dataclass(frozen=True)
class A:
    a: int


@dataclasses.dataclass(frozen=True)
class D:
    a: A
    b: float
    c: str


def test_asdict_without_any() -> None:
    a = A(1)
    d = D(a, 2.0, "3")
    dict_d = asdict_without_copy(d)
    assert dict_d["a"] is a
    assert dict_d["b"] == 2.0
    assert dict_d["c"] == "3"


@dataclasses.dataclass(frozen=True)
class D2:
    a: float
    b: torch.Tensor


def test_asdict_as_float() -> None:
    b = torch.rand([], dtype=torch.float32)
    d = D2(a=1.0, b=b)
    dict_d = asdict_as_float(d)
    assert dict_d["a"] == 1.0
    assert dict_d["b"] == b.numpy()
