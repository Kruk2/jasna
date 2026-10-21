from types import SimpleNamespace

import pytest
import torch

from jasna.tensor_utils import pad_batch_with_last

REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_pad_batch_with_last_no_padding() -> None:
    x = torch.randn(4, 3, 8, 8)
    result = pad_batch_with_last(x, batch_size=4)
    assert result is x


def test_pad_batch_with_last_pads_to_batch_size() -> None:
    x = torch.randn(2, 3, 8, 8)
    result = pad_batch_with_last(x, batch_size=4)
    assert result.shape == (4, 3, 8, 8)
    assert torch.equal(result[0], x[0])
    assert torch.equal(result[1], x[1])
    assert torch.equal(result[2], x[1])
    assert torch.equal(result[3], x[1])


def test_pad_batch_with_last_single_frame() -> None:
    x = torch.randn(1, 3, 8, 8)
    result = pad_batch_with_last(x, batch_size=3)
    assert result.shape == (3, 3, 8, 8)
    for i in range(3):
        assert torch.equal(result[i], x[0])


def test_pad_batch_with_last_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        pad_batch_with_last(torch.empty((0, 1)), batch_size=4)


def test_pad_batch_with_last_rejects_oversized_batch() -> None:
    with pytest.raises(ValueError, match="exceeds target batch_size 4"):
        pad_batch_with_last(torch.ones((5, 1)), batch_size=4)


@pytest.mark.parametrize("major, dtype", [(7, torch.float16), (8, torch.bfloat16)])
def test_half_dtype_is_fp16_before_sm80(major, dtype, monkeypatch) -> None:
    from jasna.tensor_utils import half_dtype

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=major))
    assert half_dtype(torch.device("cuda")) == dtype
    assert half_dtype(torch.device("cpu")) == torch.bfloat16


def test_compile_static_runs_bf16_eager_where_inductor_has_no_bf16(monkeypatch) -> None:
    from jasna.tensor_utils import compile_static

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=7))
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: pytest.fail("compiled bf16 before SM 8.0"))

    def double(x):
        return x * 2

    assert compile_static(double, mode="default", device=torch.device("cuda"), dtype=torch.bfloat16) is double


@pytest.mark.parametrize("major, dtype", [(7, torch.float16), (8, torch.bfloat16)])
def test_compile_static_compiles_fp16_everywhere_and_bf16_from_sm80(major, dtype, monkeypatch) -> None:
    from jasna.tensor_utils import compile_static

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=major))
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: calls.append(kwargs) or "compiled")

    assert compile_static(lambda x: x, mode="default", device=torch.device("cuda"), dtype=dtype) == "compiled"
    assert calls == [{"dynamic": False, "mode": "default"}]


def test_compile_static_runs_eager_without_triton(monkeypatch) -> None:
    import torch.utils._triton

    from jasna.tensor_utils import compile_static

    monkeypatch.setattr(torch.utils._triton, "has_triton_package", lambda: False)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: pytest.fail("compiled without Triton"))

    def double(x):
        return x * 2

    assert compile_static(double, mode="default", device=torch.device("cuda"), dtype=torch.float16) is double
