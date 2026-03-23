import pytest
import torch

from jasna.tensor_utils import to_device, pad_batch_with_last

REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_to_device_same_device_returns_same_tensor() -> None:
    t = torch.randn(3, 4)
    result = to_device(t, torch.device("cpu"))
    assert result is t


@REQUIRES_CUDA
def test_to_device_cpu_to_cuda_contiguous() -> None:
    src = torch.randn(3, 4)
    result = to_device(src, torch.device("cuda:0"))
    assert result.device.type == "cuda"
    assert result.is_contiguous()
    assert torch.equal(result.cpu(), src)


@REQUIRES_CUDA
def test_to_device_cpu_to_cuda_non_contiguous() -> None:
    src = torch.randn(4, 4).t()
    assert not src.is_contiguous()
    result = to_device(src, torch.device("cuda:0"))
    assert result.device.type == "cuda"
    assert result.is_contiguous()
    assert torch.equal(result.cpu(), src)


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
