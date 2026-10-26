import threading

import pytest
import torch

from jasna import accelerator
from jasna.accelerator import KernelBackend, kernel_backend, load_gpu_libraries_ahead


@pytest.mark.parametrize(("hip", "triton", "capability", "expected"), [
    (None, True, (12, 0), KernelBackend.NVIDIA),
    (None, True, (8, 0), KernelBackend.NVIDIA),
    (None, True, (7, 5), KernelBackend.NVIDIA_PRE_SM80),
    ("7.15", True, (12, 0), KernelBackend.AMD_TRITON),
    ("7.15", False, (12, 0), KernelBackend.AMD_PREBUILT),
])
def test_each_gpu_resolves_to_one_kernel_backend(monkeypatch, hip, triton, capability, expected):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", None if hip else "13.0")
    monkeypatch.setattr(accelerator, "has_triton", lambda: triton)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)

    backend = kernel_backend(torch.device("cuda"))

    assert backend is expected
    assert backend.is_amd == (hip is not None)


def test_the_cpu_has_no_gpu_kernels():
    assert kernel_backend(torch.device("cpu")) is KernelBackend.CPU


def test_gpu_libraries_load_ahead_only_on_a_gpu(monkeypatch):
    monkeypatch.setattr(accelerator.F, "linear", lambda *_args: pytest.fail("no GPU library to load on the CPU"))
    load_gpu_libraries_ahead(torch.device("cpu"), torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gpu_libraries_load_ahead_on_another_thread(dtype):
    worker = threading.Thread(target=load_gpu_libraries_ahead, args=(torch.device("cuda"), dtype))
    worker.start()
    worker.join()
