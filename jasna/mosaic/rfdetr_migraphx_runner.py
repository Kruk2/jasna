# SPDX-License-Identifier: AGPL-3.0
"""AMD MIGraphX (ROCm) runner for RF-DETR via ONNX Runtime.

Runs the detector through ONNX Runtime's classic MIGraphX execution provider instead of
the torch path. Measured on an RX 7900 XT (benchmarks/2026-10-07_amd_migraphx_ep_windows.md):
a 480 fp16 export is ~1.5-1.9x faster per frame than the fp16 torch path (5.0-6.8 vs
9.63 ms/frame), with identical detection decisions on real frames; the fp32 export is
numerically exact but slower than torch, so this runner exports fp16.

Setup that has to happen before a session can be created (all verified on Windows,
ROCm 10.1 / EP package MicrosoftCorporationII.WinML.AMD.GPU.EP.1.8):

* the ``windowsml`` package (EP catalogue bindings) with its matching ORT build,
  ``windowsml==1.8.2192[with-ort]``; the plugin route of the 2.x rail does not work yet,
* ``onnxruntime_providers_migraphx.dll`` copied next to ORT's own onnxruntime.dll - ORT's
  provider bridge resolves providers from its own capi folder only,
* the EP package's sibling runtimes (migraphx*.dll, amdhip64_7.dll, ...) preloaded by
  full path, because bare-name loads do not find them once ORT narrows the search path,
* no ``HIP_VISIBLE_DEVICES``: the plugin routes on the HIP device it sees, and setting it
  to the discrete card makes it see the integrated GPU instead and fall back to DirectML,
* ``migraphx_model_cache_dir`` for the compiled-program cache; the first session compiles
  the model (~2.5 min per model/shape) and later processes load the cached .mxr.

The ONNX graph itself is exported from the same ``.pt`` checkpoint the torch path uses,
at the registry resolution and the engine batch size, and cached next to the weights.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import shutil
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import torch

from jasna.engine_paths import model_weights_dir
from jasna.mosaic.rfdetr_torch_runner import TorchTensorInfo, _VARIANT_CLASSES

logger = logging.getLogger(__name__)

_ONNX_OUTPUT_NAMES = ("dets", "labels", "masks")
_LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
_ORT_DTYPE = {"tensor(float)": torch.float32, "tensor(float16)": torch.float16}


def migraphx_cache_dir() -> Path:
    return model_weights_dir() / "migraphx-cache"


def migraphx_onnx_path(weights_path: Path, resolution: int, batch_size: int) -> Path:
    """Where the fp16 MIGraphX ONNX export for this model/shape lives."""
    return weights_path.with_name(
        f"{weights_path.stem}.migraphx.r{int(resolution)}.b{int(batch_size)}.fp16.onnx"
    )


def _clean_hip_env() -> None:
    # See module docstring: routing must see the discrete GPU (gfx1100), and any
    # HIP_VISIBLE_DEVICES value reroutes the plugin to the iGPU / DirectML.
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ.pop(name, None)


def _ep_package() -> tuple[Path, Path]:
    """The EP package folder and its classic MIGraphX provider library."""
    try:
        from windowsml import EpCatalog
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "The MIGraphX engine needs the Windows ML EP bindings: "
            "pip install 'windowsml==1.8.2192[with-ort]'"
        ) from exc
    with EpCatalog() as catalog:
        providers = {p.name: p for p in catalog.find_all_providers()}
        ep = providers.get("MIGraphXExecutionProvider")
        if ep is None:
            raise RuntimeError(
                "Windows ML does not offer a MIGraphX execution provider on this machine"
            )
        if int(ep.ready_state) != 0:  # 0 = Ready, 1 = NotReady, 2 = NotPresent
            ep.ensure_ready()
        try:
            library = Path(ep.library_path)
        except OSError:
            # a fresh process can only read the package path once Windows has attached
            # the provider to its dependency graph, which ensure_ready() does
            ep.ensure_ready()
            library = Path(ep.library_path)
    return library.parent, library


def _prepare_ort_for_migraphx() -> None:
    """Copy the provider next to ORT's own DLLs and preload the package runtimes."""
    import onnxruntime as ort

    ep_dir, provider_lib = _ep_package()
    os.add_dll_directory(str(ep_dir))

    capi = Path(ort.__file__).parent / "capi"
    target = capi / provider_lib.name
    if provider_lib.parent != capi and not target.exists():
        shutil.copy2(provider_lib, target)
        logger.info("MIGraphX provider installed to %s", target)

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.LoadLibraryExW.restype = wintypes.HMODULE
    k32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    loaded = 0
    for dll in sorted(glob.glob(str(ep_dir / "*.dll"))):
        if Path(dll).name == provider_lib.name:
            continue
        if k32.LoadLibraryExW(dll, None, _LOAD_WITH_ALTERED_SEARCH_PATH):
            loaded += 1
    logger.debug("MIGraphX: preloaded %d EP package DLLs", loaded)


def export_rfdetr_onnx_fp16(
    weights_path: Path,
    *,
    resolution: int,
    batch_size: int,
    variant: str,
    output_path: Path,
) -> Path:
    """Export the trained checkpoint to a static-shape fp16 ONNX graph.

    Traced on the GPU: half-precision deformable conv / grid_sample have no CPU kernels.
    Takes about a minute; the result is cached next to the weights.
    """
    import rfdetr

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    num_classes = int(state["class_embed.weight"].shape[0]) - 1
    del checkpoint

    cls_name = _VARIANT_CLASSES.get(variant)
    if cls_name is None:
        raise RuntimeError(f"unsupported RF-DETR variant {variant!r}")
    wrapper = getattr(rfdetr, cls_name)(
        num_classes=num_classes,
        resolution=int(resolution),
        pretrain_weights=str(weights_path),
        device="cpu",
    )
    core = wrapper.model.model
    if core is None:
        raise RuntimeError("rfdetr model is unavailable after load")
    core.eval()
    core.export()

    x = torch.randn(
        int(batch_size), 3, int(resolution), int(resolution),
        generator=torch.Generator().manual_seed(0),
    )
    t0 = time.perf_counter()
    torch.onnx.export(
        core.to("cuda").half(), (x.cuda().half(),), str(output_path),
        input_names=["input"], output_names=list(_ONNX_OUTPUT_NAMES),
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    logger.info(
        "RF-DETR MIGraphX export: %s (%.1f MB) in %.1f s",
        output_path, output_path.stat().st_size / 1e6, time.perf_counter() - t0,
    )
    return output_path


class RfDetrMigraphxRunner:
    """Same runner contract as ``RfDetrTorchRunner``, backed by ORT + MIGraphX."""

    def __init__(
        self,
        weights_path: Path,
        batch_size: int,
        resolution: int,
        device: torch.device,
        *,
        variant: str,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "The MIGraphX engine needs ONNX Runtime: "
                "pip install 'windowsml==1.8.2192[with-ort]'"
            ) from exc

        self.device = device
        self.batch_size = int(batch_size)
        self.resolution = int(resolution)
        self.weights_path = Path(weights_path)

        self.onnx_path = migraphx_onnx_path(self.weights_path, resolution, batch_size)
        if not self.onnx_path.is_file():
            logger.info(
                "RF-DETR MIGraphX: exporting fp16 ONNX to %s (one-time, ~1 min)",
                self.onnx_path,
            )
            export_rfdetr_onnx_fp16(
                self.weights_path,
                resolution=self.resolution,
                batch_size=self.batch_size,
                variant=variant,
                output_path=self.onnx_path,
            )

        _clean_hip_env()
        _prepare_ort_for_migraphx()
        cache_dir = migraphx_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=ort.SessionOptions(),
            providers=[
                ("MIGraphXExecutionProvider",
                 {"device_id": "0", "migraphx_model_cache_dir": str(cache_dir)}),
                "CPUExecutionProvider",
            ],
        )
        used = self._session.get_providers()
        if "MIGraphXExecutionProvider" not in used:
            raise RuntimeError(
                f"MIGraphX engine did not engage (session providers: {used}); "
                "the model would run on the CPU"
            )
        logger.info(
            "RF-DETR MIGraphX session ready: %s (providers=%s, setup %.1f s)",
            self.onnx_path.name, used, time.perf_counter() - t0,
        )

        self.input_names = ["input"]
        self.input_dtypes: dict[str, torch.dtype] = {"input": torch.float16}
        self.output_names = list(_ONNX_OUTPUT_NAMES)
        self.outputs: dict[str, TorchTensorInfo] = {}
        for out in self._session.get_outputs():
            shape = tuple(dim if isinstance(dim, int) else -1 for dim in out.shape)
            self.outputs[out.name] = TorchTensorInfo(shape, _ORT_DTYPE.get(out.type, torch.float32))

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = inputs["input"]
        feed = {self.input_names[0]: x.detach().to(torch.float16).cpu().numpy()}
        outs = self._session.run(None, feed)
        return {
            name: torch.from_numpy(np_out.astype(np.float32)).to(self.device)
            for name, np_out in zip(self.output_names, outs)
        }

    def close(self) -> None:
        self._session = None
