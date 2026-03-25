from __future__ import annotations

import logging
import os
import warnings

import torch

logger = logging.getLogger(__name__)


from jasna.engine_paths import engine_precision_name, engine_system_suffix  # noqa: E402, F401


def get_workspace_size_bytes() -> int:
    free, _total = torch.cuda.mem_get_info()
    return int(free * 0.95)


def load_torchtrt_export(*, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
    import torch_tensorrt  # noqa: F401

    logger.debug("Loading TensorRT export from %s", checkpoint_path)
    fake_reg_logger = logging.getLogger("torch._library.fake_class_registry")
    prev_level = fake_reg_logger.level
    fake_reg_logger.setLevel(logging.ERROR)
    try:
        export_logger = logging.getLogger("torch.export")
        prev_export_level = export_logger.level
        export_logger.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*PytorchStreamReader.*")
                try:
                    with open(checkpoint_path, "rb") as f:
                        trt_module = torch.export.load(f).module()
                except Exception:
                    trt_module = torch.load(checkpoint_path, map_location=device, weights_only=False)
        finally:
            export_logger.setLevel(prev_export_level)
        return trt_module.to(device)
    finally:
        fake_reg_logger.setLevel(prev_level)


def compile_and_save_torchtrt_dynamo(
    *,
    module: torch.nn.Module,
    inputs: list,
    output_path: str,
    dtype: torch.dtype,
    workspace_size_bytes: int,
    message: str,
    device: torch.device | None = None,
    optimization_level: int = 3,
) -> str:
    """Compile a module to TensorRT and save the result.

    ``inputs`` may be plain tensors (static shapes) or
    ``torch_tensorrt.Input`` specs (dynamic shapes).  Pass ``device``
    explicitly when using ``torch_tensorrt.Input`` objects.
    """
    import torch_tensorrt  # type: ignore[import-not-found]

    has_dynamic = any(isinstance(inp, torch_tensorrt.Input) for inp in inputs)
    if device is None:
        device = inputs[0].device

    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
    with torch_tensorrt.logging.errors():
        logger.info("%s", message)
        with torch.cuda.device(device):
            trt_gm = torch_tensorrt.compile(
                module,
                ir="dynamo",
                inputs=inputs,
                min_block_size=1,
                workspace_size=int(workspace_size_bytes),
                enabled_precisions={dtype},
                use_fp32_acc=False,
                use_explicit_typing=False,
                sparse_weights=False,
                optimization_level=int(optimization_level),
                hardware_compatible=False,
                use_python_runtime=False,
                cache_built_engines=False,
                reuse_cached_engines=False,
                truncate_double=True,
            )
            fake_reg_logger = logging.getLogger("torch._library.fake_class_registry")
            prev_level = fake_reg_logger.level
            fake_reg_logger.setLevel(logging.ERROR)
            try:
                if has_dynamic:
                    _save_with_dynamic_shapes(trt_gm, output_path, inputs, device, dtype)
                else:
                    torch_tensorrt.save(trt_gm, output_path, inputs=inputs)
            finally:
                fake_reg_logger.setLevel(prev_level)
    del trt_gm
    return output_path


def _save_with_dynamic_shapes(
    trt_gm: torch.nn.Module,
    output_path: str,
    inputs: list,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Save a TRT-compiled GraphModule that uses dynamic shapes.

    Builds sample tensors and a ``dynamic_shapes`` spec from
    ``torch_tensorrt.Input`` objects so that ``torch.export.export``
    records the correct symbolic dimension constraints.
    """
    import torch_tensorrt  # type: ignore[import-not-found]
    from torch.export import Dim

    sample_args: list[torch.Tensor] = []
    dyn_shapes: list[dict[int, Dim] | None] = []

    for inp in inputs:
        if isinstance(inp, torch_tensorrt.Input):
            shape_dict = inp.shape
            opt = shape_dict["opt_shape"]
            sample_args.append(torch.randn(*opt, dtype=dtype, device=device))

            min_s, max_s = shape_dict["min_shape"], shape_dict["max_shape"]
            dim_map: dict[int, Dim] = {}
            for d in range(len(opt)):
                if min_s[d] != max_s[d]:
                    dim_map[d] = Dim(f"d{d}", min=int(min_s[d]), max=int(max_s[d]))
            dyn_shapes.append(dim_map if dim_map else None)
        else:
            sample_args.append(inp)
            dyn_shapes.append(None)

    try:
        ep = torch.export.export(
            trt_gm,
            tuple(sample_args),
            dynamic_shapes=tuple(dyn_shapes),
            strict=False,
        )
        torch.export.save(ep, output_path)
    except RuntimeError:
        logger.debug(
            "torch.export.export failed (multi-subgraph dynamic shapes); "
            "falling back to torch.save for %s", output_path,
        )
        torch.save(trt_gm, output_path)

