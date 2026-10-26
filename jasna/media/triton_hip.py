"""Triton kernels compiled ahead of time into AMD code objects, so that GPUs without Triton (Windows
ROCm, AMD releases) launch them through ``cuda_kernel``. Build-time only: needs a ROCm Triton."""

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

WARP_SIZE = 32


def compile_hip(
    kernel: triton.JITFunction, arguments: dict[str, str], constexprs: dict[str, object], options: dict, architecture: str
):
    """``kernel`` for ``architecture`` with the hints the Triton JIT gives the same launch on AMD.

    ``arguments`` maps each runtime argument to its Triton type: every tensor is 16-byte aligned
    and inside the 2 GB range of AMD buffer loads unless typed ``"*<dtype>:wide"``; ``"i32:16"`` is
    an int the caller guarantees divisible by 16. ``options`` are Triton compile options
    (``num_warps``, ``num_stages``, ...).
    """
    names = kernel.arg_names
    signature = {name: kind.split(":")[0] for name, kind in arguments.items()}
    signature.update({name: "constexpr" for name in constexprs})
    attrs = {}
    for name, kind in arguments.items():
        if kind.startswith("*"):
            attrs[(names.index(name),)] = [["tt.divisibility", 16]] + ([] if kind.endswith(":wide") else [["tt.pointer_range", 32]])
        elif kind.endswith(":16"):
            attrs[(names.index(name),)] = [["tt.divisibility", 16]]
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs, attrs),
        target=GPUTarget("hip", architecture, WARP_SIZE),
        options=options,
    )
    metadata = compiled.metadata
    if metadata.global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{metadata.name} on {architecture} needs Triton scratch memory, which no launcher passes")
    return compiled
