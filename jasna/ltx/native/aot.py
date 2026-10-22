"""Ahead-of-time cubins / AMD code objects of the ``jasna/ltx/kernels.py`` Triton kernels for the
native block.

Each kernel variant the native block launches is compiled for one GPU architecture the way
``triton.tools.compile`` compiles a kernel (``triton.compile`` of an ``ASTSource`` with
explicit constexprs, divisibility hints and target), so no GPU is needed and a release can
build every architecture of ``kernel_policy.INT8_MM_TILES`` / ``CUBLAS_INT8_MM`` on one
machine; AMD architectures (``kernel_policy.AMD_INT8_MM_TILES``, keyed by gfx name) are compiled with
the hints and options the Triton JIT uses there and need a ROCm Triton. ``triton.tools.compile``
itself emits C launchers that cannot pass the stream-K GEMM's TMA descriptors, so the images go
to a manifest that the native launcher reads instead: kernel name, shared memory, warps and the
expanded argument list.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import expand_signature

from jasna.ltx import kernel_policy as policy
from jasna.ltx import kernels
from jasna.ltx.kernel_policy import BlockDims, is_amd


@dataclass(frozen=True)
class Variant:
    family: str
    kernel: triton.JITFunction
    types: tuple[str, ...]  # one per non-constexpr argument, in order
    divisible: tuple[str, ...]  # arguments promised to be multiples of 16 (pointers: 16-byte aligned)
    constexprs: tuple[tuple[str, object], ...]
    num_warps: int
    num_stages: int
    options: tuple[tuple[str, object], ...]  # extra Triton compile options, as the JIT launch passes them

    @property
    def key(self) -> str:
        return self.family + "".join(f"|{name}={int(value)}" for name, value in self.constexprs)


def _rotate_quantize(k: int, modulate: bool) -> Variant:
    return Variant(
        "rotate_quantize", kernels._rotate_quantize_kernel,
        ("*bf16", "*bf16", "*bf16", "*i8", "*fp32", "i32", "i32"),
        ("x_ptr", "shift_ptr", "scale_ptr", "q_ptr", "s_ptr", "stride_x"),
        (("K", k), ("MODULATE", modulate)), policy.ROTATE_QUANTIZE_WARPS, 3, (),
    )


def _gated_rotate_quantize(k: int, head_dim: int) -> Variant:
    return Variant(
        "gated_rotate_quantize", kernels._gated_rotate_quantize_kernel,
        ("*bf16", "*bf16", "*i8", "*fp32", "i32"), ("o_ptr", "g_ptr", "q_ptr", "s_ptr"),
        (("K", k), ("HEAD_DIM", head_dim)), policy.ROTATE_QUANTIZE_WARPS, 3, (),
    )


_MM_POINTERS = ("as_ptr", "bs_ptr", "bias_ptr", "res_ptr", "gate_ptr", "c_ptr")
_MM_TYPES = ("*fp32", "*fp32", "*bf16", "*bf16", "*bf16", "*bf16")


def _flags(bias: bool, gelu: bool, residual: bool) -> tuple[tuple[str, object], ...]:
    return ("HAS_BIAS", bias), ("GELU", gelu), ("RESIDUAL", residual)


def _mm(tile, bias: bool, gelu: bool, residual: bool) -> Variant:
    bm, bn, bk, stages, warps = tile
    return Variant(
        "mm", kernels._int8_mm_kernel,
        ("*i8", "*i8", *_MM_TYPES, "i32", "i32", "i32", "i32", "i32"),
        ("a_ptr", "b_ptr", *_MM_POINTERS, "M", "N", "K"),
        (*_flags(bias, gelu, residual), ("BM", bm), ("BN", bn), ("BK", bk), ("GROUP_M", policy.GROUP_M)),
        warps, stages, tuple(kernels.GEMM_OPTIONS.items()),
    )


def _mm_stream_k(tile, bias: bool, gelu: bool, residual: bool) -> Variant:
    bm, bn, bk, stages, warps = tile
    return Variant(
        "mm_stream_k", kernels._int8_mm_stream_k_kernel,
        (f"tensordesc<i8[{bm}, {bk}]>", f"tensordesc<i8[{bn}, {bk}]>", *_MM_TYPES, "*i32", "*i32", *("i32",) * 8),
        (*_MM_POINTERS, "ws_ptr", "count_ptr", "M", "N", "K"),
        (*_flags(bias, gelu, residual), ("BM", bm), ("BN", bn), ("BK", bk), ("GROUP_M", policy.GROUP_M)),
        warps, stages, (),
    )


def _epilogue(bias: bool, gelu: bool, residual: bool) -> Variant:
    bm, bn = policy.CUBLAS_EPILOGUE_TILE
    return Variant(
        "epilogue", kernels._int8_epilogue_kernel,
        ("*i32", *_MM_TYPES, "i32", "i32", "i32", "i32"),
        ("acc_ptr", *_MM_POINTERS, "M", "N"),
        (*_flags(bias, gelu, residual), ("BM", bm), ("BN", bn)), policy.CUBLAS_EPILOGUE_WARPS, 3,
        tuple(kernels.GEMM_OPTIONS.items()),
    )


def _quantize_fp4(gelu: bool) -> Variant:
    return Variant(
        "quantize_fp4", kernels._quantize_fp4_kernel,
        ("*bf16", "*fp32", "*u8", "*u8", "i32", "i32", "i32"),
        ("x_ptr", "q_ptr", "s_ptr", "M", "K", "stride_x"),
        (("GELU", gelu), ("BM", policy.FP4_QUANTIZE_ROWS), ("BK", policy.FP4_QUANTIZE_COLS)),
        policy.FP4_QUANTIZE_WARPS, 3, (),
    )


def variants(architecture: tuple[int, int] | str, dims: BlockDims, *, fp4: bool) -> list[Variant]:
    """Every kernel variant an INT8 (``fp4`` False) or NVFP4 block launches on ``architecture``,
    for linears with and without a bias."""
    if fp4:
        return [_quantize_fp4(False), _quantize_fp4(True)]
    found = [_rotate_quantize(dims.dim, True), _rotate_quantize(dims.dim, False), _rotate_quantize(dims.ff_dim, False)]
    if is_amd(architecture):  # the AMD attention epilogue, as transformer._gated_heads runs it there
        found.append(_gated_rotate_quantize(dims.dim, dims.dim // dims.heads))
    projections = (  # (n, k, gelu, residual)
        (dims.dim, dims.dim, False, False),  # Q/K/V, cross-attention Q
        (dims.dim, dims.dim, False, True),  # attention output + gated residual
        (dims.ff_dim, dims.dim, True, False),  # feed-forward up + GELU
        (dims.dim, dims.ff_dim, False, True),  # feed-forward down + gated residual
        (dims.heads, dims.dim, False, False),  # per-head gate logits
    )
    for n, k, gelu, residual in projections:
        for bias in (False, True):
            if architecture in policy.CUBLAS_INT8_MM:
                found.append(_epilogue(bias, gelu, residual))
                continue
            tile = policy.gemm_tile(n, k, architecture)
            found.append(_mm(tile, bias, gelu, residual))
            if policy.has_tma(architecture) and n >= policy.SMALL_MM_MAX_N:
                found.append(_mm_stream_k(tile, bias, gelu, residual))
    return list({variant.key: variant for variant in found}.values())


def _source_digest(architecture: tuple[int, int] | str, chosen: list[Variant]) -> str:
    # each variant's options follow from the architecture and the kernels source
    digest = hashlib.sha256()
    digest.update(triton.__version__.encode())
    digest.update(inspect.getsource(kernels).encode())
    digest.update(repr(architecture).encode())
    for variant in chosen:
        digest.update(repr((variant.key, variant.types, variant.divisible, variant.num_warps, variant.num_stages)).encode())
    return digest.hexdigest()[:16]


def _compile(variant: Variant, architecture: tuple[int, int] | str, directory: Path) -> dict:
    if is_amd(architecture):
        return _compile_amd(variant, architecture, directory)
    names = variant.kernel.arg_names
    constexprs = dict(variant.constexprs)
    runtime = [name for name in names if name not in constexprs]
    signature = dict(zip(runtime, variant.types, strict=True)) | {name: "constexpr" for name in constexprs}
    attrs = {(names.index(name),): [["tt.divisibility", 16]] for name in variant.divisible}
    target = GPUTarget("cuda", architecture[0] * 10 + architecture[1], 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": variant.num_warps, "num_stages": variant.num_stages})
    source = triton.compiler.ASTSource(fn=variant.kernel, signature=signature, constexprs=constexprs, attrs=attrs)
    compiled = triton.compile(source, target=target, options=options.__dict__)
    metadata = compiled.metadata
    _check_scratch(variant, metadata)
    cubin = directory / (hashlib.sha256(variant.key.encode()).hexdigest()[:16] + ".cubin")
    cubin.write_bytes(compiled.asm["cubin"])
    tensordesc = [dict(meta) for meta in metadata.tensordesc_meta or ()]
    args = expand_signature([signature[name] for name in runtime], tensordesc)
    divisible, position = [], 0
    for name, kind in zip(runtime, variant.types):
        width = len(expand_signature([kind], [tensordesc[position]] if kind.startswith("tensordesc") else []))
        position += kind.startswith("tensordesc")
        divisible += [name in variant.divisible] + [False] * (width - 1)
    return _entry(variant, cubin.name, metadata, args, divisible, tensordesc)


def _check_scratch(variant: Variant, metadata) -> None:
    if metadata.global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{variant.key} needs Triton scratch memory, which the native launcher does not provide")


def _compile_amd(variant: Variant, architecture: str, directory: Path) -> dict:
    """An AMD code object specialized as the Triton JIT specializes the same launch: every pointer
    16-byte aligned and inside the 2 GiB range of buffer loads, the variant's options added."""
    if not kernels.HIP.value:
        raise RuntimeError("AMD native-block kernels must be compiled with a ROCm PyTorch and Triton")
    names = variant.kernel.arg_names
    constexprs = dict(variant.constexprs)
    runtime = [name for name in names if name not in constexprs]
    signature = dict(zip(runtime, variant.types, strict=True)) | {name: "constexpr" for name in constexprs}
    pointers = {name for name, kind in zip(runtime, variant.types) if kind.startswith("*")}
    attrs = {(names.index(name),): [["tt.divisibility", 16], ["tt.pointer_range", 32]] for name in pointers}
    attrs |= {(names.index(name),): [["tt.divisibility", 16]] for name in variant.divisible if name not in pointers}
    target = GPUTarget("hip", architecture, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(
        {"num_warps": variant.num_warps, "num_stages": variant.num_stages, **dict(variant.options)}
    )
    source = triton.compiler.ASTSource(fn=variant.kernel, signature=signature, constexprs=constexprs, attrs=attrs)
    compiled = triton.compile(source, target=target, options=options.__dict__)
    metadata = compiled.metadata
    _check_scratch(variant, metadata)
    image = directory / (hashlib.sha256(variant.key.encode()).hexdigest()[:16] + ".hsaco")
    image.write_bytes(compiled.asm["hsaco"])
    divisible = [name in pointers or name in variant.divisible for name in runtime]
    return _entry(variant, image.name, metadata, list(variant.types), divisible, [])


def _entry(variant: Variant, image: str, metadata, args: list[str], divisible: list[bool], tensordesc: list[dict]) -> dict:
    return {
        "family": variant.family,
        "constexprs": {name: int(value) for name, value in variant.constexprs},
        "cubin": image,
        "function": metadata.name,
        "shared": metadata.shared,
        "warps": variant.num_warps,
        "args": args,
        "divisible": divisible,
        "tensordesc": [(m["swizzle"], m["elem_size"], m["elem_type"], list(m["block_size"])) for m in tensordesc],
    }


def build_kernels(architecture: tuple[int, int] | str, dims: BlockDims, *, fp4: bool, cache: Path) -> Path:
    """Compile the block's kernels for ``architecture`` (``kernel_policy.gpu_architecture``) into
    ``cache/<digest>/`` unless done already; returns the manifest path."""
    chosen = variants(architecture, dims, fp4=fp4)
    name = f"{policy.profile_name(architecture, fp4=fp4)}-{_source_digest(architecture, chosen)}"
    directory = cache / name
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        entries = [_compile(variant, architecture, directory) for variant in chosen]
        partial = manifest.with_suffix(".part")
        partial.write_text(json.dumps(entries))
        partial.replace(manifest)
    return manifest
