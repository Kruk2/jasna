"""The LTX VAE encoder on AMD GPUs without Triton (``KernelBackend.AMD_PREBUILT``: every AMD release): the
convs on comfy-kitchen (fp16 head) and MIOpen (channels-last), everything between them in the ahead-of-time
``encoder_triton`` kernels (``scripts/build_ltx_encoder_hipfbs.py``), as the compiled encoder fuses them.

``norm_pad``: a conv's input, from a layer output plus a pending residual add, optionally PixelNorm + SiLU,
written into a buffer with the causal leading frames (and the spatial zero border kitchen needs).
``downsample_add``: ``SpaceToDepthDownsample``'s space-to-depth of its conv plus the group-mean skip.
"""

import functools

import torch
import torch.nn.functional as F

from jasna.media.cuda_kernel import aot_kernel
from jasna.models.ltx_vae.encoder_hip_launch import KERNEL_LAUNCH
from jasna.models.ltx_vae.ops import patchify
from jasna.models.ltx_vae.video_encoder import (
    CausalConv3d,
    ResnetBlock3D,
    SpaceToDepthDownsample,
    UNetMidBlock3D,
    VideoEncoder,
    _release_cache,
)

WARPS = 4
ELEMENTS_PER_PROGRAM = 4096
_DTYPES = {torch.float16: "fp16", torch.bfloat16: "bf16"}


NORM_PAD_BUILDS = (  # input and output dtype, channel block, PixelNorm + SiLU, residual add
    ("fp16", "fp16", 64, False, False),  # conv_in's patchified pixels
    ("fp16", "fp16", 128, True, False),  # the fp16 head's residual blocks (kitchen adds their residual)
    ("fp16", "bf16", 128, False, False),  # the head's output into the first bf16 downsample
    *(("bf16", "bf16", channels, norm, add) for channels in (256, 512, 1024)
      for norm, add in ((True, False), (True, True), (False, True))),
)
DOWNSAMPLE_WIDTHS = (256, 512, 1024)


def block_rows(channels: int) -> int:
    return max(1, ELEMENTS_PER_PROGRAM // channels)


def norm_pad_module(source: str, target: str, block_c: int, norm: bool, add: bool) -> str:
    return f"ltx_encoder_{'norm' if norm else 'pad'}{'_add' if add else ''}_{source}_{target}_{block_c}"


def norm_pad_arguments(source: str, target: str) -> dict[str, str]:
    return {"x_ptr": f"*{source}:wide", "bias_ptr": f"*{source}", "residual_ptr": f"*{target}:wide",
            "sum_ptr": f"*{target}:wide", "out_ptr": f"*{target}:wide",
            **{name: "i32" for name in ("rows", "T", "H", "W", "C", "lead", "pad", "stride_n", "stride_t",
                                        "stride_h", "stride_w", "stride_c")}}


def downsample_module(channels: int) -> str:
    return f"ltx_encoder_downsample_{channels}"


DOWNSAMPLE_ARGUMENTS = {"y_ptr": "*bf16:wide", "bias_ptr": "*bf16", "x_ptr": "*bf16:wide", "out_ptr": "*bf16:wide",
                        **{name: "i32" for name in ("rows", "T", "H", "W", "stride_t", "stride_h", "stride_w", "group",
                                                    "y_stride_n", "y_stride_t", "y_stride_h", "y_stride_w",
                                                    "x_stride_n", "x_stride_t", "x_stride_h", "x_stride_w")}}


@functools.cache
def _zeros(channels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.zeros(channels, dtype=dtype, device=device)


def _block_c(channels: int) -> int:
    return 1 << (channels - 1).bit_length()


def _channels_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last_3d)


def norm_pad(
    x: torch.Tensor, dtype: torch.dtype, *, lead: int, pad: int, norm: bool, bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``x`` (+ ``bias`` per channel, + ``residual``), through PixelNorm + SiLU when ``norm``, as a channels-last
    ``dtype`` conv input ``(N, C, lead + T, H + 2 pad, W + 2 pad)``: frame 0 repeated ``lead`` times ahead, a zero
    border of ``pad``. With both ``norm`` and ``residual`` also returns the sum (the next residual)."""
    if residual is not None and (residual.dtype != dtype or residual.shape != x.shape):
        raise ValueError(f"a residual must match its layer output's shape and the target dtype, got {residual.dtype}")
    n, channels, frames, height, width = x.shape
    module = norm_pad_module(_DTYPES[x.dtype], _DTYPES[dtype], _block_c(channels), norm, residual is not None)
    kernel = aot_kernel(KERNEL_LAUNCH, module, x.device, norm_pad_arguments(_DTYPES[x.dtype], _DTYPES[dtype]))
    out = torch.empty((n, channels, lead + frames, height + 2 * pad, width + 2 * pad), dtype=dtype, device=x.device,
                      memory_format=torch.channels_last_3d)
    if pad:
        for border in (out[..., :pad, :], out[..., -pad:, :], out[..., :pad], out[..., -pad:]):
            border.zero_()
    total = None
    if residual is not None:
        residual = _channels_last(residual)
        if norm:
            total = torch.empty_like(residual)
    rows = n * frames * height * width
    stride_n, stride_c, stride_t, stride_h, stride_w = x.stride()
    bias = _zeros(channels, x.dtype, x.device) if bias is None else bias
    kernel.launch((-(-rows // block_rows(_block_c(channels))), 1, 1), torch.cuda.current_stream(x.device).cuda_stream,
                  x, bias, out if residual is None else residual, out if total is None else total, out,
                  rows, frames, height, width, channels, lead, pad, stride_n, stride_t, stride_h, stride_w, stride_c)
    return out, total


def downsample_add(
    y: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, block: SpaceToDepthDownsample,
) -> torch.Tensor:
    """``space_to_depth(y + bias) + group mean of space_to_depth(x)`` as ``block`` computes them,
    channels-last."""
    y, x = _channels_last(y), _channels_last(x) if x.stride(1) != 1 else x
    n, _, frames, height, width = y.shape
    stride_t, stride_h, stride_w = block.stride
    channels = y.shape[1] * stride_t * stride_h * stride_w
    out_shape = (n, channels, frames // stride_t, height // stride_h, width // stride_w)
    out = torch.empty(out_shape, dtype=y.dtype, device=y.device, memory_format=torch.channels_last_3d)
    kernel = aot_kernel(KERNEL_LAUNCH, downsample_module(channels), y.device, DOWNSAMPLE_ARGUMENTS)
    rows = out_shape[0] * out_shape[2] * out_shape[3] * out_shape[4]
    y_strides, x_strides = y.stride(), x.stride()
    kernel.launch((-(-rows // block_rows(channels)), 1, 1), torch.cuda.current_stream(y.device).cuda_stream,
                  y, bias, x, out, rows, *out_shape[2:], stride_t, stride_h, stride_w, block.group_size,
                  y_strides[0], *y_strides[2:], x_strides[0], *x_strides[2:])
    return out


def _kitchen(conv: CausalConv3d, padded: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
    return torch.ops.comfy_kitchen.fp16_conv3d(padded, conv.conv.weight, conv.conv.bias, residual, [1, 1, 1])


def _conv(conv: CausalConv3d, padded: torch.Tensor) -> torch.Tensor:
    """The conv on MIOpen without its bias, which the next kernel adds."""
    return F.conv3d(padded, conv.conv.weight, None, padding=conv.conv.padding)


def _causal_lead(conv: CausalConv3d) -> int:
    return conv.time_kernel_size - 1


def _miopen_input(
    x: torch.Tensor, pending: tuple[torch.Tensor, torch.Tensor] | None, dtype: torch.dtype, *, lead: int, norm: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``norm_pad`` of ``x`` plus a pending conv output and its bias, as a MIOpen conv's input."""
    if pending is None:
        return norm_pad(x, dtype, lead=lead, pad=0, norm=norm)
    conv_output, bias = pending
    return norm_pad(conv_output, dtype, lead=lead, pad=0, norm=norm, bias=bias, residual=x)


def _resnet(
    block: ResnetBlock3D, x: torch.Tensor, pending: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """``block`` on ``x`` (+ ``pending``): its output, or its input and its last conv's output and bias,
    still to be added."""
    lead = _causal_lead(block.conv1)
    if block.conv1.kitchen and block.conv1.conv.weight.dtype == torch.float16:
        hidden, _ = norm_pad(x, torch.float16, lead=lead, pad=1, norm=True)
        hidden, _ = norm_pad(_kitchen(block.conv1, hidden), torch.float16, lead=lead, pad=1, norm=True)
        return _kitchen(block.conv2, hidden, x), None
    dtype = block.conv1.conv.weight.dtype
    hidden, total = _miopen_input(x, pending, dtype, lead=lead, norm=True)
    x = x if total is None else total
    hidden, _ = norm_pad(_conv(block.conv1, hidden), dtype, lead=lead, pad=0, norm=True, bias=block.conv1.conv.bias)
    return x, (_conv(block.conv2, hidden), block.conv2.conv.bias)


def _downsample(
    block: SpaceToDepthDownsample, x: torch.Tensor, pending: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    lead = _causal_lead(block.conv)
    extra = 1 if block.stride[0] == 2 else 0
    padded, _ = _miopen_input(x, pending, block.conv.conv.weight.dtype, lead=lead + extra, norm=False)
    return downsample_add(_conv(block.conv, padded), block.conv.conv.bias, padded[:, :, lead:], block)


def encode_prebuilt(encoder: VideoEncoder, sample: torch.Tensor) -> torch.Tensor:
    """``video_encoder.encode_cropped`` on the prebuilt kernels: a channels-last encoder whose head runs on
    kitchen (``AMD_ENCODER_HEAD``) and tail on MIOpen."""
    _release_cache(encoder)
    pixels, _ = norm_pad(patchify(sample, patch_size_hw=encoder.patch_size), encoder.conv_in.conv.weight.dtype,
                         lead=_causal_lead(encoder.conv_in), pad=1, norm=False)
    x, pending = _kitchen(encoder.conv_in, pixels), None
    for block in encoder.down_blocks:
        _release_cache(encoder)
        if isinstance(block, UNetMidBlock3D):
            for resnet in block.res_blocks:
                x, pending = _resnet(resnet, x, pending)
        else:
            x, pending = _downsample(block, x, pending), None
    _release_cache(encoder)
    hidden, _ = _miopen_input(x, pending, encoder.conv_out.conv.weight.dtype, lead=_causal_lead(encoder.conv_out),
                              norm=True)
    means = encoder.conv_out.conv(hidden)[:, :-1, ...]
    return encoder.per_channel_statistics.normalize(means)
