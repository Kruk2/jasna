from __future__ import annotations

import gc
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from jasna.engine_paths import (
    BASICVSRPP_DIRECTIONS as DIRECTIONS,
    _basicvsrpp_sub_engine_dir as _sub_engine_dir,
    engine_precision_name,
    engine_system_suffix,
    get_basicvsrpp_sub_engine_paths as get_sub_engine_paths,
    all_basicvsrpp_sub_engines_exist as all_sub_engines_exist,
)
from jasna.trt.torch_tensorrt_export import (
    compile_and_save_torchtrt_dynamo,
    get_workspace_size_bytes,
    load_torchtrt_export,
)

logger = logging.getLogger(__name__)

# DIRECTIONS imported from engine_paths
FEATURE_SIZE = 64
INPUT_SIZE = 256
MAX_DYNAMIC_BATCH = 180
OPT_DYNAMIC_BATCH = 60


class _PropagateBodyWrapper(nn.Module):
    """Fuses grid_sample + deform_align + backbone + residual add."""

    def __init__(self, deform_align: nn.Module, backbone: nn.Module):
        super().__init__()
        self.conv_offset = deform_align.conv_offset
        self._max_res = int(deform_align.max_residue_magnitude)
        self._flow_rep = int(deform_align.deform_groups * 3 * 3 // 2)
        self.dc_weight = deform_align.weight
        self.dc_bias = deform_align.bias
        self._stride = deform_align.stride
        self._padding = deform_align.padding
        self._dilation = deform_align.dilation
        self.backbone = backbone

    def forward(
        self,
        feat_prop: torch.Tensor,
        grid_n1: torch.Tensor,
        feat_n2: torch.Tensor,
        grid_n2: torch.Tensor,
        feat_current: torch.Tensor,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor,
        backbone_prefix: torch.Tensor,
    ) -> torch.Tensor:
        cond_n1 = F.grid_sample(feat_prop, grid_n1, mode="bilinear", padding_mode="zeros", align_corners=True)
        cond_n2 = F.grid_sample(feat_n2, grid_n2, mode="bilinear", padding_mode="zeros", align_corners=True)

        x = torch.cat([cond_n1, feat_current, cond_n2, flow_1, flow_2], dim=1)
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self._max_res * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, self._flow_rep, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, self._flow_rep, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        inp = torch.cat([feat_prop, feat_n2], dim=1)
        feat_prop_new = torchvision.ops.deform_conv2d(
            inp, offset, self.dc_weight, self.dc_bias,
            self._stride, self._padding, self._dilation, mask,
        )

        feat = torch.cat([backbone_prefix, feat_prop_new], dim=1)
        return feat_prop_new + self.backbone(feat)


class _UpsampleWrapper(nn.Module):
    def __init__(
        self,
        reconstruction: nn.Module,
        upsample1: nn.Module,
        upsample2: nn.Module,
        conv_hr: nn.Module,
        conv_last: nn.Module,
    ):
        super().__init__()
        self.reconstruction = reconstruction
        self.upsample1 = upsample1
        self.upsample2 = upsample2
        self.conv_hr = conv_hr
        self.conv_last = conv_last
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reconstruction(x)
        x = self.lrelu(self.upsample1(x))
        x = self.lrelu(self.upsample2(x))
        x = self.lrelu(self.conv_hr(x))
        return self.conv_last(x)


class _SPyNetWrapper(nn.Module):
    """Flattened SPyNet that explicitly unrolls the 6-level pyramid.

    The original SPyNet uses Python lists and a for-loop to build the image
    pyramid and process each level. torch_tensorrt can't compile that as a
    single graph – it splits into multiple sub-graphs whose dynamic shapes
    break ``torch.export.export`` at save time.

    This wrapper hard-codes the 6 levels so the entire forward is a single
    traceable graph.  Valid only for inputs whose spatial size is already a
    multiple of 32 (true for our 64×64 downsampled frames).
    """

    def __init__(self, spynet: nn.Module):
        super().__init__()
        self.register_buffer("mean", spynet.mean.clone())
        self.register_buffer("std", spynet.std.clone())
        self.bm0 = spynet.basic_module[0]
        self.bm1 = spynet.basic_module[1]
        self.bm2 = spynet.basic_module[2]
        self.bm3 = spynet.basic_module[3]
        self.bm4 = spynet.basic_module[4]
        self.bm5 = spynet.basic_module[5]

    @staticmethod
    def _warp_border(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        theta = torch.eye(2, 3, device=x.device, dtype=x.dtype).unsqueeze(0).expand(n, -1, -1)
        grid = F.affine_grid(theta, (n, c, h, w), align_corners=True)
        flow_perm = flow.permute(0, 2, 3, 1)
        flow_x = flow_perm[..., 0] * (2.0 / max(w - 1, 1))
        flow_y = flow_perm[..., 1] * (2.0 / max(h - 1, 1))
        return F.grid_sample(
            x, grid + torch.stack((flow_x, flow_y), dim=-1),
            mode="bilinear", padding_mode="border", align_corners=True,
        )

    def _level(self, bm: nn.Module, ref_l: torch.Tensor, supp_l: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
        warped = self._warp_border(supp_l, flow_up)
        return flow_up + bm(torch.cat([ref_l, warped, flow_up], dim=1))

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n = ref.shape[0]
        h, w = ref.shape[2:]

        r0 = (ref - self.mean) / self.std
        s0 = (supp - self.mean) / self.std

        r1 = F.avg_pool2d(r0, 2, 2, count_include_pad=False)
        s1 = F.avg_pool2d(s0, 2, 2, count_include_pad=False)
        r2 = F.avg_pool2d(r1, 2, 2, count_include_pad=False)
        s2 = F.avg_pool2d(s1, 2, 2, count_include_pad=False)
        r3 = F.avg_pool2d(r2, 2, 2, count_include_pad=False)
        s3 = F.avg_pool2d(s2, 2, 2, count_include_pad=False)
        r4 = F.avg_pool2d(r3, 2, 2, count_include_pad=False)
        s4 = F.avg_pool2d(s3, 2, 2, count_include_pad=False)
        r5 = F.avg_pool2d(r4, 2, 2, count_include_pad=False)
        s5 = F.avg_pool2d(s4, 2, 2, count_include_pad=False)

        flow = ref.new_zeros(n, 2, h // 32, w // 32)
        warped = self._warp_border(s5, flow)
        flow = flow + self.bm0(torch.cat([r5, warped, flow], dim=1))

        flow = self._level(self.bm1, r4, s4, flow)
        flow = self._level(self.bm2, r3, s3, flow)
        flow = self._level(self.bm3, r2, s2, flow)
        flow = self._level(self.bm4, r1, s1, flow)
        flow = self._level(self.bm5, r0, s0, flow)

        return flow


class _PreprocessWrapper(nn.Module):
    """Fuses feat_extract + bicubic downsample + bidirectional SPyNet flow.

    Single engine call replaces three separate stages, eliminating
    kernel-launch overhead between feat_extract, interpolate, and SPyNet.
    Input: (T, 3, H, W)  where H=W=INPUT_SIZE
    Output: (feats, flows_fwd, flows_bwd)
    """

    def __init__(self, feat_extract: nn.Module, spynet: nn.Module):
        super().__init__()
        self.feat_extract = feat_extract
        self.spynet = _SPyNetWrapper(spynet)

    def forward(self, lqs_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.feat_extract(lqs_flat)
        lqs_ds = F.interpolate(lqs_flat, scale_factor=0.25, mode="bicubic")
        lqs_1 = lqs_ds[:-1]
        lqs_2 = lqs_ds[1:]
        flows_bwd = self.spynet(lqs_1, lqs_2)
        flows_fwd = self.spynet(lqs_2, lqs_1)
        return feats, flows_fwd, flows_bwd


def _loop_body_engine_path(engine_dir: str, direction: str, fp16: bool) -> str:
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    return os.path.join(engine_dir, f"loop_body_{direction}.trt_{prec}{suf}.engine")


def _upsample_engine_path(engine_dir: str, fp16: bool, max_clip_size: int) -> str:
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    return os.path.join(engine_dir, f"upsample_dyn_b{max_clip_size}.trt_{prec}{suf}.engine")


def _preprocess_engine_path(engine_dir: str, fp16: bool, max_clip_size: int) -> str:
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    return os.path.join(engine_dir, f"preprocess_b{max_clip_size}.trt_{prec}{suf}.engine")


def _get_inference_generator(model: nn.Module) -> nn.Module:
    if hasattr(model, "generator_ema") and model.generator_ema is not None:
        return model.generator_ema
    return model.generator


def compile_basicvsrpp_sub_engines(
    model: nn.Module,
    device: torch.device,
    fp16: bool,
    model_weights_path: str,
    max_clip_size: int = 60,
    optimization_level: int = 5,
) -> dict[str, str]:
    import torch_tensorrt  # type: ignore[import-not-found]

    dtype = torch.float16 if fp16 else torch.float32
    engine_dir = _sub_engine_dir(model_weights_path)
    os.makedirs(engine_dir, exist_ok=True)
    workspace_size = get_workspace_size_bytes()

    generator = _get_inference_generator(model)
    mid = generator.mid_channels

    paths: dict[str, str] = {}

    # ── loop_body engines (fused deform_align + backbone, static batch=1) ──
    cond_channels = 3 * mid
    for i, direction in enumerate(DIRECTIONS):
        path = _loop_body_engine_path(engine_dir, direction, fp16)
        paths[f"loop_body_{direction}"] = path
        if os.path.isfile(path):
            logger.info("Sub-engine already exists: %s", path)
            continue

        prefix_channels = (1 + i) * mid
        wrapper = _PropagateBodyWrapper(
            generator.deform_align[direction],
            generator.backbone[direction],
        ).to(device=device, dtype=dtype).eval()
        inp_fp = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_g1 = torch.randn(1, FEATURE_SIZE, FEATURE_SIZE, 2, dtype=dtype, device=device)
        inp_fn2 = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_g2 = torch.randn(1, FEATURE_SIZE, FEATURE_SIZE, 2, dtype=dtype, device=device)
        inp_fc = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_f1 = torch.randn(1, 2, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_f2 = torch.randn(1, 2, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_bp = torch.randn(1, prefix_channels, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        compile_and_save_torchtrt_dynamo(
            module=wrapper,
            inputs=[inp_fp, inp_g1, inp_fn2, inp_g2, inp_fc, inp_f1, inp_f2, inp_bp],
            output_path=path,
            dtype=dtype,
            workspace_size_bytes=workspace_size,
            message=f"Compiling sub-engine {i + 1}/6: loop_body [{direction}]",
            optimization_level=optimization_level,
        )
        del wrapper, inp_fp, inp_g1, inp_fn2, inp_g2, inp_fc, inp_f1, inp_f2, inp_bp

    # ── preprocess engine (feat_extract + downsample + bidirectional SPyNet, dynamic batch) ──
    path = _preprocess_engine_path(engine_dir, fp16, max_clip_size)
    paths["preprocess"] = path
    if os.path.isfile(path):
        logger.info("Sub-engine already exists: %s", path)
    else:
        wrapper = _PreprocessWrapper(
            generator.feat_extract, generator.spynet,
        ).to(device=device, dtype=dtype).eval()
        dyn_input = torch_tensorrt.Input(
            min_shape=[3, 3, INPUT_SIZE, INPUT_SIZE],
            opt_shape=[max_clip_size, 3, INPUT_SIZE, INPUT_SIZE],
            max_shape=[max_clip_size, 3, INPUT_SIZE, INPUT_SIZE],
            dtype=dtype,
        )
        compile_and_save_torchtrt_dynamo(
            module=wrapper,
            inputs=[dyn_input],
            output_path=path,
            dtype=dtype,
            workspace_size_bytes=workspace_size,
            message=f"Compiling sub-engine 5/6: preprocess (batch=3..{max_clip_size})",
            device=device,
            optimization_level=optimization_level,
        )
        del wrapper

    # ── upsample engine (dynamic batch – called once for all frames) ──
    path = _upsample_engine_path(engine_dir, fp16, max_clip_size)
    paths["upsample"] = path
    if os.path.isfile(path):
        logger.info("Sub-engine already exists: %s", path)
    else:
        in_ch = 5 * mid
        wrapper = _UpsampleWrapper(
            generator.reconstruction,
            generator.upsample1,
            generator.upsample2,
            generator.conv_hr,
            generator.conv_last,
        ).to(device=device, dtype=dtype).eval()
        dyn_input = torch_tensorrt.Input(
            min_shape=[1, in_ch, FEATURE_SIZE, FEATURE_SIZE],
            opt_shape=[max_clip_size, in_ch, FEATURE_SIZE, FEATURE_SIZE],
            max_shape=[max_clip_size, in_ch, FEATURE_SIZE, FEATURE_SIZE],
            dtype=dtype,
        )
        compile_and_save_torchtrt_dynamo(
            module=wrapper,
            inputs=[dyn_input],
            output_path=path,
            dtype=dtype,
            workspace_size_bytes=workspace_size,
            message=f"Compiling sub-engine 6/6: upsample (batch=1..{max_clip_size})",
            device=device,
            optimization_level=optimization_level,
        )
        del wrapper

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return paths


def load_sub_engines(
    model_weights_path: str,
    device: torch.device,
    fp16: bool,
    max_clip_size: int = 60,
) -> tuple[dict[str, nn.Module], nn.Module, nn.Module] | None:
    """Returns ``(loop_body_engines, preprocess, upsample)`` or *None*."""
    paths = get_sub_engine_paths(model_weights_path, fp16, max_clip_size)
    if not all(os.path.isfile(p) for p in paths.values()):
        return None

    loop_body_engines: dict[str, nn.Module] = {}
    for d in DIRECTIONS:
        loop_body_engines[d] = load_torchtrt_export(
            checkpoint_path=paths[f"loop_body_{d}"], device=device,
        )
    preprocess_engine = load_torchtrt_export(
        checkpoint_path=paths["preprocess"], device=device,
    )
    upsample_engine = load_torchtrt_export(
        checkpoint_path=paths["upsample"], device=device,
    )
    return loop_body_engines, preprocess_engine, upsample_engine


class BasicVSRPlusPlusNetSplit(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        loop_body_engines: dict[str, nn.Module],
        preprocess_engine: nn.Module,
        upsample_engine: nn.Module,
    ):
        super().__init__()
        self.mid_channels = generator.mid_channels
        self.backbone = generator.backbone

        self._loop_body_engines = loop_body_engines
        self._preprocess_engine = preprocess_engine
        self._upsample_engine = upsample_engine

    @staticmethod
    def _make_identity_grid(
        h: int, w: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        theta = torch.eye(2, 3, device=device, dtype=dtype).unsqueeze(0)
        return F.affine_grid(theta, (1, 1, h, w), align_corners=True)

    @staticmethod
    def _flow_warp_cached(
        x: torch.Tensor, flow: torch.Tensor, grid: torch.Tensor,
    ) -> torch.Tensor:
        n, _, h, w = x.shape
        g = grid.expand(n, -1, -1, -1) if n > 1 else grid
        flow_x = flow[..., 0] * (2.0 / max(w - 1, 1))
        flow_y = flow[..., 1] * (2.0 / max(h - 1, 1))
        return F.grid_sample(
            x, g + torch.stack((flow_x, flow_y), dim=-1),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )

    def _precompute_accumulated_flows(
        self,
        flows: torch.Tensor,
        flow_idx: list[int],
        frame_count: int,
        grid: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        if frame_count <= 2:
            return {}
        indices = list(range(2, frame_count))
        fn1_list = [flows[:, flow_idx[i], :, :, :] for i in indices]
        fn2_list = [flows[:, flow_idx[i - 1], :, :, :] for i in indices]
        fn1_batch = torch.cat(fn1_list, dim=0)
        fn2_batch = torch.cat(fn2_list, dim=0)
        warped = self._flow_warp_cached(
            fn2_batch, fn1_batch.permute(0, 2, 3, 1), grid,
        )
        acc = fn1_batch + warped
        return {i: acc[j : j + 1] for j, i in enumerate(indices)}

    def _precompute_flow_data(
        self,
        flows: torch.Tensor,
        direction: str,
        grid: torch.Tensor,
    ) -> tuple[
        list[int], list[int],
        dict[int, torch.Tensor], torch.Tensor, dict[int, torch.Tensor],
    ]:
        """Precompute acc_flows, flows_grid, acc_grids for a flow direction.

        Returns (frame_idx, flow_idx, acc_flows, flows_grid, acc_grids).
        """
        n, t, _, h, w = flows.size()

        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        if direction == "backward":
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        acc_flows = self._precompute_accumulated_flows(
            flows, flow_idx, len(frame_idx), grid,
        )

        scale_x = 2.0 / max(w - 1, 1)
        scale_y = 2.0 / max(h - 1, 1)

        flows_grid = flows.permute(0, 1, 3, 4, 2).contiguous()
        flows_grid[..., 0].mul_(scale_x)
        flows_grid[..., 1].mul_(scale_y)
        flows_grid.add_(grid.unsqueeze(1))

        acc_grids: dict[int, torch.Tensor] = {}
        if acc_flows:
            acc_keys = sorted(acc_flows.keys())
            acc_batch = torch.cat([acc_flows[k] for k in acc_keys], dim=0)
            acc_batch_nhwc = acc_batch.permute(0, 2, 3, 1).contiguous()
            acc_batch_nhwc[..., 0].mul_(scale_x)
            acc_batch_nhwc[..., 1].mul_(scale_y)
            acc_batch_nhwc.add_(grid)
            acc_grids = {
                k: acc_batch_nhwc[j : j + 1] for j, k in enumerate(acc_keys)
            }

        return frame_idx, flow_idx, acc_flows, flows_grid, acc_grids

    def propagate(
        self,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
        grid: torch.Tensor,
        frame_idx: list[int],
        flow_idx: list[int],
        acc_flows: dict[int, torch.Tensor],
        flows_grid: torch.Tensor,
        acc_grids: dict[int, torch.Tensor],
    ) -> dict[str, list[torch.Tensor]]:
        n, t, _, h, w = flows.size()
        mid = self.mid_channels

        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        lbe = self._loop_body_engines[module_name]
        backbone_pt = self.backbone[module_name]
        other_keys = [k for k in feats if k not in ["spatial", module_name]]

        zero_feat = flows.new_zeros(n, mid, h, w)
        zero_flow = flows.new_zeros(n, 2, h, w)

        feat_prop = flows.new_zeros(n, mid, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            backbone_prefix = torch.cat(
                [feat_current] + [feats[k][idx] for k in other_keys],
                dim=1,
            )
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                g_n1 = flows_grid[:, flow_idx[i]]

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = acc_flows[i]
                    g_n2 = acc_grids[i]
                else:
                    feat_n2 = zero_feat
                    flow_n2 = zero_flow
                    g_n2 = grid

                feat_prop = lbe(feat_prop, g_n1, feat_n2, g_n2, feat_current, flow_n1, flow_n2, backbone_prefix)
            else:
                feat = torch.cat([backbone_prefix, feat_prop], dim=1)
                feat_prop = feat_prop + backbone_pt(feat)
            feats[module_name].append(feat_prop)

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(
        self, lqs: torch.Tensor, feats: dict[str, list[torch.Tensor]]
    ) -> torch.Tensor:
        t = lqs.size(1)
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        hr_list: list[torch.Tensor] = []
        for i in range(t):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr_list.append(torch.cat(hr, dim=1))

        hr_batch = torch.cat(hr_list, dim=0)
        out_batch = self._upsample_engine(hr_batch)
        out_batch = out_batch + lqs.squeeze(0)
        return out_batch.unsqueeze(0)

    _PREPROCESS_MIN_BATCH = 3

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lqs.size()

        lqs_flat = lqs.view(-1, c, h, w)
        padded = t < self._PREPROCESS_MIN_BATCH
        if padded:
            pad_count = self._PREPROCESS_MIN_BATCH - t
            lqs_flat = torch.cat(
                [lqs_flat, lqs_flat[-1:].expand(pad_count, -1, -1, -1)], dim=0,
            )
            t_engine = self._PREPROCESS_MIN_BATCH
        else:
            t_engine = t

        feats_, flows_fwd, flows_bwd = self._preprocess_engine(lqs_flat)
        h_f, w_f = feats_.shape[2:]

        feats_ = feats_[:t]
        flows_fwd = flows_fwd[:t - 1] if t > 1 else flows_fwd[:0]
        flows_bwd = flows_bwd[:t - 1] if t > 1 else flows_bwd[:0]

        feats_ = feats_.view(n, t, -1, h_f, w_f)

        feats: dict[str, list[torch.Tensor]] = {}
        feats["spatial"] = [feats_[:, i, :, :, :] for i in range(0, t)]

        flows_forward = flows_fwd.view(n, max(t - 1, 0), 2, h // 4, w // 4)
        flows_backward = flows_bwd.view(n, max(t - 1, 0), 2, h // 4, w // 4)

        grid = self._make_identity_grid(h_f, w_f, lqs.device, lqs.dtype)

        flow_data: dict[str, tuple] = {}
        for direction in ["backward", "forward"]:
            flows = flows_backward if direction == "backward" else flows_forward
            flow_data[direction] = self._precompute_flow_data(flows, direction, grid)

        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"
                feats[module] = []
                flows = flows_backward if direction == "backward" else flows_forward
                fi, fli, af, fg, ag = flow_data[direction]
                feats = self.propagate(feats, flows, module, grid, fi, fli, af, fg, ag)

        return self.upsample(lqs, feats)


def create_split_forward(
    model: nn.Module,
    model_weights_path: str,
    device: torch.device,
    fp16: bool,
    max_clip_size: int = 60,
) -> BasicVSRPlusPlusNetSplit | None:
    result = load_sub_engines(model_weights_path, device, fp16, max_clip_size)
    if result is None:
        return None
    loop_body_engines, preprocess_engine, upsample_engine = result
    generator = _get_inference_generator(model)
    split = BasicVSRPlusPlusNetSplit(
        generator, loop_body_engines, preprocess_engine, upsample_engine,
    )
    return split
