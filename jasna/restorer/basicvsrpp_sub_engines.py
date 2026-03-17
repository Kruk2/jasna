from __future__ import annotations

import gc
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from jasna.trt.torch_tensorrt_export import (
    compile_and_save_torchtrt_dynamo,
    engine_precision_name,
    engine_system_suffix,
    get_workspace_size_bytes,
    load_torchtrt_export,
)

logger = logging.getLogger(__name__)

DIRECTIONS = ["backward_1", "forward_1", "backward_2", "forward_2"]
FEATURE_SIZE = 64
INPUT_SIZE = 256
MAX_DYNAMIC_BATCH = 180
OPT_DYNAMIC_BATCH = 60


class _BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _DeformAlignWrapper(nn.Module):
    """Fuses offset+mask computation with ``deform_conv2d`` into one module."""

    def __init__(self, deform_align: nn.Module):
        super().__init__()
        self.conv_offset = deform_align.conv_offset
        self._max_res = int(deform_align.max_residue_magnitude)
        self._flow_rep = int(deform_align.deform_groups * 3 * 3 // 2)
        self.dc_weight = deform_align.weight
        self.dc_bias = deform_align.bias
        self._stride = deform_align.stride
        self._padding = deform_align.padding
        self._dilation = deform_align.dilation

    def forward(
        self,
        extra_feat: torch.Tensor,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor,
        feat_prop: torch.Tensor,
        feat_n2: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self._max_res * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, self._flow_rep, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, self._flow_rep, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        inp = torch.cat([feat_prop, feat_n2], dim=1)
        return torchvision.ops.deform_conv2d(
            inp, offset, self.dc_weight, self.dc_bias,
            self._stride, self._padding, self._dilation, mask,
        )


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


def _sub_engine_dir(model_weights_path: str) -> str:
    stem = os.path.splitext(os.path.basename(model_weights_path))[0]
    return os.path.join(os.path.dirname(model_weights_path), f"{stem}_sub_engines")


def _backbone_engine_path(engine_dir: str, direction: str, fp16: bool) -> str:
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    return os.path.join(engine_dir, f"backbone_{direction}.trt_{prec}{suf}.engine")


def _upsample_engine_path(engine_dir: str, fp16: bool) -> str:
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    return os.path.join(engine_dir, f"upsample_dyn.trt_{prec}{suf}.engine")


def _feat_extract_engine_path(engine_dir: str, fp16: bool) -> str:
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    return os.path.join(engine_dir, f"feat_extract_dyn.trt_{prec}{suf}.engine")


def _deform_align_engine_path(engine_dir: str, direction: str, fp16: bool) -> str:
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    return os.path.join(engine_dir, f"deform_align_{direction}.trt_{prec}{suf}.engine")


def get_sub_engine_paths(model_weights_path: str, fp16: bool) -> dict[str, str]:
    engine_dir = _sub_engine_dir(model_weights_path)
    paths: dict[str, str] = {}
    for d in DIRECTIONS:
        paths[f"backbone_{d}"] = _backbone_engine_path(engine_dir, d, fp16)
        paths[f"deform_align_{d}"] = _deform_align_engine_path(engine_dir, d, fp16)
    paths["upsample"] = _upsample_engine_path(engine_dir, fp16)
    paths["feat_extract"] = _feat_extract_engine_path(engine_dir, fp16)
    return paths


def all_sub_engines_exist(model_weights_path: str, fp16: bool) -> bool:
    return all(os.path.isfile(p) for p in get_sub_engine_paths(model_weights_path, fp16).values())


def _get_inference_generator(model: nn.Module) -> nn.Module:
    if hasattr(model, "generator_ema") and model.generator_ema is not None:
        return model.generator_ema
    return model.generator


def compile_basicvsrpp_sub_engines(
    model: nn.Module,
    device: torch.device,
    fp16: bool,
    model_weights_path: str,
) -> dict[str, str]:
    import torch_tensorrt  # type: ignore[import-not-found]

    dtype = torch.float16 if fp16 else torch.float32
    engine_dir = _sub_engine_dir(model_weights_path)
    os.makedirs(engine_dir, exist_ok=True)
    workspace_size = get_workspace_size_bytes()

    generator = _get_inference_generator(model)
    mid = generator.mid_channels

    paths: dict[str, str] = {}

    # ── backbone engines (static batch=1, per-frame sequential) ──
    for i, direction in enumerate(DIRECTIONS):
        path = _backbone_engine_path(engine_dir, direction, fp16)
        paths[f"backbone_{direction}"] = path
        if os.path.isfile(path):
            logger.info("Sub-engine already exists: %s", path)
            continue

        in_channels = (2 + i) * mid
        wrapper = _BackboneWrapper(generator.backbone[direction]).to(device=device, dtype=dtype).eval()
        inp = torch.randn(1, in_channels, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        compile_and_save_torchtrt_dynamo(
            module=wrapper,
            inputs=[inp],
            output_path=path,
            dtype=dtype,
            workspace_size_bytes=workspace_size,
            message=f"Compiling backbone sub-engine [{direction}] ({in_channels}\u2192{mid} ch)",
        )
        del wrapper, inp

    # ── deform_align engines (fused offset + deform_conv2d, static batch=1) ──
    cond_channels = 3 * mid
    for direction in DIRECTIONS:
        path = _deform_align_engine_path(engine_dir, direction, fp16)
        paths[f"deform_align_{direction}"] = path
        if os.path.isfile(path):
            logger.info("Sub-engine already exists: %s", path)
            continue

        wrapper = _DeformAlignWrapper(
            generator.deform_align[direction],
        ).to(device=device, dtype=dtype).eval()
        inp_cond = torch.randn(1, cond_channels, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_f1 = torch.randn(1, 2, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_f2 = torch.randn(1, 2, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_fp = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        inp_fn2 = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        compile_and_save_torchtrt_dynamo(
            module=wrapper,
            inputs=[inp_cond, inp_f1, inp_f2, inp_fp, inp_fn2],
            output_path=path,
            dtype=dtype,
            workspace_size_bytes=workspace_size,
            message=f"Compiling deform_align sub-engine [{direction}] (fused offset+deform_conv2d)",
        )
        del wrapper, inp_cond, inp_f1, inp_f2, inp_fp, inp_fn2

    # ── upsample engine (dynamic batch – called once for all frames) ──
    path = _upsample_engine_path(engine_dir, fp16)
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
            opt_shape=[OPT_DYNAMIC_BATCH, in_ch, FEATURE_SIZE, FEATURE_SIZE],
            max_shape=[MAX_DYNAMIC_BATCH, in_ch, FEATURE_SIZE, FEATURE_SIZE],
            dtype=dtype,
        )
        compile_and_save_torchtrt_dynamo(
            module=wrapper,
            inputs=[dyn_input],
            output_path=path,
            dtype=dtype,
            workspace_size_bytes=workspace_size,
            message=f"Compiling upsample sub-engine ({in_ch}\u21923 ch, dynamic batch)",
            device=device,
        )
        del wrapper

    # ── feat_extract engine (dynamic batch) ──
    path = _feat_extract_engine_path(engine_dir, fp16)
    paths["feat_extract"] = path
    if os.path.isfile(path):
        logger.info("Sub-engine already exists: %s", path)
    else:
        feat_extract = generator.feat_extract.to(device=device, dtype=dtype).eval()
        fe_input = torch_tensorrt.Input(
            min_shape=[1, 3, INPUT_SIZE, INPUT_SIZE],
            opt_shape=[OPT_DYNAMIC_BATCH, 3, INPUT_SIZE, INPUT_SIZE],
            max_shape=[MAX_DYNAMIC_BATCH, 3, INPUT_SIZE, INPUT_SIZE],
            dtype=dtype,
        )
        compile_and_save_torchtrt_dynamo(
            module=feat_extract,
            inputs=[fe_input],
            output_path=path,
            dtype=dtype,
            workspace_size_bytes=workspace_size,
            message=f"Compiling feat_extract sub-engine (3\u2192{mid} ch, dynamic batch)",
            device=device,
        )
        del feat_extract

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return paths


def load_sub_engines(
    model_weights_path: str,
    device: torch.device,
    fp16: bool,
) -> tuple[dict[str, nn.Module], nn.Module, nn.Module, dict[str, nn.Module]] | None:
    """Returns ``(backbone_engines, upsample, feat_extract, deform_align_engines)`` or *None*."""
    paths = get_sub_engine_paths(model_weights_path, fp16)
    if not all(os.path.isfile(p) for p in paths.values()):
        return None

    backbone_engines: dict[str, nn.Module] = {}
    deform_align_engines: dict[str, nn.Module] = {}
    for d in DIRECTIONS:
        backbone_engines[d] = load_torchtrt_export(
            checkpoint_path=paths[f"backbone_{d}"], device=device,
        )
        deform_align_engines[d] = load_torchtrt_export(
            checkpoint_path=paths[f"deform_align_{d}"], device=device,
        )
    upsample_engine = load_torchtrt_export(
        checkpoint_path=paths["upsample"], device=device,
    )
    feat_extract_engine = load_torchtrt_export(
        checkpoint_path=paths["feat_extract"], device=device,
    )
    return backbone_engines, upsample_engine, feat_extract_engine, deform_align_engines


class BasicVSRPlusPlusNetSplit(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        backbone_engines: dict[str, nn.Module],
        upsample_engine: nn.Module,
        feat_extract_engine: nn.Module,
        deform_align_engines: dict[str, nn.Module],
    ):
        super().__init__()
        self.spynet = generator.spynet
        self.mid_channels = generator.mid_channels

        self._backbone_engines = backbone_engines
        self._upsample_engine = upsample_engine
        self._feat_extract_engine = feat_extract_engine
        self._deform_align_engines = deform_align_engines


    def compute_flow(self, lqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, t, c, h, w = lqs.size()
        if t == 1:
            empty = lqs.new_zeros(n, 0, 2, h, w)
            return empty, empty
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

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

    def propagate(
        self,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
        grid: torch.Tensor,
    ) -> dict[str, list[torch.Tensor]]:
        n, t, _, h, w = flows.size()
        mid = self.mid_channels

        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
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

        backbone_engine = self._backbone_engines[module_name]
        dae = self._deform_align_engines[module_name]
        other_keys = [k for k in feats if k not in ["spatial", module_name]]

        zero_feat = flows.new_zeros(n, mid, h, w)
        zero_flow = flows.new_zeros(n, 2, h, w)
        zero_cond = zero_feat

        feat_prop = flows.new_zeros(n, mid, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = F.grid_sample(
                    feat_prop, flows_grid[:, flow_idx[i]],
                    mode="bilinear", padding_mode="zeros", align_corners=True,
                )

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = acc_flows[i]
                    cond_n2 = F.grid_sample(
                        feat_n2, acc_grids[i],
                        mode="bilinear", padding_mode="zeros",
                        align_corners=True,
                    )
                else:
                    feat_n2 = zero_feat
                    flow_n2 = zero_flow
                    cond_n2 = zero_cond

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = dae(cond, flow_n1, flow_n2, feat_prop, feat_n2)

            feat = [feat_current] + [
                feats[k][idx] for k in other_keys
            ] + [feat_prop]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + backbone_engine(feat)
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

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lqs.size()

        lqs_downsample = F.interpolate(
            lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic"
        ).view(n, t, c, h // 4, w // 4)

        feats: dict[str, list[torch.Tensor]] = {}
        feats_ = self._feat_extract_engine(lqs.view(-1, c, h, w))
        h_f, w_f = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h_f, w_f)
        feats["spatial"] = [feats_[:, i, :, :, :] for i in range(0, t)]

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        grid = self._make_identity_grid(h_f, w_f, lqs.device, lqs.dtype)
        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"
                feats[module] = []
                flows = flows_backward if direction == "backward" else flows_forward
                feats = self.propagate(feats, flows, module, grid)

        return self.upsample(lqs, feats)


def create_split_forward(
    model: nn.Module,
    model_weights_path: str,
    device: torch.device,
    fp16: bool,
) -> BasicVSRPlusPlusNetSplit | None:
    result = load_sub_engines(model_weights_path, device, fp16)
    if result is None:
        return None
    backbone_engines, upsample_engine, feat_extract_engine, deform_align_engines = result
    generator = _get_inference_generator(model)
    split = BasicVSRPlusPlusNetSplit(
        generator, backbone_engines, upsample_engine, feat_extract_engine,
        deform_align_engines,
    )
    return split
