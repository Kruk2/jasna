from __future__ import annotations

import gc
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from jasna.models.basicvsrpp.mmagic.flow_warp import flow_warp
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


class _BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


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
    return os.path.join(engine_dir, f"upsample.trt_{prec}{suf}.engine")


def get_sub_engine_paths(model_weights_path: str, fp16: bool) -> dict[str, str]:
    engine_dir = _sub_engine_dir(model_weights_path)
    paths: dict[str, str] = {}
    for d in DIRECTIONS:
        paths[f"backbone_{d}"] = _backbone_engine_path(engine_dir, d, fp16)
    paths["upsample"] = _upsample_engine_path(engine_dir, fp16)
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
    dtype = torch.float16 if fp16 else torch.float32
    engine_dir = _sub_engine_dir(model_weights_path)
    os.makedirs(engine_dir, exist_ok=True)
    workspace_size = get_workspace_size_bytes()

    generator = _get_inference_generator(model)
    mid = generator.mid_channels

    paths: dict[str, str] = {}

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

    path = _upsample_engine_path(engine_dir, fp16)
    paths["upsample"] = path
    if os.path.isfile(path):
        logger.info("Sub-engine already exists: %s", path)
    else:
        wrapper = _UpsampleWrapper(
            generator.reconstruction,
            generator.upsample1,
            generator.upsample2,
            generator.conv_hr,
            generator.conv_last,
        ).to(device=device, dtype=dtype).eval()
        inp = torch.randn(1, 5 * mid, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)
        compile_and_save_torchtrt_dynamo(
            module=wrapper,
            inputs=[inp],
            output_path=path,
            dtype=dtype,
            workspace_size_bytes=workspace_size,
            message=f"Compiling upsample sub-engine ({5 * mid}\u21923 ch)",
        )
        del wrapper, inp

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return paths


def load_sub_engines(
    model_weights_path: str,
    device: torch.device,
    fp16: bool,
) -> tuple[dict[str, nn.Module], nn.Module] | None:
    paths = get_sub_engine_paths(model_weights_path, fp16)
    if not all(os.path.isfile(p) for p in paths.values()):
        return None

    backbone_engines: dict[str, nn.Module] = {}
    for d in DIRECTIONS:
        backbone_engines[d] = load_torchtrt_export(
            checkpoint_path=paths[f"backbone_{d}"], device=device,
        )
    upsample_engine = load_torchtrt_export(
        checkpoint_path=paths["upsample"], device=device,
    )
    return backbone_engines, upsample_engine


class BasicVSRPlusPlusNetSplit(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        backbone_engines: dict[str, nn.Module],
        upsample_engine: nn.Module,
    ):
        super().__init__()
        self.feat_extract = generator.feat_extract
        self.spynet = generator.spynet
        self.deform_align = generator.deform_align
        self.mid_channels = generator.mid_channels

        self._backbone_engines = backbone_engines
        self._upsample_engine = upsample_engine

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

    def propagate(
        self,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
    ) -> dict[str, list[torch.Tensor]]:
        n, t, _, h, w = flows.size()

        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        backbone_engine = self._backbone_engines[module_name]
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(
                        flow_n2, flow_n1.permute(0, 2, 3, 1)
                    )
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](
                    feat_prop, cond, flow_n1, flow_n2
                )

            feat = [feat_current] + [
                feats[k][idx]
                for k in feats
                if k not in ["spatial", module_name]
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
        outputs: list[torch.Tensor] = []
        num_outputs = len(feats["spatial"])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)

            hr = self._upsample_engine(hr)
            hr += lqs[:, i, :, :, :]
            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lqs.size()

        lqs_downsample = F.interpolate(
            lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic"
        ).view(n, t, c, h // 4, w // 4)

        feats: dict[str, list[torch.Tensor]] = {}
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        h_f, w_f = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h_f, w_f)
        feats["spatial"] = [feats_[:, i, :, :, :] for i in range(0, t)]

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"
                feats[module] = []
                flows = flows_backward if direction == "backward" else flows_forward
                feats = self.propagate(feats, flows, module)

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
    backbone_engines, upsample_engine = result
    generator = _get_inference_generator(model)
    split = BasicVSRPlusPlusNetSplit(generator, backbone_engines, upsample_engine)
    return split
