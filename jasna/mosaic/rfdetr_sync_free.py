"""rfdetr's export forward without host synchronization, for the AMD torch runner.

``LWDETR.forward_export`` is the graph the NVIDIA ONNX/TensorRT engine is built
from: no padding masks, no ``masked_fill``, no encoder mask head. Run eagerly it
still blocks the host on every call: ``Transformer.forward`` copies the level
shapes from pageable memory to the GPU, and every decoder ``MSDeformAttn``
asserts on them and splits its values by their ``.item()``. The replacements here
keep the level shapes as Python ints and are otherwise the same ops, so the
detector thread never waits for the GPU (or for other threads' work on it).
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F


class _LevelShapes:
    def __init__(self) -> None:
        self._tensors: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.hw_by_tensor_id: dict[int, tuple[tuple[int, int], ...]] = {}

    def tensors(self, hw: tuple[tuple[int, int], ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (hw, device)
        if key not in self._tensors:
            spatial_shapes = torch.tensor(hw, dtype=torch.long, device=device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            self._tensors[key] = (spatial_shapes, level_start_index)
            self.hw_by_tensor_id[id(spatial_shapes)] = hw
        return self._tensors[key]


def export_without_host_syncs(core: torch.nn.Module) -> None:
    """Switch ``core`` (an rfdetr ``LWDETR``) to its export forward, minus the host syncs."""
    from rfdetr.models.ops.modules import MSDeformAttn

    transformer = core.transformer
    if (
        transformer.use_grouppose_keypoints
        or transformer.num_registers
        or not transformer.two_stage
        or transformer.dec_layers == 0
        or core.backbone[0].cross_attn_projector is not None
    ):
        raise ValueError("RF-DETR layout differs from the one the sync-free export forward supports")
    core.export()
    level_shapes = _LevelShapes()
    transformer.forward = types.MethodType(_two_stage_transformer_forward(level_shapes), transformer)
    for module in core.modules():
        if isinstance(module, MSDeformAttn):
            module.forward = types.MethodType(_box_ms_deform_attn_forward(level_shapes), module)


def _two_stage_transformer_forward(level_shapes: _LevelShapes):
    from rfdetr.models.transformer import gen_encoder_output_proposals

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None):
        hw = tuple((src.shape[2], src.shape[3]) for src in srcs)
        memory = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], 1)
        pos = torch.cat([pos_embed.flatten(2).transpose(1, 2) for pos_embed in pos_embeds], 1)
        spatial_shapes, level_start_index = level_shapes.tensors(hw, memory.device)

        output_memory, output_proposals = gen_encoder_output_proposals(memory, None, hw, unsigmoid=not self.bbox_reparam)
        output_memory = self.enc_output_norm[0](self.enc_output[0](output_memory))
        enc_class = self.enc_out_class_embed[0](output_memory)
        if self.bbox_reparam:
            delta = self.enc_out_bbox_embed[0](output_memory)
            enc_coord = torch.concat(
                [
                    delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2],
                    delta[..., 2:].exp() * output_proposals[..., 2:],
                ],
                dim=-1,
            )
        else:
            enc_coord = self.enc_out_bbox_embed[0](output_memory) + output_proposals
        topk = min(self.num_queries, enc_class.shape[-2])
        topk_proposals = torch.topk(enc_class.max(-1)[0], topk, dim=1)[1]
        boxes_ts = torch.gather(enc_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        memory_ts = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))

        batch_size = memory.shape[0]
        tgt = query_feat.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        refpoint_embed = refpoint_embed.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        ts_len = boxes_ts.shape[-2]
        refpoint_ts = refpoint_embed[..., :ts_len, :]
        if self.bbox_reparam:
            refpoint_ts = torch.concat(
                [
                    refpoint_ts[..., :2] * boxes_ts[..., 2:] + boxes_ts[..., :2],
                    refpoint_ts[..., 2:].exp() * boxes_ts[..., 2:],
                ],
                dim=-1,
            )
        else:
            refpoint_ts = refpoint_ts + boxes_ts
        refpoint_embed = torch.concat([refpoint_ts, refpoint_embed[..., ts_len:, :]], dim=-2)

        hs, references = self.decoder(
            tgt,
            memory,
            pos=pos,
            refpoints_unsigmoid=refpoint_embed,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
        )[:2]
        return hs, references, memory_ts, boxes_ts if self.bbox_reparam else boxes_ts.sigmoid()

    return forward


def _box_ms_deform_attn_forward(level_shapes: _LevelShapes):
    from rfdetr.models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        input_spatial_shapes_hw=None,
    ):
        batch_size, len_query, _ = query.shape
        len_input = input_flatten.shape[1]
        value = self.value_proj(input_flatten)
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, len_query, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, len_query, self.n_heads, self.n_levels * self.n_points
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        )
        attention_weights = F.softmax(attention_weights, -1)
        value = value.transpose(1, 2).contiguous().view(
            batch_size, self.n_heads, self.d_model // self.n_heads, len_input
        )
        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=level_shapes.hw_by_tensor_id[id(input_spatial_shapes)],
        )
        return self.output_proj(output)

    return forward
