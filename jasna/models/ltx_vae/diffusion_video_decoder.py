"""Single-step x0 diffusion video VAE decoder (``NADiffusionDecoder``), ``chunked_eager`` mode.

Ported from ``ltx_core/model/video_vae/diffusion_video_decoder.py`` and
``ltx_core/model/transformer/timestep_embedding.py``.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Tuple

import torch
from torch import nn

from jasna.models.ltx_vae import diffusion_tiling
from jasna.models.ltx_vae.ops import PerChannelStatistics, patchify, unpatchify
from jasna.models.ltx_vae.tiling import (
    SpatioTemporalScaleFactors,
    Tile,
    TileSizeConfig,
    VideoLatentShape,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
    scale_by_masks_1d,
)
from jasna.models.ltx_vae.transformer import (
    AdaLNZero,
    ChunkedDiffusionNABlock,
    LinearPixelShuffleUpsample,
    NABlock,
)

_TIMESTEP_PROJ_DIM = 256


def get_timestep_embedding(timesteps: torch.Tensor) -> torch.Tensor:
    """Sinusoidal ``[cos, sin]`` embedding of 1-D ``timesteps`` (256 dims, max period 10000)."""
    half_dim = _TIMESTEP_PROJ_DIM // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=_TIMESTEP_PROJ_DIM, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        return self.timestep_embedder(get_timestep_embedding(timestep).to(dtype=hidden_dtype))


class DiffusionVideoDecoder(nn.Module):
    """Diffusion video VAE decoder with a neighborhood-attention backbone.

    Stages 1-4 deterministically upsample the latent into a context volume; stage 5 runs
    ``ChunkedDiffusionNABlock``s that predict clean patchified pixels (x0) from per-tile
    noise in one step, guided by that context. The last latent frame is replicated
    ``(stage1_K_t // 2) * 2`` times through stages 1-4 against the NA last-frame border and
    cropped off before stage 5. Latents below ``stage_min_tile_sizes`` are edge-padded first;
    leftover pad is cropped from the final pixels.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        head_dim: int,
        stage_channels: Tuple[int, ...],
        stage_depths: Tuple[int, ...],
        stage_kernels: Tuple[Tuple[int, int, int], ...],
        upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...],
        stage5_kernel: Tuple[int, int, int],
        stage5_channels: int,
        t_emb_dim: int,
        timestep_scale_multiplier: float,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.stage_channels = stage_channels
        self.video_downscale_factors = SpatioTemporalScaleFactors.default()
        self.stage5_kernel = tuple(stage5_kernel)
        self._natten_trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)
        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_i in range(len(stage_channels) - 1):
            c = stage_channels[stage_i]
            self.det_stages.append(
                nn.ModuleList(
                    [
                        NABlock(dim=c, kernel_size=stage_kernels[stage_i], head_dim=head_dim)
                        for _ in range(stage_depths[stage_i])
                    ]
                )
            )
            stride, reduction = upsamples[stage_i]
            self.upsamples.append(
                LinearPixelShuffleUpsample(in_channels=c, stride=stride, out_channels_reduction_factor=reduction)
            )

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim)

        self.context_channels = stage_channels[-1]
        noised_pixel_channels = out_channels * (patch_size**2)

        self.stage_min_tile_sizes = diffusion_tiling.all_stages_min_tile_size(stage_kernels, upsamples, stage5_kernel)
        up3_stride = upsamples[3][0]
        self.tile_min_sizes = diffusion_tiling.compute_tile_min_size(stage_kernels[3], stage5_kernel, up3_stride)
        self.tile_halos = diffusion_tiling.compute_tile_halos(
            stage_kernels[3], stage_depths[3], stage5_kernel, stage_depths[-1], up3_stride
        )
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, stage5_channels, bias=True)
        self.shared_adaln = AdaLNZero(dim=stage5_channels, t_emb_dim=t_emb_dim)
        self.diff_blocks = nn.ModuleList(
            [
                ChunkedDiffusionNABlock(
                    dim=stage5_channels,
                    kernel_size=stage5_kernel,
                    context_channels=self.context_channels,
                    head_dim=head_dim,
                )
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = nn.RMSNorm(stage5_channels, eps=1e-6)
        self.conv_out = nn.Linear(stage5_channels, noised_pixel_channels, bias=True)
        self.timestep_scale_multiplier = timestep_scale_multiplier

    def recommended_tiling_config(self, *, height: int, width: int, num_frames: int, free_bytes: int) -> TileSizeConfig:
        """Stage-4/5 overlaps + tile sizes that fit ``free_bytes`` of activations."""
        return diffusion_tiling.recommended_decode_tiling_config(
            tile_halos=self.tile_halos,
            pixel_scale=diffusion_tiling.stage4_to_pixel_scale_factors(tuple(self.upsamples[3].stride), self.patch_size),
            min_tile_size_s4=self.tile_min_sizes,
            patch_size=self.patch_size,
            height=height,
            width=width,
            num_frames=num_frames,
            free_bytes=free_bytes,
            stage5_channels=self.conv_in_x_t.out_features,
            stage4_channels=self.stage_channels[3],
            upsample_strides=tuple(tuple(u.stride) for u in self.upsamples),
            model_bytes=sum(p.numel() * p.element_size() for p in self.parameters()),
            natten_trailing_pad_latent_frames=self._natten_trailing_pad_latent_frames,
            out_channels=self.out_channels,
        )

    def _run_det_stage(self, x: torch.Tensor, stage_i: int, drop_leading_frame: bool) -> torch.Tensor:
        for block in self.det_stages[stage_i]:
            x = block(x)
        return self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)

    def forward_stages_1_to_3(self, z_noisy: torch.Tensor) -> torch.Tensor:
        """Stages 1-3 on the full (ghost-padded) latent -> channels-last stage-4 input feature."""
        z_noisy = self.per_channel_statistics.un_normalize(z_noisy)
        x = self.conv_in(z_noisy.permute(0, 2, 3, 4, 1))
        for stage_i in range(3):
            x = self._run_det_stage(x, stage_i, drop_leading_frame=True)
        return x

    def forward_stage_4(self, x: torch.Tensor, pad_trailing: bool) -> torch.Tensor:
        """Stage-4 NA blocks on one tile (the upsample is deferred into the diffusion blocks)."""
        for block in self.det_stages[3]:
            x = block(x)
        if pad_trailing:
            up_t = int(self.upsamples[3].stride[0])
            x = diffusion_tiling.crop_trailing_context_natten_pad(
                x,
                n_latent_frames=self._natten_trailing_pad_latent_frames,
                time_scale=self.video_downscale_factors.time // up_t,
                stage5_kernel_t=max(1, -(-self.stage5_kernel[0] // up_t)),
            )
        return x

    def forward_diff_step(
        self, x_t: torch.Tensor, stage4_feat: torch.Tensor, t: torch.Tensor, *, drop_leading_frame: bool
    ) -> torch.Tensor:
        """The stage-5 x0 prediction in pixel space from noise ``x_t`` and stage-4 context."""
        x = self.conv_in_x_t(patchify(x_t, patch_size_hw=self.patch_size).permute(0, 2, 3, 4, 1))
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)
        for block in self.diff_blocks:
            x = block(x, stage4_feat, modulation, self.upsamples[3], drop_leading_frame=drop_leading_frame)
        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return unpatchify(x, patch_size_hw=self.patch_size)

    def _decode_temporal_group_isolated(
        self,
        tiles: List[Tile],
        feat_s4: torch.Tensor,
        content_s4_frames: int,
        timestep: torch.Tensor,
        full_video_shape: VideoLatentShape,
        curr_temporal_slice: slice,
        generator: torch.Generator | None,
        *,
        complementary: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Decode every tile of one temporal group in isolation and blend into an fp16 accumulator."""
        group_temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start
        group_shape = full_video_shape._replace(frames=group_temporal_len)
        full_torch_shape = full_video_shape.to_torch_shape()
        accum_dtype = torch.float16 if feat_s4.dtype == torch.bfloat16 else feat_s4.dtype
        buffer = torch.zeros(group_shape.to_torch_shape(), device=feat_s4.device, dtype=accum_dtype)
        weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)
        local_temporal_slice = slice(0, group_temporal_len)

        randn_device = generator.device if generator is not None else feat_s4.device
        up3_stride = tuple(self.upsamples[3].stride)

        for tile in tiles:
            feat_tile, is_origin, pad_trailing, content_thw = diffusion_tiling.slice_stage4_tile(
                feat_s4, tile, content_frames=content_s4_frames
            )
            content_pixel_shape = diffusion_tiling.pixel_tile_shape(full_torch_shape, tile.out_coords)
            stage5_f, stage5_h, stage5_w = diffusion_tiling.stage5_pixel_shape_from_stage4(
                *content_thw,
                upsample_stride=up3_stride,
                patch_size=self.patch_size,
                stage5_kernel_t=self.stage5_kernel[0],
                drop_leading_frame=is_origin,
                pad_trailing=pad_trailing,
            )
            x_t = torch.randn(
                (content_pixel_shape[0], content_pixel_shape[1], stage5_f, stage5_h, stage5_w),
                dtype=feat_s4.dtype,
                generator=generator,
                device=randn_device,
            ).to(feat_s4.device)

            context_tile = self.forward_stage_4(feat_tile, pad_trailing=pad_trailing)
            pixel_tile = self.forward_diff_step(x_t, context_tile, timestep, drop_leading_frame=is_origin)
            pixel_tile = diffusion_tiling.crop_pixels_to_content(
                pixel_tile, content_pixel_shape[2], content_pixel_shape[3], content_pixel_shape[4]
            ).to(buffer.dtype)

            masks = tuple(m.to(device=buffer.device, dtype=torch.float32) for m in tile.masks_1d)
            local_coords = (
                tile.out_coords[0],
                tile.out_coords[1],
                local_temporal_slice,
                tile.out_coords[3],
                tile.out_coords[4],
            )
            buffer[local_coords] += scale_by_masks_1d(pixel_tile, masks)
            if weights is not None:
                strength = torch.ones(pixel_tile.shape, device=buffer.device, dtype=buffer.dtype)
                weights[local_coords] += scale_by_masks_1d(strength, masks)

        return buffer, weights

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TileSizeConfig,
        generator: torch.Generator | None,
    ) -> Iterator[torch.Tensor]:
        """Decode latent to ``(B, C, F, H, W)`` pixels in ~[-1, 1], yielding temporal chunks.

        Stages 1-3 run once on the full volume; stages 4-5 run per tile with pixel blend.
        Across temporal groups only the trailing overlap is kept between iterations.
        """
        content_shape = VideoLatentShape.from_torch_shape(latent.shape)
        content_pixel = content_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)

        latent, (_t_pad, h_pad, w_pad) = diffusion_tiling.ensure_min_latent_shape(latent, self.stage_min_tile_sizes)
        spatial_scale = (self.video_downscale_factors.height, self.video_downscale_factors.width)
        work_shape = VideoLatentShape.from_torch_shape(latent.shape)
        full_video_shape = work_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        target_shape = full_video_shape.to_torch_shape()

        strides = [tuple(u.stride) for u in self.upsamples]
        s4_t, s4_h, s4_w = diffusion_tiling.stage4_thw_from_latent(
            strides, latent.shape[2], latent.shape[3], latent.shape[4], drop_leading_frame=True
        )
        tiles = diffusion_tiling.prepare_tile_schedule(
            torch.Size([latent.shape[0], latent.shape[1], s4_t, s4_h, s4_w]),
            tiling_config,
            upsample3_stride=tuple(self.upsamples[3].stride),
            patch_size=self.patch_size,
            min_tile_size=self.tile_min_sizes,
        )

        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            latent, self._natten_trailing_pad_latent_frames
        )
        feat_s4 = self.forward_stages_1_to_3(latent_padded)
        timestep = torch.ones(latent.shape[0], dtype=torch.float32, device=latent.device)

        complementary = masks_are_complementary(tiles, target_shape)
        groups = group_tiles_by_temporal_slice(tiles)
        group_slices = [slice(*group[0].out_coords[2].indices(target_shape[2])[:2]) for group in groups]

        overlap_stub: torch.Tensor | None = None
        overlap_stub_weights: torch.Tensor | None = None

        def _finalize(buf: torch.Tensor, wts: torch.Tensor | None) -> torch.Tensor:
            if complementary:
                return buf.to(latent.dtype)
            wts = wts.clamp(min=diffusion_tiling.weight_floor(wts.dtype))
            return (buf / wts).to(latent.dtype)

        def _crop_emit(buf: torch.Tensor, wts: torch.Tensor | None, global_start: int) -> torch.Tensor | None:
            if global_start >= content_pixel.frames or buf.shape[2] < 1:
                return None
            frames_keep = min(buf.shape[2], content_pixel.frames - global_start)
            chunk = _finalize(buf[:, :, :frames_keep], None if wts is None else wts[:, :, :frames_keep])
            return diffusion_tiling.crop_pixels_to_content(
                chunk,
                frames_keep,
                content_pixel.height,
                content_pixel.width,
                h_pad=h_pad,
                w_pad=w_pad,
                spatial_scale=spatial_scale,
            )

        for gi, group in enumerate(groups):
            curr_temporal_slice = group_slices[gi]
            buffer, weights = self._decode_temporal_group_isolated(
                group,
                feat_s4,
                s4_t,
                timestep,
                full_video_shape,
                curr_temporal_slice,
                generator,
                complementary=complementary,
            )

            if overlap_stub is not None:
                overlap_len = int(overlap_stub.shape[2])
                if overlap_len > 0:
                    overlap_stub += buffer[:, :, :overlap_len]
                    buffer[:, :, :overlap_len] = overlap_stub
                    if not complementary:
                        overlap_stub_weights += weights[:, :, :overlap_len]
                        weights[:, :, :overlap_len] = overlap_stub_weights
                overlap_stub = None
                overlap_stub_weights = None

            if gi + 1 < len(groups):
                next_start = group_slices[gi + 1].start
                exclusive_len = min(max(0, next_start - curr_temporal_slice.start), buffer.shape[2])
                emitted = _crop_emit(
                    buffer[:, :, :exclusive_len],
                    None if weights is None else weights[:, :, :exclusive_len],
                    curr_temporal_slice.start,
                )
                if emitted is not None:
                    yield emitted
                overlap_stub = buffer[:, :, exclusive_len:].clone()
                if not complementary:
                    overlap_stub_weights = weights[:, :, exclusive_len:].clone()
                del buffer, weights
            else:
                emitted = _crop_emit(buffer, weights, curr_temporal_slice.start)
                if emitted is not None:
                    yield emitted
