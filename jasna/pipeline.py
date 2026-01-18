from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.mosaic import Detections


class Pipeline:
    def __init__(
        self,
        *,
        input_video: Path,
        output_video: Path,
        detection_model,
        restoration_pipeline,
        stream: torch.cuda.Stream,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.input_video = input_video
        self.output_video = output_video
        self.detection_model = detection_model
        self.restoration_pipeline = restoration_pipeline
        self.stream = stream
        self.batch_size = int(batch_size)
        self.device = device

    def run(self) -> None:
        stream = self.stream
        metadata = get_video_meta_data(str(self.input_video))

        with (
            NvidiaVideoReader(str(self.input_video), batch_size=self.batch_size, device=self.device, stream=stream) as reader,
            NvidiaVideoEncoder(str(self.output_video), device=self.device, stream=stream, metadata=metadata, stream_mode=False) as encoder,
            torch.inference_mode(),
            torch.cuda.stream(stream),
            tqdm(total=reader.total_frames if reader.total_frames else None, unit="frame", dynamic_ncols=True) as pb,
        ):
            target_hw = (int(reader.decoder.Height), int(reader.decoder.Width))

            for frames, pts_list in reader.frames():
                effective_bs = len(pts_list)
                if effective_bs == 0:
                    continue

                frames_eff = frames[:effective_bs]
                if effective_bs < self.batch_size:
                    pad = frames_eff[-1:].expand(self.batch_size - effective_bs, -1, -1, -1)
                    frames_in = torch.cat([frames_eff, pad], dim=0)
                else:
                    frames_in = frames

                detections: Detections = self.detection_model(frames_in, target_hw=target_hw)
                restored = self.restoration_pipeline.restore(frames_in, detections)[:effective_bs]

                for i in range(effective_bs):
                    encoder.encode(restored[i], int(pts_list[i]))

                pb.update(effective_bs)

