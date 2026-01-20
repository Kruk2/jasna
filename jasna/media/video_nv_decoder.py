import torch
import PyNvVideoCodec as nvc
from jasna.media import VideoMetadata
from typing import Iterator

class NvidiaVideoReader:
    def __init__(self, file: str, batch_size: int, device: torch.device, stream: torch.cuda.Stream, metadata: VideoMetadata):
        self.device = device
        self.file = file
        self.stream = stream
        self.batch_size = batch_size
        self.metadata = metadata

    def __enter__(self):
        self.decoder = nvc.SimpleDecoder(
            enc_file_path=self.file,
            gpu_id=self.device.index,
            output_color_type=nvc.OutputColorType.RGBP,
            use_device_memory=True,
            decoder_cache_size=self.batch_size,
            cuda_stream=self.stream.cuda_stream,
        )

        if self.metadata.is_10bit:
            self._bayer8 = torch.tensor(
                [
                    [0, 48, 12, 60, 3, 51, 15, 63],
                    [32, 16, 44, 28, 35, 19, 47, 31],
                    [8, 56, 4, 52, 11, 59, 7, 55],
                    [40, 24, 36, 20, 43, 27, 39, 23],
                    [2, 50, 14, 62, 1, 49, 13, 61],
                    [34, 18, 46, 30, 33, 17, 45, 29],
                    [10, 58, 6, 54, 9, 57, 5, 53],
                    [42, 26, 38, 22, 41, 25, 37, 21],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            self._bayer8 = (self._bayer8 + 0.5) / 64.0
            self._y_mod8 = torch.arange(self.decoder.Height, device=self.device) & 7
            self._x_mod8 = torch.arange(self.decoder.Width, device=self.device) & 7
            t = self._bayer8[self._y_mod8][:, self._x_mod8].unsqueeze(0)
            self._dither2 = torch.floor(t * 4.0).to(torch.int32)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.decoder

    def frames(self) -> Iterator[tuple[torch.Tensor, int]]:
        frame_idx = 0
        with torch.cuda.stream(self.stream):
            while True:
                batch_size = min(self.batch_size, self.metadata.num_frames - frame_idx)
                frames = self.decoder.get_batch_frames(batch_size)
                if len(frames) == 0:
                    break

                positive_pts = [f for f in frames if f.getPTS() >= 0]
                if len(positive_pts) == 0:
                    continue

                batch_tensor = torch.empty((len(positive_pts), 3, self.metadata.video_height, self.metadata.video_width), device=self.device, dtype=torch.uint8)
                for i, f in enumerate(positive_pts):
                    batch_tensor[i] = torch.from_dlpack(f)
                    frame_idx += 1
                yield batch_tensor, [f.getPTS() for f in positive_pts]