import torch
import python_vali as vali
from typing import Iterator
import av

class NvidiaVideoReader:
    def __init__(self, file: str, batch_size: int | None, device: torch.device, stream: torch.cuda.Stream):
        self.device = device
        self.file = file
        self.stream = stream
        self.batch_size = batch_size

    def __enter__(self):
        self.decoder = vali.PyDecoder(
            self.file,
            {},
            gpu_id=self.device.index,
            stream=self.stream.cuda_stream,
        )
        container = av.open(self.file)
        stream = container.streams.video[0]
        self.duration = stream.duration
        self.timebase = stream.time_base
        self.average_rate = stream.average_rate
        self.guessed_rate = stream.guessed_rate
        self.total_frames = stream.frames
        container.close()
        self.is_vfr = self.decoder.IsVFR
        self.rgb_planar_surface = vali.Surface.Make(
            format=vali.PixelFormat.RGB_PLANAR if self.decoder.Format == vali.PixelFormat.NV12 else vali.PixelFormat.RGB10_PLANAR,
            width=self.decoder.Width,
            height=self.decoder.Height,
            gpu_id=self.device.index)
        self.py_cvt = vali.PySurfaceConverter(gpu_id=self.device.index, stream=self.stream.cuda_stream)
        self.decode_surface = vali.Surface.Make(
            format=self.decoder.Format,
            width=self.decoder.Width,
            height=self.decoder.Height,
            gpu_id=self.device.index,
        )
        self.rgb_surface = vali.Surface.Make(
            format=vali.PixelFormat.RGB if self.decoder.Format == vali.PixelFormat.NV12 else vali.PixelFormat.RGB10,
            width=self.decoder.Width,
            height=self.decoder.Height,
            gpu_id=self.device.index)
        
        if self.decoder.Format == vali.PixelFormat.P10:
            self.nv12_surface = vali.Surface.Make(
                format=vali.PixelFormat.NV12,
                width=self.decoder.Width,
                height=self.decoder.Height,
                gpu_id=self.device.index)
        else:
            self.nv12_surface = self.decode_surface

        color_space = vali.ColorSpace.BT_709 if self.decoder.ColorSpace == vali.ColorSpace.UNSPEC else self.decoder.ColorSpace
        color_range = vali.ColorRange.MPEG if self.decoder.ColorRange == vali.ColorRange.UDEF else self.decoder.ColorRange
        self.cc_ctx = vali.ColorspaceConversionContext(color_space, color_range)

        if self.decoder.Format == vali.PixelFormat.P10:
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

    def frames(self, frame_seek: int|None=None) -> Iterator[tuple[torch.Tensor, int]]:
        frame_idx = 0
        pkt_data = vali.PacketData()
        eof = False
        with torch.cuda.stream(self.stream):
            while True:
                batch_tensor_nv = torch.empty((self.batch_size, 3, self.decoder.Height, self.decoder.Width), device=self.device, dtype=torch.uint8)
                pkts = []
                seek_ctx = None if frame_seek is None else vali.SeekContext(seek_frame=frame_seek)

                for i in range(self.batch_size):
                    success, details = self.decoder.DecodeSingleSurfaceAsync(self.decode_surface, pkt_data, seek_ctx)
                    if not success:
                        if details.name == 'END_OF_STREAM':
                            eof = True
                            break
                        raise Exception(details)

                    self.py_cvt.RunAsync(self.decode_surface, self.rgb_surface, self.cc_ctx)
                    self.py_cvt.RunAsync(self.rgb_surface, self.rgb_planar_surface)

                    tensor_nv = torch.from_dlpack(self.rgb_planar_surface)
                    if tensor_nv.dtype == torch.uint16:
                        frame10 = (tensor_nv.to(torch.int32) >> 6)
                        tensor_nv = ((frame10 + self._dither2) >> 2).clamp(0, 255).to(torch.uint8)
                    batch_tensor_nv[i].copy_(tensor_nv)

                    frame_idx += 1
                    pkts.append(pkt_data.pts)
                yield batch_tensor_nv, pkts
                if eof:
                    break