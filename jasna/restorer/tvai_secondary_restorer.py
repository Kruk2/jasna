from __future__ import annotations

import torch


def _parse_tvai_args_kv(args: str) -> dict[str, str]:
    args = (args or "").strip()
    if args == "":
        return {}
    out: dict[str, str] = {}
    for part in args.split(":"):
        part = part.strip()
        if part == "":
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --tvai-args item: {part!r} (expected key=value)")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "":
            raise ValueError(f"Invalid --tvai-args item: {part!r} (empty key)")
        out[k] = v
    return out


class TvaiSecondaryRestorer:
    name = "tvai"
    num_workers = 1
    _INPUT_SIZE = 256

    def __init__(self, *, ffmpeg_path: str, tvai_args: str, scale: int, num_workers: int) -> None:
        self.ffmpeg_path = str(ffmpeg_path)
        self.tvai_args = str(tvai_args)
        self.scale = int(scale)
        self.num_workers = int(num_workers)
        if self.scale not in (1, 2, 4):
            raise ValueError(f"Invalid tvai scale: {self.scale} (valid: 1, 2, 4)")
        kv = _parse_tvai_args_kv(self.tvai_args)
        parts: list[tuple[str, str]] = []
        if "model" in kv:
            parts.append(("model", kv["model"]))
        parts.append(("scale", str(self.scale)))
        for key, value in kv.items():
            if key in {"model", "scale", "w", "h"}:
                continue
            parts.append((key, value))
        self.tvai_filter_args = ":".join(f"{key}={value}" for key, value in parts)

    def build_ffmpeg_cmd(self) -> list[str]:
        size = f"{self._INPUT_SIZE}x{self._INPUT_SIZE}"
        return [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            size,
            "-r",
            "25",
            "-i",
            "pipe:0",
            "-sws_flags",
            "spline+accurate_rnd+full_chroma_int",
            "-filter_complex",
            f"tvai_up={self.tvai_filter_args}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        raise NotImplementedError("TVAI integration is disabled")

    def close(self) -> None:
        pass
