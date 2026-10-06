# AMD secondary upscaling (`--secondary-restoration amd-upscale`)

Host: MSI PRO B650M-P, Ryzen 7 7800X3D, **RX 7900 XT** (20 GB), Windows,
`torch 2.9.0+rocmsdk20251116` (HIP 7.1), FFmpeg 8.1.2 (gyan, `--enable-amf`).
Secondary restoration upscales each restored 256x256 crop and the blend stage
resamples the result back, exactly like RTX Super Res on NVIDIA.

## End-to-end (Jasna pipeline)

Clip: 10 s / 300 frames / 640x480 H.264 with two mosaic-like blocks, detection
forced with `--detection-score-threshold 0.01 --max-clip-size 60`, which yields
**12 clips / 348 crops**. Secondary restoration runs concurrently with detection,
so it barely moves the wall clock:

| Secondary restoration | wall | `[timing] secondary` restore | upscale throughput | GPU util (avg / peak) |
| --------------------- | ---: | ---------------------------: | -----------------: | --------------------: |
| `none`                | 49.9 s | 0.0 s | — | 128 % / 246 % |
| `amd-upscale` amf-sr 4x (`sr1-0`) | 49.8 s | 6.8 s | 55 fps | 122 % / 223 % |
| `amd-upscale` amf-sr 2x (`sr1-0`) | 45.4 s | 4.8 s | 76 fps | 63 % / 138 % |
| `amd-upscale` amf-sr 4x (`bicubic`) | 45.4 s | 6.5 s | 58 fps | 66 % / 144 % |

GPU utilisation is the sum of the 7900 XT's engine counters (`luid_0x13C63`),
sampled every 500 ms; the adapter was verified separately with
`AMFDeviceDX11Impl ... deviceID=0x744c` and a torch GEMM control run.

## Micro-benchmarks (256x256 crops, one FFmpeg process per clip)

| Engine | Scale | Output | Throughput |
| ------ | ----: | -----: | ---------: |
| `amf-sr` (`sr1-0`) | 4x | 1024x1024 | **59-60 fps** |
| `amf-sr` (`sr1-0`) | 2x | 512x512 | **91-94 fps** |
| ~~`libplacebo` (`ewa_lanczos`)~~ (engine removed) | 4x | 1024x1024 | 31 fps |
| `realesrgan` (Real-ESRGAN RRDBNet x4plus, fp16) | 4x | 1024x1024 | 16.5 fps |

Real-ESRGAN engine batch sizes (same shapes): batch 1 → 15.7 fps, batch 2 → **16.5 fps**,
batch 4 → 0.24 fps, batch 8 → 13.6 fps. Batch 4 makes MIOpen pick a pathological
kernel, hence the hard-coded default of 2.

## Notes for whoever repeats this

* **AMF on Windows is D3D11, not Vulkan.** FFmpeg logs
  `AMF initialisation succeeded via D3D11` and creates the device on
  `deviceID=0x744c` (the dGPU, not the Raphael iGPU). AMF is AMD's media
  framework and has no ROCm build; on Linux it rides Vulkan/OpenCL instead.
* **One FFmpeg process per clip.** The AMF filters buffer the tail of the stream
  inside the AMF component and only release it when the input ends — FFmpeg's own
  source notes the AMF filters ran off a `filter_frame` callback "which has no way
  to tell a component that no more input is coming". Reusing one process across
  clips therefore stalls exactly one frame short of every clip (observed: 53/54
  frames, restore=125 s). Closing stdin per clip flushes the tail deterministically
  and also gives the restorer a hard timeout via `subprocess`.
* The first `realesrgan`-engine `restore()` pays MIOpen algorithm autotuning
  (≈25 s cold, cached afterwards).
