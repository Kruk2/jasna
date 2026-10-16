# AMD 编码路径私有流修复 — 2026-10-07

## 问题

检测模型发现「无码片段」后，整体速度没有明显加快。原因不在检测，而在**编码**：
干净片段跳过修复后，流水线的限速点从修复模型转移到编码 worker。

## 根因

`jasna/media/video_encoder.py` 里，NVIDIA 编码在私有流 `new_stream` 上跑、和修复重叠；
但 AMD 分支被特判成 `current_stream`（默认流），并且每帧做一次 `self.stream.synchronize()`。

解码 / 检测 / 修复 / 编码全都跑在**同一个默认流**上，所以那次 `synchronize()` 是
**全设备屏障**：每编码一帧，就等所有已排队的 GPU 工作（包括其它线程的解码转换、
检测、修复内核）全部结束。修复越少，这个串行化越突出，于是「编码跟不上」。

当年用 `current_stream` 是为了绕 issue #252（ROCm 回收在途转换缓冲导致花屏），
但那个问题的真正修法早已落地：eager RGB→YUV 转换（`yuv_scratch.py`）只写进持久
scratch + `_packed`，**不再分配每帧临时张量**。所以切回私有流是安全的。

## 修复

`video_encoder.py` `__enter__` 的 AMD 分支：

```python
# 之前
self.stream = current_stream(self.device)
# 之后
self.stream = new_stream(self.device)
```

阻塞 D2H（`_host_yuv.copy_(packed, non_blocking=False)`）保持原样：它排空私有流后，
`_packed`/scratch 才能在下一帧复用，issue #252 的竞态不会复现。`stream.synchronize()`
现在只同步私有流（等价于等本帧转换完成），不再是全设备屏障。

## 验证

- `tests/test_video_encoder_unit.py` + `test_video_encoder_mux.py`：132 passed。
  34 个失败均在 `test_video_encoder_mux.py`，是 `_encode_synthetic_vfr` 里
  `assert enc._cuda_ctx.cuda_stream == ...`（第 692 行）这个 **NVIDIA 专属断言**在
  AMD 机上的预存失败，与本次改动无关（`_cuda_ctx` 只在 NVIDIA 分支赋值）。
- 真实 AMD 编码冒烟（`F:\Jasna\amd_cache\smoke_enc_amd.py`）：90 帧 1080p HEVC
  10-bit（quality=quality）编码 0.82 s ≈ **109 fps**，解码回读 90 帧、1920x1080 一致。

## 收益与后续

- 干净片段场景：编码 YUV 转换与修复在 GPU 上并行，不再被每帧全设备屏障串行化。
- 进一步（非必需）：`_host_yuv` 目前是单缓冲，D2H 与 AMF 编码按帧串行；改成
  2-3 个 pinned 环形缓冲 + 异步 D2H 可再重叠一档。另：流水线把视频**解码了两遍**
  （检测一遍、`blend_encode_loop` 一遍），干净片段时第二遍解码是纯开销，属架构级优化，
  需要另行评估。
