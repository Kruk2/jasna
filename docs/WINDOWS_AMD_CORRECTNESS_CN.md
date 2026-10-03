# Windows AMD 解码正确性后端

## 目的与边界

本候选只重建 Windows AMD 的既有正确性解码策略，不引入 Linux 原生
AMF 路由、NVIDIA 改动、GUI 改动或新的产品默认值。它保留
`JASNA_DECODE_BACKEND=auto` 的既有选择顺序，只在 Windows、AMD 和下表指定的
输入交集上作出明确选择。

| 条件 | `auto` 解码/上传路线 | PyAV 线程设置 |
| --- | --- | --- |
| Windows AMD H.264 8-bit | AMF host frame → 既有 ROCm 上传 | 既有 AMF 设置 |
| Windows AMD HEVC 8-bit | AMF host frame → 既有 ROCm 上传 | 既有 AMF 设置 |
| Windows AMD HEVC Main10/P010 | FFmpeg/PyAV 软件解码 → P010 规范化/上传 ROCm | `thread_count=1`，`thread_type="SLICE"` |
| Windows AMD AV1 8-bit / Main10 | FFmpeg/PyAV 软件解码 → NV12/P010 规范化/上传 ROCm | `thread_type="AUTO"` |

这里的“软件解码”只限解码阶段：`_frames_software()` 仍将已规范化的 NV12/P010
帧批量上传至 Torch 的 ROCm 设备内存，并在 GPU 上执行共享的 YUV→RGB 转换；它不是
返回 CPU tensor 的降级路线。

H.264 10-bit、其他 codec/profile，以及非 Windows AMD 平台不匹配这项策略，继续走
当前分支原有的路径。本候选当时没有把 Linux-only 兼容后端扩展到 Windows；后续独立
PR 已彻底删除该后端，Windows 路由本身不变。

## Windows ROCm 的 MMEngine 导入边界

Windows ROCm 的 Torch wheel 不提供 distributed c10d，
`torch.distributed.is_available()` 因此为 false。MMEngine 0.10.x 即使只用于
BasicVSR++ 单进程推理，也会在导入阶段访问 `ReduceOp` 和 FSDP 模块。Jasna 只在
`win32 + torch.version.hip + distributed unavailable` 的精确交集内补齐这两个导入
表面；FSDP 构造仍会显式报错，且 `torch.distributed.is_available()` 保持 false。

Linux ROCm、Windows/Linux CUDA，以及真正具备 distributed 的 Torch 环境完全不进入
该兼容层。该修复只解除 MMEngine 的模块导入阻断，不启用分布式训练、不改变 BasicVSR++
模型、checkpoint、推理路线或产品默认值。

## 选择与失败语义

- 仅 `auto` 自动应用 HEVC Main10/AV1 的 Windows AMD 软件解码策略。它会记录一条
  明确 warning，说明原因及“上传到 ROCm”。
- `JASNA_DECODE_BACKEND=pyav-hw` 保留为诊断入口，故意绕过这项自动策略，仍尝试既有
  AMF 设置；这不等同于把问题格式提升为已验证的 Windows AMF 路线。
- 显式 `pyav-sw` 仍会强制软件解码；其中 Windows AMD HEVC Main10 同样使用单线程
  slice 限制，避免绕过已知的线程稳定性约束。
- 受控软件路线无法打开或读取时，错误直接作为 `VideoDecodeError` 交给调用者。Windows
  不会尝试已删除的 Linux-only 兼容后端，也不会悄悄继续为 CPU-only 输出。
- H.264/HEVC 8-bit 的既有 AMF 打开失败会沿既有逻辑记录硬件失败 warning，并进入已有的
  软件解码加 GPU 上传路径；本候选没有把这种失败吞掉或替换为未记录的 fallback。

## 实现来源

实现仅重建下列历史提交中与 Windows 解码相关的窄逻辑，而不是恢复旧文件或
cherry-pick 整个提交：

- `fc10db2`：Windows AMD HEVC Main10/P010 自动软件解码上传；
- `f8d4a3d`：Windows AMD AV1 自动软件解码上传；
- `892801a`：Windows AMD HEVC Main10 的单线程 slice 限制。

当前实现入口在 `jasna/media/video_decoder.py` 的
`_requires_windows_amd_software_decode()` 和
`_requires_single_slice_pyav_threads()`。二者均先检查 `sys.platform == "win32"`
及 AMD vendor，因此不会改变 Linux 原生、NVIDIA 或其他平台的选择。

## 当前验证范围与 Windows 验收清单

本 Linux 工作区只能运行 mock/structural 测试：它验证平台/vendor/codec 选择、AMF
是否被跳过、线程设置、显式 `pyav-hw` 隔离，以及 Windows 软件打开失败时错误直返。
它不能证明 Windows DLL、驱动、AMF host frame、ROCm 上传或真实视频的正确性。

本次重建在 Linux 上执行：

```bash
/home/user/vr_toolbox_jasna_linux/.venv/bin/python -m pytest -q \
  tests/test_video_decoder_backends.py tests/test_video_decoder_amd_path.py
```

结果为 `40 passed, 1 skipped`；`compileall`（变更的 decoder 与测试）和
`git diff --check` 也通过。该结果只证明上述结构契约，不能替代下列 Windows 真机矩阵。

Windows 真机验收前先使用固定 runtime 预检：

```powershell
cd D:\AI\jasna_windows_amd_dev
.\scripts\run_jasna_unified_windows.ps1 -PreflightOnly
```

随后以 `JASNA_DECODE_BACKEND=auto` 分别用短真实素材运行产品入口，并记录输入、输出、
首个 decoder/frame format、日志、帧数、严格递增 PTS、时长、输出 bit depth 与关闭/取消
结果。最少需要覆盖：

| 样本 | 预期路线 | 必查项 |
| --- | --- | --- |
| H.264 Main/High 8-bit NV12 | AMF host | 帧数/PTS、GPU tensor、关闭 |
| HEVC Main 8-bit NV12 | AMF host | 帧数/PTS、GPU tensor、关闭 |
| HEVC Main10 P010 | 受控软件上传 | 单线程 slice、P010、帧数/PTS、关闭 |
| AV1 Main 8-bit | 受控软件上传 | NV12、帧数/PTS、关闭 |
| AV1 Main10 P010 | 受控软件上传 | P010、帧数/PTS、关闭 |

每个输出还必须用独立 FFmpeg/FFprobe 做严格解码、时长和帧数检查；错误、短输出、PTS
不连续、CPU tensor、AMF/FFmpeg native failure 或关闭异常均为阻断项。完成这些 Windows
真机检查前，本候选只具备 Linux 结构验证，不宣称 Windows 实机验收通过。
