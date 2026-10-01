# Linux AMD HIP 色彩内核与 BasicVSR++ B1 MIGraphX 优化

更新时间：2026-09-04

## 结论

本次把两项已经用真实视频通过正确性与性能门的优化接入 Linux AMD 产品自动路线：

1. `gfx1100` 使用预编译 HIP code object 完成 NV12/P010 ↔ RGB 融合颜色转换；
2. FP16 BasicVSR++ 只把四个传播方向中重复的 `i > 0` 主体交给静态 B1
   Torch-MIGraphX artifact。

两项优化都不改变检测阈值、Tracker、ROI、BasicVSR++ checkpoint、输出码控、双 GOP、
Smart Render、AMF D2D 解码、`record_stream`、pinned-host owner、blocking D2H 或显存回收
合同。它们也不恢复已经删除的 rocDecode，不恢复已经裁决失败的 HIP → AMF/Vulkan
编码零拷贝实验。

## 自动选择与失败语义

HIP 色彩内核的自动门为：Linux、AMD/ROCm `cuda` device、`gfx1100`，并且
`jasna/media/yuv_to_rgb.gfx1100.hsaco` 与
`jasna/media/rgb_to_yuv.gfx1100.hsaco` 同时存在。覆盖开关：

- `JASNA_AMD_HIP_COLOR_KERNELS=auto`：默认；满足门时自动启用；
- `=0`：回到既有 Torch eager 转换；
- `=1`：强制要求 HIP 路线，平台、架构或文件不符时直接失败，不静默回退。

BasicVSR++ B1 的自动门为：Linux、AMD/ROCm、`gfx1100`、FP16，并在当前 checkpoint
同目录发现完整的 `basicvsrpp-b1-migraphx-gfx1100/` sidecar。覆盖开关：

- `JASNA_BASICVSRPP_MIGRAPHX_B1=auto`：默认；完整安装时自动启用；
- `=0`：保留完整 PyTorch eager BasicVSR++；
- `=1`：强制要求 artifact，缺失或不兼容时直接失败；
- `JASNA_BASICVSRPP_MIGRAPHX_B1_DIR`：只用于覆盖等价的已校验 artifact 目录。

artifact loader 会逐项验证 manifest/SHA、checkpoint 与语义源码 SHA、Torch/ROCm/
MIGraphX/Torch-MIGraphX 版本、GPU 名称与架构、四个 artifact 的静态输入 ABI、根
GraphModule 28 个 node 和三个 MIGraphX partition。加载时禁止 JIT/重新编译；复用的
artifact 输出在返回调用方前立即 `clone()`，避免下一次 dispatch 覆盖旧结果。RF-DETR
先初始化 Torch-MIGraphX 时，loader 会复用其已经载入且 SHA 完全相同的 native extension；
允许 artifact 目录与 Torch extension 缓存中的等内容副本路径不同，但不同内容仍直接失败。

## 实现与构建

相关产品文件：

- `jasna/media/hip_kernel.py`
- `jasna/media/yuv_to_rgb.py`
- `jasna/media/rgb_to_yuv.py`
- `jasna/restorer/basicvsrpp_migraphx_b1.py`
- `jasna/restorer/basicvsrpp_mosaic_restorer.py`

HIP code object 由现有 `.cu` 源码生成：

```bash
scripts/build_hip_code_objects.sh
scripts/build_hip_code_objects.sh gfx1100 yuv_to_rgb
```

当前提交的两个对象只接受 `gfx1100`，运行不需要 `hipcc`。BasicVSR++ 的四个 `.torch`
artifact、manifest 和 `_torch_migraphx.so` 属于编译模型/发布资产，依照仓库规则不进入
Git。当前开发机安装目录为：

```text
/mnt/D/AI/jasna_windows_amd_dev/model_weights/basicvsrpp-b1-migraphx-gfx1100
```

源码 checkout 的 checkpoint 是指向同一模型目录的软链接，因此 `Path.resolve()` 后可
自动发现该目录。正式 Linux AMD 冻结包必须额外携带两个 `.hsaco`、完整 B1 sidecar、
Torch-MIGraphX Python 包及其 manifest 匹配的 native extension。当前公开 checkout
没有开发文档提到的私有 `jasna/protection/keytool/build_nuitka.py`，因此本次不能直接
修改其资产清单；发布构建前必须在私有构建仓补齐并验证。

## 真实视频 A/B

硬件/运行边界：Linux ROCm `gfx1100`（Radeon RX 7900 XTX）、FP16、同一源、同一范围、
同一 ROI/检测/编码设置，`MIOPEN_FIND_MODE=FAST`。所有 A/B 均保持 AMF Vulkan → HIP
D2D、现有内存所有权与双 GOP 逻辑。

### 8K Main10/P010，20.103411 秒，1202 帧

源：

```text
/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-vbr-peak-20260901/source-8k-main10-20s-multigop.mp4
```

| 指标 | eager 基线 | HIP + B1 MIGraphX | 变化 |
|---|---:|---:|---:|
| 总墙钟 | 197.31 s | 164.08 s | 快 16.8% |
| BasicVSR++ restore | 121.6 s | 71.3 s | 快 41.4% |
| decode-detect decode | 70.9 s | 42.9 s | 快 39.5% |
| blend-encode decode | 71.5 s | 40.9 s | 快 42.8% |
| Torch 峰值 VRAM | 8.45 GiB | 7.60 GiB | 低约 0.85 GiB |

两版都是 1202/1202 帧、20.103411 秒；PTS 序列完全一致，DTS 严格递增且一致，关键帧
位置均为 0/250/500/750/1000，FFmpeg 软件严格解码通过。两个 reader 的 AMF D2D
均为 1202/1202，host/map/staging/D2H/bridge 五类禁止项均为 0。输出间 PSNR
37.75 dB、SSIM 0.9955；差值主要由旧 eager 基线一次异常首帧拖低，新 HIP 路线连续三次
首帧均正常。该旧首帧异常不作为新路线质量回归。

### 8K Main10/P010，200 帧

总墙钟 `71.54 → 65.31 s`（快 8.7%），restore `17.4 → 13.6 s`；200/200 帧和严格
解码通过。输出间 PSNR 54.88 dB、SSIM 0.9985，与源码的质量指标基本不变。

### 3840×2160 HEVC Main/NV12，300 帧

总墙钟 `29.44 → 28.61 s`（快 2.8%）；restore `11.8 → 9.3 s`，decode-detect decode
`3.3 → 2.4 s`，blend/write `6.7 → 6.1 s`，整卡峰值 `8.29 → 7.78 GiB`。两版均为
300/300 帧、10.000 秒、NV12，严格解码通过。相对源码 PSNR 为 35.1131 dB 与
35.1174 dB，优化版没有质量下降。总收益较小是因为样片只有 10 秒，模型/引擎启动开销
占比高；不把这一短样片数字外推为长片总加速。

## 自动默认产品冒烟

在不设置四个实验/目录环境变量的全新进程中先确认：

- `YuvToRgbConverter.uses_kernel=True`；
- `RgbToYuvConverter.uses_kernel=True`；
- BasicVSR++ 自动加载模型目录的 B1 artifact；
- loader 没有现场 JIT。

随后用统一 FFmpeg 8/AMF runtime 对 3840×2160 HEVC Main/NV12 真实样片跑完整产品流程：

```text
/mnt/D/AI/amf-unified-work/transactions/jasna-ubuntu-amf-host-native-20260903/jasna/assets/test_clip1_2160p.mp4
```

最终树结果为墙钟 26.95 秒、300/300 帧、10.000 秒、3840×2160 NV12；FFmpeg `-xerror`
软件严格解码输出为空。两条 AMF D2D audit 都为 300/300，五类禁止回退计数全部为 0；
整卡峰值 6369 MiB，无 offload、pressure episode 或 critical reclaim。证据目录：

```text
/tmp/jasna-auto-product-smoke-final-GaDOiFRO
```

该 smoke 只证明自动产品路由、媒体完整性与资源合同，不替代上面的同条件 A/B 性能门。

## 测试与已知限制

两个保留的开发探针用于后续源码/runtime 变更时重复做同条件门，不接产品路由。颜色
探针会在临时目录重新编译 HIP 对象，并把 oracle 显式锁定到 Torch eager，避免产品
`auto` 让基线失真；B1 探针同时运行当前 eager 与已校验 artifact：

```bash
python scripts/probe_amd_hip_color_kernels.py \
  --height 4096 --width 8192 --rounds 8 --output /tmp/hip-colour.json

python scripts/probe_basicvsrpp_migraphx_b1.py \
  --artifact-dir /path/to/basicvsrpp-b1-migraphx-gfx1100 \
  --weights model_weights/lada_mosaic_restoration_model_generic_v1.2.pth \
  --input /path/to/real-video.mp4 --frames 60 --rounds 5 \
  --output /tmp/basicvsrpp-b1.json
```

最终树又以 64×96/一轮执行颜色探针：24 个正确性组合全部通过；以真实 4K 视频的
两帧 ROI/一轮执行 B1 探针：四个方向各实际 dispatch 两次，`max_abs=0.00048828125`、
`mean_abs=0.0000130873`、allclose 通过，四个 artifact 均保持 28 nodes/3 partitions。
这些极短复验只验证探针和合同本身，不作为长片性能数字。

聚焦回归覆盖开关解析、平台/架构/文件门、缺失资产 fail-closed、静态 ABI、重复 output
所有权、所有 BT.601/709/2020 的 full/limited 8/10-bit 颜色组合，以及产品 restorer
dispatch/close：

```bash
MIOPEN_FIND_MODE=FAST python -m pytest -q \
  tests/test_hip_kernel.py \
  tests/test_hip_colour_kernel_product.py \
  tests/test_basicvsrpp_migraphx_b1_product.py \
  tests/test_basicvsrpp_mosaic_restorer.py \
  tests/test_rgb_to_yuv_kernel.py \
  tests/test_yuv_to_rgb.py \
  tests/test_yuv_scratch_reuse.py
```

最终树聚焦结果为 `84 passed`。完整 pytest 结果为 `2517 passed, 36 skipped,
169 failed, 6 errors`；失败清单不含本次新增/修改的测试，仍集中在当前 AMD venv 不含
TensorRT、统一 FFmpeg 的精简 codec 集不含测试用 FFV1/libx264/libx265 等 encoder、
以及 NVIDIA/GUI 环境假设的既有基线，未伪造为通过。最终提交前另执行 compileall 与
diff check。

当前限制：

- HIP AOT 仅验收并自动用于 Linux AMD `gfx1100`；其他 AMD 架构保留 Torch eager；
- B1 MIGraphX 仅接受当前 checkpoint、语义源码、FP16、GPU 与 runtime 的精确 manifest
  组合；任一版本变化都必须重新生成和验收 artifact；
- Windows AMD、NVIDIA、FP32、H.264/HEVC/AV1 的编码器选择和双 GOP 资格门均未被本功能
  修改；源 codec/bit depth 不决定 BasicVSR++ B1 是否适用，输出像素格式只决定选用
  NV12 还是 P010 色彩内核；
- 不允许把 B1 `.torch`、native extension、模型、生成视频或日志提交进 Git。
