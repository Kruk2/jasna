# BasicVSR++ 主模型优化：HIP Graph 回放 + 网格缓存（2026-10-08）

## 背景

AMD 路径下主修复模型一直是**纯 eager** 运行（TRT 子引擎/CUDA Graph 仅 NVIDIA）。
此前结论：MIGraphX 引擎无正收益（launch-bound）、torch.compile 无正收益（用户实测）。
本次用 profiler 重新定位 eager 前向的真实瓶颈。

## 定位（torch.profiler，T=60 fp16，RX 7900 XT）

- 每次前向约 **17 万次算子调用** —— 派发/启动受限；
- **`flow_warp` 每次调用都重建单位网格**（`eye(2,3)` + `affine_grid`），
  每次前向约 1400 次，`affine_grid_generator + eye + linspace + repeat`
  合计约占设备时间 **25%** —— 而基准网格完全不依赖输入数据；
- SPyNet 光流缩放每次前向做一次 pageable H2D 拷贝（`torch.tensor([...], device=cuda)`）；
- torchvision `deform_conv2d` 仅 ~9%，不是瓶颈。

## 三项优化

1. **`flow_warp` 单位网格缓存**（`mmagic/flow_warp.py`）：
   按 `(n, h, w, dtype, device)` 键控，`no_grad` 下构建一次。
2. **SPyNet 光流缩放张量缓存**（`mmagic/basicvsr_plusplus_net.py`）：
   消除每次前向的 pageable H2D（该项同时是图捕获的阻断点）。
3. **HIP Graph 回放**（`restorer/basicvsrpp_mosaic_restorer.py`）：
   在**构造时**（流水线线程启动前）为两种主导 clip 长度
   （`max_clip_size` 与 `max_clip_size-2*temporal_overlap`）各捕获一张图，
   前向改为「拷贝静态输入 → replay → clone 静态输出」；
   其余长度（片尾/切镜头片段）自动走 eager。
   `JASNA_BV_HIP_GRAPH=0` 可关闭；NVIDIA TRT 路径不受影响。

## 实测（RX 7900 XT，fp16，256×256）

### 微基准

| 场景 | eager | graph 回放 | 提升 |
|---|---:|---:|---|
| T=60 | 534.9 ms | 290.2 ms | **-46%** |
| T=90 | 839.1 ms | 491.7 ms | **-41%** |

网格缓存单独贡献：T=60 585.4→534.9 ms（-8.6%）、T=90 927.9→839.1 ms（-9.6%）。

### 全路径 A/B（600 帧 1080p，含 10 秒马赛克区间，同参数开关对比）

| 指标 | Graph OFF | Graph ON |
|---|---:|---:|
| primary restore | 11.4s | **2.5s（-78%）** |
| 稳态流水线阶段 | ~16.5s | **~9.3s（-44%）** |
| 一次性启动预热 | — | +3~4s |

### 被否掉的路线（实测数据）

| 变体 | 结果 | 判定 |
|---|---|---|
| torch.compile | 用户此前实测无正收益（inductor 缓存存在但无产出） | 跳过 |
| channels_last | T=60 711.8 ms（基线 585.4） | **-22%，拒绝** |
| cudnn.benchmark=True | ~1-2% | 边际，不启用 |
| MIGraphX EP | launch-bound，无正收益（前次已证） | 维持 |

### 质量验证

- 网格缓存：与逐次构建的网格**位级一致**（多形状断言）；
- Graph 回放：与 eager 输出 `max|diff| = 0.000000`（T=60/90/74）；
- 全路径输出：开关 Graph 两个输出 **PSNR = inf、0 帧差异**；
- 相关单测 29 passed，无回归。

## 关键工程教训

1. **流水线中捕获不安全**：`capture_error_mode="thread_local"` 在本 ROCm 构建
   （torch 2.14+rocm10.1）下，捕获窗口内其它线程的 CUDA 调用会以
   `operation not permitted when stream is capturing` 直接中止（实测打死
   decode/blend 线程）。捕获必须在流水线线程启动前完成。
2. pageable H2D 拷贝（`torch.tensor([...], device=cuda)`）是图捕获的阻断点，
   也是每次前向的纯开销 —— 这类"Python 常量 → 设备张量"都应缓存。
3. HIP Graph 在 Windows ROCm 上可用且收益巨大（-45%），与 torch.compile
   的失败结论互不矛盾：前者消除派发开销，后者在 ROCm Windows 上收益有限。

## 状态

- 4 个文件（flow_warp.py、basicvsr_plusplus_net.py、basicvsrpp_mosaic_restorer.py、
  session_factory.py）已提交并推送到 PR #475（`be10baf`）；
- 便携版追加补丁包已更新（`E:\JasnaDist\Jasna-AMD-Windows-补丁-20261008.7z`，
  10 个文件含说明）；
- 全链路剩余瓶颈：decode-detect 稳态 ~9s（decode 3.5s + detect 5.6s）、
  blend-encode 稳态 ~9.3s（decode 2.7s + write 0.9s + queue-wait 3.2s），
  修复已不再是最慢环节。
