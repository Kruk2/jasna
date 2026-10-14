# AMD 检测速度诊断与修复（无码片段 ~50 FPS → ~100 FPS）— 2026-10-07

## 现象

检测模型跑无马赛克片段时只有 ~50 FPS，N 卡同场景 100~120 FPS。

## 实测定位

环境：RX 7900 XT，ROCm 10.1 torch 2.14，rfdetr-v6 @480，fp16，GUI 默认 B4。
（AOTriton 镜像存在、flash SDP 已启用——排除 attention 回退假设。）

### 1. 模型级基准（1080p 帧，含预处理/推理/后处理）

| 引擎 | 批次 | 优化前 | 优化后 |
|---|---|---:|---:|
| torch | 8 | 10.50 ms/f（95 fps）| 不变 |
| torch | 4 | 11.81 ms/f（85 fps）| 不变 |
| migraphx | 8 | 13.68 ms/f（73 fps）| **7.24 ms/f（138 fps）** |
| migraphx | 4 | 13.24 ms/f（75 fps）| **7.03 ms/f（142 fps）** |

### 2. 串行 decode+detect（复现真实扫描路径，594 帧真实片段）

| 引擎 | 批次 | 优化前 | 优化后 |
|---|---|---:|---:|
| migraphx | 4 | **58.9 fps** | **100.4 fps** |
| torch | 4 | 65.0 fps | 65.0 fps（未动）|

用户的 ~50 FPS 即 migraphx b4 的串行扫描路径（MIGraphX 是 AMD 默认引擎）。

### 3. 逐段分解（migraphx b4，优化前）

| 阶段 | 耗时/次 |
|---|---:|
| 输入 D2H | 0.9 ms |
| session.run（纯 GPU 推理仅 5.4 ms/帧）| 21.1 ms |
| **输出 H2D（masks 23 MB！）** | **25.8 ms** |

## 根因

MIGraphX runner 每次调用把 **23 MB 的 masks 张量（4×200×120×120）走 GPU→CPU→GPU
一个来回**：推理在 GPU 完成 → 输出 D2H 回 CPU → `astype` 再复制 → H2D 推回 GPU
→ 在 GPU 上只做一个阈值判断，99.9% 的数据当场丢弃。仅这一趟 PCIe 就吃掉
~26 ms/次，比推理本身（21.5 ms/次）还贵，把理论 ~140 fps 拖到 59 fps。
（N 卡走 TensorRT 引擎，无此开销，故 100~120。）

torch 引擎不受影响（推理始终在 GPU 上），它慢在 ROCm 上 RF-DETR 推理本身约
12 ms/帧（约为 TensorRT 的 2 倍），这是另一个层面的差距。

## 修复（3 处小改动）

1. `rfdetr_migraphx_runner.py infer()`：masks 不再 `.astype(float32)+.to(device)`，
   以 `torch.from_numpy` 零拷贝 CPU 视图返回（fp16 原样）；dets/labels 照旧上 GPU。
2. `rfdetr.py _postprocess()`：gather/阈值在 masks 自己的设备上做（CPU），
   只把阈值后存活的少量 (N_i, 120, 120) bool 掩码传回 GPU（无码帧为 0 个）。
   torch 路径（GPU masks）行为不变。
3. `rfdetr.py scan_scores_masks()`（GUI「扫描视频」快速路径）：同样在 CPU 做
   阈值 + 合并，仅把 (B, mask_h, mask_w) 的小结果传回 GPU。

## 验证

- 单测：`test_detection_engine.py` 全过；`test_rfdetr_postprocess.py` 13 passed
  （2 个失败为环境性：`jasna.trt` 遍历、tmp 权重缺失，与本改动无关）；
  `test_detection_registry.py` 的失败均为厂商后缀断言（本机 AMD vs 测试假设
  NVIDIA）的预存问题，与本次改动无关。
- 复测数据见上表：migraphx b4 串行扫描 **58.9 → 100.4 fps**，达到 N 卡水平。

## 备注

- torch 引擎的 RF-DETR 推理在 ROCm 上约 12 ms/帧（TensorRT 约 5-6），差距来自
  后端本身，不是本次流水线问题；MIGraphX 引擎（AMD 默认）优化后已反超。
- 本修复应同步进 PR #475（`amd-migraphx-detection-engine` 分支），待确认后提交。
- 脚本：`D:\Strata-data\bench_det_amd.py`、`bench_scan_loop.py`。
