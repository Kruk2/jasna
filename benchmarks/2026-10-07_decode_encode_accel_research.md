# 视频解码/编码加速调研（AMD / Windows）— 2026-10-07

问题：解码、编码两端还能不能加速？新出的 rocDecode 值不值得上？

## 结论先行

1. **rocDecode / rocPyDecode：Windows 上是死路**。rocDecode 1.10.0（2026-10-05 文档）
   明确标注 "Applies to Linux"，HIP 互操作走 VA-API（Linux 显示栈）；rocPyDecode
   绑定也是 Linux。与此前 45 个 PR 分析中 #10/#41 的结论一致，不用再惦记。
2. **1080p 解码不是瓶颈，实测四种后端全部一样快**：软件 / hevc_amf / d3d11va /
   vulkan hwaccel 解 1080p HEVC 10-bit 都是 **~600 fps（20x）**，7800X3D 的软解完全
   跟得上（对比 e2e 流水线只有 30-40 fps）。解码换成什么都改变不了墙钟。
   （d3d12va 也能跑，20x；4K/8K 源上硬解才有意义，而那时 AMF 已在用。）
3. **编码端找到一处真实收益：D3D11 零拷贝直喂 AMF 编码器**（ffmpeg CLI 场景）：
   `-hwaccel d3d11va -hwaccel_output_format d3d11 -i in -c:v hevc_amf`
   vs 系统内存路径：**CPU 时间 6.0 s → 0.25 s（24 倍），墙钟快 ~25-30%**
   （602 帧 1080p10bit：rtime 1.34 s → 0.88-1.08 s）。d3d12va 路径 Conversion
   failed，不可用。
4. 但 jasna 的重编码输入是 **HIP 显存里的修复帧**（GPU 上 `RgbToYuvConverter` 转
   YUV → PyAV 下载到 CPU → AMF 编码），不是解码器输出的 D3D11 帧。要把零拷贝搬进
   主流水线，唯一路线是 **HIP↔D3D11 互操作**（`hipExternalMemoryHandleTypeD3D11Resource`
   / `D3D11ResourceKmt`，HIP 官方在 Windows 支持）把显存帧导入 D3D11 纹理再交 AMF
   —— 这是一个 native 工程，不是配置改动。
5. **当前流水线里解码+编码合计占比太小**：600 fps 解码、~690 fps AMF 编码，e2e
   30-40 fps，且阶段间流水重叠——解码/编码端任何改进对墙钟的期望收益都在
   个位数百分比。真正的墙还是修复模型本身（torch 路径）。

## 实测数据

片段：`F:\Jasna\amd_cache\ab_e2e_migraphx_1.mkv`（1080p HEVC 10-bit，602 帧，
9.94 s，gyan full build ffmpeg 8.1.2，RX 7900 XT，Adrenalin 32.0.31041）。

### 解码吞吐（`-benchmark`，各 2 次）

| 方式 | speed | 备注 |
|---|---:|---|
| 软件解码 | 20.2-20.6x | ~600 fps |
| `-c:v hevc_amf` | 18-19.2x | 无优势 |
| `-hwaccel d3d11va` | 18.1-18.6x | 无优势 |
| `-hwaccel d3d12va` | 21.5x | 可用 |
| `-hwaccel vulkan` | 21x | **能跑**（FFmpeg 8 Vulkan video decode 在 AMD Windows 驱动上工作）|

### 编码吞吐（hevc_amf，rc=cqp 25，602 帧）

| 输入路径 | utime+stime (CPU) | rtime (墙钟) |
|---|---:|---:|
| 系统内存（现状等效） | ~6.0 s | 1.34-1.38 s |
| **D3D11 零拷贝**（quality 档） | **0.23 s** | **1.08 s** |
| **D3D11 零拷贝**（speed 档） | **0.25 s** | **0.88 s** |
| D3D12 零拷贝 | — | 失败 |

CPU 时间的差别就是省掉的 `yuv420p10le→nv12` 软件转换 + 每帧上传；零拷贝路径里
解码输出直接以 D3D11 纹理进 AMF。

## 社区经验核对

| 途径 | 状态 |
|---|---|
| rocDecode / rocPyDecode / rocJPEG | Linux 专属（官方文档 "Applies to Linux"），Windows 无移植 |
| rocDecode 的 HIP 互操作 | 走 VA-API（Linux）；Windows 等价物是 D3D11 互操作，HIP 官方支持导入 D3D11/D3D12 资源 |
| FFmpeg Hardware/AMF wiki（2026-01 更新） | 推荐 `-hwaccel d3d11va -hwaccel_output_format d3d11` 全硬件链路，正是本次实测的零拷贝方案 |
| FFmpeg 8 "Huffman" | Vulkan video decode 已进主线；gyan full build 带 `--enable-vulkan --enable-libshaderc`，AMD Windows 驱动可用 |
| "FFmpeg-AMD-Optimized"（GitHub） | RDNA2/3 优化第三方构建，主要卖点还是 AMF 完整性，我们用 gyan full 已覆盖 |

## 可落地的动作

1. **立即（低风险）**：`amd-upscale` 二级修复（`sr_amf` ffmpeg CLI 链路）与未来任何
   「解码→AMF 编码」的 ffmpeg 子进程场景，套用
   `-hwaccel d3d11va -hwaccel_output_format d3d11` 零拷贝链（前提：源是 h264/hevc
   且目标编码器是 AMF）；CPU 占用大幅下降，多任务时收益最大。
2. **中期（native 工程，需要再评估收益）**：主流水线零拷贝需要
   HIP↔D3D11（修复帧 HIP 张量 → D3D11 纹理 → PyAV/AMF 硬编码帧），省掉
   GPU→CPU→GPU 往返与逐帧上传；官方 API 在 Windows 可用，但 PyAV 侧要自建
   hwframe 封装，工作量大，且 e2e 期望收益个位数百分比。
3. **不建议**：为 Windows 找 rocDecode 替代、换 Vulkan 解码（无速度差）、d3d12va
   进 AMF（不支持）。

原始日志：`F:\Jasna\amd_cache\dec_*.log`、`enc_*.log`；脚本
`bench_decode.bat`、`bench_encode.bat`。
