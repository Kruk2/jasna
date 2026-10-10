# Windows HIP 缩放后端：限定范围的产品代码验收

日期：2026-09-08（北京时间）。此项不是整个 Windows 优化目标完成报告。

## 实现与默认行为

`ResizeNormalizer` 的原有公开接口、20 参数 ABI、检测预处理/后处理继续共用。
Windows AMD 增加预编译 HIP 后端，使用现有 `hip_kernel` 上下文缓存与当前流，
不引入运行时 JIT，不复制 Linux/Windows 两套业务流水线。原 CUDA 源文件未改。
离线构建只派生四个已验收的 reciprocal 表达式，重建产物必须逐字节符合固定 SHA。

`JASNA_WINDOWS_HIP_RESIZE` 默认关闭。显式启用只接纳 Windows、cuda:0、gfx1100、
Torch HIP 7.16.26354、runtime 71626354 和固定 DLL/代码对象/manifest。
不匹配的 bundle/runtime 报错；超出几何范围的检测输入继续使用原 Torch 表达式。
支持 B=1..4、C=3、源尺寸至8192、输出至640、正向且不重叠的跨行/通道/批次步长。
NVIDIA/Linux 默认选择不变。既有 HIP color 和 D3D11 resident 两个开关仍为0。

普通可见 GUI 的默认策略、冻结包资源收集和研究检测/解码后端的产品化尚未完成；
不能只设置这个开关便宣称普通 GUI 已具有完整研究路线的性能。

## 实测验收

证据目录：`D:\AI\jasna_windows_amd_dev\Temp\windows-product-hip-resize-20260908-a1`。

- `PRODUCT_RESIZE_build_A1`：受保护的离线构建成功。代码对象11968字节，
  SHA256 `3c93e066930ad74bfa90f50ffdc059b230bc3ccfcc1ead3b1aa40034d29d0a10`，
  与先前研究验收版本完全一致。
- `PRODUCT_RESIZE_probe_A1`：实际产品构造器选中 HIP；26组 FP16/FP32、letterbox、
  非默认流及 strided/sentinel 测试通过；真实8192×4096 Main10帧及左右眼 FP32
  输入与原 Torch 逐元素精确一致。输入保持不变、边界哨兵完整、模块缓存和两个
  函数实际解析、GPU同步、全卡采样器关闭均已检查。
- 整帧 FP16 微测试 eager 2.300/2.298ms，HIP 0.483/0.225ms；
  临时分配203317248字节降为1990656字节。仅两次/路线的小样本，不是整片FPS。
- `PRODUCT_RESIZE_detector_A1`：实际 RfDetr 产品构造器使用新后端，首帧两眼
  各4次 eager/HIP/HIP/eager 推断；输入、raw输出、框、掩码全精确一致。
  两眼各检测到1个框。两组使用同一 bounded-leaf/depthwise 和 stable-proposal
  诊断控制；这些控制不是此项产品改动，也不构成默认检测后端验收。

以上编译、缩放、检测运行全部串行，180秒保护上限，退出0、active0、forced0。

## 真实600帧完整流程

证据目录：`D:\AI\jasna_windows_amd_dev\Temp\windows-product-resize600-20260908-a1\native-runs\PRODUCT600_20260907T194929Z_35571dfbeaf2`。

只处理已准入的 `source-main10-native8k-first600.mp4`（600帧、10.010秒、无音频），
未访问原始长视频。沿用受保护的共享 GUI worker/Processor/session/pipeline，
保留已验收的 FRAME2、源生命周期、小YUV块、native FFmpeg callback 和研究检测后端；
仅把研究缩放构造器替换为真实产品后端。观察器只计数调用，不更换构造器或算法。

处理完成，总耗时149.766秒（约4.01fps，含启动/退出），pipeline阶段134.320秒
（约4.47fps）。缩放实际调用1200次，每帧双眼各一次。此运行不是相对原研究路线
的新速度A/B，因为研究路线此前已经使用相同内核。

主线程独立完整CPU解码验证47.016秒，600帧全部可解码、8192×4096、Main10、
yuv420p10le，帧号0..599连续，PTS/DTS严格为2100+1001*n，末PTS601699，
timebase1/60000，无缺失、重复或时间戳漂移。
输出69397397字节，SHA256
`9392bdf323753e50c4b12f77199fdbe52cd4205784a4b99fb6debb9b41d05476`。

处理/验证保护均退出0，无残留进程或强制终止。处理峰值Job6176550912字节，
低于6144MiB额度；全卡显存最小余量18208632832字节，1286次产品采样、无offload，
采样线程及reader正常关闭。主机提交/物理余量分别至少19903287296/16674304000字节。
600秒处理额度与180秒验证额度未扩张，未安装/改驱动、未运行CPU AI修复。

## 责任与结论边界

Terra worker负责纯CPU bundle/几何契约与9项测试，主线程逐文件审查并独立复跑
其9项和主线程8项集成测试（共17项）。主线程负责产品接线、离线构建器、所有
串行native试验、固定600帧准入及metadata变异测试6项和独立完整输出解码。
Terra另提交只读证据核验器及3项变异测试；主线程审查最终文件后独立复跑通过，
核对build/probe两份日志、10个当前源文件pin、bundle及全部矩阵/资源/生命周期证据。

结论：新产品后端作为默认关闭、固定匹配栈的限定功能，已通过组件精确性与真实
600帧完整流程验收。未证明跨GPU/跨SDK泛化、普通可见GUI、冻结分发包或最终修复
像素画质。此前3600帧60秒结构/音频验收仍有效，但不是本次产品代码的3600帧运行。
默认启用及完整Windows路线产品化仍需后续验收；不能报告整个性能目标完成。
