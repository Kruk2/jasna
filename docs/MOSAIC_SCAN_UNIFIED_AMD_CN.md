# 自动粗扫的统一 Linux AMD 路由

更新时间：2026-08-29

GUI 段落编辑器的整片粗扫和单帧 mask preview 都直接构造共享
`NvidiaVideoReader`，不传 scan-specific backend，也没有独立解码分支。因此在合格
Linux AMD H.264/HEVC 上，它们会随产品 `auto` 使用 AMF Vulkan → HIP
private-deferred D2D；AV1 同样遵守共享 reader 的 stable-cache/native gate 与普通 PyAV
边界。

检测模型同样只通过共享 `build_detection_model` 构造。Linux AMD gfx1100、FP16、
`rfdetr-v6` 且同目录存在已安装 manifest 时，registry 自动选择 RF-DETR MIGraphX；否则
保留原 PyTorch/ROCm detector。粗扫不复制 MIGraphX 判定、不修改阈值、Tracker、mask、
VR SBS adapter 或 Pipeline，也不会为迁移路线引入兼容层。

contract test 固定以下边界：

- whole-video scan 和 preview 都不出现 `decode_backend` 特例；
- scan 把 B4/B8、FP16、模型名、权重与既有 `SCAN_SCORE_FLOOR` 原样交给共享 registry；
- MIGraphX manifest 的发现与 fail-closed 仍由已验收的 registry/runner 唯一负责。

因此本阶段的“迁移”是让粗扫自然消费统一产品后端，而不是另写一套粗扫解码或检测实现。

精扫的调度与后端选择分开：reader 仍不传 `decode_backend`，继续使用共享 `auto`；但当
共享能力门已经确认输入会走 Linux AMD 原生 AMF D2D、视频不少于 3000 万像素且时长
不少于 10 秒时，精扫把时间轴等分给两条共享 reader，以同时使用 RX 7900 XTX 的两套
媒体解码引擎。Linux AMD 的 4K/短片/非原生格式、Windows AMD，以及其他既有单 reader
边界均不变；NVIDIA 原有的 4K 双 reader 门也不变。这不是恢复 rocDecode，也没有复制或
改变 `auto` 的 codec/profile/platform 判定。

## Linux AMD 实片验收

在 Ubuntu AMD gfx1100 上使用 4096×2048 H.264 High 8-bit、120 帧实片执行共享产品
`auto` 粗扫，按约 0.5 秒间隔抽取 4 帧，全部正常完成：

- detector：`RfDetrMosaicDetectionModel`；
- runner：`RfDetrMigraphxRunner`；
- engine：`rfdetr-seg-medium.static-b1.dot-projector-fp16-gfx1100.mxr`；
- mask shape：4 份 `[4, 90, 160]`；
- GPU 最高 junction 62°C、memory 65°C；
- 本次运行窗口未发现 GPU reset、ring timeout、page fault 或 OOM。

这项验收同时证明粗扫从共享 reader 取得统一 AMF D2D 解码，并从共享 registry 取得
RF-DETR MIGraphX 产品选择；没有建立 scan-only 的 AMD 分支。

## 8K 精扫双 reader 性能验收

2026-08-30 使用同一份 60.06 秒、3600 帧、8192×4096、60000/1001 fps 的 HEVC
Main10/P010 实片，固定 0.5 秒间隔、RF-DETR MIGraphX FP16 和 B4，对单/双共享
PyAV/AMF reader 做公平 A/B：

- 单 reader：41.501 秒，86.745 原始帧等效 fps；
- 双 reader 两次：24.451/24.721 秒，147.231/145.624 fps；
- 双 reader 中位数 146.428 fps，相对单 reader 加速 1.688 倍、耗时减少 40.76%；
- 三次均返回 120 个采样；双 reader 两次的有序时间、checkpoint PTS、检测分数和
  90×160 mask 均与单 reader 逐字节一致；
- 每次双 reader 的 D2D copy 为 64+60，FD close、fixed-context session、HIP stream
  与 event 生命周期全部配平；host transfer、CPU Map、staging、D2H、non-D2D 和
  failed bridge 全为 0；
- 双 reader 峰值显存约 23.57 GB，junction 70°C；运行窗口没有 OOM、GPU reset、
  ring timeout 或 page fault，也没有残留进程组。

证据保存在仓库外本地事务
`jasna-linux-amf-dual-reader-scan-ab-20260830/VERIFICATION.txt`。
这恢复了旧双 rocDecode 路线约 150 fps 的性能等级，同时产品后端仍只有统一
PyAV/AMF 路线。
