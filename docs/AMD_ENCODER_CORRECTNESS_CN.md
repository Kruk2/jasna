# AMD 编码正确性合同

本功能只收口 AMD AMF encoder 自身的跨平台正确性，不改变 GUI 质量默认、共享 Pipeline、解码路由或 NVIDIA。

## AV1 Main10

Linux 和 Windows AMD 的 P010 AV1 禁用 AMF PreAnalysis，并使用 `vbr_peak`。PyAV codec context 的目标码率
优先取源码率；源没有码率时按像素率计算并限制在 2--100 Mbps。AV1 使用自身支持的 `aq_mode=caq`，不再传
HEVC/H.264 的 `vbaq`。8-bit AV1 和 NVIDIA AV1 不应用此策略。

## 缓冲帧所有权与 PTS

编码器的有界重排窗口将 frame、PTS、插入顺序和 LUT 标记保存在同一个 heap item 中。AMD native decoder 的
batch storage 可能在下一批复用，因此入缓冲时 clone 一份 GPU tensor；NVIDIA 保持已有不复制路径。这样乱序
PTS 不会把另一帧或另一 LUT 标记配错，也不会把已复用的 decoder storage 交给异步 encoder。

## 码率参数范围

源码率派生的 `maxrate` 或两倍 `bufsize` 超过 FFmpeg 32-bit encoder option 范围时，明确记录 warning 并省略
该 ceiling，避免把溢出的无效参数传给 native encoder。已有 CQ/质量值本身不变。

## 验收边界

Linux 聚焦测试覆盖选项、位深隔离、码率边界、frame/PTS/LUT 绑定和 AMD/NVIDIA ownership 隔离。Windows 仍需
用真实 AV1 Main10/P010 和 H.264/HEVC 输出做 FFmpeg strict decode、帧数、PTS、位深、文件大小、关闭/取消和
AMF 错误检查；当前 Linux 结构测试不替代 Windows 实机门。
