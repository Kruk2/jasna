# 最终输出持久化与 Stop 线性化

本阶段把完整渲染和 Smart Render 的成功条件统一为“已验证、已刷盘、已原子发布”，
并让 Stop、任务开始和最终完成状态在同一个锁域中裁决。它不修改检测、Tracker、修复、
编码质量、批大小或 NVIDIA/AMD backend 选择。

## 产品合同

- 完整渲染先写到最终目录中的唯一 staging 文件；只有结构、codec、时长和尾部读取检查
  通过后，才使用同目录原子替换发布到目标路径。
- 发布前刷写 staging 文件；发布后再次刷写目标文件和 POSIX 目录元数据，并重新验证目标。
  Windows 没有通过 Python 暴露可移植的目录句柄，因此执行文件级 `_commit` 等价刷写和
  发布后复验。
- Smart Render 继续复用同一个 `commit_video_output` 发布合同。HEVC mixed 输出在进入该
  合同前额外检查所有 copy seam hash、源片等帧数、严格递增且对源最大漂移不超过 1 ms
  的 PTS。独立 native HEVC 软件解码器的完整 FFmpeg strict decode 保留给显式开发/版本验收，
  不由 GUI 对每个长片固定执行，也不是产品处理路径或 CPU fallback。相同参数集 ID 的内容重定义
  本身只作诊断。
- Stop 与 job claim、输出目录创建、完整渲染发布、最终 `COMPLETED` 标记共用 completion
  lock。Stop 胜出时任务恢复为 `PENDING`，不会继续创建后续 job 的目录或发布 partial。
- 覆盖已有文件时，pipeline 未产生新输出或输出指纹未变化都视为失败；post-export 命令
  只在最终媒体验证通过后运行。
- 普通 full staging 在成功、失败和取消路径都会做 best-effort 清理；清理失败会保留日志，
  不把未发布 staging 伪装成成功成片。HEVC Smart 最终门失败是刻意的诊断例外：不覆盖
  目标文件，并保留 `.smart-render` assembly 和 signature 工作区，不自动改跑 full。
- PyAV 的定制 AMF runtime 可能把已完成 bitstream 的 decoder implementation 名称报告为
  `hevc_amf`、`h264_amf` 或 `av1_amf`。最终输出验收只把这三个精确名称分别归一到
  `hevc`、`h264`、`av1`；不会用通配后缀吞掉未知 codec 或真正的 codec mismatch。

## 验收范围

Linux 聚焦单测覆盖：缺失/未变化输出、验证与 fsync 次序、Windows 文件刷写模式、Stop
竞态、job claim、目录创建、post-export、同目录 staging、已有目标保护、Smart Render
通用发布包装以及失败清理。真实 Windows 文件系统、AMF 输出和 GUI Stop 竞态仍需在累计
PR 链的 Windows 实机矩阵中复验。

旧 checkpoint 的 Linux AMD isolated-child job API 不存在于本次从原始 v0.10 重建的
产品链中，因此本 PR 不重新引入一套没有调用方的子进程协议；当前产品路径的开始、发布
和完成均已由同一 `Processor` completion lock 覆盖。

2026-08-30 又以 8K HEVC Main10、60000/1001、3600 帧真实素材完成一次 B4/FP16/
RF-DETR MIGraphX/BasicVSR++/HEVC AMF CQ18 全流程：原子发布成功，软件 strict decode
和 3600 帧严格递增 PTS 通过。正向 `hevc_amf` 与负向未知 AMF codec mismatch 均有
聚焦回归测试；完整证据位于
`jasna-gui-8k60s-prescan-amf-reuse-20260830/VERIFICATION.txt`。

2026-08-31 又对历史 HEVC collision 事务进行离线复核。两份保留的 copy/render/copy
assembly 均通过新最终门：完整 strict decode 成功、帧数恰好 1003，源/输出 PTS 最大差
`1/180000 s`。聚焦 unit/media 回归 90 passed；本轮没有启动 GPU 或 8K 长任务。
此外，已安装统一 runtime 用当前 HEVC normalize 实现从保留 AMF raw NUT 重建 render TS，
当前 mux/commit 前验收链生成 2,355,782-byte copy/render/copy 成片并通过全部门；测试结束后
临时目录已自动清理。
