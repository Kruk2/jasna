# GUI 结束状态修正与验收边界（2026-09-07）

## 已修正

旧版 `JasnaApp._handle_complete()` 把处理器的“本轮结束”通知一律当成成功，
即使用户已停止或队列中有失败任务，也会显示 100% 和 `All jobs completed`。
主会话独立执行原方法 AST，已复现停止和失败两种误报。

本次仅修改共享 GUI 逻辑，不另建 Windows 队列流程：

- `Processor.was_stopped()` 只读当前/上一轮的停止事件；Start 原有清零逻辑不变。
- 非空队列全部完成或跳过、且没有 Stop 请求，才调用 `set_completed()`。
- 停止、失败、未完成、空队列均重置控制栏，不强制显示 100%。
- 停止记录 INFO；失败记录 ERROR 和失败数量；未完成/空队列记录 WARNING。
- 原有“必须重启应用”分支仍优先，按钮、队列状态、日志关闭逻辑保留。

## 分工与独立验收

Terra worker 在独立暂存目录完成源代码追踪、修正和 13 项契约检查。
主会话审阅实际差异、复跑 worker 检查，并另写独立验收：

- 13 项 GUI 结束状态及只读停止接口测试。
- 27 项处理器、协议、线程和假进程生命周期测试。

候选第一版缺少失败汇总，在主会话验收中 12 通过 / 1 失败；补齐后 13 项全过。
部署后再次针对产品源码执行 13 + 27 项，全部通过。

这些测试使用 CPU Python、源码 AST 和假 UI/后端对象，禁止加载 GPU/媒体/Tk
运行时。处理器契约测试省略四处重型导入并使用显式禁止调用的替身；方法主体
未重写。清理和进程测试不证明真实 HIP 释放或实际 FFmpeg 子进程树退出。

**这不是原生 GUI 视频处理/停止验收，也不是 60 秒 8K 修复或性能验收。**

主会话证据目录：
`D:\AI\jasna_windows_amd_dev\Temp\gui-lifecycle-cpu-20260907-a1`

- `STATUS_BASELINE_DEFECTS.json`：两项预期失败，只证明原缺陷存在。
- `STATUS_CANDIDATE_A1.json`：第一版独立验收失败，保留记录。
- `STATUS_CANDIDATE_A2.json`：修正版 13 项通过。
- `STATUS_DEPLOYED_A2.json`：产品部署后 13 项通过。
- `CPU_DEPLOYED_A2.json`：产品部署后 27 项通过。

Worker 源码、不可变原版样本和测试：
`D:\AI\jasna_windows_amd_dev\Temp\gui-lifecycle-status-fix-20260907-a1`

## 部署与恢复

部署前逐一核对原文件和候选 SHA-256，确认没有 Python/FFmpeg/Jasna 进程。
保留原有未提交修改，没有替换用户其他工作。原文件可从以下独立备份恢复：

`D:\AI\amf-unified-work\transactions\jasna-windows-triton38-matched-stack-20260906\gui-terminal-status-backup-5ef00effd53e472193665599dcc03167`

部署后哈希与验收候选逐字节一致：

- `jasna/gui/app.py`: `0d9d3c17c737b458707f63c5a9358547715aba338eec8b12dcb02ca0cfe4de04`
- `jasna/gui/processor.py`: `73049ac98a889d262b5d9ece81e8b58c5f74491f7b441614e6f82f74269333cc`

## 未解决与未变更

Windows 默认仍是进程内视频处理；Linux AMD 的独立进程恢复没有直接启用到
Windows。现有 Windows 终止分支不保证 FFmpeg 子进程树清理，排队中的旧进度
回调也未增加轮次隔离。本次只修正结束误报，不宣称这些生命周期缺口已解决。

模型、编解码、共享队列/跟踪/修复逻辑、运行时和实验开关未变。
分页文件仍是 2176 MiB，未修改系统设置、未重启、未关机。
完整 8K 测试仍受已复现的系统提交容量保护限制；没有重新启动同样的失败测试，
也没有把上述 CPU 测试冒充真实视频通过。

### 后续系统配置更新（2026-09-07 10:57，用户已批准）

上述“未修改分页文件”描述的是本次 GUI 验收时的状态。随后用户明确批准了
C 盘分页文件初始 16 GiB、最大 32 GiB；管理员脚本已成功写入，并经主会话
独立核对 WMI 和注册表。但实际分页文件仍为 2176 MiB、提交容量尚未增加，
需要重启后再次确认。没有重启，也没有启动新的 8K 测试。

修改前后记录与恢复执行入口说明：
`D:\AI\jasna_windows_amd_dev\Temp\pagefile-16g-32g-authorized-20260907-a1\STATUS.md`
