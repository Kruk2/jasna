# 共享 GUI 进度回调隔离：CPU 验收记录

2026-09-07。本次仅部署 `jasna/gui/app.py` 的局部共享 GUI 修复；不改变
Linux/Windows 视频流程、Processor 执行语义、GPU 后端或任何测试资源保护。
这不是原生 GUI 操作验收，也不是连续 60 秒 8K 修复验收或速度提升证据。

## 改动与边界

- 每次实际 Start 分配 UI generation；排队回调携带该 generation。
- Stop 后拒绝 PROCESSING，但保留同一轮权威 PENDING/终态的队列行刷新。
- 旧轮次完成通知不能重置新轮次；重复完成通知不重复清理。
- 已移除的任务不更新界面，已结束的任务不被旧 PROCESSING 通知复活。
- close 使排队回调失效；Start 异常使当前 generation 不再活动；捕获窗口
  销毁时 `after()` 的 TclError/RuntimeError。Start 异常的完整控件回滚不在本次范围。
- 原有“停止/失败/空队列不显示全部完成”的逻辑保持不变。

Processor 回调接口本身没有 run ID。本次保证隔离旧 UI run **已排队**的
回调，不能识别假设性逃逸旧 worker 在新 Start 后才新发出的消息。
当前 Processor 的 Start/is_running 以线程是否存活为准，正常生命周期不允许
上一轮线程仍存活时启动新轮次。本次不改变该契约，也不修改日志回调通道。

## 独立验收

Terra worker 暂存候选并运行 11 项 AST/fake-UI 测试。主线程审阅实际 diff，
指出并要求补回 Stop 后的最终队列行通知，随后独立重跑该 11 项，全部通过。

主线程自编 9 项回调检查，在原产品代码复现 7 种过期/重复回调问题，并确认
2 项正常行为；修复后全部通过。部署后再次针对真实产品文件运行：

- 9 项回调流程检查；
- 13 项终态状态回归；
- 27 项 Processor/视频任务生命周期 CPU 回归。

共 49 项通过。测试执行真实方法 AST，替代 GUI/后端表面；没有启动 Tk、
Torch、GPU、FFmpeg 或视频处理。部署文件与已验收候选 SHA256 完全一致。
局部 diff 空白检查没有发现新增空白错误。

证据目录：`D:/AI/jasna_windows_amd_dev/Temp/`。

- `gui-progress-epoch-main-20260907-a1/BASELINE_DEFECTS_A3.json`
- `gui-progress-epoch-main-20260907-a1/DEPLOYED_A1.json`
- `gui-progress-epoch-20260907-a1/CPU_PROGRESS_EPOCH_MAIN_RERUN_A1.json`
- `gui-lifecycle-cpu-20260907-a1/STATUS_EPOCH_DEPLOYED_A1.json`
- `gui-lifecycle-cpu-20260907-a1/CPU_EPOCH_DEPLOYED_A1.json`

生命周期 harness 的旧默认 Processor pin 在第一次预检拒绝运行；主线程核对
当前 Processor 与先前已验收 `gui-lifecycle-status-fix-20260907-a1/processor.py`
逐字相同后，显式传入当前 SHA256 才重跑，没有绕过源码一致性检查。

## 源码与恢复

部署前 app SHA256：
`0d9d3c17c737b458707f63c5a9358547715aba338eec8b12dcb02ca0cfe4de04`

部署后 app SHA256：
`f44adf4f6e935562ee2ff2ddfff9fbfe081bf688acd4c337031d8188ce536db8`

基线保存在 `D:/AI/jasna_windows_amd_dev/Temp/gui-progress-epoch-20260907-a1/app.baseline.py`。
恢复时应只反向应用本次局部 hunks；若产品已有后续修改，不应整文件覆盖。
工作树原有未提交修改均保留，没有执行 Git reset/checkout。

下一步仍需原生 GUI 完整处理/Stop 操作验收，以及连续 60 秒 native 8K Main10
输出验收。600 帧的 10 分钟测试上限等待用户确认，当前仍为 180 秒；超时
不等价于卡死或优化无效。
