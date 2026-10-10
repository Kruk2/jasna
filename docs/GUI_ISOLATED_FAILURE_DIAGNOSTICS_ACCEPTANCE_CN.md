# 共享隔离任务错误详情修正

2026-09-08，产品 `jasna/gui/processor.py`。

原终态分支先检查非零退出码，导致guard/runtime/backend返回的protocol_error
被通用“isolated video job exited with code N”覆盖。现在保留退出码并附加具体详情，
详情规范为单行、清除控制字符并限定2048字符；无有效详情时保持原通用文本。
零退出码附带协议错误仍失败，Stop优先级、75/86重试分支及完成输出验证未改。

Terra负责窄改动与6项纯CPU测试，主线程审查实际diff后独立复跑通过。
测试直接执行实际formatter及终态AST，未加载Torch/PyAV/Tk或启动子进程。
另由主线程复跑guard adapter16项、runtime worker9项、GPU recovery假对象17项，
均通过；停止/清理/身份handoff/恢复原合同未变化。

Windows普通GUI尚未选择该隔离backend，因此此修正不是可见Windows窗口验收；
它修复共享Processor错误呈现，并补足后续Windows接入的诊断依赖。
未改变平台默认选择、guard时限、运行时路径策略或媒体后端。
