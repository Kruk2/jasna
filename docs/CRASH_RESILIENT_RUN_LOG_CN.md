# GUI 崩溃诊断运行日志

高级设置新增“保存诊断运行日志”，默认关闭。启用后，每次批处理创建独立、不可覆盖的
UTF-8 日志；选择输出目录时写入其 `.jasna-logs/`，使用“与输入相同位置”时写入 Jasna
用户配置目录下的 `run-logs/`。递归媒体扫描会忽略 `.jasna-logs`，不会把日志目录中的
同名测试文件误加入处理队列。

## 行为合同

- GUI、processor 和 Python logging handler 只做非阻塞 enqueue；专用 daemon thread 独占
  建目录、打开、写入、flush 和 fsync，不把日志 I/O 放进媒体热路径。
- 内存队列固定上限。日志风暴时丢弃最旧事件，并在文件中写入带准确序号缺口的 dropped
  marker；日志故障始终 fail-open，不改变任务成功、失败或 Stop 语义。
- 默认每 1 秒 flush、每 5 秒 fsync；即使 GUI/驱动随后异常退出，已经完成周期 fsync 的
  记录仍具备持久化屏障。正常完成、Stop 或关闭时再请求最终 flush/fsync，并以有界时间
  等待，不允许日志线程卡住 GUI 退出。
- 日志记录 GUI/Python/平台、输出配置、队列输入、处理消息和当前 job。Linux 每 30 秒只读
  `/proc` 与 AMD DRM sysfs，记录 CPU load、RAM、GPU busy、VRAM、edge/junction/memory
  温度和功耗；不导入 GPU runtime、不执行外部工具，也不启动 kernel journal 抓取。
- kernel OOM、GPU reset、ring timeout 和 page fault 仍应在失败后由 `journalctl`/pstore
  独立核对；运行日志会明确写出这个边界，不能把缺少 kernel 行误判为没有 kernel 故障。
- 编码 stall 看门狗从第一次实际进入 encoder `write()` 才开始计时；BasicVSR++ 为首批帧
  正常计算超过 30 秒时不再误报编码卡死或输出整段线程栈。第一次及后续原生编码提交本身
  若超过阈值，仍保留完整队列、显存和线程栈诊断。

## 验收范围

Linux 聚焦单测覆盖路径隔离与防碰撞、周期 fsync、正常关闭最终同步、bounded queue 丢失
标记、I/O 失败 fail-open、`/proc`/AMD sysfs 解析、遥测格式和递归扫描隔离。GUI 控件、
Windows 文件系统刷写及 Windows 实际崩溃后的保留范围仍需在累计 PR 链的 Windows 实机
验收中复核。
