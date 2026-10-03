# Windows GUI 父进程有界后端：接线及真实 Stop 验收

2026-09-08。阶段性通过；不是完整视频/吞吐验收，不自动启用 Windows 隔离。

## 产品变更

- `Processor` 新增默认 `None` 的 `video_job_attempt_backend` 注入点。原 Linux
  工作进程路径保留；自动平台选择未变。父进程仍使用共用事件处理、Stop 和
  重试逻辑，Windows 后端不复制 AI 模型或视频时序管线。
- 新增 `windows_guarded_attempt.py`：固定资源额度的 Windows Job 启动、精确
  Popen 句柄清理、独立 stdout/stderr 有界读取、严格退出码验证。控制命令
  是8条有界队列，每条至多4KiB，由独立 writer 写管道，GUI只做非阻塞入队。
- `isolated_worker_streams.py` 与 `windows_guard_exit_result.py` 从已验收版本
  按字节一致引入，SHA分别为
  `5544e107f1e6011e6f712d6515bac7112abbe9ef9b416f4b8b0e57de8f3a5ebd`、
  `e23287ab9e8ad88f361bde9a137920460eb10b39cb93f076bb8c2ddf761c4023`。
- 守护普通终止失败后，使用同一 Popen 句柄 kill/wait；不按整数 PID 查找或
  杀进程。清理不完整时拒绝该后端的后续任务，要求重启恢复。

资源：Job6144MiB，host commit reserve10240MiB，physical reserve8192MiB，
processes16，poll1s；当前后端时限仅接受1..180秒。1秒仅用于CPU超时测试。
媒体功能开关仍为0，全卡显存预算路线未改。没有开启 CPU AI 修复。

## 分工与独立验收

主线程实现生产代码并运行全部真实进程/视频试验。Terra 仅拥有
`tests/test_windows_guarded_attempt.py`，编写纯CPU假管道/假进程测试并指出
cleanup缺少kill回退及join异常跳过解绑的问题；主线程修正生产代码。

Terra测试16项通过；主线程审阅实际测试代码、独立重跑16项通过，覆盖固定
额度、唯一报告、UTF-8命令及队列溢出、共用stdout解析、stderr隔离、协议丢失、
重复终态、回调异常、超时、精确句柄kill回退、清理失败后禁止续跑及默认None
注入边界。Linux路径仅做源代码未改与CPU边界核查，不冒充原生Linux验收。

主线程还运行实际 `Processor` attempt/control 方法体（AST按原样提取以免
CPU父进程加载GPU库），配真实Windows guard和共用worker的FakeProcessor。
最后一组完成、重试75、错误1、暂停/恢复/Stop、超时拒绝五项均通过，无读写
线程/后代残留。记录位于：
`D:/AI/jasna_windows_amd_dev/Temp/windows-guard-exit-adapter-20260908-a1/product-parent-cpu-runs`。

## 真实 8K Main10 活跃处理中 Stop

运行：
`D:/AI/jasna_windows_amd_dev/Temp/windows-shared-worker-20260907-a1/native-runs/GUI600_20260907T164613Z_3c349b60a1f5`。

原生8192x4096、Main10、固定600帧片段（10.010秒），输入SHA
`3c630edaefe53d242bbc864c0b11c3c48c92e12ef1d2b48dd73431697c7af2fc`。
实际产品父进程attempt/control/event方法体，实际共用GUI worker/session/pipeline，
eager恢复、已验收研究解码/resize设置。没有打开GUI窗口。

处理到128/600帧、phase=restoring时，经产品原有命令接口发送Stop。总耗时
50.047秒，Stop到guard退出2.734秒；guard completed/child0/active0，Peak Job
6,179,934,208 bytes，低于6,442,450,944上限，全部资源门槛满足。131条进度事件。
父/子终态均为result pending；pipeline取消true、完成false、worker线程空，
全卡显存reader及offloader线程关闭，清理观察器恢复。主线程另用独立PowerShell
检查原始parent/guard/child/request、当前源码SHA、输入身份、资源额度、Stop
触发和关闭状态，通过；最终进程清单为空。

前一次 `GUI600_20260907T164431Z_d3f565b0b90a` 的真实处理也正常停止（2.782秒、
guard completed/active0），但主验收脚本重复读取一个已自行清空的reaper引用，
导致AttributeError。保留原始失败和脚本备份；改为先持有本地线程引用后重跑，
上述新运行完整通过。未把第一次的failed标记改成pass。

取消的MP4仅为局部失败/取消证据，不是完整可播放产品输出的验收。

## 尚未完成

自动Windows平台选择、匹配运行时完整视频启动验收、guard打包/定位、Windows全卡
显存恢复后重试接线、真实GUI窗口交互及完整输出验证仍需完成。没有把事务
绝对路径写入产品代码。当前显式后端配置不适用于任意长度生产视频的默认时限。
不得直接开启自动Windows隔离或把研究后端当成已打包产品。

用户明确 Linux 是成熟AMD/ROCm实现参考，不要求Windows FPS相同。编译残差
候选仍被native8K数值差异拒绝；完整60秒8K Main10修复验收仍未完成。

## 选定运行时启动准备与真实进程预检（2026-09-08）

新增 `windows_video_worker.py` 的显式 Windows source-mode 配置；复用现有
`build_runtime_environment` 和 `scripts/run_jasna_unified.py --_product-child`，
保留统一 DLL 目录句柄及子进程实际加载 ABI 校验。产品不硬编码事务路径；
frozen/其他平台或缺失路径拒绝启动。`Processor` 仍共用请求文件、事件及重试
逻辑；后端未注入时沿用原命令。Windows 自动隔离仍未开启。

主线程独立重跑新增启动测试6项及 Terra 先前提交的守护测试16项，全部通过。
主线程串行运行固定180秒、原资源额度的真实 guard + 匹配 Python：

- `selected-runtime-preflight-runs/fa7a9fb76f73435eb2b25246323637b1`：
  预检通过，PyAV18.1.0、FFmpeg ABI、来源路径和 bridge API 验证通过，child0。
  Windows required FFmpeg help topics 为空，不能声称已运行 FFmpeg CLI 功能检查；
  Windows 当前也不枚举已加载 FFmpeg DLL 路径，空列表不是该项验收。
- `selected-runtime-preflight-runs/b052bc962e5742038a273d008f73a2a0`：
  原样执行新 builder 产生的完整产品子命令，空请求由共用 worker 正确拒绝；
  一条 fatal（unsupported isolated video job request schema），child1，
  guard child_failed，而非超时、崩溃或资源强杀。

上列目录均位于
`D:/AI/jasna_windows_amd_dev/Temp/windows-guard-exit-adapter-20260908-a1`。
两次 active0、无通信错误和读写线程残留。主线程另用 PowerShell 独立校验原始
报告、当前源码SHA、实际命令与guard命令逐字匹配、退出码及内存额度，全部通过；
进程清单为空。产品媒体开关仍0。没有媒体输入、AI执行或GUI窗口；先前 Stop
结果是旧源码的历史证据，不把它当作新启动路径的真实视频验收。

Terra 只读梳理全卡显存/恢复路线，无文件变更或测试执行；主线程复核实际
`processor.py`、`vram_offloader.py`、`system_stats.py` 及研究版 reader 源码，
独立执行研究 reader 的13项CPU测试通过。Windows 默认全卡采样仍误用 Linux
reader，得到None；研究版按 HIP LUID/node 匹配 PDH Dedicated Usage 的字节值。
接线前必须解决 GUI 父进程身份获取不重新创建 HIP 上下文的问题，并做实际
退出→采样恢复→重试验证。Linux 4GiB启动预算与约1GiB运行时全卡压力预留不同，
此轮未调整任何水位或将研究 reader 自动启用。
