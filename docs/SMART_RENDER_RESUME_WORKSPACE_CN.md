# Smart Render 可恢复工作区与完整处理签名

更新时间：2026-09-17

Smart Render 的片段工作区改为固定、可恢复且由完整处理签名绑定。取消、进程异常或
机器重启后，下一次只复用同一输入、同一输出目标、同一 splice plan、同一模型和同一
处理设置产生的完整片段。

## 工作区合同

- 路径由输出名和 canonical signature SHA-256 派生，不依赖一次性临时目录。
- manifest 采用原子 JSON 写入；`running` 状态在下一次打开时回退为 `pending`。
- complete artifact 记录绝对路径、size、mtime 和完整 SHA-256；任一不符即拒绝复用。
- source identity 包含绝对路径、size、mtime 以及首尾各 1 MiB 的 hash。
- model identity 包含 detection、BasicVSR++ checkpoint 和 LUT 的 size、mtime、完整 hash。
- splice identity 包含 time base、start/end PTS、keyframes、B-frame 合同、decode delay 和
  每个 copy/render span。
- processing identity 由 session factory 构造，包含 Jasna 版本、设备、FP16、B4/B8、
  detector 阈值、Tracker/clip 参数、去噪、Primary TensorRT、完整 secondary 设置、VR、
  sharpen 与高帧率 retarget。
- encoding identity 包含 codec 和已解析的 encoder settings；算法版本为
  `jasna-smart-render-workspace-v4`。v4 同时固定 HEVC 参数集注入/检查语义，拒绝复用 v3
  时代可能带有错误 CodecPrivate 注入顺序或 copy PTS 的片段。

成功路径先验证临时成片的 codec、duration 和尾部可读性，再 fsync、同目录原子替换、
复验最终文件，最后才删除工作区。取消和一般失败保留证据及可复用片段。HEVC 最终门失败
同样保留工作区及同目录 `.smart-render` assembly；它不会覆盖已有成片，也不会在同一进程
自动改跑 full。修正设置或实现后，完整 signature 会决定片段能否安全复用。

## 验收

- manifest 损坏会保留 invalid backup 并重建；
- source/settings/model/algorithm 任一变化会得到不同工作区；
- complete hash 被篡改后不会复用；
- path traversal 和 workspace 外 artifact 会被拒绝；
- 已完成 render span 只增加进度，不制造虚假速度样本；
- TVAI 与 RTX secondary 的全部有效参数都进入签名；
- Linux 聚焦回归：194 passed、1 skipped。

2026-08-31 的后续复核确认，原失败 assembly 使用了 AMF NUT CodecPrivate 的错误
SPS/PPS/VPS 注入顺序。去掉 HEVC `dump_extra` 后，保留的相同 ID 重定义样本可以完整严格
解码、保持 1003 帧并通过 PTS/copy seam gate。因此 v4 不再仅凭参数集 hash 不同清理工作区；
只有缺失/错序 header 或最终真实成片验收失败才 fail closed 并保留证据。

## Linux AMD native worker 自动续跑（2026-09-02）

GUI 的 Linux AMD 视频任务继续以 per-video child 为 native 资源边界。Smart Render
render span 成功写入并标记 complete 后，如果该 pass 发生过整卡 pressure episode 或
critical reclaim，且后面还有 render span，child 会请求 parent 回收自身。parent 等待
至少 4 GiB 整卡余量稳定恢复后创建新 child；新 child 打开相同 signature 的 workspace，
把异常退出前的 `running` span 恢复为 `pending`，并只复用 size、mtime、SHA-256 均通过的
complete fragment。

AMF decoder open 另有 60 秒 isolated-worker watchdog。超时必须结束整个 child，不能在
仍持有 native AMF/Vulkan call 的线程旁继续运行；parent 最多自动重试两次。pressure
recycle 最多 32 次。达到上限、显存 60 秒内不能恢复到启动预算或两次 open 仍卡住时，
当前文件明确失败并保留 workspace，不会 CPU fallback、静默跳过或继续制造输出。

如果 HIP/MIGraphX 在 native 调用中触发 `SIGABRT`、`SIGBUS`、`SIGILL` 或
`SIGSEGV`，子进程可能来不及发出结构化 `retry` 事件。Linux AMD 隔离父进程会把这
类信号视为 native fault，等待整卡启动余量稳定恢复后，在同一 canonical workspace
上最多重启两个 fresh worker；已完成且哈希通过的片段继续复用，崩溃时处于
`running` 的片段回到 `pending`。超过两次、显存未恢复或收到用户停止信号时保持
fail-closed，不重试 SIGTERM/SIGKILL，也不会把部分输出标记为成功。

### 有界全片重试的稳定工作区标识（2026-09-14）

Linux AMD 的有界全片路线为了原子发布，会把每次尝试写入一个带 UUID 的临时成片路径。
该临时路径不能参与可恢复工作区的签名：worker 因 AMF session 边界退出后，新的 worker
必须继续打开同一 canonical 输出目标对应的工作区，才能复用已经完成的片段。现在由 GUI
把 canonical 输出路径单独传给 pipeline；临时 staging 路径只用于最终发布。旧版本把 staging
路径写入签名，会在每次回收后创建新工作区并重复渲染 fragment 0；修复后有回归测试覆盖
签名和工作区 slug 在不同尝试路径下保持稳定。

### 有界全片恢复范围哨兵修复（2026-09-15）

有界全片路线的 `SpliceSpan` 使用空 `effect_ranges` 记录计划元数据，但检测线程把
`None` 定义为“全片可检测”、把空 tuple 定义为“没有帧可检测”。旧实现把该空 tuple
直接传入每个 fragment 的 pass，导致检测/恢复队列为空，片段仍能编码并通过封装门，
最终成片却没有实际修复。现在传递给 pass 的值会把空 tuple 转回 `None`，同时保留显式
非空范围；回归测试锁定这两个哨兵语义。

该语义修复会改变 bounded-full 的处理签名（`all-frames-none-v1`）。因此旧版本生成的
“成功”片段不会被复用，下一次运行会在新的签名工作区中重新检测和恢复，避免继续沿用
无修复内容的历史 fragment。Linux AMD 的双 GOP、AMF/Vulkan D2D 和显存回收边界不变。

### H.264 AMF 多片段死锁隔离（2026-09-17）

一部 4096×2048、59.94 FPS、H.264 High/yuv420p 实片在第二个 Smart Render
render span 中复现了 AMF native encode 永久阻塞：第一个 480 帧 span 正常完成，第二个
368 帧 span 已完成解码、检测和 BasicVSR++ 恢复，但 encoder worker 卡在
`out_stream.encode()`；此时 GPU busy 为 0、显存和磁盘余量充足，对对应源片范围的软件
解码也正常，因此不是 OOM、坏码流或最终合并错误。

Linux AMD H.264 Smart Render 现在每完成一个 render span，就在下一 render span 前正常
回收 isolated worker。完成片段仍由 canonical workspace 的 size、mtime 和 SHA-256 门复用，
copy span 和最终拼接合同不变。该边界只作用于 Linux AMD GUI 的隔离视频任务，不改变
NVIDIA、Windows 或非 Smart Render 路线，也不会启用 rocDecode。

编码心跳超过 30 秒只记录一次紧凑警告，不再每 30 秒重复写入完整线程栈；isolated AMD
worker 连续 90 秒没有编码产出时，以专用退出码 fail closed。GUI parent 等待整卡启动余量
恢复后最多用两个 fresh worker 重试，并从已完成 fragment 继续。正常 session 边界的重复
初始化日志会被静默；只有 restoration/finalization 的同阶段进度采用高水位，不会把粗扫的
0–15% 与恢复阶段的 0–100% 混为同一进度。恢复重新进入 restoring 后才更新 FPS 和 ETA。
单元验收覆盖 H.264 span 边界、编码卡死退出码、重试上限、
日志去重以及进度单调性；真实视频验收由用户在 GUI 中执行。

2026-09-18 的后续实片证明仅靠缩短 span 和 fresh-worker 重试不能修复根因：同一长
render 区间会在相近位置重复停在 `out_stream.encode()`，当时整卡仍有约 20 GiB 空闲，
因此不是 OOM、合并或扫描故障。进一步实片启动日志确认当前 AMF runtime 的 H.264 QVBR
反而强制要求开启 PreAnalysis；把 PA 关闭但保留 QVBR 会在 encoder open 阶段明确失败。
因此 QVBR/CQ 与禁用这个长期停滞源在该 runtime 上不能同时成立。

现在 Linux AMD H.264 Smart Render 使用与 HEVC 自动路线相同原则的源码率 `vbr_peak`：
target 与 peak 取源视频平均码率，buffer 为其两倍，并强制 `preanalysis=0`、`vbaq=0`；
源 GOP、profile 与 B 帧结构继续匹配，GUI CQ 不参与这条自动源码率路线。人工 30 秒切段
已经撤销。新 RC 字段进入完整处理签名，因此旧配置片段不会被复用。原有跨自然 render
span 的进程隔离、编码停滞 fail-closed 和可恢复工作区仍作为安全边界保留；不会启用
rocDecode，也不改变 HEVC 双 GOP 路线。
