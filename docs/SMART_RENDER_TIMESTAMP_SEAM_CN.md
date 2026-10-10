# Smart Render 时间戳与 HEVC seam 收口

更新时间：2026-08-31

本阶段在统一 PyAV 产品链上收口局部处理后的直接装配、精确 PTS 和 HEVC 拼接安全性，
不改变检测、Tracker、修复、阈值或产品 B4 默认值。

## 实现边界

- 最终 mux 直接读取 fragment concat manifest，并从原片复制兼容的音频、字幕、章节、
  metadata、attachment 与视频 disposition，不再先生成一份中间 assembled video。
- secondary reader 按 `FrameMeta.pts` 读取原帧。若 decoder 偶发偏移，先丢弃有限数量的
  旧帧，再以相同产品 decode backend 最多重开两次；仍不一致就明确失败，不转入隐藏的
  `pyav-sw`/CPU fallback。
- keyframe probe 记录最后 packet tail 与源关键帧 decode delay；仅对没有自身 B-frame
  重排的 render fragment 补回 DTS delay，避免复制段和重编码段的 PTS/DTS 语义分裂。
- full render 和 Smart Render 都保持源 8/10-bit 合同；HEVC render fragment 继续使用
  已有 source VUI/fps resolver。
- HEVC 拼接前逐 RAP 比较 VPS/SPS/PPS。共享 parameter-set ID 内容变化时 fail-closed；
  装配后只在 render seam 两侧各取一个有界 copy GOP，比较帧 hash、duration、size 和
  归一化 PTS。
- framemd5 比较允许 seek 边界最多一帧差异和一个恒定的亚帧 PTS origin 偏移，但不允许
  内部 hash 改变或时间轴漂移。

## 验收

- `tests/test_splice.py`：packet tail/decode delay、fragment timestamp、HEVC parameter-set
  顺序、非零 stream start、copy window 归一化。
- `tests/test_splice_media.py`：H.264 decode-delay 实片、HEVC hvcC/Annex-B parameter-set、
  collision fail-closed、copy seam hash。
- `tests/test_pipeline_segments.py`：HEVC gate、VUI resolver 与 seam 两侧有界 GOP。
- `tests/test_pipeline_threads.py`：精确 PTS、相同 backend 重开、取消和无 CPU fallback。
- Linux 聚焦回归：167 passed、1 skipped；独立 encoder 单测中的旧 NVIDIA 默认断言在
  当前 AMD 主机未做伪装，本阶段没有修改那些共享/NVIDIA 断言。

手动选择的 Smart Render 范围遇到 seam 不兼容时保持显式失败。自动粗扫/精扫通过
`automatic_segments=True` 把范围来源传到产品执行层；如果在 Smart Pipeline 建立之前的
预检就能确认输入不适合 Smart Render，可以直接选择 full 路径。若必须等 Smart GPU 工作和
render fragment 生成后才能发现 HEVC parameter-set ID 冲突等运行期 seam 不兼容，则现在
明确 fail-closed，不在同一产品进程内自动启动第二条 full Pipeline。该边界不改变 auto
判定、检测/修复阈值或 B4，也不增加软件解码/CPU fallback。

## 自动 Smart 到 full 的实机否决结论

attempt006 表明，在捕获 `SmartRenderCompatibilityError` 的 `except` 作用域内直接启动
full Pipeline，会让 traceback 和关闭时序继续保留足以压满 8K decoder/native pool 的资源。
当时提出的候选方案是：保存纯文本原因、清除 traceback、退出异常作用域并完成同步与
allocator cleanup 后，再启动同一 GPU-only full Pipeline。

attempt007 对该候选方案进行了唯一一次授权实机验收，并将其否决。Smart 阶段两路 reader
完整完成 `1204/1204` 和 `1200/1200`，资源审计配平且禁止的 host/CPU Map/staging/D2H/
failed bridge 均为 0；HEVC seam 门也正确发现 VPS/SPS/PPS ID 0 冲突。但随后同进程 full
阶段的 secondary reader 在第 16 次 Vulkan -> HIP 导入失败（前 15 次成功，HIP 2），AMF
又报告 `Copy thread timeout`。独立 sysfs 显存峰值为
25,701,306,368/25,753,026,560 bytes（99.80%），运行窗口 kernel 同时出现
`Not enough memory for command submission` 和 framebuffer pin failure。最高 junction
只有 75°C，因此不是温控触发。

这次压力最终导致 Electron/Codex 退出、桌面失去响应并被迫重启；D 盘 NTFS dirty 已由
用户在 Windows 用 chkdsk 修复。由于强制重启，attempt007 的 guard/result 终态文件来不及
落盘，这本身属于失败现场，不能当作正常退出。

因此，同进程运行期 Smart -> full 自动重试已经撤回，不能进入产品，也不允许启动
attempt008。当前只保留 MPEG-TS runtime 构建/合同修复与 HEVC seam 安全门；运行期 seam
不兼容明确失败。撤回后的 processor/Stop/pre-scan/mosaic-scan/
splice/runtime/build/installer/AMF-contract/GUI queue/i18n CPU 聚焦回归为 459 passed，`git diff --check`
通过；没有再次启动 GUI、模型、FFmpeg 或 GPU 工作负载。

回归测试还固定了这一安全边界：自动扫描范围若在 `Pipeline.run()` 内才抛出
`SmartRenderCompatibilityError`，processor 只能关闭该 Pipeline 并原样抛出；
`build_pipeline` 必须恰好调用一次，不能在同一进程偷偷创建 full Pipeline。预检阶段尚未
创建 Smart Pipeline 时直接选择 full 的行为不受影响。

此外，Linux AMD native 路径在 `Pipeline.run()` 内出现该错误后，会把当前 GUI 进程标记为
必须重启：本轮队列立即结束，后续文件保持 Pending，Start 按钮禁用，用户会看到“重启后为
该文件选择完整视频”的明确提示。即使用户再次调用 `Processor.start()` 也会被拒绝，避免
Smart 失败留下的 driver-owned 资源被下一文件复用。这个锁只由已经执行过 Linux AMD native
GPU Smart 工作后的晚期兼容错误触发；Smart Pipeline 建立前的预检回退、NVIDIA、Windows、
普通 full 和其他队列错误均不改变。

在上述安全锁完成后，用户另行明确授权了可恢复的正式 runtime 安装。MPEG-TS 候选 runtime
已通过仓库安装器原子替换到默认 Linux AMD runtime 目录，旧 runtime 保留为
`linux-amd.backup-20260831-041146`。正常 launcher 的全新子进程 `--preflight-only` 验证通过
PyAV 18.1.0、全部固定 FFmpeg ABI、AMF bridge 来源和 13 项 FFmpeg 能力，其中明确包含
MPEG-TS muxer 与 demuxer。该步骤没有启动 GUI、媒体、模型或 GPU；不改变 attempt007 的
FAILED 结论，也不是 attempt008。

## 真实 8K HEVC 长片接缝结论复核

2026-08-31 的 8192x4096、59.94 fps、Main 10 实片运行完整处理了 47,686 帧，AMF D2D
审计为 `copies=47688/47688`、FD `47688/0`，禁止的 host/Map/staging/D2H/bridge 全为 0。
最终装配仍被 strict decoder 以 `VPS 0 does not exist`、`SPS 0 does not exist`、
`PPS id out of range` 拒绝，因此没有发布成片。

独立片段和保留事务复核确认了直接问题与此前错误推断：

- AMF NUT CodecPrivate 的参数集顺序是 SPS/PPS/VPS；旧的
  `hevc_mp4toannexb,dump_extra=freq=keyframe` 会把这组错误顺序内容注入已有正确
  VPS/SPS/PPS 的关键帧。HEVC 归一化现改为只用 `hevc_mp4toannexb`，参数集扫描也只读
  random-access packet，避免把数 GB 普通帧复制进 Python。
- AMF render 段与原片 copy 段会用相同 ID 0 表示不同 VPS/SPS/PPS 内容，但这不能单独证明
  接缝失败。旧 rocDecode 产品路线的几十个 8K VR 成功运行仍使用 AMF render 与源 copy；
  保留事务的 `assembled-with-colliding-ids.mp4` 和 header/decode-delay 修正版也都 strict
  decode 通过、恰好 1003 帧，并保持源 PTS（最大量化差 5.56 微秒）。

因此删除了 Linux AMD HEVC mixed plan 的一刀切 collision 门，自动与手动范围都继续使用
Smart Render，不再产生“部分 AI 修复但整片重编码”的 4 小时回退。安全门改为观察真实产物：

- fragment 每个 RAP 都必须有依赖顺序正确的 VPS -> SPS -> PPS；
- 所有 untouched copy seam 必须通过 bounded strict decode 与帧 hash/PTS 对照；
- 临时成片帧数必须等于源片，PTS 必须严格递增并与源片相差不超过 1 ms；
- 临时成片必须完成 FFmpeg `-xerror -err_detect explode` 全视频严格解码；
- 全部通过后才 fsync 并原子发布；失败保留 assembly/workspace，不覆盖原成片，也不自动 full
  重跑。

这次改变不触碰 auto 扫描判定、检测/Tracker/修复阈值、B4、H.264/AV1、Windows/NVIDIA
路线或 CPU fallback。离线媒体与聚焦回归合计 90 passed；两份保留 collision 成片也都通过
新最终门。另从保留 AMF raw NUT 出发，用已安装统一 runtime 和当前
`normalize_fragment()` 重新生成 render TS，再由当前代码完成 copy/render/copy assembly；
2,355,782-byte 临时成片通过全部最终门后由测试临时目录清理。30 分钟 8K 用户长片没有自动
重启，需新 GUI 进程再由用户手动验收。
