# HEVC Smart Render 编码兼容合同

Smart Render 会把重编码片段与源码 copy 片段拼接；仅“每段都能编码”不够，重编码 HEVC 的 level、颜色 VUI、
时间基和关键帧合同必须与源码参数集兼容。

Linux AMD HEVC fragment 在用户没有显式指定 `level` 时，将 FFprobe 的 level_idc 映射为 AMF dotted level。
未显式指定 `rc`、`maxrate`、`bufsize` 或原生 QP 时，Smart Render fragment 自动使用源码率
`vbr_peak`；GUI 选择“自动（匹配源码率）”时，完整编码也使用同一合同：target 为源视频平均码率，
peak 为 target 的 1.25 倍，buffer 为 peak 的 2 倍，并强制 `preanalysis=0`、`vbaq=0`，同时保留
closed-GOP/forced-IDR 合同。显式码率策略继续优先；`JASNA_AMF_HEVC_VBR_PEAK=0` 可回退旧
fragment CQP（便携 CQ+2）。CLI 完整编码、非 Linux AMD、非 HEVC 的产品默认均不变。

第一次真正需要渲染片段时，产品从源 HEVC codec context/首帧读取 SPS VUI 可见的 framerate、color range、
matrix、primaries 和 transfer，只将编码器已支持的值覆盖到 fragment 专用 metadata。读取失败会记录 warning，
后续 splice 参数集检查仍 fail closed，绝不靠猜测绕过 seam guard。

除 GUI 显式选择自动源码率时的 Linux AMD HEVC 完整编码外，本功能不改变 CLI 完整编码、NVIDIA 默认、
解码路由、Tracker/Pipeline 调度或手动 CQ 值。Windows 的 AMF stream 语义、真实 HEVC Main/Main10
copy/render seam、严格 PTS/时长仍需要在 Windows 真机验收；完整 FFmpeg strict decode 属于版本级独立
验收，不放入每个 GUI 文件的产品完成路径。

## Linux AMD 实片接缝边界复核（2026-08-31）

8K Main 10 长片曾在最终拼接时报 VPS/SPS/PPS 未初始化；复核旧 rocDecode 产品路线的几十个
8K VR 成功记录及保留事务后，不能再把“源 copy 与 AMF render 使用相同参数集 ID、内容不同”
本身判成必然失败。rocDecode 只负责修复区读取，接缝实际仍是源 HEVC copy 与 AMF render，
因此该成功历史对新 PyAV/AMF 路线同样是有效兼容性证据。

同时确认 AMF NUT extradata 为 SPS/PPS/VPS 顺序；因此 HEVC fragment 标准化不得再使用
`dump_extra=freq=keyframe` 注入该 extradata，只保留 `hevc_mp4toannexb`。参数集预检只扫描
关键帧，并要求 VPS -> SPS -> PPS 依赖顺序。

此前 8K 失败的直接原因是错误的 CodecPrivate 注入顺序，不是已证明的 ID collision：保留事务
中的 `assembled-with-colliding-ids.mp4` 与修正 header/decode-delay 的候选均完成 FFmpeg
`-xerror -err_detect explode` 严格解码，帧数都是 1003，PTS 与源片只存在 90-kHz MP4 时间基
量化造成的最大 5.56 微秒差异。

产品因此恢复 Linux AMD HEVC copy/render 混合 Smart Render，不再把自动范围退成整片编码。
每个 copy/render fragment 的起始 random-access packet 必须携带依赖顺序正确的
VPS -> SPS -> PPS；最终临时成片还必须依次通过 copy seam 解码 hash、源片等帧数、严格递增
且最大漂移不超过 1 ms 的 PTS，之后才允许原子发布。完整 native HEVC 软件 strict decode
保留为显式独立验收函数，仅用于开发/版本矩阵，不由 GUI 对每个已完成长片重复执行，也不是
处理失败后的 CPU fallback。相同 ID 的合法重定义只记录诊断信息。任一产品最终门失败都不发布、
不自动 full 重跑，并保留 assembly 与可恢复工作区。

Linux 统一 runtime 必须注册 AAC decoder。PyAV 只有在对应 codec 注册后才能从 MP4 音频
stream 创建 copy template；若 runtime 缺失 decoder，产品现在显式失败，禁止再以 warning
跳过音轨并生成静音成片。

## 断点 copy 片段时间轴复核（2026-09-01）

真实 8K Main10、59.94 fps 长片完成 47,688/47,688 帧两条 AMF D2D reader 后，最终
copy seam gate 在 `419.419s` 拒绝候选成片。源片窗口有 300 帧，候选窗口有 301 帧；
跳过候选首帧后，其余 300 帧像素 hash 全部与源片一致。packet 级复核确认复用的旧
`0000.ts` 在 GOP 内产生约 11 微秒的异常 PTS 间隔，而正常帧间隔应为
16.683/16.684 毫秒。使用当前 `create_copy_fragment -> normalize_fragment` 从同一源片、
同一 GOP 重新生成的片段时间轴正常，证明故障来自旧工作区 copy artifact，而不是当前
修复、编码或最终 seam gate。

工作区文件 hash/大小只能证明 artifact 未被外部修改，不能证明旧版本生成的媒体语义仍
有效。产品现在复用 HEVC copy span 前，会无解码地读取该 span 和源片对应区间的完整 packet
PTS，移除容器原点后逐项比较，允许不超过 1 ms 的 muxer 时间基量化。数量不等或时间轴不符
时，只废弃并重建该 copy span；已完成的 render span 继续复用，不重跑耗时修复。最终成片
仍必须经过 bounded seam 逐帧 hash、全片 PTS 和帧数门，不能用复用检查代替最终验收。

GUI 独立 worker 的原条件复现进一步确认，旧 copy artifact 不是唯一根因：Linux 统一 runtime
只注册 AMF 视频解码器时，FFmpeg CLI 无法从 PyAV NUT 中恢复源片 B-frame DTS，长 copy span
在 NUT -> MPEG-TS stream-copy 规范化时会把每组多帧压成约 11 微秒间隔。系统 FFmpeg 因含
原生 HEVC 解析能力而没有复现，故不能用系统 FFmpeg 的手工候选代替产品验收。H.264/HEVC
copy span 现在由 PyAV 直接保留源 packet PTS/DTS 写入最终 MPEG-TS，跳过该有损中间步骤；
这仍是纯 remux，不解码像素、不进入 CPU fallback，也不改变 render/AMF 编码路线。AV1 和
render span 继续使用既有规范化流程。

## GUI 完成门收口（2026-09-01）

真实 8K Main10 全长候选以 111,286 帧通过六个 copy/render 接缝 hash、全片严格递增 PTS、
源片等帧数及最大 `0.000005556s` PTS 差门，证据位于
`hevc-smart-direct-copy-20260901.Ujw5rz/PASSED.txt`。在这些片段和最终媒体语义已经通过后，
GUI 再用软件解码器逐帧解完整长片只会重复数小时工作，且旧 rocDecode 产品路线也没有该固定
完成成本，因此不再作为每文件发布条件。

同时收紧最终阶段的实际 I/O：参数集预检只读每个 fragment 边界的首个 random-access packet，
因为这是 concat 独立进入该 fragment 时需要建立解码状态的位置；不再为重复检查内部关键帧而
顺序扫完整个数 GB fragment。该只读 packet 探针的 `unspecified pixel format` FFmpeg 警告在
局部 capture 中抑制，缺少或乱序 VPS/SPS/PPS 仍由结构解析 fail closed。每个接缝 copy 侧的
像素 hash 窗口封顶约 1 秒，全片 packet 数、PTS 严格递增和 1 ms 对齐门保持不变。GUI 日志会
分别报告参数集、stream-copy 拼接、接缝进度和全片 packet/PTS 阶段。

## AMD HEVC CQ 与成片体积实测（2026-09-01）

真实 8192x4096、Main10、60000/1001 的 1856.6825 秒素材中，源视频 packet 为
4,148,809,281 bytes（约 17.88 Mb/s），CQ18 Smart Render 候选的视频 packet 为
10,771,781,441 bytes。四个 copy span 只占 2,238,560,119 bytes；三个 CQ18 render span
达到 8,533,221,322 bytes，证明 4.2 GB 膨胀到 10.8 GB 来自修复区 CQP 码率，而不是 concat
重复、音频或容器异常。Linux AMD HEVC Smart Render 将界面 CQ 加 2 后作为 AMF CQP，因此
CQ18 实际为 CQP20；CQP 不使用源码率 `maxrate` 上限。

对同一已修复 8K 内容的前 600 帧做原帧率实编码，CQ25/CQP27、CQ28/CQP30、
CQ31/CQP33 分别为 22.393789、13.863673、8.752040 Mb/s。按源片 packet 总预算扣除固定
copy span 后，render span 的目标是 16.707467 Mb/s；补测 CQ27/CQP29 得到
16.706837 Mb/s。由此预测本片使用 CQ27 的视频加原音频约 4,163,870,295 bytes，与源文件
4,209,930,944 bytes 同一量级。该值是本片及相似 8K HEVC VR 素材的可靠起点，不是 CQP
对所有素材的体积保证；跨内容稳定贴近原片仍应研究 AMD 官方推荐的 `vbr_peak` 源码率合同。

此前“CQ18 短片接近原片”的印象不能直接外推到这里。旧证据使用 4096x2048 H.264 源：原始
60 秒 packet 为 129,072,304 bytes，但当时受 1201 帧边界限制的 CQ18 对照实际是 20 fps
fixture，输出 73,282,334 bytes，比原 60 秒 packet 小 43.22%，本来就不是同帧率体积等价。
随后原 60000/1001 帧率的 20 秒 HEVC CQ18 矩阵输出为 35,609,583 bytes。旧素材是 4K H.264、
当前素材是 8K HEVC，画面复杂度、源编码效率和 Smart Render 的 copy/render 占比也不同；
AMD 官方同样明确 CQP 体积随内容复杂度变化，因此两次 CQ18 结果并不矛盾。

## Linux AMD HEVC 源码率 vbr_peak 验收（2026-09-01）

旧实验只设置 `maxrate`/`bufsize`、没有设置 `codec_context.bit_rate`，曾得到约 292.7 Kb/s 的
异常低码率；这不是 `vbr_peak` 本身失效。正式合同同时设置 target、peak 和 buffer。多 fragment
路线继续禁止 QVBR + PreAnalysis；真实 native abort 证据仍要求 Linux AMD HEVC fragment
保持 `preanalysis=0`。

P010/Main10 8K、60000/1001 的 333 帧真实 GOP 使用 target 13,663,282 bit/s、peak
17,079,102 bit/s、buffer 34,158,204 bit。两轮输出逐字节相同，都是 333 帧、2 个关键帧，
全部 packet PTS 与源一致并通过 FFmpeg `-xerror -err_detect explode` 软件严格解码。输出视频码率
12.613728 Mb/s，视频 packet 8,812,135 bytes，比源 GOP 小 7.1%；最大 1 秒 packet bucket
15.905656 Mb/s，没有超过 peak。同参数 CQP25 为 29.100308 Mb/s、20,329,900 packet bytes；
两者逐帧代理为 PSNR 48.666916 dB、SSIM 0.994644。该指标只比较两种编码输出，不是修复真值。

同一进程版本的公平性能对照为 CQP `blend-encode=64.7s`，两轮 vbr_peak 为 67.7s 与 64.3s，
没有可声明的稳定提速。内部显存峰值分别为 CQP 9474 MiB、vbr_peak 9498/9490 MiB；24/16 MiB
差异远低于运行波动。整卡最小余量在三轮为 423/88/197 MiB，第二轮恢复证明 88 MiB 不是稳定
额外占用；没有 OOM、native abort、少帧或尾部泄漏。因此本改动的已证收益是体积控制，不宣传性能提升。

NV12/Main 4K、30 fps、300 帧实片使用 target 2,840,512 bit/s，输出 2.760412 Mb/s，源为
2.840512 Mb/s；最大 1 秒 bucket 3.435280 Mb/s，低于 3.550640 Mb/s peak。输出保持 yuv420p，
300 帧、2 个关键帧、PTS 全同并严格解码通过。同参数 CQP 为 6.611747 Mb/s；两者代理为
PSNR 46.487994 dB、SSIM 0.989310。blend-encode 14.6s 对 14.5s，显存峰值 5132 MiB 对
5418 MiB，同样没有性能或显存回退。

正常产品 `--segments 1-2,11-12` 还在无重编码截取的 20.09 秒真实 8K Main10、多 GOP 输入上
完成两次独立 render encoder/session、两个 copy fragment 和三处真实接缝。四个 fragment 的首个
random-access packet 均为依赖顺序正确的 VPS(0) -> SPS(0) -> PPS(0)，最终成片通过产品三段
bounded seam gate、1202/1202 全 packet/PTS 门及额外软件严格解码；归一化最大 PTS 差约 6 微秒，
AAC elementary-stream SHA256 与源一致。源文件 38,589,234 bytes，输出 38,567,096 bytes，
只小 22,138 bytes（0.057%）；视频码率 15.125274 -> 15.114667 Mb/s。

产品默认因此在 Linux AMD HEVC Smart Render 且没有显式码率策略时自动启用该合同；GUI 选择
“自动（匹配源码率）”时，自动预扫描最终选择的 Smart Render 或完整编码都会保留该合同。源码率
缺失或超出安全选项范围时记录 warning 并保留旧 CQP；CLI 完整编码、Windows、NVIDIA、H.264/AV1
以及显式 `rc`/码率/原生 QP 均不变。GUI 手动 CQ 会显式写入 `rc=cqp`；
`JASNA_AMF_HEVC_VBR_PEAK=0` 是可复现的旧 CQP 回退，`=1` 仍可在 Linux AMD HEVC 完整编码上
强制该合同用于诊断。

## GUI 完整编码源码率修复（2026-09-02）

真实第三个 8K Main10 队列视频的自动粗扫覆盖率为 87.0%，因此从 Smart Render 选择切换到
`full`。旧 GUI 只把“自动（匹配源码率）”落实到 Smart Render；完整编码没有收到对应策略，
也没有输出自动 `vbr_peak` 日志，实际回退到 CQ。源片为 4,764,495,424 bytes、视频码率
20.941312 Mb/s，旧成片达到 21,725,705,318 bytes、96.424384 Mb/s；两者均为 107,743 帧、
1797.512383 秒，AAC 均为 256 Kb/s，排除了重复帧、音频和容器开销，确认膨胀来自完整编码 CQP。

GUI 的码率模式现作为显式 session 配置贯通到 Pipeline 和完整编码器。Linux AMD HEVC 自动模式
对 `full` 与 Smart Render 都使用源码率 `vbr_peak`；手动质量继续显式使用 `rc=cqp`。CLI 完整
编码默认仍为原行为，Windows、NVIDIA、H.264 和 AV1 也没有被扩展到尚未验收的自动源码率路线。

真实 GUI isolated worker 用同一 8192x4096、Main10/P010、60000/1001、1202 帧素材强制走
`full`，日志确认 `target=15125274`、`peak=18906592`、`buffer=37813184`、`preanalysis=0`。
源文件为 38,589,234 bytes，输出为 37,865,244 bytes（小 1.88%）；视频码率从
15.125274 降至 14.803080 Mb/s。输出保持 Main10/yuv420p10le、1202 帧和 941 帧 AAC，
源/输出视频 packet PTS 集合、首 PTS `0.038000` 和末 PTS `20.124733` 完全一致，输出
PTS/DTS 严格递增，且通过 FFmpeg `-hwaccel none -xerror -err_detect explode` 软件严格解码。
源素材带有非零视频起点和 B-frame DTS；输出容器报告时长比源多 0.050050 秒，但没有新增或
丢失帧、没有新增 PTS 间隙，音频时长和 packet 数不变，并在产品 0.5 秒时长容差内。

本轮 D2D audit 为 1202/1202、FD close/fail 为 1202/0，全部 host/Map/staging/D2H/bridge
计数为 0。内部显存峰值 8398 MiB，整卡峰值 19471 MiB、最小余量 5089 MiB，pressure、
critical reclaim 和 offload 均为 0；未发现 AMF/D2D/显存错误。

## Linux AMD 8K Main10 AMF host-native 输入（2026-09-03）

编码前 profile 确认 PyAV 的 `HWAccel("amf")` 会在 `avcodec_send_frame()` 前将软件
P010/NV12 上传到 AMF hardware-frame pool，因此 FFmpeg `amfenc` 的软件 host-frame
分支不可达。旧产品实际上只有这一次 host -> AMF surface 全帧复制，并不存在此前假设的
第二次 FFmpeg 内部复制。新的精确优化在已完成 blocking D2H 后绕过 PyAV upload，让
`amfenc` 用默认关闭的 `host_zero_copy` 选项调用 `CreateSurfaceFromHostNative()`，只消除
这一遍 host -> AMF surface 复制；Vulkan -> HIP D2D、AMD `frame.clone()`、
`stream.synchronize()`、blocking D2H 和 AMF keyframe reset 均保持不变，也不使用 rocDecode。

每个 8192x4096 P010 帧约 96 MiB。每个仍在途的提交帧必须有独立 pinned host owner，
FFmpeg 复用现有 attached-frame property 将 PyAV AVFrame/DLPack 引用持有到对应 AMF 输出
完成；单会话的 `async_depth=4` 约束其在途 owner。双 GOP writer 另使用共享、有界、惰性分配
的 pinned owner 池，按输出 packet PTS 确认 AMF 已消费后复用，避免长片中持续向 ROCm caching
allocator 申请同尺寸大块。FFmpeg 对输入执行 fail-closed 检查：
只接受 NV12/P010、编码尺寸完全一致、Y/UV 均有引用、正且相等的 stride、UV 紧跟 Y 的
vertical pitch，并复核 AMF 返回 plane native pointer、HPitch 和 VPitch。任何不匹配直接失败，
绝不静默复制或 CPU fallback。

同一真实 20 秒 8K Main10/P010、1,202 帧的两轮正反 A/B 结果：生产深度 copy writer
均值 122.35 秒，host-native writer 均值 114.9 秒，快 7.45 秒（6.1%）；深度匹配的
copy4 均值 126.6 秒，host-native 快 11.7 秒（9.2%）。blend/encode 跟踪均值约从
240.25 秒降到 230.50 秒（4.1%）。所有有效输出的 1,202 帧、P010、时长、大小和 MP4
SHA-256 完全一致，软件严格解码通过；无 `AMF_INPUT_FULL`、blocking loop 或 native error。

正式提升门使用同一真实长片的连续 24:00–29:00 Smart Render，覆盖用户报告的 24:09、
24:14、24:19、28:15 和 28:22。三个 bounded render span 共 18,300 次 wrap、0 次
AMF-owned copy；最终全长输出为 111,286 帧、8192x4096 P010、1856.621433 秒。产品内建
两个 copy seam 与全 packet/PTS gate 通过；独立 `-xerror -err_detect explode` 软件解码
通过完整五分钟及两个接缝窗口。源/输出 111,286 个显示 PTS 无重复、无缺口，最大配对偏差
约 6 微秒。用户检查五个历史坏帧点的压缩联系表后确认全部正常。

4K HEVC Main/NV12 的 300 帧实片也得到逐字节一致输出并严格解码通过，但短样本收益很小且
出现一次 drain 离群，不能证明稳定加速。后续 5K Main/NV12 单会话矩阵同样只快约 3.9%，
所以 Main8 不单独自动启用 host-native；只有合格的双 GOP 请求才从 5760×2880 起把它作为
必要输入合同。Main10/P010 经 5K 实片矩阵验证，并按等量 host 帧负载推断从 `3840x2160`
等效像素量起自动启用。Windows、NVIDIA、H.264
输出和 AV1 输出保持旧路线。`JASNA_AMF_HOST_ZERO_COPY=0` 是明确回退；`auto` 或未设置采用
上述自动范围。实现位于
`jasna/media/video_encoder.py`，固定 FFmpeg 补丁位于
`patches/ffmpeg/0006-amfenc-wrap-contiguous-host-input.patch`。

## Linux AMD 双持久编码会话按 GOP 并行（2026-09-03）

RX 7900 XTX 当前 Linux AMF runtime 的 capability 只报告一个可显式寻址的 HEVC
hardware instance，`VCNINSTANCE=1` 会被 AMD 样例拒绝，但两个独立 AMF 会话并发仍可由
驱动分配到整卡聚合编码能力。上游 `TranscodeHW` 也把 `THREADCOUNT` 定义为并行 session
数。因此实验路线不是给单个会话设置不存在的 instance 1，而是保留一套解码/AI producer，
将每 250 帧的 closed GOP 交替送入两个在文件生命周期内持续存在的 AMF encoder session。
每个 GOP 都从 IDR 及 VPS -> SPS -> PPS 开始，完成后按源码显示顺序无重编码拼接。

该选择与公开方案的关系：

- AMD `HevcMultiHwInstanceEncode` 是同一帧的横向 split-frame，官方工程师确认当前只允许
  DX11 memory，并且只是驱动 hint；PreAnalysis/preencode、high-motion boost、filler、
  非 frame output、picture transfer 或不兼容码控都会使它失效。因此它只进入 Windows
  待办，不能套到 Linux Host/Vulkan 路线。
- AMD SmartAccess Video 负责在多个 VCN 或 APU+dGPU 间重定向 decode/encode session；
  AMD 对当前问题明确说明是 Windows-only，本机 capability 也报告 `false`。它不能代替
  Linux 的单流双会话。
- NVIDIA SDK 13 的 `AppEncMultiInstance` 同样把输入拆成独立 GOP，由多个持久 session
  thread 编码后按原顺序回写，是本设计最接近的公开实现。NVIDIA 的 SFE 则是单帧横向
  分片；官方注明会降低画质，并且当多 session 已经占满编码引擎时不会再增加总吞吐。
- Intel Media Delivery 的 8K Hyper Encode 样例在单 GPU node、双 VDBOX 上同样按 GOP
  并行；其开发指南还指出 GOP 变大可能增加第二个 adapter 等待时间，异步深度增加会提高
  内存占用，必须按实际流水线找平衡点。它不能直接用于 AMD，但进一步支持“持久 session、
  GOP 级分工、受限队列、实片调参”这组架构选择，而不是为追求双引擎把整套 AI 流水线复制。
- Intel oneVPL 的 multiple-segment 文档同样要求每段从独立随机访问点开始并保持参数集
  兼容，同时提醒拼接段不一定保持全局 HRD。当前 AMF 输出实测
  `vui_hrd_parameters_present_flag=0`，但仍逐 GOP 检查参数集和完整软件解码。

针对当前 Linux AMD 高分辨率 HEVC 产品形态，各候选路线的取舍如下：

| 路线 | 当前平台可用性 | 主要收益 | 主要代价或风险 | 决策 |
| --- | --- | --- | --- | --- |
| 单 AMF session + host-native 输入 | 已可用 | 最简单、显存最低、没有跨 session 码控 | writer 仍是明显瓶颈 | 保留为 CLI 默认和显式关闭双 GOP 时的基线 |
| 两个持久 AMF session 按 closed GOP 交替 | 已可用 | 5K Main8 完整路线快 23.5%，5K Main10 比默认 copy 快 26.4%，8K Main8 快 18.0%；完整 #1 GUI 路线快约 19.9%；writer 通常明显缩短 | 更多随机访问点、约 0.5–0.65 GB 进程 RSS、两个独立码控状态；必须严格检查参数集、PTS、DTS、HRD 和解码 | 已通过 Main8/Main10、全片/Smart Render 正确性门；GUI 默认偏好启用并自动回落，CLI 仍显式启用且严格拒绝不合格请求 |
| 每 GOP 重建 AMF session 或并行子进程 | 技术上可做 | 调度实现直观 | 初始化开销、显存尖峰、失败清理和时间戳交接更差 | 已实测否决 |
| 复制两套解码/AI/编码流水线 | 技术上可做 | producer 也可并行 | 当前模型额外约 7.57 GiB，容易爆显存且结果排序复杂 | 否决 |
| AMD 单帧 multi-HW split-frame | 当前仅 Windows DX11 条件满足时可尝试 | 降低单帧编码延迟，不需要 GOP 重排 | 只是 driver hint、功能组合受限、可能改变 slice/画质；Linux Host/Vulkan 不可用 | 仅列 Windows 真机待办 |
| AMD SmartAccess Video | 当前 Linux 主机不可用 | 在多 VCN/APU+dGPU 间自动调度 session 与跨 adapter 传输 | 官方说明 Windows-only；依赖硬件 capability | 不作为当前方案 |
| Vulkan Video + RADV 多 context | API/驱动已有基础 | 有机会建立 HIP -> Vulkan 反向零拷贝，删除当前 D2H | 应用自行管理 DPB、参考帧和参数；码控行为由实现决定；没有标准化 split-frame/指定 VCN 控制 | 后续独立实验，不能替换当前 AMF 路线 |

主要上游依据：

- [AMD HEVC encoder API](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/master/amf/doc/AMF_Video_Encode_HEVC_API.md)
- [AMD 对双 VCN、SAV 与 split-frame 的说明](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/issues/585#issuecomment-4165598416)
- [AMD SmartAccess Video primer](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/wiki/Smart-Access-Video-Primer)
- [NVIDIA AppEncMultiInstance](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/read-me/index.html#appencmultiinstance)
- [NVIDIA SFE 编程说明](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/index.html#multi-nvenc-split-frame-encoding-in-hevc-and-av1)
- [Intel 8K Hyper Encode GOP 并行样例](https://github.com/intel/media-delivery#running-8k-with-intel-deep-link-hyper-encode)
- [Intel Hyper Encode 开发指南](https://github.com/intel/vpl-gpu-rt/blob/main/doc/HyperEncode_FeatureDeveloperGuide.md)
- [Intel oneVPL multiple-segment encoding](https://intel.github.io/libvpl/latest/appendix/VPL_apnds_b.html)
- [FFmpeg segment/closed-GOP 说明](https://ffmpeg.org/ffmpeg-formats.html#segment)

短样本结果不能外推为长片整体收益。同一真实 20 秒 8K Main10 完整 Jasna 的两轮基线均值为 241.71 秒，两个持久会话、
每会话固定 8 帧 pinned P010 队列的两轮均值为 201.93 秒：整链路快 16.46%，writer
从 114.05 秒降到 58.25 秒，快 48.93%。1,202 帧的两个候选输出逐字节相同，并与基线
逐帧软件解码 hash 相同；关键帧固定在 0/250/500/750/1000，严格解码和参数集检查通过。
峰值整卡显存约 19.92 GB，仍余约 5.83 GB。队列 4 会破坏首帧内容；每 GOP 重开 AMF
会把峰值推到约 24.3 GiB；复制整套 AI/解码流水线会额外占约 7.57 GiB，三者均已否决。

长测还暴露了两个只在较长时间轴出现的收尾问题。旧实现先写 Matroska，容器把
60000/1001 的源码 PTS 量化到 1 ms，72 个 GOP 的最大相对误差达到 1.005556 ms，正确地
被产品 1 ms gate 拒绝。持久会话输出已改为 NUT，以源码 `1/60000` time base 保存，再由
PyAV 在同一进程内按记录的精确帧数直接 remux 成 MPEG-TS；这也删除了每 GOP 启动一次
FFmpeg 的 72-process 收尾成本。501 帧和新的完整 1,202 帧实片均把最大 PTS 差恢复到
5.56 微秒，DTS、关键帧、VPS/SPS/PPS 和软件严格解码全部通过。少于 251 帧时允许没有
收到 GOP 的第二个 session 正常关闭；输入门进一步约束为 HEVC Main10、P010 兼容 4:2:0，
同时兼容当前 bundled ffprobe 将 Main10 profile 报为空但 pix_fmt 明确为 P010 的情况。

该功能在 GUI 工厂预设中默认开启，也可通过 GUI 开关关闭；CLI 仍只通过显式
`--amd-dual-gop-encode` 启用。产品范围是 Linux AMD、HEVC 输出、原帧率、非 fMP4 离线
输出、自动源码率 VBR Peak，以及 Main/NV12 至少 5760×2880，或 Main10/P010 至少达到
3840×2160 等效像素量。完整编码按最终 HEVC 输出合同判断，因此源可以是 H.264 或 AV1；Smart Render
还会复制源包，所以必须是 profile/位深兼容的 HEVC 源码流。完整编码与 Smart Render
`--segments` 的 render fragment 均已接入。每个独立 GOP 固定 250 帧且禁止 B 帧（`bf=0`）；
不支持 60→30 FPS、H.264/AV1 输出、Windows 或 NVIDIA。GUI 对低于各格式门槛的任务自动使用
既有单会话，CLI 显式请求则 fail closed；两者都不静默切换 CPU 或 rocDecode。

正式长测使用从原片 24:00–29:00 无重编码截出的连续五分钟 Main10/P010 输入，共
17,985 帧。双会话路线在 2597.6297 秒内完成，生成 72 个 closed GOP（71×250 帧和一个
235 帧尾段）。输出保持 17,985 个 packet；相对源码最大 PTS 偏差 22.22 微秒，所有 DTS
存在且严格递增，72 个关键帧均带依赖顺序正确的 VPS/SPS/PPS。AAC 的 14,065 个 packet、
时长、extradata 和 elementary hash 与源码完全一致；FFmpeg 原生 HEVC 软件解码器使用
`-xerror -err_detect explode` 完成 17,985/17,985 帧，stderr 为空。D2D audit 也是
17,985/17,985，全部 forbidden host/map/staging/D2H/bridge 计数为 0。整卡峰值约
24,143 MiB、最小余量 417 MiB，没有 pressure、critical reclaim、offload 或致命 AMF 错误。

该长测还发现收尾并不是编码：PyAV 参数集校验和 FFmpeg concat demuxer分别对 72 个约
4.17 秒 MPEG-TS 片段使用默认五秒分析预算，耗时约 257 秒和 256 秒。现在只对 `.ts`
片段使用 `probesize=32768`、`analyzeduration=0`、`fpsprobesize=0`；PyAV 没有在首个
随机访问 packet 找到完整 VPS/SPS/PPS 时仍立即失败。ffconcat 对每个 TS 写入同样的
`option key value`，这是 FFmpeg 官方定义的逐文件 access/open/probe 参数，非 TS 输入
保持原探测行为。72 段实测参数集校验降至 23.989 秒，concat 加产品 packet/PTS 校验降至
19.291 秒，合计约 43.28 秒，较约 513 秒减少 91.6%。同一旧片段集的新旧 MP4 SHA-256
一致；从已验收五分钟成片重拆的 72 段再走正式函数后，17,985 个 VCL NAL 与基线逐字节
一致，包数、PTS/DTS、参数集、音频和严格软件解码全部通过。

另用 200 帧真实 8192x4096 Main10/P010 输入验收短尾：只有 encoder 0 收到一个不足
250 帧的 GOP，encoder 1 没有收到帧也能正常关闭。完整 CLI 用时 74.50 秒，输出恰好
200 帧，最大 PTS 偏差 5.56 微秒，单个随机访问点参数集有效并通过 200/200 严格软件解码。

### 完整 #2 Smart Render 性能与正确性门（2026-09-03）

完整 `hevc-main10-8k-sample.mp4` 使用历史成片逐帧 VCL 对比恢复出的四段 Smart Render
范围，直接跳过粗扫和精扫。最终 88,492 帧中 49,020 帧重编码、39,472 帧源码复制；总墙钟
6,988.69 秒，9 次受控 AMF session recycle，峰值整卡显存 21,714,534,400 bytes
（约 20.22 GiB）。全程没有 pressure episode、critical reclaim、offload、CPU 回退或
rocDecode。

十个 bounded render span 的 `blend-encode` 合计为 6,272.5 秒，即 7.815 fps。为排除短段
初始化开销，只比较六个同为 7,192 帧的长段：双 GOP 平均 902.47 秒、7.969 fps；同素材、
同长度、稳定显存回收后的单 session 实测为 900.2 秒、7.989 fps，双 GOP 慢约 0.25%，属于
持平。双 GOP 的 writer 子阶段从 332.1 秒降至平均 305.6 秒，约快 8%，但检测、修复和混合
成为上游瓶颈，没有转化为整链路加速。早期显存回收尚未完善时观察到的 4.5 fps 不再作为
性能基线；用户后来重跑 #1 的稳定 7.x fps 与这组同素材长段结果一致。

新输出为 3,373,614,343 bytes，旧 #2 成片为 3,367,720,920 bytes，只增加约 5.9 MB
（0.175%），源码率 VBR Peak 的体积行为正常。独立验收确认源片与输出均为 88,492 个视频包，
最大相对 PTS 偏差 11.111 微秒，PTS 唯一、DTS 完整且严格递增；334 个关键帧位置与 copy span
及每段 GOP=250 的预期完全一致，且都带依赖顺序正确的 VPS/SPS/PPS。39,472 个 copy 帧的
HEVC VCL 与源码逐字节一致；69,206 个 AAC packet 的时长、extradata 和 elementary SHA-256
也完全一致。FFmpeg 使用 `-xerror -err_detect explode` 软件严格解码 88,492/88,492 帧，
return code 为 0、stderr 为空，产品的全部接缝窗口与全片 packet/PTS 门同时通过。

以用户此前已确认没有花屏的旧 #2 成片为参考，在四段 render range 内的 325、440、784、
1100 秒各取连续 1 秒、60 帧做软件解码 PSNR/SSIM。新旧成片的平均 PSNR 依次为
48.493547、49.764118、43.539054、45.253110 dB，all-channel SSIM 依次为
0.994270、0.994930、0.983876、0.989047。该指标只表示与旧验收成片的解码像素相似度，
不是修复真值评分。用户随后检查完整 #2 的 16 张接缝压缩联系表并确认没有问题；助手未打开
图片。

### Main 8-bit/NV12 扩展与最终提升门（2026-09-03）

为避免用单个短样本外推，Main8 使用同源正反顺序各两轮比较默认 copy、host-native 单会话和
host-native 双 GOP。3840×2160 的双 GOP 完整路线比基线慢约 1.8%，所以 4K 明确不自动启用。
真实 5760×2880、59.94 fps、1201 帧样片中，默认基线、单会话、双 GOP 的平均 wall time
分别为 102.978、98.962、78.783 秒；单会话只快 3.9%，双 GOP 快 23.5%，writer 从
49.65 秒降至 25.50 秒。8192×4096、1202 帧 Main8 样片中，三条路线分别为 196.394、
191.327、161.105 秒；双 GOP 快 18.0%。双 GOP 的进程 RSS 约增加 647 MiB，但整卡显存
与基线基本相同，5K 矩阵中约为 9.35 GiB。

5K 与 8K 各六个输出均通过软件严格解码、帧数、持续时间、PTS/DTS 和每个 GOP 的
VPS/SPS/PPS 检查，两轮双 GOP decoded framemd5 一致。5K 源到基线/双 GOP 的规范时间基
PSNR 分别为 52.522811/52.797188 dB，基线到双 GOP 为 54.911185 dB；8K 源到两者为
45.532213/45.403337 dB，只差 0.129 dB，没有实质质量退化。最后把强制 host-native 环境
改为 `auto`，再跑同一 5K 正式参数传递路线，79.786 秒完成；日志确认自动选择
Main/NV12 host-native 与双 GOP，1201 帧、5 个随机访问点及全部严格门再次通过。

一份从中段关键帧截取的旧 5K 样片还保留了两个 discard 标记的前导 B 包：容器有 1417 个包，
软件和 AMF 都只输出 1415 幅画面。两个会话和六个分片本身全部成功，最终安全门仍按包数差异
拒绝发布。这是独立的 open-GOP 容器边缘问题，不应通过放宽帧数门掩盖；性能矩阵改用从干净
IDR 开始、无 discard 前导包的 1201 帧纯 stream-copy 样片。

用户随后用 GUI 和复用的预扫描文件完成整条 #1 Smart Render：同为 111,286 帧和 13 个
fragment，新双 GOP/host-native 路线从 15:46:29 到 17:50:44，约 2 小时 4 分；此前单会话
路线从 23:00:26 到 01:35:36，约 2 小时 35 分，端到端快约 19.9%。新成片通过产品完整
packet/PTS 门，最大 PTS 偏差 11.111 微秒；用户确认画面正常。这项长片结果与 5K/8K矩阵
方向一致，双 GOP 因此不再标记为实验：GUI 默认偏好开启，合格任务自动使用，不合格任务回到
既有单会话；CLI 和通用 session 默认仍关闭，只有显式请求才严格启用。

### Main8/NV12 全长显存与 pinned 内存收口（2026-09-04）

短样本没有暴露旧双 GOP writer 的长时间内存行为。旧实现对每帧执行一次
`torch.empty(..., pin_memory=True)`；即使对应 PyAV/AMF owner 已随输出 packet 释放，ROCm
pinned-host caching allocator 仍会保留不断申请过的 48/96 MiB block。真实 8K Main8 长测在
30,564 个修复帧后进程 RSS 峰值达到 14,176,100,352 bytes，整卡显存峰值约 24,176 MiB、
最小余量 384 MiB，并触发一次显存 pressure 安全退出。AMF 审计同时显示每会话 hardware
surface 峰值只有 2，排除了 AMF 持有数千帧的假设。

双 GOP writer 现在让两个 encoder worker 共享一个惰性分配的 pinned NV12/P010 帧池。容量
固定为 `2 * (queue_depth 8 + async_depth 4) + producer 1 = 25`；正常路径依据编码 packet
PTS 回收对应 owner，重复/缺失 PTS 直接失败，worker 异常、未处理队列和 abort 路径也必须配平。
90 秒 Main8 对照中，新旧 writer 的 MP4 SHA-256 完全一致，证明池化只改变 host allocation
生命周期，不改变输入像素、码控、GOP、画质或最终码流。

仅池化后，首个长 worker 的 RSS 峰值已降至 9,765,859,328 bytes，实际只分配 17/25 个槽且
结束时 pinned active 为 0；但单进程连续处理 300–500 秒 render span 后再创建下一组 AMF
reader/encoder，仍会累积 native working set 并触发 pressure。产品原先只给 8K Main10 设置
120 秒 isolated worker 边界；现在精确扩展为：8192×4096 Main8 仅在双 GOP 实际启用时使用
同一边界。每个 worker 完成当前受限 span 后以既有 `amf_session_limit` 正常退出，GUI 父进程
复用已完成 Smart Render fragment 接着运行；没有关闭 Main8/NV12 双 GOP，也不改变较低分辨率
Main8 或单会话路线。

最终正式长测复用已有扫描断点，将原四个 render span 切成八个受限 span，共处理 37,460 个
修复帧并生成 103,092 帧、1719.918200 秒的完整 8192×4096 Main8/NV12 成片。八个 worker
全部为 pressure/critical reclaim/offload 0；最低显存余量约 4.56 GiB，进程 RSS 峰值约
7.45 GiB，每段 pinned active 最终均为 0。跳过扫描后的总墙钟约 3,916 秒（65.3 分钟）。
完整成片与源均有 103,092 个视频 packet 和 80,624 个 AAC packet；最大配对 PTS 偏差约
11 微秒，PTS 唯一且严格递增、DTS 严格递增，13 个片段参数集和 8 个 copy seam 全部通过。
独立 FFmpeg `-hwaccel none -xerror -err_detect explode` 软件严格解码完成 103,092/103,092
帧、返回码 0；AAC ADTS elementary SHA-256 与源完全一致。

共享池同时覆盖 P010，因此又用既有 5760×2880 Main10/P010、20.036678 秒、1,201 帧实片
跑了一次当前产品自动路线。双 GOP 自动启用，池实际分配 8/25 个槽、峰值同时在用 7 个，
结束 active=0；wall time 69.629 秒、writer 15.6 秒，未比此前已验收路线回退。输出保持
P010、1,201 帧和 0/250/500/750/1000 五个关键帧，PTS 唯一、DTS 严格递增、五个参数集入口
有效，软件严格解码返回 0。整卡峰值约 11.0 GiB、最低余量约 13.0 GiB，pressure、critical
reclaim、offload 均为 0。该单次结果用于排除 P010 功能和明显性能回退，不把速度差异单独
归因于池化。

### Main10/P010 5K 扩展（2026-09-03）

对 E/F/G/H 实际挂载盘的约 2,288 个视频做只读探测后，没有找到 4K Main10/P010，找到两部
5760×2880 Main10 候选。其中一部在起始几帧即出现 `Invalid NAL unit size`，按严格输入门
排除；另一部从首个 IDR 无重编码截取 20.036 秒、1,201 帧样片，完整软件严格解码、包数、
PTS/DTS 与 Main10/yuv420p10le 属性均通过后才进入矩阵。按用户要求没有测试 6K/7K。

同源正序和反序各两轮比较默认 AMF-owned copy、host-native P010 单会话与 host-native P010
双 GOP。三者 wall time 中位数分别为 116.652、97.447、85.889 秒；双 GOP 相对默认 copy
快 26.4%，相对 host-native 单会话快 11.9%。writer 从单会话 63.05 秒降到 30.95 秒，
快 50.9%。双 GOP 进程 RSS 中位数增加约 512 MiB，整卡显存没有增加。两个双 GOP 输出
逐帧 MD5 完全一致；六个输出都通过 1,201 帧、20.036688 秒、唯一 PTS、严格递增 DTS、
五个独立 VPS/SPS/PPS 接入点和 FFmpeg 软件严格解码。

去掉实验用 `JASNA_AMF_HOST_ZERO_COPY=1` 后，又用产品 `auto` 跑了一次 5K Main10/P010：
日志自动选择 host-native P010 和双 GOP，86.868 秒完成，writer 31.2 秒；1,201 帧、
20.036678 秒、五个参数集接入点、PTS/DTS 与严格软件解码再次全部通过。

双 GOP 文件比单会话小约 12.1%，这是两个独立码控状态按 GOP 重启产生的码率分配差异。
单会话与双 GOP 成片逐帧比较为 PSNR 56.810522 dB、SSIM 0.998881，未见实质画质退化。
P010 每像素约 3 byte，NV12 每像素约 1.5 byte，因此 3840×2160 P010 与已实测提速 23.5%
的 5760×2880 Main8/NV12 单帧 host 数据量都约为 24.9 MB。用户明确接受这项等量负载推断，
产品门因此扩展到 3840×2160 等效像素量及以上 Main10/P010；4K Main10 尚无直接实片 A/B，
若后续出现质量或性能异常再做定向验证。4K Main8/NV12 仍因实测慢 1.8% 而保持单会话。

继续查询的下一阶段候选是 Vulkan Video：Khronos 的
[`VK_KHR_video_encode_queue`](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_video_encode_queue.html)
提供编码队列、源/DPB 图像、同步和反馈查询，也定义 CBR/VBR 的 average/peak bitrate；
但标准明确保留具体码控行为为 implementation-specific，不能据此假定其输出体积和 AMF
`vbr_peak` 一致。
[`VK_KHR_video_encode_h265`](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_video_encode_h265.html)
提供 GOP/IDR 与 HEVC 码控指引，同时要求应用负责提供符合 HEVC 规则的 codec-specific
参数；Mesa 25.3 也只记录了 RADV 在多 VCN 硬件上为视频编码队列使用额外 context，并不
等价于自动把单流分配到两个 VCN。理论上新建 HIP -> Vulkan 反向零拷贝和同步链可删除
目前必需的 HIP -> pinned-host P010 D2H，但 Vulkan 标准没有 AMD 单帧双 VCN split-frame
或 VCN instance 选择控制，且当前没有 Main10、源码率 VBR Peak、画质和稳定性 A/B，
因此只作为后续独立实验，不替换当前已完成正确性验收的双 AMF GOP 实验路线。

机器可读证据保存在：

- `/mnt/D/AI/amf-unified-work/transactions/jasna-linux-amf-dual-gop-20260903/runs/20260903T011017Z-dual-q8-dual5m/validation-5m.json`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-linux-amf-dual-gop-20260903/runs/20260903T011017Z-dual-q8-dual5m/finalize-fast/validation-finalize-fast.json`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-linux-amf-dual-gop-20260903/runs/20260903T023240Z-dual-q8-short200-product/validation.json`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-linux-amf-dual-gop-20260903/runs/20260903T035623Z-dual-q8-full2/validation-full2.json`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-linux-amf-dual-gop-20260903/runs/20260903T035623Z-dual-q8-full2/quality-comparison-old2.json`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main8-nv12-20260903/RUN_MAIN8_MATRIX.py`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main8-nv12-20260903/quality/`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main8-nv12-20260903/runs/20260903T122113Z-wrap-dual-main8-5k-real-20s/validation.json`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main8-nv12-20260903/runs/20260903T113103Z-wrap-dual-main8-5k-real-24s/run.log`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main10-p010-5k-20260903/inputs/source-5k-main10-ayami-start20s.mp4`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main10-p010-5k-20260903/runs/`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main10-p010-5k-20260903/runs/20260903T143357Z-wrap-dual-main10-5k-real-20s/validation.json`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-main8-long-repro-20260904/runs/20260904T065913Z-gui-lifecycle-main8-8k-savr1062-full/`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-dual-gop-pinned-pool-20260904/runs/20260904T092000Z-gui-main8-pool-bounded-full/`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-dual-gop-pinned-pool-20260904/runs/20260904T092000Z-gui-main8-pool-bounded-full/strict-software-decode.log`
- `/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main10-p010-5k-20260903/runs/20260904T105259Z-wrap-dual-main10-5k-real-20s/validation.json`

### 双 GOP 生命周期与 full 路线 OOM 防护（2026-09-14）

后续修复的重点是资源生命周期，而不是恢复已删除的 rocDecode 或隐藏 AMF
错误。双 GOP 的共享 pinned-host pool 现在有明确的 close 边界：只有全部 packet
PTS lease 释放后才关闭，关闭时排空空闲 tensor、记录 peak allocated，并主动请求
ROCm allocator trim；异常队列、abort 和重复/缺失 PTS 仍然 fail closed。DLPack
`VideoFrame` 在每次提交后也会立即丢弃 Python owner，避免最后一帧跨越下一次队列等待。

AMF decoder teardown 显式调用 `CodecContext.close()`，再关闭输入容器；AMF→HIP
interop 在 close 或校验失败后会停用 reader lease，解除 bridge 和 consumer-stream
引用，并保留已关闭的 audit 计数供诊断，失败的 session 不能在同一进程中复用，同时
保留原始 native 异常。带 resource-cache 的路径会先同步 consumer stream，再关闭
cache/session，避免 Vulkan surface 在 HIP 仍有引用时被回收。现有 null-stream source
release、private-deferred device-wait 与 transport audit 合同没有放宽。

对会触发 Linux AMF 全局 working-set 累积的隔离 AMD 8K HEVC full 路线，pipeline
现在按受限时长切成独立 closed-GOP render fragments，并把每个已完成片段写入签名绑定
的 Smart Render workspace；片段完成后退出整个 isolated worker，由 GUI 父进程等待
整卡资源回落并从下一个片段恢复。这样把实际 native 生命周期边界从“同一进程重建
decoder/encoder”收紧为“每个片段退出进程”，同时不会重复渲染已完成前缀。当前
fragment 在 native 压力下保持 `running`，新 worker 会安全重做；失败 workspace 会
保留用于诊断。

原生 HIP/MIGraphX 错误有时会在子进程内以 `SIGABRT`（或同类 `SIGBUS`、`SIGILL`、
`SIGSEGV`）终止，无法先发出 Python 层的结构化错误。父进程现将这些信号纳入同一
隔离恢复边界：等待整卡余量稳定后，最多启动两个 fresh worker，复用 canonical
workspace 中已完成且校验通过的片段；连续崩溃后 fail-closed 并保留工作区。该恢复
策略不改变 B1、双 GOP、AMF D2D 或 rocDecode 禁用状态，也不把 SIGTERM/SIGKILL
误判为可恢复故障。

隔离 worker 另有 host RSS/可用内存监视：连续达到 75% RSS 或可用内存低于 10% 时
主动取消并发出结构化 `host_memory_pressure` 事件，避免等待 Linux OOM killer 直接杀掉
GUI；双 GOP 取消竞态也不会继续对部分 NUT 做分包，从而避免把次级 packet 错误覆盖在
原始压力原因之上。该路径仍需在目标 8K Main8/Main10 实片上完成长测后，才能宣称最终
性能和稳定性验收通过。
