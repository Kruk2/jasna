# Linux AMD AMF Vulkan→HIP D2D Core

本文记录从 `codex/linux-amd-amf-interop-core` 开始的独立研究入口。基础 PR 建立
H.264/HEVC 的 native bridge；后续 `codex/linux-amd-amf-av1-native` 复用同一
bridge 扩展固定格式 AV1。下文保留其显式研究入口，同时记录后来已经通过真实媒体
验收并进入 Linux AMD `auto` 的窄范围产品策略。

## 使用边界

诊断时可以用下列环境变量显式强制进入此路径：

```bash
JASNA_DECODE_BACKEND=amf-interop
```

接受范围如下：

| 编码 | profile / 位深 | native surface |
| --- | --- | --- |
| H.264 | Main 或 High，8-bit | NV12 |
| HEVC | Main，8-bit | NV12 |
| HEVC | Main10，10-bit | P010 |
| AV1 | Main，8-bit | NV12 |
| AV1 | Main，10-bit | P010 |

reader batch 只能是 `1`、`2`、`4` 或 `8`；Linux 以外、非 AMD、其他
profile/pixel format 均会在打开前报错。本 core 只接受一个 session 内分辨率、
位深和 surface format 不变的输入，不覆盖中途重配。HEVC 的旧 ffprobe 元数据若
没有 profile，只有在 NV12/8-bit 或 P010/10-bit 的同一范围内才会推断为对应的
Main/Main10。

本 core 最初提交时没有修改 `auto`、`pyav-hw`、`pyav-sw` 或 VALI 的选择。后续产品
接入已经让上述合格 Linux AMD 输入由 `auto` 选择本路线。旧 rocDecode 后端及其
Python/C++ bridge 已永久删除：运行时没有自动选择、诊断入口或 fallback；遗留配置值
`rocdecode` 只会作为未知后端 fail closed。AMF 入口失败时也不会转为 CPU、host Map、
staging、D2H 或其他 decoder fallback。

## bridge 与生命周期

`scripts/amf_surface_probe.pyx` 只包含 decode 方向：

1. 验证 PyAV frame 是 `AV_PIX_FMT_AMF_SURFACE`、AMF Vulkan memory 和有效的
   external-memory handle。
2. 查询 AMF context/Vulkan device，并由 reader 固定 AMF frame-context、AMF
   context、Vulkan device 和 HIP device identity。
3. 导出 Vulkan opaque FD，导入为 HIP external memory 后立即由调用方关闭该
   dma-buf FD，再映射并以两个 `hipMemcpy2D(..., hipMemcpyDeviceToDevice)` 复制
   NV12/P010 的 Y/UV 平面。ROCm 7.2.1 不会替调用方关闭成功导入的 FD，不能把
   `hipImportExternalMemory` 成功误当作 FD 所有权转移。
4. 在 PyAV frame 可释放前同步 HIP null stream；随后释放 mapped buffer 和
   external-memory import。任一步失败均保留 native 原因并报错。

bridge 和 reader 都输出 transport counters。reader 会拒绝非 D2D copy、host
frame transfer、CPU Map、staging、D2H、`av_hwframe_transfer_data`、失败 copy、
FD close 失败、identity 变化，或 export/FD close/import/map/release/destroy 与
copy 数不配平的情况。每次 export 必须恰有一次可审计 close，最后一次 close
errno 必须为 0。固定 context session 也必须 create/close 配平；关闭失败不会被
吞掉。

产品 INFO 日志只输出一次紧凑 D2D 审计摘要；带
`AMF interop transport stats reader=` 前缀的完整 JSON counters 保留在 DEBUG
日志。这样正常 GUI 不会被长 transport 字典刷屏，诊断事务仍可在 DEBUG 级别完整
解析生命周期证据。

2026-08-30 的 8K HEVC Main10、3600 帧真实 GUI 全流程中，粗扫/检测/对齐 reader
分别得到 12、3600、3600 次 D2D；三者的 Vulkan export FD close、fixed-context
create/close 全部配平，host/CPU Map/staging/D2H/non-D2D/failed bridge 全为 0。
正常 INFO 每条 reader 只输出一行紧凑摘要，完整 transport JSON 只出现在 DEBUG。
证据位于 `jasna-gui-8k60s-prescan-amf-reuse-20260830/VERIFICATION.txt`。

固定 runtime 在 `surface_pool_size` 保持默认 `-1` 时会自行派生 36 个 AMF decode
surfaces，并给 AVHWFrames pool 追加 8 个。
曾验证显式写成 `0` 会使 AVHWFrames pool 只剩 8 个，B8 reader 在持有首批 8 个
surface 后等待下一帧而停滞；因此使用 upstream 默认计算，不把 decoder pool 误当成
本 core 默认关闭的 external-memory mapping cache。

2026-09-01 又在真实 8K HEVC Main10/B4 Smart Render 上验证了
`surface_pool_size=8`：固定 runtime 会创建 16 个 AVHWFrames surface，但
decode/detect reader 仍会在 `CodecContext.decode()` 内发生 surface starvation；其余
队列为空、整卡显存余量充足，证明这不是显存压力。`0` 和 `8` 都已否决。

同日继续验证 `surface_pool_size=16`。在 8192×4096、HEVC Main10/P010、B4、
fisheye、Smart Render 自动 `vbr_peak` 的 1202 帧真实全流水线中，两条 reader 都完成
1202/1202 D2D，FD/export/import/map/release/destroy 配平，host/Map/staging/D2H/
failed bridge 全为 0。整卡峰值由 runtime 默认池的约 23,944 MiB 降至 16,713 MiB，
最小余量由 616 MiB 提高到 7,847 MiB，且不再触发 whole-card pressure episode。
两份输出逐字节相同，SHA-256 都是
`02fa2b695d5a7f60e69c8b1ee17fb8accc968fa93b20c74de2618a48899051ad`；均为
1202 帧、20.103411 秒、HEVC Main10/yuv420p10le，含 941 帧 AAC，系统 FFmpeg
`-xerror -err_detect explode` 严格解码通过。

完整流水线 pool16/default 的 wall 分别为 224.39/217.03 秒，但分段 timing 中 pool16
的 decode/detect、primary 和 blend decode 都更快；差异来自一次 AMF encoder write
波动。额外的双 reader decode-only A/B 对同一 1202 帧源得到完全相同的 PTS SHA-256，
两边均为 1202×2 次 D2D；pool16/default wall 为 43.86/44.13 秒，整卡峰值为
11,127/17,887 MiB。该对照确认 pool16 没有引入可测的解码性能损失。

因此产品只在 `Linux + AMD + HEVC Main10/P010 + 8192×4096 + B4` 自动使用 16；
统一 runtime 若省略 HEVC profile，只有 `is_10bit + P010` 才允许按 Main10 推断。
Windows、NVIDIA、8-bit、其他分辨率和 B8 仍保留 runtime 默认池。
`JASNA_AMF_INTEROP_SURFACE_POOL_SIZE` 继续作为诊断覆盖，显式值必须不小于 16。

本 core 不实现资源 mapping cache。`JASNA_AMF_INTEROP_RESOURCE_CACHE` 默认
`false`；若显式设为 true，explicit backend 会 fail closed，而不是启用未验证的
缓存。

## reader→caller stream 所有权修复（2026-09-01）

一条 8192×4096、HEVC Main10/P010、B4 的正式 Smart Render 成片曾在四处产生连续
花屏。四处首坏帧分别为全局帧 101035、106975、107815、108115，全部是 B4 的最后
一帧（`frame % 4 == 3`），并且全部位于下一个源关键帧前第 5 帧。异常随后通过 HEVC
预测传播；源关键帧提供干净输入后才逐步恢复。四组都位于同一个 render span 中段，
不是拼接 seam；encoder 输入 clone 也早已存在，不能解释该模式。

AMF reader 的 private stream 在写完一个 B4 后会同步再 yield，但同步只保证像素已经
可读，不会把后续 caller stream 的使用登记给 Torch caching allocator。generator
推进并释放旧 batch 后，下一批同尺寸分配可能在 caller 仍读取旧 view 时复用 storage。
一个真实 ROCm allocator 最小复现直接验证了这一点：旧路径三次中两次复用同一
pointer，延迟 consumer 读到覆盖值；在实际 caller stream 对 source batch 调用
`record_stream()` 后，三次都没有提前复用且 consumer 值正确。

正式策略因此限定为：只有已经打开且 `_amf_interop_enabled is True` 的 reader 默认
使用 `record-stream`，并且 DecodeDetect 与 BlendEncode/PtsAlignedFrameReader 各自在
自己的实际 caller current stream 登记整批 storage。NVIDIA、CPU、VALI、PyAV 软件
上传和其他非 AMF reader 默认仍为 `off`。PTS recovery 每次重开 reader 后重新根据
该 reader 的实际 route 选择策略。

`JASNA_AMF_READER_CALLER_HANDOFF` 保留以下显式覆盖：

- `off`：回滚到旧所有权行为；
- `record-stream`：强制只登记 caller stream；
- `record-stream-clone-batch`：先登记 source，再在 caller stream clone 整个 B4，仅供
  诊断。

环境变量未设置时采用上述按实际 AMF reader 自动选择的产品策略。8K 全流水线只剩约
182 MiB 整卡物理显存余量，因此没有冒险运行 whole-B4 clone；它不作为产品默认。

同一份从真实关键帧无重编码截取的 8K Main10/P010 样本共 333 帧，本地第 295 帧正好
对应首个 `K-5`，第 300 帧为源关键帧。`off` 与 `record-stream` 输出逐字节相同，
SHA-256 均为
`383e0cc52b2d99d2f9c53e03403994fe7589654b91d6f74aa0f1fe2b7ce0862b`；两者都严格
软件解码通过，输出 333 帧，最大 PTS 差为 0，目标窗口 PSNR 约 45.69–46.70 dB。
blend/encode wall-clock 为 100.5/100.6 秒。旧路径内部/整卡峰值为 6708/24371 MiB，
`record-stream` 为 6914/24378 MiB；最小整卡余量分别为 189/182 MiB。本次短片旧
路径没有自然触发非确定性花屏，因此 allocator 复现承担因果证据，真实短片承担格式、
内容、时间戳和性能回归证据。

另用现有 3840×2160、HEVC Main、8-bit yuv420p/NV12、300 帧素材验证完整双 reader
与 AMF encoder 路线。显式 `off`、显式 `record-stream` 和未设置环境变量的自动默认
三个输出逐字节相同，SHA-256 均为
`0a6b76dc15bf00825b56ea53139bad50812b1b4a49b29eff54ba3c2ac1835830`；三者都是
300 帧、10 秒、最大 PTS 差 0，严格解码通过。自动默认日志同时确认两条 reader 均为
`hevc_amf + explicit AMF Vulkan→HIP D2D`，且两个 caller role 都启用
`record-stream`。与源对比的逐帧 PSNR 最低 33.208879 dB、平均 35.113118 dB，没有
全帧崩坏。显式旧/新路径的内部峰值为 5196/5400 MiB，整卡峰值为 7411/7611 MiB，
最小余量为 17149/16949 MiB；约 204 MiB 的差异符合 allocator 延后复用，未发生
offload、native/FFmpeg 错误或 CPU fallback。

代码焦点测试覆盖 AMF 默认登记、非 AMF 默认不变、显式关闭与 clone 顺序，以及 PTS
recovery 重开 reader 后重新选取正确模式。媒体与完整复现记录保存在仓库外本地事务
`jasna-hevc-b4-caller-handoff-20260901/`，不进入仓库。

## AMF HEVC 关键帧状态重置（2026-09-02）

caller-side `record_stream()` 修正了真实的 Torch storage 生命周期缺口，但后续逐层
指纹证明，它不是这批随机花屏的充分修复。稳定复现从源片 1438.438 秒附近开始连续
解码，在 1444.30–1444.48 秒窗口得到四个错误 P010 帧，源 time base 下 PTS 分别为
86661575、86662576、86663577、86664578。AMF 原生 P010 surface 在任何 RGB 转换、
ROI 修复、blend、AMF encode 或 Smart Render 拼接之前已经与 FFmpeg 软件基准不同，
因此 bridge/caller storage、restorer、encoder 和接缝都不是这四帧的首个出错层。

同一 11 帧窗口若在 1439.438 秒关键帧新建 AMF decoder，P010 指纹完全一致；对旧
component 只做 `Drain + Flush` 不能清除错误状态，而 `Drain + Terminate + Init` 后再
提交该关键帧可以稳定得到 `p010_mismatches=[]`。这把根因收窄为 Linux RADV/AMF 在
长生命周期 8K HEVC Main10 decoder 中跨 GOP 保留了错误的内部状态，而不是源片本身、
时间戳或编码后传播。

固定 FFmpeg 基线因此增加默认关闭的 `reset_on_keyframe` decoder 选项。第二个及以后
的关键帧到来时，decoder 先完整 drain 旧 GOP 的全部 reorder 输出，再
`Terminate()` / 以原尺寸 `Init()`，最后重新提交保留的关键帧包。HEVC decoder 同时
启用 `hevc_mp4toannexb`，保证重初始化后的关键帧携带 VPS/SPS/PPS；`SurfaceCopy`
属性改为 BOOL。启用 reset 时必须同时设置 `copy_output=1`，因为禁用独立输出 surface
会在下游仍持有旧 AVFrame 时导致 AMF 退出码 139，不能作为产品路径。

Jasna 只在 `Linux + AMD + 8192×4096 + HEVC Main10/P010 + B4` 的已验范围自动设置：

```text
surface_pool_size=16
copy_output=1
reset_on_keyframe=1
```

Windows、NVIDIA、8-bit/NV12、其他尺寸和 B8 均不启用；FFmpeg 选项本身默认关闭。
这条修复没有引入 CPU 或其他 decoder fallback，也没有恢复已经永久删除的 rocDecode。

正式验收使用同一用户提供的 8192×4096 HEVC Main10/B4 源片的
24:00–29:00 连续五分钟效果范围，并由正式 Smart Render 路线保留其余 copy span。
三个隔离 child 依次完成两个约 120 秒 render span 和最后约 65 秒 render span：

| child | decode/detect D2D | blend/encode D2D | 整卡峰值 | 最小余量 | pressure / critical / offload |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 7196/7196 | 7192/7192 | 22,301 MiB | 2,259 MiB | 0 / 0 / 0 |
| 2 | 7196/7196 | 7192/7192 | 22,892 MiB | 1,668 MiB | 0 / 0 / 0 |
| 3 | 3920/3920 | 3916/3916 | 23,008 MiB | 1,552 MiB | 0 / 0 / 0 |

三轮的 FD close 均与 copy 配平，host/CPU Map/staging/D2H/bridge failure 全为 0；
运行窗口没有 GPU reset、ring timeout、page fault 或 OOM。显存随每个隔离 child
退出回落，没有按 GOP 或 span 线性泄漏，也没有用运行时回收循环限制吞吐。

同一源、同一 24:00–29:00 范围的修复前 process-recycle 候选路线可作近似性能对照：
旧/新三轮 `blend-encode` tracked 合计为 3013.3/2966.9 秒，新路线快 1.54%；按三个
child 日志/`time` 计的 wall 合计约为 3183/3161.24 秒，新路线快 0.68%。这不是隔离
其他系统负载后的微基准，但至少没有显示 keyframe reset 降低端到端性能。两份成片大小
仅差 65,121 bytes（约 64 KiB）。

`copy_output=1` 会为独立 AMF output surface 付出 native 显存，因此新路线三轮整卡
峰值比旧候选的对应三轮高约 1.67–4.29 GiB；这是避免下游仍持有 surface 时退出码 139
所必需的正确性成本。24 GiB 实机最差仍有 1,552 MiB 物理余量，且三个 child 都是
0 pressure episode、0 critical reclaim、0 offload，未发生爆显存或回收限速。该成本
只落在上述精确 8K/Main10/B4 自动范围，不影响其他格式和平台。

最终 MP4 为 8192×4096、HEVC Main10/yuv420p10le，视频 111,286 帧、
1856.621433 秒，与源片帧数和视频时长完全一致。全片包级独立对照得到 PTS 唯一、DTS
严格递增，输出换为 1/90000 time base 后最大归一化 PTS 偏差 5.555556 微秒；五个
fragment 的 VPS/SPS/PPS 检查与两个 copy seam（1438.438、1744.743 秒）严格软件解码
均通过。系统 FFmpeg 另对 1438.438 秒起连续 302 秒执行
`-xerror -err_detect explode -hwaccel none`，完成 18,101 帧且零错误。

最后对精确 24:00–29:00 的 17,982 帧同时软件解码源片与成片，缩至 256×128 后逐帧
比较。两路帧数完全一致、FFmpeg stderr 为空；用户确认过的 24:09、24:14、24:19
附近，99% 像素绝对差均不超过 4，绝对差不小于 32 的像素比例为 0–0.0061%，没有
旧成片的全帧/大块异常特征。验收产物和数值日志保存在仓库外本地 AMF 诊断事务
`keyframe-reset-5m/`，不进入仓库。

通过用户联系表确认后，验收构建已用原子安装器发布到
`~/.local/share/jasna/unified-runtime/linux-amd`；旧 runtime 保存在同级
`linux-amd.backup-20260902-200444`。安装后 `libavcodec.so.62.36.101` SHA-256 为
`cc2dc10463d13aacf36a5e5be252f2761a518ab356a30b48ecc0b6774b90f4c3`，bridge SHA-256
为 `1f6b3ee57b329c422a7ab65243df4504184883cf0b8c87c9b6559d03107eb158`；完整 runtime
preflight 通过。再用安装目录本身重复固定 11 帧指纹，仍得到
`p010_mismatches=[]`。

## 构建与 runtime 前提

用以下 helper 在源码树外构建 extension：

```bash
python scripts/build_amf_surface_probe.py \
  --pyav-source /path/to/pyav-source \
  --amf-include /path/to/amf-include \
  --ffmpeg-include /path/to/ffmpeg-include \
  --ffmpeg-lib /path/to/ffmpeg-lib \
  --vulkan-include /path/to/vulkan-headers/include \
  --rocm-include /opt/rocm/include \
  --output-dir /tmp/jasna-amf-interop-bridge
```

extension 必须与实际 PyAV/FFmpeg ABI 匹配。当前 unified runtime 的 runtime
contract 不会自动把 `bridge/` 加入 `PYTHONPATH`；这是本 PR 保持的显式研究入口
前提。运行实验时由操作者同时提供 ABI-matched runtime `site-packages`、bridge
目录和 FFmpeg `lib` 到对应的 Python/loader 搜索路径。普通开发 venv 没有 bridge
时会明确 fail closed。

## 独立验收

焦点测试位于 `tests/test_amf_interop_core.py`，覆盖 backend/env、Linux AMD
scope/batch 矩阵、Windows/NVIDIA/不支持格式拒绝、auto 范围判定、bridge 缺失、non-native
frame、transport counter 拒绝、close 配平和 cache 默认值。现有 decoder backend
与 AMD software-path 测试也作为回归检查。

主会话用 accepted Linux AMD unified runtime（PyAV 18.1.0，固定 FFmpeg/PyAV/AMF
source pin）重新在源码树外构建 bridge，产物 SHA-256 为
`b5efc58113e55a64545db647b3fce5723b490cc857e3fc08a52b46347dd2a9dc`。随后以
`JASNA_DECODE_BACKEND=amf-interop`、batch 4 对三份静态 fixture 做完整只读验收，
没有生成输出媒体：

| 输入 | 完整帧数 | 输出 tensor | PTS |
| --- | ---: | --- | --- |
| H.264 Main 8-bit，3840×2160 | 120/120 | `N×3×2160×3840` uint8 HIP | 0–122122，与独立 FFprobe 逐帧一致且严格递增 |
| HEVC Main 8-bit，4096×2048 | 120/120 | `N×3×2048×4096` uint8 HIP | 0–121121，与独立 FFprobe 逐帧一致且严格递增 |
| HEVC Main10，4096×2048 | 120/120 | `N×3×2048×4096` uint8 HIP | 0–119119，与独立 FFprobe 逐帧一致且严格递增 |

每一份 120 帧输入均得到 120 次 Vulkan export、HIP import/map/release/destroy、
120 次 source-release stream synchronize 和 240 次 D2D plane copy；固定 context
session 均为 create 1 / close 1。host transfer、CPU Map、staging、D2H、
`av_hwframe_transfer_data`、failed bridge、cache hit/miss 均为 0。三次运行后的最高
GPU junction 为 66°C、memory sensor 为 68°C，低于既定停止门槛。另以 H.264
Main fixture 做过 B8 中途关闭：消费首批 8 帧后主动关闭 iterator，8 次
export/import/map/release/destroy、16 次 D2D plane copy 和 session 1/1 全部配平，
没有预取未消费的下一组 AMF surfaces，也没有残留测试进程。

最终集合另用 8192×4096、60000/1001 fps 的 HEVC Main10 实际素材完成 B4
private-deferred decode-only 1400 帧回归，超过修复前约 999 次 export 后失败的
位置。1400 次 Vulkan export、FD close、HIP import/map/release/destroy 全部配平；
每 100 帧读取一次 `/proc/self/fd`，从 100 到 1400 帧均为总 FD 9、dma-buf FD 0，
reader 关闭后回到总 FD 6、dma-buf FD 0。PTS 严格递增，junction 峰值 68°C、
显存使用峰值 11,259,826,176 bytes，运行窗口无 GPU reset、ring timeout、page
fault 或 OOM。证据保存在
`transactions/jasna-upstream-pr-desktop-cutover-20260830/fd-leak-fix/`；测试媒体不
进入仓库。

`dynamic-fixtures-rebuilt/hevc-dynamic.mkv` 不是静态 HEVC Main fixture：它在第 31
帧从 640×320 8-bit 切换为 1280×640 10-bit。该输入违反 fixed-context/fixed-format
边界，AMF decoder 在重配后的 frame 能交给 reader 拒绝前于 Vulkan fence 等待中
终止进程。因此本 PR 不声明支持中途分辨率/位深重配；该限制不能用首批成功掩盖，
也不启用 CPU fallback。运行后 kernel journal 没有 GPU reset、ring timeout、page
fault 或 OOM。

动态重配仍保持独立工作项；当前 auto route、编码与产品流水线的后续实机结果记录在
本文后续章节。AV1 的额外元数据边界与实机结果见 `docs/AMF_AV1_NATIVE_CN.md`。

## Linux AMD GUI 每视频进程隔离（2026-08-31）

Linux AMD GUI 视频任务现在以“一条视频一个 child 进程”承载完整生命周期：自动
预扫描、RF-DETR MIGraphX、BasicVSR++、两条 AMF reader 和 AMF encoder 都留在同一
child 内；下一条视频必须使用新的 PID。长驻 GUI 不再预热 HIP context。该边界只对
Linux + ROCm 视频生效，不改变 Windows、NVIDIA、图片任务、`auto` 判定顺序或
其他现存 decoder 的边界；永久删除的 rocDecode 不会由隔离 child 恢复。父进程仍负责
Stop/Pause、最终输出新鲜度检查和原子完成提交；
child 报出的 100% 在父进程验收前只作为 99.9% 转发。

真实验收使用同一份 120 帧、8192×4096、HEVC Main10/P010 素材连续运行两次
“精扫→full 修复”。两个 child PID 分别为 389535 和 390733，每个 child 均得到
12/16 次扫描 D2D copy，以及 120/120 次完整流水线两 reader D2D copy。所有 compact
audit 的 cache hit/miss 为 0/0，FD close 与 copy 配平，host/CPU Map/staging/D2H/
failed bridge 全为 0；compact audit 只有在 fixed-context session close 等完整
transport 断言通过后才会输出。两个 child 退出后的显存分别回到 1651.8 MiB 和
1647.8 MiB，接近 1653.8 MiB 启动基线。运行峰值显存 23207.4 MiB、junction 75°C，
磁盘最低可用 77.272 GiB，窗口内无 OOM、GPU reset、ring timeout 或 page fault，
且没有残留进程组或 partial/staging 输出。

两个输出均为 120 帧、12.012 秒、8192×4096 P010，PTS 严格递增；系统 FFmpeg
`-xerror -err_detect explode` 严格解码通过。两个 MP4 的 SHA-256 同为
`1f805d6859d630bbd937a5522b6dfd6724424278a6da2dbb97931fe35b9f403c`，逐帧
SHA-256 framehash 流汇总同为
`1732b9bae44018ab70569fca955b5d763050ffd9460a5c1c592a5a7a72c25be0`。

同一事务的 attempt003 曾在首个 child 重载 MIGraphX 后、正式 reader 启动前停在
`VramOffloader` 初始化日志，900 秒内显存约 5.26 GiB、温度 54–55°C，journal 也没有
GPU 故障。该次没有生成输出，不能伪装成成功；精确 PGID 终止后显存回到约 1.65 GiB。
attempt005 随后的两个独立 child 都完整成功，因此进程隔离与退出回收已实证成立，
但这一次无 GPU 错误的随机启动停顿仍保留为已知观察项。正常编码尾部会在修复进度
99.9% 后保持约 20 GiB 显存约 35 秒，随后输出 `blend-encode` timing 并完成；不能仅凭
短时间没有进度更新把它误判为 attempt003 的启动停顿。

完整证据位于仓库外本地事务 `jasna-8k-vram-fix-20260831/`。

## 8K 整卡显存余量与 AMF session 启动边界（2026-09-01）

长 Smart Render 事故中，Torch/ROCm 只报告约 8–10 GiB，DRM sysfs 却显示整卡已
长期使用 24.1–24.37/24.56 GiB；随后 kernel 出现 display framebuffer pin `-12`
和 KFD queue eviction，AMF 最终报 `CopySurfaceRegion() Copy thread timeout`。
原 `VramOffloader` 只看 Torch allocator，不能统计 AMF/Vulkan decoder/encoder 或
MIGraphX 的原生分配，因此不能作为整卡保护。

2026-09-01 的同一真实长片后续失败进一步确认了这一点：第三个 47,686 帧 render
span 的 blend reader 只完成 8/9 次 D2D，第 9 次在
`hipImportExternalMemory` 返回 HIP 2，报
`private-deferred AMF-to-HIP D2D copy failed`。失败前 DRM 最小物理余量只有 209 MiB，
前一个 span 曾低至约 10 MiB；host/Map/staging/D2H 均为 0。这不是坏帧、CPU fallback
或 `vbr_peak`，而是 runtime 默认 decoder pool 把 native 显存推到极限。失败的
Smart workspace 保留在本地输出目录、不进入仓库；span 0–4 complete，失败 span
下次打开会由 running 自动恢复为 pending。

Linux AMD 正式流水线现在同时记录 Torch 与 DRM sysfs 两套口径。4 GiB 只保留为
8K/B4/P010 native session 的启动前预算，不再作为每 100 ms 检查一次的运行时回收线。
Torch 侧按帧尺寸、位深、batch 和两条 reader 的真实工作集推导 reserve；当前
8K/B4/P010 约为 2,862 MiB。整卡运行时水位按显卡容量缩放并设上下界：24 GiB 卡的
pressure/recovery/critical 约为 1.0/1.5/0.5 GiB。

普通 pressure 必须持续 2 秒，每个 episode 最多有限卸载 256 MiB restoration tensor
并只 trim allocator 一次；余量恢复到 recovery 以上还必须持续 5 秒才重新布防，随后
普通 episode 另有 30 秒冷却。因此 4 GiB 以下不会反复搬数据，也不会因短暂水位抖动
频繁卸载。只有余量低于 critical 持续 1 秒时可以绕过冷却，并且每个 episode 仍只做
一次 emergency cache reclaim。运行中的 AMF/Vulkan surface pool 不动态缩容，
B4/B8 也不会在同一视频中途改写用户设置。DRM 统计包含 Jasna、桌面/远程软件、
AMF/Vulkan 与其他进程的整卡显存总和；日志摘要给出整卡峰值、最小余量、pressure
episode 和 critical reclaim 数。

Smart Render 不再让两条大型 AMF decoder pool 跨多个 render span 常驻；每个 span
由正式 pass 自己打开并关闭 reader，关闭时同时断开 PyAV decoder、stream 与
container 的 Python owner，避免已结束 span 的 Vulkan surface pool 被 reader 对象
继续引用。上述已验收的 8K/Main10/B4 路线每次打开使用 pool16，其余路线仍保持
runtime 默认值。

另一个与显存无关的低温、低显存启动停顿被收窄到两条正式 reader 同时进入 AMF
`CodecContext.open()`：同一源的两个 FD 已打开，但两条成功日志都没有出现。现在只对
一次性的 AMF decoder create/open 使用进程内锁；两个 session 打开后，逐帧解码、
检测、修复、blend 和编码继续并行。该锁不改变 Windows/NVIDIA、共享 auto 判定顺序或
每帧热路径。

最终实机验收连续运行两个独立 child，均完成 8K HEVC Main10、B4、自动精扫到 full
修复的 120 帧闭环。两个 child 的整卡峰值分别为 22,211/22,278 MiB，最小物理余量
2,349/2,282 MiB，分别卸载 118/117 MiB；温度峰值 74°C，退出后显存回到约
760 MiB。每个 child 的两条正式 reader 都为 120/120 D2D，FD close 配平，cache
hit/miss 为 0/0，host/CPU Map/staging/D2H/failed bridge 全为 0。两个输出均为
120 帧、12.012 秒、8192×4096、HEVC Main10/yuv420p10le，PTS 严格递增且 strict
decode 通过；文件与逐帧 framehash 都完全相同。运行窗口没有新增 OOM、GPU reset、
ring timeout、page fault、framebuffer pin failure 或 queue eviction，也没有残留
worker。

同一正式 8K Main10 长片随后完成 47,686 帧修复跨度和 111,286 帧最终 Smart Render
成片。pool16、自动 `vbr_peak` 路线的 `blend-encode` 全程 timing 为 7,100.3 秒，平均
6.716 fps；改动前同一源、同一 47,686 帧跨度、runtime 默认 pool 的记录为 7,047.5 秒，
平均 6.766 fps，当前差值为 -0.74%。这两轮之间编码控制已由 CQP 改为自动
`vbr_peak`，因此不是严格单变量 A/B，但长跨度结果没有显示有实际意义的性能回退；
上文同版本 decode-only 单变量 A/B 仍以 pool16 的 43.86 秒对默认 pool 的 44.13 秒确认
解码吞吐没有损失。长片整卡峰值由旧轮的 24,298 MiB 降至 20,748 MiB，最小余量由
262 MiB 增至 3,812 MiB，offload 由 16,769 次降至 0，pressure episode 和 critical
reclaim 也均为 0。最终日志保存在本地输出目录的 `.jasna-logs/` 中。

另以同一 8K Main10 源构造两个连续 Smart Render 修复区段（7+8 帧）验证 per-span
reader 生命周期。两个 span 都分别打开、审计并关闭两条 reader，每条审计为 8/8
D2D、FD close 配平且 forbidden counters 全 0；整卡峰值分别为 19,330/20,857 MiB，
最小余量 5,230/3,703 MiB。最终输出 15 帧、0.250244 秒、PTS 严格递增且 strict
decode 通过；运行窗口无新增 kernel GPU 错误，进程退出后整卡显存回到约 0.8 GiB。

证据位于仓库外本地事务 `jasna-8k-system-vram-headroom-20260901/`；
其中 `default-pool-release-isolation-attempt002` 保留串行 open 修复前“首 child 成功、
第二 child 低显存停顿”的反例，`default-pool-serialized-open-attempt003` 是最终通过
记录。

本轮 pool16、水位状态机和输出验收证据位于仓库外本地事务
`jasna-8k-vram-watermark-pool-20260901/`。
无环境变量的真实 15 帧 P010 reader 重开测试明确输出
`Automatic 8K/B4 AMF decoder surface pool: 16`；连续两轮各新建、完整消费并关闭两条
reader，每轮均为 15/15×2 D2D，FD close 配平，所有 forbidden counters 为 0。另一个
基于原 20 秒源前 15 帧的两 render-span 诊断中，两个 span 都完成两条 reader 的
打开、D2D 审计和关闭，第二 span 开始前显存已从第一 span 结束后的约 6,226 MiB 回落到
约 4,114 MiB，没有跨 span pool 累积；由于该诊断故意只给 1202 帧源提交前 15 帧，
最终全源帧数校验按设计拒绝 15 != 1202，不能把保留的 rejected assembly 当成成片。

另一次把原 HEVC packet-copy 截成仅 15 帧、约 0.25 秒的独立 MP4 再走两段完整 mux
时，第一段 AMF encoder 报 `HevcInitialVBVBufferFullness AMF_OUT_OF_RANGE`，随后第二段
decoder open 停顿；该进程已按精确 PID 终止，输出未被接受。这是亚秒级合成源的
`vbr_peak`/AMF 生命周期边界，不是长片 pool16 的通过证据，也没有据此修改正式长片
策略；后续若声明支持这类亚秒输入，需另开独立最小复现和修复。

## 多 render span 高压回收与 decoder-open watchdog（2026-09-02）

第二个正式 8K Main10 视频完成 fresh pre-scan 后连续处理多个 Smart Render 区段。前三个
render span 的整卡峰值依次为 20,775、23,860、24,433 MiB；第三段最小余量只有
127 MiB，触发一次 critical reclaim，并出现 AMF `CopySurfaceRegion() Copy thread
timeout`。该段仍完成两条 2,340/2,340 D2D reader；但下一段在打印 pool16 后卡在
`CodecContext.open()`，没有 AMF decoder success、没有新帧，GPU 降到约 0–7%，
整卡显存稳定在约 15–16 GiB。日志保存在本地输出目录的 `.jasna-logs/` 中。

现在 isolated Linux AMD Smart Render 每完成一个 render fragment，都会检查该 pass 的
整卡 pressure episode/critical reclaim。若仍有待处理 render span 且本段发生过压力，
先把当前 fragment 原子标记为 complete，再以专用协议退出 child；GUI parent 等待显存
至少连续两次恢复到 4 GiB 启动预算后，创建全新 worker，并由同一 signature 工作区复用
已验 fragment。pressure recycle 最多 32 次，防止异常输入无限循环。没有 pressure 的
长 span 不增加进程重启，因此此前 47,686 帧、0 episode 的正式路线不会增加开销。

另在 isolated child 的 AMF `decoder.open()` 外加入 60 秒 watchdog。native open 无法由
Python 安全取消，所以超时后以专用退出码结束整个 child；parent 等待 native 显存释放并
最多自动重试两次，`running` span 在工作区重开时恢复为 `pending`。两次仍卡住会明确将
当前文件标错，不会 CPU fallback、静默跳过或继续后续 queue。watchdog 和 pressure
recycle 只在 Linux AMD GUI 的 per-video isolated worker 启用；Windows、NVIDIA、普通
CLI 和每帧热路径不变。

2026-09-04 合并 GUI“保持输入目录的子文件夹结构”后，最终输出路径仍只由 GUI parent
按每个任务记录的输入根目录解析一次。parent 把已经解析完成的精确目录与文件名交给
isolated child，因此自动预扫描的 `copy`、`full`、Smart Render、pressure recycle 和
decoder-open retry 始终复用同一路径。批处理续跑时 parent 会先对该路径已有的视频做
编码、时长和尾部可读性校验：完整输出不启动 child，残缺输出以 `overwrite` 合同原位
重建；Stop 与 skip/replace 决策共用 completion lock，不会在停止后新建后续输出目录。

阶段性聚焦回归为 181 passed；阶段性扩展集合为 632 passed、1 skipped。原先
`test_app_prepares_entire_queue_before_starting_processor` 的 Tk `__new__` fixture 没有
初始化后来新增的 `_run_log` 字段，最终验收已补齐 fixture，不涉及产品路径。真实 20 秒
8K HEVC Main10/P010、B4、pool16、自动 `vbr_peak`
验收完成 1,202 帧，两条 reader 均为 1,202/1,202 D2D，forbidden counters 全 0，
整卡峰值约 20,417 MiB、最小余量约 4,143 MiB，pressure/reclaim 均为 0；watchdog 没有
误触发，最终 20.103411 秒成片通过软件 `-xerror -err_detect explode` 严格解码，运行窗口
无新增 kernel GPU 错误。证据位于仓库外本地事务
`jasna-8k-vram-watermark-pool-20260901/pool16-watchdog-success-20260902/`。

最终 AMF keyframe/reset/build/backend 集合为 216 passed、1 skipped；包含 GUI child
隔离、显存、Pipeline、Smart Render、splice、runtime 安装合同等交界面的集合为
653 passed、1 skipped。`tests/test_main_entry.py` 会按测试目的从 `sys.modules` 删除
`jasna.gui`，因此在同一 pytest 进程中把它排在已 collection 的 pre-scan 测试之前会让
后者持有旧函数对象；pre-scan routing 另开干净进程为 29 passed。完整仓库此前在当前
Linux AMD venv 得到 2409 passed、35 skipped、156 failed；失败集中于缺少
TensorRT/NVIDIA 依赖、普通 venv 未装 unified AMF bridge、模型/媒体 fixture 与其他
既有 GPU 环境用例，不属于本改动的聚焦通过集合。
