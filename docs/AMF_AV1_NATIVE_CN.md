# Linux AMD 显式 AMF AV1 原生解码

> 当前产品状态（2026-08-30）：用户在确认安全 cache 与旧 rocDecode 基准的长稳态差距后，
> 接受该性能差距。Linux AMD AV1 Main NV12/P010 已由研究入口提升为产品 `auto`
> 默认路线；旧 rocDecode 后端已在后续独立 PR 中删除。下面“不进 auto/cache off”及
> rocDecode 对照描述保留为各历史阶段的证据，最终状态以本文最后一节为准。

## 范围

本阶段只扩展显式诊断入口：

```bash
JASNA_DECODE_BACKEND=amf-interop
```

支持 Linux AMD 上固定格式的 AV1 Main 8-bit（AMF Vulkan NV12）与 AV1
Main 10-bit（AMF Vulkan P010），batch size 为 1、2、4、8。输出沿用
`scripts/amf_surface_probe.pyx` 的 Vulkan external-memory → HIP D2D bridge。

本阶段不修改 `auto` 选择顺序，不删除临时 rocDecode fallback，不影响 Windows、
NVIDIA、GUI、共享修复编排、编码或产品默认。resource cache 仍默认关闭，显式开启
仍会 fail closed。动态分辨率或位深重配不在支持范围内。

## AV1 位深元数据

统一 AMF 运行时只带硬件 AV1 decoder；在真正 decode 前，FFprobe 可以给出标准
`mime_codec_string`，但可能不给 `pix_fmt`。`jasna.media` 因此只对明确的
`av01.…08` 和 `av01.…10` 分别补成 `yuv420p` 与 `yuv420p10le`，不猜测 12-bit
或缺失 codec string 的输入。显式 reader 随后仍要求第一帧以及全部后续帧的 AMF
surface format、尺寸和 fixed context 一致；不符合时禁止 host fallback。

## 改码前只读探针

接受运行时：

```text
/home/user/.local/share/jasna/unified-runtime/linux-amd
```

在未扩展源码 scope 前用运行时 monkeypatch 仅放行 AV1，分别完整读取：

```text
av1-main8-1536x960-60f.mkv
roi003-positive-av1-p010-60f.mkv
```

两份输入均为 60/60 帧，PTS 与系统 FFprobe 逐帧相同且严格递增。每次运行有 60
次 Vulkan export、HIP import/map/release/destroy，120 次 D2D plane copy，固定
context session create/close 为 1/1；host、CPU Map、staging、D2H、failed bridge、
cache hit/miss 均为 0。

源码扩展后又在不使用 monkeypatch、也不强制覆写 10-bit metadata 的条件下重复
上述 B4 完整读取，结果相同。两种位深也分别以 B8 消费第一批 8 帧后主动关闭；
每次都只有 8 次 export/import/map/release/destroy、16 次 D2D plane copy，session
create/close 1/1，没有预取下一批未消费 surfaces。

随后补齐 B1、B2、B8 的完整 60 帧读取；连同 B4，两种位深的四个 batch size
全部保持 60/60、PTS 完全匹配、每帧资源配平和 forbidden transport=0。矩阵完成后
GPU junction 为 64°C、memory sensor 为 68°C，D 盘剩余 43 GiB。

最终聚焦回归为 `126 passed, 1 skipped`，runtime contract 为 `6 passed`；排除
`test_media_init.py` 中依赖 NVIDIA 参数集合、但会随本机 AMD vendor 自动切换的两组
既有编码参数测试后，元数据集合为 `63 passed, 19 deselected`。不排除时的 10 个
失败均属于上述既有 vendor 假设，本阶段没有为通过测试而修改共享编码参数逻辑。

探针期间的 `[av1] Missing Sequence Header` 来自 `get_video_meta_data()` 在容器没有
`nb_frames` 时调用 OpenCV 计帧的既有路径；单独运行该元数据函数可复现，AMF reader
本身和统一运行时 FFprobe 均没有这些错误。该日志不用于掩盖 decoder 失败：正式验收
仍必须同时满足完整帧数、PTS、native surface、D2D telemetry 和资源配平。

10-bit fixture 的统一运行时报告 1552×960，并附带 `crop_right=48`；系统 FFprobe
显示 1600×960 coded frame。reader 采用 AMF 解码后的可见尺寸 1552×960，与运行时
crop metadata 一致，不把裁剪区域作为有效画面补回。

## 2026-08-29 最终真实素材验收

在 RX 7900 XTX（gfx1100）上，用固定 unified runtime、`private-deferred`、cache off
和产品默认 B4 完成了 Main8/Main10、4K/8K 的最终正确性边界。新增的真实画面来源夹具
为 Main8 4096×2048、Main8 8192×4096、Main10 8192×4096，各 120 帧、原始
`60000/1001` 帧率。三份夹具都通过独立 libdav1d `-xerror -err_detect explode`
严格解码、完整 120 帧和 PTS 严格递增。

连同既有 Main8 1536×960 60 帧与 Main10 3840×1920 240 帧，五组样本分别用
`amf-interop`、PyAV/libdav1d reference、rocDecode 完整读取。每组内三条路线的帧数、
PTS SHA-256 与完整 RGB SHA-256 完全一致。10,000 帧 Main10 B4 native 生命周期也
通过：fixed-context create/close `1/1`、stream create/destroy `1/1`、event
create/destroy `6/6`、source acquire/release `10000/10000`、import/map/release/
destroy 各 `10000`、final in-flight `0`。主动在第 20 帧停止后同样全部配平。
host、CPU Map、staging、D2H、non-D2D、failed bridge 与 cache hit/miss 均为 `0`。

但是性能替换门没有通过。同一 1536×960 Main10 素材、前 1200 帧、无 RGB 回读、
三轮交错的中位数为：native B4 `412.756 fps`、native B8 `417.220 fps`、rocDecode
B4 `1000.787 fps`、rocDecode B8 `898.528 fps`。native B8 对 B4 只快 `1.08%`，
native B4 比 rocDecode B4 慢 `58.76%`。因此 AV1 **不进入产品 auto**，rocDecode
不能删除；本路线继续只作为显式研究后端。该结论不改变 H.264/HEVC 已验收的 auto
路线、共享检测/修复编排、resource cache 默认关闭或产品默认 B4。

完整证据位于：

```text
/mnt/D/AI/amf-unified-work/transactions/jasna-ubuntu-av1-native-final-20260829
```

运行窗口峰值 junction 为 69°C，峰值 VRAM 为 11,891,666,944 bytes；D 盘最终
剩余 43 GiB，无残留测试进程或 partial，kernel journal 没有 OOM、GPU reset、ring
timeout 或 page fault 记录。

最终聚焦回归覆盖 AMF interop core、decoder backend/AMD path、rocDecode/reuse 与
runtime contract，结果为 `165 passed, 1 skipped`。

## 2026-08-29 interop 瓶颈与 cache 生命周期结论

后续阶段用同一 1536×960 Main10 前 1200 帧、B4、cache off 做七阶段三轮矩阵。
AMF/rocDecode 纯解码中位分别为 `675.583/685.051 fps`，只差 `1.38%`；加入
surface inspection 后 AMF 仍为 `675.427 fps`。差距从 Vulkan external memory
导入 HIP 开始：AMF copy 为 `528.060 fps`，AMF full 为 `410.885 fps`；对应
rocDecode copy/full 为 `638.576/610.303 fps`。这里的 full 指同一 profiler 的
copy 加 YUV-to-RGB 阶段，与上一节 decoder 入口吞吐矩阵不是同一测量口径。

`rocprof` 在 AMF full 的 1200 帧中记录 `hipImportExternalMemory` 共
`653.678 ms`、`hipFree(mapped)` 共 `246.206 ms`、`hipStreamSynchronize` 共
`105.759 ms`。逐帧 import 是最大固定成本，mapped pointer 的 free 又等待尚未完成的
GPU 工作；decoder、surface inspection、D2D copy、converter private/default stream
均不是主瓶颈。

旧 fixed-session raw-identity cache 作为隔离性能上限，把 1200 次 import/map/release/
destroy 降为 `36`，三轮均为 `1164 hit + 36 miss`，中位 `554.732 fps`：比当前
AMF full 快 `35.01%`，达到 rocDecode full 的 `90.90%`。但它不能恢复到产品。
独立 `AMFSurfaceObserver` 探针在 1200 帧中得到 1200 次 add、1200 次
`OnSurfaceDataRelease` callback、1200 个 generation、0 次 active-surface repeat，
虽然底层只轮换 36 个 Vulkan memory identity。也就是说，同一 raw handle 跨越了
不同 AMF surface-data generation；按官方 callback 做安全失效会得到 0 hit，跨 callback
保留 HIP import 则不满足生命周期证明。

随后已完成 session-owned Vulkan staging surface 研究。该路线的 packed P010 逐帧
等价、owned observer、固定 context 身份和 HIP mapping → owned surface → decoder/context
关闭顺序都通过；但两轮 10,000 帧 B4 批量全线路中位为 `1016.185 fps`，rocDecode 为
`1138.412 fps`，仍慢 `10.74%`。B4 owned ring 相对单 owned allocation 也没有有意义的
收益，因此该路线因长期稳态性能不足否决。

官方 `AMFComponent::SetOutputDataAllocatorCB` 方向也已用隔离 FFmpeg/AMF 副本验证。
注册成功且 allocator callback acquire/release 配平，但 AV1 decoder 从未调用
`AMFDataAllocatorCB::AllocSurface`，计数为 `surfaces=0`，仍使用内部 36 个 Vulkan
allocation。也就是说当前组件/运行时接受但忽略 AV1 decode output allocator，不能借此
建立 caller-owned decoder pool 或消除拷贝。

最后验证了不依赖可复用 `VkDeviceMemory` 数值的 dma-buf 稳定身份 cache。它每帧导出
FD，并以 `fstat(fd)` 的 `(st_dev, st_ino)` 识别真实 backing：hit 关闭本次 FD 并复用
HIP import/map，miss 才把 FD 所有权交给 HIP。120 帧逐帧等价验收得到 36 个 Vulkan
memory 与 36 个 dma-buf identity 一一对应，`36 miss + 84 hit`；packed P010 与当前逐帧
bridge 完全相等，PTS SHA-256 为
`115e01bde728624531c8963aeddeae77dd98716061dfed7950485b3732daf847`。36 个 mapping/import
均在 decoder 关闭前释放，host、CPU Map、staging、D2H、failed copy 和 identity change
全部为 0。

1200 帧三轮交错的短样本中位一度达到 `614.814 fps`，rocDecode 为 `612.213 fps`，
说明短样本受启动波动影响，不能据此接入。最终两轮 10,000 帧交错结果为：cache
`1024.278/1029.970 fps`，rocDecode `1137.875/1137.582 fps`；中位
`1027.124/1137.729 fps`，cache 长稳态仍慢 `9.72%`。两轮都保持
`9964 hit + 36 miss`，raw handle/dma-buf backing 没有变化，36/36 mapping/import 在关闭
时配平，PTS 严格递增且 hash 一致。这证明稳定身份解决了旧 raw-handle cache 的生命周期
漏洞，但没有消除每帧 HIP D2D/同步瓶颈。

至此本研究分支的三条安全候选均已收口：owned staging 正确但慢，AMF AV1 忽略 output
allocator，稳定 dma-buf cache 安全但长稳态慢 `9.72%`。本分支停止，不接产品、不继续
无依据参数扫描。

本阶段仍未修改产品实现：AV1 不进 auto、rocDecode 不删除、resource cache 默认关闭、
默认 B4，Windows/NVIDIA/共享检测修复编排均不变。完整记录位于：

```text
/mnt/D/AI/amf-unified-work/transactions/jasna-ubuntu-av1-native-bottleneck-20260829
```

最终运行窗口峰值 junction 为 `76°C`，D 盘仍有约 `43 GiB`，无残留测试进程；kernel
journal 未发现 OOM、GPU reset、ring timeout 或 page fault。主要新证据为：

```text
results/dmabuf-identity-cache-equality-lifecycle-120f.json
results/dmabuf-formal-r{1,2,3}-cache-overlap-batched-full.json
results/dmabuf-formal-r{1,2,3}-roc-full.json
results/dmabuf-long-r{1,2}-cache-overlap-batched-full-10000f.json
results/dmabuf-long-r{1,2}-roc-full-10000f.json
results/output-allocator-lifecycle-amf-raw-120f.json
```

## 2026-08-29 用户放宽性能门后的产品接入

用户明确裁决：在没有新的安全优化空间时，可以接受 cache 与旧 rocDecode 基准的剩余差距并
启用 cache。产品实现因此只在 Linux AMD、AV1 Main、NV12/P010、B1/B2/B4/B8 的既定
格式门内，把 `auto` 固定到 AMF Vulkan → HIP 稳定 dma-buf identity cache；H.264/HEVC
继续原有 `private-deferred`，Windows、NVIDIA、共享检测/Tracker/BasicVSR++ 和默认 B4
均未改变。后续独立 PR 已删除 rocDecode；native gate 外输入继续使用普通 PyAV，失败
直接上报，不扩大本 native gate，也不新增 CPU fallback。

cache 以单个 reader session 为 epoch，用每次导出 FD 的 `(st_dev, st_ino)` 识别真实
dma-buf backing，不信任会复用的 `VkImage`/`VkDeviceMemory` 数值。cache hit 关闭本次
导出 FD；miss 才由 HIP 接管 FD 并保存 import/mapping。reader teardown 必须先关闭
全部 HIP mapping/import，再销毁 decoder-owned Vulkan allocation。context、device、
格式、可见尺寸、位深、allocation size 或 plane layout 变化全部 fail closed。

产品 B4 路径同时使用批量 AMD YUV conversion 和两个 packed 输入 slot 交错 copy/convert。
RGB 输出 tensor 不复用，因为 `_flat_frames()` 会把 batch view 交给下游线程；复用已 yield
的 RGB slot 会有覆盖尚未消费帧的风险。主动停止最多已经预取一个 B4，预取 surface 在
关闭前完成 copy/release，不会跨 reader epoch 留存。

用产品代码、临时 ABI-matched 新 bridge 与不设置 decode 实验变量的 `auto` 完成真实 GPU
验收：Main8 4K/8K 各 120 帧、Main10 4K 60 帧和 Main10 8K 120 帧均完整通过；8K
Main10 为 `36 miss + 84 hit`、约 `53.33 fps`，host/CPU Map/staging/D2H 均为 0。
10,000 帧 Main10 B4 为 `9964 hit + 36 miss`，36/36 import/destroy 配平，PTS 严格递增，
hash 为 `1f223a31a3cc1c1b636b726718b1094eec9c21cb07d5b4663e2578e7f064b89c`，
吞吐 `924.696 fps`；同口径 rocDecode 已知基准为 `1137.729 fps`，产品实现慢
`18.72%`。这比隔离实验的 `9.72%` 差距大，原因是产品不能复用已经交给下游的 RGB
output slot；该正确性边界不为追求实验数字而放宽。

Main10 reader 又分别从开头、1 秒和 2 秒执行 `auto` seek/reopen。三次各返回 12 帧并
严格递增，各自 cache session create/close `1/1`；因 overlap 每次实际 copy 16 帧，
16/16 import/destroy 在 decoder 关闭前配平，forbidden transport 为 0。Cython/C 编译
通过；AMF interop/YUV 聚焦回归为 `101 passed`，最终 decoder/AMD/rocDecode/runtime、
installer/build/YUV 较宽回归当时为 `215 passed, 1 skipped`。另 6 个旧 seek 用例在本机
AMD `auto` 环境仍暴露既有 B24/未加载 unified bridge 假设；只读对照确认提交前代码同样
失败，本提交不借机修改共享 seek 或批大小策略。

Windows 不能直接使用本 cache。Linux 实现依赖 `vkGetMemoryFdKHR`、opaque FD、
`fstat`、POSIX `close` 与 `hipExternalMemoryHandleTypeOpaqueFd`。Windows 若要获得同类
收益，必须另做 `vkGetMemoryWin32HandleKHR`、opaque Win32/KMT handle、
`hipExternalMemoryHandleTypeOpaqueWin32/Win32Kmt`，以及 Win32 object identity 或固定
decoder-pool epoch 的生命周期证明，并在 Windows AMD 实机单独验收。当前 Windows AMD
AV1/HEVC Main10 仍保持软件解码 → ROCm upload，不能把 Linux 的通过结论标成 Windows
已支持。

新 bridge 已通过独立候选 runtime preflight 后由原子安装器发布到本机正式
`~/.local/share/jasna/unified-runtime/linux-amd`；旧 runtime 保存在同级
`linux-amd.backup-20260829-143157`。正式 runtime 的 AV1 Main10 60 帧 `auto` smoke 为
60/60、PTS 严格递增、`36 miss + 24 hit`、import/destroy `36/36`、session `1/1`，
forbidden transport=0。正在运行的旧 GUI 进程不会热换已加载模块，需下次启动才使用新
bridge。
