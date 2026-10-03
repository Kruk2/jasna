# 统一 FFmpeg / PyAV 构建流水线

更新时间：2026-09-02

## 范围

本页描述两个**显式运行**的构建器：

- Linux：`scripts/build_unified_ffmpeg_pyav_ubuntu.sh`
- Windows：`scripts/build_unified_ffmpeg_pyav_windows.ps1`

它们只从固定源码构建一组共享 FFmpeg 库、`ffmpeg`/`ffprobe` 和 PyAV wheel，并写出
`build-manifest.txt`。受限 FFmpeg 配置显式包含产品 Smart Render 使用的
MP4/Matroska/NUT/MPEG-TS、concat/framemd5/pipe、AMF encoder 以及对应 bitstream filter；
这些能力是产品运行合同的一部分，不能只依赖开发机系统 FFmpeg。Linux 构建器还会针对同一 PyAV/FFmpeg ABI 构建 AMF Vulkan/HIP
bridge；它不会安装 runtime、不会改变默认解码/编码路由，也不会接入 MIGraphX、
Smart Render 或 GUI。

FFmpeg configure 中 extradata bitstream filter 的组件名是 `dump_extradata`；构建后的
CLI 名称是产品命令使用的 `dump_extra`。构建器启用 `--fatal-warnings`，configure 出现
未匹配组件等警告时必须直接失败，不能产出可误装的半完整 runtime。

已有的 `jasna.runtime_contract` 和 `scripts/install_unified_runtime.py` 是单独的责任：前者
定义可接受 runtime，后者只安装已经通过其 pin 与哈希检查的构建产物。本流水线与安装器
之间唯一的预期交接物是构建目录及其中的 `build-manifest.txt`；不要为了安装新构建而修改
runtime 合同或安装器。

## 固定源码

所有源码 checkout 必须恰好位于下表 commit。构建器在开始时验证 `HEAD`，不接受分支名、
tag 或近似版本。

| 组件 | 固定 commit | Linux | Windows |
|---|---|:---:|:---:|
| FFmpeg | `44d082edc87381d978e8588b148116b99fefdb43` | 是 | 是 |
| PyAV | `7e3d950a8b72062502c1a60d672f8ca565313af5` | 是 | 是 |
| AMF headers | `c35f613aea2e5057a688c979e75b1cf24253297e` | 是 | 是 |
| dav1d | `b546257f770768b2c88258c533da38b91a06f737` | 否 | 是 |

> 源码 checkout 必须可丢弃。两个构建器都会在 FFmpeg checkout 中应用所选补丁；Windows
> 还会把补丁目标转换为 LF，并在固定的 `configure` 中做 MSVC 兼容性替换。请使用工作副本，
> 不要指向唯一的干净源码树。

## FFmpeg 补丁

`0001` 与 Linux 的 `0005`、`0006` 总是应用。`0005`、`0006` 只增加默认关闭的 FFmpeg
decoder/encoder 选项，并不会自行改变其他调用方；Jasna 仅在已通过真片验证的精确范围内
启用它们。其余补丁仍需要明确选项，避免把研究性行为变成默认构建行为。

| 补丁 | 作用 | 默认 |
|---|---|---|
| `0001-amf-transfer-use-context-sw-format.patch` | AMF transfer 仅声明当前 frames context 的 `sw_format` | 始终应用 |
| `0002-amfdec-replace-stale-frames-context.patch` | 实际 surface 格式或尺寸与旧 context 不同时分配并替换 frames context | `--apply-frames-context-fix` / `-ApplyFramesContextFix` |
| `0003-amfdec-fix-dynamic-resolution-reinit.patch` | 分辨率变化时 `Terminate()` 后以未知尺寸 `Init()` | `--apply-dynamic-resolution-fix` / `-ApplyDynamicResolutionFix`；自动包含 `0002` |
| `0004-matroska-projection-tag-spherical.patch` | 仅当 coded spherical side data 缺失时，从 `projection=equirectangular` metadata 回退 | `--apply-spherical-metadata-patch` / `-ApplySphericalMetadataPatch` |
| `0005-amfdec-reset-state-at-keyframes.patch` | 增加默认关闭的 `reset_on_keyframe`；启用时在关键帧前完整 drain，再 `Terminate()` / `Init()` AMF，并修正 `SurfaceCopy` 属性类型与启用 HEVC Annex-B BSF | Linux 始终应用补丁；选项默认关闭 |
| `0006-amfenc-wrap-contiguous-host-input.patch` | 增加默认关闭的 `host_zero_copy`；严格校验连续 NV12/P010 host planes，以 `CreateSurfaceFromHostNative()` 包装并持有 AVFrame 到输出完成 | Linux 始终应用补丁；选项默认关闭 |

每个补丁均通过 `git apply --check` 处理；若已应用，构建器会用反向 check 接受它，而不是
重复应用。若 checkout 既不匹配原始源码也不匹配已应用补丁，构建会停止。

`0005` 已分别验证可直接应用到固定 FFmpeg commit，以及应用在 `0002 + 0003` 之后；
外部源码树用 `make -j8` 编译通过。真实 8K HEVC Main10/B4 复现中，原 AMF decoder
在固定 11 帧窗口有 4 个 P010 mismatch，启用 reset 后为 0；随后 24:00–29:00 连续
五分钟正式 Smart Render 验收完成，三轮 reader D2D/FD 审计配平，最终 111,286 帧成片
与源帧数、时长一致，软件严格解码、PTS 和 copy seam 检查全部通过。完整实片证据见
`docs/AMF_INTEROP_CORE_CN.md` 的“AMF HEVC 关键帧状态重置”章节。

`0006` 在相同 8K Main10/P010、相同 `async_depth=4` 的两轮正反 A/B 中将 writer
平均耗时从 126.6 秒降到 114.9 秒（9.2%）；相对生产深度的 writer 平均快 6.1%。
20 秒对照输出逐字节一致。随后真实连续 24:00–29:00 Smart Render 覆盖历史花屏点，
18,300 个 render 帧全部走 host-native wrap、AMF-owned copy 为 0；最终 111,286 帧成片
通过严格软件解码、copy seam、packet/PTS/时长检查和用户缩略图确认。

随后完成 Main 8-bit/NV12 的同源正反矩阵。4K 的完整路线双 GOP 比基线慢约 1.8%，因此
仍使用单会话；真实 5760×2880、59.94 fps、1201 帧 5K 样片中，双 GOP wall time
从平均 102.978 秒降到 78.783 秒（快 23.5%），writer 从 49.65 秒降到 25.50 秒；
8192×4096、1202 帧 8K 样片中 wall time 从 196.394 秒降到 161.105 秒（快 18.0%）。
5K 六个输出与 8K 六个输出均通过软件严格解码、帧数、持续时间、PTS/DTS 和每个 GOP
VPS/SPS/PPS 检查。5K 源到基线/双 GOP 的全片 PSNR 分别为 52.523/52.797 dB，
基线到双 GOP 为 54.911 dB；8K 源到基线/双 GOP 分别为 45.532/45.403 dB，差值
仅 0.129 dB。双 GOP 没有实质质量退化。

随后又以首个 IDR 无重编码截取的 5760×2880、Main10/P010、59.94 fps、1,201 帧实片
完成同样的六轮正反矩阵。默认 copy、host-native 单会话、双 GOP 的 wall time 中位数为
116.652、97.447、85.889 秒；双 GOP 相对默认 copy 快 26.4%，相对单会话快 11.9%，
writer 从 63.05 秒降到 30.95 秒。双 GOP 增加约 512 MiB 进程 RSS但没有增加整卡显存；
六个输出均通过软件严格解码、帧数、持续时间、PTS/DTS 和逐 GOP VPS/SPS/PPS 检查。
单会话与双 GOP 输出的 PSNR 为 56.811 dB、SSIM 为 0.998881。
另一次不带实验强制变量的产品 `auto` 复验在 86.868 秒完成，自动 host-native/双 GOP
选择及全部严格门再次通过。
P010 每像素约 3 byte，NV12 每像素约 1.5 byte；3840×2160 P010 与已实测提速 23.5% 的
5760×2880 Main8/NV12 单帧 host 数据量同为约 24.9 MB。按用户接受的等量负载推断，产品范围因此扩展为 Linux AMD
HEVC Main/NV12 至少 5760×2880，Main10/P010 至少达到 3840×2160 等效像素量。4K Main10
尚无直接实片 A/B；4K Main8、Windows、NVIDIA、H.264 和 AV1 输出保持原路线。

## Linux（Ubuntu）

准备 Git、Bash、GNU Make、可用的 C/C++ 构建工具、Python/PyAV wheel 构建依赖，以及
Vulkan 和 SPIR-V headers。默认会查找 `/usr/include/vulkan/vulkan.h` 与
`/usr/include/spirv/unified1/spirv.h`；如 headers 在别处，分别显式传入
`--vulkan-headers` 和 `--spirv-headers`。不能用忽略 configure warning 或
`--disable-x86asm` 绕过依赖检查。AMF checkout 可以是包含 `AMF/core/Factory.h` 的布局，也可以使用上游
`amf/public/include` 布局。

Linux runtime 的固定文件 hash 之外，安装与启动预检还会逐项查询 FFmpeg 的
Smart Render 合同：MPEG-TS/NUT/MP4/Matroska mux/demux、concat demux、AAC decoder、
H.264/HEVC Annex-B bitstream filter，以及 file/pipe protocol。任何一项被裁剪都会
直接拒绝安装或启动，不能等到长视频精扫与修复完成后才在片段归一化阶段失败。

最小构建示例：

```bash
./scripts/build_unified_ffmpeg_pyav_ubuntu.sh \
  --ffmpeg-source /work/src/ffmpeg \
  --pyav-source /work/src/pyav \
  --amf-source /work/src/AMF \
  --vulkan-headers /work/deps/Vulkan-Headers \
  --spirv-headers /work/deps/SPIRV-Headers \
  --output-root /work/out/unified-linux \
  --python /work/venv/bin/python \
  --jobs 16
```

要试验动态分辨率修复与 Matroska metadata 回退，额外加入：

```bash
  --apply-dynamic-resolution-fix \
  --apply-spherical-metadata-patch
```

输出目录的稳定部分如下：

```text
unified-linux/
├── build-manifest.txt
├── amf-interop-bridge/
│   └── _jasna_amf_surface_probe.<python-soabi>.so
├── ffmpeg-install/
│   ├── bin/ffmpeg
│   ├── bin/ffprobe
│   └── lib/
└── wheels/
    └── av-*.whl
```

Linux bridge 使用同一个 PyAV source、FFmpeg install、AMF/Vulkan headers 与指定的 ROCm
headers 构建。默认 ROCm include 是 `/opt/rocm/include`，也可用 `--rocm-include` 显式
覆盖。manifest 同时记录 bridge 二进制和 `scripts/amf_surface_probe.pyx` 的 SHA-256；
安装器会再次校验二进制、源码和当前 Python SOABI。

构建成功不等同于 runtime 已接受。只有当 wheel 和 FFmpeg 文件哈希也符合现有 runtime
合同的白名单时，才可把同一输出目录交给已有安装器，例如：

```bash
python3 scripts/install_unified_runtime.py \
  --build-root /work/out/unified-linux \
  --target-root /work/out/runtime-test
```

安装器会独立验证 manifest 中的固定 pin、wheel 和动态库；拒绝时应保留构建目录以便诊断，
而不是放宽合同或静默使用系统 PyAV。

## Windows

Windows 构建需要：PowerShell、Git、Python（能够运行 `python -m mesonbuild.mesonmain`）、
Ninja、Visual Studio 的 x64 MSVC 工具链与 `VsDevCmd.bat`、以及包含 `bash.exe` 和
`cygpath.exe` 的 MSYS2。dav1d 会通过 Meson/Ninja 以 shared library 方式构建，之后
`dav1d.dll` 会复制到 `ffmpeg-install\\bin`。

使用 ASCII 路径的独立工作目录。当前脚本生成的 `.cmd` 文件使用 ASCII 编码，因此不声明
非 ASCII 路径可用；如本机 VS 安装位置不同，请明确传入 `-VsDevCmd`。

```powershell
.\scripts\build_unified_ffmpeg_pyav_windows.ps1 `
  -FfmpegSource D:\work\src\ffmpeg `
  -PyAvSource D:\work\src\pyav `
  -AmfSource D:\work\src\AMF `
  -Dav1dSource D:\work\src\dav1d `
  -OutputRoot D:\work\out\unified-windows `
  -Python D:\work\venv\Scripts\python.exe `
  -Ninja ninja `
  -MsysRoot C:\msys64 `
  -VsDevCmd 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat' `
  -Jobs 16
```

可选修复使用 PowerShell switch，例如：

```powershell
  -ApplyFramesContextFix
  -ApplyDynamicResolutionFix
  -ApplySphericalMetadataPatch
```

Windows 输出与 Linux 同样包含 `build-manifest.txt`、`wheels\av-*.whl` 和
`ffmpeg-install`；Windows DLL（包括 `dav1d.dll`）位于 `ffmpeg-install\bin`，FFmpeg
import libraries 位于 `ffmpeg-install\lib`。将其安装到测试 runtime 时必须传入
`--platform win32`：

```powershell
D:\work\venv\Scripts\python.exe scripts\install_unified_runtime.py `
  --build-root D:\work\out\unified-windows `
  --target-root D:\work\out\runtime-test `
  --platform win32
```

## 当前验证边界

2026-08-31 的 Linux AMD runtime 重新构建恢复了 AAC decoder，并保持既有 FFmpeg/PyAV/
AMF 固定 commit、四项 FFmpeg 补丁和 AMF bridge。候选 wheel、CLI 和全部动态库已更新到
运行合同固定哈希；真实 8K HEVC Main 10 + AAC 48 kHz 双声道原片可由候选 PyAV 正确创建
AAC codec context。独立 5.03 秒 packet-copy 得到 236 个 DTS 严格递增的 AAC packet，
随后直接解码得到 236 帧、241,664 samples，PTS 严格递增。

该 runtime 已由安装器原子替换到默认 Linux AMD 目录，旧 runtime 保存在
`linux-amd.backup-20260831-221612`。正常 launcher 的全新子进程 `--preflight-only`
通过 PyAV 18.1.0、全部固定 FFmpeg ABI、AMF bridge 来源和 14 项 FFmpeg 能力（包含 AAC）。
这只验收 runtime 与真实音轨，不等同于重新运行被用户停止的 8K 长视频，也不代表最终成片
已通过；长片必须在关闭旧 GUI、重新启动后由用户重新测试。

Windows 的真实构建、生成 DLL 的加载/ABI 验证、以及 PowerShell parser/static-analyzer
验证仍必须在 Windows 环境完成；在完成前，不应把 Windows 产物或这组可选补丁宣称为已通过
Windows 实机验收。
