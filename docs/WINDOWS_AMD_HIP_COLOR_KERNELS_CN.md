# Windows AMD gfx1100 AOT HIP 色彩内核

## 产品范围

本优化把 NV12/P010 与 RGB 之间的逐像素 Torch eager 色彩转换替换为离线编译的
HIP module kernel。它覆盖 BT.601、BT.709、BT.2020 的 full/limited range，并保留
现有 Windows 解码、双 reader、BasicVSR++ 和 AMF 编码路径。

当前只接受以下精确合同：Windows、AMD ROCm PyTorch、`gfx1100`、HIP 7，以及随产品
安装并通过 manifest 校验的两个 code object。该路线不会自动开启；只有显式设置：

```powershell
$env:JASNA_AMD_HIP_COLOR_KERNELS = "1"
```

才会请求启用。未设置、空值或 `auto` 继续使用 Torch eager；`0` 明确关闭。显式启用时，
GPU 架构、Torch HIP 版本、已加载的 HIP runtime DLL/精确 SHA-256、runtime API 版本、
ABI、源码或 artifact 哈希任一不符都会报错，不允许静默回退。

## Runtime 与安全边界

Windows 机器可能同时安装多个 ROCm SDK。运行时不会按 `PATH` 搜索同名 DLL，而是复用
PyTorch 已加载的 `amdhip64_<HIP主版本>.dll` handle，再校验其真实路径、文件哈希及 API
版本。产品运行时不调用 `hipcc`、HIPRTC 或其他 JIT 编译器。

code object 按 HIP context 缓存到进程结束。转换器持有 module function handle，因而不会
在单个任务结束时提前 unload；进程销毁 HIP context 时统一释放，避免使仍存活的转换器
句柄失效。

## 离线构建与 frozen 打包

必须用当前 ROCm PyTorch 对应的 HIP SDK 和 Python 环境执行：

```powershell
.\scripts\build_hip_code_objects_windows.ps1 `
  -Architecture gfx1100 `
  -Hipcc C:\path\to\matching\HIP\bin\hipcc.exe `
  -Python C:\path\to\rocm-python.exe
```

脚本固定 `HIP_PATH`/`ROCM_PATH` 到所选 compiler 的 SDK，生成并校验 raw ELF AMDGPU HSA
code-object ABI v4，然后写入 manifest。提交或发布前应重新运行产品 parity 探针。

源码运行时文件位于 `jasna/media/`。Nuitka/frozen Windows 产品必须把以下三个文件复制到
可执行文件同目录：

- `yuv_to_rgb.gfx1100.windows.co`
- `rgb_to_yuv.gfx1100.windows.co`
- `hip_color_kernels.gfx1100.windows.json`

`.cu` 只用于离线重建，不需要随 frozen 产品发布。manifest 仍保留其精确 SHA-256，作为
构建来源身份记录。

## 本机验收结论

验收环境为 Radeon RX 7900 XTX (`gfx1100`)、PyTorch ROCm 7.2.1/HIP
`7.2.53211-158bd99533`。8K microbenchmark 的四类转换比 eager 快 10.33–25.35 倍；
24 个 bit-depth/colorspace/range/direction 组合全部通过，8-bit 最大误差不超过 1 code，
P010 最大误差不超过 64（1 code）。

正式产品路径的 600 帧 ABBA 中：Main8 平均墙钟由 119.928 秒降至 118.279 秒（1.375%），
显存峰值约下降 0.75 GiB；Main10 由 162.693 秒降至 161.684 秒（0.620%），显存峰值约
下降 0.68 GiB。Main8 输出逐字节一致；Main10 为 PSNR 53.795 dB、SSIM 0.998068。
两种 bit depth 均通过帧/packet 数、PTS/DTS、IDR、参数集、音频保持和严格软件解码验证。

转换核心和显存收益明确，但整线墙钟收益只有 0.6–1.4%，且当前 manifest 绑定精确 HIP
runtime，因此 Windows `auto` 仍保持关闭。这是有边界的显式产品优化，不应外推到其他
GPU、HIP runtime 或自动默认策略。
