# Jasna 统一 PyAV/FFmpeg runtime 合同

更新时间：2026-10-01

## 目的

PyAV wheel 与它加载的 FFmpeg 动态库必须作为一个 ABI 单元使用。普通开发虚拟环境仍负责
PyTorch、GUI 和模型依赖；只有在显式使用本页启动器时，Jasna 才把固定的 PyAV、FFmpeg/
FFprobe 和动态库放到子进程最前面，并在导入媒体模块前执行完整预检。

启动器只选择、验证运行库，不另外实现解码、编码、检测、修复或 GUI 路由。当前集合以
上游 0.11 公共流水线为基础；供应商与平台差异由各后端处理。各优化的默认门和验收范围
见对应功能文档，不能把历史 0.10 验收当作 0.11 或新 ROCm SDK 的验收。

## 固定合同

- FFmpeg commit：`44d082edc87381d978e8588b148116b99fefdb43`
- PyAV commit：`7e3d950a8b72062502c1a60d672f8ca565313af5`
- AMF headers commit：`c35f613aea2e5057a688c979e75b1cf24253297e`
- PyAV：`18.1.0`
- FFmpeg ABI：libavcodec 62、libavformat 62、libavutil 60，以及文件中列出的完整小版本

Linux AMD 与 Windows AMD 各自有独立的文件哈希白名单。Windows runtime 另外固定 dav1d
commit。校验失败时启动器直接退出，不回退到系统 PyAV、系统 FFmpeg 或 CPU 路线。

## runtime 目录

```text
runtime-root/
├── runtime.json
├── site-packages/av/
├── bridge/             # Linux AMD AMF Vulkan/HIP extension
├── bin/ffmpeg[.exe]
├── bin/ffprobe[.exe]
└── lib/                 # Linux；Windows DLL 位于 bin/
```

`runtime.json` 至少包含：

```json
{
  "schema_version": 1,
  "platform": "linux-amd",
  "wheel_sha256": "...",
  "source_pins": {
    "FFMPEG_COMMIT": "...",
    "PYAV_COMMIT": "...",
    "AMF_COMMIT": "..."
  }
}
```

构建器与安装器见 `UNIFIED_BUILD_PIPELINE_CN.md`、`UNIFIED_RUNTIME_INSTALL_CN.md`。
构建/安装与运行库合同按功能分别拆分，但需要组合验收；未验收的资产不得冒充可用运行库。

## 使用

Linux：

```bash
scripts/run_jasna_unified.sh --preflight-only
scripts/run_jasna_unified.sh -- <jasna 参数>
```

可用 `JASNA_PYTHON=/path/to/python` 选择开发环境，用
`JASNA_UNIFIED_RUNTIME_ROOT=/path/to/runtime` 选择 runtime。没有显式 Python 时，启动器优先使用
仓库 `.venv/bin/python`，再查找 `python3`。

Windows PowerShell：

```powershell
scripts\run_jasna_unified_windows.ps1 -PreflightOnly
scripts\run_jasna_unified_windows.ps1 -- <jasna 参数>
```

可用 `-Python` 与 `-RuntimeRoot` 覆盖默认位置。默认 runtime 位于
`%LOCALAPPDATA%\Jasna\unified-runtime\windows-amd`。

成功预检会在用户状态目录记录 `jasna/runtime-preflight.json`；状态目录不可写不会改变已通过的
runtime 结论。

Linux runtime 的 manifest 还固定 bridge 文件名、二进制 SHA-256 与 bridge source
SHA-256。启动器把 `bridge/` 放在仓库源码之前，并在 preflight 中验证 extension 从所选
runtime 加载且具有 surface inspection、dependency probe 和 fixed-context session
入口。Windows runtime 不包含这个 Linux-only extension。

Linux ROCm 10 wheel SDK 的库位于所选 Python 环境的 `_rocm_sdk_core/lib` 与
`_rocm_sdk_libraries/lib`。启动器将这两处加入子进程的动态库路径，并排除系统
`/opt/rocm/lib`，防止同一 SONAME 加载不同 ROCm 版本。Python 常为指向系统解释器
的 venv 软链接，查找包目录时保留 venv 路径；不沿软链接改查系统包目录。SDK 只安装
一半时直接失败，旧的非 wheel SDK 环境保持原路径规则。

HIP 模块加载器从 Torch 的已载入依赖定位真实 HIP 库，再以 `RTLD_NOLOAD` 复用。
不能仅通过 `libamdhip64.so` 名字再载入系统运行库，也不能因为 HIP major 相同就认为
二进制来源相同。MIGraphX artifact 的版本、扩展 SHA、模型/源码 SHA 与 ABI 校验仍
各自保留；升级 Torch/ROCm 不自动使旧 artifact 合法。

Windows Python 3.8+ 不再只依赖子进程 `PATH` 搜索 extension 的 DLL 依赖。预检子进程和
正式 Jasna 子进程都会在导入 PyAV 前通过 `os.add_dll_directory()` 显式登记所选 runtime
经策略指定且已通过哈希校验的 DLL 目录（当前为 `bin/`），并让登记句柄保持到进程结束。
登记失败会直接终止，不会回退到环境中的 FFmpeg/PyAV。

## 边界与回滚

- 不自动修改当前 shell 的 `PATH`、`PYTHONPATH` 或 `LD_LIBRARY_PATH`；只构造 Jasna 子进程环境。
- 不接入 MIGraphX 或产品 GUI 自动启动；已删除的 rocDecode 不属于 runtime；Linux AMF bridge 只作为已校验的
  runtime ABI 组件随子进程加载，解码路由仍由独立 PR 决定。
- 不修改 B4、CQ、codec、VR auto 或任何媒体路由默认值。
- 回滚本功能只需停止使用 `scripts/run_jasna_unified*`；普通 `python -m jasna` 行为未改变。
