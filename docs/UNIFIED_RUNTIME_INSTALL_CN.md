# 统一 runtime 原子安装器

更新时间：2026-09-03

## 功能边界

`scripts/install_unified_runtime.py` 只安装已经构建并被
`jasna.runtime_contract` 接受的 PyAV/FFmpeg ABI 单元。Linux 构建产物中的 ABI-matched
AMF Vulkan/HIP bridge 会一并原子安装；安装器不负责构建源码，也不安装 MIGraphX
artifact、旧 rocDecode artifact 或模型文件。

因此本功能不会改变 GUI、解码/编码路由、B4、VR auto 或普通 `python -m jasna` 启动方式。

## 输入目录

```text
build-root/
├── build-manifest.txt
├── amf-interop-bridge/     # Linux only
│   └── _jasna_amf_surface_probe.<python-soabi>.so
├── wheels/
│   └── av-*.whl
└── ffmpeg-install/
    ├── bin/
    └── lib/                 # Linux
```

安装前会验证：

- FFmpeg、PyAV、AMF，以及 Windows dav1d 的固定 source commit；
- PyAV wheel 哈希；
- FFmpeg/FFprobe、动态库和 Windows DLL 的逐文件哈希；
- wheel 内部路径不能逃逸 `site-packages`；
- Linux bridge 的二进制哈希、源码哈希与当前 Python SOABI；
- Linux FFmpeg 的 `hevc_amf` decoder help 必须包含默认关闭的
  `reset_on_keyframe`，encoder help 必须包含默认关闭的 `host_zero_copy`；Smart Render
  所需 mux/demux、AAC decoder、Annex-B BSF 与 file/pipe protocol 也必须全部可查询；
- runtime 中的符号链接不能指向安装目录外部。

## 安装与回滚

先使用独立测试目录，不要直接覆盖正在使用的 runtime：

```bash
python3 scripts/install_unified_runtime.py \
  --build-root /path/to/accepted-build \
  --target-root /path/to/test-runtime
```

安装器在目标同级目录完成 staging 和全量合同校验，全部通过后才原子发布。已有目标默认拒绝覆盖。
显式使用 `--force` 时，旧 runtime 会重命名为带时间戳的 `.backup-*` 目录，不会自动删除；新目录发布
失败时会恢复旧目录。

禁止把文件系统根目录、用户主目录、仓库、build root，或它们的父/子危险位置作为安装目标；目标本身
也不能是符号链接。

## 集中 Windows 验收清单

本 PR 与 runtime 合同 PR 可以在同一次 Windows 会话中逐条验收：

1. 对 runtime 合同 PR 执行：

   ```powershell
   scripts\run_jasna_unified_windows.ps1 -PreflightOnly
   ```

2. 对本 PR 使用已验收 Windows build 安装到新的临时目标，不加 `--force`：

   ```powershell
   .venv\Scripts\python.exe scripts\install_unified_runtime.py `
     --build-root D:\path\to\accepted-build `
     --target-root D:\path\to\temporary-runtime `
     --platform win32
   ```

3. 用 `-RuntimeRoot` 指向临时目标再次执行 preflight。
4. 验证重复安装在不加 `--force` 时被拒绝，临时目标保持不变。
5. 不覆盖当前正式 runtime；`--force` 备份行为只在另一个可丢弃的测试目标中验证。

上述检查不需要跑完整视频。涉及 Windows AMF 解码/编码的真实视频验收属于后续后端 PR。
