# 流媒体

流媒体可以让你不必先处理完整个文件，就能实时观看修复后的视频。
支持跳转。

## 浏览器播放器

流媒体模式目前仅支持 CLI。它会在浏览器中打开一个播放器 — 选择视频
文件即可开始观看:

```bash
jasna --stream
```

在 Windows 上，串流使用与应用相同的文件: `jasna.exe --stream`。

实用选项: `--stream-port`（默认 `8765`），以及想自己打开播放器时的
`--no-browser`。

## Stash 集成

Jasna 可以通过自定义 Stash 分支在 [Stash](https://github.com/stashapp/stash)
内使用。播放场景时，Stash 会自动启动 Jasna，边看边处理。

自定义分支:
**[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

设置:

1. 从上方链接下载 Stash 分支。
2. 启动 Stash 前设置环境变量:
   - `JASNA_CLI_PATH`: `jasna.exe` 的完整路径，除非你自己重命名了它。
   - `JASNA_WORKING_DIR`: 包含该可执行文件的文件夹完整路径。
3. **重要:** 使用 Stash 前，先用你打算在 Stash 中使用的相同设置，在
   一个短视频上串流一次。这样可以准备好 GPU 专用检测缓存，避免第一次
   健康检查超时。
4. 启动 Stash 并播放场景。

如果 Stash 日志出现 `timeout waiting for jasna-cli to become healthy`，
请先检查 `JASNA_CLI_PATH`，然后按上面的方法预编译。
