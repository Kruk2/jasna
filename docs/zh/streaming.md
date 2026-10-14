# 流媒体

流媒体让你在 Jasna 还在修复时就能观看修复后的视频。不用等整个文件处理完。
支持跳转。

## 浏览器播放器

从命令行开始串流。Jasna 会在浏览器中打开一个播放器。在那里选择视频文件并播放。

```bash
jasna --stream
```

在 Windows 上，使用与应用相同的文件：`jasna.exe --stream`。

实用选项：`--stream-port`（默认 `8765`），以及想自己打开播放器时用的
`--no-browser`。

流媒体总是使用 BasicVSR++（不使用 LTX）和 H.264。如果想在应用里观看，请使用
[视频播放器](gui.md#视频播放器)。

## Stash

你可以通过一个定制的 Stash 版本，在 [Stash](https://github.com/stashapp/stash)
中使用 Jasna。播放场景时，Stash 会启动 Jasna，边看边修复。

下载：**[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

设置：

1. 下载这个 Stash 版本。
2. 启动 Stash 前，设置以下环境变量：
   - `JASNA_CLI_PATH`：`jasna.exe` 的完整路径。
   - `JASNA_WORKING_DIR`：包含 `jasna.exe` 的文件夹的完整路径。
3. **重要：**先用你在 Stash 中要用的相同设置，在一个短视频上串流一次。第一次
   运行要为你的 GPU 准备文件，耗时很长。如果跳过这一步，Stash 会等不及而放弃。
4. 启动 Stash 并播放场景。

如果 Stash 日志出现 `timeout waiting for jasna-cli to become healthy`，请先检查
`JASNA_CLI_PATH`，然后执行第 3 步。
