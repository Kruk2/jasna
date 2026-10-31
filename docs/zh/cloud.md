# 云 GPU（RunPod、Vast、Docker）

租用的 GPU 也能用命令行运行 Linux NVIDIA 版 Jasna，许可证码同样可用：只有 Jasna 运行时
这台机器才占用 3 个设备名额中的 1 个，使用支持者模型时需要联网。GitHub 源码版不能使用
支持者模型。

## 需要准备

- NVIDIA 显卡，驱动 **580 或更新**，以及至少 **80 GB** 磁盘。
- 在模板或实例设置里加上 `NVIDIA_DRIVER_CAPABILITIES=all`，否则 Jasna 读不了也存不了视频。
  Vast 上把 `-e NVIDIA_DRIVER_CAPABILITIES=all` 加到 Docker 选项里。
- 用 `root` 登录。RunPod 和 Vast 默认就是。
- 要用 LTX 请选 Vast。目前 RunPod 和 Modal 上不能用 LTX（BasicVSR++ 可以用）。
- Docker：不要用 `--pid=host` 启动容器。

## 安装

把下面几行粘贴到租来机器的终端里。它们会把最新的 Jasna 下载到 `/root`：

```bash
apt-get update && apt-get install -y curl zstd wget libgl1 libglib2.0-0

cd /root
curl -s https://api.github.com/repos/Kruk2/jasna/releases/latest | grep -o 'https://[^"]*jasna-linux-nvidia-[^"]*part[0-9]*' | sort -u | xargs -n1 wget
cat jasna-linux-nvidia-*.part* | zstd -dc | tar -xpf - --same-owner
cd jasna-linux-nvidia-*/
```

把 Jasna 放在机器自己的磁盘上。它不能从 `/workspace`、网络驱动器或网络卷启动。

## 检查视频编码器

有些租来的机器，显卡的视频编码器用不了。激活前先检查一下，只要一秒：

```bash
./tools/ffmpeg -loglevel error -f lavfi -i testsrc2=duration=1:size=1280x720 -c:v hevc_nvenc -f null - && echo OK
```

没看到 `OK`？删掉这台机器，换一台。在这台上 Jasna 存不了视频。

## 激活

```bash
export XDG_CONFIG_HOME=/root/jasna-config
./jasna --license-code 你的许可证码
```

出现 `installation verified` 后，可能会像卡住一样最多一分钟。等着就行，
别按 Ctrl+C。

每次启动 Jasna 前都先运行 `export`。`jasna-config` 没了（例如换了新机器）的话，重新激活即可。
不要把它放在 RunPod 的 `/workspace`，也别给别人：谁拿到它都能用你的激活。

## 修复视频

```bash
export XDG_CONFIG_HOME=/root/jasna-config
./jasna --input /root/video.mp4 --output /root/video-restored.mp4 --restoration-model-name ltx
```

第一次会问要不要下载 LTX 模型（15 GB），输入 `y`。

RTX 50 系显卡加上 `--ltx-fast`，修复大约快 1.4 倍（会多下载约 8 GB）。

上传视频和取回结果，可以用服务商的文件管理器，或在电脑上运行：

```bash
scp -P PORT video.mp4 root@HOST:/root/
scp -P PORT root@HOST:/root/video-restored.mp4 .
```

## 常见问题

| 看到 | 怎么办 |
| ---- | ------ |
| `unsigned or modified Jasna runtime` | 用 `root` 重新解压到 `/root`。 |
| `libGL.so.1: cannot open shared object file` | 运行安装里的 `apt-get` 那一行。 |
| `Failed to open hevc encoder` | 这台机器的视频编码器坏了，换一台。 |
| `in use on three devices` | 关闭其他设备上的 Jasna；如果刚刚关闭，等待 15 分钟，再试一次。 |
| `could not reach the license service` 或 `lost the connection` | 检查机器的网络，然后重新开始。 |
| `No activation found` 或 `Activate Jasna with your License code` | 先运行 `export` 那一行，再重新启动 Jasna，或者重新激活。 |
| `unsafe Jasna launch environment` | 用 `env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD ./jasna ...` 启动。 |

看到 `Bit depth 10 ... not supported` 没关系，只是这个视频读得慢一点。
