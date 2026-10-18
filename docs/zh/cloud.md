# 云 GPU（RunPod、Vast、Docker）

> 云许可证目前暂不出售，以后会开放。

租用的 GPU 也能运行 Jasna。用 Linux NVIDIA 官方版，在受支持容器的命令行里运行。
云激活仅限 CLI，Windows 和 GitHub 源码版不支持。

## 需要准备

- 请向你获得许可证码的地方申请为它打开**云权限**。
- 普通容器即可。RunPod 和 Vast 都能用。用 `--pid=host` 启动的 Docker 容器不行。
- NVIDIA 显卡，驱动 **580 或更新**。
- 至少 **80 GB** 磁盘。
- 在模板或实例设置里加上 `NVIDIA_DRIVER_CAPABILITIES=all`。
  不加的话 Jasna 读不了也存不了视频。Vast 上把
  `-e NVIDIA_DRIVER_CAPABILITIES=all` 加到 Docker 选项里。
- 用 `root` 登录。RunPod 和 Vast 默认就是。
- 固定的 `/etc/machine-id`。有些容器（例如 Modal 或部分 Docker 镜像）没有它，
  激活会失败。请把它加到镜像里。

## 安装

把下面几行粘贴到租来机器的终端里。用的是更新的 Jasna 的话，把 `0.11.1`
改成你的版本。

```bash
apt-get update && apt-get install -y zstd wget libgl1 libglib2.0-0

cd /root
for part in 000 001 002 003; do wget https://github.com/Kruk2/jasna/releases/download/v0.11.1/jasna-linux-nvidia-0.11.1.tar.zst.part$part; done

cat jasna-linux-nvidia-0.11.1.tar.zst.part* | zstd -dc | tar -xpf - --same-owner
cd jasna-linux-nvidia-0.11.1
```

像上面这样，用运行 Jasna 的同一个用户解压到 `/root`。把 Jasna 放在机器自己的
硬盘上（ext4、Btrfs、XFS、ZFS 或容器自己的文件系统）。放在 `/workspace` 或
网络盘里 Jasna 启动不了，放在网络共享或网络卷上 LTX 可能无法加载。

## 检查视频编码器

有些租来的机器，显卡的视频编码器用不了。激活前先检查一下，只要一秒：

```bash
./tools/ffmpeg -loglevel error -f lavfi -i testsrc2=duration=1:size=1280x720 -c:v hevc_nvenc -f null - && echo OK
```

没看到 `OK`？删掉这台机器，换一台。在这台上 Jasna 存不了视频。

## 激活

```bash
export XDG_CONFIG_HOME=/root/jasna-config
./jasna --license-code 你的许可证码 --cloud-workspace
```

出现 `installation verified` 后，可能会像卡住一样最多一分钟。等着就行，
别按 Ctrl+C。

RunPod 的 `/workspace` 可能不保留私有文件权限。请像上面一样将激活信息
保存在 `/root/jasna-config`。其他服务商可以使用保留私有权限的持久文件夹。

每次启动 Jasna 前都先运行 `export`。删除 Pod 前，将 `jasna-config` 私下下载
到自己的电脑。下次将它恢复到 Pod 的 `/root` 并保持私有，就能继续使用同一
激活名额。丢失这个文件夹后，需要空闲名额才能重新激活。

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
| `Cloud workspace activation requires a supported Linux container` | 使用最新的 Linux 版和不带 `--pid=host` 的容器。RunPod 上请将设置放在 `/root`。 |
| `Failed to open hevc encoder` | 这台机器的视频编码器坏了，换一台。 |
| `enable cloud activation` | 请向你获得许可证码的地方申请打开云权限。 |
| `No activation found` 或 `Cloud workspace identity is missing` | 先运行 `export` 那一行，再重新启动 Jasna。 |
| `unsafe Jasna launch environment` | 用 `env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD ./jasna ...` 启动。 |

看到 `Bit depth 10 ... not supported` 没关系，只是这个视频读得慢一点。

别把 `jasna-config` 文件夹给别人。谁拿到它都能用你的激活。
