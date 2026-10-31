# クラウド GPU (RunPod、Vast、Docker)

レンタルした GPU でも、Linux NVIDIA 版をコマンドラインで使えます。ライセンスコードも
そのまま使えます。そのマシンが 3 台の枠の 1 つに数えられるのは Jasna を動かしている
間だけで、支援者向けモデルを使う間はインターネット接続が必要です。GitHub のソース版
では支援者向けモデルを使えません。

## 必要なもの

- ドライバー **580 以降**の NVIDIA GPU と、**80 GB** 以上のディスク。
- テンプレートかインスタンスの設定に `NVIDIA_DRIVER_CAPABILITIES=all`。
  これがないと動画を読み込めず、保存もできません。Vast では Docker
  オプションに `-e NVIDIA_DRIVER_CAPABILITIES=all` を追加します。
- `root` でログイン。RunPod と Vast は最初からそうなっています。
- LTX には Vast を使ってください。今のところ RunPod と Modal では LTX が使えません
  （BasicVSR++ は使えます）。
- Docker: コンテナーを `--pid=host` で起動しないでください。

## インストール

借りたマシンのターミナルに次の行を貼り付けます。最新の Jasna を `/root` に
ダウンロードします:

```bash
apt-get update && apt-get install -y curl zstd wget libgl1 libglib2.0-0

cd /root
curl -s https://api.github.com/repos/Kruk2/jasna/releases/latest | grep -o 'https://[^"]*jasna-linux-nvidia-[^"]*part[0-9]*' | sort -u | xargs -n1 wget
cat jasna-linux-nvidia-*.part* | zstd -dc | tar -xpf - --same-owner
cd jasna-linux-nvidia-*/
```

Jasna はマシン自身のディスクに置いてください。`/workspace`、ネットワークドライブ、
ネットワークボリュームからは起動できません。

## 動画エンコーダーを確認

レンタルマシンの中には、GPU の動画エンコーダーが動かないものがあります。
有効化の前に確認してください。1 秒で終わります:

```bash
./tools/ffmpeg -loglevel error -f lavfi -i testsrc2=duration=1:size=1280x720 -c:v hevc_nvenc -f null - && echo OK
```

`OK` が出ない場合は、そのマシンを削除して別のマシンを借りてください。
そこでは Jasna は動画を保存できません。

## 有効化

```bash
export XDG_CONFIG_HOME=/root/jasna-config
./jasna --license-code YOUR-LICENSE-CODE
```

`installation verified` の後、最大 1 分ほど止まって見えることがあります。
Ctrl+C は押さずに待ってください。

Jasna を起動する前に毎回 `export` を実行してください。`jasna-config` がなくなった
場合（新しいマシンなど）は、もう一度有効化するだけです。RunPod の `/workspace` には
置かず、人にも渡さないでください。持っている人は誰でもあなたの有効化を使えてしまいます。

## 動画を復元する

```bash
export XDG_CONFIG_HOME=/root/jasna-config
./jasna --input /root/video.mp4 --output /root/video-restored.mp4 --restoration-model-name ltx
```

最初の 1 回だけ LTX モデル (15 GB) のダウンロードを聞かれます。`y` を入力します。

RTX 50 シリーズでは `--ltx-fast` を付けると約 1.4 倍速く復元できます
(さらに約 8 GB をダウンロードします)。

動画のアップロードと結果のダウンロードは、レンタル先のファイル画面を使うか、
PC で次を実行します:

```bash
scp -P PORT video.mp4 root@HOST:/root/
scp -P PORT root@HOST:/root/video-restored.mp4 .
```

## 困ったとき

| 表示 | やること |
| ---- | -------- |
| `unsigned or modified Jasna runtime` | `root` で `/root` に展開し直す。 |
| `libGL.so.1: cannot open shared object file` | インストールの `apt-get` の行を実行する。 |
| `Failed to open hevc encoder` | このマシンの動画エンコーダーが壊れています。別のマシンを借りる。 |
| `in use on three devices` | 別の端末の Jasna を閉じるか、閉じたばかりなら15分待ってから、もう一度試す。 |
| `could not reach the license service` または `lost the connection` | マシンのインターネット接続を確認して、もう一度開始する。 |
| `No activation found` または `Activate Jasna with your License code` | `export` の行を実行してから Jasna を起動し直すか、もう一度有効化する。 |
| `unsafe Jasna launch environment` | `env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD ./jasna ...` で起動する。 |

`Bit depth 10 ... not supported` は大丈夫です。その動画の読み込みが少し遅く
なるだけです。
