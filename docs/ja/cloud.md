# クラウド GPU (RunPod、Vast、Docker)

レンタルした GPU でも Jasna を使えます。Linux NVIDIA 版をコマンドラインで
使います (アプリの画面は開きません)。

## 必要なもの

- 販売者にライセンスコードの**クラウド利用**をオンにしてもらう。
- ドライバー **580 以降**の NVIDIA GPU。
- **80 GB** 以上のディスク。
- テンプレートかインスタンスの設定に `NVIDIA_DRIVER_CAPABILITIES=all`。
  これがないと動画を読み込めず、保存もできません。Vast では Docker
  オプションに `-e NVIDIA_DRIVER_CAPABILITIES=all` を追加します。
- `root` でログイン。RunPod と Vast は最初からそうなっています。

## インストール

借りたマシンのターミナルに次の行を貼り付けます。新しい Jasna を使う場合は
`0.11.0` をそのバージョンに変えてください。

```bash
apt-get update && apt-get install -y zstd wget libgl1 libglib2.0-0

cd /root
for part in 000 001 002; do wget https://github.com/Kruk2/jasna/releases/download/v0.11.0/jasna-linux-nvidia-0.11.0.tar.zst.part$part; done

cat jasna-linux-nvidia-0.11.0.tar.zst.part* | zstd -dc | tar -xpf - --same-owner
cd jasna-linux-nvidia-0.11.0
```

上のとおり `/root` に展開してください。`/workspace` やネットワークドライブ
からは起動できません。

## 有効化

```bash
export XDG_CONFIG_HOME=/workspace/jasna-config
./jasna --license-code あなたのライセンスコード --cloud-workspace
```

`installation verified` の後、1 分ほど止まって見えることがあります。
Ctrl+C は押さずに待ってください。

有効化は `/workspace/jasna-config` に保存されます。Jasna を起動する前に
毎回 `export` の行を実行してください。別のマシンを借りるときは、この
フォルダーも持っていきます (RunPod のボリュームか、PC にダウンロード)。
なくすと、もう一度有効化するのに空き枠が必要です。

## 動画を復元する

```bash
export XDG_CONFIG_HOME=/workspace/jasna-config
./jasna --input /root/video.mp4 --output /root/video-restored.mp4 --restoration-model-name ltx
```

最初の 1 回だけ LTX モデル (15 GB) のダウンロードを聞かれます。`y` を入力します。

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
| `This container needs a Cloud workspace key` | `--cloud-workspace` を付ける。 |
| `enable cloud activation` | 販売者にクラウド利用をオンにしてもらう。 |
| `No activation found` または `Cloud workspace identity is missing` | `export` の行を実行してから Jasna を起動し直す。 |
| `unsafe Jasna launch environment` | `env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD ./jasna ...` で起動する。 |

`Bit depth 10 ... not supported` は大丈夫です。その動画の読み込みが少し遅く
なるだけです。

`jasna-config` フォルダーは人に渡さないでください。持っている人は誰でも
あなたの有効化を使えてしまいます。
