# クラウド GPU (RunPod、Vast、Docker)

> クラウドライセンスは現在販売していません。今後、販売する予定です。

レンタルした GPU でも Jasna を使えます。Linux NVIDIA 版をコマンドラインで
対応するコンテナー内で使います。クラウド有効化は Linux 公式版の CLI 専用で、
Windows と GitHub のソース版には対応していません。

## 必要なもの

- ライセンスコードの入手先に**クラウド利用**をオンにしてもらう。
- 普通のコンテナー。RunPod と Vast は使えます。`--pid=host` で起動した
  Docker コンテナーは使えません。
- ドライバー **580 以降**の NVIDIA GPU。
- **80 GB** 以上のディスク。
- テンプレートかインスタンスの設定に `NVIDIA_DRIVER_CAPABILITIES=all`。
  これがないと動画を読み込めず、保存もできません。Vast では Docker
  オプションに `-e NVIDIA_DRIVER_CAPABILITIES=all` を追加します。
- `root` でログイン。RunPod と Vast は最初からそうなっています。
- 固定の `/etc/machine-id`。これがないコンテナ（Modal や一部の Docker イメージ
  など）では有効化に失敗します。イメージに追加してください。

## インストール

借りたマシンのターミナルに次の行を貼り付けます。新しい Jasna を使う場合は
`0.11.1` をそのバージョンに変えてください。

```bash
apt-get update && apt-get install -y zstd wget libgl1 libglib2.0-0

cd /root
for part in 000 001 002 003; do wget https://github.com/Kruk2/jasna/releases/download/v0.11.1/jasna-linux-nvidia-0.11.1.tar.zst.part$part; done

cat jasna-linux-nvidia-0.11.1.tar.zst.part* | zstd -dc | tar -xpf - --same-owner
cd jasna-linux-nvidia-0.11.1
```

上のとおり `/root` に、Jasna を実行するのと同じユーザーで展開してください。
Jasna はマシン自身のディスク（ext4、Btrfs、XFS、ZFS、またはコンテナ自身の
ファイルシステム）に置きます。`/workspace` やネットワークドライブからは起動
できず、ネットワーク共有やネットワークボリュームでは LTX が読み込めない
ことがあります。

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
./jasna --license-code YOUR-LICENSE-CODE --cloud-workspace
```

`installation verified` の後、最大 1 分ほど止まって見えることがあります。
Ctrl+C は押さずに待ってください。

RunPod の `/workspace` ではファイルを非公開にする権限が反映されない場合が
あります。有効化は上のとおり `/root/jasna-config` に保存してください。
ほかのサービスでは、非公開の権限を保つ永続フォルダーも使えます。

Jasna を起動する前に毎回 `export` を実行してください。Pod を削除する前に、
`jasna-config` を自分の PC に非公開で保存してください。次の Pod の `/root`
に非公開のまま戻すと、同じ有効化の枠を使えます。なくすと、もう一度有効化するには空き枠が必要です。

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
| `Cloud workspace activation requires a supported Linux container` | 最新の Linux 版と `--pid=host` のないコンテナーを使う。RunPod では設定を `/root` に置く。 |
| `Failed to open hevc encoder` | このマシンの動画エンコーダーが壊れています。別のマシンを借りる。 |
| `enable cloud activation` | ライセンスコードの入手先にクラウド利用をオンにしてもらう。 |
| `No activation found` または `Cloud workspace identity is missing` | `export` の行を実行してから Jasna を起動し直す。 |
| `unsafe Jasna launch environment` | `env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD ./jasna ...` で起動する。 |

`Bit depth 10 ... not supported` は大丈夫です。その動画の読み込みが少し遅く
なるだけです。

`jasna-config` フォルダーは人に渡さないでください。持っている人は誰でも
あなたの有効化を使えてしまいます。
