# CLI リファレンス

コマンドラインでは、GUI でできることのほとんどができます。GUI だけの機能は、
動画プレーヤー、LTX のシードのプレビュー、マスクの提案、対話型画像復元です。
`jasna --help` で常にすべてのオプションを表示できます。このページでは説明と
例を紹介します。

インストールや更新後の初回起動は、Jasna がまずファイルを確認するため時間が
かかります。

```bash
# 動画 1 本
jasna --input input.mp4 --output output.mkv

# 静止画（自動的に SD 1.5 を使います）
jasna --input photo.png --output restored.png

# フォルダー全体（画像が先、次に動画）
jasna --input input_folder --output output_folder
```

Windows では、CLI もアプリ本体と同じファイルです: `jasna.exe --input ...`。

## Windows の一時ファイル

Jasna の一時ファイルは、ImDisk（NTFS）などの RAM ディスクに置けます。
SSD の負担を減らせます。Jasna を起動する前に `TMP` と `TEMP` を設定してください。

```powershell
$env:TMP = 'Z:\Temp'
$env:TEMP = 'Z:\Temp'
.\jasna.exe --version
```

`--working-directory` は区間と LTX のファイル用のフォルダーを別に指定します。

## 全般

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--version` | — | Jasna のバージョンを表示して終了します。 |
| `--input` | — | 動画、画像、またはフォルダー。 |
| `--output` | — | 出力ファイル。`--input` がフォルダーの場合は出力フォルダー。 |
| `--output-pattern` | `{original}_out` | フォルダー入力時のファイル名テンプレート。`{original}` は入力ファイル名（拡張子なし）です。画像は元の拡張子を保持し、動画はテンプレートに拡張子があればそれを使います。Jasna は処理前に予定される出力を確認し、2 つの入力が同じファイルに対応する場合はエラーで終了します。 |
| `--device` | `cuda:0` | GPU の選択。AMD のカードも ROCm 経由で同じ `cuda:N` の名前を使います。 |
| `--batch-size` | `4`、AMD は `8` | 検出のバッチサイズ。旧 `rfdetr-v5` は常に 4 を使います。 |
| `--fp16` / `--no-fp16` | オン | 半精度計算。VRAM が減り、速くなることが多く、見た目の画質は落ちません。 |
| `--log-level` | `error` | `debug`、`info`、`warning`、`error`。 |
| `--no-progress` | オフ | プログレスバーを無効にします。 |

## 復元

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--restoration-model-name` | `basicvsrpp` | 動画復元モデル：`basicvsrpp`（高速）、`ltx-basic`（基本画質の LTX：`ltx` より約 1.7 倍速く、少しやわらかい）、`ltx`（標準画質の LTX：より精細、かなり遅い。おすすめ）、`ltx-4step`（良好画質：`ltx` より少し精細で約 1.5 倍遅い）、`ltx-8step`（高画質：`ltx` より精細で約 2.5 倍遅い）、`ltx-undistilled`（最高画質：少し良く、`ltx` より約 5 倍遅い）。[モデルの選び方](models.md#復元モデル)を参照。 |
| `--restoration-model-path` | `model_weights/lada_mosaic_restoration_model_generic_v1.2.pth`、LTX では `model_weights/ltx-restore` | 復元モデルの重み（LTX はフォルダー）。 |
| `--ltx-fast` / `--no-ltx-fast` | オフ | LTX：約 1.4 倍高速、ディテールはわずかに減ります。RTX 50 シリーズのみ。他の GPU ではエラーで止まります。 |
| `--ltx-large-canvas` / `--no-ltx-large-canvas` | オフ | LTX：大きなモザイクをより精細に。ほとんどの動画に大きなモザイクがあるため、動画全体で最大約 3 倍時間がかかります。空き VRAM が約 12 GB 必要です。足りない場合、そのモザイクは通常どおり復元します。 |
| `--ltx-overlap` | `17` | LTX：LTX が復元する区間どうしで共有するフレーム数（`17`、`25`、`33`、`41`、`49`）。モザイクが 4 秒以上続く場面では少ないほど速くなります。`33` は重なりの多い従来のモードです。 |
| `--ltx-vae-encoder` | `auto` | LTX：LTX が動画を読み込む方法。`full`（最もシャープで最も遅い）、`half`（速い）、`max`（最速、少しやわらかい）。`auto`: `ltx-basic` は `half`、それ以外は `full`。アプリの **LTX エンコーダー**と同じです。 |
| `--ltx-vae-decoder` | `auto` | LTX：LTX の結果を動画に戻す方法。`full`、`half`、`max`（上と同じ）。`auto`: `ltx-basic` は `max`、それ以外は `full`。アプリの **LTX デコーダー**と同じです。 |
| `--ltx-seed` | `20260923` | LTX：シードを変えると結果が少し変わります。同じシードなら同じ結果になります。 |
| `--ltx-trial` / `--no-ltx-trial` | オフ | LTX：モデルやライセンスなしで速度を試します。動画は復元されません。 |
| `--ltx-auto-split` | — | LTX：大きなモザイク（動画の高さのこの割合以上、例 `12`）を LTX、小さなものを BasicVSR++ で復元します。フォルダーにも使えます。`--segments`、`--fmp4`、`--frame-generation` とは併用できません。[区間](segments.md#cli---ltx-auto-split)を参照。 |
| `--compile-basicvsrpp` / `--no-compile-basicvsrpp` | オン | BasicVSR++ をお使いの GPU 向けに準備します。約 3 倍速く、VRAM はほとんど増えません。初回は 15～60 分かかります。NVIDIA のみ。詳しくは[調整ガイド](tuning.md)。 |
| `--max-clip-size` | `90` | 一度に復元するフレーム数。大きいほど安定して速くなりますが、VRAM を多く使います。収まる最大の値を使ってください。 |
| `--temporal-overlap` | `8` | 隣り合うクリップが共有するフレーム数で、つなぎ目のちらつきを隠します。`8`～`20` が目安で、`--max-clip-size` の半分未満にしてください。 |
| `--denoise` | `none` | 復元した部分のノイズを除去します: `low`、`medium`、`high`。 |

`--ltx-*` オプションには `--restoration-model-name ltx-basic`、`ltx`、`ltx-4step`、`ltx-8step` または `ltx-undistilled` が必要です。
`--no-compile-basicvsrpp` は `basicvsrpp` でのみ使えます。

## 検出

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--detection-model` | `rfdetr-v6` | インストール済みモデルは `model_weights/` から検出されます。`rfdetr-v6`（高速）、`rfdetr-vr-v1`（VR180）、`lada-yolo-v4` は同梱、`rfdetr-v6-large` と `zelefans-vr-yolo-v2` は任意のダウンロードです。詳しくは[モデル](models.md)。 |
| `--detection-model-path` | 自動 | デフォルトは `model_weights/<detection-model>` で、カードに合ったファイル形式を使います。RF-DETR は NVIDIA で `.onnx`、AMD で `.pt`。YOLO は常に `.pt`。 |
| `--detection-score-threshold` | 自動 | `0` から `1`。デフォルトはモデルの推奨値です（`rfdetr-v6`：0.35、`rfdetr-v6-large` と `rfdetr-vr-v1`：0.40、YOLO モデル：0.25）。モザイクを見逃す場合は下げ、通常の領域が誤検出される場合は上げてください。 |
| `--max-detection-gap` | `2` | モザイクが同じ位置に再出現する場合、最大 N フレームの検出途切れを補完します。`0` で無効。LTX は常に 3 フレーム以下の途切れを補完します（4 以上で LTX にも適用）。 |
| `--min-detection-duration` | `2` | N フレーム未満の検出を誤検出として破棄します（該当フレームは未処理のまま）。`0` で無効。 |
| `--scene-detection` | オン | ハードカット（シーン切り替え）を検出し、その位置で追跡中のモザイククリップをすべて終了します。クリップが 2 つのショットにまたがりません。`--no-scene-detection` で無効化。 |

## 二次復元

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--secondary-restoration` | `none` | `unet-4x`（NVIDIA・AMD）、`tvai`、または `rtx-super-res`（NVIDIA）。詳しくは[モデル](models.md)。 |
| `--rtx-scale` | `4` | RTX Super Res の拡大倍率（`2` または `4`）。LTX では常に `2`。 |
| `--rtx-quality` | `high` | `low`、`medium`、`high`、`ultra`。 |
| `--rtx-denoise` | `medium` | `none`、`low`、`medium`、`high`、`ultra`。 |
| `--rtx-deblur` | `none` | `none`、`low`、`medium`、`high`、`ultra`。 |
| `--tvai-ffmpeg-path` | Topaz のデフォルトインストールパス | Topaz Video の `ffmpeg.exe` のパス（Windows のみ）。 |
| `--tvai-model` | `iris-2` | 例: `iris-2`、`prob-4`、`iris-3`。 |
| `--tvai-scale` | `4` | 出力サイズは `256*scale`。`1` = 拡大なし。 |
| `--tvai-args` | `--help` を参照 | 追加の `tvai_up` パラメータ。 |
| `--tvai-workers` | `2` | 並列で動かす TVAI ffmpeg ワーカー数。 |
| `--tvai-denoise` | オフ | Topaz モデルの前にノイズを除去します。遅くなり、VRAM を多く使います。 |

## 動画全体を拡大

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--upscale` | `none` | `span`（高速、NVIDIA・AMD）、`realplksr`（高精細・低速、NVIDIA・AMD）、`rtx-super-res`（NVIDIA）、または `dlss5`（Windows、RTX 30 以降、ランタイムは別途）。詳しくは[高度な処理](advanced_processing.md)。 |
| `--upscale-height` | `2160` | `1080`、`1440`、`2160`。すでにこの高さ以上の動画は拡大しません。 |
| `--upscale-strength` | `1.0` | 0〜1。下げると普通の拡大が混ざります。 |
| `--upscale-rtx-quality` | `high` | `low`、`medium`、`high`、`ultra`。 |
| `--upscale-rtx-denoise` | `none` | `none`、`low`、`medium`、`high`、`ultra`。 |
| `--upscale-rtx-deblur` | `none` | `none`、`low`、`medium`、`high`、`ultra`。 |
| `--upscale-dlss5-preset` | `m` | `default`、`j`、`k`、`l`、`m`：DLSS の拡大モデル。 |
| `--upscale-dlss5-style` | `default` | `default`、`natural`、`cinematic`。 |
| `--upscale-dlss5-intensity` | `1.0` | DLSS 5 が光や質感を描き直す強さ、0〜2。 |

## SD 1.5 画像復元

静止画は自動的にここへルーティングされます。`--restoration-model-name` は動画専用です。

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--image-restoration-model-name` | `sd-15-jav` | 現在唯一の値。 |
| `--sd15-steps` | `25` | 画像を仕上げる回数。増やすと細部が増えることがありますが、遅くなります。 |
| `--sd15-strength` | `0.6` | モザイク部分をどれだけ描き直すか。最大 `0.7`。高いほど強いモザイクに効きますが、元の画像と似なくなることがあります。 |
| `--sd15-freeu` / `--no-sd15-freeu` | オン | 形がくっきりすることが多いです。硬いエッジが出るときはオフにしてください。 |
| `--sd15-seed` | `0` | シード。同じシードと設定なら同じ結果になります。 |
| `--sd15-variants` | `1` | N 個のバージョンを `name_v1` … `name_vN` として保存します（シード `seed` … `seed+N-1`）。最も良いものは自分で選んでください。 |

## VR

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--vr-mode` | `auto` | `auto`、`off`、`sbs`、`sbs-fisheye`、`tb`。詳しくは [VR180](vr180.md)。 |

## ジョブの再開

Ctrl+C、エラー、アプリ終了後は、同じコマンドを実行すると、LTX のジョブや `--segments` のジョブは
保存済みの処理から再開します。[一時停止と再開](models.md#一時停止と再開)をご覧ください。

## エンコード

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--codec` | `hevc` | オフライン出力用の `hevc`、`h264`、`av1`。HLS ストリーミングは常に H.264 を使います。 |
| `--cq` | GPU/コーデック別 | 動画の品質。低いほど高品質でファイルは大きくなります。NVIDIA のデフォルト: H.264 25、HEVC 28、AV1 35。AMD: H.264 24、HEVC 25、AV1 16。 |
| `--encoder-settings` | — | 高度な設定を JSON オブジェクトまたはカンマ区切りの `key=value` で指定。例: `{"rc-lookahead":32}` または `rc-lookahead=32,bf=4`。下記参照。 |
| `--lut` | — | 動画の書き出し時に適用される `.cube` カラー LUT（1D または 3D）。アプリでは**出力**タブの**カラー LUT (.cube)** でも設定できます。 |
| `--sharpen` | `0` | エンコード前に映像をシャープにします。`0`（無効）〜`1`（最強）。ffmpeg の `cas` フィルターと同じ結果になるため、再エンコードは不要です。詳しくは[高度な処理](advanced_processing.md)。 |
| `--retarget-high-fps` | オフ | 1 フレームおきに残して 60 → 30 FPS（および 59.94 → 29.97）に変換。他のレートは変わらず、音声もずれません。`--stream` や `--segments` とは併用できません。 |
| `--frame-generation` | `none` | `2x` で NVIDIA RTX Video Frame Generation によりフレームレートを 2 倍にし、動きを滑らかにします。RTX 40 シリーズ以降の GPU が必要です。音声と長さは変わりません。`--stream`、`--segments`、`--ltx-auto-split` とは併用できません。 |
| `--fmp4` | オフ | 作成中の `.mp4` / `.mov` 出力をそのまま再生できます。中断してもファイルは再生可能なままです。`--stream`、`--segments`、`--ltx-auto-split` とは併用できません。詳しくは[高度な処理](advanced_processing.md)。 |
| `--segments` | — | 1 本の動画のこの区間だけを復元します。秒または `mm:ss` で指定。例: `10-25,01:10-01:30.5`。`--stream`、`--retarget-high-fps`、`--fmp4`、`--ltx-auto-split`、`--frame-generation` とは併用できず、画像やフォルダーの入力にも使えません。詳しくは[区間](segments.md)。 |
| `--working-directory` | 出力ディレクトリ | 一時ファイルと保存済み進行状況の書き込み先。詳しくは[区間](segments.md)。 |

### コーデックの選び方

- **`hevc`**（デフォルト）: ファイルサイズの割に品質が良く、10 ビットです。
  最近のデバイスならほとんどで再生できます。迷ったらこれを使ってください。
- **`h264`**: ほぼどこでも再生できます（古いテレビ、ブラウザー、動画編集
  ソフト）。8 ビットのみで、同じ品質ならファイルは大きくなります。
  ストリーミングでは常にこれを使います。
- **`av1`**: 同じ品質で最も小さなファイルになり、10 ビットです。AV1 で
  エンコードできる GPU（NVIDIA RTX 40 シリーズ、AMD RX 7000 以降）と、新しめの
  プレーヤーが必要です。

`--segments` を使う場合、出力は常に入力と同じコーデックになります。別の
`--codec` を指定するとエラーで止まります。`--cq` と `--encoder-settings` は、
`--codec`（既定は `hevc`）が入力のコーデックと同じときだけ使われます。

### エンコーダー設定

`--cq` が主な品質設定です。低いほど高品質で、ファイルは大きくなります。
Jasna は数値をそのままエンコーダーに渡します（AMD の AV1 だけは 5 倍します）。
コーデックを変えても数値は変換されません。

| GPU | H.264 デフォルト | HEVC デフォルト | AV1 デフォルト | 許容範囲 |
| --- | ---: | ---: | ---: | --- |
| NVIDIA | 25 | 28 | 35 | H.264/HEVC は 1–51、AV1 は 1–63 |
| AMD | 24 | 25 | 16 | H.264/HEVC は 0–51、AV1 は 1–51 |

NVIDIA では CQ 0 は「自動」を意味するため、Jasna では 1 以上の値が必要です。
GUI はコーデックごとに別の CQ 値を記憶します。

`--encoder-settings` はその他のエンコーダー設定を変えます。Jasna はすべての
キーを確認します。エンコーダーが知らないキーがあると、受け付けるキーの一覧を
示すエラーで処理が止まります:

```bash
# 高品質（大きなファイル）にするには CQ を下げます。
jasna --input in.mp4 --output out.mkv --cq 22

# CQ と高度なキー
jasna --input in.mp4 --output out.mkv --cq 22 --encoder-settings "rc-lookahead=32,bf=4"
```

`--cq` を使わない場合は、`--encoder-settings` 内の `cq=22` も引き続き使えます。
両方で CQ を指定するとエラーになります。GUI では、CQ は CQ コントロールでのみ
設定でき、**カスタム引数**では設定できません。

#### NVIDIA（NVENC）のキー、全コーデック共通

| キー | 説明 |
| --- | ------------ |
| `cq` | VBR の目標品質。低いほど高品質でファイルが大きくなります。H.264/HEVC の範囲は 1–51（デフォルト 25/28）、AV1 は 1–63（デフォルト 35）。出力サイズの自動上限により、近い値で同じ結果になる場合があります。 |
| `preset` | 速度と品質のトレードオフ。`p1`（最速）から `p7`（最高品質）。デフォルト `p5`。 |
| `tune` | `hq`（デフォルト）、`ll`、`ull`、または `lossless`。 |
| `rc` | レート制御モード: `vbr`（デフォルト）、`cbr`、`constqp`。 |
| `qmin` / `qmax` | VBR の品質の下限/上限。デフォルト 17/34（H.264/HEVC のみ。AV1 は別の 0–255 QP スケールを使うため未設定のままです）。 |
| `init_qpI` / `init_qpP` / `init_qpB` | フレームタイプごとの初期量子化値。デフォルト 17（H.264/HEVC）。 |
| `g` | キーフレーム間隔（フレーム数）。デフォルト 250。小さいほどシークしやすく、ファイルは大きくなります。 |
| `bf` | 連続 B フレームの最大数。デフォルト 4。 |
| `b_ref_mode` | B フレームを参照として使用: `disabled`、`each`、`middle`（デフォルト）。 |
| `b_adapt` | 適応的な B フレーム配置。 |
| `nonref_p` | 非参照 P フレーム。デフォルトで有効。 |
| `spatial_aq` / `spatial-aq` | 空間適応量子化: 目につきやすい部分により多くのビットを割り当てます。デフォルトでオン。AV1 はハイフン付きの表記のみ受け付けます。 |
| `temporal-aq` | 時間適応量子化。デフォルトでオン。 |
| `aq-strength` | AQ の強さ。1–15。デフォルト 8。 |
| `rc-lookahead` | レート制御のために先読みするフレーム数。デフォルト 32。 |
| `lookahead_level` | 先読みの品質。0–3。HEVC/AV1 のみ。H.264 では警告付きで無視されます。 |
| `maxrate` / `bufsize` | ビットレート上限と VBV バッファサイズ（bit/秒）。Jasna がソースのビットレートから自動設定します（下記参照）。`maxrate` を指定するとその値が使われます。 |
| `multipass` | 2 パスエンコード: `disabled`、`qres`、`fullres`。 |
| `weighted_pred` | 重み付き予測。NVENC は `bf=0` と組み合わせた場合のみ対応します。それ以外（および AV1 では常に）警告付きで無視されます。 |
| `tf_level` | 時間フィルタリングのレベル。 |

コーデック別の追加キー:

| コーデック | 追加キー |
| ----- | ---------- |
| `hevc` | `profile`（デフォルト `main10`。`main` は古いプレーヤーや機器向けの 8-bit 映像）、`tier` |
| `h264` | `profile`（`baseline`、`main`、`high` — デフォルト `high`）、`coder`（`cabac`/`cavlc`） |
| `av1` | `tier`、`tile-rows`、`tile-columns`（大きなフレームのデコードを並列化） |

#### 出力サイズの自動上限

`cq` は、ソースがどうエンコードされていても一定の品質を狙います。そのため
ビットレートの低いソースは、出力が数倍に大きくなってしまいます。これを防ぐため、
Jasna はソース映像のビットレートから `maxrate` を決め、`bufsize` をその 2 倍に
します:

| 条件 | 上限 |
| ---- | ---- |
| NVIDIA H.264 出力 | ソース映像ビットレートの 2.0 倍 |
| その他の出力、HEVC ソース | ソース映像ビットレートの 1.25 倍 |
| その他の組み合わせ | ソース映像ビットレートの 1.0 倍 |

NVIDIA H.264 は復元したディテールを保つためにより多くのビットが必要なので、
余裕を持たせています。上限が関係するのはビットレートの低いソースだけです。
ビットレートの高いソースは、もともと上限より下に収まります。上限に達すると、
近い CQ 値で同じファイルサイズになることがあります。

自分で `maxrate` を指定すると上限を置き換えられます。非常に大きな値にすると
オフになります。ソースにビットレートの情報がない場合、Jasna は警告を記録し、
上限なしでエンコードします。AMD はデフォルトで固定 QP を使うため、上限はありません。

#### AMD（AMF）のキー、全コーデック共通

| キー | 説明 |
| --- | ------------ |
| `cq` | AMF の品質値。固定 QP として渡します（AV1 は内部の目盛りが細かいため CQ を 5 倍して渡します）。H.264 で `rc=qvbr` を指定した場合は QVBR の品質レベルになります。低いほど高品質。デフォルトは H.264 24、HEVC 25、AV1 16。 |
| `qvbr_quality_level` | 従来の CQ 別名。`--cq` を省略した場合は CLI の高度な設定で利用できますが、GUI のカスタム引数では利用できません。 |
| `usage` | エンコーダーの用途プロファイル。デフォルト `high_quality`。 |
| `quality` | 速度/品質プリセット: `speed`、`balanced`、`quality`（デフォルト）。 |
| `rc` | レート制御モード。デフォルト `cqp`。 |
| `preset` | AMF プリセット。 |
| `g` | キーフレーム間隔（フレーム数）。デフォルト 250。 |
| `bf` | 連続 B フレームの最大数。 |
| `preanalysis` | 事前分析パス。デフォルトで無効。 |
| `maxrate` / `bufsize` | ビットレート上限と VBV バッファサイズ（bit/秒）。CQP モード以外ではソースのビットレートから自動設定されます。 |
| `profile` / `level` | コーデックのプロファイルとレベル。 |

コーデック別の追加キー:

| コーデック | 追加キー |
| ----- | ---------- |
| `hevc` | `tier`、`bitdepth`（デフォルト 10。`8` または `profile=main` で 8-bit 映像）、`vbaq`（デフォルトで無効） |
| `h264` | `coder`、`bf_ref`（B フレーム参照）、`pa_adaptive_mini_gop`（適応型 B フレーム配置）、`vbaq`（デフォルトで無効） |
| `av1` | `bitdepth`（デフォルト 10。`8` で 8-bit 映像）、`aq_mode`（デフォルトで無効） |

## ストリーミング

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--stream` | オフ | HLS ストリーミングモード。ファイル出力はありません。詳しくは[ストリーミング](streaming.md)。 |
| `--stream-port` | `8765` | HTTP ポート。 |
| `--stream-segment-duration` | `4.0` | HLS セグメント長（秒）。 |
| `--no-browser` | オフ | ブラウザウィンドウを開きません。 |

## エクスポート後

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--post-export-action` | `none` | すべての動画が終わった後に実行します。`shutdown` はすぐに PC の電源を切ります（GUI と違いカウントダウンはありません）。`command` は `--post-export-command` を実行します。 |
| `--post-export-command` | — | `--post-export-action command` 用のシェルコマンド。 |
| `--post-export-video-command` | — | 正常に書き出された各動画の後に実行するシェルコマンド。`{input}`、`{output}`、`{output_dir}`、`{output_stem}`、`{output_suffix}` を使用できます。 |

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
jasna --input folder_in --output folder_out --post-export-video-command "ffmpeg -i {output} -map 0 -map_metadata 0 -map_chapters 0 -c copy -movflags +faststart {output_dir}/{output_stem}_remuxed{output_suffix}"
```

最後の例は元の出力を残し、リマックスした 2 つ目のファイルを書き出します。
ファイルを置き換えるには、一時ファイルを使うスクリプトを使ってください。
`ffmpeg` は `PATH` に入れるか、フルパスで書いてください。

## 有効化

| オプション | 説明 |
| ------ | ----- |
| `--license-code CODE` | GPU の準備前に、ライセンスコードで有効化します。処理オプションがなければ有効化して終了します。TPM 2.0 セキュリティチップのある PC では、その後オフラインで動きます。それ以外の場合と、Linux のコンテナーや仮想マシンの中では常に、支援者向けモデルを使う間、ライセンスをオンラインで確認します。このオプションの代わりに環境変数 `JASNA_LICENSE_CODE` を設定すると、コードがコマンドラインに出ません。まだ有効化されていない PC でだけ使われます。 |
| `--online` | `--license-code` または `JASNA_LICENSE_CODE` と一緒に使います。オフラインで使わず、ライセンスをオンラインで確認します。レンタル・共用の PC で使うと、3 台の枠を使い続けません。 |
| `--activation-status` | 有効化されているか、使用中のデバイスとその ID を表示して終了します。有効化されていなければ 1 で終了します。 |

```bash
jasna --license-code "YOUR_LICENSE_CODE"
jasna --license-code "YOUR_LICENSE_CODE" --input input.mp4 --output output.mkv
jasna --activation-status
```

- オンラインで動く場合は、メッセージに理由が表示されます。Linux で `sudo usermod -aG tss $USER` が必要なときは、Jasna が知らせます。
- レンタル GPU（RunPod、Vast）: `XDG_CONFIG_HOME` を残しておく非公開の
  フォルダーに設定します。このフォルダーに有効化の情報が入るので、毎回同じ
  フォルダーを使ってください:

  ```bash
  export XDG_CONFIG_HOME=/root/jasna-config
  ./jasna --license-code "YOUR_LICENSE_CODE"
  ```

  RunPod の `/workspace` は使わないでください。詳しい手順: [クラウド GPU](cloud.md)。
- 台数の枠と、有効化に失敗したときの対処:
  [支援者向けモデルの有効化](gui.md#支援者向けモデルの有効化)。

## ベンチマーク

| オプション | 説明 |
| ------ | ----- |
| `--benchmark` | 処理の代わりにベンチマークを実行します。 |
| `--benchmark-filter` | 名前にこの文字列を含むベンチマークのみ実行します。 |
| `--benchmark-video` | ベンチマークに使う動画のパス。複数回指定できます。 |
