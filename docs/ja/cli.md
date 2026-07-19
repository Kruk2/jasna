# CLI リファレンス

Jasna の CLI は GUI と同じ機能を提供します。`jasna --help` で常に最新の全オプション一覧を表示できます。このページは補足とサンプルを提供します。

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image (routes to SD 1.5 automatically)
jasna --input photo.png --output restored.png

# Whole folder (images first, then videos)
jasna --input input_folder --output output_folder
```

Windows では、CLI もアプリ本体と同じファイルです: `jasna.exe --input ...`。

## 全般

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--version` | — | Jasna のバージョンを表示して終了します。 |
| `--input` | — | 動画、画像、またはフォルダ。 |
| `--output` | — | 出力ファイル。`--input` がフォルダの場合は出力フォルダ。 |
| `--output-pattern` | `{original}_out` | フォルダ入力時のファイル名テンプレート。`{original}` は入力ファイル名（拡張子なし）です。画像は元の拡張子を保持し、動画はテンプレートに拡張子があればそれを使います。Jasna は処理前に予定される出力を確認し、2 つの入力が同じファイルに対応する場合はエラーで終了します。 |
| `--device` | `cuda:0` | GPU の選択。AMD Linux では MIGraphX の GPU も選択します。 |
| `--batch-size` | `4` | 検出のバッチサイズ。 |
| `--fp16` / `--no-fp16` | オン | 対応箇所（復元 + TensorRT）で FP16 を使用。VRAM を抑え、速度が上がる場合があります。 |
| `--log-level` | `error` | `debug`、`info`、`warning`、`error`。 |
| `--no-progress` | オフ | プログレスバーを無効にします。 |

## 復元

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--restoration-model-name` | `basicvsrpp` | 動画復元モデル（現在は `basicvsrpp` のみ）。 |
| `--restoration-model-path` | `model_weights/lada_mosaic_restoration_model_generic_v1.2.pth` | 復元モデルの重み。 |
| `--compile-basicvsrpp` / `--no-compile-basicvsrpp` | オン | TensorRT コンパイル: 大幅な高速化、VRAM 増。詳しくは[調整ガイド](tuning.md)。 |
| `--max-clip-size` | `90` | 追跡するクリップの最大フレーム数。VRAM の主な調整項目です。 |
| `--temporal-overlap` | `8` | クリップ分割位置でのオーバーラップ+破棄マージン。境界のフリッカーを軽減します。 |
| `--enable-crossfade` / `--no-enable-crossfade` | オン | 処理済みフレームを再利用してクリップ境界をクロスフェード。追加の GPU コストはありません。 |
| `--denoise` | `none` | 復元済みクロップの空間ノイズ除去: `low`、`medium`、`high`。 |
| `--denoise-step` | `after_primary` | ノイズ除去をセカンダリの前（`after_primary`）に適用するか、合成の直前（`after_secondary`）に適用するか。 |

## 検出

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--detection-model` | `rfdetr-v5` | インストール済みモデルは `model_weights/` から検出されます。`zelefans-vr-yolo-v2` は同梱です。詳しくは[モデル](models.md)。 |
| `--detection-model-path` | 自動 | デフォルトは `model_weights/<detection-model>.onnx`（RF-DETR）または `.pt`（YOLO）。 |
| `--detection-score-threshold` | `0.25` | モザイクを見逃す場合は下げ、通常の領域が誤検出される場合は上げてください。 |
| `--max-detection-gap` | `2` | モザイクが同じ位置に再出現する場合、最大 N フレームの検出途切れを補完します。`0` で無効。 |
| `--min-detection-duration` | `2` | N フレーム未満の検出を誤検出として破棄します（該当フレームは未処理のまま）。`0` で無効。 |

## セカンダリ復元

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--secondary-restoration` | `none` | `unet-4x`、`tvai`、または `rtx-super-res`。詳しくは[モデル](models.md)。 |
| `--rtx-scale` | `4` | RTX Super Res の拡大倍率（`2` または `4`）。 |
| `--rtx-quality` | `high` | `low`～`ultra`。 |
| `--rtx-denoise` | `medium` | `none` で無効。 |
| `--rtx-deblur` | `none` | `none` で無効。 |
| `--tvai-ffmpeg-path` | Topaz のデフォルトインストールパス | Topaz Video の `ffmpeg.exe` のパス。 |
| `--tvai-model` | `iris-2` | 例: `iris-2`、`prob-4`、`iris-3`。 |
| `--tvai-scale` | `4` | 出力サイズは `256*scale`。`1` = 拡大なし。 |
| `--tvai-args` | `--help` を参照 | 追加の `tvai_up` パラメータ。 |
| `--tvai-workers` | `2` | 並列で動かす TVAI ffmpeg ワーカー数。 |

## SD 1.5 画像復元

静止画は自動的にここへルーティングされます。`--restoration-model-name` は動画専用です。

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--image-restoration-model-name` | `sd-15-jav` | 現在唯一の値。 |
| `--sd15-steps` | `25` | 拡散ステップ数。 |
| `--sd15-strength` | `0.6` | SDEdit のノイズ除去強度。`<= 0.7` に制限されます。 |
| `--sd15-freeu` / `--no-sd15-freeu` | オン | FreeU による UNet の調整。 |
| `--sd15-seed` | `0` | ベースシード。 |
| `--sd15-variants` | `1` | シード `seed..seed+N-1` で N 個のバリエーションを生成し、最も良いものを残します。 |

## VR

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--vr-mode` | `auto` | `auto`、`off`、`sbs`、`sbs-fisheye`。詳しくは [VR180](vr180.md)。 |

## エンコード

| オプション | デフォルト | 説明 |
| ------ | ------- | ----- |
| `--codec` | `hevc` | オフライン出力用の `hevc`、`h264`、`av1`。HLS ストリーミングは常に H.264 を使います。 |
| `--encoder-settings` | — | JSON オブジェクトまたはカンマ区切りの `key=value`。例: `{"cq":22}` または `cq=22,rc-lookahead=32`。下記参照。 |
| `--lut` | — | エンコード前に GPU で適用される `.cube` カラー LUT（1D または 3D）。GUI のエンコードセクションでも設定できます。 |
| `--retarget-high-fps` | オフ | 1 フレームおきに処理して 60 → 30 FPS（および 59.94 → 29.97）に変換。他のレートは変更せず、音声のタイミングは維持されます。 |
| `--segments` | — | 選択した範囲だけを復元します。例: `10-25,01:10-01:30.5`。`--stream` および `--retarget-high-fps` とは併用できません。詳しくは[区間](segments.md)。 |
| `--working-directory` | 出力ディレクトリ | 区間処理の一時ファイルの書き込み先。詳しくは[区間](segments.md)。 |

### コーデックの選び方

- **`hevc`**（デフォルト）: 品質とファイルサイズのバランスが最も良く、
  10 ビットでエンコードします。最近のデバイスとプレーヤーならどれでも
  再生できます。特別な理由がなければこれを使ってください。
- **`h264`**: 最大の互換性（古いテレビ、ブラウザ、編集ソフト）。
  8 ビットのみで、同じ品質ならファイルは大きくなります。ストリーミングで
  使われるコーデックでもあります。
- **`av1`**: 最高の圧縮率 — 同じ品質で最も小さなファイルになり、
  10 ビットです。AV1 エンコードに対応した世代の GPU（NVIDIA RTX 40
  シリーズ以降）と、比較的新しいプレーヤーが必要です。

`--segments` を使う場合、コーデックは入力動画のコーデックに固定され、
`--codec` は適用されません。

### エンコーダー設定

`--encoder-settings` はハードウェアエンコーダーを細かく調整します。キーは
使用中のエンコーダーに対して検証され、未対応のキーは、そのエンコーダーが
受け付けるキーの一覧付きの分かりやすいエラーになります。ほとんどの場合、
`cq` 以外は必要ありません:

```bash
# Higher quality (bigger file): lower cq. Default is 25 (HEVC), 24 (H.264), 32 (AV1).
jasna --input in.mp4 --output out.mkv --encoder-settings "cq=22"

# Multiple keys
jasna --input in.mp4 --output out.mkv --encoder-settings "cq=22,rc-lookahead=32,bf=4"
```

#### NVIDIA（NVENC）のキー — 全コーデック共通

| キー | 説明 |
| --- | ------------ |
| `cq` | VBR の目標品質。**主な品質調整項目です。** 低いほど高品質でファイルが大きくなります。H.264/HEVC は 0–51（デフォルト 24/25）、AV1 は 0–63（デフォルト 32）。 |
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
| `spatial_aq` / `spatial-aq` | 空間適応量子化 — 目につきやすい部分にビットを割り当てます。デフォルトでオン。AV1 はハイフン付きの表記のみ受け付けます。 |
| `temporal-aq` | 時間適応量子化。デフォルトでオン。 |
| `aq-strength` | AQ の強さ。1–15。デフォルト 8。 |
| `rc-lookahead` | レート制御のために先読みするフレーム数。デフォルト 32。 |
| `lookahead_level` | 先読みの品質。0–3。HEVC/AV1 のみ — H.264 では警告付きで無視されます（エンコーダーが使用できません）。 |
| `maxrate` / `bufsize` | ビットレート上限と VBV バッファサイズ。ビットレートに厳密な上限が必要な場合に使います。 |
| `multipass` | 2 パスエンコード: `disabled`、`qres`、`fullres`。 |
| `weighted_pred` | 重み付き予測。NVENC は `bf=0` と組み合わせた場合のみ対応します。それ以外（および AV1 では常に）警告付きで無視されます。 |
| `tf_level` | 時間フィルタリングのレベル。 |

コーデック別の追加キー:

| コーデック | 追加キー |
| ----- | ---------- |
| `hevc` | `profile`（`main`、`main10` — デフォルト `main10`）、`tier` |
| `h264` | `profile`（`baseline`、`main`、`high` — デフォルト `high`）、`coder`（`cabac`/`cavlc`） |
| `av1` | `tier`、`tile-rows`、`tile-columns`（大きなフレームのデコードを並列化） |

#### AMD（AMF）のキー — 全コーデック共通

| キー | 説明 |
| --- | ------------ |
| `cq` | 汎用の品質調整項目。AMF の `qvbr_quality_level` に自動変換されます。低いほど高品質。デフォルトは 24（H.264）、25（HEVC）、32（AV1）。 |
| `qvbr_quality_level` | AMF ネイティブの品質レベル。直接設定したい場合に使います。 |
| `usage` | エンコーダーの用途プロファイル。デフォルト `high_quality`。 |
| `quality` | 速度/品質プリセット: `speed`、`balanced`、`quality`（デフォルト）。 |
| `rc` | レート制御モード。デフォルト `qvbr`。 |
| `preset` | AMF プリセット。 |
| `g` | キーフレーム間隔（フレーム数）。デフォルト 250。 |
| `bf` | 連続 B フレームの最大数。 |
| `preanalysis` | 事前分析パス。デフォルトで有効。 |
| `vbaq` | 分散ベースの適応量子化。デフォルトで有効。 |
| `maxrate` / `bufsize` | ビットレート上限と VBV バッファサイズ。 |
| `profile` / `level` | コーデックのプロファイルとレベル。 |

コーデック別の追加キー:

| コーデック | 追加キー |
| ----- | ---------- |
| `hevc` | `tier`、`bitdepth`（デフォルト 10） |
| `h264` | `coder` |
| `av1` | `bitdepth`（デフォルト 10） |

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
| `--post-export-action` | `none` | `shutdown` または `command`。すべてのエクスポート完了後に実行されます。 |
| `--post-export-command` | — | `--post-export-action command` 用のシェルコマンド。 |

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```

## ライセンス

| オプション | 説明 |
| ------ | ----- |
| `--license-email` | キーに紐付いた支援者メールアドレス（unet-4x と SD 1.5 を解除）。 |
| `--license-key` | そのメールアドレスに発行されたライセンスキー。 |

GUI は初回入力後にこれらを保存します。CLI フラグはスクリプトでの利用向けです。

## ベンチマーク

| オプション | 説明 |
| ------ | ----- |
| `--benchmark` | 処理の代わりにベンチマークを実行します。 |
| `--benchmark-filter` | 名前にこの文字列を含むベンチマークのみ実行します。 |
| `--benchmark-video` | ベンチマークに使う動画のパス。複数回指定できます。 |
