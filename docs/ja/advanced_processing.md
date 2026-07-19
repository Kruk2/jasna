# 高度な処理

特別な場面のためのオプション機能です。ここにあるものはすべて GUI
（対応する設定を探してください。それぞれにツールチップがあります）と
CLI の両方で使えます。

## ノイズ除去

復元された領域にはノイズが残ることがあります。ノイズ除去の設定
（`--denoise low|medium|high`）は、復元された領域だけに穏やかな空間
ノイズ除去を適用します — フレームのそれ以外の部分はそのままです。
まず `low` から始め、ノイズが残る場合だけ上げてください。

デフォルトではセカンダリ復元の前に実行されます。
`--denoise-step after_secondary` を指定すると、合成の直前に移動します。

## 60 FPS から 30 FPS への書き出し

60（または 59.94）FPS の入力では、**60 FPS を 30 FPS に変換**
（`--retarget-high-fps`）が 1 フレームおきに処理して 30（または 29.97）
FPS で出力します — 処理量は半分になります。音声のタイミングと再生速度は
維持されます。他のフレームレートは変更されません:

```bash
jasna --input input.mp4 --output output.mp4 --retarget-high-fps
```

[区間処理](segments.md)とは併用できません。

## カラー LUT

`.cube` カラー LUT（1D または 3D）を出力に適用できます — カラー
グレーディングや、決まったルックに合わせるためです。GUI のエンコード
セクションか、`--lut path/to/look.cube` で設定します。LUT はエンコード
直前に GPU で適用されるため、コストはほとんどかかりません。

## カスタムエンコーダー設定

**カスタム引数**フィールド（`--encoder-settings`）は、ハードウェア動画
エンコーダーを細かく調整します — 品質レベル、ビットレート上限、
キーフレーム間隔など。主な調整項目は `cq` です（低いほど高品質で
ファイルが大きくなります）:

```bash
jasna --input in.mp4 --output out.mkv --encoder-settings "cq=22"
```

各コーデックで使えるすべてのキーは [CLI リファレンス](cli.md)に
記載されています。

## エクスポート後のアクション

キュー全体が完了したときに何かを実行できます: **PC をシャットダウン**、
または**カスタムコマンド**（たとえば通知スクリプト）です。GUI の
エクスポート後のアクションセクションか、CLI で設定します:

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```
