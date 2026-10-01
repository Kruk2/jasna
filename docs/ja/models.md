# モデルの選び方

## 検出モデル

検出モデルは各フレーム内のモザイクを見つけます。

- **最新の RF-DETR モデル**（`rfdetr-v6`）を使ってください — デフォルトかつ
  高速で、最もバランスの取れた選択です。Jasna に同梱されています。
- **`rfdetr-v6-large`** は高品質だが低速の派生モデルで、4K 動画ではこちらの
  ほうが良い場合があります。別途ダウンロードの
  任意モデルです。お使いのグラフィックスカード用のファイルを
  `model_weights/` に置くと自動的に検出されます。
  - NVIDIA:
    [rfdetr-v6-large.onnx](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx)
  - AMD:
    [rfdetr-v6-large.pt](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.pt)
- **Lada YOLO** モデルは 2D アニメーションでより良い場合があります。
- **rfdetr-vr-v1**（同梱）は VR180 用の RF-DETR 検出モデルで、VR180 動画に最適です。
- **zelefans-vr-yolo-v2**（任意ダウンロード）は VR180 の代替検出モデルです。
- **AMD では** RF-DETR はグラフィックスカードで動き、`.pt` のモデルファイルを
  使います（NVIDIA は `.onnx`）。NVIDIA より遅いので、検出品質より速度を
  優先する場合は `lada-yolo-v4` を選んでください。

各モデルは既定で推奨検出しきい値を適用します（`rfdetr-v6`：0.35、
`rfdetr-v6-large`：0.40）。`--detection-score-threshold` で上書きできます。

旧 `rfdetr-v5` モデルも引き続き利用できます。

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v6
```

[区間エディター](segments.md)の中で、動画ごとに別の検出モデルを
設定することもできます。

## 復元モデル

**BasicVSR++** が既定です。高速で、対応するすべての GPU で動きます。
**LTX** はかなり遅いですが、より精細で自然な仕上がりになります。特に
大事な動画やシーンに使ってください。**修復モデル**セクションで選ぶか、
次のように指定します。

```bash
jasna --input input.mp4 --output output.mkv --restoration-model-name ltx
```

- NVIDIA RTX のみ。VR180 サイドバイサイド動画とストリーミングには
  使えません。LTX ではセカンダリ復元とノイズ除去はオフになります。
- 動画プレーヤーと**復元プレビュー**は常に BasicVSR++ を使います。LTX を
  試すには[区間エディター](segments.md)の**シードを試す**を使います。
- 時間は動画の長さではなく、モザイクが映っている長さで決まります。初回は
  Jasna が LTX をお使いの GPU 向けに準備するため、さらに時間がかかります。
- モデルは同梱されていません。約 15 GB（高速モードは約 9 GB）です。
  ダウンロードが公開されると、Jasna はダウンロード前に確認します。

### LTX のオプション

- **モデル**: **蒸留**（既定）または**非蒸留**。非蒸留は少し高品質で、
  約 2 倍遅くなります（さらに 13 GB のダウンロード）。CLI:
  `--restoration-model-name ltx-undistilled`。
- **高速モード**: RTX 50 シリーズのみ。約 1.4 倍速く、ディテールは
  わずかに減り、ダウンロードも小さくなります。CLI: `--ltx-fast`。
- **大きなモザイクをより鮮明に**: 大きなモザイク（クローズアップ、4K）を
  より精細にします。その部分は約 3 倍遅くなります。空き VRAM が 10 GB
  必要で、足りない場合は効果がありません。CLI: `--ltx-large-canvas`。
- **シード**: シードを変えると結果が少し変わり、同じシードなら同じ結果に
  なります。CLI: `--ltx-seed 7`。
- **お試しモード**: モデルファイルなしで LTX を実際の速度で動かし、
  お使いの PC での速さを確認できます。動画は復元されません。CLI:
  `--ltx-trial`。

### 対応 GPU と所要時間

**モザイクが常に映っている 1080p 動画 1 分**あたりのおおよその時間
（蒸留モデル）です。モザイクのないシーンはずっと速く、非蒸留は約 2 倍
かかります。

| GPU | LTX | 時間 | 高速モード |
| --- | --- | ---: | ---: |
| RTX 5090 | 最速 | 約 5 分 | 約 3 分 |
| RTX 4090 | 快適 | 約 7 分 | — |
| RTX 5080 | 快適 | 約 10 分 | 約 6 分 |
| RTX 3090 | 動くが遅め | 約 15 分 | — |
| RTX 2080 Ti | 動くがとても遅い | 約 45 分 | — |
| GTX 16 シリーズ | 未検証 | — | — |
| AMD | 非対応 | — | — |

- **VRAM**: 24 GB ならモデル全体が GPU に載ります。16 GB でも十分で、
  11～12 GB でも動きますが遅くなります。11 GB 未満は未検証です。
- **システム RAM**: 32 GB を推奨します（モデルの読み込みに約 19 GB 使います）。16 GB の場合は、約 10 GB で動く Fast モードを使ってください。

## セカンダリ復元

Jasna は各モザイク領域の 256x256 クロップを復元します。そのため、
大きなモザイク領域、クローズアップ、4K 動画は一次復元の後にぼやけて
見えることがあります。セカンダリモデルは、復元済みクロップを元の映像へ
合成する前に 512x512 または 1024x1024 へアップスケールし、目に見えて
シャープにします。

現在、セカンダリ復元は NVIDIA GPU でのみ利用できます。AMD 版は
一次復元のみ対応しています。

- **unet-4x**: 支援者モデル。現在のテストでは TVAI より高速で同程度の
  品質です。ドメイン内データセットで訓練されており、見た目は TVAI
  `iris-2` に近いです。
  SLS Discord の例
  をご覧ください。支援者キーで解除します — 詳しくは
  [プロジェクトを支援する](../../README.ja.md)をご覧ください。
- **RTX Super Resolution**: 非常に高速で無料、追加のものは不要です。
  品質はまずまずです。一部の動画ではフリッカーが出る場合があるため、
  まず短いクリップで試してください。
- **TVAI**: RTX Super Resolution より高品質で unet-4x と同程度ですが、
  非常に遅いです。[Topaz Video](https://www.topazlabs.com/topaz-video)
  が必要です。有料で Windows のみです。推奨モデル: `iris-2`。

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

TVAI では、環境変数 `TVAI_MODEL_DATA_DIR` と `TVAI_MODEL_DIR` を
Topaz Video のモデルフォルダに、以下のように設定してください
（`--tvai-args` で Topaz モデルのパラメータをさらにカスタマイズできます）:

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### 速度と VRAM の比較

| セカンダリ種別           | CAWD 1080p        | K109 1080p      |
| ------------------------ | -----------------:| -----------------:|
| セカンダリなし           | 22秒 / 10.0 GB VRAM | 11秒 / 10.7 GB VRAM |
| unet-4x                  | 29秒 / 12.5 GB VRAM | 14秒 / 12.6 GB VRAM |
| RTX Super-Res            | 25秒 / 11.7 GB VRAM | 13秒 / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52秒 / 12.1 GB VRAM | 24秒 / 12.4 GB VRAM |

## 静止画復元（SD 1.5）

静止画では、Jasna は動画パイプラインの代わりに、ファインチューニング済みの
Stable Diffusion 1.5 inpaint モデルを使います。GUI のキューに画像を追加する
（または CLI で渡す）だけで、画像ジョブは自動的に SD 1.5 へルーティング
されます:

```bash
jasna --input photo.png --output restored.png
```

- モデルは**同梱されておらず**、約 **6.9 GB** です。Jasna は
  [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav)
  からダウンロードする前に確認します。
- 現在は支援者のみ利用でき、unet-4x と同じキーを使います — 詳しくは
  [プロジェクトを支援する](../../README.ja.md)をご覧ください。
- 推論中は約 **7 GB VRAM**、大きな 4K 画像ではもう少し多く必要です。

SD 1.5 経路は実験的です。結果はシーンによって変わりますが、うまく合う
画像では非常に良い結果になることがあります。バリエーションをいくつか
生成して、最も良いものを残してください:

```bash
jasna --input photo.png --output restored.png --sd15-variants 4
```

すべての調整項目（`--sd15-steps`、`--sd15-strength`、`--sd15-seed` など）は
[CLI リファレンス](cli.md)に一覧があります。

例:
SLS Discord の SD 1.5 例
とその他の例。
