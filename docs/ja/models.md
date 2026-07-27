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

## セカンダリ復元

Jasna は各モザイク領域の 256x256 クロップを復元します。そのため、
大きなモザイク領域、クローズアップ、4K 動画は一次復元の後にぼやけて
見えることがあります。セカンダリモデルは、復元済みクロップを元の映像へ
合成する前に 512x512 または 1024x1024 へアップスケールし、目に見えて
シャープにします。

- **unet-4x**: 支援者モデル。現在のテストでは TVAI より高速で同程度の
  品質です。JAV ドメイン内データセットで訓練されており、見た目は TVAI
  `iris-2` に近いです。
  [SLS Discord の例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)
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

| セカンダリ種別           | CAWD 1080p        | KV-109 1080p      |
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
[SLS Discord の SD 1.5 例](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106)
と[その他の例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)。
