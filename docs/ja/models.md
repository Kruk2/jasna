# モデルの選び方

## 検出モデル

検出モデルは各フレームのモザイクを見つけます。

- **`rfdetr-v6`**（既定、同梱）: 高速で、ほとんどの動画に合います。迷ったら
  これを使ってください。
- **`rfdetr-v6-large`**: 検出は良くなりますが遅くなります。4K では良い場合が
  あります。別途ダウンロードです。お使いの GPU 用のファイルを
  `model_weights/` に置くと、Jasna が自動で見つけます。
  - NVIDIA:
    [rfdetr-v6-large.onnx](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx)
  - AMD:
    [rfdetr-v6-large.pt](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.pt)
- **`rfdetr-vr-v1`**（同梱）: VR180 動画に最適です。
- **`zelefans-vr-yolo-v2`**（別途ダウンロード）: もう 1 つの VR180 用モデルです。
- **Lada YOLO** モデル: 2D アニメーションでは良い場合があります。
- 旧 `rfdetr-v5` も引き続き使えます。

AMD では RF-DETR は NVIDIA より遅くなります。検出品質より速度を優先する
場合は `lada-yolo-v4` を使ってください。

各モデルには既定の信頼度があります（`rfdetr-v6`: 0.35、
`rfdetr-v6-large`: 0.40）。`--detection-score-threshold` で変更できます。

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v6
```

[区間エディター](segments.md)では、動画ごとに別の検出モデルを選べます。

## 復元モデル

| | **BasicVSR++**（既定） | **LTX** |
| --- | --- | --- |
| 速度 | 速い | かなり遅い（動画 1 分あたり数分） |
| 品質 | 良い | より精細で自然 |
| GPU | 対応するすべての GPU | NVIDIA のみ |
| 料金 | 無料 | 有料（[別途購入](../../README.ja.md#プロジェクトを支援する)） |

LTX は、特に大事な動画やシーンに使ってください。買う前に
SLS Discord のチャンネルで
サンプルを見て、**お試しモード**で PC が LTX を動かせるか、どれくらい時間が
かかるかを確認してください。**修復モデル**セクションで
選ぶか、次のように指定します。

```bash
jasna --input input.mp4 --output output.mkv --restoration-model-name ltx
```

### LTX に必要なもの

- **ライセンス**: LTX を含むアクティベーションキー（25 USD、unet-4x と SD 1.5 も含む、
  [プロジェクトを支援する](../../README.ja.md#プロジェクトを支援する)を参照）。ない場合、アプリでは
  代わりに**お試しモード**を選べます（下記参照）。CLI はエラーで止まります。
- **GPU**: NVIDIA RTX 20 シリーズ以降。GTX 16 シリーズは未検証です。AMD には
  対応していません。
- **VRAM**: 16 GB 以上が最適です。それより少ないカードでも動きますが遅く
  なります。モデルの一部をシステム RAM に置くためです。これまでに試した
  最小のカードは 11 GB です。
- **システム RAM**: 32 GB を推奨します（モデルの読み込みに約 19 GB 使います）。
- **ディスク**: モデルは同梱されていません。初めて LTX のジョブを始めるとき、
  Jasna はダウンロード前に確認します。

  | モデル | ダウンロード |
  | --- | ---: |
  | 標準 | 約 15 GB |
  | 高速モード | 約 9 GB |
  | 高画質（標準に追加） | さらに約 13 GB |

  Hugging Face に接続できない場合（中国など）は、自動で hf-mirror.com を
  使います。

LTX は VR180 サイドバイサイド動画とストリーミングでは使えません。LTX では
セカンダリ復元とノイズ除去はオフになります。

### LTX のオプション

- **モデル**: **標準**（既定）または**高画質**。高画質は少し良く、約 1.5 倍
  遅くなります。CLI: `--restoration-model-name ltx-undistilled`。
- **高速モード**（RTX 50 シリーズのみ）: 約 1.4 倍速く、ディテールはわずかに
  減り、ダウンロードも小さくなります。CLI: `--ltx-fast`。
- **大きなモザイクをより鮮明に**: 大きなモザイク（クローズアップ、4K）を
  より精細にします。その部分は約 3 倍遅くなります。空き VRAM が約 12 GB
  必要です（区間エディターの区間では 16 GB のカード）。足りない場合は通常
  どおり復元します。CLI: `--ltx-large-canvas`。
- **シード**: シードを変えると結果が少し変わり、同じシードなら同じ結果に
  なります。CLI: `--ltx-seed 7`。
- **お試しモード**: モデルやライセンスなしで LTX を実際の速度で動かし、
  お使いの PC での速さを確認できます。動画は復元**されません**。
  CLI: `--ltx-trial`。
- **大きなモザイクだけ LTX**: 大きなモザイクは LTX、それ以外は高速な
  BasicVSR++ で復元します。長い動画では時間を大きく節約できます。
  [動画の一部だけを復元する](segments.md#cli---ltx-auto-split)を参照してください。

LTX をプレビューするには、[区間エディター](segments.md)を開いて区間を選び、
LTX を選んで**復元プレビュー**をオンにします。動画プレーヤーは常に
BasicVSR++ を使います。

### 所要時間

**モザイクが常に映っている 1080p 動画 1 分**あたりのおおよその時間
（標準モデル）です。モザイクのないシーンはずっと速く終わります。高画質は
約 1.5 倍かかります。

| GPU | 時間 | 高速モード |
| --- | ---: | ---: |
| RTX 5090 | 約 5 分 | 約 3 分 |
| RTX 4090 | 約 7 分 | 使えません |
| RTX 5080 | 約 10 分 | 約 7 分 |
| RTX 3090 | 約 15 分 | 使えません |
| RTX 2080 Ti | 約 45 分 | 使えません |

- 初回の LTX は、Jasna がお使いの GPU 向けに準備するため時間がかかります。
- 進行状況は 4 つのステップで表示されます: モザイク検出、準備、復元、保存。
  GPU で最初の LTX 動画（更新後の最初の動画も）は、短い速度テストから始まります。
  その後は、ステップ 1 の終わりから残り時間が正確になります。
- 他のアプリが GPU を使うと LTX は遅くなります。ゲームやブラウザーを閉じて
  ください。
- **Windows**: Windhawk が Jasna 内で動いていると、LTX は「アクセスが拒否
  されました」で失敗します。Windhawk を終了するか、`jasna.exe` を除外
  プロセスに追加してください。起動時に Jasna が警告します。

## セカンダリ復元

Jasna は各モザイクを 256x256 の画像として復元します。大きなモザイク、
クローズアップ、4K 動画では、ぼやけて見えることがあります。セカンダリモデルは
この画像をフレームに戻す前に大きく（512x512 または 1024x1024）するので、
結果がよりシャープになります。

NVIDIA のみです。AMD ではセカンダリ復元は使えません。

- **unet-4x**（有料）: 復元用に訓練されており、品質は TVAI `iris-2` に近く、
  TVAI より高速です。
  SLS Discord の例
  と[プロジェクトを支援する](../../README.ja.md#プロジェクトを支援する)を
  ご覧ください。
- **RTX Super Resolution**（無料）: 非常に高速で、インストールは不要です。
  品質はまずまずです。一部の動画ではフリッカーが出るので、まず短いクリップで
  試してください。
- **TVAI**: unet-4x と同程度ですが、非常に遅いです。
  [Topaz Video](https://www.topazlabs.com/topaz-video) が必要です（有料、
  Windows のみ）。`iris-2` モデルを使ってください。

LTX では使われません。

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

TVAI では、環境変数 `TVAI_MODEL_DATA_DIR` と `TVAI_MODEL_DIR` を
Topaz Video のモデルフォルダーに、下のように設定してください。
`--tvai-args` で他の Topaz の設定を変えられます。

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### 速度と VRAM の比較

1080p 動画 2 本での時間と VRAM:

| セカンダリ種別           | CAWD 1080p        | K109 1080p      |
| ------------------------ | -----------------:| -----------------:|
| セカンダリなし           | 22秒 / 10.0 GB VRAM | 11秒 / 10.7 GB VRAM |
| unet-4x                  | 29秒 / 12.5 GB VRAM | 14秒 / 12.6 GB VRAM |
| RTX Super-Res            | 25秒 / 11.7 GB VRAM | 13秒 / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52秒 / 12.1 GB VRAM | 24秒 / 12.4 GB VRAM |

## 静止画復元（SD 1.5）

画像では、Jasna は別のモデルを使います。訓練済みの Stable Diffusion 1.5
モデルです。キューに画像を追加する（または CLI で渡す）と、Jasna は自動で
このモデルを使います。

```bash
jasna --input photo.png --output restored.png
```

- 有料です。1 つのアクティベーションキーに SD 1.5、unet-4x、LTX を含められます。
  [プロジェクトを支援する](../../README.ja.md#プロジェクトを支援する)を
  ご覧ください。
- ダウンロードには含まれていません。約 **6.9 GB** です。Jasna は
  [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav)
  からダウンロードする前に確認します。
- 約 **7 GB の VRAM** が必要です。大きな 4K 画像ではもう少し必要です。

このモデルは実験的です。結果は画像ごとに大きく変わりますが、とても良く
なるものもあります。いくつか作って、最も良いものを残してください。

```bash
jasna --input photo.png --output restored.png --sd15-variants 4
```

GUI でシードを 1 つずつ試すには、
[画像を対話的に処理](gui.md#画像を-1-枚ずつ処理)を使ってください。すべての
オプション（`--sd15-steps`、`--sd15-strength`、`--sd15-seed` など）は
[CLI リファレンス](cli.md#sd-15-画像復元)にあります。

例:
SLS Discord の SD 1.5 例
とその他の例。
