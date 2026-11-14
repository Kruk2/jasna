> **支援者向けモデルの正規ライセンスコードを入手できるのは、[Ko-fi](https://ko-fi.com/kruk2)、[Unifans](https://app.unifans.io/c/kruk2)、中国では[中国語版 README](README.zh.md#在闲鱼购买) の販売者だけです。**

[English](README.md) | [**日本語**](README.ja.md) | [中文](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna 

Jasna は動画と画像のモザイクを復元します。無料で、シンプルなアプリと
コマンドラインがあり、GPU で動きます。

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />

## 目次

**はじめに**

- [Jasna でできること](#jasna-でできること)
- [必要なもの](#必要なもの)
- [クイックスタート](#クイックスタート)
- [ヘルプとコミュニティ](#ヘルプとコミュニティ)
- [プロジェクトを支援する](#プロジェクトを支援する)（支援者向けモデル）

**上級者向け**

- [Jasna のしくみ](#jasna-のしくみ)
- [コマンドライン](#コマンドライン)
- [すべてのガイド](#すべてのガイド)
- [ベンチマーク](#ベンチマーク)
- [ライセンス](#ライセンス)
- [謝辞](#謝辞)

## Jasna でできること

- **動画のモザイクを復元。** 変わるのはモザイクの部分だけです。[クイックスタート](#クイックスタート)
- **速さか精細さかを選ぶ。** BasicVSR++ は高速で無料、LTX は遅いけれどより精細です（支援者向け）。[モデルの選び方](docs/ja/models.md#復元モデル)
- **LTX を無料で試す。** お試しモードで、お使いの PC での LTX の速さが分かります。[LTX のオプション](docs/ja/models.md#ltx-のオプション)
- **必要な所だけ LTX。** 大きなモザイクは LTX、それ以外は BasicVSR++ で復元します。[モザイクスキャン](docs/ja/segments.md#モザイクスキャン)
- **一部のシーンだけ復元。** 区間エディターでプレビューしながら区間を選べます。[区間エディター](docs/ja/segments.md#区間エディターgui)
- **一時停止と再開。** 長い LTX のジョブも進捗が残ります。[一時停止と再開](docs/ja/models.md#一時停止と再開)
- **復元した部分をよりシャープに。** UNet 4x、RTX Super Res、Topaz。[二次復元](docs/ja/models.md#二次復元)
- **動画全体を拡大**して 1080p、1440p、4K に。[動画全体を拡大](docs/ja/advanced_processing.md#動画全体を拡大)
- **動きを滑らかに。** RTX 40 と 50 でフレームレートを 2 倍にします。[フレーム生成](docs/ja/gui.md#フレーム生成)
- **VR180 動画**（サイドバイサイド、トップボトム）。[VR180](docs/ja/vr180.md)
- **静止画**を SD 1.5 で復元（支援者向け）。[静止画](docs/ja/models.md#静止画復元sd-15)
- **復元しながら見る。** アプリ内でもブラウザーでも。[動画プレーヤー](docs/ja/gui.md#動画プレーヤー)、[ストリーミング](docs/ja/streaming.md)
- **たくさんの動画をまとめて処理。** キュー、プリセット、キュー完了後の動作。[GUI の使い方](docs/ja/gui.md#キュー)
- **出力の追加機能。** カラー LUT、シャープネス、60→30 FPS、処理中でも再生可能。[高度な処理](docs/ja/advanced_processing.md)
- **コマンドラインとレンタル GPU。** [CLI](docs/ja/cli.md)、[クラウド GPU](docs/ja/cloud.md)

## 必要なもの

- **NVIDIA** の GTX 16 シリーズ / RTX 20 シリーズ以降と、新しいドライバー
  （Windows で 610 以上、Linux で 580 以上）。GTX 10 シリーズ以前では動きません。
- または **AMD** Radeon RX 6000 以降（実験的）。検証済みは RX 7000/9000 で、
  RX 6000 は未検証です。
- ローカルディスク上のフォルダー。パスに日本語、中国語、アクセント付きの文字
  など、英語以外の文字を含まないこと。

<details>
<summary>詳細</summary>

- NVIDIA: NVIDIA の [GPU 一覧](https://developer.nvidia.com/cuda/gpus)で
  コンピュート能力が 7.5 以上である必要があります。A100、H100、B200 などの
  データセンター向け GPU では動きません。
- AMD: 下の表をご覧ください。UNet 4x の二次復元に対応しています。
- ディスク形式: Windows では NTFS、Linux では ext4、Btrfs、XFS、ZFS。ネットワーク
  ディスクは非対応です。SSD を使ってください。ハードディスク（HDD）では起動と
  モデルの読み込みが遅くなります。
- LTX には NVIDIA RTX または AMD Radeon RX 7000/9000 GPU（AMD 内蔵 GPU は未検証）、8 GB 以上の VRAM、16 GB 以上の RAM が必要です。
  [LTX に必要なもの](docs/ja/models.md#ltx-に必要なもの)をご覧ください。

| AMD GPU | Jasna | LTX |
| --- | --- | --- |
| Radeon RX 9060/9070、Radeon AI PRO R9700 | 対応 | 対応（Windows はベータ版） |
| Radeon RX 7600–7900、Radeon PRO W7000 | 対応 | 対応（Windows はベータ版） |
| Radeon 740M/760M/780M/880M/890M、Ryzen AI Max（8050S/8060S） | 対応（未検証） | 対応（未検証） |
| Radeon RX 6000、Radeon PRO W6000 | 対応（未検証） | 非対応 |
| Radeon 610M/660M/680M/820M/840M/860M | 対応（未検証） | 非対応 |
| Radeon RX 5000、Radeon PRO W5000、RX Vega、RX 500 以前 | 非対応 | 非対応 |

</details>

## クイックスタート

1. お使いのシステムと GPU に合うリリースをダウンロードします。
2. パスに日本語、中国語、アクセント付きの文字など、英語以外の文字を含まない
   フォルダーへ展開します。
3. アプリを起動します:
   - Windows: `jasna.exe` をダブルクリックします。AMD では先に
     [ROCm 10.1 用の AMD ドライバー](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html)
     （**26.10.41.05**、[.exe をダウンロード](https://drivers.amd.com/drivers/amd-software-adrenalin-edition-26.10.41.05-win11-b-204865.exe)）
     をインストールし、PC を再起動してください。ROCm のインストールは不要です。
   - Linux NVIDIA: `jasna` ファイルを実行します。
   - Linux AMD: 最初に一度だけ AMD の AMF 動画ランタイムをインストールします。
     [AMF 26.20](https://repo.radeon.com/amf/26.20/) で動作確認しています:

     ```bash
     curl -fLO https://repo.radeon.com/amf/26.20/amf_installer_26.20.zip
     unzip amf_installer_26.20.zip
     sudo bash amf_installer_26.20.sh
     ```

     求められたら AMD のライセンスに同意し、`run_jasna_amd.sh` を実行します。
     Distrobox を使う場合は、Jasna と同じコンテナー内に AMF をインストールしてください。
4. 動画を追加し、設定を選んで、画面下の **▶** ボタンを押します。

**最初のジョブは時間がかかります。** NVIDIA では、Jasna がまずお使いの GPU
専用にモデルを準備します。約 **15～60 分**かかりますが、一度だけです。この間は
他のアプリを閉じ、PC を使わないでください。Jasna を更新するときは、
`model_weights` フォルダーを新しいバージョンへコピーしてください。もう一度待たずに済みます。
更新後、LTX はモデルの再ダウンロードを求めることがあります。AMD では、
この準備が必要なのは UNet 4x を初めて使うときだけです。

すべての設定にはツールチップがあります: ⓘ アイコンにマウスを合わせてください。
GPU メモリ（VRAM）が足りない場合は、**最大クリップサイズ**を下げてください。例: `90` から `60`。
[VRAM と GPU 使用量の調整](docs/ja/tuning.md)をご覧ください。

## ヘルプとコミュニティ

- [アプリガイド](docs/ja/gui.md): キュー、プリセット、出力名、動画プレーヤー、
  有効化。
- 問題がある場合: アプリ下部の**ログ**を押し、**エクスポート**でログファイルを
  保存します。質問するときに添付してください。

## プロジェクトを支援する

Jasna は無料です。支援は新しいモデルの訓練に使われます。主に GPU レンタル代
です。支援者には支援者向けモデルのライセンスコードが届きます:

| 支援額 | モデル |
| --- | --- |
| **合計 15 USD**（Ko-fi は **14 EUR**）以上 | **UNet 4x**: 復元した部分をよりシャープにします。<br>**SD 1.5**: 静止画用の実験的なモデル。 |
| **25 USD**（Ko-fi は **23 EUR**） | **UNet 4x** と **SD 1.5**、さらに<br>**LTX**: BasicVSR++ よりずっと遅いですが、より細かく復元します。この LTX モデルの今後のアップデートも含みます。 |

**2026 年 10 月 14 日までに 15 USD 以上**支援した方は、LTX を無料で使えます。

**LTX を選ぶ前に:**

- PC で LTX を**お試しモード（速度テスト）**で動かしてください。無料で、お使いの GPU で
  LTX が動くか、どれくらい時間がかかるかが分かります。動画は復元されません。
  [LTX に必要なもの](docs/ja/models.md#ltx-に必要なもの)をご覧ください。

### 支援できる場所

ライセンスコードは次の場所で**のみ**入手できます:

- [Ko-fi](https://ko-fi.com/kruk2)
- [Unifans](https://app.unifans.io/c/kruk2)
- 中国の方: [中国語版 README](README.zh.md#在闲鱼购买) をご覧ください

それ以外で Jasna、コード、改変ファイルを販売している人は詐欺です。そのキーは
使えず、そのファイルは PC 上で隠れたコードを実行する可能性があります。Jasna は
GitHub からのみダウンロードしてください。

### 有効化の手順

1. 上のどれかで支援します。**ライセンスコード**が届きます。
2. Jasna で **Jasna を有効化**を開き、ライセンスコードを入力し、
   **自分のPC**か**レンタル・共用のPC**を選んで**有効化**を押します。

1 つのライセンスコードは**同時に 3 台**まで使えます。TPM 2.0 セキュリティチップの
ある自分の PC（多くの Windows 11 PC）では、その後オフラインで動きます。それ以外では、
支援者向けモデルを使う間インターネットが必要です。詳しい説明とトラブル対応:
[支援者向けモデルの有効化](docs/ja/gui.md#支援者向けモデルの有効化)。レンタル GPU:
[クラウド GPU](docs/ja/cloud.md) をご覧ください。

## 上級者向け

### Jasna のしくみ

Jasna は各モザイクを見つけ、フレームごとに追いかけ、その部分だけを復元して元に
戻します。画面のほかの部分には手を加えません。ほぼすべて GPU で動きます。LTX は
遅いですが、より本物らしい細部を加えられます。[モデルの選び方](docs/ja/models.md#復元モデル)をご覧ください。

### コマンドライン

コマンドラインでは、アプリでできることのほとんどができます（動画プレーヤー、
LTX のシードのプレビュー、マスクの提案、対話型画像復元は使えません）。

```bash
# 動画 1 本
jasna --input input.mp4 --output output.mkv

# 画像 1 枚
jasna --input photo.png --output restored.png

# フォルダー全体
jasna --input input_folder --output output_folder
```

`jasna --help` で全オプションを表示できます。[CLI リファレンス](docs/ja/cli.md)を
ご覧ください。

### すべてのガイド

- [GUI の使い方](docs/ja/gui.md): キュー、プリセット、出力名、動画プレーヤー、
  有効化。
- [モデルの選び方](docs/ja/models.md): 検出モデル、BasicVSR++ か LTX か、
  二次復元、画像用の SD 1.5。
- [動画の一部だけを復元する](docs/ja/segments.md): 区間エディター、
  モザイクスキャン、大きなモザイクだけ LTX。
- [VR180 動画](docs/ja/vr180.md): サイドバイサイドとトップボトム VR の扱い方。
- [VRAM と GPU 使用量の調整](docs/ja/tuning.md): クリップサイズ、
  オーバーラップ、VRAM が足りないときの対処。
- [高度な処理](docs/ja/advanced_processing.md): ノイズ除去、60→30 FPS、
  カラー LUT、シャープネス、動画全体の拡大、エンコーダー品質、動画に保存される設定、
  キュー完了後のアクション。
- [ストリーミング](docs/ja/streaming.md): ブラウザーや Stash で見る。
- [クラウド GPU](docs/ja/cloud.md): レンタル GPU（RunPod、Vast）で Jasna を使う。
- [CLI リファレンス](docs/ja/cli.md): すべてのコマンドラインオプション。
- [ソースから実行](docs/en/development.md): 開発者向け（英語）。ソースからは
  無料の BasicVSR++ を使えます。有効化と支援者向けモデルには公式リリースが必要です。

### ベンチマーク

RTX 5090（Linux）で、Jasna 0.9.1 は 3:22 の 1080p 動画を BasicVSR++ で 34 秒で
復元しました。テスト設定、GPU メモリ、以前のバージョンは
[完全なベンチマークレポート](benchmarks/2026-07-26_release_matrix.md)に、新しい結果は
[benchmarks](benchmarks/) にあります。

### ライセンス

Jasna の公開アプリケーションソースは AGPL-3.0 です。モデルと同梱依存関係には、
それぞれのライセンスが適用されます。[ライセンスとソース公開範囲](LICENSING.md)、
[モデルの通知](assets/THIRD_PARTY_MODELS.md)、
[リリースソース対応表](RELEASE_SOURCES.md)をご覧ください。

### 謝辞

- **Lada**: Jasna は Lada に
  着想を得ており、一部は Lada をベースにしています。Jasna で使っている
  `mosaic_restoration_1.2` モデルは、Lada 作者の ladaapp によって訓練されました。
- **ZeLeFans**: VR 検出モデルと、モザイクの形状および VR 投影に関する詳細な
  分析に感謝します。
