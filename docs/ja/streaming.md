# ストリーミング

ストリーミングを使うと、Jasna が復元している途中の動画を見られます。
ファイル全体を待つ必要はありません。シークもできます。

## ブラウザープレーヤー

コマンドラインからストリーミングを始めます。Jasna がブラウザーでプレーヤーを
開きます。そこで動画ファイルを選んで再生してください。

```bash
jasna --stream
```

Windows ではアプリ本体と同じファイルを使います: `jasna.exe --stream`。

便利なオプション: `--stream-port`（デフォルト `8765`）と、プレーヤーを自分で
開きたい場合の `--no-browser`。

ストリーミングは常に BasicVSR++（LTX ではありません）と H.264 を使います。
アプリの中で見たい場合は、[動画プレーヤー](gui.md#動画プレーヤー)を使って
ください。

## Stash

専用の Stash を使うと、[Stash](https://github.com/stashapp/stash) の中で
Jasna を使えます。シーンを再生すると Stash が Jasna を起動し、見ている間に
動画が復元されます。

ダウンロード: **[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

セットアップ:

1. この Stash をダウンロードします。
2. Stash を起動する前に、次の環境変数を設定します:
   - `JASNA_CLI_PATH`: `jasna.exe` のフルパス。
   - `JASNA_WORKING_DIR`: `jasna.exe` があるフォルダーのフルパス。
3. **重要:** Stash で使うのと同じ設定で、短い動画を一度ストリーミングして
   ください。初回はお使いの GPU 用のファイルを準備するので時間がかかります。
   これを省くと、Stash が待ちきれずに止まります。
4. Stash を起動してシーンを再生します。

Stash のログに `timeout waiting for jasna-cli to become healthy` と出る場合は、
まず `JASNA_CLI_PATH` を確認してください。その後で手順 3 を行ってください。
