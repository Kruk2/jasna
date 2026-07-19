# ストリーミング

ストリーミングを使うと、ファイル全体を先に処理せずに、復元された動画を
リアルタイムで視聴できます。シークに対応しています。

## ブラウザプレーヤー

ストリーミングモードは現在 CLI のみです。ブラウザでプレーヤーが
開きます — 動画ファイルを選んで視聴を開始してください:

```bash
jasna --stream
```

Windows では、ストリーミングもアプリ本体と同じファイルを使います:
`jasna.exe --stream`。

便利なオプション: `--stream-port`（デフォルト `8765`）と、プレーヤーを
自分で開きたい場合の `--no-browser`。

## Stash 連携

Jasna はカスタム Stash フォークを通じて
[Stash](https://github.com/stashapp/stash) の中から使えます。シーンを
再生すると Stash が Jasna を自動起動し、視聴しながら処理します。

カスタムフォーク:
**[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

セットアップ:

1. 上のリンクから Stash フォークをダウンロードします。
2. Stash を起動する前に環境変数を設定します:
   - `JASNA_CLI_PATH`: `jasna.exe` のフルパス（リネームした場合はその名前）。
   - `JASNA_WORKING_DIR`: その実行ファイルがあるフォルダのフルパス。
3. **重要:** Stash を使う前に、Stash で使う予定と同じ設定で、短い動画を
   一度ストリーミングしてください。GPU 固有の検出キャッシュが準備され、
   初回のヘルスチェックタイムアウトを避けられます。
4. Stash を起動してシーンを再生します。

Stash のログに `timeout waiting for jasna-cli to become healthy` と出る
場合は、まず `JASNA_CLI_PATH` を確認し、その後で上記の方法で事前
コンパイルしてください。
