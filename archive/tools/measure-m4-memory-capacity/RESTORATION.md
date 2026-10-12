# M4メモリ容量測定ツールのソース回収

このツールは過去のメモリ容量調査を保全するためのアーカイブです。論文の6手法の主比較には使いません。

- 回収元commit: `b197786118fea8bfaa980d326cca2d6b471f695f`
- 回収元ディレクトリ: `measure-m4-memory-capacity/`
- 回収したファイル: `Cargo.toml`、`Cargo.lock`、`src/main.rs`
- 回収日: 2026-10-13（Asia/Tokyo）
- 上記3ファイルは回収元の内容を変更せず保存しています。

## 既存の実行ファイル

移動前から存在していた `target/release/measure-m4-memory-capacity`（macOS arm64）を保持しています。

SHA-256:

```text
ca0bbab92ade380095292021d979afc56e72e690aea5166476ec3d03b7dc11b9
```

この実行ファイルのビルド元commitは確認できていません。回収したソースとの一致や動作は保証していないため、研究の再現実行ではソースから再ビルドしてください。今回の整理では再ビルドと容量測定は行っていません。

## 再ビルド

リポジトリのルートから実行します。

```sh
cargo build --locked --release --manifest-path archive/tools/measure-m4-memory-capacity/Cargo.toml
```

生成先は `archive/tools/measure-m4-memory-capacity/target/release/measure-m4-memory-capacity` です。再ビルドすると保持中の実行ファイルは置き換わります。ビルドに必要な依存はCargo経由で取得し、リポジトリ内の別crateへのパス依存はありません。

## 出力パス

歴史的ソースの既定出力は `CARGO_MANIFEST_DIR/../output/m4-memory-capacity` です。移動後の再ビルドでは `archive/tools/output/m4-memory-capacity` を指すため、将来再測定する場合は `--output-dir` にリポジトリ直下の `output/m4-memory-capacity` を明示してください。主比較の実験結果へは混ぜません。
