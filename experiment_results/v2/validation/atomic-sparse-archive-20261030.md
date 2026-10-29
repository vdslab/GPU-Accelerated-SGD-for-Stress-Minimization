# Atomic Sparse診断記録の保存（2026-10-30）

2026-10-27の未追跡の診断コード・実験記録を整理してGitへ保存した。
本番実装・既存の実験結果は変更せず、大規模実験の再実行は行っていない。
今回のRR高速化の測定群とは別の記録である。

## 保存方針

- 診断ソース、Cargo.lock、再現スクリプト、測定JSON/CSV、ログ、ソースsnapshotをGit管理。
- target/はリポジトリ直下の既存ルールにより全階層で除外。
- 座標・頂点対応表と保存済み実行バイナリはローカルに保持し、Gitから除外。
- 各ディレクトリのlocal-artifacts-manifest.jsonに相対パス・サイズ・SHA-256を記録。
  manifest単独で元ファイルは復元できない。過去のGPU実行順序を伴う結果の再評価には元の座標を保管する必要がある。
- ファイルの移動や削除は行っていないため、既存のローカル参照パスは維持。

| 記録 | ローカル成果物数 | 容量 bytes |
|---|---:|---:|
| [atomic-sparse-audit-20261027](atomic-sparse-audit-20261027/) | 18 | 88,856,186 |
| [atomic-sparse-fenced-20261027](atomic-sparse-fenced-20261027/) | 91 | 636,898,275 |
| [atomic-sparse-no-shuffle-20261027](atomic-sparse-no-shuffle-20261027/) | 135 | 16,770,509 |
| [atomic-sparse-no-shuffle-large-20261027](atomic-sparse-no-shuffle-large-20261027/) | 45 | 198,169,085 |
| [atomic-sparse-no-shuffle-xlarge-20261027](atomic-sparse-no-shuffle-xlarge-20261027/) | 45 | 755,077,346 |
| [atomic-sparse-smallworld-order-20261027](atomic-sparse-smallworld-order-20261027/) | 27 | 152,376,112 |

## 診断ツール

[使い方](../../../tools/diagnostics/sparse-audit/README.md)。
2026-10-30に以下を実行して成功した。ログはatomic-sparse-organize-tests-20261030.log。

```sh
cargo test --locked --manifest-path tools/diagnostics/sparse-audit/Cargo.toml -- --test-threads=1
```
