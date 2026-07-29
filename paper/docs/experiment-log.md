# 実験台帳

実験は成功・失敗を問わず 1 行ずつ登録します。生ログは上書きせず、論文に採用した集計データだけを `tables/` に保存します。

| ID | 日時 | commit | 実装 | データ | seed / 反復 | 環境 | コマンドまたはログ | 結果 | 採否 |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 2026-07-28 | `14ead51` + dirty | 5手法の共通実験出力 | USPowerGrid | seed 0, 1, 2 / 15 iterations | Apple M4 Pro / Metal | `output/experiments/e0-uspowergrid/` | 15/15成功、開始条件を確認 | 計測系検証のため速度値は不採用 |

## E0: 共通実験基盤の開始条件確認

- 目的: 5手法の共通JSON契約、自動実行、再開、比較可能性を実データで検証する
- commit: `14ead51da42e617a297f88f6a90eb204d083e302` + 未コミットの実装変更
- 環境: Apple M4 Pro、Metal
- コマンド: `python3 experiments/run_experiments.py experiments/manifests/e0-uspowergrid.json --output-root output/experiments`
- 入力: `data/USpowerGrid.mtx`、SHA-256 `8fda7edfc1844d73d4f5be7d9a49963104349292aae64e73782c1a55c3b5f05f`
- seed / 反復: seed 0, 1, 2、各15 iterations、warm-up 1回、正式計測15条件
- 出力ログ: `output/experiments/e0-uspowergrid/results.jsonl`
- 結果: 最新recordは15/15成功。各seedでFull 3手法の初期座標hashが一致し、Sparse 2手法の初期座標hashと前処理hashが一致した。時間内訳の合計誤差、NaN/Inf stress、成果物欠損、atomic retry失敗はいずれも0件
- 異常・失敗: 初回のRR-Sparse 3条件は、各compute passへtimestamp queryを2個割り当ててQuerySet上限4096を超えた。各iterationの先頭・末尾だけを計測するよう修正し、`--resume`で失敗3条件だけを再実行して成功した
- 解釈: 共通フォーマットとrunnerは本実験へ進める状態。E0はdirty worktree上の計測系検証なので、記録された速度値は論文の性能比較には使わない
- 次のアクション: 実装をcommitした後、runbookに従って本実験用manifestを確定する
- 論文への採用: 不採用（実験基盤の検証記録のみ）

## 実験メモのテンプレート

### E___: 実験名

- 目的:
- 仮説・主張 ID:
- commit:
- 環境:
- コマンド:
- 入力:
- seed / 反復:
- 出力ログ:
- 結果:
- 異常・失敗:
- 解釈:
- 次のアクション:
- 論文への採用:
