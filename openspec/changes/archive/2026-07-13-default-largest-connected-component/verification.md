## 最大連結成分を既定にする検証結果

### 動作

- CPU版・GPU版は追加フラグなしで、Matrix Market入力の最大連結成分を自動選択する。
- 起動時には常に次の統計を表示する。

```text
Input graph: nodes=N, edges=M, components=C
Largest component used: nodes=k/N (p%), edges=e/M (q%)
```

- 結果prefixごとに`-vertex-map.txt`を生成し、初期・最終TXTの両方から同じmapを参照する。

### bcsstk31

`bcsstk31.mtx`はリポジトリに同梱されていないため、SuiteSparse公式アーカイブから一時取得して検証した。

```sh
cd sparse-sgd-gpu
cargo run --release -- /tmp/bcsstk31.HGRkc8/bcsstk31/bcsstk31.mtx \
  --pivots 200 --iterations 15 --epsilon 0.1 --seed 0
```

| 項目 | 値 |
|---|---:|
| 元頂点数 | 35,588 |
| 元辺数 | 572,914 |
| 連結成分数 | 2 |
| 採用頂点数 | 35,586 |
| 採用辺数 | 572,913 |
| 頂点採用率 | 99.99% |
| GPU compute | 332.409 ms |

CPU版・GPU版の35,586行のvertex mapは一致した。どちらも追加の`--largest-component`指定なしで実行している。

### bcsstm35

```sh
cargo run --release -- ../data/bcsstm35.mtx \
  --pivots 200 --iterations 15 --epsilon 0.1 --seed 0
```

| 項目 | 値 |
|---|---:|
| 元頂点数 | 30,237 |
| 元辺数 | 2,408 |
| 連結成分数 | 29,228 |
| 採用頂点数 | 6 |
| 採用辺数 | 15 |
| 頂点採用率 | 0.02% |
| 辺採用率 | 0.62% |

CPU版・GPU版とも通常起動でこの統計を表示し、GPU出力は`gpu_visualizer`でfull stress `1.15`と2048×2048 PNG生成を確認した。

### USPowerGrid回帰

追加フラグなしのGPU実行で、連結入力の統計は`4,941/4,941`頂点、`6,594/6,594`辺、成分数1となった。`gpu_visualizer` full stressは既存値と同じ`720,695.22`で、2048×2048 PNGを生成した。

### 静的検査

- `baseline-sparse-sgd-non-gpu`: `cargo fmt --check`、`cargo clippy --all-targets -- -D warnings`、`cargo test`（24 tests）成功。
- `sparse-sgd-gpu`: `cargo fmt --check`、`cargo clippy --all-targets -- -D warnings`、`cargo test`（29 tests）成功。
- テストは再番号付け、同率成分、自己ループのみのエラー、連結入力の恒等map、旧`--largest-component`の互換受理、vertex map形式を含む。
