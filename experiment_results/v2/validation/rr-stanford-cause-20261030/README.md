# web-Stanford：RR SparseとCPUのstress差の原因調査

2026-10-30、Apple M4 Pro / Metal、seed=0、P=200、15反復、epsilon=0.1。
対象はブラウザのRR実行03:09:19とCPU実行03:10:02。生のrecordをbrowser-records.jsonに保存。

## 結論

今回のseedでは、RRの更新順序と数値丸めの相互作用で大きな品質差が生じることをCPU上でも再現した。
GPUでの並列競合を仮定しなくても悪化する。単なる「f32は一律に品質が悪い」という説明ではない。
更新順序を固定して初期座標だけをf32へ丸めても、f64演算のRR再生で大きく悪化する。
初期座標の最大差は2.974926638366071e-8。入力の微小差に対して最終配置が敏感である。

今回の高速化前のRRとブラウザRRの最終座標SHAは完全一致。
最適化によって新しく発生した差ではなく、以前からのRR Sparseの品質問題を維持したもの。
以前の「stress不変」は変更前RRとの同等性であり、CPUと同等の品質を意味しない。
本調査では本番の更新式・更新順序は変更していない。

## 入力・評価・再現性

両実行で入力SHA、f64共通初期座標SHA、前処理SHA、頂点・辺・制約数、P、seed、反復数、epsilon、評価始点数・評価seedが一致。
初期SHAはGPU転送前のf64配列なので、転送時のf32丸めまで同じという意味ではない。
GPUは255,265頂点、1,941,926辺、52,973,023制約、38,625 matching rounds。
診断で再実行したGPU・CPUの最終座標SHAは、それぞれブラウザの保存座標と一致。
過去の `rr-common-optimization-20261030/suite-web-Stanford-s0.jsonl` の標準RR座標とも一致。
照合結果はverification.json。

| 評価始点数 | RR / CPU stress |
|---|---:|
| 64、seed 0（ブラウザ） | 1.219870 |
| 256、seed 0 | 1.231939 |
| 4096、seed 20261027 | 1.255305 |

4096の共通始点でCPU=5,043,851,029.634286、RR=6,331,569,034.477455。
paired差=1,287,718,004.843171、近似95%区間=[1,200,406,912.9491706, 1,375,029,096.7371712]。
これは固定した2配置についての始点サンプリング差の区間で、最適化seed間の信頼区間やexact full stressの証明ではない。
64始点のサンプリング誤差だけでは今回の悪化を説明できない。

## 更新順序×数値精度の実験

`order_precision_probe`は診断専用。RRのpivot permutation、頂点ごとのrotation、matching順序、片側/両側の先後を本番と同じ乱数列でCPU上に再現する。
片側更新はpivotのみを読み、各matching内は頂点を共有しないので、このCPU直列化は同じ依存順序を持つ。
f64条件では元のf64重み・eta・初期座標を使う。距離は元から整数なのでf32距離からの復元が正確。
元制約とスケジュールの件数・端点・f32重みをassertで照合する。
通常CPU f64は既存の独立CPU実装をそのまま呼び出す。通常CPU f32は同じ全制約shuffle規則と前処理後RNGから開始する。
CPU f32はMetalのfast math/FMAをbit単位で再現するものではない。

以下は同じ評価始点でのstress。単位は10^9。64と256を混ぜて比を求めないこと。

| 更新順序・演算 | 64始点 | 256始点 | 距離<1e-12の回避分岐回数 |
|---|---:|---:|---:|
| 通常CPU順序 / CPU f64 | 5.092621 | 5.103171 | 未計測 |
| 通常CPU順序 / CPU f64、初期座標のみf32へ丸める | 5.070903 | 5.522952 | 未計測 |
| 通常CPU順序 / CPU f32 | 4.757446 | 5.028527 | 0 |
| RR順序 / CPU f64 | 5.167165 | 5.192271 | 93 |
| RR順序 / CPU f32 | 7.298773 | 7.534045 | 181 |
| RR順序 / GPU f32（本番） | 6.212335 | 6.286796 | 未計測 |
| RR順序 / CPU f64、初期座標のみf32へ丸める | 6.443000 | 6.680305 | 81 |
| RR順序 / CPU f64、重み・etaのみf32へ丸める | 5.915742 | 5.773188 | 92 |
| RR順序 / CPU f64、初期座標・重み・etaをf32へ丸める | 6.212289 | 5.785562 | 80 |

丸める条件でも、その後の更新演算はf64。対照実験では丸めの要素だけを変更した。
初期座標だけの丸めで、256始点評価では通常CPUも元CPU比+8.23%、RRは元RR f64比+28.66%変化した。
通常CPUにも初期摂動への感度はあり、RRだけに生じる現象とまでは主張しない。今回のseedではRRの影響が大きい。
各丸めの影響は非線形で加算できない。最後の行とGPUは64始点では偶然近いが256始点では異なり、同じ配置ではない。
CPU f32直列再生がGPUと異なる最終配置になることも、この系の算術差への感度と整合するが、GPU実装全体の無欠陥を証明するものではない。
近接距離回避分岐はRRでのみ観測した。これが劣化のどの割合を引き起こしたかは今回分離しておらず、原因として断定しない。

## 順序を変える介入

RRは51,011,197本（96.2966%）の片側制約をpivot位置を固定した相で処理し、残り1,961,826本をmatchingで更新する。
CPU通常版は全制約を混ぜてshuffleし、pivot位置の変化と他の更新が交互に入る。
この大きな相分離と非線形更新により、同じ制約を同じ回数処理しても同じ最適化の軌道にはならない。

診断として、片側pivot列と両側round列を分割して交互に処理。初期配置、式、P、反復数、乱数列は維持し、全制約を毎反復1回処理する。

| GPU相の分割数 | 64始点 stress / 10^9 | 256始点 stress / 10^9 |
|---|---:|---:|
| 1（本番） | 6.212335 | 6.286796 |
| 4（診断） | 4.763104 | 4.895465 |
| 16（診断） | 5.626299 | 6.051529 |

このseedでは4分割でCPU以下になるが、分割数を増やせば単調に改善するわけではない。
過去の同じ共通変更の検証では、4分割でEVA seed3が元RR比+7.916%、16分割でluxembourg_osm seed1が+47.051%悪化。
根拠は `../rr-common-optimization-20261030/suite-EVA-s3.jsonl` と `suite-luxembourg_osm-s1.jsonl`。
したがってStanford専用分岐や4分割の無条件採用は行わない。

修正候補はこの順序の数値的な不安定さを抑える共通スケジュール・初期配置精度・近接距離処理などだが、単一seedで良い案を全グラフへ採用してはいけない。
今回はユーザー提示ケースの原因調査であり、全グラフに対する品質改善の実装完了ではない。

## 再現

リポジトリルートから実行。診断再生の所要時間を本番GPU/CPUの性能比較に用いないこと。

```sh
cargo build --release --locked --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --example order_precision_probe
implementations/sparse/sparse-sgd-gpu/target/release/examples/order_precision_probe data/web-Stanford.mtx 0 200 compare
implementations/sparse/sparse-sgd-gpu/target/release/examples/order_precision_probe data/web-Stanford.mtx 0 200 rounding
implementations/sparse/sparse-sgd-gpu/target/release/examples/order_precision_probe data/web-Stanford.mtx 0 200 rounding-init
implementations/sparse/sparse-sgd-gpu/target/release/examples/order_precision_probe data/web-Stanford.mtx 0 200 rounding-coeff
implementations/sparse/sparse-sgd-gpu/target/release/examples/order_precision_probe data/web-Stanford.mtx 0 200 cpu-rounding-init
implementations/sparse/sparse-sgd-gpu/target/release/examples/order_precision_probe data/web-Stanford.mtx 0 200 gpu-phases
cargo test --release --locked --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --example order_precision_probe
```

例のテスト8件（CPU参照7件と非退化更新でのGPU/CPU再生一致1件）通過。clippy warnings deniedも通過。
本番のソースは変更していない。ソース・入力・バイナリSHAはsource-sha256.jsonに保存。

4096始点のpaired評価（元のブラウザ保存ファイルが必要）：

```sh
tools/diagnostics/sparse-audit/target/release/sparse-audit paired data/web-Stanford.mtx output/web-Stanford/2026-10-30-03-10-02_sparse_sgd/2026-10-30-03-10-02_info.txt output/web-Stanford/2026-10-30-03-09-19_rr_sparse_sgd/2026-10-30-03-09-19_info.txt
```
