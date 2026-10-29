# RR Sparse SGD: 100万ノード実走と改善候補（2026-10-29）

## 対象・条件

- ソース: e4d6e7ccd96e08c52e63f3ab33a612e4233ecccd。既存未追跡ファイルあり。実装ソースは変更せず `cargo build --release` で確認。
- Apple M4 Pro / 20 GPU cores / 24 GiB unified memory。
- data/delaunay_n20.mtx: 1,048,576 vertices, 3,145,686 undirected edges, connected。
- seed=0, 15 iterations, epsilon=0.1。pivot=16,64を各1回。200 pivotと他のグラフは今回未実走。
- JSONのiteration_time_msは15反復の合計のホスト計測値。GPU timestampではない。
- 成功時には実装内のNaN/Inf・末尾canary検査を通る。ただし競合がないことや品質の証明ではない。
- sampled full stressは固定64始点からのBFS推定。各pivot条件は乱数消費が異なり、seedが同じでも初期座標が異なる。品質比較の確定には共通初期座標・複数seed・CPU基準が必要。

## 再現方法

リポジトリルートで以下を実行。16を64に置き換えると第2条件。

```sh
cargo build --release --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml
/usr/bin/time -l implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu data/delaunay_n20.mtx --pivots 16 --iterations 15 --epsilon 0.1 --seed 0 --run-id rr-million-h16-s0 --output-format json --verbose --output-dir experiment_results/v2/validation/rr-million-audit-20261029/artifacts
```

生データ: h16.json / h64.json、時間・最大RSS: h16.stderr / h64.stderr。配置ファイルはartifacts/。

## 200 pivotでの見積もり

以下はn=1,000,000、h=200、十進GB。実測ピークではなく配列のサイズ見積もり。

- pivot距離表: usize × n × h = 1.6 GB。
- CPU EdgeInfo: 40 bytes × 約nh = 約8 GB（実辺などの分を追加）。
- BTreeMapのkeyとvalueだけで約56 bytes × nh = 約11.2 GB。木の空き領域・ポインタ・アロケータ分は別。
- one-sided配列: 16 bytes × nh = 3.2 GB。CPU配列に加えてGPUバッファを生成する。
- 座標f32x2そのものは8 MB。

距離表・木構造・CPU制約・GPU配列は寿命が異なるので上記全部を単純合計してピークとはしない。共有メモリでもCPU配列とGPUバッファの別確保はある。現行経路はメモリ圧迫の可能性が高く、200 pivotの完走を今回の結果から保証できない。

limits.rsで同じwgpu 27依存を使い実機照会した結果はlimits.txtに保存。storage binding上限=4,294,967,295 bytes、buffer上限=12,884,901,888 bytes。よって100万×200の3.2 GBはこの実機の単一binding上限内。『WebGPU既定値128 MiBなので動かない』という結論にはならない（実装がadapter.limits()を要求）。現行16-byte表の単一binding限界はh=200で約134万頂点。一般化にはチャンク化が必要。

## 改善候補（優先順）

1. **正しさ: one-sided相のpivot書き戻しを止める。** shader.wgsl:58-81は全頂点をdispatchし、pivot行も末尾でpositions[vertex]を書き戻す。一方、他invocationはpositions[entry.pivot]を読む。同じ値を書き戻しても非atomic read/writeの競合になる。pivot maskで早期returnするか、非pivot頂点ID列だけをdispatchする。非pivot座標が更新された場合だけstoreする方法も検討。既存の有限値/canaryテストでは検出できない。WGSLメモリモデル: https://www.w3.org/TR/WGSL/ 。品質悪化の原因と断定する証拠はない。

2. **I/O: BufWriterと明示flush、出力時間の記録。** main.rs:320-390のsave_result/save_vertex_mapはFileへwriteln!を直接実行。100万頂点で初期・最終の両方に全辺も出力するため大量のwriteが発生。保存する辺を共通ファイルに分ける・初期座標保存を任意にする案もある。JSONのcli_total_time_cold_msは入力+アルゴリズムの和で、stress評価・出力等を含む実プロセス時間ではない。エンドツーエンド計測の追加が必要。

3. **CPU前処理: 全制約BTreeMapを作らずRR形式に直接出力。** graph.rs:364-450はO(nh)件を木に挿入し、さらにEdgeInfo Vecにしてschedule.rsでGPU形式へ変換。one-sidedは(vertex,pivot-slot)で直接配置し、pivot間は小さな三角表、実辺は別列にする。実辺の上書き・方向別weight・pivot対の統合・学習率min/max・fingerprintの同等性を小グラフで照合する。CPU基準経路を残してRR専用前処理を設けるのが安全。

4. **メモリ: one-sided 16→8 bytes。** pivot IDは列から共通pivot-ID配列を引けばよく、各セルにはdijとweightのf32だけを置ける。paddingも不要。n=100万,h=200でCPU表/GPU表は各3.2→1.6 GB。疎という名前だけでCSRにするより、ほぼnh個あるpivot制約の冗長フィールド削除が先。必要なら頂点チャンクで単一binding上限にも対応。

5. **距離表・BFS: usize→u32、CSR隣接、バッファ再利用。** 距離表は1.6→0.8 GB。到達不能sentinelと範囲検査を保つ。pivot選択は次のpivotが直前までの距離に依存するので、200本を単純に独立並列化して同じ選択結果を保つことはできない。BFS内部の並列化、または保存量を抑えて後で再BFSする方式を比較。regionの最短pivotラベルをpivot選択時に更新すればassign_regionsの後走査を減らせる（同距離tieを維持）。

6. **不要な保持・cloneを削減。** gpu.rs:119-123でtwo_sidedを丸ごとcloneしている。空配列の場合だけdummyを用意して通常は借用する。schedule完成後のparams.pairsはGPU計算で使わないため、GPU確保より前に明示的に解放する。メモリ寿命を明確にする。効果はRSS/footprintで確認。

7. **ハブ対策: matchingだけでは最大次数が下限になる。** 正確に各辺の両端を1回ずつ更新するmatching方式は少なくともΔ roundsが必要。既存summary.csvのweb-NotreDame（325,729頂点）ではmedian 10,721 rounds、15反復6.27秒、CPUに対するstress比約1.244。小さいroundのdispatch overheadと品質を調べる。round内で独立な辺の融合は可能だが、共有頂点を持つround同士を無同期でまとめるのは不可。ハブ別の逐次owner処理やsnapshot/reductionは演算順序・手法を変えるため別variantとして品質評価する。

8. **GPUメモリアクセスと待機。** one-sidedはvertex-major表に加えvertexごとにpivot順をrotationするため、隣接laneの読出しが揃いにくい。pivot-major/AoSoAと共通順序のvariant、pivot座標の小さなread-only配列を比較する。転置だけでrotationの不規則性が消えるわけではない。gpu.rsは最大256 passごとにsubmit+CPU waitを行う。単純撤去はせず、Metal実行制限・メモリ・順序保証を保ちながら複数batchのin-flight化を測る。

9. **品質/収束: pivot数・初期配置・反復数を分けて検討。** h=16/32/64/128/200を共通初期座標と複数seedで比較し、同じ品質に達する総時間を見る。multilevel初期配置やcoarse-to-fine、段階的pivot追加は有望な実験案だが今回未検証。目的関数/学習率変更を伴う案は現行RRと区別する。

10. **評価と観測: phase別GPU timestampと保存時間。** 現状gpu_device_time_ms=nullでone/two-sidedの分離測定もない。GPU timestamp対応確認後に測定する。64始点BFSのstress評価も大規模化すると無視できない。最適化試験では評価を分離・結果を再利用し、最終比較では共通サンプルを保つ。

## 次の検証順

pivotの書込み修正 → 出力buffering/時間計測 → CPU制約の保持削減 → RR専用直接前処理と8-byte表 → 200 pivotで100万ノード再測定 → ハブの多いグラフ・複数seed・共通初期座標で品質確認。

本調査では実装の変更や速度改善を適用していない。成功した単一seedのDelaunayグラフを、任意の100万ノードグラフでの速度・品質保証には拡張しない。

## 実測結果

| pivots | 前処理 s | schedule構築 s | 15反復合計 s | アルゴリズム計 s | stress評価 s | プロセス実時間 s | 最大RSS GiB | 推定stress |
|---|---|---|---|---|---|---|---|---|
| 16 | 2.431 | 0.808 | 0.181 | 3.538 | 1.911 | 48.420 | 1.953 | 6.32578e+10 |
| 64 | 10.570 | 1.693 | 0.891 | 13.416 | 1.899 | 62.060 | 4.862 | 5.34045e+10 |

各条件1回。最大RSSはプロセス指標でGPUを含む全システム使用量とは異なる。stressは条件間で初期配置が異なるため、値の大小だけでpivot数の優劣は断定しない。実時間とJSON内のCLI時間の大きな差を区別する。
