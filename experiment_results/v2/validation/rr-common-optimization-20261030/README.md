# RR Sparse共通高速化・品質検証（2026-10-30）

## 結論と未解決事項

**全グラフ共通の高速化は実装・検証済み。従来RRのCPUに対する品質劣化は未解決。**
グラフ名・種類・次数による分岐や特別な手法は追加していない。標準CLIの更新順序・pivot数・反復数・epsilonは維持した。
8グラフ×3 seedの24条件で、初期座標・前処理・最終座標のSHAとsampled stressが変更前のRRと完全一致した。
これはCPU並みの品質を保証しない。参考としてCPUの1.05倍以下を要求した独立のquality gateは、比較可能な18条件のうち14条件で失敗する。失敗を成功扱いしていない。

## v3の確認と以前の説明の訂正

以前の「web-NotreDameで約24%悪化」はv2集計であり、v3の値ではない。
v3 basic/P200で対応するCPU成功seedは0,1,3,4の4件。RRのCPU比は次のとおり。

| seed | RR / CPU stress |
|---|---:|
| 0 | 1.082225 |
| 1 | 1.157143 |
| 3 | 1.442364 |
| 4 | 1.013275 |

中央値は1.119684（約12%悪化）。RR自体は5 seed成功だがCPUとの対応比較は4件。
round数は10,721。v3の大規模stress評価は256始点、v2/mainの既定評価は64始点なので混同しない。
原表の場所・SHA・抽出行は `v3-notredame.json` に保存した。

seed 3をv3と同じ入力・P200・15反復で再実行したところ、初期座標/前処理SHAが一致し、
修正後GPU stress=10,071,990,543.659239、独立CPU再実行=6,982,973,940.667906で、いずれもv3と同値。
pivot同値書き戻しの競合修正だけではこの品質劣化は消えなかった。

## 採用した共通変更

1. 片側表を16→8 bytesに圧縮。pivot IDとpaddingを各セルから除き、共通pivot ID列を参照。
2. pivot行には片側更新がないため、その行をstoreしない。従来のpivot read/write競合を除去。
3. u32 BFS距離・再利用queue・pivot選択時のregion更新。距離同率は以前のpivotを維持。
4. 辞書順の制約を行ごとのmergeで直接生成。巨大なBTreeMapと全EdgeInfo Vecを作らない。
   CPU参照版は変更せず、f64の制約値・順序・学習率・初期配置・乱数状態・fingerprintの同等性を試験。
   両側制約だけを保持し、元と同じfirst-fit matching順序を作る。
5. two-sidedのcloneを除去。GPU転送前に不要なCPU制約を解放。初期座標もcloneせず所有権移動。
6. 最大256 pass/command bufferを保ちつつqueueへ順にsubmitし、完了待ちを反復末尾へ移動。
   次反復のuniform/permutation変更前には必ず待機する。uniformのCPU配列も反復間で再利用。
7. 出力を1 MiB BufWriterと明示flushへ変更。既存のテキスト形式は維持。
8. GPU storage buffer容量とdispatch上限を明示チェックし、範囲超過をエラーにする。

CSR化、GPU timestamp計測、multilevel初期配置、pivot削減などは今回の採用変更ではない。
特に初期配置やpivot削減は問題設定・品質を変え得るため、「品質不変の高速化」として混ぜていない。

## 100万ノード実測

Apple M4 Pro / 24 GiB、delaunay_n20、1,048,576頂点、3,145,686辺、seed=0、15反復、epsilon=0.1。
時間は各1回の観測であり、統計的な性能保証ではない。GPU時間はtimestampでなくホスト計測。

| 条件 | SGD 15反復合計 | 前処理等を含むalgorithm cold | 評価・保存込み実時間 | 最大プロセスRSS |
|---|---:|---:|---:|---:|
| 変更前 P64（10/29保存結果） | 0.891秒 | 13.416秒 | 62.06秒 | 4.862 GiB |
| 変更後 P64 | 0.389秒 | 8.030秒 | 12.15秒 | 1.874 GiB |
| 変更後 P200 | 1.438秒 | 24.749秒 | 29.06秒 | 3.704 GiB |

P64では前処理SHA・初期座標SHA・stressが完全一致。
最終ファイルの`# Positions`以降のテキストSHAも一致：
`87073f32574458b7a653cd89b5e628d5a9a3bd46d0474c72ed8b4ec57eba7aaa`。
P200には同規模の変更前実走を今回追加していないため、P200の速度比は算出しない。
P64/P200間は初期配置と制約が変わるため、stressの大小だけをpivot数の純粋な効果とは呼ばない。
最大RSSはGPUを含む全システム使用量ではない。`time -l`のpeak footprintもstderrに保持。
JSONのcli_total_time_cold_msはinput+algorithmであり、評価/保存込みの実時間ではない。

実行例（リポジトリルート）：

```sh
cargo build --release --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml
/usr/bin/time -l implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu data/delaunay_n20.mtx --pivots 200 --iterations 15 --seed 0 --output-format json --run-id million-optimized-h200 --output-dir experiment_results/v2/validation/rr-common-optimization-20261030/artifacts
```

## 品質・同期の回帰検証

対象：EVA、USpowerGrid、coAuthorsCiteseer、luxembourg_osm、preferentialAttachment、smallworld、web-NotreDame、web-Stanford。
各seed=0,1,3、P200、15反復。全24条件で次の4点が完全一致した。

- preprocess SHA
- initial-position SHA
- final-position SHA
- 同じ256始点のsampled full stress

`suite-*.jsonl`のchunks=1は安全なデータ圧縮・直接前処理を適用し、従来のbatchごとのwaitを保った対照。
`optimized-*.jsonl`はwait削減・走査改善後。対照のpreprocess/initial SHAはv3とも照合済み。
さらにv3に保存されたgzip座標ファイルを実際に展開し、24条件すべての最終座標SHAが高速化版と一致することも確認した（`v3-final-coordinate-equivalence.json`）。
最終座標を変えずにwaitを削減できている。web-NotreDameではこの同一セッション対照に対しSGD区間が約1.7〜1.8倍高速。
全グラフで常に同じ倍率になるとは主張しない。小さい時間には実行時の変動もある。

`quality_probe`は小グラフも256始点sampledで評価する。v3のsmall exact評価とは直接比較せず、
CPU gateは同じ初期配置・前処理SHAと256始点sampled条件が一致した18件のみを対象にした。
CPU +5%は今回の診断用の基準で、ユーザー合意済みの品質許容値ではない。

```sh
cargo run --release --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --example quality_probe -- /Users/kose/master/atomic-scheduling-investigation/data/v3_lcc/web-NotreDame.mtx 200 3 cpu 1
python3 tools/diagnostics/rr-common/compare_quality.py experiment_results/v2/validation/rr-common-optimization-20261030 --v3-runs /Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/tables/runs.csv
# 次は既知のCPU品質問題を検出してexit 1となる。通ったことにはしない。
python3 tools/diagnostics/rr-common/compare_quality.py experiment_results/v2/validation/rr-common-optimization-20261030 --v3-runs /Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/tables/runs.csv --max-cpu-ratio 1.05
```

`exact-regression.json`は失敗0件、`cpu-quality-gate.json`はCPU比超過14件を保存。
Rustの35テスト、clippy --all-targets -D warnings、git diff --checkは成功。
RustテストはCPU参照との72構成の制約同等性検査（ランダムグラフ64＋path/star 8）、GPU実走、CLI JSON契約等を含む。

## 品質改善の対照実験（標準には不採用）

各反復で全制約を1回ずつ処理することを保ち、片側相を4/16分割して両側roundと交互にする案を、同じ8グラフ×3 seedで比較。
全72条件をsuiteファイル、比を`order-experiments.json`に保存。

- web-NotreDame seed 3: 16分割でCPU比1.442→約1.068まで改善。
- しかしluxembourg_osm seed 1: 同じ16分割で元のRRからstressが約47%悪化。
- 4分割でもEVA seed 3で元のRRから約7.9%悪化、web-Stanford seed 3で約2.4%悪化。

さらに頂点ごとに互いに素なstrideでpivot巡回順を変える案、1 pivotずつの200分割も診断した。
200分割ではweb-NotreDame seed 3のCPU比が約1.056に縮むが、通常の巡回ではluxembourg_osm seed 1が大きく悪化。
strideを使うとこの道路グラフの悪化は回避できた一方、別の品質差が残る。
有限の試行から全グラフで品質を維持すると判断できないため、いずれも標準経路では無効。
診断用API/exampleとして再現可能にしているが、グラフに応じた自動切替は実装していない。

片側/両側のまとまり方が品質に影響することは対照実験で確認できた。
ただしCPUとの差全体をその原因だけへ帰属する証明ではなく、CPU/GPUのf64/f32や微小距離処理等の差も残る。
**今回の高速化を、CPUに対する品質問題まで修正済みと解釈してはいけない。**

## 証拠

- `million-h64.json`, `million-h200.json`, 各stderr: 実測・メモリ
- `notredame-s3-safe.jsonl`: v3同値のGPU/独立CPU再実行
- `suite-*.jsonl`, `optimized-*.jsonl`: 24条件の同値性と順序の対照
- `stride-*.jsonl`, `fine-*.jsonl`: 不採用の追加順序実験
- `v3-notredame.json`, `exact-regression.json`, `cpu-quality-gate.json`
- `cargo-test.log`, `cargo-clippy.log`, `source-sha256.json`

v3の固定済みソース・結果・表は変更していない。今回の変更はmainの作業ツリーにあり、commit/pushはしていない。
