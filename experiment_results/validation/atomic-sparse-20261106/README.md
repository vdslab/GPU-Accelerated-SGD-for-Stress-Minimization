# Atomic Sparse SGD: 原因調査とCPUとの再比較

2026-11-06、Apple M4 Pro。添付図の38グラフを対象に、Atomic Fullの排他更新に寄せた実装を検証する。
結果の表と図は [RESULTS.md](RESULTS.md)。全数値は [paired-summary.csv](paired-summary.csv)、機械集計は [summary.json](summary.json)、各実行の生データは `paired-raw/`。
測定の完了件数・失敗理由は [paired-status.jsonl](paired-status.jsonl) を参照。

## 採用判断

改良版をAtomic Sparseの既定として採用する。図の38グラフすべてが成功し、
反復速度比は中央値2.80倍（2.63–3.37倍）、初回準備・転送込みでも中央値2.28倍。
共通前処理まで含めた比は中央値1.56倍。5グラフ×未使用2 seedの10件も成功し、反復速度比の中央値2.79倍だった。

CPUに対する主比較の推定stress比は中央値1.0035、範囲0.9165–1.0719。
したがって「CPUと常に同品質」とは言わない。
上昇が大きかった2グラフ×2 seedを256始点で追加比較すると、新方式のCPU比は0.9720–1.0077、
旧Atomic比は0.9465–1.0174だった。旧方式から一律に品質を落として速度を稼いだ結果ではないが、
この4件が全グラフ・全seedでの非劣性を証明するわけではない。
64始点の主比較と256始点の再実行には評価始点数とGPU更新順の両方の差があるため、差をサンプリングだけに帰属しない。

本番ソースとrelease CLIへ反映済み。既存v1–v4の測定値・論文図は書き換えていない。
今後の正式実験では新しい実装タグとbinary SHAを記録し、旧Atomic結果と混ぜずに集計する。

## 原因

図のv3ソースを保存して照合した。`sources/v3-snapshot-shader.metal` は今回のv3対照shaderとSHA-256が一致する。
図側のホストは毎反復 `all_pending.shuffle(rng)` を実行し、128回のロック試行で失敗した制約をCPUで抽出して再投入する。
数千万件のCPUシャッフルと共有バッファ更新・完了待ち・再投入がSGD時間に含まれる。
Atomic Fullには毎反復のCPUシャッフルと未完了制約のCPU再投入がない。

同条件の追加比較（kron_g500-logn18、seed 0）では、旧方式の15反復は15.917秒、GPU timestamp合計は5.039秒。
差の10.878秒はCPUシャッフル・共有バッファ処理・再投入・投入待ちなど、GPU command実行区間の外にある。
本番改良版は15反復4.572秒、GPU timestamp 4.315秒で、その差は0.257秒まで減った。
初回準備・転送1.320秒を加えても5.892秒で、同じCPUの15.761秒より短い。
この差分だけからシャッフル単体の時間を厳密に分離したとは主張しない。

この作業ディレクトリの変更前コードは、すでに毎反復シャッフルを既定で無効にしていたが、図側とは別実装だった。
さらに、Full/v3は個々のロック取得・解放にfenceを置く一方、このディレクトリの旧Sparseは両端取得後と更新後だけだった。
今回、2番目のロック取得失敗時のrollbackも含め、Fullと同じfence配置に統一した。
fenceを省いて得た速度は採用根拠にしない。

## 実装

- 32スレッドのthreadgroupにつき1 lane・1制約、両端を頂点ID順に排他ロック、座標読取り後に方向別重みでatomic加算。Fullと同じ形を維持。
- Fullと同じGPU内再試行。2番目のロックに失敗すると1番目を解放する。1反復1dispatch。
- 準備時に一度だけ制約をシャッフルし、以後は固定投入順。同じ頂点を連続して争う偏りを減らす。初回シャッフルは準備時間に計上。
- 反復ごとの全indexコピー・未完了制約の抽出・CPU再投入を削除。
- 完了epochを制約ごとに記録し、毎反復検証する。過去の完了フラグで未完了を隠さない。
- `--cpu-shuffle` は毎反復CPUシャッフルの対照として維持する。通常実装でもGPU内再試行は新方式。
- JSONの `atomic_schedule` と `atomic_pair_order`、座標ヘッダー、バイナリハッシュで実装条件を区別。

Fullとの差はSparse固有の方向別重みと、初回だけ行う順序分散。ロックの省略、pivot座標の古い値への固定、更新数削減、追加反復は行わない。
GPUはf32、CPUは従来のf64。GPUのロック取得順序には非決定性がある。
Full同様の無制限GPU内再試行には実行時間の上限を理論的には保証できないため、測定runnerが240秒でプロセスを停止する。

## 実験条件と時間の定義

全38グラフでseed 0、15反復、200 pivots、epsilon 0.1。最大連結成分のv4入力を用いる。
同じ前処理結果をcloneし、CPU参照実装と新しい本番Metal runnerへ渡す。
CPUの反復時間には従来どおり毎反復シャッフルを含む。
GPUの反復時間にはGPU投入・GPU待機・完了epoch検証を含む。GPU timestampだけをCPU時間と比較しない。

- `iteration_speedup`: CPU反復時間 / Atomic反復時間。
- `setup_inclusive_speedup`: CPU反復時間 / (Atomic初回順序分散・変換・バッファ準備・upload + 反復時間)。
- `preprocessing_inclusive_speedup`: 共通Sparse前処理も双方に加えた比。入力読込み・Metal初期化・stress評価・ファイル出力は含まないため、CLI全体の速度比ではない。

主比較のstressは同じ64始点・評価seed 0によるsampled full stress。
追加検証はluxembourg_osm、smallworld、kron_g500-logn17、web-NotreDame、coAuthorsCiteseerの未使用seed 1,2で、同じ256始点を用いる。
主比較のseed 0と追加検証は始点数が違うため、stress絶対値をまとめて平均しない。
全グラフ1 seedの比較は採用可否を判断する探索測定であり、論文用の複数seedによる確定評価ではない。
グラフごとの独立warmupは行っていない。予備測定と小規模smoke後にCPU/GPUを逐次実行し、追加seed 1では実行順を反転した。

同一プロセス内の比較用workloadコピーを保持するため、単独CLIよりメモリ使用量が多い。
1ケース240秒・RSS 12 GiB上限。上限停止やエラーは速度・品質集計から除外し、成功として扱わない。
グラフ入力・実行バイナリ・ソースのSHAは `paired-config.json`、`paired-status.jsonl`、`provenance.json` に保存。

## 予備実験

`pilot-correct-fences.jsonl` はluxembourg_osmで同じロックfenceを使った対照。
GPU内再試行だけでなく初回順序分散も切り分ける。`full-once` は旧ホストの余分なコピー・走査が残る初期候補で、本番ホストの最終値とは別。
`pilot-quality.jsonl` の `v3-each` はv3 shader・毎反復CPUシャッフルの再現対照。
CPU、旧方式、初期候補を同じ入力で比較した結果を残す。
`pilot.jsonl` の `full` 系はfenceが重複した初期診断なので、最終方式の速度集計には使わない。

## 品質悪化例の追加比較

主比較のCPU比stress上昇が最大だったkron_g500-logn18と、その次のcom-DBLPを追加選択した。
`experiments/run_atomic_sparse_quality_followup.py` は主測定48ケースの完了後に実行し、
各グラフのseed 0,1でCPU・v3対照・本番版を同一前処理・256共通始点で比較する。
seed 1では実行順を反転する。各ケースの240秒上限、`time -l` のRSS記録、実行コマンドを保存する。
結果は `quality-followup/` と `quality-followup-summary.csv`。主比較と混ぜず `RESULTS.md` の別表にする。

## 除外記録と検証

`invalid-stale-runner/` はビルド失敗で旧試験バイナリが使われた測定。
新しい `production` 指定を旧版が固定順の診断モードとして処理し、stress始点数も指定値と異なったため、すべて除外した。
修正後は未知モードを拒否し、smoke測定と各結果のmode・stress始点数・バイナリSHAを検証する。除外値は最終表・図に混入させない。

`tests.log` は本番GPUとCLIのテスト結果。片側重み、mu上限、座標一致、共有頂点競合、65,536制約×3反復に加え、
第2反復で完了epochを書かない負の対照が確実に失敗することを確認する。
`python-tests.log` は既存実験runnerの回帰テスト。

再実行:

```sh
cargo build --release --locked --manifest-path tools/diagnostics/sparse-audit/Cargo.toml --bin speed-probe
python3 experiments/run_atomic_sparse_probe.py
python3 experiments/summarize_atomic_sparse_probe.py
```

runnerは保存済みバイナリSHAが異なる場合に停止する。新実装を測り直す場合は別の結果ディレクトリを用いる。

最終検証: Rust 31テスト、Python 108テスト成功。診断bin群の `cargo check --locked --bins` 成功。
通常CLIの既定／`--cpu-shuffle` の両方でJSON Schema、更新数、dispatch数、前処理・初期配置一致を確認した。
`cli-smoke/validation.json` と各ログを参照。

`sources/` は測定時のコードsnapshot。診断用Rustファイルは測定後に整形のみを行った。仕上げ後のファイルSHAは `final-source-hashes.json` に保存し、本番のMetal・Rust実装が測定時snapshotと一致することも確認した。
