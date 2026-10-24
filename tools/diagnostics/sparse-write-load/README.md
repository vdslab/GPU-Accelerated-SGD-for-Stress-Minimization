# v3 Sparse予定更新負荷の補助診断

このツールは正式v3の計測完了後だけに使う。計測runner、6実装、shaderは変更しない。GPUとSGDは実行せず、CPU前処理を同じソース・seed・P・15反復の学習率条件で再現する。

既定対象はseed 0・P200のLuxembourg、web-Stanford、coAuthorsCiteseer、preferentialAttachment、kron_g500-logn17。正式basicに成功recordがあるものだけを選ぶ。同じseedで成功したSparse手法の入力・初期配置・前処理hashの一致を確認し、CPU→Atomic→RRの優先順で参照recordを選ぶ。成功recordがない対象は欠測として残す。

実行例（全3210計画recordの確定後）：

```sh
python3 experiments/analyze_v3_sparse_write_load.py
```

スクリプトは未完了の計画recordや生存する正式runnerがあれば、ビルドより前に拒否する。初回は`target/v3-diagnostics`へ独立ビルドし、各グラフを逐次処理する。公式binaryの出力場所は使わない。前処理自体は対象グラフ相応の時間・メモリを使うため、正式計測中には直接ツールを起動しない。

## 集計の定義

- 正重み端点の予定更新数：各制約で`weight_u > 0`ならuに1、`weight_v > 0`ならvに1を加える。1反復当たりの意味上の更新対象数であり、hardware store命令数ではない。
- 参考のAtomic lock対象数：各制約の両端へ1ずつ加える。Atomicが重み0の端点もlockするため別系列で保存する。実際の試行回数・retry・競合・待ち時間を計測するものではない。
- `all / pivot / nonpivot`ごとに頂点数、合計、平均、最大、母標準偏差÷平均のCV、p50/p90/p99、上位ceil(頂点数÷100)の割合、全頂点の負荷合計に占める群の割合、負荷値別の頂点数ヒストグラムを保存する。空の群や平均0のCVはnull。
- 全反復の予定数は1反復値×15。どの更新でも実移動量が非0になることは保証しない。Sparse RRでは複数の予定更新をlocalへ累積して最後にstoreし、Atomicはzero deltaの加算も行うので、両者の実書込み回数はこの指標と一致しない。

出力は`experiment_results/v3/analysis/sparse-write-load/`。`selection.json`に計測・診断の出所、`*.request.json`に照合条件、`*.result.json`に分布、`summary.csv`に集約値、`outcomes.json`に成功・失敗・欠測を保存する。入力SHA、共有graphソースSHA、前処理SHA、初期座標SHA、頂点数、制約数が参照recordと一致しなければ診断値を成功扱いしない。

`sparse-preprocess-v1` fingerprintのdomain・field順序は計測済みSparseのmain.rsと同じ。乱数は正式Sparseと同じrand 0.9.5を指定する。Cargo.lockは計測後の初回ビルド時に生成されるため、そのlockと診断ソースも解析の証跡として保存する。

この補助診断の数値は正式v3の速度・stress測定へ混ぜない。No.4の頂点別予定負荷を説明する少数グラフの補足であり、全データの因果推定ではない。

## 再開とサマリの整合

`*.provenance.json`は診断binary・Rustソース・Cargo manifest/lock・共有graphソース・orchestrationのSHA、要求した測定identityと、結果JSONのSHAを束縛する。sidecarがない、又は不一致なら、新しい一時出力先で診断をやり直し、検証後に結果を採用する。以前の結果JSONは`*.retired-*.json`へ残す。

CSVは毎回atomic置換し、全件失敗・欠測でもheaderだけの最新版へ更新する。各グラフの6統計行を全て検証してから採用する。`outcomes.json`の完了状態・CSV SHA・成功status・結果SHAとsidecarが一致しない行を、サマリreaderは表示しない。解析途中に前回の図用データを再利用しない。
