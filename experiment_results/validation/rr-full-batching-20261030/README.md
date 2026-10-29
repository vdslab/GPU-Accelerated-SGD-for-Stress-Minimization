# RR Fullの追加バッチ化・反復間CPU待機除去の評価（2026-10-30）

## 判断

**標準実装には不採用。更新結果は維持できたが、既定設定で一貫した高速化は確認できなかった。**

ラウンドごとのCPU待機を反復末尾へ集約する既存の高速化は維持した。今回は、さらに実行区間を集約し、CPUの待機を全反復末尾へ移す案を比較した。論文v4のFull図表・esaは変更していない。

## 候補の変更

- 全外側ラウンドのタイル割り当てを事前転送し、ラウンドごとのコピーを除去。
- 1反復分の全ラウンドの内側シャッフル列を個別領域に保持し、dynamic storage offsetで選択。
- 最大256 dispatchを1 compute passに記録。
- queue writeが次のsubmitの前に実行される順序を利用し、反復間のCPU完了待ちを除去。全反復末尾では必ず待機する。
- shader、更新式、乱数系列、学習率、f32、15反復、T/W、同一ラウンドのmatchingは維持。グラフ別の切替はない。
- 設定バッファが1反復分の全ラウンドを保持するため、元よりメモリを使う。

CPU待機を除くことはGPUの依存関係を除くことではない。ラウンド内のタイルは独立で、ラウンド間のpositionsへの読み書きはwgpu/backendの同期に従う。CPU書込元はwrite_buffer内でコピーされ、次反復の転送は前反復の最後のsubmitより後に実行される。

## 測定条件

Apple M4 Pro、Metal、release build。通常設定T=1024、W=1024、shared、内外shuffleあり。15反復、epsilon=0.1。各グラフseed 0/1/3で各1測定、追加でseed 0を各3回測定。実行は直列、before/after順を交互にし、CPUは同一入力・初期配置で測定。両GPU版は事前warmupを各1回行った。通常seed測定ではCPU比較も実行し、追加反復測定ではCPUを挟まない。温度・クロックの固定や信頼区間の推定は行っていない。

SGD時間はnativeのiteration_time_ms。シャッフル生成、設定転送の投入、command記録、GPU完了待ちを含むwall timeであり、CPU投入だけの時間ではない。前処理・画像生成・stress評価は区間外。gpu_device_time_msも別に保持した。

## 通常設定の全測定

時間は15反復合計ms。前/後は同一seedの時間比で、1より大きければ候補が速い。CPUも今回測定した値で、過去の参照値ではない。

| グラフ | seed | 現行RR ms | 候補RR ms | CPU ms | 前/後 | CPU/候補 | 候補/CPU stress |
|---|---:|---:|---:|---:|---:|---:|---:|
| fe_4elt2 | 0 | 684.612 | 665.415 | 17336.291 | 1.029 | 26.053 | 1.010130 |
| fe_4elt2 | 1 | 618.772 | 657.309 | 17630.079 | 0.941 | 26.822 | 1.023934 |
| fe_4elt2 | 3 | 623.935 | 588.641 | 17591.514 | 1.060 | 29.885 | 1.023866 |
| delaunay_n14 | 0 | 1074.529 | 1034.229 | 48888.029 | 1.039 | 47.270 | 0.926381 |
| delaunay_n14 | 1 | 1185.835 | 1048.067 | 40523.756 | 1.131 | 38.665 | 1.153997 |
| delaunay_n14 | 3 | 1437.590 | 1056.455 | 40053.972 | 1.361 | 37.914 | 1.164555 |

通常3 seedの速度比中央値はfe_4elt2が1.029倍、delaunay_n14が1.131倍。ただし、追加測定では次のとおり逆転した。

## 同一seedの追加測定

| グラフ | 追加回 | 現行RR ms | 候補RR ms | 前/後 |
|---|---:|---:|---:|---:|
| fe_4elt2 | 0 | 545.054 | 520.292 | 1.048 |
| fe_4elt2 | 1 | 557.247 | 550.359 | 1.013 |
| fe_4elt2 | 2 | 508.178 | 572.406 | 0.888 |
| delaunay_n14 | 0 | 1320.992 | 1415.448 | 0.933 |
| delaunay_n14 | 1 | 1342.960 | 1444.037 | 0.930 |
| delaunay_n14 | 2 | 1207.594 | 1134.362 | 1.065 |

追加3回の比の中央値はfe_4elt2が1.013倍、delaunay_n14が0.933倍。後者では3回中2回が遅くなった。追加測定のGPU timestampはfe_4elt2が現行468〜479 ms/候補470〜480 ms、delaunay_n14が現行855〜867 ms/候補859〜863 msでほぼ同水準だった。wall timeの変動が大きいため、原因をCPU待機・投入手間だけに帰属したり、安定した速度向上を主張したりしない。

細かいタイルの追加診断では改善例もあるが、グラフや設定で手法を切り替える採用は行わない。現行の共通実装を維持した。

## 品質・安全性の検証

通常6組、追加反復6組、特殊設定8組の**計20組で最終座標テキストSHAとstressが完全一致**。入力・初期配置・dispatch数・反復数・seed・制約数・評価設定も照合した。CPUとのstress差は従来のままで、この案による改善を主張しない。stressはCLI既定の固定64始点・評価seed 0のsampled値。全頂点対の厳密stressではない。

特殊設定はUSpowerGrid T127/W64/shared、EVA T32/W32/global、1138_bus T4/W32/shared、1138_bus T127/W64/global（内外shuffleなし）、各seed 0/3。T4は外側286ラウンド/反復で256 dispatchのバッチ境界をまたぐ。奇数・端数ブロック、W<T、shared/global、shuffle無効を含む。

候補のcargo testは7 unit + 1 integration testが成功。既存のcenter/wijのdead_code警告2件あり。追加したstorage sliceの境界検査も通過。

## 再現・保存

- baseline commit: `6f8c2c19d87548eac40201136a42baf7b354c7e5`。今回の候補以外のソースは同じ。
- `candidate.patch`: 上記baselineに適用するホストコードの変更。標準ソースには適用していない。
- `provenance.json`: ソース・binary・入力・patchのSHA、復元の記録。
- `pairs.json`, `regressions.json`, `repeat-pairs.json`, `summary.json`: 全対応比較。
- 各JSONにコマンド、binary SHA、native値、最終座標SHAを保存。記録中のafter実行パスは測定当時のパスで、現在その標準パスのbinaryはbaselineへ復元済み。
- `candidate-tests.log`: テスト結果。`artifacts/`は座標と画像をローカル保持しGit対象外。

ローカルの保存済み候補binaryを使う再現コマンド（新規測定では新しい出力先を指定）：

```sh
python3 tools/diagnostics/rr-full-batching/benchmark.py \
  --before target/rr-full-batching-baseline/rr-gpu \
  --after target/rr-full-batching-candidate/rr-gpu \
  --output output/rr-full-batching-recheck
```

別環境ではbaselineソースにcandidate.patchを適用してrelease buildし、before/afterのbinaryを別々に保持する。binary自体はGit対象外。
