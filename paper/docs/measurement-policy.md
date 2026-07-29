# 計測・報告方針

## 目的

本研究では、GPU並列化によってStress SGDの解品質を保ちながら計算時間を短縮できるかを評価する。速度だけを独立に評価せず、同じdataset・seed・初期配置・反復条件における最終full stressと組み合わせて判定する。

比較する6手法は次のとおりである。

- Full Stress: `SGD`, `atomicSGD`, `RR-SGD`
- Sparse Stress: `SparseSGD`, `atomicSparseSGD`, `RR-SparseSGD`

Full Stress群とSparse Stress群では目的関数と制約数が異なるため、GPU方式は同じ群のCPU基準と比較する。

## 主要な評価指標

全runで、速度3指標と品質1指標を記録する。

| 指標 | 定義 | 目的 |
|---|---|---|
| Iteration time | 15 iterationsの実行時間 | GPU並列化した更新処理そのものの性能を測る |
| Algorithm time | 共通前処理から最終座標取得まで | 本研究の主要な速度指標 |
| CLI total time | 入力ファイル読込から最終座標取得まで | 1回の実利用で利用者が待つ時間を測る |
| Final full stress | 15 iterations後の座標を共通評価器で計算したfull stress | CPU基準と同等の解品質を保てたか判定する |

論文の中心的な主張にはAlgorithm timeとFinal full stressを使用する。Iteration timeは高速化方式の分析、CLI total timeは実用上の効果を説明する補助指標とする。

## 時間区間

計測値は次の区間に分けて保存する。

| フィールド | 含める処理 |
|---|---|
| `input_time` | Matrix Market読込、無向化、自己ループ処理、最大連結成分抽出 |
| `common_preprocess_time` | 最短路、pivot選択、Sparse制約生成、重み、学習率、初期座標生成 |
| `method_setup_time` | RRのschedule・round生成、atomicのlock/atomic用データ生成など方式固有の準備 |
| `runtime_init_time` | GPU adapter/device取得、shader・pipeline生成 |
| `upload_time` | 座標、制約、schedule、uniformなどのCPUからGPUへの転送 |
| `iteration_time` | 15 iterations全体。command生成、queue submit、必要なGPU完了待ちを含む |
| `gpu_device_time` | timestamp queryで測定したGPU commandの実行時間。取得可能な場合のみ |
| `readback_time` | 最終座標のGPUからCPUへの転送とmap |
| `postprocess_time` | 座標のcenteringなど、最終座標を返すまでに必要な後処理 |

RRのschedule生成は、iteration開始前に行われるため広い意味では前処理である。ただしRR方式だけに必要なコストなので、`common_preprocess_time`へ統合せず、`method_setup_time`として独立保存する。Algorithm timeとCLI total timeには必ず含める。

## 合計時間の定義

初回実行を表すcold値は次で定義する。

```text
Algorithm time (cold)
  = common_preprocess_time
  + method_setup_time
  + runtime_init_time
  + upload_time
  + iteration_time
  + readback_time
  + postprocess_time

CLI total time (cold)
  = input_time + Algorithm time (cold)
```

GPU runtimeとpipelineを再利用できる反復利用を想定する場合は、`runtime_init_time`を除いたAlgorithm time (warm)も報告する。ただし、coldとwarmを同じspeedupとして混在させない。主結果は1回のレイアウト生成を想定したcold値とし、warm値は補助結果とする。

Iteration平均は独立した計測値ではなく、次の派生値とする。

```text
Mean iteration time = Iteration time / 15
```

## Stressによる品質判定

Stressは小さいほどよい。全手法の最終座標を同じfull stress評価器で評価し、実装内部の近似stressだけでは品質を判定しない。

同じdataset・seedのCPU基準に対して、次の比率を計算する。

```text
Full Stress群:
  GPU stress ratio = GPU final full stress / SGD final full stress

Sparse Stress群:
  GPU stress ratio = GPU final full stress / SparseSGD final full stress
```

GPU方式のstress悪化は5%以内を基準とし、paired stress ratioの95%信頼区間上限が1.05以下であることを非劣性の判定条件とする。

速度向上率は同じdataset・seedのCPU基準から計算する。

```text
Iteration speedup = CPU iteration time / GPU iteration time
Algorithm speedup = CPU algorithm time / GPU algorithm time
CLI speedup       = CPU CLI total time / GPU CLI total time
```

## 計測に含めない処理

次の処理は速度計測から除外する。

- 結果TXT/CSVのファイル書出し
- PNGなどの描画
- full stressの事後計算
- 正解検証
- デバッグログとiterationごとの標準出力

これらは計測終了後に実行する。atomic方式の完了update数やretry失敗数は必要だが、毎iterationの同期・表示が速度を変える場合は、正式な速度runと詳細診断runを分ける。

## 実験手順

1. 全手法をrelease buildする。
2. 同じdataset・seed・初期座標・iterations・epsilonを使用する。
3. Sparse群では同じpivot、方向別重み、制約集合を使用する。
4. GPUは計測前に同条件を1回warm-upし、warm-up結果を統計へ含めない。
5. 正式runでは計測区間内の進捗表示を無効化する。
6. GPU時間は必ず処理完了を待ってからhost timerを停止する。
7. 各条件を最低10 runs実行し、品質評価の主要条件はseed 0〜24を使用する。
8. mean/min/maxに加えてmedian/IQRを報告する。
9. OOM、timeout、失敗runをゼロ時間として扱わず、失敗理由とともに記録する。

## runごとに保存するデータ

```text
method
dataset
seed
commit
cpu
gpu
n
m
constraint_count
pivots
iterations
epsilon
input_time_ms
common_preprocess_time_ms
method_setup_time_ms
runtime_init_time_ms
upload_time_ms
iteration_time_ms
gpu_device_time_ms
readback_time_ms
postprocess_time_ms
algorithm_time_cold_ms
algorithm_time_warm_ms
cli_total_time_cold_ms
final_full_stress
attempted_updates
completed_updates
retry_failures
rounds
dispatches
status
```

CPU手法やtimestamp query非対応環境など、該当しない値は0で埋めず`N/A`として扱う。

## 論文で示す図表

主要な図表は次の4点とする。

1. 手法別のAlgorithm timeとAlgorithm speedup
2. Algorithm timeの内訳
3. CPU基準に対するFinal full stress ratioと5%境界
4. Stress対wall-clock timeの収束曲線

Iteration speedupとCLI speedupは表または補助図として示す。要旨・結論で高速化率を述べる場合は、Iteration、Algorithm、CLIのどの範囲かを必ず明記する。
