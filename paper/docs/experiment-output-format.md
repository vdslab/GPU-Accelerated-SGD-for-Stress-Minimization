# 実験出力フォーマット

## 位置づけ

この文書は、論文実験で1 runから取得するデータとコンソール出力の形式を固定する。速度指標の意味は[measurement-policy.md](measurement-policy.md)、比較条件は[experiment-plan.md](experiment-plan.md)を正本とする。

この形式はOpenSpec change `standardize-experiment-output`で実装した共通契約である。機械可読な正本は`experiments/schema/experiment-record-v1.json`とし、現在の`schema_version`は`1`である。

## 対象手法

| method | family | 実装 |
|---|---|---|
| `sgd` | `full` | `baseline-sgd-non-gpu` |
| `atomic_sgd` | `full` | `vram-lock-native` |
| `rr_sgd` | `full` | `rr_gpu` |
| `sparse_sgd` | `sparse` | `baseline-sparse-sgd-non-gpu` |
| `rr_sparse_sgd` | `sparse` | `sparse-sgd-gpu` |
| `atomic_sparse_sgd` | `sparse` | 未実装。完成後に同じschemaへ追加 |

Full群とSparse群は目的関数・制約数が異なるため、速度とstressの基準は同じfamilyのCPU手法とする。

## コンソール出力の契約

実験ランナーは各binaryを`--output-format json`で実行する。

- 成功時のstdoutは、改行を含まないJSONオブジェクト1行だけとする。
- 進捗、iteration番号、GPU adapter情報、schedule情報、保存通知をstdoutへ出さない。
- `--verbose`指定時の進捗と診断はstderrへ出す。
- エラー理由はstderrへ出し、processは非0で終了する。成功JSONを出してはならない。
- runnerはstdoutを1 JSON行として検証し、raw dataの`results.jsonl`へ1行ずつ追記する。
- CSVはraw dataの正本にせず、論文表を作る段階でJSONLから生成する。

通常の人間向け実行は残してよいが、正式実験では使用しない。

## 共通CLI

全手法が次を受け付ける。

```text
--run-id <ID>
--input <PATH>
--iterations <N>
--epsilon <FLOAT>
--seed <U64>
--output-format json
--output-dir <PATH>
--run-mode benchmark|diagnostic
--verbose
```

Sparse群は`--pivots <N>`も受け付ける。正式な速度計測は`benchmark`、追加同期やreadbackが必要な統計取得は`diagnostic`とし、両者の時間を同じ集計へ混ぜない。

## JSON record

### 識別・再現情報

| field | 型 | 必須 | 内容 |
|---|---|---:|---|
| `schema_version` | integer | yes | 初期版は`1` |
| `run_id` | string | yes | runnerが条件から決定的に生成 |
| `run_mode` | string | yes | `benchmark`または`diagnostic` |
| `status` | string | yes | `success`または`failure` |
| `method` | string | yes | 上表の固定ID |
| `family` | string | yes | `full`または`sparse` |
| `binary` | string | yes | 実行したrelease binary |
| `git_commit` | string | yes | 実験コードのcommit SHA |
| `git_dirty` | boolean | yes | 未commit変更の有無。正式実験は`false` |
| `dataset` | string | yes | データセットの固定名 |
| `input_path` | string | yes | 実行時の入力パス |
| `input_sha256` | string | yes | 入力ファイルのSHA-256 |
| `seed` | integer | yes | アルゴリズム用seed |
| `initial_positions_sha256` | string | yes | 初期座標列のhash |
| `preprocess_sha256` | string/null | yes | Sparse群のpivot・制約・学習率fingerprint。Full群は`null` |

### 問題規模・パラメータ

| field | 型 | 非該当時 | 内容 |
|---|---|---|---|
| `nodes` | integer | - | 採用した最大連結成分の頂点数。失敗recordでは`null`可 |
| `edges` | integer | - | 無向化・自己ループ除去後の辺数。失敗recordでは`null`可 |
| `constraints` | integer/null | `null` | 実際に更新対象とした制約数 |
| `pivots` | integer/null | `null` | Sparse群のpivot数 |
| `iterations` | integer | - | 正式実験の基本値は15 |
| `epsilon` | number | - | 正式実験の基本値は0.1 |

### 実行環境

| field | 型 | 非該当時 | 内容 |
|---|---|---|---|
| `cpu_model` | string | - | CPU名 |
| `gpu_name` | string/null | `null` | 使用GPU |
| `gpu_backend` | string/null | `null` | Metal、Vulkanなど |

OS、compiler、driverなどrun間で共通の情報は、各recordへ重複させずexperiment directoryの`environment.json`にも保存する。

### 時間

単位はすべてmillisecond、host wall-clockである。

| field | 非該当時 | 区間 |
|---|---|---|
| `input_time_ms` | - | 読込、無向化、自己ループ処理、最大連結成分抽出 |
| `common_preprocess_time_ms` | - | 最短路、pivot、制約、重み、学習率、初期座標 |
| `method_setup_time_ms` | `null` | RR schedule、atomic用データなど方式固有準備 |
| `runtime_init_time_ms` | `null` | GPU adapter/device、shader、pipeline |
| `upload_time_ms` | `null` | CPUからGPUへの転送 |
| `iteration_time_ms` | - | 全iterationsのcommand生成、submit、GPU完了待ち |
| `gpu_device_time_ms` | `null` | timestamp queryで取得できたdevice実行時間 |
| `readback_time_ms` | `null` | GPUからCPUへの最終座標転送とmap |
| `postprocess_time_ms` | - | centeringなど最終座標を返すまでの処理 |
| `algorithm_time_cold_ms` | - | input以外の上記区間の和 |
| `algorithm_time_warm_ms` | `null`可 | coldからruntime initを除いた派生値 |
| `cli_total_time_cold_ms` | - | input + algorithm cold |

主要な速度指標は次の3つである。

```text
Iteration time
Algorithm time (cold)
CLI total time (cold)
```

RRのscheduleは`method_setup_time_ms`へ分離するが、Algorithm timeとCLI total timeには含める。ファイル保存、stress評価、PNG描画、検証、ログ出力は含めない。

CPU手法でGPU固有区間が存在しない場合、recordには`null`を保存する。合計を計算するときだけ非該当区間を0コストとして扱う。取得できなかった値を0msにしてはならない。

### Stress

| field | 型 | 内容 |
|---|---|---|
| `stress_kind` | string | `exact`または`sampled` |
| `stress_value` | number | 共通評価器による最終座標のstress |
| `stress_eval_time_ms` | number | 速度指標に含めない評価時間 |
| `stress_samples` | integer/null | sampledで使用した始点数 |
| `stress_seed` | integer/null | sampledの始点選択seed |

8,000頂点以下はexact、8,000頂点超は既定64始点のsampledとする。sampled値を`final_full_stress`と呼ばない。品質の主比較でexactとsampledを混ぜない。

### 方式固有統計と成果物

| field | 非該当時 | 内容 |
|---|---|---|
| `attempted_updates` | `null` | atomicで試行した更新数 |
| `completed_updates` | `null` | 完了した更新数 |
| `retry_failures` | `null` | retry上限などで完了しなかった数 |
| `rounds` | `null` | RR scheduleのround数 |
| `dispatches` | `null` | GPU dispatch数 |
| `final_positions_path` | `null`不可 | 最終座標成果物 |
| `vertex_map_path` | `null`可 | 元頂点番号との対応 |

追加同期を行わないと取得できない詳細統計は`diagnostic` runだけで取得する。

### 失敗情報

成功recordでは次を`null`にする。method processが失敗した場合は、runnerが可能な範囲を埋めた`failure` recordを作る。

| field | 内容 |
|---|---|
| `error_stage` | `build`、`input`、`runtime_init`、`execute`、`timeout`、`parse`、`validate`など |
| `error_message` | 短い失敗理由 |
| `exit_code` | process終了code。timeoutなら`null` |
| `stderr_log_path` | run別stderr log |

失敗を0msの成功runとして扱ってはならない。

## 成功recordの例

見やすさのため整形しているが、実際のstdoutとJSONLでは1行にする。

```json
{
  "schema_version": 1,
  "run_id": "e0-uspowergrid-rr_sparse_sgd-s000-r00",
  "run_mode": "benchmark",
  "status": "success",
  "method": "rr_sparse_sgd",
  "family": "sparse",
  "binary": "target/release/sparse-sgd-gpu",
  "git_commit": "0123456789abcdef",
  "git_dirty": false,
  "dataset": "USPowerGrid",
  "input_path": "data/USpowerGrid.mtx",
  "input_sha256": "sha256-value",
  "seed": 0,
  "initial_positions_sha256": "positions-sha256",
  "preprocess_sha256": "sparse-preprocess-sha256",
  "nodes": 4941,
  "edges": 6594,
  "constraints": 123456,
  "pivots": 200,
  "iterations": 15,
  "epsilon": 0.1,
  "cpu_model": "Apple CPU",
  "gpu_name": "Apple GPU",
  "gpu_backend": "Metal",
  "input_time_ms": 12.4,
  "common_preprocess_time_ms": 318.7,
  "method_setup_time_ms": 41.2,
  "runtime_init_time_ms": 74.8,
  "upload_time_ms": 8.9,
  "iteration_time_ms": 26.3,
  "gpu_device_time_ms": null,
  "readback_time_ms": 1.8,
  "postprocess_time_ms": 0.6,
  "algorithm_time_cold_ms": 472.3,
  "algorithm_time_warm_ms": 397.5,
  "cli_total_time_cold_ms": 484.7,
  "stress_kind": "exact",
  "stress_value": 724120.5,
  "stress_eval_time_ms": 92.1,
  "stress_samples": null,
  "stress_seed": null,
  "attempted_updates": null,
  "completed_updates": null,
  "retry_failures": null,
  "rounds": 231,
  "dispatches": 3465,
  "final_positions_path": "output/experiments/e0/artifacts/e0-uspowergrid-rr_sparse_sgd-s000-r00/final.txt",
  "vertex_map_path": "output/experiments/e0/artifacts/e0-uspowergrid-rr_sparse_sgd-s000-r00/vertex-map.txt",
  "error_stage": null,
  "error_message": null,
  "exit_code": null,
  "stderr_log_path": null
}
```

## 収集時の検証

runnerはrecordを追記する前に、少なくとも次を検証する。

1. stdoutがJSON 1行だけである。
2. `schema_version`を解釈できる。
3. manifestのrun ID、method、seed、パラメータとrecordが一致する。
4. 必須値が有限で、時間が非負である。
5. Algorithm/CLI合計が内訳と丸め許容誤差内で一致する。
6. `exact`と`sampled`に必要なfieldの組合せが正しい。
7. 非該当値が0や空文字ではなく`null`である。
8. 成功時の成果物が存在する。
9. 同じFull条件では初期座標hash、同じSparse条件では初期座標hashと前処理fingerprintが手法間で一致する。
