## Context

比較対象の実装済み5手法は、別々のRust crateとして発展してきたため、CLI、乱数seed、ログ、計測開始・終了位置、GPU同期、結果メタデータが一致していない。特にFull Stress群の3手法は入力やseedがハードコードされており、現在表示される「実行時間」をそのまま横並びにしても同じ処理範囲の比較にならない。

論文では、同じdataset・seed・初期配置に対する速度とストレスを対にして評価する。速度はIteration、Algorithm、CLI totalの3区間、品質はタイミング外で計算する最終ストレスを使用する。RRのschedule生成は方式固有の前処理であり、内訳では分離するがAlgorithm timeからは除外しない。CPUに存在しないGPU区間やtimestamp query非対応値は`null`として扱う。

## Goals / Non-Goals

**Goals:**

- 5手法が同じ実験用CLIと同じJSONレコード契約を実装する。
- seed、入力、主要ハイパーパラメータを外部から固定し、同一条件を再実行できるようにする。
- 時間区間の境界とGPU同期条件を揃え、合計値を機械検証できるようにする。
- ストレスを同じ実装でタイミング後に評価し、厳密値と標本化推定を区別する。
- manifestから実験条件を列挙し、途中失敗から再開可能なJSONLデータセットを作る。
- 未実装のAtomicSparseSGDを、スキーマ変更なしで後から追加できるようにする。

**Non-Goals:**

- この変更内でAtomicSparseSGD自体を実装しない。
- Full Stress群とSparse Stress群のアルゴリズムや目的関数を統一しない。
- 各アルゴリズムの更新式、round構築法、atomic競合処理を変更しない。
- 論文用の統計集計、グラフ描画、PDF生成までは自動化しない。
- 既存の人間向け通常実行を完全に廃止しない。

## Decisions

### 1. 実験契約を共通Rust crateへ集約する

新しい`experiment-common` crateに、実験用CLI共通項目、method ID、schema version、record型、時間区間型、ストレス結果型、JSON直列化、合計値検証を置く。各手法が独自のJSONを組み立てる方式は、field名や`null`処理のdriftを起こすため採用しない。

共有するのは実験契約と評価器であり、SGDのアルゴリズム実装そのものはこの段階では移動しない。大きな共通化と計測変更を同時に行うと、性能差の原因を切り分けにくくなるためである。

### 2. 実験モードのstdoutをJSON 1行に限定する

各binaryは`--output-format json`を受け付ける。このモードの成功時は、改行を含まないJSONオブジェクトをstdoutへ1行だけ出力する。進捗、GPU adapter情報、保存通知、警告はstdoutへ出さない。`--verbose`指定時だけ診断情報をstderrへ出す。

通常の人間向け出力は当面維持する。全面的な既定出力変更より移行リスクが小さく、実験ランナーは常に`--output-format json`を指定するため、収集データの厳密性も保てる。

### 3. method IDと共通CLIを固定する

実装済み5手法のmethod IDとcrateを次のように対応付ける。

| method | family | crate |
|---|---|---|
| `sgd` | `full` | `baseline-sgd-non-gpu` |
| `atomic_sgd` | `full` | `vram-lock-native` |
| `rr_sgd` | `full` | `rr_gpu` |
| `sparse_sgd` | `sparse` | `baseline-sparse-sgd-non-gpu` |
| `rr_sparse_sgd` | `sparse` | `sparse-sgd-gpu` |

全手法は少なくとも`--run-id`、`--input`、`--iterations`、`--epsilon`、`--seed`、`--output-format`、`--output-dir`、`--run-mode`を受け取る。Sparse群は`--pivots`も受け取る。AtomicSparseSGDは`atomic_sparse_sgd`として同じ契約へ追加する。

初期配置を確実に比較できるよう、同一family・dataset・seedで生成した初期座標のhashもレコードへ保存する。Full群とSparse群の直接一致は要求せず、各family内でCPU/GPU間の一致を検証する。

### 4. 時間区間を単一所有にし、合計を派生させる

各処理は`input`、`common_preprocess`、`method_setup`、`runtime_init`、`upload`、`iteration`、`readback`、`postprocess`のちょうど1区間へ所属する。ファイル保存、ストレス評価、検証、デバッグ出力は全区間の外に置く。

`algorithm_time_cold_ms`と`cli_total_time_cold_ms`は別timerの曖昧な値ではなく、区間値の和として共通crateで計算する。`algorithm_time_warm_ms`はcold値から`runtime_init_time_ms`を除いた派生値とする。CPU方式でGPU固有区間が存在しない場合、その区間は`null`だが合計上は0コストとして扱う。

GPUのhost wall timeは、対象queue処理の完了を待ってからtimerを停止する。`gpu_device_time_ms`はtimestamp query対応時だけ記録し、Iteration timeの代替にはしない。Iteration timeには各iterationに必要なCPU側command生成、submit、完了待ちを含める。

### 5. ストレス評価器を共有し、タイミングから除外する

`gpu_visualizer`の厳密・標本化ストレス計算を共有モジュールへ移し、visualizerと5手法から同じ実装を利用する。頂点数8,000以下は`exact`、それを超える場合は既定64始点・固定stress seedの`sampled`を使用する。recordは`stress_kind`、`stress_value`、`stress_samples`、`stress_seed`、`stress_eval_time_ms`を持ち、標本化値をfull stressとは表記しない。

ストレスは最終座標をCPUへ取得し、Algorithm/CLI timerを停止した後に計算する。これにより品質データを同じrunへ紐付けつつ、速度値を汚染しない。

### 6. JSONLを正本とし、該当しない値はnullにする

1実行のrecordはschema version、run ID、status、method、入力checksum、commit、パラメータ、グラフ規模、実行環境、時間内訳、ストレス、方式固有統計、成果物パスを含む。成功recordはmethod binaryが生成し、失敗recordはrunnerがexit code、失敗段階、stderr log pathを付けて生成する。

CSVは入れ子や`null`、将来field追加に弱いためraw dataの正本にしない。`results.jsonl`をappend-onlyの正本とし、論文表用CSVは後段で生成する。schema version 1のfield追加はoptional fieldに限り、意味や単位を変える場合はversionを上げる。

### 7. 標準ライブラリだけのPython runnerとJSON manifestを使う

`experiments/run_experiments.py`が`experiments/manifests/*.json`を読み、method、dataset、seed、iteration、epsilon、pivot、repetitionの直積を作る。Python標準ライブラリの`subprocess`と`json`だけを使い、追加パッケージを不要にする。runnerは先にrelease binariesの存在を検証し、`--dry-run`で予定run一覧を表示できる。

各条件は統計に含めないwarm-upを指定回数実行した後、正式runを逐次実行する。run IDはmanifest内容と条件から決定的に生成し、既存の成功recordがあれば再開時にskipする。失敗はゼロ時間として扱わず、失敗recordとstderr logを残して次のrunへ進む。並列実行はGPU競合で計測を歪めるため初期版では行わない。

出力構造は次とする。

```text
output/experiments/<experiment-id>/
  manifest.json
  environment.json
  results.jsonl
  stderr/<run-id>.log
  artifacts/<run-id>/...
```

### 8. 速度runと詳細診断runを区別する

atomicのattempted/completed/retry統計やRRのround/dispatch統計は、通常の実行で追加同期なしに得られる場合だけ速度recordへ含める。取得のためにiteration内同期やreadbackが増える場合は`run_mode: diagnostic`で別実行し、`run_mode: benchmark`の速度値と混在させない。

## Risks / Trade-offs

- [共通計測の追加自体が性能へ影響する] → iteration内の出力と不要な同期を禁止し、release buildで計測前後のtimerだけを使用する。
- [GPU host時間とdevice時間が一致しない] → 主指標を同期済みhost wall timeに固定し、device timestampは補助fieldとして分離する。
- [既存コードの処理区間が重なっており分類しにくい] → 先に区間境界をコメントとテストで固定し、合計値の整合性検査を各binaryへ共通適用する。
- [乱数ライブラリ差で同じseedでも初期配置がずれる] → familyごとに初期配置生成を共通関数化し、initial position hashを検証する。
- [大規模グラフの厳密ストレスが実用時間を超える] → exact/sampleの種別をrecordに必須化し、sampled結果をfull stressとして比較しない。
- [JSONLが中断時に末尾破損する] → 1 recordごとにflushし、再開時に最終不完全行を検出して隔離する。
- [通常出力と実験出力の二経路がdriftする] → 実験recordを内部の正本にし、人間向け表示も同じrecordから生成する方向で段階移行する。

## Migration Plan

1. `experiment-common`にschema、record、時間集計、共通seed設定、stress評価を追加する。
2. CPU 2手法へ共通CLIとJSON出力を導入し、schemaとタイミング合計のcontract testを通す。
3. GPU 3手法へ同じ契約を導入し、queue完了待ちとGPU固有区間を揃える。
4. 5手法のstdout snapshot test、同一seed検証、stress評価一致テストを追加する。
5. runner、manifest例、resume/failureテストを追加し、小規模fixtureでE0相当のend-to-end試験を行う。
6. 既存の人間向け出力を残したまま、論文実験はrunner経由へ切り替える。

問題が見つかった場合は、runnerを使わず従来の通常実行へ戻せる。JSON schema version 1を公開後にfieldの意味を変更する場合は、旧readerを残してversionを上げる。

## Open Questions

- GPU timestamp queryを全対象環境で利用できるかは実装時に確認し、非対応時は`null`を正式な値とする。
- peak CPU RAM/GPU memoryの移植性のある取得方法は別変更で検討し、本変更では任意fieldに留める。
- 将来AtomicSparseSGDをどのcrate名で追加するかは未確定だが、method IDは`atomic_sparse_sgd`に固定する。
