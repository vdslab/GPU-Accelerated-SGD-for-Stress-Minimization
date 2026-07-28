## 1. 共通実験契約

- [x] 1.1 `experiment-common` crateを追加し、5手法と`gpu_visualizer`から参照できるworkspace/path dependencyを設定する
- [x] 1.2 schema version、method/family/run mode/status、共通CLI項目を型として定義する
- [x] 1.3 実験record、時間内訳、ストレス結果、方式固有統計、成果物情報を`serde`型として定義し、非該当値を`null`へ直列化する
- [x] 1.4 Algorithm cold/warmとCLI totalを内訳から計算し、有限性・非負性・合計誤差を検証する共通処理を実装する
- [x] 1.5 dataset SHA-256、Git commit、実行環境、初期座標hashを収集する共通処理を実装する
- [x] 1.6 schema version 1のJSON Schemaまたは同等のvalidator fixtureを`experiments/schema/`へ追加する

## 2. seedとストレス評価の共通化

- [x] 2.1 Full Stress群の初期座標生成をseed指定可能な共通関数へ移し、3手法で同じ座標hashになるテストを追加する
- [x] 2.2 Sparse Stress群のpivot・制約・学習率・初期座標生成結果を比較可能にし、2手法で同じ識別hashになるテストを追加する
- [x] 2.3 `gpu_visualizer`のexact/sampled stress実装を`experiment-common`へ移し、既存visualizerを共通実装へ切り替える
- [x] 2.4 8,000頂点の境界、64始点の既定sample、固定stress seed、exact/sample一致を検証するunit testを追加する
- [x] 2.5 stress評価を全速度timer停止後に呼び出し、種別・値・評価時間・sample情報をrecordへ格納する共通APIを実装する

## 3. CPU手法の実験モード

- [x] 3.1 `baseline-sgd-non-gpu`へ`--run-id`、`--input`、`--iterations`、`--epsilon`、`--seed`、`--output-format`、`--output-dir`、`--run-mode`、`--verbose`を追加し、ハードコードされた入力と非決定的乱数を除去する
- [x] 3.2 `baseline-sgd-non-gpu`の処理を共通時間区間へ分割し、`sgd` recordと座標成果物をタイミング後に生成する
- [x] 3.3 `baseline-sparse-sgd-non-gpu`の既存CLIを共通名・既定値へ揃え、`sparse_sgd` recordを生成する
- [x] 3.4 `baseline-sparse-sgd-non-gpu`の前処理、iteration、postprocess境界を再計測し、ファイル保存とstressを時間外へ移す
- [x] 3.5 CPU 2手法のJSONモードでstdoutが1行だけになり、verboseログがstderrだけに出るintegration testを追加する

## 4. GPU手法の実験モード

- [x] 4.1 `vram-lock-native`へ共通CLIとseedを追加し、`atomic_sgd` method IDで共通recordを生成する
- [x] 4.2 `vram-lock-native`のruntime init、upload、iteration、readback、方式固有setupを分離し、GPU完了待ち後にhost timerを停止する
- [x] 4.3 `rr_gpu`へ共通CLIとseedを追加し、`rr_sgd` method IDで共通recordを生成する
- [x] 4.4 `rr_gpu`のschedule構築を`method_setup_time_ms`へ分離し、runtime init、upload、iteration、readbackを共通境界で計測する
- [x] 4.5 `sparse-sgd-gpu`の既存CLIを共通名・既定値へ揃え、`rr_sparse_sgd` method IDで共通recordを生成する
- [x] 4.6 `sparse-sgd-gpu`のschedule、runtime init、upload、iteration、readbackを共通境界へ分割し、iterationにcommand生成・submit・GPU完了待ちを含める
- [x] 4.7 timestamp query対応を検出して`gpu_device_time_ms`を取得し、非対応GPUでは処理を失敗させず`null`を記録する
- [x] 4.8 GPU 3手法のJSONモードでstdoutが1行だけになり、GPU情報・進捗・保存通知がstdoutへ混在しないintegration testを追加する

## 5. 方式固有統計と計測モード

- [x] 5.1 atomicSGDのattempted/completed/retry統計をrecordへ対応付け、追加同期の有無を確認する
- [x] 5.2 RR-SGDとRR-SparseSGDのround数・dispatch数をrecordへ対応付ける
- [x] 5.3 追加同期が必要な統計を`diagnostic` runへ分離し、`benchmark` recordと混在しない検証を追加する
- [x] 5.4 将来の`atomic_sparse_sgd`用method ID、nullable atomic統計、Sparse共通CLIをschema compatibility testへ追加する

## 6. 自動実験ランナー

- [x] 6.1 `experiments/run_experiments.py`へJSON manifest読込、設定検証、条件直積、決定的run ID生成を実装する
- [x] 6.2 5手法のrelease binary解決、共通CLI command生成、逐次subprocess実行、timeoutを実装する
- [x] 6.3 method stdoutの1 JSON行・schema version・method・run ID・時間合計を検証して`results.jsonl`へflush付き追記する
- [x] 6.4 stderrのrun別保存と、非0終了・timeout・不正JSON用の失敗record生成を実装する
- [x] 6.5 manifest snapshot、environment metadata、成果物、stderr、resultsをexperiment ID配下へ保存する
- [x] 6.6 warm-up除外、repetition、fail-fast設定、`--dry-run`、成功run skipによるresumeを実装する
- [x] 6.7 JSONL末尾破損の検出・隔離と、失敗runだけを再試行できる挙動を実装する
- [x] 6.8 小規模fixture manifestで条件列挙、warm-up、失敗継続、再開、重複防止を検証するrunner testを追加する

## 7. 文書とE0検証

- [x] 7.1 `paper/docs/experiment-output-format.md`を実装済みschemaと照合し、field・単位・`null`規則・JSON例を確定する
- [x] 7.2 `paper/docs/experiment-runbook.md`を実装済みrunnerと照合し、build、dry-run、実行、再開、失敗確認、集計前検査を確定する
- [x] 7.3 5手法をrelease buildし、小規模graphで共通JSON contract testを完走させる
- [x] 7.4 USPowerGrid・seed 0〜2のE0 manifestをdry-runし、期待する5手法15条件とcommandを確認する
- [x] 7.5 E0を実行し、同一familyの初期状態、時間合計、stress評価、NaN/Inf、成果物追跡、失敗0件を確認する
- [x] 7.6 `cargo test`、runner test、`openspec validate standardize-experiment-output`を実行し、実験開始条件を満たすことを記録する
