## ADDED Requirements

### Requirement: 比較手法の共通実験インターフェース
システムは、実装済みの`sgd`、`atomic_sgd`、`rr_sgd`、`sparse_sgd`、`rr_sparse_sgd`の各手法に、少なくとも入力、反復数、epsilon、seed、出力形式、成果物出力先を指定できる共通の実験用CLIを提供しなければならない（MUST）。Sparse群はpivot数も指定できなければならない（MUST）。

#### Scenario: 同じFull Stress条件を指定する
- **WHEN** Full Stress群の3手法へ同じ入力、反復数、epsilon、seedを指定する
- **THEN** 全手法が指定値を使用し、同じfamilyの初期座標を生成する
- **AND** 実行recordのパラメータと初期座標hashが一致する

#### Scenario: 同じSparse Stress条件を指定する
- **WHEN** Sparse Stress群の2手法へ同じ入力、pivot数、反復数、epsilon、seedを指定する
- **THEN** 両手法が同じpivot列、制約集合、学習率、初期座標を使用する
- **AND** 実行recordのパラメータと初期座標hashが一致する

#### Scenario: 将来AtomicSparseSGDを追加する
- **WHEN** AtomicSparseSGD実装が`atomic_sparse_sgd`として比較対象へ追加される
- **THEN** 既存のschemaを変更せず、Sparse群の共通CLIとrecordを実装できる

### Requirement: 実験モードの機械可読な標準出力
各手法は、`--output-format json`を指定した成功runで、schema version付きの有効なJSONオブジェクトをstdoutへちょうど1行だけ出力しなければならない（MUST）。進捗、警告、GPU情報、保存通知、iterationログをstdoutへ混在させてはならない（MUST NOT）。

#### Scenario: JSON実験を成功させる
- **WHEN** 有効な条件で手法を`--output-format json`により実行する
- **THEN** stdoutは1行のJSONとして解析できる
- **AND** JSON以外の文字列や空行を含まない

#### Scenario: 詳細ログを要求する
- **WHEN** `--output-format json --verbose`を指定する
- **THEN** 成功recordだけをstdoutへ出力する
- **AND** 任意の進捗・診断ログはstderrへ出力する

#### Scenario: JSON実験が失敗する
- **WHEN** method processが入力不正、GPU初期化失敗、OOMまたは計算失敗で非0終了する
- **THEN** stdoutへ成功recordを出力しない
- **AND** runnerが失敗理由とstderr log pathを含む失敗recordを保存する

### Requirement: version付き共通実験record
実験recordは、schema version、run ID、run mode、status、method、family、dataset識別子とchecksum、commit、seed、グラフ規模、実行パラメータ、実行環境、時間内訳、ストレス、方式固有統計、成果物パスを共通のfield名と単位で保持しなければならない（MUST）。該当しない値および取得不能な値は0や空文字ではなくJSONの`null`としなければならない（MUST）。

#### Scenario: CPU SGDのrecordを生成する
- **WHEN** `sgd`が正常終了する
- **THEN** GPU device、upload、device timestamp、readback、pivot、round、dispatchの該当しないfieldを`null`で記録する
- **AND** method、family、seed、グラフ規模、時間、stress、成果物を同じschemaで記録する

#### Scenario: RR-SparseSGDのrecordを生成する
- **WHEN** `rr_sparse_sgd`が正常終了する
- **THEN** pivot数、constraint数、schedule round数、dispatch数、GPU情報、全時間区間を共通schemaで記録する

#### Scenario: schemaの意味を変更する
- **WHEN** fieldの単位、意味、必須性に後方互換でない変更を加える
- **THEN** `schema_version`を更新する
- **AND** runnerは未知のschema versionを受理せず失敗recordとして扱う

### Requirement: 比較可能な3つの時間指標
システムは、全手法について`iteration_time_ms`、`algorithm_time_cold_ms`、`cli_total_time_cold_ms`を記録しなければならない（MUST）。各処理は`input`、`common_preprocess`、`method_setup`、`runtime_init`、`upload`、`iteration`、`readback`、`postprocess`の高々1区間へ属し、結果ファイル保存、ストレス評価、描画、検証、ログ出力をこれらの時間へ含めてはならない（MUST NOT）。

#### Scenario: cold Algorithm timeを計算する
- **WHEN** 1 runの全時間区間が確定する
- **THEN** `algorithm_time_cold_ms`をcommon preprocess、method setup、runtime init、upload、iteration、readback、postprocessの和として計算する
- **AND** CPU手法の該当しないGPU区間は合計上0コストとして扱い、record上は`null`を維持する

#### Scenario: CLI total timeを計算する
- **WHEN** `algorithm_time_cold_ms`と`input_time_ms`が確定する
- **THEN** `cli_total_time_cold_ms`を両者の和として計算する

#### Scenario: RR scheduleを構築する
- **WHEN** RR手法がiteration前にscheduleまたはmatching roundsを生成する
- **THEN** その時間を`method_setup_time_ms`へ記録する
- **AND** Algorithm timeとCLI total timeへ含める

#### Scenario: GPU iterationを計測する
- **WHEN** GPU手法がiteration commandをsubmitする
- **THEN** 対象queue処理の完了を待ってからhost側の`iteration_time_ms`計測を終了する
- **AND** command生成、submit、完了待ちをIteration timeへ含める

#### Scenario: device timestampを取得できない
- **WHEN** GPUが必要なtimestamp queryをサポートしない
- **THEN** host側Iteration timeを通常どおり記録する
- **AND** `gpu_device_time_ms`を`null`として記録する

### Requirement: タイミング外の共通ストレス評価
システムは、全手法の最終座標を同じストレス評価実装へ渡し、速度計測終了後に品質を評価しなければならない（MUST）。評価recordは値、`exact`または`sampled`の種別、評価時間、標本数、stress seedを区別して保持しなければならない（MUST）。

#### Scenario: 8,000頂点以下を評価する
- **WHEN** 最終座標を持つグラフが8,000頂点以下である
- **THEN** 厳密ストレスを計算し、`stress_kind`を`exact`として記録する
- **AND** stress評価時間をAlgorithm timeとCLI total timeへ含めない

#### Scenario: 8,000頂点を超えるグラフを評価する
- **WHEN** 最終座標を持つグラフが8,000頂点を超える
- **THEN** 固定した標本数とstress seedで標本化推定を計算する
- **AND** `stress_kind`を`sampled`として標本数とstress seedを記録する
- **AND** その値をfull stressとして表示または集計しない

#### Scenario: visualizerとmethodの評価を比較する
- **WHEN** 同じグラフ、座標、評価mode、標本数、stress seedを使用する
- **THEN** visualizerとmethod recordのstress値が浮動小数点許容誤差内で一致する

### Requirement: 実験条件と成果物の追跡可能性
成功recordは、入力ファイルのSHA-256、Git commit、実行binary、dataset、採用後の頂点数・辺数、constraint数、seed、パラメータ、CPU/GPU/backend情報、初期座標hash、最終座標成果物パスを記録しなければならない（MUST）。

#### Scenario: 同名datasetの内容が変わる
- **WHEN** dataset名が同じで入力ファイル内容が異なる2 runを収集する
- **THEN** 異なる入力SHA-256を記録し、同一条件として集約しない

#### Scenario: 座標成果物を保存する
- **WHEN** method runが正常終了する
- **THEN** タイミング停止後に最終座標と必要なvertex mapを保存する
- **AND** recordから成果物を一意に参照できる

### Requirement: manifestによる再開可能な自動実験
実験ランナーは、JSON manifestで指定したmethod、dataset、seed、iteration、epsilon、pivot、repetitionの条件を決定的に列挙し、release binaryを逐次実行して、各正式runをappend-onlyの`results.jsonl`へ保存しなければならない（MUST）。

#### Scenario: 実行前に条件を確認する
- **WHEN** 利用者がmanifestを`--dry-run`で指定する
- **THEN** runnerは実行順、run数、run ID、commandを表示する
- **AND** method binaryを実行せずresultsを変更しない

#### Scenario: warm-up後に正式runを実行する
- **WHEN** manifestが1回以上のwarm-upを指定する
- **THEN** runnerは同条件のwarm-upを正式runより先に実行する
- **AND** warm-up結果を`results.jsonl`の統計対象recordへ保存しない

#### Scenario: 中断した実験を再開する
- **WHEN** 同じmanifestの実験を既存出力dirに対して再開する
- **THEN** runnerは決定的なrun IDに対応する成功recordをskipする
- **AND** 未完了または失敗したrunだけを実行する

#### Scenario: 1 runが失敗する
- **WHEN** あるmethod processが非0終了、timeout、または不正なstdoutを返す
- **THEN** runnerは失敗を0秒の成功として扱わない
- **AND** status、exit code、失敗段階、stderr log pathを持つrecordを追記する
- **AND** manifestの停止方針がfail-fastでない限り次のrunを続行する

### Requirement: benchmarkとdiagnosticの分離
システムは、正式な速度値を取得する`benchmark` runと、追加同期を伴い得る詳細統計用`diagnostic` runをrecord上で区別しなければならない（MUST）。異なるrun modeの時間を同一の速度分布へ混在させてはならない（MUST NOT）。

#### Scenario: 更新統計に追加同期が不要である
- **WHEN** attempted updates、completed updates、retry failures、rounds、dispatchesを計測区間へ追加同期せず取得できる
- **THEN** benchmark recordへ該当統計を保存する

#### Scenario: 更新統計に追加readbackが必要である
- **WHEN** 詳細統計の取得がiteration内の同期またはreadbackを増やす
- **THEN** `run_mode`を`diagnostic`として別runを実行する
- **AND** その時間をbenchmark speedupへ使用しない

### Requirement: 共通契約の自動検証
システムは、5手法のJSON出力、必須field、`null`規則、時間合計、seed再現性、ストレス一致、runnerの条件列挙・失敗・再開を自動テストしなければならない（MUST）。

#### Scenario: 5手法の出力contract testを実行する
- **WHEN** 小規模fixtureに対して5手法の実験モードを実行する
- **THEN** 各stdoutが同じschemaの1 JSON行として検証される
- **AND** 時間合計が許容誤差内で内訳と一致する
- **AND** stdoutに進捗文字列を含まない

#### Scenario: 同一seedを再実行する
- **WHEN** 同じfamily、dataset、seed、パラメータを2回実行する
- **THEN** 初期座標hashと前処理識別情報が一致する

#### Scenario: runnerの再開テストを実行する
- **WHEN** 複数条件のfixture実験を途中失敗させて再開する
- **THEN** 既存成功runを重複実行せず、残りのrunだけを完了する
