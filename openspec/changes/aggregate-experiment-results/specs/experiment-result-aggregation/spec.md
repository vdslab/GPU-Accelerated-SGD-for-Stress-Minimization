## ADDED Requirements

### Requirement: raw experiment recordの検証と正規化

集計システムは、1つ以上のexperiment directoryからmanifest snapshot、environment metadata、`results.jsonl`を読み込み、schema version 1、JSON構文、run ID、manifest条件、必須field、有限性、時間合計を検証しなければならない（MUST）。manifestの決定的な条件展開を再現し、各recordへseedとrepetitionを含む計画上の条件を対応付けなければならない（MUST）。

#### Scenario: 正常なexperiment directoryを読み込む

- **WHEN** manifestから計画された全runを含む有効なexperiment directoryを指定する
- **THEN** 各run IDを計画条件へ一意に対応付け、型付きの正規化rowとして読み込む

#### Scenario: JSONまたはmanifest対応が壊れている

- **WHEN** JSONLに壊れた行、未知のschema version、manifestに存在しないrun ID、またはrecordと計画条件の不一致がある
- **THEN** 該当箇所と理由を報告し、正式表を生成しない

### Requirement: append-only履歴からのrun選別

集計システムは、同じrun IDのrecordが複数ある場合にJSONLのappend順で最新recordだけを現在状態として採用し、過去の失敗と再試行履歴を検証reportへ残さなければならない（MUST）。最新recordが失敗または欠損であるrunを成功標本として扱ってはならない（MUST NOT）。

#### Scenario: 失敗後の再開に成功する

- **WHEN** 同じrun IDについてfailure recordの後にsuccess recordが追記されている
- **THEN** 最新のsuccessを集計候補とし、過去のfailure件数と失敗段階を検証reportへ記録する

#### Scenario: 最新の試行が失敗している

- **WHEN** 同じrun IDの過去にsuccessがあっても最新recordがfailureである
- **THEN** そのrunを失敗として扱い、過去のsuccessで隠蔽しない

### Requirement: publication用の集計前ゲート

集計CLIは`timing`、`quality`、`validation` profileを提供しなければならない（MUST）。`timing`と`quality`では、計画runの欠損・失敗が0件、`git_dirty=false`、比較group内のcommit・入力checksum・実行環境・共通パラメータが一致し、必要標本数を満たす場合だけ`publication_ready=true`の表を生成しなければならない（MUST）。`validation`は不足条件を明示したdraft出力を許可してもよい（MAY）が、論文採用可能と表示してはならない（MUST NOT）。

#### Scenario: timing表の反復数が不足する

- **WHEN** 比較対象のいずれかが10回未満の正式benchmark標本しか持たない
- **THEN** timing profileを失敗させ、不足しているdataset・method・条件と期待数・実数を報告する

#### Scenario: quality表のseed数が不足する

- **WHEN** 比較対象のいずれかが25個未満の共通seedしか持たない
- **THEN** quality profileを失敗させ、paired比較に不足するseedを報告する

#### Scenario: E0をvalidationとして集計する

- **WHEN** 3 seedのdirty worktree由来E0をvalidation profileで指定する
- **THEN** draft表と検証reportを生成できる
- **AND** metadataへ`publication_ready=false`と全警告を保存する

### Requirement: 比較可能な集計groupの分離

集計システムは、dataset、input SHA-256、family、run mode、iterations、epsilon、pivots、stress kind、実行環境を集計keyへ含めなければならない（MUST）。`benchmark`と`diagnostic`、FullとSparse、exactとsampled、異なる入力またはパラメータを同じ分布へ混在させてはならない（MUST NOT）。

#### Scenario: FullとSparseのrecordが同じ入力に存在する

- **WHEN** 1つのexperimentにFull 3手法とSparse 2手法が含まれる
- **THEN** Fullは`sgd`、Sparseは`sparse_sgd`を基準とする別groupへ分離する

#### Scenario: exactとsampled stressが混在する

- **WHEN** 同じmethodにexact recordとsampled recordが存在する
- **THEN** 品質分布とpaired ratioをstress kind別に生成し、両者を1つの統計量へ集約しない

### Requirement: 速度統計とpaired speedup

集計システムは、`iteration_time_ms`、`algorithm_time_cold_ms`、`cli_total_time_cold_ms`の各指標について、標本数、seed数、repetition数、median、Q1、Q3、IQR、mean、sample standard deviationを計算しなければならない（MUST）。主要表示はmedianとIQRとし、同じdataset・family・seed・repetition・条件にあるCPU基準と対象手法を対応付け、`baseline_time / method_time`としてpaired speedupを計算しなければならない（MUST）。

#### Scenario: Full timingを集計する

- **WHEN** `sgd`、`atomic_sgd`、`rr_sgd`が同一条件のseed・repetitionをすべて持つ
- **THEN** 3速度指標の分布と、`sgd`を1.0とする各手法のpaired speedup分布を生成する

#### Scenario: 対応する基準runがない

- **WHEN** 対象手法のseed・repetitionに対応するCPU基準runが存在しない
- **THEN** その比較を黙ってunpaired比へ置換せず、publication profileを失敗させる

### Requirement: stress品質統計とpaired stress ratio

集計システムは、stressについて標本数、seed数、mean、sample standard deviation、median、Q1、Q3、IQRを計算しなければならない（MUST）。同じfamily・dataset・seed・stress kind・条件のCPU基準に対して`method_stress / baseline_stress`を計算し、同じseedに複数repetitionがある場合はseed内代表値を固定した規則で算出してからpaired ratioを集計しなければならない（MUST）。

#### Scenario: Full品質をseed対応で比較する

- **WHEN** 25個の共通seedについてFull 3手法のexact stressが存在する
- **THEN** 手法別stressのmeanとsample standard deviation、および`sgd`基準のpaired stress ratioを生成する

#### Scenario: stressが非有限または0以下である

- **WHEN** 成功recordのstressがNaN、Inf、またはratioを定義できない値である
- **THEN** 該当runを除外して続行せず、入力検証エラーとして正式表を生成しない

### Requirement: 論文用table artifactの生成

集計システムは、正規化済みrunのCSV、速度集計CSV、品質集計CSV、方式固有統計CSV、検証report、生成metadataを出力しなければならない（MUST）。同じ集計値から、人間確認用Markdownと論文組込み用LaTeXの速度表・品質表を別々に生成しなければならない（MUST）。

#### Scenario: timingとquality artifactを生成する

- **WHEN** 有効なexperimentを集計する
- **THEN** `runs.csv`、`speed-summary.csv`、`quality-summary.csv`、`method-stats.csv`、`validation-report.json`、`aggregation-metadata.json`を生成する
- **AND** 速度と品質それぞれの`.md`と`.tex`を同じ数値・丸め規則で生成する

#### Scenario: LaTeX特殊文字を含むdataset名を出力する

- **WHEN** datasetまたはmethod表示名にLaTeX特殊文字が含まれる
- **THEN** CSVの値を変更せず、LaTeX表では安全にescapeした表示名を使用する

### Requirement: 決定的な生成とprovenance

集計出力は、同じ入力bytes、集計profile、設定、集計実装から再生成した場合にbyte単位で一致しなければならない（MUST）。metadataには入力JSONL・manifest・environmentのSHA-256、experiment ID、Git commit、集計profile、集計規則version、生成command、各出力fileのSHA-256を含めなければならない（MUST）。現在時刻など再生成ごとに変わる値を決定的artifactへ含めてはならない（MUST NOT）。

#### Scenario: 同じ入力を2回集計する

- **WHEN** 同じ入力と引数を別directoryへ2回出力する
- **THEN** provenance内で出力先に依存する部分を除く全生成fileの内容とSHA-256が一致する

#### Scenario: raw JSONLを変更する

- **WHEN** 1つのrecord値を変更して再集計する
- **THEN** 入力checksumと影響する集計artifactのchecksumが変化し、元入力との対応を追跡できる

### Requirement: 集計CLI・文書・自動検証

システムは、experiment directory、profile、出力directoryを明示する非対話CLIを提供し、入力検証だけを行うdry-runと既存出力の意図しない上書き防止を実装しなければならない（MUST）。fixtureとgolden fileにより、統計量、paired比較、失敗ゲート、CSV、Markdown、LaTeX、決定性を自動検証し、runbookへ実行・確認・論文repository移送手順を記載しなければならない（MUST）。

#### Scenario: 集計をdry-runする

- **WHEN** 有効なexperiment directoryと`--dry-run`を指定する
- **THEN** 読み込むrun数、group、警告、生成予定fileを表示し、table fileを作成しない

#### Scenario: 既存table directoryへ出力する

- **WHEN** 生成済みfileを含む出力directoryを上書き許可なしで指定する
- **THEN** 既存fileを変更せず、明示的なエラーを返す

