## 1. runner計画情報の共有

- [x] 1.1 manifest検証、条件直積、決定的run ID生成を`run_experiments.py`からimport可能な共通moduleへ抽出する
- [x] 1.2 既存E0 manifestとfixtureについて、抽出前後でrun ID・command・列挙順が変わらない回帰テストを追加する
- [x] 1.3 manifest展開結果へrepetition番号とpaired比較keyを保持し、FullとSparseのpivots規則をテストする

## 2. experiment入力とrun正規化

- [x] 2.1 `experiments/aggregate_results.py`へ複数`--experiment-dir`、`--profile`、`--output-dir`、`--dry-run`のCLIを追加する
- [x] 2.2 manifest、environment、results JSONLの存在・JSON構文・SHA-256を検証して読み込む処理を実装する
- [x] 2.3 schema version、必須field、有限性、非負性、時間合計、成果物参照を検証するrecord validatorを実装する
- [x] 2.4 manifest計画とrecordのrun ID・method・dataset・seed・パラメータ・run modeを照合し、repetitionを付与する
- [x] 2.5 run IDごとにappend順の最新recordを選び、過去failure・再試行履歴を保持する処理を実装する
- [x] 2.6 壊れたJSON、未知run、条件不一致、failure→success、success→failureを検証するfixture testを追加する

## 3. publication gateと集計group

- [x] 3.1 dataset、input checksum、family、run mode、パラメータ、stress kind、environmentを含む比較group keyを実装する
- [x] 3.2 Fullの`sgd`、Sparseの`sparse_sgd`を基準へ対応付け、benchmark/diagnosticとexact/sampledを分離する
- [x] 3.3 計画runの欠損・失敗、dirty commit、commit・環境・checksum・条件不一致を検出するvalidation reportを実装する
- [x] 3.4 timingの最低10標本、qualityの最低25共通seedとpaired基準runを強制するpublication gateを実装する
- [x] 3.5 validation profileで警告付きdraftを許可し、常に`publication_ready=false`となることをテストする
- [x] 3.6 Full/Sparse、benchmark/diagnostic、exact/sampled、異なる環境が同一分布へ混在しないtestを追加する

## 4. 統計とpaired比較

- [x] 4.1 固定quartile規則と、n・seed数・repetition数・median・Q1・Q3・IQR・mean・sample SDを返す共通統計関数を実装する
- [x] 4.2 3速度指標の手法別分布と、同じseed・repetitionに基づくpaired speedup分布を実装する
- [x] 4.3 同一seedの複数repetitionをmedianへまとめ、stress分布とpaired stress ratioを実装する
- [x] 4.4 attempted/completed/retry、rounds、dispatches、GPU device timeを方式固有の補助集計へ対応付ける
- [x] 4.5 既知値fixtureでquartile、sample SD、speedup、stress ratio、欠損baseline、非有限値を検証するunit testを追加する

## 5. CSV・Markdown・LaTeX表

- [x] 5.1 安定した列順・行順・float表現で`runs.csv`、`speed-summary.csv`、`quality-summary.csv`、`method-stats.csv`を生成する
- [x] 5.2 速度を`median [Q1, Q3]`とpaired speedupで表示する共通table modelを作り、MarkdownとLaTeXへrenderする
- [x] 5.3 stressをmean・sample SDとpaired ratioで表示する品質table modelを作り、MarkdownとLaTeXへrenderする
- [x] 5.4 欠損値、表示名、単位、固定丸め、LaTeX特殊文字escapeを実装する
- [x] 5.5 golden fixtureによりCSV・Markdown・LaTeXの値一致、列順、escapeを検証する

## 6. provenanceと安全な生成

- [x] 6.1 入力checksum、experiment ID、source commit、profile、集計規則version、正規化commandを含む`aggregation-metadata.json`を実装する
- [x] 6.2 warning・error・再試行履歴・publication readinessを含む`validation-report.json`を実装する
- [x] 6.3 全artifactを一時directoryへ生成・検証してから配置し、既存出力を既定で拒否する処理を実装する
- [x] 6.4 現在時刻と出力先依存値を除外し、同じ入力を2回集計した全fileとchecksumがbyte単位で一致するtestを追加する
- [x] 6.5 `--dry-run`がrun数・group・警告・生成予定fileだけを表示し、出力fileを作らないintegration testを追加する

## 7. E0受入と文書化

- [x] 7.1 `paper/docs/experiment-runbook.md`へ集計dry-run、validation、timing、quality、出力確認、再生成のcommandを追記する
- [x] 7.2 `paper/docs/experiment-output-format.md`へraw JSONLと集計CSV・表示表の正本関係、統計量、`null`規則を追記する
- [x] 7.3 `paper/tables/README.md`へtable directory構成、provenance、論文repositoryへ移送するfileと禁止する手作業編集を記載する
- [x] 7.4 E0の15 runをvalidation profileで集計し、5手法、Full/Sparse分離、dirty・標本不足警告、draft表を確認する
- [x] 7.5 fixtureでtiming・qualityのpublication profileを完走し、正式表の全artifactと`publication_ready=true`を確認する
- [x] 7.6 runner test、集計test、`openspec validate aggregate-experiment-results --strict`、決定性再生成を実行して結果を記録する
