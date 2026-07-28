## Why

実験runnerは各runを再現可能な`results.jsonl`へ保存できるが、そのままでは論文の速度表・品質表へ利用できず、手作業の転記は計算間違い、失敗runの混入、集計条件の不一致を招く。raw JSONLを正本として検証・集計し、CSV、Markdown、LaTeXの表を同じ操作から再生成できる経路が必要である。

## What Changes

- experiment directoryまたは`results.jsonl`を読み込み、schema、実験条件、成功・失敗、重複・再試行recordを検査する集計CLIを追加する
- append順でrunごとの最新recordを選別し、benchmarkとdiagnostic、FullとSparse、exactとsampledを混在させない集計単位を定義する
- 主要速度3指標についてseed・repetitionを集計し、同一familyのCPU基準に対するspeedupを算出する
- stress値と同一seedのpaired stress ratioを集計し、速度表と品質表を分離する
- 生データ抽出CSV、集計CSV、Markdown表、LaTeX表、集計metadataを決定的に生成する
- 欠損run、非有限値、dirty commit、条件不一致、基準run不足などを検出し、不完全な正式表を生成しない
- 集計・表生成・再生成手順を論文用ドキュメントへ追加する

## Capabilities

### New Capabilities

- `experiment-result-aggregation`: 実験JSONLの検証、run選別、統計集計、基準手法比較、論文用CSV・Markdown・LaTeX表の決定的生成を定義する

### Modified Capabilities

なし。

## Impact

- `experiments/`へ集計CLI、統計・表生成処理、fixture、テストを追加する
- `paper/tables/`を生成物の保存先とし、表ごとのsource metadataと生成commandを保存する
- `paper/docs/experiment-runbook.md`と実験出力関連文書へ集計手順を追記する
- Python標準ライブラリを基本とし、外部統計依存を追加する場合は再現可能なversion固定が必要になる
- schema version 1の`results.jsonl`は変更せず、その下流処理として追加する
