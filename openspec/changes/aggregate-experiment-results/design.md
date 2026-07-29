## Context

`standardize-experiment-output`により、各methodはschema version 1の1行JSONを返し、runnerはmanifest、environment、append-onlyの`results.jsonl`、stderr、座標成果物をexperiment ID単位で保存する。現在はraw dataから論文表への変換処理がなく、集計条件、失敗除外、基準手法、丸めを手作業で決める余地が残っている。

集計対象には、再開によって同じrun IDのfailureとsuccessが複数行存在し得る。またrecord自体にはrepetition番号がないため、manifestとrunnerの決定的run展開を使わなければpaired比較を安全に構成できない。正式な速度値はbenchmarkだけ、FullとSparseは別目的関数、exactとsampled stressは別推定量という制約もある。

## Goals / Non-Goals

**Goals:**

- raw JSONLを正本として、検証から論文用table生成までを1 commandで再現する
- seed・repetitionを復元し、同じ条件のCPU基準とpaired比較する
- 速度、品質、方式固有統計を分離し、集計規則とprovenanceをartifactへ固定する
- publication用の不完全・不正な入力をfail closedで拒否する
- Python標準ライブラリだけでCIと研究環境の双方から実行できるようにする

**Non-Goals:**

- raw JSONL、manifest、座標成果物を変更または削除すること
- 実験runnerやmethod binaryを集計処理から再実行すること
- グラフ・誤差棒・stress推移などのfigure生成
- 統計的有意差検定や論文の主張を自動決定すること
- 異なるhardware、入力checksum、family、stress kindを補正して直接比較すること

## Decisions

### 1. 単一の標準ライブラリCLIと内部moduleにする

入口を`experiments/aggregate_results.py`とし、JSONL読込、manifest照合、正規化、統計、render、provenanceをテスト可能な内部関数へ分ける。CLIは1つ以上の`--experiment-dir`、`--profile timing|quality|validation`、`--output-dir`、`--dry-run`を受け取る。

Pandas notebookは探索には便利だが、依存versionと手作業状態が表へ影響しやすい。正式表はPython標準ライブラリの`json`、`csv`、`statistics`、`hashlib`、`decimal`を使い、探索用notebookは生成CSVを読む下流利用に限定する。

### 2. manifestを再展開してrepetitionを復元する

runnerの条件展開とrun ID生成を共有moduleへ切り出すか、同一の純粋関数をimport可能にする。集計器はmanifest snapshotを再展開し、`run_id -> dataset, method, seed, repetition, iterations, epsilon, pivots, run_mode`の計画表を作る。record内の条件と照合し、run IDに対応しない行や条件差を拒否する。

run ID文字列のhash部分を逆解析する案は採用しない。format変更に弱く、repetitionを確実に復元できないためである。

### 3. runの現在状態はappend順の最後のrecordとする

JSONLを先頭から読み、run IDごとに最後に現れたrecordをcurrent recordとする。履歴は別に保持し、failure回数、再試行回数、失敗段階をvalidation reportへ出す。最後がfailureなら過去にsuccessがあっても失敗である。

「最後のsuccessを探す」方式は後続失敗を隠すため採用しない。

### 4. profileでpublication gateを固定する

- `timing`: benchmarkのみ。比較groupの全計画run成功、clean commit、同一環境を要求し、各method・条件で最低10標本を要求する
- `quality`: 同じstress kindの共通seedを要求し、各method・条件で最低25 seedを要求する
- `validation`: E0や開発中の確認用。dirty、標本不足、失敗をreportへ出し、可能な範囲のdraft表を生成するが`publication_ready=false`とする

publication profileに汎用`--ignore-errors`を置かない。例外を許すと採用条件がcommand履歴に埋もれるため、除外が必要な場合は将来、理由付き除外manifestを別仕様として追加する。

### 5. group keyと基準手法を固定する

group keyはexperiment、dataset、input SHA-256、family、run mode、iterations、epsilon、pivots、stress kind、environment fingerprintを含む。Fullの基準は`sgd`、Sparseの基準は`sparse_sgd`とする。commitやenvironmentが異なるrecordは同じ比較groupに入れず、publication profileでは不一致をエラーにする。

複数datasetは1つの表に行として並べられるが、統計はdatasetをまたいでpoolしない。

### 6. 統計量とquartile定義をversion化する

速度3指標は全formal benchmark標本について`n`、seed数、repetition数、median、Q1、Q3、IQR、mean、sample SDを保存する。主要表は`median [Q1, Q3]`を表示する。quartileは端点を含む線形補間法を実装し、規則versionとテストfixtureを固定する。

speedupは同じseed・repetitionの`baseline_time / method_time`を先に求め、そのratio分布を集計する。ratio of mediansは補助列にできるが、paired speedupの代替にはしない。

品質はseedを統計単位とする。同一seedに複数repetitionがある場合はmethod・seed内のstress medianを代表値とし、stressのmean、sample SD、median、quartileを算出する。paired stress ratioは`method_seed_stress / baseline_seed_stress`であり、1に近いほど基準と同等、1未満ほど低stressである。

### 7. 機械可読CSVを正本にして表示形式を派生する

出力directoryには少なくとも次を生成する。

```text
runs.csv
speed-summary.csv
quality-summary.csv
method-stats.csv
speed-table.md
speed-table.tex
quality-table.md
quality-table.tex
validation-report.json
aggregation-metadata.json
```

CSVは丸め前の十分な精度を持つ値を保存する。MarkdownとLaTeXは同じformat関数から生成し、速度、speedup、stressごとの固定桁数を適用する。欠損値はCSVでは空欄、表示表では`—`とし、0へ変換しない。

### 8. byte再現性とprovenanceを優先する

入力と行・列を安定sortし、UTF-8、LF、固定JSON key順、固定float表現で書き出す。現在時刻、絶対出力path、一時directory名はartifactへ入れない。metadataには入力file checksum、experiment ID、record schema version、集計規則version、profile、source commit、正規化した生成command、出力checksumを保存する。

metadata自身のchecksumをmetadataへ再帰的に含めず、その他のartifactだけを列挙する。既存出力は既定で拒否し、一時directoryへ全fileを書いて検証後に配置することで部分生成を避ける。

## Risks / Trade-offs

- [runnerの条件展開と集計器がずれる] → run展開をimport可能な共有moduleへ抽出し、同じmanifestからrun IDとrepetitionが一致するcontract testを置く
- [小標本でquartileやsample SDが不安定] → publication profileの最低標本数を強制し、validationでは`n`と警告を必ず表示する
- [異なるexperimentを結合して環境差を混ぜる] → environment fingerprintとcommitをgroup key・publication gateへ含める
- [丸め後のMarkdownとLaTeXが食い違う] → 両形式を同じ中間table modelとformat関数から生成しgolden testで照合する
- [再試行履歴を重複標本として数える] → run IDごとの最新recordだけを統計対象にし、履歴はreport専用にする
- [表の列数が増えて論文幅を超える] → 主要表は3速度指標とspeedup、品質表はstressとratioに限定し、全統計はCSVへ残す

## Migration Plan

1. runnerのmanifest展開・run ID生成を共有可能なmoduleへ整理し、既存run IDが変わらない回帰テストを追加する
2. 集計CLIの検証・正規化・統計処理をfixtureで実装する
3. CSV、Markdown、LaTeX、metadata生成とgolden testを追加する
4. 既存E0を`validation` profileで集計し、15 run、5 method、警告内容、決定性を確認する
5. runbookへdry-run、生成、検査、論文repositoryへの移送手順を追記する

集計は下流追加でraw dataを変更しないため、rollbackは集計scriptと生成tableを削除するだけでよい。

## Open Questions

- LaTeX表を最終論文templateの`booktabs`前提にするか、依存しない標準`tabular`にするかは実装時に論文repositoryのpreambleを確認して決める
- 将来、95% confidence intervalやbootstrapを主要値へ加える場合は、集計規則versionを上げて既存表と区別する
