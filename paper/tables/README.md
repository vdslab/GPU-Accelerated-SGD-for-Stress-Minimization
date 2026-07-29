# Tables

論文に採用する表と、その元になる集計済みデータを保存する。表は`experiments/aggregate_results.py`でraw JSONLから生成し、手作業で数値を編集しない。

## Directory構成

1つの表セットを1 directoryへ保存する。

```text
paper/tables/<table-id>/
├── runs.csv
├── speed-summary.csv
├── quality-summary.csv
├── method-stats.csv
├── speed-table.md
├── speed-table.tex
├── quality-table.md
├── quality-table.tex
├── validation-report.json
└── aggregation-metadata.json
```

CSVが丸め前の集計値、Markdownが確認用、LaTeXが論文組込み用である。欠測、OOM、timeoutを0として扱わず、CSVでは空欄、表示表では`—`にする。

## 採用条件

- 正式表は`timing`または`quality` profileで生成する。
- `validation-report.json`の`errors`が空で、`publication_ready=true`であることを確認する。
- `aggregation-metadata.json`にraw JSONL、manifest、environment、集計script、生成fileのchecksumがあることを確認する。
- 表のcaptionまたは本文に単位、反復数、集計量、基準手法を明記する。
- `validation` profileのdraft表を性能比較へ採用しない。

## 論文repositoryへ移すもの

採用するtable directoryを10 fileまとめて移す。少なくとも表示用`.tex`、元となる集計CSV、`validation-report.json`、`aggregation-metadata.json`を分離しない。raw JSONL自体は大容量のため実験保管場所に残し、metadataのchecksumから追跡する。

表の値を修正したい場合はCSVや`.tex`を直接編集せず、raw data、manifest、または集計規則を修正し、新しいtable IDへ再生成する。
