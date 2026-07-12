## Context

`sparse-sgd-gpu/src/gpu.rs` の `GpuContext::execute` は、学習率配列を走査するループ内で各iterationのGPU command encoderを作成・submitしている。現在はループ外で総iteration数を表示し、ループ終了後に合計compute時間を報告するだけである。CPU版のSparse SGDは同じ処理段階で `Iteration: N` を出力しているため、GPU版にも同じ進捗観測点が必要である。

## Goals / Non-Goals

**Goals:**

- GPU計算が開始した各iterationを、1始まりの番号で標準出力へ表示する。
- 表示を実際のGPU command submit処理と同じループへ置き、表示と計算進行の順序を一致させる。
- 既存のGPU計算、乱数、iteration数、Timing集計、結果ファイルを変更しない。

**Non-Goals:**

- iterationごとのGPU処理時間測定やETA表示を追加しない。
- shader、スケジュール、ラウンド数、CLI引数を変更しない。
- ログをファイルへ保存したり、構造化ログ形式へ変更したりしない。

## Decisions

- `for (iteration, &eta) in params.etas.iter().enumerate()` のループ先頭で `println!("Iteration: {}", iteration + 1);` を実行する。CPU版と同じ文字列形式にすることで、既存利用者の目視確認とログ検索を統一する。
- 表示はuniform生成やcommand encoder作成より前に置く。これにより、GPUが該当iterationのdispatch準備へ入ったことを示す。iteration処理後に表示する案では、GPU停止時に最後に開始したiterationが分からないため採用しない。
- テストはGPUを必須にせず、進捗行を生成する小さな純粋関数またはログフォーマット関数を検証する。GPU実行の既存テストは計算結果の回帰に集中させる。

## Risks / Trade-offs

- [大量iterationでは標準出力行が増える] → 既存CPU版と同じ1行/iterationに限定し、追加の詳細ログは出さない。
- [標準出力のバッファリングにより表示が遅れる環境がある] → 行末改行付きの`println!`を使う。明示flushは今回の範囲外とする。
- [ログを厳密に解析する利用者がいる] → 既存の開始・終了ログを変更せず、追加行の形式を固定する。

## Migration Plan

1. GPU iterationループへ表示を追加する。
2. CPUテストまたは実GPUの小規模実行で、`Iteration: 1`から最終番号までが順序どおり出ることを確認する。
3. 既存の`cargo test`、`cargo clippy`、GPU回帰実行を行う。ロールバックは追加したprintln行を削除するだけで可能である。

## Open Questions

- なし。iteration番号だけをCPU版と同じ形式で表示する方針で確定している。
