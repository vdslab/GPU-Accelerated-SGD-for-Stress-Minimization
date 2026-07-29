## Context

`sparse-sgd-gpu/src/schedule.rs` は two-sided 制約をseed付き順序でfirst-fit edge coloringし、競合しないmatching roundへ分割する。現在は生成中に、制約値を複製した `graph_pairs`、全roundの `Vec<Vec<EdgeInfo>>`、全roundの使用頂点を表す `Vec<HashSet<usize>>` を同時に保持し、検証後にもう一度平坦な `Vec<GpuPair>` へコピーしている。

web-Stanfordの最大連結成分は255,265頂点・1,941,926辺、最大次数38,625である。pivot 1の実測では38,625 roundを生成し、スケジューリングに約261.5秒、プロセス全体のピーク私有メモリに約2.44GBを要した。BFSの追加メモリは約4MiBであり、今回の対象はtwo-sided schedule生成時の辺コピーとround管理である。

GPU実行はすでに平坦な `two_sided` と `round_offsets` を全iterationで再利用している。この形式と既存のmatching semanticsを維持し、そこへ到達するまでの一時構造だけを置き換える。

## Goals / Non-Goals

**Goals:**

- Bound WGPU/Vulkan command-recording memory by limiting each command encoder and submission to 256 compute passes.
- Preserve the exact shuffled invocation order and in-place update visibility across submission boundaries.

- 全roundの辺コンテナと頂点 `HashSet` を同時に保持せず、two-sided scheduleを構築する。
- 各制約を一度だけroundへ割り当て、round ID、件数、offsetを使う2-pass処理で平坦なGPU配列を生成する。
- 同一入力・pivot・seedに対するfirst-fit round割当、round内の無競合性、制約の完全性、round上限、再現性を維持する。
- 完成したscheduleを全annealing iterationで再利用し、roundごとの再生成・再アップロードを避ける。
- 高次数グラフに対する補助メモリを、roundごとのハッシュテーブル群ではなく、頂点・制約・round数に比例する連続配列中心の構成へ制限する。

**Non-Goals:**

- Reducing the number of matching rounds or compute dispatches. Batching changes submission grouping only.

- 最大次数に由来する38,625 roundやdispatch数そのものを削減すること。
- two-sided in-place更新をatomic、Jacobi、頂点所有方式などへ変更すること。
- dense one-sided `n × h` 配列、pivot距離、CPU Sparse制約生成を最適化すること。
- WGSL、CLI、結果ファイル形式、stress品質契約を変更すること。

## Decisions

### 1. 既存round走査と同じfirst-fit結果を、頂点ごとの使用roundから求める

pivot–pivot制約は従来のcircle methodでbase round IDを決める。各頂点について、すでに使用したround IDのリストを保持する。seedでshuffleした各graph edge `(u, v)` に対し、`u` と `v` の使用roundをgeneration mark配列へ印付けし、最小の未使用roundを選択する。

これは既存実装の「round 0から順に、両端点が未使用の最初のroundを選ぶ」判定と等価である。roundごとの全頂点集合を保持する代わりに、各割当は両端点の使用roundリストへ1回ずつ追加されるため、round-membership値の総数はtwo-sided制約数の2倍に制限される。

graph edgeのshuffle対象は `EdgeInfo` のコピーではなく、`prepared.pairs` を参照するindex列とする。これにより約194万件の40-byte制約コピーをindexへ置き換える。

代替案として、roundを1つ作るたびに全未処理辺を再走査する方式を検討した。この方式はround状態を最小化できるが、各iterationでの再利用には別のmembership保存が必要であり、高次数グラフで `rounds × remaining_edges` の走査を招くため採用しない。

### 2. round IDから2-passで平坦なScheduleを構築する

割当段階では各two-sided制約について `u32` round IDを保持する。全割当後に次の処理を行う。

1. round IDごとの件数を数える。
2. checked prefix sumで既存形式の `round_offsets` を作る。
3. roundごとのwrite cursorを使い、正確な長さで確保した `Vec<GpuPair>` へ各制約をscatterする。
4. 一時index、round ID、cursorを破棄する。

round内の順序はseed付き入力順に対して安定にし、同一seedでバイト単位に再現可能とする。`usize`から`u32`への変換、prefix sum、round数はすべてchecked conversionとし、GPU表現を超える入力ではGPU初期化・出力前に明示的なエラーを返す。

代替案としてroundごとにGPUへアップロードして即時破棄する方式を検討した。この方式はGPU pair bufferを小さくできるが、38,625 round × 15 iterationsでCPU–GPU転送が増え、完成scheduleを再利用できないため採用しない。

### 3. production検証もflat schedule上で有界に行う

既存の `validate_rounds` は全roundのネスト配列と全pairの `HashSet` を要求する。新しい検証は、flat `two_sided` と `round_offsets` を走査し、頂点ごとのgeneration mark配列で各round内の重複端点を検出する。入力分類数と出力件数を照合し、欠落・重複のないことを構築上の一対一対応と件数で確認する。

小規模テストでは集合比較も使用し、one-sidedとtwo-sidedの和が元制約集合と一致することを独立に検証する。高次数のstar graphテストではround数が最大次数に一致し、補助membership数がtwo-sided制約数の2倍であることを確認する。

### 4. GPU側と公開結果形式は変更しない

`Schedule.one_sided`、`Schedule.two_sided`、`Schedule.round_offsets`、round統計は維持する。`GpuContext::execute` は従来通り、1回のone-sided dispatchとshuffleしたtwo-sided round dispatchを1 command encoderへ記録する。したがって、この変更はCPU schedule生成のメモリ表現に限定される。

### 5. Bound command recording with ordered submission batches

`GpuContext::execute` builds the same invocation list as before and partitions it into contiguous chunks of at most 256 invocations. Each chunk is recorded into a fresh command encoder, submitted immediately, and completed with `device.poll(wait_indefinitely())` before the next chunk is recorded.

Dynamic uniform offsets use the invocation's global slot index, not its chunk-local index. Queue ordering and the completion boundary therefore preserve the existing in-place position-update semantics. The flat schedule and GPU buffers remain resident and reusable; no pair data is uploaded per round.

## Risks / Trade-offs

- [Many batches add submission and synchronization overhead] → Keep the limit at 256 passes, measure web-Stanford compute time, and leave graphs with 256 or fewer dispatches on the single-submission path.
- [A chunk-local dynamic offset could select the wrong uniforms] → Enumerate each chunk with its global start offset and test contiguous batch ranges at 256/257 and web-Stanford-scale boundaries.

- [頂点ごとの使用round走査は高次数頂点で依然として高コスト] → 今回はメモリ改善を優先し、既存first-fitと同等の結果を維持する。時間とdispatch数の改善は別変更で扱う。
- [頂点ごとの小さい `Vec<u32>` が多数のallocationを生む] → グラフ次数からcapacityを事前確保し、値を`u32`へ圧縮する。実測で必要なら将来CSR型arenaへ置換できる内部境界を設ける。
- [scatterにより入力index・round ID・最終pair配列が一時的に重なる] → いずれも制約数に線形であり、全roundの辺コピーとHashSet群を保持しない。phase終了直後に明示的にdropできる構造にする。
- [first-fit結果またはround内順序が変わりstress再現性へ影響する] → 同じpivot base round、seed付きedge順、最小使用可能round、stable scatterをテストし、既存実装とのgolden比較を小規模グラフで行う。
- [generation counterのoverflow] → checked incrementを使用し、overflow時はmark配列をclearして安全にepochを再開する。

## Migration Plan

1. 現行スケジューラをテスト用referenceとして小規模グラフで利用できる状態を保つ。
2. compact round assignmentとflat schedule構築を内部helperとして追加する。
3. 同一seedでreferenceとcompactの分類、round ID、offset、pair順を比較する。
4. `build_schedule` をcompact実装へ切り替え、既存GPU・stressテストを実行する。
5. USPowerGridとweb-Stanfordでround統計、生成時間、ピークメモリを記録する。

問題が発生した場合は `build_schedule` の呼出先をreference実装へ戻せるよう、GPU側の `Schedule` 契約を変更しない。

## Open Questions

- web-Stanfordでのピークメモリ合格値はOS・GPU driver・allocatorに依存するため、絶対RSSではなく、scheduler固有の補助要素数と実測前後差の両方を記録する。
- round数とdispatch数の根本削減は、two-sided更新方式を変更する別提案で扱う。
