## ADDED Requirements

### Requirement: Bounded GPU command batching

The system MUST record no more than 256 Sparse SGD compute passes in one command encoder and submission. It MUST submit contiguous batches in the exact generated invocation order and MUST complete each submitted batch before recording the next batch. The system MUST reuse the resident flat schedule and GPU buffers and MUST NOT upload pair data per round.

#### Scenario: Invocation count exceeds the batch limit

- **WHEN** an iteration contains more than 256 one-sided and two-sided invocations
- **THEN** the system partitions them into contiguous batches of at most 256, submits every invocation exactly once, and preserves global dynamic-uniform slot offsets

#### Scenario: Invocation count is within the batch limit

- **WHEN** an iteration contains 256 or fewer invocations
- **THEN** the system records and submits the iteration in one batch

#### Scenario: Batches preserve in-place update semantics

- **WHEN** an iteration crosses one or more submission boundaries
- **THEN** every later batch observes position updates completed by all earlier batches, and seed-derived invocation ordering and dispatch count remain unchanged

### Requirement: コンパクトなtwo-sided schedule生成

システムは、全matching roundの辺コンテナおよびround別の使用頂点HashSetを同時に保持せず、各two-sided制約へ決定的なround IDを1回だけ割り当て、件数集計とprefix sumによる2-pass処理で平坦なGPU pair配列とround offsetを構築しなければならない（MUST）。スケジューラ固有のround-membership値はtwo-sided制約数に対して線形でなければならない（MUST）。

#### Scenario: 高次数グラフをスケジュールする

- **WHEN** 最大次数が大きく多数のmatching roundを必要とするグラフをスケジュールするとき
- **THEN** 全roundの `Vec<Vec<EdgeInfo>>` と `Vec<HashSet<vertex>>` を構築せず、制約index、制約ごとのround ID、頂点ごとの使用round、round件数・offsetを使ってscheduleを完成する

#### Scenario: flat GPU scheduleを構築する

- **WHEN** 全two-sided制約のround ID割当が完了したとき
- **THEN** checked prefix sumとstable scatterにより各制約をちょうど1回だけ平坦な `two_sided` 配列へ格納し、`round_offsets` の各範囲を対応するmatchingとして公開する

### Requirement: compact scheduleの競合回避と再現性

コンパクトスケジューラは、既存のpivot circle method、seed付きgraph edge順、最小の利用可能roundを選ぶfirst-fit規則、round上限を維持し、各round内で同じ頂点を高々1回だけ使用しなければならない（MUST）。完成したscheduleは全annealing iterationで再利用し、iterationごとまたはroundごとに再生成してはならない（MUST NOT）。

#### Scenario: 同一seedでscheduleを再生成する

- **WHEN** 同じグラフ、Sparse制約、pivot列、seedからscheduleを2回生成したとき
- **THEN** one-sided配列、two-sided pair順、round offset、base round数、spill round数が一致する

#### Scenario: round内競合と制約完全性を検証する

- **WHEN** compact schedule生成が完了したとき
- **THEN** 各two-sided制約がちょうど1回だけ存在し、各roundの全制約について端点集合が互いに素であり、one-sided件数とtwo-sided件数の和が入力Sparse制約数と一致する

#### Scenario: 高次数star graphをスケジュールする

- **WHEN** 中心頂点の次数が `Δ` のstar graphを1 pivotでスケジュールしたとき
- **THEN** graph edgeは `Δ` 個の競合しないtwo-sided roundへ配置され、roundごとの頂点集合を全round分保持しない

#### Scenario: 複数iterationでscheduleを利用する

- **WHEN** compact scheduleを複数annealing iterationのGPU実行へ渡したとき
- **THEN** 同じflat pair bufferとround offsetを再利用し、各iterationでは既存どおりround順だけをseedからshuffleする
