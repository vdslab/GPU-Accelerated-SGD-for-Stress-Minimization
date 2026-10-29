# RR Sparse SGD: f32・同じ15反復での品質改善

## 採用した変更

CLIと単発実験の標準を、頂点ごとに独立したpivot順序＋4分割の交互更新へ変更した。
全グラフで同じ処理を使う。GPU座標・演算は単一のf32、移動係数の上限は従来の1のまま。
反復数・学習率・重み・初期座標の生成・RR matchingは維持し、追加反復やseedの選別は行わない。
4段Feistel置換とcycle walkingにより、任意のpivot数で各制約を1回ずつ処理する。
グラフ名、次数、測定stressによる切り替えはない。

## 品質検証

追加の[CPUシャッフル調査](../rr-cpu-shuffle-20261030/README.md)では、EVA seed 1が256始点評価と厳密評価で順位逆転し、seed 11は厳密評価でもGPUが良いことを確認した。以下の40条件の集計は256始点の推定値であり、厳密評価の集計ではない。

8グラフ×4seed（0, 1, 3, 7）で比較し、候補を`shuffle / 4@0`に固定してから未使用seed 11を検証した。
合計40条件でGPU/CPU stress比の中央値は **1.00154（CPU比＋0.154%）**。
未使用seed 11の8条件は最大でもCPU比＋2.15%で、すべて従来RRより改善した。
全40条件の最大差はCPU比＋6.39%、従来RR比では最大＋3.95%の悪化が残る。
完全一致や全入力での改善は主張しない。従来RRのCPU比最大悪化は同じ40条件で＋45.68%だった。

各行は5seedをまとめた結果。評価は固定256始点、評価seed 0のsampled full stress。
CPU・GPUの初期座標と前処理SHAを照合した。小規模グラフでもsampledなので、CLIのauto exactとは区別する。

| グラフ | GPU/CPU stress比の中央値 | CPU比の最大値 | 新/旧RR stress比の中央値 |
| --- | ---: | ---: | ---: |
| EVA | 1.00731 | 1.06241 | 0.96449 |
| USpowerGrid | 1.00095 | 1.00544 | 1.00148 |
| luxembourg_osm | 0.99976 | 1.00213 | 0.99884 |
| preferentialAttachment | 1.01258 | 1.03469 | 0.86465 |
| smallworld | 1.00870 | 1.01100 | 0.84664 |
| coAuthorsCiteseer | 1.00641 | 1.02145 | 0.85271 |
| web-NotreDame | 1.00617 | 1.04404 | 0.74162 |
| web-Stanford | 0.98675 | 1.06386 | 0.86892 |

## Stanford seed 0の追加評価

同じ4,096始点（評価seed 20261027）で、CPUは **5,043,851,029.634**、新GPUは **5,210,893,656.347**。
CPUとの差は **＋3.31%**。従来GPUの **＋25.53%** から縮小した。
対応のある差は167,042,626.713、始点抽出に対する近似95%区間は[93,209,194.973, 240,876,058.452]。
これは固定した配置の始点抽出に関する区間であり、最適化seed間のばらつきや未検証グラフの保証ではない。
CPUと同値とはいえず、約3.3%の差は残っている。

ブラウザと同じ64始点では、CPU 5,092,620,939.590、旧GPU 6,212,335,104.265、
新GPU **4,924,176,872.803**。256始点では新GPU/CPU=1.06386となるため、
64始点の数字だけでCPUより良いと結論しない。

## 大規模グラフの反復時間

Apple M4 Pro / Metal。同じ初期座標・前処理・15反復・epsilon 0.1で比較した。
各方式を同じ初期状態から3回実行した中央値。各再実行の最終座標の一致を確認した。
他のベンチマークと重ねず順に実行し、入力・前処理・GPU初期化・画像生成・stress評価を除く。
CPUの計測区間もSGD反復のみ。下表のstressは固定256始点で、Stanfordの4,096始点評価とは別。

| グラフ | 頂点数 | pivots / seed | GPU 15反復 | CPU 15反復 | CPU/GPU時間 | stressのCPU比差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coAuthorsCiteseer | 227,320 | 200 / 0 | 0.270 s | 13.904 s | 51.4倍 | +1.43% |
| web-NotreDame | 325,729 | 200 / 3 | 0.677 s | 20.689 s | 30.5倍 | +4.40% |
| web-Stanford | 255,265 | 200 / 0 | 1.560 s | 16.001 s | 10.3倍 | +6.39% |
| delaunay_n20 | 1,048,576 | 64 / 0 | 0.285 s | 23.988 s | 84.2倍 | -1.35% |

100万ノードのdelaunay_n20はpivot数64の追加検証であり、上記40条件には含めない。
この時間は処理全体の速度倍率ではない。生の3回分の時間、stress、SHAとコマンドは
`isolated-timing/`に保存した。

## 再現と記録

CLI標準の結果には`rr_pivot_order=per-vertex-feistel4-cyclewalk`、`rr_phase_chunks=min(4,pivots)`、
`rr_step_cap=1`、`rr_arithmetic=f32`を保存する。`quality_search`の`shuffle / 4@0`が対応する。
従来RRは`standard / 1@0`で再生できる。`quality_probe`は旧循環順序の対照実験として残している。

単発実験と同じ`experiments/run_single.py`でStanford / seed 0 / 200 pivots / 15反復 /
2048×2048を実行し、成功・画像2枚・方式メタデータ・stress **4,924,176,872.802872**を確認した。
出力は`output/web-Stanford/2026-10-30-04-30-13_rr_sparse_sgd/`。
初回確認で新規メタデータが厳密なschemaに拒否されたため、共通validator・JSON schema・v2 runnerも
対応した。旧形式も受け入れ、新規項目の欠落・別手法への混入・不正な型や値は拒否する。
成功記録は`single-run-verification.json`に保存した。

Rust releaseの全64テスト、all-targets clippy (`-D warnings`)、fmtチェックが通過。
GPUの実行時入力を使った精度補償テスト、置換の重複・欠落、境界guardも確認した。
実験ランナーのPython全103テストも通過した（`python3 -m unittest discover -s experiments/tests -q`）。

採用しなかった精度補償、上限変更、中心移動、他の相分割の結果も以下に残す。

---

# 診断実験の詳細 (2026-10-30)

The user requires the same iteration count as CPU. Every quality comparison in
the diagnostic search uses 15 iterations, epsilon 0.1, 200 pivots, fixed initial positions,
and the same evaluator/source seed (0). No extra refinement or best-of-seed
selection is used. The diagnostic CLI supports a general iteration argument,
but no increased-iteration experiment is part of this investigation.

## Arithmetic experiments

- `compensated`: f32 high/low coordinates, compensated additions, f32 distances.
- `extended`: also uses double-single distance and displacement operations.
- `coefficients`: additionally retains the f32 residuals of the original f64
  weights and learning rates. GPU arithmetic still uses only f32. Preprocessing
  and final stress evaluation remain the existing CPU f64 implementations.
- `cap=VALUE`: original single-f32 coordinates/arithmetic, with a graph-independent
  upper bound on the directional step mu. This changes the clipping constant,
  and must not be described as a numerically equivalent implementation.

The initial unrounded arithmetic experiments failed a runtime GPU numerical
probe: the algebraic residuals were optimized away on this Metal path. A test
using only literal constants had missed this because it was constant-foldable.
The final tests load inputs from storage buffers. The shaders now force rounding
boundaries through a runtime all-ones integer mask. Both retention of a unit
below f32 spacing and the product residual are checked on the actual GPU.

`stanford-s0.jsonl`, `stanford-s0-extended.jsonl`, and
`stanford-s0-coefficients.jsonl` are preliminary, unrounded results; they are NOT
valid evidence that the intended compensated arithmetic was executed. Results
whose filename contains `rounded` and the later `screen-*` use the corrected
rounding boundaries. These are precision experiments, not native GPU f64 or a
claim of equivalence to f64.

The corrected 256-source Stanford seed 0 results are:

| Variant | Stress | Iteration seconds |
| --- | ---: | ---: |
| Addition compensation | 6245951240.779 | 1.596675 |
| Distance compensation | 5809508322.644 | 1.851439 |
| Distance and coefficient compensation | 7045411545.523 | 2.342507 |

Original RR stress is 6,286,796,052.030 at 256 sources; CPU stress is
5,103,171,456.527. These results do not establish CPU-quality equivalence for
compensated arithmetic. More preserved digits did not monotonically improve
layout quality.

## Common step cap and schedule experiments

`screen-*` uses 64 evaluation sources. It includes the same schedule variants as
the previous investigation: each constraint is visited exactly once per iteration,
and two-sided rounds remain disjoint matchings. A cap of 0.75 and multiple phase
orders were screened on Stanford seeds 0/3, Luxembourg seed 1, EVA seed 3,
NotreDame seed 3, and coAuthorsCiteseer seed 0. These are exploratory results,
not a held-out quality claim.

`cap09-validation/` compares a fixed 0.9 cap and original RR order against an
independent CPU run and the archived original RR results. Every row checks
initial-position and preprocessing hashes. Times are single observations, not
repeated warmed performance measurements. `comparison.csv` retains every case,
including regressions; a process exit of zero only means the experiment completed.

## Reproduction

From the repository root:

```sh
cargo build --release --locked --manifest-path implementations/sparse/sparse-sgd-gpu/Cargo.toml --example quality_search
# Corrected arithmetic experiment (same 15 iterations):
implementations/sparse/sparse-sgd-gpu/target/release/examples/quality_search data/web-Stanford.mtx 0 '1@0' 256 200 extended cpu
python3 tools/diagnostics/rr-common/validate_f32_arithmetic.py --output /tmp/rr-cap09-validation
```

The validation runner records the executable SHA-256 and command for each run.
It resumes only complete matching runs. Archived original RR records come from
`../rr-f32-quality-20261030/validation/` and use the same 256 evaluation sources.

## Additional order / translation diagnostics

`recenter` translates all positions by a GPU-computed mean after each iteration.
It leaves pair distances invariant in exact arithmetic; it changes f32 rounding.
A separate scratch vector follows the two guard vectors. A runtime test covers
257 vertices (a non-multiple of the workgroup size), exact translation, and guards.

`shuffle` replaces cyclic shifts of the common pivot order with independent
per-vertex permutations. It uses a four-round Feistel permutation and cycle walking
on the smallest power-of-four domain covering the pivot count. Every pivot is
visited exactly once, including when the phase is split into chunks. RR matchings,
weights, clipping at one, learning rates, iterations, and f32 coordinate storage
are unchanged. Runtime GPU tests check coverage at pivot counts 1, 2, 3, 199, 200,
256, and 257 with two keys each. This is an order change, not higher precision.

`explore-*` contains exploratory 64-source results for recentering and shuffling.
`shuffle-safety/` checks known adverse cases with 256 sources. The subsequent
`shuffle-validation/` and `cap075-interleaved-validation/` each test both schedules
on all 32 graph/seed conditions, using the CPU results in `cap09-validation/`.
`heldout-reference/` prepares original RR and CPU results for seed 11, which is
reserved for validating the selected candidate rather than selecting it.

Some exploratory and validation runs overlap other quality experiments. Their
timing columns are NOT isolated performance benchmarks and must not be used to
claim speedups. Final speed comparisons require separate isolated runs.
The temporary `_quality_explore` executable wraps the same `quality_search` source
so current experiments can keep their original executable while a new candidate
is built. Recorded executable hashes distinguish these runs. Reproduction can
use the final `quality_search` binary with the documented arithmetic mode.
