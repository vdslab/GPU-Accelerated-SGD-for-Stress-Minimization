# 実験の自動実行・収集手順

## 位置づけ

このrunbookは、実験条件をmanifestで固定し、5手法を同じ流れで実行して`results.jsonl`へ収集する手順を定める。比較条件は[experiment-plan.md](experiment-plan.md)、時間区間は[measurement-policy.md](measurement-policy.md)、1 runのfieldは[experiment-output-format.md](experiment-output-format.md)を参照する。

ここに記載するrunnerと共通CLIは、OpenSpec change `standardize-experiment-output`の実装予定である。実装完了まではコマンド例を実行できない。

## 基本方針

- 実験条件を手入力のcommand列ではなくJSON manifestで保存する。
- release binaryだけを使用し、正式run中は他の重いCPU/GPU処理を動かさない。
- GPUを含むためrunは逐次実行し、同一GPU上で並列化しない。
- 同じdataset・seed・初期配置・iterations・epsilonをfamily内で揃える。
- Sparse群はpivot、制約、学習率も揃える。
- warm-upは正式runより先に行い、統計へ含めない。
- raw dataはappend-onlyのJSONLとし、失敗も削除せず理由を残す。
- 正式実験はcleanなGit commitから実行する。

## 配置

実装後の自動化コードはrepository rootの`experiments/`、生の実験結果は`output/experiments/`へ置く。

```text
experiments/
├── run_experiments.py
├── schema/
│   └── experiment-record-v1.json
└── manifests/
    ├── e0-uspowergrid.json
    ├── e2-quality.json
    └── e3-timing.json

output/experiments/<experiment-id>/
├── manifest.json
├── environment.json
├── results.jsonl
├── stderr/
│   └── <run-id>.log
└── artifacts/
    └── <run-id>/
        ├── final.txt
        └── vertex-map.txt
```

`output/`は大容量の作業データであり、原則としてGitへcommitしない。論文で採用した集計表と生成情報だけを`paper/tables/`、図と生成情報を`paper/figures/`へ移す。

## manifest

E0で実装済み5手法をseed 0〜2に対して確認する例を示す。条件の配列は直積として展開する。`pivots`はSparse群だけに適用し、Full群のrecordでは`null`にする。

```json
{
  "experiment_id": "e0-uspowergrid",
  "run_mode": "benchmark",
  "methods": [
    "sgd",
    "atomic_sgd",
    "rr_sgd",
    "sparse_sgd",
    "rr_sparse_sgd"
  ],
  "datasets": [
    {
      "name": "USPowerGrid",
      "path": "data/USpowerGrid.mtx",
      "sha256": "取得後に固定する"
    }
  ],
  "seeds": [0, 1, 2],
  "iterations": [15],
  "epsilon": [0.1],
  "pivots": [200],
  "repetitions": 1,
  "warmups": 1,
  "timeout_seconds": 3600,
  "fail_fast": false
}
```

このmanifestの正式runは`5 methods × 1 dataset × 3 seeds = 15 runs`である。加えて15 warm-up runsがあるが、`results.jsonl`の正式統計には含めない。

実時間を最低10回測る場合は`repetitions`を10にする。品質のseed 0〜24は`seeds`へ列挙する。条件を変更したら同じexperiment IDを再利用せず、新しいIDにする。

## 実行手順

### 1. 実験前チェック

```bash
git status --short
git rev-parse HEAD
```

次を確認する。

- 作業ツリーがcleanである。
- 入力datasetのSHA-256がmanifestと一致する。
- 電源、GPU、OS、driver、Rust toolchainが実験中に変わらない。
- timeoutと必要disk容量に余裕がある。
- 前回の別experiment出力dirを上書きしない。

### 2. release build

実装後はrunnerが必要binaryの存在とcommitを検証する。初回は対象5手法をrelease buildする。

```bash
cargo build --release --manifest-path baseline-sgd-non-gpu/Cargo.toml
cargo build --release --manifest-path vram-lock-native/Cargo.toml
cargo build --release --manifest-path rr_gpu/Cargo.toml
cargo build --release --manifest-path baseline-sparse-sgd-non-gpu/Cargo.toml
cargo build --release --manifest-path sparse-sgd-gpu/Cargo.toml
```

### 3. dry-run

```bash
python3 experiments/run_experiments.py \
  experiments/manifests/e0-uspowergrid.json \
  --dry-run
```

確認項目:

- 表示される正式run数と手計算が一致する。
- methodとbinaryの対応が正しい。
- Full群に`--pivots`が付かず、Sparse群だけに付く。
- dataset、seed、iterations、epsilon、output dirが意図どおりである。
- run IDが重複していない。

### 4. 正式実行

```bash
python3 experiments/run_experiments.py \
  experiments/manifests/e0-uspowergrid.json \
  --output-root output/experiments
```

runnerは各条件について次を行う。

1. 必要回数のwarm-upを実行し、結果を統計対象から除外する。
2. 正式run用の決定的run IDと成果物directoryを作る。
3. method binaryを`--output-format json --run-mode benchmark`で逐次実行する。
4. stderrをrun別logへ保存する。
5. stdoutのJSON、schema、条件、時間合計、成果物を検証する。
6. 成功または失敗recordを`results.jsonl`へ追記してflushする。

### 5. 中断後の再開

既存のexperiment directoryへ書く場合、runnerは既定で拒否する。意図した再開であることを確認して`--resume`を付ける。

```bash
python3 experiments/run_experiments.py \
  experiments/manifests/e0-uspowergrid.json \
  --output-root output/experiments \
  --resume
```

resume時はmanifest snapshotと現在のmanifestが一致することを確認する。既存の`success` run IDはskipし、未完了と`failure`だけを再実行する。成功recordを削除して帳尻を合わせない。

## benchmarkとdiagnostic

正式な速度比較は`run_mode: benchmark`だけを使う。

atomicのretry詳細やGPU counterを取るために追加同期・readbackが必要な場合は、manifestを分けて`run_mode: diagnostic`とする。diagnostic runの時間をbenchmarkのspeedupへ混ぜない。追加同期なしで取得できるattempted/completed/retry、round、dispatchはbenchmark recordへ保存してよい。

## 実験後チェック

実行後、少なくとも次を確認する。

1. 期待した正式run数と`success + failure`の一意なrun ID数が一致する。
2. 重複run IDと壊れたJSON行がない。
3. `git_dirty`が全正式runで`false`である。
4. dataset名ごとに`input_sha256`が1つに固定されている。
5. 同一family・dataset・seedの`initial_positions_sha256`が一致する。
6. Sparse群ではpivot、constraint、前処理識別情報がCPU/GPU間で一致する。
7. 時間が非負で、Algorithm/CLI合計が内訳と一致する。
8. GPU benchmarkでIteration timeがqueue完了待ちを含む。
9. stressが有限で、`exact`と`sampled`が混在した集計をしていない。
10. 成功recordが参照する座標成果物が存在する。
11. failure、timeout、OOMを0秒の成功として扱っていない。
12. stderrにpanic、validation error、NaN/Inf、未完了update警告がない。

## E0から本実験への進み方

### E0: 計測系の検証

- dataset: `USPowerGrid`
- methods: 実装済み5手法
- seeds: 0〜2
- iterations: 15
- Sparse pivots: 200
- 正式run数: 15

E0では速度の結論を出さず、共通出力、seed再現性、初期状態、制約、時間境界、stress、成果物の整合性を確認する。AtomicSparseSGD実装後は同じmanifestへ追加し、正式run数を18として再検証する。

### 本実験

E0合格後、[experiment-plan.md](experiment-plan.md)のE1〜E6を別manifestとして固定する。

- 品質: seed 0〜24を基本とし、同じseedのpaired stress ratioを使う。
- 速度: 各条件を最低10 repetitions、median/IQRを主に報告する。
- GPU: 同条件のwarm-upを少なくとも1回行う。
- 主速度指標: Algorithm time cold。
- 補助速度指標: Iteration time、CLI total time cold。
- 品質: exact stressを主とし、大規模sampled stressは別集計にする。

## 論文用データへの移送

raw JSONLは加工せず保管し、集計scriptの入力とする。論文repositoryへ移すのは次に限定する。

- 採用した表のCSV/TSVと、元experiment ID・集計script・commit
- 採用した図と、元JSONLのchecksum・生成command
- 実験環境、dataset checksum、除外runと理由
- 中央値、IQR、paired ratio、信頼区間の計算条件

手作業で表の数値を書き換えず、JSONLから再生成できる状態を保つ。

