# GPU高速化の実験計画

先行研究の条件と根拠は `prior-work-experiments.md` にまとめる。このファイルは本研究で実行する比較条件を固定する。

## 1. 研究上の比較単位

6手法を、同じ目的関数・制約集合を持つ2群に分ける。

### Full Stress群

| ID | 手法 | 役割 | 現在の実装候補 |
|---|---|---|---|
| F-CPU | SGD | CPU逐次基準 | `baseline-sgd-non-gpu/` |
| F-AT | atomicSGD | GPU atomic/lock方式 | `vram-lock-native/`を主候補、`vram-lock/`は旧WGPU版として要整理 |
| F-RR | RR-SGD | GPU round-robin方式 | `rr_gpu/` |

### Sparse Stress群

| ID | 手法 | 役割 | 現在の実装候補 |
|---|---|---|---|
| S-CPU | SparseSGD | CPU逐次基準 | `baseline-sparse-sgd-non-gpu/` |
| S-AT | atomicSparseSGD | GPU atomic/lock方式 | 未実装。新規binary/branchを要固定 |
| S-RR | RR-SparseSGD | GPU round-robin方式 | `sparse-sgd-gpu/` |

`atomicSparseSGD`はこれから実装する。pivot選択、Sparse制約、方向別重み、初期座標、学習率は`SparseSGD`および`RR-SparseSGD`と共通化し、座標更新の競合処理だけをatomic方式へ置き換える。`RR-SparseSGD`は現在の`sparse-sgd-gpu/`が持つone-sided phaseとtwo-sided matching roundsを合わせた1手法として扱う。

主比較は次の4本である。

1. `atomicSGD / SGD`
2. `RR-SGD / SGD`
3. `atomicSparseSGD / SparseSGD`
4. `RR-SparseSGD / SparseSGD`

GPU方式同士の `RR-SGD / atomicSGD` と `RR-SparseSGD / atomicSparseSGD` は、競合処理方式を比較する副比較とする。Full群とSparse群の速度を直接比べる場合は、制約数と近似目的関数が異なることを明記する。

## 2. 仮説と判定

| ID | 仮説 | 主要指標 | 合格条件 |
|---|---|---|---|
| H1 | Full Stress GPU法はSGD品質を保つ | paired full-stress ratio | GPU/CPU平均比の95% CI上限が1.05以下 |
| H2 | Sparse Stress GPU法はSparseSGD品質を保つ | paired full-stress ratio | GPU/CPU平均比の95% CI上限が1.05以下 |
| H3 | GPU法は計算を高速化する | speedup | CPU/GPUのmedianが1より大きく、95% CIが1をまたがない |
| H4 | 規模増加でGPUの利点が増える | 規模別speedup・throughput | 頂点数、辺数、制約数に対する傾向を図示して説明可能 |

Stressは小さいほどよい。比率は同じdataset・seedのCPU結果を分母とする。5%は現在のGPU Sparse SGD仕様でも採用済みの基準であり、最終計画確定時に変更しない。

## 3. 全手法で固定する条件

| 項目 | 条件 |
|---|---|
| 入力 | 同じMatrix Marketファイル |
| 連結成分 | 同じ最大連結成分と同じ頂点再番号付け |
| 初期配置 | 同じseedから生成した1×1正方形内の座標 |
| iterations | 15 |
| epsilon | 0.1 |
| annealing | 先行研究の指数減衰 |
| 次元 | 2 |
| 重み | `w_ij = d_ij^-2` |
| Sparse pivots | 主比較は200 |
| pivot選択 | 同じseed・同じMaxMinRandomSP結果を全Sparse手法で共有 |
| build | `--release`、同一commitの成果物 |
| 可視化 | 計測区間外 |
| full stress | 全手法で同じ評価器 |

時間区間、合計時間、stress比、runごとの保存項目は`measurement-policy.md`の定義に従う。

atomic方式にretry上限や制約skipがある場合は、`attempted_updates, completed_updates, retry_failures`も必須とする。処理できなかった制約を無視したまま高速化として報告しない。

## 4. 時間の定義

速度は次の3指標を報告する。

1. **Iteration time**: 15 iterations全体。
2. **Algorithm time**: 共通前処理、方式固有前処理、GPU初期化、転送、iterations、readback、後処理。
3. **CLI total time**: 入力読込から最終座標取得まで。

主speedupはAlgorithm timeで計算する。RRのschedule生成は方式固有前処理として独立計測し、Algorithm timeとCLI total timeの両方へ含める。詳細な計測境界、cold/warmの区別、除外処理は`measurement-policy.md`を正本とする。

## 5. データセット方針

### 5.1 先行研究の代表8グラフ

Full Stress比較の標準セットとして、Figure 10と同じ次を取得する。

`btree9`, `qh882`, `1138_bus`, `dwt_1005`, `poli`, `dwt_2680`, `USPowerGrid`, `3elt`

このセットは約1000頂点以下から約5000頂点までを含み、木、電力網、mesh、細長いグラフなどの形状差を含む。取得後、頂点数・辺数・SuiteSparse URL・checksumを台帳へ記録する。

### 5.2 現在保有する主要グラフ

| データセット | 行列サイズ | 非ゼロ要素数 | 用途 |
|---|---:|---:|---|
| `USpowerGrid.mtx` | 4,941 | 6,594 | 論文再現、全6手法、pivot感度 |
| `luxembourg_osm.mtx` | 114,599 | 119,666 | 先行研究最大規模のSparse再現 |
| `web-Stanford.mtx` | 281,903 | 2,312,497 | 先行研究を超えるSparseスケーリング（入力は非対称） |

Matrix Marketヘッダの第3値は非ゼロ要素数であり、実装が対称化・自己ループ除去・最大連結成分抽出を行った後の実辺数とは区別してログへ残す。

`web-Stanford`はMatrix Market上で`general`な有向入力であり、先行研究の対称行列とは条件が異なる。無向化を「`(u,v)`または`(v,u)`があれば1辺」と固定し、変換後のchecksumと辺数を保存するまでは、主たる品質比較ではなく追加のスケーリング評価として扱う。

### 5.3 トポロジー選定

最低でも次を含める。

- 小規模: GPU起動コストが支配的になるケース
- mesh/road: Sparse近似が効きやすいケース
- low-diameter/high-degree: Sparse近似が難しいケース
- long/twisted: 局所解へ入りやすいケース
- 大規模疎グラフ: GPU throughputとメモリを評価するケース

`EVA` は先行研究でSparse近似が難しい例なので、取得可能ならpivot感度実験へ加える。

## 6. 実験の実行順

### E0. 計測系の検証

- データ: `USPowerGrid`
- 手法: 6手法
- seed: 0〜2
- 条件: 15 iterations、Sparseは200 pivots
- run数: 18

確認項目:

- 同じseedで初期座標が一致する
- Full/Sparseそれぞれで制約集合がCPU/GPU間で一致する
- 全手法のfull stressを同じ評価器で計算できる
- NaN/Inf、未更新頂点、範囲外書き込みがない
- 時間区間とログ形式が揃っている

E0に合格するまで大規模実験へ進まない。

### E1. 先行研究のSparse SGD再現

- データ: `USPowerGrid`
- 手法: `SparseSGD`
- pivots: 200
- seeds: 0〜24
- iterations: 15
- 期待値: 既存再現結果の平均full stress `726,046.63`
- 先行研究Figure 14の読み取り値: 約`720,000`

既存の25-run結果は利用できるが、最終論文用commit・環境で再測定する。

### E2. 6手法の品質非劣性

- 標準8グラフ
- 6手法
- seeds: 0〜24
- 主条件: 15 iterations、Sparseは200 pivots
- 最大run数: `8 × 6 × 25 = 1,200`

計算資源を抑える場合は、まず全8グラフをseed 0〜9で実行する。その後、代表3グラフ（易しい・難しい・最大）だけseed 10〜24を追加する。

出力:

- datasetごとのfull stress分布
- CPUに対するpaired stress ratio
- 95% bootstrap CI
- 最良レイアウトだけでなく、中央値に最も近いrunの画像

### E3. 実時間と収束

- 標準8グラフ
- 6手法
- 最低10 runs
- GPUは計測前に同条件を1回warm-up
- iterationごとの経過時間とfull stressを取得

同じrunから品質と時間を得られる場合はE2と共有し、重複実行しない。Stress計算の時間はSGD計測から除外する。

出力:

- 先行研究Figure 10に対応するstress対wall-clock time
- Algorithm timeのspeedup
- Compute/前処理/転送/readbackのstacked breakdown
- updates/s または constraints/s

### E4. Sparse pivot感度

- データ: `USPowerGrid`, `EVA`, `3elt`（EVA取得までは代替グラフを明記）
- 手法: `SparseSGD`, `atomicSparseSGD`, `RR-SparseSGD`
- pivots: 10, 50, 200
- seeds: 0〜24を推奨、最低0〜9
- iterations: 15

可能ならCPU `SparseSGD`のみfull stress条件も測る。GPU方式は同じ近似制約を高速化する研究なので、主比較は同じpivot数同士で行う。

出力:

- pivot数対full stress
- pivot数対時間
- pivot数対制約数、round数、dispatch数

### E5. 大規模Sparseスケーリング

- データ: `luxembourg_osm`, `web-Stanford`
- 手法: `SparseSGD`, `atomicSparseSGD`, `RR-SparseSGD`
- pivots: 200
- seeds: 0〜9
- iterations: 15
- run数: `2 × 3 × 10 = 60`

記録:

- 完走可否
- peak CPU RAM / GPU memory
- 前処理、compute、readback、総時間
- constraints/s
- full stress評価に要した時間

`web-Stanford`で全点対full stressが現実的でない場合、品質指標を勝手に変更せず、共通のsampled stressを補助指標として追加し、そのsampling seedとpair数を固定する。

`web-Stanford`の結果は、有向入力を無向化した追加実験として、先行研究準拠の対称グラフ結果と分けて示す。

### E6. ボトルネック分析

- atomic競合回数または再試行数
- atomic方式の完了update率とskip率
- round数・dispatch数
- GPU occupancyを直接取得できない場合のconstraints/s
- graph degree、制約数、pivot数との関係

この実験は「速い/遅い」の結果を、方式上の理由へ結びつけるために行う。

## 7. 統計と図表

### 報告統計

- 先行研究との対応: mean、min、max
- 安定性: median、IQR
- 推論: dataset・seedで対応づけたbootstrap 95% CI
- 速度向上率: `CPU Algorithm time / GPU Algorithm time`
- 品質比: `GPU full stress / CPU full stress`

### 必須図表

1. 6手法の構造と書き込み競合処理の比較表
2. dataset別のfinal full stress分布
3. CPU基準に対するpaired stress ratioと5%境界
4. stress対wall-clock timeの収束曲線
5. dataset規模対speedup
6. 時間内訳
7. pivot数対stress・時間
8. 代表レイアウト

## 8. 公平性チェック

- [ ] 同じ入力グラフと最大連結成分を使用
- [ ] 同じseedの初期座標をファイルまたはhashで照合
- [ ] 同じiterations、epsilon、学習率
- [ ] Sparse手法間でpivotと方向別重みが一致
- [ ] atomic方式の完了update数を記録し、skipを品質・時間とともに報告
- [ ] Full/Sparseの全手法を同じfull stress評価器で評価
- [ ] release build
- [ ] 描画とstress再計算を計測区間から除外
- [ ] GPU warm-up条件を記録
- [ ] 失敗、OOM、timeoutを欠測として明記し、ゼロ扱いしない
- [ ] 結果ファイルにcommitとハードウェア情報を保存

## 9. 実験停止基準

次をすべて満たした時点で、追加実装より論文完成を優先する。

- [ ] E0の計測系検証に合格
- [ ] 4つの主比較について品質非劣性を判定可能
- [ ] 4つの主比較についてAlgorithm timeのspeedupを報告可能
- [ ] 標準8グラフのうち、トポロジーの異なる最低5グラフで結果がある
- [ ] `USPowerGrid`の25-seed再現がある
- [ ] `luxembourg_osm`の大規模Sparse結果がある
- [ ] 少なくとも1つの失敗条件または性能限界を説明できる
- [ ] 全図表をcommit・コマンド・元ログへ追跡できる
