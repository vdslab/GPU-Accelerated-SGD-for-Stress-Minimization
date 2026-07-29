# 研究課題・仮説・貢献

## 研究課題

Stress minimization に基づくグラフレイアウトにおいて、先行研究の SGD と同等の最終 stress を保ちながら、競合処理または競合回避を用いた GPU 並列化によって計算時間を短縮できるか。

## 背景にある問題

ZhengらのSGDは、stress majorizationより少ない反復で低いstressへ到達し、Sparse Stress近似との組み合わせにより10万頂点規模にも適用できる。一方、各制約を逐次更新するSGDの計算時間は依然として大きい。単純なGPU並列化では、複数の制約が同じ頂点座標を同時更新する書き込み競合が発生する。本研究では、atomic更新と競合しないround-robin型スケジュールを比較し、解品質を損なわずに高速化できる条件を明らかにする。

## 仮説

- H1: `atomicSGD` と `RR-SGD` は、CPU `SGD` に対する最終full stressの悪化を5%以内に抑えながら、SGD計算時間を短縮できる。
- H2: `atomicSparseSGD` と `RR-SparseSGD` は、CPU `SparseSGD` に対する最終full stressの悪化を5%以内に抑えながら、Sparse SGD計算時間を短縮できる。
- H3: GPU法の速度向上率はグラフおよび制約数が大きくなるほど高くなるが、atomic競合またはround数・dispatch数による方式ごとのボトルネックが現れる。

各仮説は `docs/experiment-plan.md` の評価指標と図表で検証できる形にします。

## 予定する貢献

1. Full Stress SGDとSparse SGDのそれぞれにおける、GPU atomic方式とround-robin方式の実装。
2. Full StressとSparse Stressの両方について、CPU基準と同じ入力・初期配置・学習率条件で比較できる再現可能な評価系。
3. atomic方式とround-robin方式の、stress・実行時間・スケーラビリティに関する実験的知見。

## 新規性

ZhengらはSGDおよびSparse SGDの解品質とCPU上の実行時間を評価した。本研究は同じstress最小化モデルをGPU上で並列実行し、書き込み競合の処理方式を比較しながら、先行研究相当のstressを保てるかを定量評価する点が異なる。

## 対象範囲

### 対象

- 無向グラフの2次元レイアウト
- Zhengらの15-iteration exponential annealing schedule
- Full Stress SGDと、200 pivotsを中心とするSparse SGD
- Full/SparseそれぞれにおけるCPU逐次、GPU atomic、GPU round-robinの比較

### 対象外

- stress majorizationとの再比較（先行研究でSGDとの比較が行われているため）
- レイアウトの美的品質に関する大規模な主観評価
- GPUごとの移植性を主題とする評価

## 研究を終える条件

- [ ] 仮説ごとの判定基準が数値で決まっている
- [ ] 必須データセットで比較実験が完了している
- [ ] 性能だけでなく解品質も検証できている
- [ ] 制約と失敗条件を説明できている
