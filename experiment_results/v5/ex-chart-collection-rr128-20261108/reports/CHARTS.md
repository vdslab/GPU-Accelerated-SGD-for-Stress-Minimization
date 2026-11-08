---
tags: [area/大学院研究, 論文, チャート]
updated: 2026-11-08
status: 作成完了
---

# EX_チャート集：CPU・Atomic・RR128

[[EX2_速度ストレス実験]] / [[EX2_速度ストレス実験_結果と考察]] / [[9_計測時間]]

## 収録データと読み方

- CPU・Atomic：前回v5通常比較の保存済み2,820実行。レイアウトは再測定していない。
- RR：EX2の1,410実行。FullはT=W128、SparseはW128・p=min(200,n)、15反復、epsilon0.1、seed0〜4×3回。
- Fullは38グラフ、Sparseは56グラフ。各族でCPU・Atomic・RRのグラフ、入力、seed、repetitionを対応させる。初期座標、Sparseの前処理fingerprint一致を確認した。
- 点は同seed3回中央値→5seed中央値。誤差棒は5seed中央値の最小〜最大で、信頼区間ではない。
- CPU＝灰色×、Atomic＝橙△、RR128＝青○。異なるグラフの測定点を線で結ばない。
- CPU速度比・Stress比は対応するrunごとに比を求めてから集計する。FullはCPU Full、SparseはCPU Sparseを分母とする。

## 論文に記載する計測条件差

CPU・Atomicは前回測定、RR128は今回測定であり、同時期の比較ではない。前回GPU timestamp無効、EX2 RRは有効。主速度図は⑤SGD経過時間/15と⑩描画なし総計算時間を使い、純GPU pass時間と混同しない。Stress評価・描画は主速度の計測終了後である。これらの差を図注・実験条件に明記し、速度差の全てをT/W変更だけの効果に帰属しない。

## 品質評価を揃えた方法

Fullは3方式すべて厳密Full Stressで比較する。前回推定値だった4グラフ（ca-HepTh、bcsstk17、PGPgiantcompo、fe_4elt2）のCPU・Atomic計120記録は、保存済み最終座標から共通評価器で厳密に再評価した。同一の座標はキャッシュ再利用し、レイアウト・速度は再実行していない。CPUはf64、Atomicはf32として保存座標を復元し、既存の小グラフ厳密値との一致を確認した。過去recordは書き換えず、このチャート集用の評価オーバーレイを保存した。

Sparseは3方式とも従来どおり、n≤8000は厳密、n>8000は同じ固定256始点・評価seed0の推定Full Stress。Sparseの最適化目的関数値とは区別する。FullとSparseの品質図を、そのまま共通評価精度の比較とは呼ばない。

## 速度比較の概要

![[EX_チャート集_図表/overview-sgd.png]]

![[EX_チャート集_図表/overview-speedup.png]]

## CPU基準の記述的集計

| 族 | 方式 | SGD速度比（幾何平均） | 総計算速度比（幾何平均） | Stress/CPU（幾何平均） |
|---|---|---:|---:|---:|
| full | CPU (previous) | 1.000 | 1.000 | 1.00000 |
| full | Atomic (previous) | 0.173 | 0.159 | 1.17840 |
| full | RR128 (EX2) | 2.193 | 0.748 | 1.04701 |
| sparse | CPU (previous) | 1.000 | 1.000 | 1.00000 |
| sparse | Atomic (previous) | 0.522 | 0.453 | 1.00833 |
| sparse | RR128 (EX2) | 2.176 | 1.141 | 1.01712 |

グラフを等しい重みで要約した記述値であり、有意差検定ではない。FullとSparseの対象集合が異なるため、幾何平均同士を方式間の直接比較に使わない。

## Full：論文用の個別図

### Full — CPU / Atomic / RR128 (38 graphs)

Mean elapsed time per SGD iteration (ms)。低いほど高速。

![[EX_チャート集_図表/full-sgd-time.png]]

### Full — CPU / Atomic / RR128 (38 graphs)

Total compute, input to CPU coordinates (s)。低いほど高速。

![[EX_チャート集_図表/full-total-time.png]]

### Full — CPU / Atomic / RR128 (38 graphs)

Paired CPU / method SGD time。1より大きいほどCPUより高速。

![[EX_チャート集_図表/full-sgd-speedup.png]]

### Full — CPU / Atomic / RR128 (38 graphs)

Paired CPU / method total compute time。1より大きいほどCPUより高速。

![[EX_チャート集_図表/full-total-speedup.png]]

### Full — Final Stress / paired family CPU

Exact Full Stress for all methods。Stress比は1より小さいほどCPUより低い。絶対Stressの大小で異なるグラフ間の品質順位を判断しない。

![[EX_チャート集_図表/full-stress-ratio.png]]

### Full — Final Full Stress (absolute)

Exact Full Stress for all methods。Stress比は1より小さいほどCPUより低い。絶対Stressの大小で異なるグラフ間の品質順位を判断しない。

![[EX_チャート集_図表/full-stress-absolute.png]]

### Full — speed and final quality

Exact Full Stress for all methods。右ほど高速、下ほど低Stress。同じ対応run集合から集計。

![[EX_チャート集_図表/full-speed-stress.png]]


## Sparse：論文用の個別図

### Sparse — CPU / Atomic / RR128 (56 graphs)

Mean elapsed time per SGD iteration (ms)。低いほど高速。

![[EX_チャート集_図表/sparse-sgd-time.png]]

### Sparse — CPU / Atomic / RR128 (56 graphs)

Total compute, input to CPU coordinates (s)。低いほど高速。

![[EX_チャート集_図表/sparse-total-time.png]]

### Sparse — CPU / Atomic / RR128 (56 graphs)

Paired CPU / method SGD time。1より大きいほどCPUより高速。

![[EX_チャート集_図表/sparse-sgd-speedup.png]]

### Sparse — CPU / Atomic / RR128 (56 graphs)

Paired CPU / method total compute time。1より大きいほどCPUより高速。

![[EX_チャート集_図表/sparse-total-speedup.png]]

### Sparse — Final Stress / paired family CPU

Exact n<=8000; same 256 sources (seed0) otherwise。Stress比は1より小さいほどCPUより低い。絶対Stressの大小で異なるグラフ間の品質順位を判断しない。

![[EX_チャート集_図表/sparse-stress-ratio.png]]

### Sparse — Final Full Stress (absolute)

Exact n<=8000; same 256 sources (seed0) otherwise。Stress比は1より小さいほどCPUより低い。絶対Stressの大小で異なるグラフ間の品質順位を判断しない。

![[EX_チャート集_図表/sparse-stress-absolute.png]]

### Sparse — speed and final quality

Exact n<=8000; same 256 sources (seed0) otherwise。右ほど高速、下ほど低Stress。同じ対応run集合から集計。

![[EX_チャート集_図表/sparse-speed-stress.png]]

## ファイルと再現情報

リポジトリ正本：`experiment_results/v5/ex-chart-collection-rr128-20261108/`

- `figures/`：各図のPNG（200 dpi）・PDF（ベクトル）。論文にはPDFを利用可能。
- `tables/paired-runs.csv`：出典と評価方法を含む4,230記録の対応表。`hierarchical.json`：282設定の階層集計。
- `tables/full-exact-stress-overlay.json`：過去の推定Stressを保存座標から厳密に再評価した120記録の対応。
- `exact-stress/cache/`：座標・入力・精度をキーにした厳密再評価結果と出典・ハッシュ。
- `validation/`：保存座標の精度確認、初期状態の対応、旧記録ハッシュ、完了検証。
- `source-snapshot/`：図生成・評価器ソースとハッシュ。

前回v5、EX1、EX2の結果・チャートは保持し、このチャート集を追加保存した。前回には存在しないGPU反復時間・描画時間をCPU/Atomicの図へ捏造して補完しない。


## 128対1024の追加比較

[[EX_チャート集_128と1024比較]]：Full38グラフ、SparseはW1024の比較条件が揃う3グラフ。一覧図とグラフ別の個別図を収録。
