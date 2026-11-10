# v5.5 Atomic Sparse：最新main、web-Stanfordまで

**52グラフ・780/780件成功、失敗0。** 2026-11-10実行。最新main `839346091717f3928be83bbcdba8cf8783a48e4e` をrelease/lockedビルドして測定した。実験途中のユーザー指定によりweb-Stanfordで終了し、com-DBLP・web-NotreDame・com-Amazon・coPapersCiteseerの4グラフ60件は開始していない。全56グラフの完了結果ではない。図表生成はユーザー指定で実施していない。

Obsidian「2_速度比較」「11_共通条件と出典」の通常比較条件。seed0〜4×各3回、15反復、epsilon0.1、p=min(200,n)。各グラフseed999のwarmupを1回、52件除外。GPU既定permutation、CPU shuffle無効、CSR CPU BFS auto、GPU timestamp無効。全件逐次実行、時間打切りなし。

## 数値と範囲

| 対応CPUとの比 | 52グラフ等重み幾何平均 |
|---|---:|
| SGD＋readback速度比（CPU/Atomic） | 0.603726 |
| 総計算速度比（CPU/Atomic） | 0.787151 |
| 最終Stress比（Atomic/CPU） | 1.005524 |

各graph/seed/repetitionで比を計算してから、seed内3回中央値→5seed中央値・最小最大で集計。時間比が1超で高速、Stress比が1未満でCPUより低い。主速度指標はGPU完了待ちを含む`sgd_time_ms + readback_time_ms`と`total_compute_time_ms`。GPU pass時間やプロセス全体の時間とは異なる。Stress評価・座標I/O・圧縮は主計時外。同じ最終座標から速度とStressを取得した。

n<=8000は厳密Full Stress、超過は固定256始点・評価seed0の推定Full Stress。CPUは過去v5の対応runであり、同時測定ではない。機材の温度・電源・常駐アプリは統制していない。今回の52グラフと、従来の56グラフ全体の集計を直接同じ対象として比較しない。

## 保存ファイル

- `raw/`：780件のrecord、native stdout、実行コマンド、stderr、資源ログ、可逆圧縮manifest。
- `warmups/`：52件、正式集計から除外。
- `tables/records.json`：今回のrunと対応CPU比。
- `tables/hierarchical.json`、`tables/summary.csv`：52グラフの階層集計。
- `tables/geometric-means.json`：上表の丸め前の数値。
- `GRAPH_TIMES.md`、`tables/graph-times.csv`：v/e、今回・v5・RR Sparseの所要時間。後続4件はexcluded。これは実験全体の所要時間で、SGD性能比ではない。
- `tables/graph-times-source-sha256.json`：v5・RR比較元のrecord SHA256。
- `config/jobs.json`：確定した780件。`original-jobs.json`と`original-plan.json`は変更前の840件を保持。
- `config/datasets.json`：初期に固定した56入力のカタログ。実行範囲はjobsとplanを正とする。
- `config/environment.json`、`logs/build*`、`source-snapshot/`：機材・実装commit・バイナリSHA256・ビルドログ・実行時ソース。
- `validation/independent-check.json`：入力・初期座標・Sparse前処理・Stress評価条件の照合、更新完了、独立した集計再計算の結果。
- `validation/relocation.json`：v5.5への配置変更とartifact参照先・versionラベルだけの変更履歴。測定値とnative stdout・実行時コマンドは変更していない。

圧縮座標（XZ）とreleaseバイナリはリポジトリ既存の`.gitignore`方針に従いローカル保持。座標は圧縮時に展開SHA256・CRCを検証し、manifestをGit管理する。Git上の数値JSON/CSVだけで速度・Stressの表や図のデータを更新できる。座標ファイルの復元にはローカル保存物が必要。

## 再集計

リポジトリ直下から以下を実行する。過去CPU・RRのrecordとローカル座標を必要とする検証もある。初回の実行管理コードはsource-snapshot、v5.5への整理後のコードはexperimentsを参照。

```sh
python3 experiments/report_atomic_sparse_main.py
python3 experiments/verify_atomic_sparse_main.py
python3 experiments/atomic_sparse_graph_times.py
```

実験を再実行する場合は保存済み結果を上書きせず、`run_atomic_sparse_main.py`の別保存先を用い、実装を上記main commitへ固定してrelease/lockedで再ビルドする。原計画と終了範囲・停止理由は`PLAN.md`と`scope-stop-request.json`を参照。
