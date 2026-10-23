> この節の初回調査対象は`prebuffer-aborted`の中断計測であり、最終v3の成否件数には含めない。最終測定後に実記録を追加する。

# v3 資源制限の事故調査

## delaunay_n14：Full方式が8 GiBのRSS制限へ到達

確定済みのrecord.jsonがある実行だけを調査した。計測・実装変更・再実行は行っていない。対象グラフは16,384頂点・49,122辺。制限はプロセス群RSS 8 GiB（8,589,934,592 bytes）。

| 実行 | 結果 | 全体秒 | peak RSS (GiB) | traceの最後 |
|---|---|---:|---:|---|
| Full CPU warmup | success | 42.862 | 7.882 | 15 |
| Full Atomic warmup | memory_limit | 4.296 | 8.321 | traceなし |
| Full CPU seed 0 | memory_limit | 3.418 | 8.230 | 0 |
| Full CPU seed 1 | memory_limit | 3.401 | 8.342 | traceなし |
| Full CPU seed 2 | memory_limit | 3.413 | 8.136 | traceなし |
| Full CPU seed 3 | memory_limit | 3.427 | 8.050 | traceなし |
| Full CPU seed 4 | memory_limit | 3.633 | 8.195 | 0 |

### 停止した段階

- Full CPUのtrace生成は、前処理完了後、execute_sgd内で初期座標と全ペア配列をcloneした後に始まる。seed 0・4は反復0だけ記録され、1反復目の完了記録がない。初回反復の開始前後で停止しており、shuffle中か実座標更新中かは既存の記録だけでは分からない。更新が0件だったとも断定できない。
- trace自体がないFull CPU実行は、少なくとも反復ループの計測開始前で停止した。前処理中と全ペアclone中のどちらかは、詳細な段階ログがないため特定できない。
- Full Atomic warmupはtraceがない。trace作成はCPU前処理・GPU用配列準備・Metal buffer作成の後、最初のdispatchより前である。したがって、この停止時点でSGD kernelの反復は始まっていない。CPU前処理/GPU用配列/buffer準備のどの確保で制限を越えたかは未特定。
- 成功したFull CPU warmupは反復0〜15が揃い、SGD時間38.343秒、全体42.862秒だった。同じ設定がメモリ境界の近くにあり、warmupの成功だけでは本測定の成功を保証しなかった。

### ソースから確認できるメモリ負荷

- Fullは全点対を保持する。16,384頂点ではペア数134,209,536。CPUの1ペアはusize×2とf64×2で32 bytesとなり、配列本体だけで約4 GiBになる。
- CPU前処理はn×nのusize距離行列（約2 GiB）とペア配列を作る。execute_sgdは所有する元ペア配列に加えて、そのcloneを作るため、ソース上で約4 GiBずつの配列2本が存在する。容量の余裕やその他のメモリも加わる。
- Full Atomicも全ペアをGPU用へ変換する。GPU用ペア配列の要素は16 bytes（約2 GiB）。さらに更新順ID約0.5 GiBと更新前4座標約2 GiBのhost配列を準備し、それぞれのMetal bufferを作る。大きな配列・bufferの準備だけで強いメモリ負荷がある。
- これらはソースから求めた配列サイズであり、そのままRSSの合計ではない。Vecの容量、変換時のallocation再利用、解放、ページの常駐状態と監視頻度で実測RSSは変わる。今回の記録だけでは、その内訳までは確定できない。

### 解釈

今回確定しているのは、runnerが観測RSSによって明示的に停止したこと。OSのメモリ枯渇や、Atomicのlock待ちによる停止が確認された事例ではない。Full CPUは第1反復完了前、Full Atomicはdispatch前であり、この失敗に最終stressやSGD完了時間を割り当てない。成功したFull RRと比較する際も、対応するCPU seedが欠測なので、そのseedの速度比は計算できない。

参照：
- [Full CPU前処理](/Users/kose/master/atomic-scheduling-investigation/implementations/full/baseline-sgd-non-gpu/src/graph.rs)、[全ペアcloneとtrace開始](/Users/kose/master/atomic-scheduling-investigation/implementations/full/baseline-sgd-non-gpu/src/algorithm.rs)
- [Full Atomic配列・buffer・trace・dispatchの順序](/Users/kose/master/atomic-scheduling-investigation/implementations/full/vram-lock-native/src/metal.rs)
- [制限による停止のrunner](/Users/kose/master/atomic-scheduling-investigation/experiments/run_v3.py)

調査に使った確定record：
- [warmup-delaunay_n14-sgd-pNone-tNone](/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/warmups/warmup-delaunay_n14-sgd-pNone-tNone/record.json)
- [warmup-delaunay_n14-atomic_sgd-pNone-tNone](/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/warmups/warmup-delaunay_n14-atomic_sgd-pNone-tNone/record.json)
- [v3-basic-delaunay_n14-sgd-s0-pNone-tNone](/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/raw/artifacts/v3-basic-delaunay_n14-sgd-s0-pNone-tNone/record.json)
- [v3-basic-delaunay_n14-sgd-s1-pNone-tNone](/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/raw/artifacts/v3-basic-delaunay_n14-sgd-s1-pNone-tNone/record.json)
- [v3-basic-delaunay_n14-sgd-s2-pNone-tNone](/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/raw/artifacts/v3-basic-delaunay_n14-sgd-s2-pNone-tNone/record.json)
- [v3-basic-delaunay_n14-sgd-s3-pNone-tNone](/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/raw/artifacts/v3-basic-delaunay_n14-sgd-s3-pNone-tNone/record.json)
- [v3-basic-delaunay_n14-sgd-s4-pNone-tNone](/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/raw/artifacts/v3-basic-delaunay_n14-sgd-s4-pNone-tNone/record.json)
