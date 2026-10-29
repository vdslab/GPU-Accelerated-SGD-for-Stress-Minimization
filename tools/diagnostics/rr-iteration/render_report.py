#!/usr/bin/env python3
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); rows=json.loads((p/'summary.json').read_text())
checks=json.loads((p/'verification.json').read_text()); sparse_extra=json.loads((p/'additional-regression.json').read_text());full_extra=json.loads((p/'full-additional-regression.json').read_text()); million=json.loads((p/'million-p200-exact.json').read_text()); assert checks['pairs']==16 and len(sparse_extra)==15 and len(full_extra)==6 and million['exact_match']
s='''# RR Full / Sparse 反復高速化と大規模CPU比較（2026-10-30）

## 変更と品質

FullはラウンドごとのGPU完了待ちを反復末尾に集約。queue writeとsubmitの順序は維持した。
Sparseは最大256 dispatchを同じcompute passに記録し、反復末尾で完了待ちする。
従来の「256個のpass/command buffer」から「1 pass内に256 dispatch」への変更。
更新式・乱数消費・シャッフル順序・pivot数・反復数・float精度・既定設定は変えていない。
グラフ名・次数などによる手法の切り替えはない。WGSLの変更は今回追加していない。

'''
s+=f'大規模CPU比較 {checks["pairs"]} 条件に加え、Sparse {len(sparse_extra)} 条件、Full {len(full_extra)} 条件で変更前後の**最終座標とstressが完全一致**した。\n'
s+=f"さらに100万頂点・P=200も前回版と座標が完全一致し、15反復は {million['iteration_time_ms']/1000:.4f} 秒。CPUとの時間比較はP=64の行を参照。\n"
s+='''Sparseは初期座標・前処理SHAも一致。Fullは初期座標SHA・入力SHAも一致。
Full追加検証は奇数タイル、端数ブロック、W<T、shared/globalメモリを含む。
CPUとRRの最終座標一致を要求したものではない。**既存のCPUに対するstress差は残る。**

wgpu 27.0.1のローカル実装も確認した。Queue::write_bufferは次のsubmitの直前に転送されるため、
Fullはsubmitをラウンド単位で維持する。Sparseのdispatchはwgpu-coreのcompute::dispatchから
flush_states→drain_barriersを通り、storage依存関係の同期を行う。passを分けること自体は更新順序の要件ではない。

## 時間比較

Apple M4 Pro（CPU 14 cores / GPU 20 cores）、24 GiB、Metal、release build。
CPU参照実装は単一スレッドの既存SGD。GPUと同じ入力・初期配置・15反復・epsilon=0.1。
FullはT=1024, W=1024, shared（現行CLI/v3の既定。v2のW=256とは異なる）。
SparseはP=200、100万頂点のCPU比較はP=64で両者を揃えた。
通常はseed 0,1,3の各1測定、100万頂点はseed 0の1測定。ベンチマークプロセスは順次実行した。
同一seedの多数反復や統計的な信頼区間を求めた実験ではない。

**下表の時間は15反復の合計秒、複数seedでは中央値。前処理・GPU初期化・転送・読み戻し・stress評価・保存はGPUの反復時間から除外。**
各反復末尾でGPU完了を待つため、CPUからの投入時間だけを測った数字ではない。
CPUは参照関数execute_sgdのwall time（Fullでは内部のpair clone、Sparse probeではcenteringを含む）。
CPUの内部準備とRR uploadを完全に同じ境界で切り分けたkernel-only計測ではない。
これは反復性能の比較であり、処理全体の高速化率ではない。

| 方式 | グラフ | 頂点数 | P | 変更前RR 秒 | 改善後RR 秒 | CPU 秒 | CPU/RR 倍 | 前RR/後RR 倍 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
for r in rows:s+=f"| {r['family']} | {r['graph']} | {r['nodes']:,} | {r['pivots'] or '—'} | {r['before_s']:.4f} | {r['after_s']:.4f} | {r['cpu_s']:.4f} | {r['speedup_cpu']:.2f} | {r['speedup_before']:.2f} |\n"
s+='\n倍率はseedごとの比の中央値なので、時間の中央値同士を割った値とは一致しない場合がある。\n\n| 方式 | グラフ | RR 1反復平均 ms（15反復時間/15） | CPU 1反復平均 ms | RR/CPU stress 中央値 | 最大値 |\n|---|---|---:|---:|---:|---:|\n'
for r in rows:s+=f"| {r['family']} | {r['graph']} | {r['after_s']*1000/15:.3f} | {r['cpu_s']*1000/15:.3f} | {r['stress_ratio']:.6f} | {r['max_stress_ratio']:.6f} |\n"
s+='''
FullのstressはCLI既定の固定64始点、Sparseはquality_probeの固定256始点（seed=0）のsampled full stress。
比較はそれぞれ同一入力・同一評価始点間で行い、FullとSparseの値を相互比較していない。
前処理と初期配置の一致はSparse CPU参照版とのassertでも検査した。
変更前RRはこの追加高速化の直前版であり、Sparseには前回の省メモリ化・反復末尾待機が既に含まれる。
従ってv3初期版に対する倍率ではない。

Fullは全頂点対を扱うため、本実験は11,143〜16,384頂点（最大約1.34億制約/反復）。
100万頂点の実測はSparseのみ。Fullを100万頂点で動作検証したとは主張しない。
既存のstress差については[前回のv3照合・品質検証](../rr-common-optimization-20261030/README.md)も参照。

## 再現・検証用ファイル

- `comparison.csv`: seed別の時間、stress、倍率、座標一致。
- `summary.json`: 集計、`verification.json`: 大規模比較の完全一致確認。
- `*-before.jsonl`, `*-after.jsonl`, `*-cpu.jsonl`: 生の計測値。SparseのafterはGPU/CPUの2行。
- `additional-regression.json`, `full-additional-regression.json`: 追加の品質検証。
- `full-tests.log`: 7 tests通過。`sparse-tests.log`: 35 tests通過。
- `sparse-clippy.log`: 全targetのclippy、warnings deniedで通過。
- `source-snapshot.tar.gz`, `sha256.json`: ソースとbinary識別情報。
- `full-before-gpu.rs.txt`, `sparse-before-gpu.rs.txt`: 変更前のGPUホストコード。
- `environment.txt`: 測定環境。

再実行はリポジトリルートで以下を使用。BASELINEには変更前のrr-gpuとsparse-probeを置く。
ソースsnapshotに対して対応するbefore-gpu.rs.txtをsrc/gpu.rsへ戻すことで変更前を構築できる。
benchmark.pyは既存の完了済み結果を再使用するため、新規測定では新しいOUTを指定する。

```sh
python3 tools/diagnostics/rr-iteration/benchmark.py --baseline-dir BASELINE --output OUT --suite full
python3 tools/diagnostics/rr-iteration/benchmark.py --baseline-dir BASELINE --output OUT --suite sparse
python3 tools/diagnostics/rr-iteration/regression.py OUT BASELINE
python3 tools/diagnostics/rr-iteration/summarize.py OUT > OUT/verification.json
python3 tools/diagnostics/rr-iteration/render_report.py OUT
```
'''
(p/'README.md').write_text(s)
