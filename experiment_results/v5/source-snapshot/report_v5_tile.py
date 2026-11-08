#!/usr/bin/env python3
"""Offline report of completed v5 tile measurements; paired ratios before aggregation."""
from pathlib import Path
import collections,csv,json,math,statistics,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/v5'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
MED=statistics.median

def hierarchy(values):
 seeds=collections.defaultdict(list)
 for seed,rep,val in values:seeds[seed].append((rep,val))
 assert set(seeds)==set(range(10,15))
 x=[]
 for seed,vs in sorted(seeds.items()):
  assert sorted(r for r,v in vs)==[0,1,2]
  x.append(MED(v for r,v in vs))
 return dict(median=MED(x),min=min(x),max=max(x),seeds=x)

def main():
 plan=json.loads((BASE/'config/plan.json').read_text());graphs=plan['tile_graphs'];tiles=[2**i for i in range(11)]
 rows=[json.loads(p.read_text()) for p in (BASE/'raw/tile').glob('*/record.json')]
 assert len(rows)==1800 and len({r['run_id'] for r in rows})==1800
 assert all(r['status']=='success' and r['archive_status']=='success' for r in rows)
 assert all(r['iterations']==15 and r['epsilon']==.1 for r in rows)
 cpu={(r['graph'],r['seed'],r['repetition']):r for r in rows if r['method']=='sgd'}
 grouped=collections.defaultdict(list)
 for r in rows:grouped[(r['graph'],r['method'],r['tile_size'])].append(r)
 summaries=[];paired=[]
 for (g,method,t),rs in grouped.items():
  assert len(rs)==15
  out=dict(graph=g,method=method,tile_size=t,success_runs=15,complete_seeds=5)
  for field in ['iteration_time_ms','total_compute_time_ms','stress_value','peak_rss_bytes']:
   out[field]=hierarchy([(r['seed'],r['repetition'],r[field]) for r in rs])
  if method=='rr_sgd':
   for r in rs:
    assert r['rr_phase_chunks']==1 and r['rr_inner_shuffle'] and r['rr_outer_shuffle'] and r['rr_workgroup_size']==1024
    c=cpu[(g,r['seed'],r['repetition'])]
    paired.append(dict(graph=g,tile_size=t,seed=r['seed'],repetition=r['repetition'],stress_ratio=r['stress_value']/c['stress_value'],sgd_speedup=c['iteration_time_ms']/r['iteration_time_ms'],total_speedup=c['total_compute_time_ms']/r['total_compute_time_ms']))
   for field in ['stress_ratio','sgd_speedup','total_speedup']:
    out[field]=hierarchy([(r['seed'],r['repetition'],r[field]) for r in paired if r['graph']==g and r['tile_size']==t])
  summaries.append(out)
 idx={(r['graph'],r['method'],r['tile_size']):r for r in summaries}
 # Initial positions must agree across all T and repetitions within a graph/seed.
 for g in graphs:
  for seed in range(10,15):
   fingerprints={r['initial_positions_sha256'] for r in rows if r['graph']==g and r['method']=='rr_sgd' and r['seed']==seed}
   assert len(fingerprints)==1,(g,seed)
 (BASE/'tables/tile-hierarchical.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
 with (BASE/'tables/tile-paired-ratios.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 metrics=[('iteration_time_ms','SGD elapsed time per iteration (s)',lambda v:v/15000,'time'),('stress_value','Final Full Stress',lambda v:v,'stress'),('stress_ratio','Final Stress / paired CPU',lambda v:v,'stress-ratio')]
 for metric,label,scale,suffix in metrics:
  fig,axs=plt.subplots(5,2,figsize=(12,16),layout='constrained')
  for ax,g in zip(axs.flat,graphs):
   stats=[idx[(g,'rr_sgd',t)][metric] for t in tiles]
   y=[scale(v['median']) for v in stats];lo=[scale(v['min']) for v in stats];hi=[scale(v['max']) for v in stats]
   ax.plot(tiles,y,'o-',color='#2563eb',label='RR Full');ax.fill_between(tiles,lo,hi,color='#2563eb',alpha=.18,label='5-seed min–max')
   if metric!='stress_ratio':
    c=idx[(g,'sgd',None)][metric];ax.axhline(scale(c['median']),color='#555555',ls='--',label='CPU Full median')
    ax.axhspan(scale(c['min']),scale(c['max']),color='#777777',alpha=.1)
   else:ax.axhline(1,color='#555555',ls='--',label='CPU ratio = 1')
   if metric=='iteration_time_ms':ax.set_yscale('log')
   ax.set_xscale('log',base=2);ax.set_xticks(tiles,labels=[str(t) for t in tiles],rotation=45);ax.set_title(g);ax.set_xlabel('Tile size T');ax.set_ylabel(label);ax.grid(alpha=.2)
  handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3)
  fig.savefig(BASE/f'figures/tile-{suffix}.png',dpi=160);fig.savefig(BASE/f'figures/tile-{suffix}.pdf');plt.close(fig)
 flat=[]
 for r in summaries:
  item={k:r[k] for k in ['graph','method','tile_size','success_runs','complete_seeds']}
  for k,v in r.items():
   if isinstance(v,dict):
    for stat in ['median','min','max']:item[k+'_'+stat]=v[stat]
  flat.append(item)
 fields=list(dict.fromkeys(k for r in flat for k in r))
 with (BASE/'tables/tile-summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
 aggregate=[]
 for t in tiles:
  rs=[idx[(g,'rr_sgd',t)] for g in graphs]
  relative=[r['iteration_time_ms']['median']/min(idx[(g,'rr_sgd',u)]['iteration_time_ms']['median'] for u in tiles) for g,r in zip(graphs,rs)]
  gm=lambda vs:math.exp(statistics.mean(math.log(x) for x in vs))
  aggregate.append(dict(tile_size=t,slowdown_vs_each_graph_best_gmean=gm(relative),stress_ratio_gmean=gm([r['stress_ratio']['median'] for r in rs]),stress_ratio_min=min(r['stress_ratio']['median'] for r in rs),stress_ratio_max=max(r['stress_ratio']['median'] for r in rs),sgd_speedup_gmean=gm([r['sgd_speedup']['median'] for r in rs])))
 (BASE/'tables/tile-cross-graph.json').write_text(json.dumps(aggregate,indent=2)+'\n')
 text='''# v5 タイルサイズ実験：結果と考察

## 実施と検証

10グラフ×（CPU Full＋RR Fullの11T）×5seed×3回＝**1,800実行すべて成功**。seed10〜14、15反復、epsilon0.1、W1024、shared、両shuffleあり、phase_chunks1。速度と最終Stressは同一実行から取得した。全120設定が5seed×3回揃い、全実行の座標圧縮も成功。各グラフ・seedのRR初期座標fingerprintは全T・3回で一致する。

n≤8000は厳密Full Stress、PGPgiantcompoとfe_4elt2は評価seed0・256始点の推定Full Stress。グラフ間で絶対Stressの大小を性能順位に使わない。GPU pass時間は補助計測の確保エラーを避けるため全GPU方式で無効にし、主指標の実経過時間を使った。変更前547実行等はvalidationに分離し、本測定に混ぜていない。

各seedの3回中央値を求め、その5seedの中央値・最小〜最大を報告する。CPU比は同じgraph・seed・repetitionのrunごとに計算してから階層集計する。

## グラフ別の結果

最速TはSGD実経過時間の中央値で選んだ観測値。Stress比はそのTでのRR/CPU。共通Tの自動決定やStress閾値での候補除外はしない。

| グラフ | 最速T | SGD時間/反復 (ms) | CPU速度比 | 最終Stress/CPU | T128の最終Stress/CPU |
|---|---:|---:|---:|---:|---:|
'''
 for g in graphs:
  best=min([idx[(g,'rr_sgd',t)] for t in tiles],key=lambda r:r['iteration_time_ms']['median'])
  text+=f"| {g} | {best['tile_size']} | {best['iteration_time_ms']['median']/15:.3f} | {best['sgd_speedup']['median']:.2f} | {best['stress_ratio']['median']:.4f} | {idx[(g,'rr_sgd',128)]['stress_ratio']['median']:.4f} |\n"
 text+='''
## 共通Tの比較

各グラフを同じ重みで扱う幾何平均。最速Tに対する時間倍率は1に近いほど速い。Stress比は1より大きいほどCPU基準よりStressが大きい。これは記述的な要約であり、統計的有意差の検定ではない。

| T | グラフ別最速に対する時間倍率（幾何平均） | Stress/CPU（幾何平均） | Stress/CPUのグラフ間範囲 | CPU速度比（幾何平均） |
|---:|---:|---:|---:|---:|
'''
 for r in aggregate:
  text+=f"| {r['tile_size']} | {r['slowdown_vs_each_graph_best_gmean']:.3f} | {r['stress_ratio_gmean']:.4f} | {r['stress_ratio_min']:.4f}〜{r['stress_ratio_max']:.4f} | {r['sgd_speedup_gmean']:.2f} |\n"
 text+='''
## 考察

1. **速度を優先した共通Tの候補は128。** グラフ別最速は128または256に集まった。T128は全体の時間倍率が小さく、T256よりもStressの増加を抑えるグラフが多い。ただし最終Stressの優位を保証する値ではない。
2. **Tを大きくするほど常によいわけではない。** T1はGPU投入回数が多く、1024は各タイル内処理が増える。128〜256付近が今回の機材・W1024・実装・15反復での速度上の良好な範囲だった。実装の投入回数と処理構造から整合する説明だが、原因の切り分け実験ではない。
3. **構造によって品質への影響が異なる。** email・CA-GrQc・p2p-Gnutella08・PGPgiantcompoなどでは大きいTでStressが増える傾向がある。メッシュ・送電網では相対的に変化が小さい。ただしadd20は回路グラフであり、5対5の構造候補を厳密なスケールフリー分類と呼ぶ根拠はない。
4. **dwt_2680のT16には非単調な品質悪化がある。** seed10・12・13・14でStressが高く、seed11では近傍Tと近い。各seed内3回は同じ値を再現した。単発の測定ノイズとして除外せず、初期条件と更新順による局所解の違いが候補となるが、原因は未確定とする。
5. **再現性と一般化には限界がある。** 同seed3回は独立した初期配置ではない。T評価と通常比較でグラフが重なるため、seedを分けても未知グラフへの一般化は検証していない。CPUとRRの比較には精度・更新順・GPU実装全体の違いが含まれ、速度差やStress差の全てをタイルだけの効果に帰属しない。

## 次の実験

分析時の速度優先候補は128だったが、通常比較の共通Tは結果確認後のユーザー指定で**1024**に固定した。通常比較4,230実行は全件完了し、速度とStressの結果・考察を別ノートへ保存済み。自動選定やStress閾値で候補を除外した結果ではない。

## 図表と生データ

- `experiment_results/v5/tables/tile-runs.csv`：全1,800実行の生データ。
- `experiment_results/v5/tables/tile-paired-ratios.csv`：対応するrunから求めた比。
- `experiment_results/v5/tables/tile-summary.csv` / `tile-hierarchical.json`：5seedの階層集計。
- `experiment_results/v5/tables/tile-cross-graph.json`：Tごとの記述的集計。
- `experiment_results/v5/figures/tile-time.png` / `tile-stress.png` / `tile-stress-ratio.png`：10グラフ別パネル。PDFも保存。
- `experiment_results/v5/raw/tile/`：各実行のコマンド、stdout、stderr、圧縮した初期・最終座標、圧縮検証記録。
'''
 (BASE/'reports/tile.md').write_text(text)
 (VAULT/'1_tilesize_考察.md').write_text(text)
 assets=VAULT/'v5_図表';assets.mkdir(exist_ok=True)
 for suffix in ['time','stress','stress-ratio']:shutil.copy2(BASE/f'figures/tile-{suffix}.png',assets/f'tile-{suffix}.png')
 p=VAULT/'1_tilesize.md';s=p.read_text();s=s.replace('測定中。10グラフ×12設定×5seed×3回＝1,800実行。','**測定完了：1,800/1,800成功。** 通常比較の共通Tはユーザー指定1024。')
 heading='## v5確定結果'
 body='''## v5確定結果

1,800実行すべて成功、全120設定で5seed×3回が揃った。結果と考察は[[1_tilesize_考察]]。通常比較の共通Tは結果確認後のユーザー指定で1024に固定し、通常比較も完了した。

![[v5_図表/tile-time.png]]

![[v5_図表/tile-stress.png]]

![[v5_図表/tile-stress-ratio.png]]
'''
 if heading in s:s=s[:s.index(heading)].rstrip()+'\n\n'+body
 else:s=s.rstrip()+'\n\n'+body
 p.write_text(s)
 save=dict(formal_runs=1800,success=1800,conditions=120,complete_seeds_per_condition=5,repetitions_per_seed=3,rr_initial_fingerprints_consistent=True,paired_ratios=True,archive_success=True,normal_comparison_started=bool(list((BASE/"raw/comparison").glob("*/record.json"))))
 (BASE/'validation/tile-completion.json').write_text(json.dumps(save,indent=2)+'\n')
 print(text)
if __name__=='__main__':main()
