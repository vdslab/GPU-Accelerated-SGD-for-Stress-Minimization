#!/usr/bin/env python3
"""Offline report of completed v5 tile measurements; paired ratios before aggregation."""
from pathlib import Path
import collections,csv,json,math,statistics,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/v5/ex1-tile-w-equal-t-20261107'
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
 assert all(r['iterations']==15 and r['epsilon']==.1 and r['stress_kind']=='exact' for r in rows)
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
    assert r['rr_phase_chunks']==1 and r['rr_inner_shuffle'] and r['rr_outer_shuffle'] and r['rr_workgroup_size']==t
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
 oldbase=ROOT/'experiment_results/v5'
 oldrows=[json.loads(p.read_text()) for p in (oldbase/'raw/tile').glob('*/record.json')]
 oldidx={(r['graph'],r['method'],r['tile_size'],r['seed'],r['repetition']):r for r in oldrows}
 comparisons=[]
 for g in graphs:
  for t in tiles:
   rs=[r for r in rows if r['graph']==g and r['method']=='rr_sgd' and r['tile_size']==t]
   ratios=[]
   for r in rs:
    old=oldidx[g,'rr_sgd',t,r['seed'],r['repetition']]
    assert r['initial_positions_sha256']==old['initial_positions_sha256']
    ratios.append((r['seed'],r['repetition'],old['iteration_time_ms']/r['iteration_time_ms']))
   comparisons.append(dict(graph=g,tile_size=t,old_w1024_over_new_w_equal_t=hierarchy(ratios)))
 (BASE/'tables/previous-w1024-time-comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
 # Compare elapsed SGD times only; prior large-graph stress values were sampled.
 oldsummary=json.loads((oldbase/'tables/tile-hierarchical.json').read_text())
 oldstats={(r['graph'],r['tile_size']):r for r in oldsummary if r['method']=='rr_sgd'}
 fig,axs=plt.subplots(5,2,figsize=(12,16),layout='constrained')
 for ax,g in zip(axs.flat,graphs):
  for label,series,color in [('W=T',[idx[g,'rr_sgd',t]['iteration_time_ms']['median']/15 for t in tiles],'#2563eb'),('Previous W=1024',[oldstats[g,t]['iteration_time_ms']['median']/15 for t in tiles],'#d97706')]:
   ax.plot(tiles,series,'o-',label=label,color=color)
  ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(tiles,labels=[str(t) for t in tiles],rotation=45);ax.set_title(g);ax.set_xlabel('Tile size T');ax.set_ylabel('SGD elapsed time per iteration (ms)');ax.grid(alpha=.2)
 handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2)
 fig.savefig(BASE/'figures/tile-previous-time.png',dpi=160);fig.savefig(BASE/'figures/tile-previous-time.pdf');plt.close(fig)
 text='''# v5 追加実験 EX1：タイル実験（W=T）の結果と考察

10グラフ×（RR Fullの11サイズ＋CPU Full）×5seed×3回＝1,800実行。
全件成功・全10グラフで厳密Full Stress。15反復、epsilon=0.1、shared、内外shuffleあり、phase_chunks=1。
T=W=1,2,4,8,16,32,64,128,256,512,1024。T<32はSIMD幅を変更する設定ではなく、32未満のスレッドをWGに用意する設定。
品質評価はSGD・総計算時間の計測終了後に行い、評価時間は別途記録した。GPU timestampは無効。

集計は同seed3回の中央値→5seedの中央値・最小〜最大。CPUとの比は対応するgraph/seed/repetitionごとに求めてから同じ順で集計した。
前回のW1024固定実験は保持。今回の初期座標fingerprintは前回の同graph/seed/T/repetitionと一致し、今回の全T・繰り返し間でも一致した。

## グラフ別の結果

最速TはSGD経過時間の中央値で選択した観測値であり、品質による候補除外や未知グラフへの一般化はしていない。

| グラフ | 最速T=W | SGD時間/反復 (ms) | CPU速度比 | 最終厳密Stress/CPU | 前回W1024の最速T |
|---|---:|---:|---:|---:|---:|
'''
 for g in graphs:
  best=min((idx[g,'rr_sgd',t] for t in tiles),key=lambda r:r['iteration_time_ms']['median'])
  oldbest=min(tiles,key=lambda t:oldstats[g,t]['iteration_time_ms']['median'])
  text+=f"| {g} | {best['tile_size']} | {best['iteration_time_ms']['median']/15:.3f} | {best['sgd_speedup']['median']:.2f} | {best['stress_ratio']['median']:.5f} | {oldbest} |\n"
 text+='''
## 評価上の注意

W=TでTを増やすと、座標再利用が増える一方、1WGの仕事量・SIMD組数・共有メモリ量が増え、配れるWG数と外側ラウンド数が減る。この実験はその総合的なトレードオフを評価する。個別のハードウェア要因の寄与率や同一サイクルの命令発行数を切り分けた実験ではない。

同seedの3回は独立した初期配置ではない。更新順が変わるためT間の最終Stressは一致を要求しない。速度と品質の両方を報告し、低い実行時間だけで品質優位とは判断しない。

前回W1024と今回W=Tの実験時刻は異なる。速度の対応比較には環境変動も含まれる。特にT1024は両方ともW1024なので、同じ設定の再測定として差を確認できる。
前回PGPgiantcompo・fe_4elt2のStressは256始点からの推定、今回は厳密値であるため、そのまま品質改善率として比較しない。CPU比は今回の厳密CPU測定を基準にする。

## 図表

![[EX1_v5_図表/tile-time.png]]

![[EX1_v5_図表/tile-stress.png]]

![[EX1_v5_図表/tile-stress-ratio.png]]

![[EX1_v5_図表/tile-previous-time.png]]

## 保存先

`experiment_results/v5/ex1-tile-w-equal-t-20261107/`

- `raw/tile/`：全1,800実行のコマンド・出力・検証済み圧縮座標・メモリ・評価時間。
- `tables/tile-runs.csv`：生データ一覧。`tile-hierarchical.json`、`tile-summary.csv`：階層集計。
- `tables/tile-paired-ratios.csv`：同条件のCPUとの対応比。
- `tables/previous-w1024-time-comparison.json`：前回との対応するSGD時間比。
- `figures/`：上記チャートのPNG・PDF。
- `config/`、`source-snapshot/`、`bin/`：入力ハッシュ・実行順・機材・ソース・新旧バイナリ。
'''
 (BASE/'reports/tile.md').write_text(text)
 (VAULT/'EX1_タイル実験(W=T)_結果と考察.md').write_text(text)
 assets=VAULT/'EX1_v5_図表';assets.mkdir(exist_ok=True)
 for suffix in ['time','stress','stress-ratio','previous-time']:shutil.copy2(BASE/f'figures/tile-{suffix}.png',assets/f'tile-{suffix}.png')
 (BASE/'validation/tile-completion.json').write_text(json.dumps(dict(formal_runs=1800,success=1800,conditions=120,complete_seeds_per_condition=5,repetitions_per_seed=3,rr_initial_fingerprints_consistent=True,paired_ratios=True,archive_success=True,stress_mode='exact',workgroup_policy='W=T'),indent=2)+'\n')
 print(text)

if __name__=='__main__':main()
