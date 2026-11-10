#!/usr/bin/env python3
"""v5.5 EX1 report: new RR runs, historical exact CPU and W=T RR references."""
from pathlib import Path
import collections,csv,json,math,statistics,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v55_ex1 import ROOT,BASE,OLD,VAULT,now,save
MED=statistics.median

def hierarchy(values):
 by_seed=collections.defaultdict(list)
 for seed,rep,value in values:by_seed[seed].append((rep,value))
 assert set(by_seed)==set(range(10,15))
 medians=[]
 for seed,values in sorted(by_seed.items()):
  assert sorted(rep for rep,v in values)==[0,1,2]
  medians.append(MED(value for rep,value in values))
 return dict(median=MED(medians),min=min(medians),max=max(medians),seed_medians=medians)

def write_csv(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 plan=json.loads((BASE/'config/plan.json').read_text());graphs=plan['tile_graphs'];tiles=plan['tiles']
 rows=[json.loads(p.read_text()) for p in (BASE/'raw/tile').glob('*/record.json')]
 assert len(rows)==1650 and len({r['run_id'] for r in rows})==1650
 assert all(r['method']=='rr_sgd' and r['status']=='success' and r['archive_status']=='success' for r in rows)
 assert all(r['stress_kind']=='exact' and r['iterations']==15 and r['epsilon']==.1 and r['rr_workgroup_size']==r['tile_size'] and r['rr_tile_size']==r['tile_size'] and r['rr_memory_mode']=='shared' and r['rr_phase_chunks']==1 and r['rr_inner_shuffle'] and r['rr_outer_shuffle'] and r['bfs_engine']=='auto' and r['iteration_times_ms'] is None and r['iteration_total_time_ms'] is None for r in rows)
 cpu_rows=json.loads((BASE/'config/cpu-reference-records.json').read_text())
 previous=json.loads((BASE/'config/previous-rr-reference-records.json').read_text())
 cpu={(r['graph'],r['seed'],r['repetition']):r for r in cpu_rows}
 old={(r['graph'],r['tile_size'],r['seed'],r['repetition']):r for r in previous}
 paired=[];groups=collections.defaultdict(list)
 for r in rows:
  c=cpu[r['graph'],r['seed'],r['repetition']];o=old[r['graph'],r['tile_size'],r['seed'],r['repetition']]
  assert c['stress_kind']==o['stress_kind']=='exact'
  assert r['input_sha256']==o['input_sha256']==c['input_sha256']
  assert r['initial_positions_sha256']==o['initial_positions_sha256']==c['initial_positions_sha256']
  paired.append(dict(graph=r['graph'],tile_size=r['tile_size'],seed=r['seed'],repetition=r['repetition'],previous_rr_run_id=o['run_id'],cpu_reference_run_id=c['run_id'],cpu_sgd_speedup=c['sgd_time_ms']/r['sgd_time_ms'],cpu_total_speedup=c['total_compute_time_ms']/r['total_compute_time_ms'],cpu_stress_ratio=r['stress_value']/c['stress_value'],previous_rr_sgd_speedup=o['sgd_time_ms']/r['sgd_time_ms'],previous_rr_total_speedup=o['total_compute_time_ms']/r['total_compute_time_ms'],previous_rr_stress_ratio=r['stress_value']/o['stress_value']))
  groups[r['graph'],r['tile_size']].append(r)
 summaries=[]
 for (g,t),rs in groups.items():
  assert len(rs)==15
  out=dict(graph=g,tile_size=t,workgroup_size=t,new_rr_runs=15,complete_seeds=5)
  for field in ['sgd_time_ms','total_compute_time_ms','stress_value','peak_rss_bytes','stress_eval_time_ms']:
   out[field]=hierarchy([(r['seed'],r['repetition'],r[field]) for r in rs])
  pairs=[r for r in paired if r['graph']==g and r['tile_size']==t]
  for field in ['cpu_sgd_speedup','cpu_total_speedup','cpu_stress_ratio','previous_rr_sgd_speedup','previous_rr_total_speedup','previous_rr_stress_ratio']:
   out[field]=hierarchy([(r['seed'],r['repetition'],r[field]) for r in pairs])
  summaries.append(out)
 idx={(r['graph'],r['tile_size']):r for r in summaries}
 for g in graphs:
  for seed in range(10,15):assert len({r['initial_positions_sha256'] for r in rows if r['graph']==g and r['seed']==seed})==1
 cpu_stats={g:{f:hierarchy([(r['seed'],r['repetition'],r[f]) for r in cpu_rows if r['graph']==g]) for f in ['sgd_time_ms','total_compute_time_ms','stress_value']} for g in graphs}
 old_stats={(g,t):{f:hierarchy([(r['seed'],r['repetition'],r[f]) for r in previous if r['graph']==g and r['tile_size']==t]) for f in ['sgd_time_ms','total_compute_time_ms','stress_value']} for g in graphs for t in tiles}
 save(BASE/'tables/tile-hierarchical.json',summaries);save(BASE/'tables/cpu-historical-reference-summary.json',cpu_stats)
 write_csv(BASE/'tables/tile-runs.csv',rows);write_csv(BASE/'tables/tile-paired-ratios.csv',paired)
 flat=[]
 for r in summaries:
  item={k:r[k] for k in ['graph','tile_size','workgroup_size','new_rr_runs','complete_seeds']}
  for k,v in r.items():
   if isinstance(v,dict):
    for stat in ['median','min','max']:item[k+'_'+stat]=v[stat]
  flat.append(item)
 write_csv(BASE/'tables/tile-summary.csv',flat)
 gm=lambda vs:math.exp(statistics.mean(math.log(v) for v in vs))
 aggregate=[]
 for t in tiles:
  ds=[idx[g,t] for g in graphs]
  aggregate.append(dict(tile_size=t,slowdown_vs_each_graph_best_gmean=gm([idx[g,t]['sgd_time_ms']['median']/min(idx[g,u]['sgd_time_ms']['median'] for u in tiles) for g in graphs]),cpu_stress_ratio_gmean=gm([r['cpu_stress_ratio']['median'] for r in ds]),previous_rr_sgd_speedup_gmean=gm([r['previous_rr_sgd_speedup']['median'] for r in ds]),previous_rr_total_speedup_gmean=gm([r['previous_rr_total_speedup']['median'] for r in ds])))
 save(BASE/'tables/tile-cross-graph.json',aggregate)
 metrics=[('sgd_time_ms','SGD elapsed per iteration (ms)',lambda x:x/15,'time'),('total_compute_time_ms','Total compute time (ms)',lambda x:x,'total'),('stress_value','Exact Full Stress',lambda x:x,'stress'),('cpu_stress_ratio','Stress / historical exact CPU',lambda x:x,'stress-ratio'),('previous_rr_sgd_speedup','Previous RR / v5.5 RR SGD time',lambda x:x,'previous-speedup')]
 for field,ylabel,scale,suffix in metrics:
  fig,axs=plt.subplots(5,2,figsize=(12,16),layout='constrained')
  for ax,g in zip(axs.flat,graphs):
   stats=[idx[g,t][field] for t in tiles]
   ax.plot(tiles,[scale(s['median']) for s in stats],'o-',color='#2563eb',label='v5.5 RR Full W=T')
   ax.fill_between(tiles,[scale(s['min']) for s in stats],[scale(s['max']) for s in stats],color='#2563eb',alpha=.18,label='Range of 5 seed medians')
   if field in ['sgd_time_ms','total_compute_time_ms','stress_value']:
    old_values=[old_stats[g,t][field] for t in tiles]
    ax.plot(tiles,[scale(s['median']) for s in old_values],'s--',color='#d97706',label='Previous EX1 RR W=T')
    c=cpu_stats[g][field];ax.axhline(scale(c['median']),color='#666',ls=':',label='Historical CPU reference')
    ax.axhspan(scale(c['min']),scale(c['max']),color='#777',alpha=.08)
   else:ax.axhline(1,color='#666',ls=':',label='Ratio = 1')
   ax.set_xscale('log',base=2);ax.set_xticks(tiles,labels=[str(t) for t in tiles],rotation=45)
   if suffix in ['time','total']:ax.set_yscale('log')
   ax.set_title(g);ax.set_xlabel('Tile size T = workgroup size W');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
  handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2)
  fig.suptitle('v5.5 EX1: 5 seeds x 3 repetitions; GPU timestamps off',y=1.025)
  for ext in ['png','pdf']:fig.savefig(BASE/f'figures/tile-{suffix}.{ext}',dpi=160,bbox_inches='tight')
  plt.close(fig)
 text='''# v5.5 EX1：タイル実験（W=T）の結果と考察

**新規測定はRR Full 1,650/1,650件成功。CPU・Atomicは取り直していない。EX2は未開始。**

10グラフ×11T×seed10〜14×各3回。15反復、epsilon0.1、T=W=1,2,4,8,16,32,64,128,256,512,1024、shared、内外shuffle有効、phase_chunks1、auto BFS。GPU timestamp無効。全グラフで厳密Full Stressを同じ最終座標から計時終了後に評価し、評価時間を別記した。各graph/Tのseed999ウォームアップは集計外。逐次実行、時間打切りなし。

前回v5 EX1のW=T RR 1,650件と、厳密評価済みCPU Full150件は読み取り専用の参照として使用。新規1,650件とCPU参照150件を「新規1,800件」と数えない。同じ入力・初期座標hashを照合した。CPU速度比は異なる日時の過去測定との参考比較である。前回RRとの速度比較にも環境の経時変動を含む。

同seed3回の中央値→5seedの中央値・最小〜最大で集計。比は対応するgraph/T/seed/repetitionごとに求めてから階層集計。誤差帯は5seed中央値の最小〜最大であり信頼区間ではない。

## グラフ別の観測最速T

速度による観測最速値であり、未測定のグラフへ一般化せず、Stressも併記する。

| グラフ | 最速T=W | SGD/反復 ms | 過去CPU速度比（参考） | 厳密Stress/CPU | 同Tの旧RR/新RR SGD速度比 |
|---|---:|---:|---:|---:|---:|
'''
 bests=[]
 for g in graphs:
  best=min((idx[g,t] for t in tiles),key=lambda r:r['sgd_time_ms']['median']);bests.append(best)
  text+=f"| {g} | {best['tile_size']} | {best['sgd_time_ms']['median']/15:.3f} | {best['cpu_sgd_speedup']['median']:.3f} | {best['cpu_stress_ratio']['median']:.5f} | {best['previous_rr_sgd_speedup']['median']:.3f} |\n"
 chosen=min(aggregate,key=lambda r:r['slowdown_vs_each_graph_best_gmean'])
 text+=f"\n各グラフの最速Tに対する相対SGD時間の幾何平均が最小だったT=Wは **{chosen['tile_size']}**。これは今回の10グラフ上の観測であり、EX2の設定を自動変更・開始していない。\n"
 text+='\n## 解釈・制約\n\nT=Wの変更は共有メモリ再利用、WGあたりの仕事量、WG数、外側ラウンド数を同時に変える。単一のハードウェア要因の寄与率を分離した実験ではない。T<32でもSIMD幅を変更した意味ではない。T間では更新順が変わるため、最終Stressの一致は要求しない。1seed内の3回は独立した初期配置ではない。\n\n今回のPR Full autoは対応サイズ内でCPU BFSを選ぶため、前回との全時間差をGPU BFSの効果とは解釈しない。常駐アプリ・温度・電源状態を専用計測で固定していないため、特に小さい差の断定は避ける。\n\n## 図表\n\n'
 for suffix in ['time','total','stress','stress-ratio','previous-speedup']:text+=f'![[EX1_v5.5_図表/tile-{suffix}.png]]\n\n'
 text+=f'## 保存先\n\n`{BASE}`\n\n`config/`：入力、順序、条件、過去CPU/RR参照、ソース・バイナリSHA。`raw/tile/`：新規1,650件のnative JSON、コマンド、ログ、検証済み圧縮座標、メモリ。`warmups/`：集計外。`tables/`：生データCSV、階層集計、対応する比。`figures/`：PNG/PDF。`source-snapshot/`・`bin/`：実装固定。前回EX1/EX2ファイルのハッシュ不変を完了時に検証。\n'
 (BASE/'reports/tile.md').write_text(text)
 (VAULT/'EX1_v5.5_タイル実験(W=T)_結果と考察.md').write_text(text)
 assets=VAULT/'EX1_v5.5_図表';assets.mkdir(exist_ok=True)
 for p in (BASE/'figures').glob('*.png'):shutil.copy2(p,assets/p.name)
 save(BASE/'validation/completion-validation.json',dict(new_rr_runs=1650,success=1650,cpu_reference_runs=150,cpu_new_runs=0,atomic_new_runs=0,conditions=110,complete_seeds_per_condition=5,repetitions=3,exact_stress_all=True,initial_fingerprints_match_cpu_and_previous_rr=True,gpu_timestamps_disabled=True,workgroup_policy='W=T',ex2_started=False))
 save(BASE/'tables/best-per-graph.json',bests)
 print('v5.5 EX1 report complete; EX2 not started',flush=True)

if __name__=='__main__':main()
