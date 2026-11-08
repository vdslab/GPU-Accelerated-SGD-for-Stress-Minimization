from pathlib import Path
import json,csv,statistics as st,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
rows=[json.loads(l) for l in (P/'results.jsonl').read_text().splitlines()]
assert len(rows)==60 and all(r['production_match'] for r in rows)
measured=[r for r in rows if not r['warmup']]
assert len(measured)==50
methods=['foreground_shuffle','background_shuffle']
metrics=['sgd_total_ms','mean_iteration_ms','sgd_to_cpu_ms','readback_ms','shuffle_cpu_ms','plan_wait_ms','dispatch_cpu_ms','gpu_wait_cpu_ms','gpu_timestamp_span_ms','upload_setup_ms','execute_wall_ms']
summary={m:{k:{'median':st.median(r[k] for r in measured if r['method']==m),'min':min(r[k] for r in measured if r['method']==m),'max':max(r[k] for r in measured if r['method']==m)} for k in metrics} for m in methods}
paired=json.loads((P/'paired.json').read_text())
summary['paired']={'count':len(paired),'background_faster':sum(r['delta']<0 for r in paired),'delta_ms_median':st.median(r['delta'] for r in paired),'change_percent_median':st.median(r['change_percent'] for r in paired),'speedup_median':st.median(r['speedup'] for r in paired),'delta_ms_min':min(r['delta'] for r in paired),'delta_ms_max':max(r['delta'] for r in paired)}
bg=[r for r in measured if r['method']==methods[1]]
summary['producer']={'all_plans_ready_ms_median':st.median(r['plan_events_ms'][-1][2] for r in bg),'first_iteration_complete_ms_median':st.median(r['iteration_events_ms'][0][2] for r in bg),'all_plans_before_first_iteration_complete':sum(r['plan_events_ms'][-1][2]<r['iteration_events_ms'][0][2] for r in bg)}
summary['correctness']={'checked_runs':60,'production_bitwise_matches':60,'seeds':5,'input_nodes':7066,'input_edges':100736,'shader_unchanged':True}
(P/'summary.json').write_text(json.dumps(summary,indent=2))
with (P/'runs.csv').open('w') as f:
 fields=['seed','repetition','warmup','method',*metrics,'production_match','final_positions_sha256'];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
with (P/'paired.csv').open('w') as f:
 w=csv.DictWriter(f,list(paired[0]));w.writeheader();w.writerows(paired)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for m,off,col,label in zip(methods,[-.12,.12],['#526978','#18816e'],['Foreground shuffle','Background shuffle']):
 for seed in range(5):
  vals=[r['sgd_to_cpu_ms'] for r in measured if r['method']==m and r['seed']==seed];mid=st.median(vals)
  axs[0].errorbar(seed+off,mid,yerr=[[mid-min(vals)],[max(vals)-mid]],fmt='o',color=col,capsize=4,label=label if seed==0 else None)
axs[0].set(xlabel='Initial-layout seed',ylabel='SGD + final readback (ms)',title='Median and range of 5 runs per seed',xticks=range(5))
axs[0].legend(frameon=False)
for seed in range(5):
 vals=[p['delta'] for p in paired if p['seed']==seed];axs[1].scatter([seed+(.07*(i-2)) for i in range(5)],vals,color='#18816e',s=28)
axs[1].axhline(0,color='#526978',lw=1);axs[1].axhline(summary['paired']['delta_ms_median'],color='#c16d35',ls='--',label='Paired median: -3.08 ms')
axs[1].set(xlabel='Initial-layout seed',ylabel='Background - foreground (ms)',title='25 paired differences; below zero = faster',xticks=range(5));axs[1].legend(frameon=False)
fig.suptitle('wiki-Vote / 15 iterations / T=W=1024 / no segmentation',fontsize=12)
fig.savefig(P/'comparison.png',dpi=180);plt.close(fig)
print(json.dumps(summary['paired'],indent=2));print(json.dumps(summary['producer'],indent=2))
