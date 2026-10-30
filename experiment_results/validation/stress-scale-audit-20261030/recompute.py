import json,csv,gzip,io,math
import numpy as np
from scipy.io import mmread
from scipy.sparse.csgraph import shortest_path, connected_components
from pathlib import Path
root=Path('/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization')
rows=[json.loads(l) for l in open(root/'experiment_results/v4/raw/results.jsonl')]
full=list(csv.DictReader(open(root/'experiment_results/v4/reference/full-v3/runs.csv')))
def sample(n,k,seed=0):
 a=list(range(n));s=seed;mask=(1<<64)-1
 for i in range(k):
  s=(s+0x9e3779b97f4a7c15)&mask;s=((s^(s>>30))*0xbf58476d1ce4e5b9)&mask;s=((s^(s>>27))*0x94d049bb133111eb)&mask;s=s^(s>>31)
  j=i+s%(n-i);a[i],a[j]=a[j],a[i]
 return a[:k]
def positions(p):
 op=gzip.open if p.endswith('.gz') else open
 with op(p,'rt') as f: txt=f.read().split('# Positions')[1].split('\n',1)[1]
 return np.loadtxt(io.StringIO(txt))
outputs=[]
for graph,pool,k in [('1138_bus',full,0),('dwt_2680',full,0),('preferentialAttachment',rows,256),('delaunay_n17',rows,1024)]:
 rs=[r for r in pool if r.get('graph',r.get('dataset'))==graph and int(r['seed'])==0 and r.get('study')=='basic' and r.get('status')=='success'][:3]
 if not rs:continue
 a=mmread(rs[0]['input_path']).tocsr();a=(a+a.T).tocsr();a.data[:]=1;a.setdiag(0);a.eliminate_zeros();n=a.shape[0]
 assert connected_components(a,directed=False,return_labels=False)==1
 xs=[positions(r['final_positions_path']) for r in rs];assert all(x.shape==(n,2) for x in xs)
 sources=sample(n,k) if k else list(range(n)); sums=[[] for r in rs];aa=np.zeros(len(rs));bb=aa.copy()
 for start in range(0,len(sources),16):
  ids=sources[start:start+16];ds=shortest_path(a,directed=False,unweighted=True,indices=ids);ds[np.arange(len(ids)),ids]=np.inf
  for j,x in enumerate(xs):
   norm=np.sqrt(((x[ids,None,:]-x[None,:,:])**2).sum(axis=2));q=norm/ds;terms=(q-1)**2;terms[np.arange(len(ids)),ids]=0
   sums[j].extend(terms.sum(axis=1));aa[j]+=(q*q).sum();bb[j]+=q.sum()
 for j,r in enumerate(rs):
  v=np.array(sums[j]);total=n/2*v.mean();old=float(r['stress_value']);record={'graph':graph,'method':r['method'],'nodes':n,'sources':len(sources),'reported':old,'recomputed':total,'rel_difference':total/old-1,'mean_pair_stress':total/(n*(n-1)/2),'relative_distance_rmse':math.sqrt(total/(n*(n-1)/2)),'best_scale':bb[j]/aa[j], 'stress_after_best_scale':(len(sources)*(n-1)-bb[j]**2/aa[j])*n/(2*len(sources))}
  if k:
   record['same256_recomputed']=n/2*v[:256].mean();record['same256_rel_difference']=record['same256_recomputed']/old-1
   record['approx95_halfwidth']=n/2*1.96*v.std(ddof=1)/math.sqrt(k)*math.sqrt((n-k)/(n-1))
   diff=v-np.array(sums[0]);record['paired_difference_vs_cpu']=n/2*diff.mean();record['paired_difference_95_halfwidth']=n/2*1.96*diff.std(ddof=1)/math.sqrt(k)*math.sqrt((n-k)/(n-1))
  outputs.append(record);print(json.dumps(record),flush=True)
Path('/tmp/stress_audit_20261030.json').write_text(json.dumps(outputs,indent=2))
