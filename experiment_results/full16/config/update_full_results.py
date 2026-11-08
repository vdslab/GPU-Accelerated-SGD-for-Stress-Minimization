from pathlib import Path
import json,csv,statistics,re
base=Path('/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/full16');a=json.loads((base/'tables/audit.json').read_text());rr={r['graph']:r for r in a['full_graphs']};arr={r['graph']:r for r in a['full_arrivals']};degree=a['degree_summary'];wiki=rr['wiki-Vote'];fe=rr['fe_4elt2'];hits=sum(r['hit'] for r in arr.values());ends=sum(r['end'] for r in arr.values())
p=Path('section/01.introduction.tex');s=p.read_text();start=s.index('The two schedules have different consequences');s=s[:start]+f'''After 15 iterations, RoundRobin Full SGD shortens iterations on {a['faster']} of {a['comparable_graphs']} comparable graphs and keeps the median stress ratio at most 1.05 on {a['quality_within_5pct']}; {a['both']} meet both criteria. RoundRobin Sparse SGD meets both criteria on 36 of 37 comparable graphs. We also measure time to a common stress target and compare split and unsplit Full schedules on preferential-attachment graphs of increasing size. A separate road-network experiment computes coordinates for more than eleven million vertices with a reduced pivot set.
''';p.write_text(s)
p=Path('section/04.experiment.tex');s=p.read_text()
s=s.replace('and preprocessing-inclusive runtime for road networks with millions of vertices.','a comparison of split and unsplit schedules on preferential-attachment graphs, and preprocessing-inclusive runtime for road networks with millions of vertices.')
s=s.replace('We fix Sparse at 200 pivots', 'We fix Full at $T=1024$ with 16 segments and retain four alternating segments per phase in Sparse. Tile-size sensitivity is outside this evaluation.\nWe fix Sparse at 200 pivots')
s=s.replace('Within each family, inputs, seeds, initial layouts, learning rates, and Sparse preprocessing are matched.','Within each family, inputs, seeds, initial layouts, learning rates, and Sparse preprocessing are matched.\nWe remeasured RoundRobin Full SGD and retained the archived CPU, Atomic, and Sparse measurements from the same machine. These main comparisons combine different measurement dates and do not isolate transient system load. The additional preferential-attachment experiment measures CPU and both Full GPU schedules during the same session.')
start=s.index('RoundRobin Full SGD is faster');end=s.index('Atomic is faster on 13 of 19',start)
s=s[:start]+f'''RoundRobin Full SGD is faster on {a['faster']} of {a['comparable_graphs']} comparable graphs, with median speedups of ${min(r['speedup'] for r in rr.values()):.3f}$--${max(r['speedup'] for r in rr.values()):.3f}\\times$.
The two slowdowns occur on \\graph{{dwt_1005}} and \\graph{{1138_bus}}.
For \\graph{{fe_4elt2}}, speedup is ${fe['speedup']:.3f}\\times$ with a stress ratio of {fe['stress_ratio']:.3f}; for \\graph{{wiki-Vote}}, the values are ${wiki['speedup']:.3f}\\times$ and {wiki['stress_ratio']:.3f}.
Eighteen graphs have a median stress ratio at most 1.05, and sixteen combine this quality criterion with faster iterations. The remaining quality gap is on \\graph{{bcsstk17}}, whose ratio is {rr['bcsstk17']['stress_ratio']:.3f}.
''' +s[end:]
s=s.replace('On \\graph{fe_4elt2}, the RoundRobin traces reach the target sooner than CPU in median elapsed SGD time (\\autoref{fig:full-time}). On \\graph{wiki-Vote}, they exhaust their 15 iterations above the target.',f"On \\graph{{fe_4elt2}} and \\graph{{wiki-Vote}}, {arr['fe_4elt2']['hit']} and {arr['wiki-Vote']['hit']} of ten RoundRobin traces reach the target, respectively (\\autoref{{fig:full-time}}). Their median attainment times are {arr['fe_4elt2']['rr_seconds']:.3f} and {arr['wiki-Vote']['rr_seconds']:.3f} s, compared with CPU times of {arr['fe_4elt2']['cpu_seconds']:.3f} and {arr['wiki-Vote']['cpu_seconds']:.3f} s.")
s=s.replace('RoundRobin Full SGD completed 99 (one memory stop).','RoundRobin Full SGD completed all 100.')
start=s.index('In Full SGD the same tile size produced');end=s.index('\\begin{figure*}',start)
s=s[:start]+f'''Across the six Full graphs, {hits} of 60 runs attain the target and {ends} remain within it at iteration 15. All 30 Sparse runs attain it and remain within it at iteration 15.
The recorded checkpoints bound the first observed attainment; they do not identify the exact crossing between saved iterations.
''' +s[end:]
# Move the road-layout figure into its own subsection.
start=s.index('\\begin{figure*}',s.index('Across the six Full graphs'));end=s.index('\\end{figure*}',start)+len('\\end{figure*}')
roadfig=s[start:end];s=s[:start]+s[end:]
start=s.index('\\paragraph{Sensitivity to initialization and graph structure.}');end=s.index('\\begin{table}[tb]',start)
new=r'''\paragraph{Degree heterogeneity and graph size.}
We generated six Barab\'asi--Albert preferential-attachment graphs using NetworkX: 2,000, 4,000, and 8,000 vertices, five attachments per new vertex, and graph seeds 0 and 1 at each size.
The edge counts are 9,975, 19,975, and 39,975, respectively; maximum degrees range from 143 to 311.
For each graph, we use five matched initial layouts (seeds 0--4), exact full stress, and the same 15-iteration schedule.
We measure CPU, unsplit RoundRobin Full SGD, and the 16-segment schedule sequentially, alternating method order across initialization seeds after a preliminary run for each setting.
Both GPU schedules wait for completion at each iteration, isolating segmentation from the earlier change in CPU wait placement.
'''
new+=f'''Across these six graphs, median CPU speedups range from ${min(r['split_speedup'] for r in degree):.2f}$ to ${max(r['split_speedup'] for r in degree):.2f}\\times$ for 16 segments.
Unsplit final stress ratios range from {min(r['unsplit_stress_ratio'] for r in degree):.3f} to {max(r['unsplit_stress_ratio'] for r in degree):.3f}, compared with {min(r['split_stress_ratio'] for r in degree):.3f} to {max(r['split_stress_ratio'] for r in degree):.3f} after segmentation (\\autoref{{fig:degree}}).
Of the 30 matched layouts, {sum(r['unsplit_within_target'] for r in degree)} unsplit runs and {sum(r['split_within_target'] for r in degree)} split runs finish within 5\\% of CPU.
The split schedule takes {min(r['split_unsplit_time_ratio'] for r in degree):.2f}--{max(r['split_unsplit_time_ratio'] for r in degree):.2f} times the unsplit iteration time in graph-level medians.
These measurements show the time--quality effect of interleaving within this graph family; they do not isolate degree distribution from other structural properties or establish behavior on all networks.
'''
new+=r'''
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{full_degree_scaling.pdf}
\caption{Full SGD on preferential-attachment graphs. Each vertex count has two graph seeds; points are medians and bars are minima and maxima over five matched initial layouts. (a) Mean iteration time on a logarithmic axis. (b) Exact final stress relative to same-seed CPU stress. The horizontal line marks 1.05. Both GPU variants use the same update rule, tile size, and per-iteration completion wait.}
\label{fig:degree}
\end{figure*}

'''
s=s[:start]+new+s[end:]
for graph,vals in arr.items():
 name='EVA' if graph=='EVA' else '\\graph{'+graph+'}'
 line=f"{name} &{vals['hit']}&{vals['end']}&{vals['cpu_seconds']:.3f}&"+(f"{vals['rr_seconds']:.3f}" if vals['rr_seconds'] is not None else '---')+r'\\'
 s=re.sub(re.escape(name)+r' &[^\n]+',lambda _:line,s)
s=s.replace('\\label{sec:large}\n','\\label{sec:large}\n\n'+roadfig+'\n')
p.write_text(s)
p=Path('section/05.conclusion.tex');s=p.read_text();start=s.index('For Full SGD,');end=s.index('\nA GPU',start)
s=s[:start]+f'''For Full SGD, interleaving 16 segments of each inner schedule across outer rounds gives shorter iterations on {a['faster']} of {a['comparable_graphs']} comparable graphs and a median stress ratio at most 1.05 on {a['quality_within_5pct']}. {hits} of 60 diagnostic runs reach the same-seed quality target within 15 iterations. On \\graph{{wiki-Vote}}, median time to attainment is {arr['wiki-Vote']['rr_seconds']:.3f} s, compared with {arr['wiki-Vote']['cpu_seconds']:.3f} s for CPU. The additional preferential-attachment study confirms a quality improvement over the unsplit schedule at an iteration-time cost. Two small graphs remain slower than CPU, and \\graph{{bcsstk17}} retains a quality gap. Time to an application-specific stress threshold therefore remains a necessary complement to iteration time.
''' +s[end:];p.write_text(s)
print(json.dumps({'hits':hits,'ends':ends,'degree':degree},indent=2))
