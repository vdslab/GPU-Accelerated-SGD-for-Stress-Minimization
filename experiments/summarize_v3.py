#!/usr/bin/env python3
"""Render the seven v3 studies, preserving failed and unavailable conditions.

Run after run_v3.py. Safe to rerun while results are incomplete. Timing data comes
only from benchmark records; diagnostic coordinate snapshots supply convergence
quality. Large graph stress uses the same SplitMix64 sampled sources as Rust.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'experiment_results/v3'
METHODS = ['sgd', 'atomic_sgd', 'rr_sgd', 'sparse_sgd', 'atomic_sparse_sgd', 'rr_sparse_sgd']
LABEL = dict(zip(METHODS, ['CPU', 'Atomic GPU', 'RR GPU'] * 2))
COLOR = dict(zip(METHODS, ['#777777', '#D55E00', '#0072B2'] * 2))
FULL_REPS = ['USpowerGrid', 'dwt_2680', 'EVA', 'fe_4elt2']
SPARSE_REPS = ['luxembourg_osm', 'web-Stanford', 'coAuthorsCiteseer', 'delaunay_n19']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 160, 'svg.fonttype': 'none'})


def finite(x, positive=False):
    return isinstance(x, (int, float)) and math.isfinite(x) and (x > 0 if positive else x >= 0)


def dump_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
    return rows


def path_of(value, base):
    path = Path(value)
    if path.is_absolute():
        return path
    for parent in [ROOT, base]:
        if (parent / path).exists():
            return parent / path
    return ROOT / path


def family(row):
    return 'sparse' if row.get('method') in METHODS[3:] else 'full'


def load(base):
    dataset_path = base/'config/datasets.json'
    datasets = json.loads(dataset_path.read_text()) if dataset_path.exists() else []
    if isinstance(datasets, dict):
        datasets = datasets.get('datasets', [])
    latest = {}
    for row in read_jsonl(base/'raw/results.jsonl'):
        row['graph'] = row.get('graph', row.get('dataset'))
        row['study'] = row.get('study', row.get('study_group', 'basic'))
        key = row.get('task_id', row.get('run_id')) or (row['study'], row['graph'], row.get('method'), row.get('seed'), row.get('tile_size'), row.get('pivots'))
        latest[str(key)] = row
    return datasets, list(latest.values())


def statistics(values):
    values = [float(v) for v in values if finite(v)]
    return (float(np.median(values)), min(values), max(values)) if values else (None, None, None)


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['study'], r['graph'], r.get('method'), r.get('pivots') if family(r)=='sparse' else None,
                r.get('tile_size', r.get('rr_tile_size', 1024)))].append(r)
    output = []
    for key, group in sorted(groups.items(), key=str):
        valid = [r for r in group if r.get('status')=='success']
        row = dict(zip(['study','graph','method','pivots','tile_size'], key))
        row.update(family=family(group[0]), records=len(group), success=len(valid), failures=len(group)-len(valid),
                   seeds=len({r.get('seed') for r in valid}))
        for field in ['iteration_time_ms', 'stress_value', 'algorithm_time_cold_ms', 'algorithm_time_warm_ms',
                      'common_preprocess_time_ms', 'method_setup_time_ms', 'process_wall_seconds', 'peak_rss_bytes']:
            med, low, high = statistics([r.get(field) for r in valid])
            row.update({field+'_median': med, field+'_min': low, field+'_max': high})
        output.append(row)
    return output


def pair_rows(rows):
    refs = {}
    for r in rows:
        if r.get('status')=='success' and r.get('method') in ['sgd','sparse_sgd']:
            refs[(r['study'], r['graph'], family(r), r.get('seed'), r.get('pivots') if family(r)=='sparse' else None)] = r
    pairs, excluded = [], []
    for r in rows:
        if r.get('status')!='success':
            continue
        reference_study='basic' if r['study']=='tiling' else r['study']
        ref = refs.get((reference_study, r['graph'], family(r), r.get('seed'), r.get('pivots') if family(r)=='sparse' else None))
        if not ref:
            excluded.append(dict(graph=r['graph'], method=r.get('method'), seed=r.get('seed'), study=r['study'], reason='No successful matching CPU baseline'))
            continue
        mismatch = [f for f in ['input_sha256','initial_positions_sha256','preprocess_sha256','iterations','epsilon','stress_kind','stress_samples','stress_seed']
                    if r.get(f)!=ref.get(f)]
        if mismatch:
            excluded.append(dict(graph=r['graph'], method=r.get('method'), seed=r.get('seed'), study=r['study'], reason='Mismatch: '+','.join(mismatch)))
            continue
        if not all(finite(x, positive=True) for x in [r.get('iteration_time_ms'),ref.get('iteration_time_ms'),ref.get('stress_value')]) or not finite(r.get('stress_value')):
            continue
        pairs.append(dict(study=r['study'],graph=r['graph'],family=family(r),method=r['method'],seed=r['seed'],
                          pivots=r.get('pivots'),tile_size=r.get('tile_size',r.get('rr_tile_size',1024)),
                          speedup=ref['iteration_time_ms']/r['iteration_time_ms'],stress_ratio=r['stress_value']/ref['stress_value'],
                          cpu_stress=ref['stress_value'],stress_value=r['stress_value']))
    return pairs, excluded


def save(fig, name, base, figures):
    fig.savefig(base/'figures'/f'{name}.png', bbox_inches='tight')
    fig.savefig(base/'figures'/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)
    figures.append(name)


def empty(ax, message='No successful measurement'):
    ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,color='#777777')
    ax.set_xticks([]); ax.set_yticks([])


def read_layout(path):
    positions, edges, section = [], [], None
    with path.open() as f:
        for line in f:
            if line.startswith('# Edges'):
                section = 'edges'
            elif line.startswith('# Positions'):
                section = 'positions'
            elif line.strip() and not line.startswith('#'):
                if section=='positions':
                    positions.append([float(v) for v in line.split()[:2]])
                elif section=='edges':
                    edges.append([int(v) for v in line.split()[:2]])
    return np.asarray(positions), np.asarray(edges,dtype=int).reshape(-1,2)


def layouts(base, rows, figures, incidents):
    choices = []
    for kind, graphs, methods in [('full', FULL_REPS, METHODS[:3]),('sparse',SPARSE_REPS,METHODS[3:])]:
        fig, axes = plt.subplots(len(graphs),3,figsize=(12,3.4*len(graphs)),squeeze=False)
        for i, graph in enumerate(graphs):
            eligible = [r for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')==methods[0] and r.get('status')=='success' and finite(r.get('stress_value'))]
            median = np.median([r['stress_value'] for r in eligible]) if eligible else 0
            ref = min(eligible,key=lambda r:(abs(r['stress_value']-median),r['seed'])) if eligible else None
            bounds=[]
            for j, method in enumerate(methods):
                ax = axes[i,j]
                ax.set_title(f'{graph} / {LABEL[method]}')
                candidates = [r for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')==method and ref and r.get('seed')==ref['seed']]
                row = candidates[0] if candidates else None
                if not row or row.get('status')!='success' or not row.get('final_positions_path'):
                    empty(ax, row.get('status','Unavailable') if row else 'Unavailable'); continue
                try:
                    xy, edges = read_layout(path_of(row['final_positions_path'],base))
                    original_edges=len(edges)
                    sampled = len(edges)>250000
                    if sampled:
                        edges = edges[np.linspace(0,len(edges)-1,250000,dtype=int)]
                    ax.add_collection(LineCollection(xy[edges],colors='#222222',linewidths=.15,alpha=.35,rasterized=True))
                    bounds.append((xy.min(axis=0),xy.max(axis=0)))
                    ax.autoscale(); ax.set_aspect('equal'); ax.set_axis_off()
                    fraction=len(edges)/original_edges if original_edges else 1.0
                    seconds=row['iteration_time_ms']/1000 if finite(row.get('iteration_time_ms')) else None
                    ax.text(0,-.04,f"seed {row['seed']}; stress {row['stress_value']:,.1f}\nSGD {fmt(seconds)} s; edges shown {fraction:.1%}",transform=ax.transAxes,fontsize=8)
                    choices.append(dict(graph=graph,method=method,seed=row['seed'],stress=row['stress_value'],edges_sampled=sampled,edges_shown_fraction=fraction,iteration_seconds=seconds,path=row['final_positions_path']))
                except (OSError, ValueError, IndexError) as e:
                    empty(ax,'Layout unavailable'); incidents.append(dict(graph=graph,stage='layout',error=str(e)))
            if bounds:
                lo=np.min([b[0] for b in bounds],axis=0); hi=np.max([b[1] for b in bounds],axis=0)
                center=(hi+lo)/2; half=max(float(np.max(hi-lo))*.54,1e-9)
                for ax in axes[i]:
                    ax.set_xlim(center[0]-half,center[0]+half); ax.set_ylim(center[1]-half,center[1]+half)
        fig.tight_layout(h_pad=2.8)
        save(fig,'01_layout_'+kind,base,figures)
    dump_csv(base/'tables/layout-selection.csv',choices)


def comparisons(base, datasets, pairs, figures, rows=None):
    category_order=['Road','Geometry / mesh','Web','Social / collaboration','Synthetic','Other / engineering']
    backgrounds=dict(zip(category_order,['#e5f5e0','#deebf7','#fee6ce','#efedf5','#fff7bc','#eeeeee']))
    catalog={r['name']:r for r in datasets}
    for kind, names, methods, name in [('full',[r['name'] for r in datasets if r.get('family')=='small'],METHODS[:3],'02_full'),
                                      ('sparse',[r['name'] for r in datasets if r.get('family')=='large'],METHODS[3:],'03_sparse'),
                                      ('sparse',[r['name'] for r in datasets if r.get('family')=='small'],METHODS[3:],'03_sparse_small_aux')]:
        if not names:
            names = sorted({r['graph'] for r in pairs if r['study']=='basic' and r['family']==kind})
        names=sorted(names,key=lambda graph:(category_order.index(structural_category(graph)),catalog.get(graph,{}).get('nodes') or catalog.get(graph,{}).get('source_nodes') or math.inf,graph))
        mapping=[dict(id=index+1,graph=graph,structural_category=structural_category(graph),category_basis='heuristic from graph name',
                      source_group=catalog.get(graph,{}).get('category'),nodes=catalog.get(graph,{}).get('nodes'),
                      source_nodes=catalog.get(graph,{}).get('source_nodes'),dataset_status=catalog.get(graph,{}).get('status')) for index,graph in enumerate(names)]
        dump_csv(base/'tables'/f'{name}-graph-order.csv',mapping)
        with_thumbnails=name=='03_sparse' and rows is not None
        fig, axes=plt.subplots(2,1,figsize=(max(13,len(names)*.30),12 if with_thumbnails else 9),sharex=True)
        count_rows=[]
        for ax,metric in zip(axes,['speedup','stress_ratio']):
            for index,graph in enumerate(names):
                category=structural_category(graph)
                ax.axvspan(index-.5,index+.5,color=backgrounds[category],alpha=.6,lw=0,zorder=0)
                for mi,method in enumerate(methods):
                    vals=[r[metric] for r in pairs if r['study']=='basic' and r['graph']==graph and r['method']==method]
                    med,low,high=statistics(vals); xx=index+(mi-1)*.18
                    if med is not None:
                        ax.errorbar(xx,med,yerr=[[med-low],[high-med]],fmt='o',markersize=3,color=COLOR[method],capsize=2,zorder=3)
                    missing=max(0,5-len(vals))
                    if missing:
                        # Axis-fraction placement communicates a missing observation,
                        # without inventing a numeric speed or stress ratio for it.
                        ax.plot([xx],[.035+mi*.055],marker='x',color=COLOR[method],markersize=4,ls='',transform=ax.get_xaxis_transform(),zorder=4)
                        ax.text(xx,.065+mi*.055,str(missing),color=COLOR[method],fontsize=5.5,ha='center',transform=ax.get_xaxis_transform())
                    if metric=='speedup': count_rows.append(dict(id=index+1,graph=graph,method=method,paired_successes=len(vals),expected_seeds=5,missing_or_unpaired=missing))
            ax.axhline(1,color='#777777',ls='--',lw=.8,zorder=1)
            ax.set_ylabel('CPU time / method time' if metric=='speedup' else 'Method stress / CPU stress')
            if metric=='speedup': ax.set_yscale('log')
            ax.grid(axis='y',alpha=.15)
        if not names:
            empty(axes[0]); empty(axes[1])
        labels=[f'{index+1}: {graph}' if len(names)<=25 else str(index+1) for index,graph in enumerate(names)]
        axes[1].set_xticks(range(len(names)),labels,rotation=60 if len(names)<=25 else 0,ha='right' if len(names)<=25 else 'center',fontsize=7)
        axes[1].set_xlim(-.6,max(.6,len(names)-.4))
        axes[1].set_xlabel('Graph ID (category, then vertices); see matching graph-order CSV')
        handles=[plt.Line2D([0],[0],marker='o',color=COLOR[m],ls='',label=LABEL[m]) for m in methods]
        axes[0].legend(handles=handles,loc='upper left',ncol=3)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=backgrounds[c],label=c) for c in category_order],loc='upper center',ncol=6,bbox_to_anchor=(.5,1),fontsize=8,title='Name-inferred categories (heuristic)')
        axes[0].set_title(f'{kind.title()}: paired-seed median and min–max; × + count = missing/unpaired seeds (of 5)',pad=14)
        fig.tight_layout(rect=(0,0,1,.69 if with_thumbnails else .93))
        if with_thumbnails:
            overview_thumbnails(fig,axes[0],names,rows,pairs,base)
        save(fig,name,base,figures)
        dump_csv(base/'tables'/f'{name}-counts.csv',count_rows)


def overview_thumbnails(fig,axis,names,rows,pairs,base):
    """Display RR layouts at the CPU-median seed, linked to graph summary points.

    This runs only during the final rendering step; it does not execute methods or
    touch measurement records. The small drawing samples edges for legibility.
    """
    from matplotlib.patches import ConnectionPatch
    choices=[]
    for slot,graph in enumerate(SPARSE_REPS):
        thumb=fig.add_axes([.045+slot*.245,.76,.225,.15])
        thumb.set_axis_off()
        cpu=[r for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')=='sparse_sgd' and r.get('status')=='success' and finite(r.get('stress_value'))]
        if not cpu:
            thumb.text(.5,.5,f'{graph}\nCPU baseline unavailable',ha='center',va='center',transform=thumb.transAxes,fontsize=8)
            choices.append(dict(graph=graph,status='cpu_unavailable')); continue
        median=float(np.median([r['stress_value'] for r in cpu]))
        ref=min(cpu,key=lambda r:(abs(r['stress_value']-median),r['seed']))
        row=next((r for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')=='rr_sparse_sgd' and r.get('seed')==ref['seed']),None)
        if not row or row.get('status')!='success' or not row.get('final_positions_path'):
            thumb.text(.5,.5,f'{graph}\nRR seed {ref["seed"]} unavailable',ha='center',va='center',transform=thumb.transAxes,fontsize=8)
            choices.append(dict(graph=graph,seed=ref['seed'],status='rr_unavailable')); continue
        try:
            xy,edges=read_layout(path_of(row['final_positions_path'],base))
            if xy.ndim!=2 or xy.shape[1]!=2 or not len(xy) or not np.isfinite(xy).all():
                raise ValueError('Invalid thumbnail coordinates')
            total=len(edges)
            if total>40000: edges=edges[np.linspace(0,total-1,40000,dtype=int)]
            fraction=len(edges)/total if total else 1.
            thumb.add_collection(LineCollection(xy[edges],colors='#222222',linewidths=.13,alpha=.35,rasterized=True))
            thumb.autoscale(); thumb.set_aspect('equal'); thumb.set_axis_off()
            graph_id=names.index(graph)+1 if graph in names else None
            thumb.set_title(f'{graph_id or "?"}: {graph}\nRR GPU; seed {row["seed"]}; stress {row["stress_value"]:,.0f}',fontsize=8,pad=3)
            speed=[r['speedup'] for r in pairs if r['study']=='basic' and r['graph']==graph and r['method']=='rr_sparse_sgd']
            anchor=float(np.median(speed)) if speed and graph in names else None
            if anchor is not None:
                connector=ConnectionPatch(xyA=(.5,0),coordsA='axes fraction',axesA=thumb,
                                          xyB=(names.index(graph)+.18,anchor),coordsB='data',axesB=axis,
                                          color='#444444',lw=.65,alpha=.7,clip_on=False,zorder=2)
                fig.add_artist(connector)
            choices.append(dict(graph=graph,graph_id=graph_id,method='rr_sparse_sgd',seed=row['seed'],
                                status='shown',stress=row['stress_value'],edges_shown_fraction=fraction,
                                source_path=row['final_positions_path'],anchor_speedup_median=anchor,
                                selection='same seed as CPU median-stress selection; connector anchors graph RR median over paired seeds'))
        except (OSError,ValueError,IndexError) as error:
            thumb.clear(); thumb.set_axis_off()
            thumb.text(.5,.5,f'{graph}\nLayout unavailable',ha='center',va='center',transform=thumb.transAxes,fontsize=8)
            choices.append(dict(graph=graph,status='layout_unavailable',error=str(error)))
    fig.text(.5,.735,'Thumbnails: RR at CPU-median-stress seed; connectors identify graph-level RR speedup medians. Edges sampled for display.',ha='center',fontsize=8)
    dump_csv(base/'tables/sparse-overview-thumbnails.csv',choices)


def structural_category(name):
    """Names provide a visual grouping only, not verified domain metadata."""
    lower=name.lower()
    if any(token in lower for token in ['_osm','roadnet','usroads']): return 'Road'
    if lower.startswith(('delaunay','rgg','fe_','naca','m6','333sp')): return 'Geometry / mesh'
    if lower.startswith(('web','cnr','in-2004','eu-2005','it-2004')): return 'Web'
    if lower.startswith(('coauthors','copapers','com-','soc-','loc-','wiki','ca-grqc','ca-hepth','eva','p2p')): return 'Social / collaboration'
    if lower.startswith(('kron','preferentialattachment','smallworld')): return 'Synthetic'
    return 'Other / engineering'


def structure(base,datasets,pairs,figures):
    catalog={r['name']:r for r in datasets}
    markers=dict(zip(['Road','Geometry / mesh','Web','Social / collaboration','Synthetic','Other / engineering'],['o','s','^','D','P','X']))
    dump_csv(base/'tables/graph-structure.csv',[dict(graph=r['name'],source_group=r.get('category'),
        structural_category=structural_category(r['name']),category_basis='heuristic from graph name, not verified domain metadata',
        nodes=r.get('nodes'),edges=r.get('edges'),mean_degree=r.get('mean_degree'),max_degree=r.get('max_degree'),degree_cv=r.get('degree_cv')) for r in datasets])
    correlations=[]; points=[]
    fields=['nodes','edges','mean_degree','max_degree','degree_cv']
    for metric,name in [('speedup','04_structure'),('stress_ratio','04_structure_quality')]:
        fig,axes=plt.subplots(2,5,figsize=(23,8),squeeze=False)
        for i,kind in enumerate(['full','sparse']):
            for j,field in enumerate(fields):
                ax=axes[i,j]
                for method in METHODS[1:3] if kind=='full' else METHODS[4:]:
                    xx=[]; yy=[]
                    for graph in sorted({r['graph'] for r in pairs if r['study']=='basic' and r['method']==method and catalog.get(r['graph'],{}).get('family')==('small' if kind=='full' else 'large')}):
                        vals=[r[metric] for r in pairs if r['study']=='basic' and r['graph']==graph and r['method']==method]
                        value=catalog.get(graph,{}).get(field)
                        if not finite(value,positive=field!='degree_cv') or not vals: continue
                        median=float(np.median(vals)); xx.append(value); yy.append(median)
                        ax.scatter([value],[median],color=COLOR[method],marker=markers[structural_category(graph)],s=30,alpha=.7)
                        points.append(dict(graph=graph,family=kind,method=method,metric=metric,feature=field,feature_value=value,paired_seed_median=median,paired_seeds=len(vals)))
                    ax.plot([],[],ls='',marker='o',color=COLOR[method],label=LABEL[method])
                    if len(xx)>2 and np.std(xx)>0 and np.std(yy)>0:
                        from scipy.stats import spearmanr
                        rho,p=spearmanr(xx,yy)
                        correlations.append(dict(family=kind,method=method,metric=metric,feature=field,n=len(xx),spearman_rho=float(rho),p_value=float(p)))
                if not np.isfinite(ax.dataLim.x0):
                    ax.set_xlim(.1,10); ax.set_ylim(.1,10)
                if field=='degree_cv': ax.set_xscale('symlog',linthresh=.01)
                else: ax.set_xscale('log')
                if metric=='speedup': ax.set_yscale('log')
                ax.set(xlabel=field,ylabel=f'{kind.title()} '+('speedup' if metric=='speedup' else 'stress / CPU stress'))
                ax.axhline(1,color='#bbbbbb',ls='--'); ax.legend()
        handles=[plt.Line2D([0],[0],ls='',marker=marker,color='#555555',label=category) for category,marker in markers.items()]
        fig.legend(handles=handles,loc='lower center',ncol=6,title='Graph category inferred from name (heuristic)',bbox_to_anchor=(.5,-.04))
        fig.tight_layout(); save(fig,name,base,figures)
    dump_csv(base/'tables/structure-correlations.csv',correlations)
    dump_csv(base/'tables/structure-points.csv',points)


def structure_supplement(base,datasets,rows,pairs,figures):
    catalog={r['name']:r for r in datasets}
    bins=[]
    schemes=[('mean_degree',[4,16],['<4','4–16','>=16']),('degree_cv',[.5,2],['<0.5','0.5–2','>=2'])]
    fig,axes=plt.subplots(2,2,figsize=(11,7),squeeze=False)
    for i,kind in enumerate(['full','sparse']):
        for j,(feature,bounds,labels) in enumerate(schemes):
            ax=axes[i,j]
            for mi,method in enumerate(METHODS[1:3] if kind=='full' else METHODS[4:]):
                grouped=defaultdict(list)
                for graph in sorted({r['graph'] for r in pairs if r['study']=='basic' and r['method']==method}):
                    info=catalog.get(graph,{})
                    if info.get('family')!=('small' if kind=='full' else 'large'): continue
                    value=info.get(feature)
                    if not finite(value): continue
                    index=0 if value<bounds[0] else 1 if value<bounds[1] else 2
                    vals=[r['speedup'] for r in pairs if r['study']=='basic' and r['graph']==graph and r['method']==method]
                    if vals: grouped[index].append(float(np.median(vals)))
                for index,values in grouped.items():
                    med,low,high=statistics(values)
                    ax.errorbar(index+(mi-.5)*.12,med,yerr=[[med-low],[high-med]],fmt='o',color=COLOR[method],capsize=3)
                    bins.append(dict(family=kind,method=method,feature=feature,bin=labels[index],graphs=len(values),graph_speedup_median=med,graph_speedup_min=low,graph_speedup_max=high))
                ax.plot([],[],color=COLOR[method],marker='o',ls='',label=LABEL[method])
            ax.set_xticks(range(3),labels); ax.set(xlabel=feature,ylabel=f'{kind} speedup',yscale='log'); ax.axhline(1,color='#bbbbbb',ls='--'); ax.legend()
    fig.tight_layout(); save(fig,'04_structure_bins',base,figures)
    dump_csv(base/'tables/structure-bins.csv',bins)
    def series(name):
        lower=name.lower()
        for prefix in ['delaunay','rgg','roadnet','usroads','kron','coauthors','copapers','web','fe_']:
            if lower.startswith(prefix): return prefix
        return None
    size_rows=[]
    for r in pairs:
        if r['study']!='basic' or r['method'] in ['sgd','sparse_sgd']: continue
        info=catalog.get(r['graph'],{}); group=series(r['graph'])
        if group and finite(info.get('nodes'),positive=True):
            size_rows.append(dict(series=group,graph=r['graph'],family=r['family'],method=r['method'],seed=r['seed'],nodes=info['nodes'],speedup=r['speedup']))
    panels=sorted({(r['series'],r['family']) for r in size_rows if len({x['graph'] for x in size_rows if x['series']==r['series'] and x['family']==r['family']})>=2})
    if panels:
        fig,axes=plt.subplots(math.ceil(len(panels)/3),3,figsize=(14,math.ceil(len(panels)/3)*3.5),squeeze=False)
        for ax in axes.flat: ax.set_visible(False)
        for ax,(group,kind) in zip(axes.flat,panels):
            ax.set_visible(True)
            for method in METHODS[1:3] if kind=='full' else METHODS[4:]:
                values=[]
                for graph in sorted({r['graph'] for r in size_rows if r['series']==group and r['method']==method}):
                    matches=[r for r in size_rows if r['series']==group and r['method']==method and r['graph']==graph]
                    values.append((matches[0]['nodes'],np.median([r['speedup'] for r in matches])))
                if values: ax.plot(*zip(*sorted(values)),marker='o',color=COLOR[method],label=LABEL[method])
            ax.set(title=f'{group} / {kind}',xlabel='Vertices',ylabel='Speedup',xscale='log',yscale='log'); ax.legend()
        fig.tight_layout(); save(fig,'04_size_series',base,figures)
    dump_csv(base/'tables/size-series.csv',size_rows)
    schedule=[dict(graph=r['graph'],method=r['method'],seed=r['seed'],pivots=r.get('pivots'),nodes=r.get('nodes'),constraints=r.get('constraints'),
                   constraints_per_vertex=r['constraints']/r['nodes'] if finite(r.get('nodes'),positive=True) and finite(r.get('constraints')) else None,
                   rr_rounds=r.get('rounds'),dispatches=r.get('dispatches'),
                   dispatches_per_iteration=r['dispatches']/r['iterations'] if finite(r.get('iterations'),positive=True) and finite(r.get('dispatches')) else None)
              for r in rows if r['study']=='basic' and r.get('status')=='success' and family(r)=='sparse']
    dump_csv(base/'tables/sparse-schedule.csv',schedule)


def sample_indices(n,count,seed=0):
    values=np.arange(n,dtype=np.int64); state=seed; mask=(1<<64)-1
    for i in range(min(n,count)):
        state=(state+0x9e3779b97f4a7c15)&mask
        state=((state^(state>>30))*0xbf58476d1ce4e5b9)&mask
        state=((state^(state>>27))*0x94d049bb133111eb)&mask
        state^=state>>31
        j=i+state%(n-i); values[i],values[j]=values[j],values[i]
    return values[:min(n,count)]


def checkpoint_stress(xy,dist,sources,exact):
    n=len(xy); total=0.
    for start in range(0,len(sources),16):
        ss=sources[start:start+16]
        dd=np.asarray(dist[start:start+16],dtype=np.float64)
        dx=xy[ss,0,None]-xy[None,:,0]
        dy=xy[ss,1,None]-xy[None,:,1]
        actual=np.sqrt(dx*dx+dy*dy)
        valid=(dd>0)&np.isfinite(dd)&(dd<np.iinfo(np.uint32).max)
        if exact:
            valid&=np.arange(n)[None,:]>ss[:,None]
        np.subtract(actual,dd,out=actual)
        np.divide(actual,dd,out=actual,where=valid)
        actual[~valid]=0
        total+=float(np.sum(actual*actual))
    return total if exact else total*n/(2*len(sources))


def load_trace(row,base):
    directory=path_of(row.get('trace_dir',str(Path(row.get('artifact_dir',''))/'trace')),base)
    trace_path=directory/'trace.jsonl'
    entries=read_jsonl(trace_path)
    for e in entries:
        e['epoch']=e.get('epoch',e.get('iteration',e.get('checkpoint',0)))
        e['elapsed_ms']=e['cumulative_algorithm_seconds']*1000 if e.get('cumulative_algorithm_seconds') is not None else e.get('cumulative_iteration_ms',e.get('cumulative_iteration_time_ms',e.get('host_elapsed_ms',e.get('elapsed_ms',e.get('iteration_time_ms',0)))))
        value=e.get('snapshot',e.get('positions_path',e.get('path',f"positions-{e['epoch']:04}.f64le")))
        if not value:
            e['snapshot']=None; continue
        p=Path(value)
        e['snapshot']=str(p if p.is_absolute() else directory/p)
    if not entries and directory.exists():
        for file in sorted(directory.glob('positions-*.f64le')):
            entries.append(dict(epoch=int(file.stem.split('-')[-1]),elapsed_ms=None,snapshot=str(file)))
    return entries


def convergence(base,datasets,rows,figures,incidents,skip=False):
    cache_path=base/'tables/convergence.csv'
    old=[]
    if cache_path.exists():
        with cache_path.open() as f:
            old=list(csv.DictReader(f))
    cached={r['cache_key']:r for r in old if r.get('cache_key')}
    output=[]
    catalog={r['name']:r for r in datasets}
    groups=defaultdict(list)
    for row in rows:
        if row['study']=='convergence' and row.get('status')=='success':
            groups[row['graph']].append(row)
    if not skip:
        from scipy.io import mmread
        from scipy.sparse.csgraph import shortest_path
        (base/'cache').mkdir(exist_ok=True)
        for graph,group in groups.items():
            snaps=[]
            for r in group:
                for e in load_trace(r,base):
                    if not e.get('snapshot'): continue
                    p=Path(e['snapshot'])
                    if not p.exists():
                        incidents.append(dict(graph=graph,stage='trace',error=f'Missing snapshot {p}')); continue
                    key=hashlib.sha256((str(p)+str(p.stat().st_size)+str(p.stat().st_mtime_ns)).encode()).hexdigest()
                    if key in cached:
                        output.append({k:float(v) if k in ['stress','elapsed_ms'] and v else int(v) if k in ['seed','epoch','stress_samples'] and v else v for k,v in cached[key].items()})
                    else:
                        snaps.append((r,e,key))
            if not snaps:
                continue
            info=catalog.get(graph,{})
            graphpath=path_of(info.get('path',group[0].get('input_path','')),base)
            matrix=mmread(graphpath).tocsr(); matrix=matrix.maximum(matrix.T)
            n=matrix.shape[0]; exact=n<=8000
            count=n if exact else min(256,n)
            sources=np.arange(n) if exact else sample_indices(n,count,0)
            signature=hashlib.sha256((str(info.get('sha256',graph))+f':{n}:{count}:uint32:v1').encode()).hexdigest()[:20]
            distpath=base/'cache'/f'distance-{signature}.u32'
            complete=distpath.with_suffix('.complete')
            if not complete.exists():
                dist=np.memmap(distpath,mode='w+',dtype='<u4',shape=(count,n))
                for start in range(0,count,16):
                    dd=shortest_path(matrix,directed=False,unweighted=True,indices=sources[start:start+16])
                    dd[~np.isfinite(dd)]=np.iinfo(np.uint32).max
                    dist[start:start+16]=dd.astype(np.uint32)
                dist.flush(); del dist
                complete.write_text(signature)
            dist=np.memmap(distpath,mode='r',dtype='<u4',shape=(count,n))
            for row,e,key in snaps:
                xy=np.fromfile(e['snapshot'],dtype='<f8').reshape(-1,2)
                if len(xy)!=n or not np.isfinite(xy).all():
                    incidents.append(dict(graph=graph,stage='trace',error=f'Invalid coordinates {e["snapshot"]}')); continue
                stress=checkpoint_stress(xy,dist,sources,exact)
                output.append(dict(cache_key=key,graph=graph,method=row['method'],family=family(row),seed=row['seed'],epoch=e['epoch'],
                                   elapsed_ms=e['elapsed_ms'],stress=stress,stress_kind='exact' if exact else 'sampled',stress_samples=count,
                                   stress_seed=0,snapshot=e['snapshot']))
                print(f'convergence {graph} {row["method"]} seed={row["seed"]} epoch={e["epoch"]}',flush=True)
                dump_csv(cache_path,output)
            del dist
    else:
        output=[{k:float(v) if k in ['stress','elapsed_ms'] and v else int(v) if k in ['seed','epoch','stress_samples'] and v else v for k,v in r.items()} for r in old]
    dump_csv(cache_path,output)
    panels=sorted({(r['graph'],r['family']) for r in output})
    fig,axes=plt.subplots(max(1,math.ceil(len(panels)/3)),3,figsize=(15,max(4,math.ceil(len(panels)/3)*3.5)),squeeze=False)
    arrivals=[]
    for ax in axes.flat: ax.set_visible(False)
    for ax,(graph,kind) in zip(axes.flat,panels):
        ax.set_visible(True); methods=METHODS[:3] if kind=='full' else METHODS[3:]
        subset=[r for r in output if r['graph']==graph and r['family']==kind]
        targets={r['seed']:r['stress']*1.05 for r in subset if r['method']==methods[0] and r['epoch']==15}
        for method in methods:
            points=[r for r in subset if r['method']==method and finite(r.get('elapsed_ms'))]
            for epoch in sorted({r['epoch'] for r in points}):
                chosen=[r for r in points if r['epoch']==epoch]
                ax.scatter(np.median([r['elapsed_ms'] for r in chosen])/1000,np.median([r['stress'] for r in chosen]),color=COLOR[method],s=10)
            medians=[(np.median([r['elapsed_ms'] for r in points if r['epoch']==e])/1000,np.median([r['stress'] for r in points if r['epoch']==e])) for e in sorted({r['epoch'] for r in points})]
            if medians: ax.plot(*zip(*medians),color=COLOR[method],label=LABEL[method])
        ax.set(title=f'{graph} / {kind}',xlabel='Cumulative SGD time (s)',ylabel='Full stress',yscale='log'); ax.legend()
    if not panels: empty(axes[0,0],'No convergence snapshots evaluated'); axes[0,0].set_visible(True)
    fig.tight_layout(); save(fig,'05_convergence',base,figures)
    arrivals=build_arrivals(base,rows,output)
    dump_csv(base/'tables/target-arrival.csv',arrivals)
    return output,arrivals


def build_arrivals(base,rows,checkpoints):
    """Retain every planned seed, including trials that produced no snapshot."""
    planpath=base/'config/plan.json'
    plan=json.loads(planpath.read_text()) if planpath.exists() else rows
    if isinstance(plan,dict): plan=plan.get('jobs',plan.get('tasks',[]))
    conditions={(r['graph'],r['method'],r['seed']):r for r in plan if r.get('study')=='convergence'}
    actual={(r['graph'],r['method'],r['seed']):r for r in rows if r['study']=='convergence'}
    traces=defaultdict(list)
    for r in checkpoints:
        if finite(r.get('elapsed_ms')): traces[(r['graph'],r['method'],r['seed'])].append(r)
    required={0,1,2,3,5,8,10,15}; arrivals=[]
    for key,condition in sorted(conditions.items()):
        graph,method,seed=key; kind=family(condition)
        seq=sorted(traces[key],key=lambda r:r['epoch'])
        observed={r['epoch'] for r in seq}; complete=required<=observed
        cpu_method='sgd' if kind=='full' else 'sparse_sgd'
        cpu_final=next((r for r in traces[(graph,cpu_method,seed)] if r['epoch']==15),None)
        own_status=actual.get(key,{}).get('status','not_run')
        for factor in [1.05,1.01]:
            target=cpu_final['stress']*factor if cpu_final else None
            hit=next((r for r in seq if target is not None and r['stress']<=target),None)
            previous=seq[max(0,seq.index(hit)-1)] if hit else None
            evaluable=own_status=='success' and complete and target is not None
            if not seq or own_status!='success': state='measurement_unavailable'; reached=None
            elif target is None: state='reference_unavailable'; reached=None
            elif not complete: state='partial_observation'; reached=True if hit else None
            else: state='reached' if hit else 'not_observed'; reached=hit is not None
            arrivals.append(dict(graph=graph,family=kind,method=method,seed=seed,target_factor=factor,target_stress=target,
                                 first_observed_epoch=hit['epoch'] if hit else None,first_observed_ms=hit['elapsed_ms'] if hit else None,
                                 interval_lower_epoch=previous['epoch'] if previous else None,interval_lower_ms=previous['elapsed_ms'] if previous else None,
                                 interval_upper_epoch=hit['epoch'] if hit else None,interval_upper_ms=hit['elapsed_ms'] if hit else None,
                                 reached=reached,evaluable=evaluable,status=state,run_status=own_status,
                                 observed_epochs=sorted(observed),missing_epochs=sorted(required-observed)))
    coverage=[]
    for key in sorted({(r['graph'],r['family'],r['method']) for r in arrivals}):
        graph,kind,method=key
        a=[r for r in arrivals if (r['graph'],r['family'],r['method'])==key and r['target_factor']==1.05]
        b=[r for r in arrivals if (r['graph'],r['family'],r['method'])==key and r['target_factor']==1.01]
        coverage.append(dict(graph=graph,family=kind,method=method,planned_seeds=len(a),evaluable_seeds=sum(r['evaluable'] for r in a),
                             reached_1_05=sum(r['evaluable'] and r['reached'] is True for r in a),reached_1_01=sum(r['evaluable'] and r['reached'] is True for r in b),
                             missing_seeds=sum(not r['evaluable'] for r in a),measurement_unavailable=sum(r['status']=='measurement_unavailable' for r in a),
                             reference_unavailable=sum(r['status']=='reference_unavailable' for r in a),partial_observation=sum(r['status']=='partial_observation' for r in a)))
    dump_csv(base/'tables/target-arrival-coverage.csv',coverage)
    return arrivals


def iteration_timings(base,rows,figures):
    """Read clean per-iteration times from ordinary benchmark traces."""
    reps=FULL_REPS+SPARSE_REPS
    records=[]
    for row in rows:
        if row['study']!='basic' or row.get('status')!='success': continue
        for entry in load_trace(row,base):
            if entry['epoch']==0 or entry.get('algorithm_seconds') is None: continue
            records.append(dict(graph=row['graph'],method=row['method'],family=family(row),seed=row['seed'],
                                iteration=entry['epoch'],algorithm_seconds=entry['algorithm_seconds'],
                                cumulative_algorithm_seconds=entry.get('cumulative_algorithm_seconds')))
    dump_csv(base/'tables/per-iteration-time.csv',records)
    panels=[(graph,kind) for graph in reps for kind in ['full','sparse']
            if any(r['graph']==graph and r['family']==kind for r in records)]
    if not panels: return
    fig,axes=plt.subplots(math.ceil(len(panels)/3),3,figsize=(15,math.ceil(len(panels)/3)*3),squeeze=False)
    for ax in axes.flat: ax.set_visible(False)
    for ax,(graph,kind) in zip(axes.flat,panels):
        ax.set_visible(True)
        for method in METHODS[:3] if kind=='full' else METHODS[3:]:
            subset=[r for r in records if r['graph']==graph and r['method']==method]
            if not subset: continue
            xx=[]; yy=[]; lo=[]; hi=[]
            for iteration in sorted({r['iteration'] for r in subset}):
                med,mn,mx=statistics([r['algorithm_seconds'] for r in subset if r['iteration']==iteration])
                if med is None: continue
                xx.append(iteration); yy.append(med); lo.append(mn); hi.append(mx)
            ax.plot(xx,yy,color=COLOR[method],label=LABEL[method])
            ax.fill_between(xx,lo,hi,color=COLOR[method],alpha=.15)
        ax.set(title=f'{graph} / {kind}',xlabel='Iteration',ylabel='SGD time (s)',yscale='log'); ax.legend()
    fig.tight_layout(); save(fig,'05_iteration_times',base,figures)
    full_graphs=sorted({r['graph'] for r in records if r['family']=='full'})
    fig,axes=plt.subplots(max(1,math.ceil(len(full_graphs)/4)),4,figsize=(18,max(3,math.ceil(len(full_graphs)/4)*3)),squeeze=False)
    for ax in axes.flat: ax.set_visible(False)
    for ax,graph in zip(axes.flat,full_graphs):
        ax.set_visible(True)
        for method in METHODS[:3]:
            subset=[r for r in records if r['graph']==graph and r['method']==method]
            if not subset: continue
            epochs=sorted({r['iteration'] for r in subset})
            yy=[np.median([r['algorithm_seconds'] for r in subset if r['iteration']==e]) for e in epochs]
            ax.plot(epochs,yy,color=COLOR[method],label=LABEL[method])
        ax.set(title=graph,xlabel='Iteration',ylabel='SGD time (s)',yscale='log'); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig,'02_full_iteration_times',base,figures)


def sparse_components(base,datasets,rows,figures):
    names=[r['name'] for r in datasets if r.get('family')=='large']
    if not names: return
    fields=[('common_preprocess_time_ms','Common preprocess','#999999'),
            ('method_setup_time_ms','Method setup','#E69F00'),
            ('runtime_init_time_ms','GPU initialization','#CC79A7'),
            ('upload_time_ms','Upload','#56B4E9'),
            ('iteration_time_ms','SGD','#0072B2'),
            ('readback_time_ms','Readback','#009E73'),
            ('postprocess_time_ms','Postprocess','#F0E442')]
    chunks=[names[i:i+20] for i in range(0,len(names),20)]
    fig,axes=plt.subplots(1,len(chunks),figsize=(7*len(chunks),8),squeeze=False)
    for ax,chunk in zip(axes[0],chunks):
        for mi,method in enumerate(METHODS[3:]):
            left=np.zeros(len(chunk))
            for field,label,color in fields:
                vals=[]
                for graph in chunk:
                    valid=[r.get(field) for r in rows if r['study']=='basic' and r['graph']==graph and r['method']==method and r.get('status')=='success' and finite(r.get(field))]
                    vals.append(float(np.mean(valid))/1000 if valid else np.nan)
                ax.barh(np.arange(len(chunk))+(mi-1)*.23,vals,height=.21,left=left,color=color,label=label if mi==0 else None)
                left+=np.nan_to_num(vals)
        ax.set_yticks(range(len(chunk)),chunk); ax.invert_yaxis()
        ax.set_xlabel('Mean measured components (s); each group: CPU / Atomic / RR')
    axes[0,0].legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=3,fontsize=7)
    fig.tight_layout(); save(fig,'03_sparse_components',base,figures)


def sensitivity(base,rows,pairs,figures):
    for study,param,name in [('tiling','tile_size','06_tiles'),('pivots','pivots','07_pivots')]:
        selected=[r for r in rows if r['study']==study]
        graphs=sorted({r['graph'] for r in selected})
        count=3 if study=='pivots' else 2
        fig,axes=plt.subplots(max(1,len(graphs)),count,figsize=(5*count,max(3,len(graphs)*2.8)),squeeze=False)
        for gi,graph in enumerate(graphs):
            for method in sorted({r['method'] for r in selected if r['graph']==graph},key=METHODS.index):
                group=[r for r in selected if r['graph']==graph and r['method']==method and r.get('status')=='success']
                for col,field in enumerate((['algorithm_time_cold_ms','stress_value','peak_rss_bytes'] if count==3 else ['iteration_time_ms','stress_value'])):
                    ax=axes[gi,col]; xx=[]; yy=[]; ll=[]; hh=[]
                    for value in sorted({r.get(param,r.get('rr_tile_size')) for r in group if r.get(param,r.get('rr_tile_size')) is not None}):
                        med,lo,hi=statistics([r.get(field) for r in group if r.get(param,r.get('rr_tile_size'))==value])
                        if med is not None:
                            divisor=1000 if field in ['iteration_time_ms','algorithm_time_cold_ms'] else 1024**3 if field=='peak_rss_bytes' else 1
                            xx.append(value); yy.append(med/divisor); ll.append((med-lo)/divisor); hh.append((hi-med)/divisor)
                    if xx: ax.errorbar(xx,yy,yerr=[ll,hh],fmt='o-',color=COLOR[method],label=LABEL[method],markersize=3,capsize=2)
                    if not xx and (not np.isfinite(ax.dataLim.x0) or ax.dataLim.x1 <= 0):
                        ax.set_xlim(1,1024); ax.set_ylim(.1,10)
                    ax.set(title=graph,xlabel=param,ylabel={'iteration_time_ms':'SGD time (s)','algorithm_time_cold_ms':'Cold algorithm total (s)','stress_value':'Full stress','peak_rss_bytes':'Peak process RSS (GiB)'}[field],xscale='log',yscale='log')
                    ax.legend()
        if not graphs:
            for ax in axes.flat: empty(ax)
        fig.tight_layout(); save(fig,name,base,figures)
    # Separate component table preserves preprocessing and cold timing not visible in the principal panels.


def fmt(value,digits=3):
    return f'{value:,.{digits}f}' if finite(value) else '—'


def report(base,datasets,rows,summary,pairs,figures,arrivals,incidents,excluded):
    status=Counter(r.get('status','unknown') for r in rows)
    counts=Counter(r.get('status','unknown') for r in datasets)
    actual=[r for r in rows if not r.get('reused_from') and finite(r.get('process_wall_seconds')) and r.get('executed',True)]
    actual_status=Counter(r.get('status','unknown') for r in actual)
    reused=[r for r in rows if r.get('reused_from')]
    blocked=[r for r in rows if r.get('status')=='warmup_failed']
    warmups=[]
    for p in sorted((base/'warmups').glob('*/record.json')):
        try: warmups.append(json.loads(p.read_text()))
        except (OSError,json.JSONDecodeError): pass
    warmup_actual=[r for r in warmups if finite(r.get('process_wall_seconds'))]
    warmup_failures=[r for r in warmup_actual if r.get('status')!='success']
    warmup_status=Counter(r.get('status','unknown') for r in warmup_actual)
    planpath=base/'config/plan.json'
    planned=None
    if planpath.exists():
        plan=json.loads(planpath.read_text()); jobs=plan.get('jobs',plan.get('tasks',[])) if isinstance(plan,dict) else plan
        if isinstance(jobs,list): planned=len(jobs)
    catalog={r['name']:r for r in datasets}
    text=['# v3 実験結果','',f'データ {len(datasets)} グラフ（'+ '、'.join(f'{k}: {v}' for k,v in counts.items())+'）。',
          f'計画条件の記録 {len(rows):,} 件：'+ '、'.join(f'{k}: {v:,}' for k,v in status.items())+'。'+(f' 計画 {planned:,} 件。' if planned is not None else ''),'',
          f'本測定で実際に起動した実行 {len(actual):,} 件（'+ '、'.join(f'{k}: {v:,}' for k,v in actual_status.items())+f'）。既存結果の再利用 {len(reused):,} 条件、warmup失敗による未起動 {len(blocked):,} 条件。',
          f'別枠のwarmupは実起動 {len(warmup_actual):,} 件（'+ '、'.join(f'{k}: {v:,}' for k,v in warmup_status.items())+'）。warmupは性能・stressの本集計に混ぜない。','',
          '速度は前処理を除いたSGD反復の実時間。速度比は同じグラフ・seed・family・pivot数のCPU時間÷各手法時間。stress比は各手法÷同じCPU。点はseedの中央値、範囲は最小〜最大。失敗・未取得の条件は0に置き換えない。',
          'FullとSparseの絶対stressは同じ評価定義を用いる。大規模グラフは固定256起点による推定値で、全点対の厳密値とは区別する。',
          'SGD時間は各反復の処理区間の合計で、CPUのshuffle、GPUへの投入と完了待ちを含む。前後の初期化やcheckpoint座標取得・保存は含まない。CPUの反復前ペアclone等はこの区間の外。cold総時間は別に保存する。',
          'esa計画との差：Sparseの頂点別書込集中度は未計測で、制約数/頂点数・RRラウンド・dispatch数のみsparse-schedule.csvに保存する。',
          '以下は現在保存されている記録の集計。実行中の再集計では未完了条件が欠ける。','']
    captions={1:'代表グラフの描画',2:'小規模20グラフ：Full SGD',3:'大規模80グラフ：Sparse SGD',4:'グラフ構造と速度',5:'収束曲線と目標stress到達',6:'タイルサイズ',7:'pivot数'}
    for number in range(1,8):
        text += [f'## {number}. {captions[number]}','']
        names=[name for name in figures if name.startswith(f'{number:02}_')]
        for name in names: text += [f'![{captions[number]}](../figures/{name}.png)','']
        if number==1:
            text += ['CPUの最終stress中央値に最も近いseedを選び、GPUも同じseedを表示する。良好なGPU実行だけを選抜していない。同じグラフの全手法は座標のunion範囲から同じx/y縮尺で描画する。25万辺を超える描画は辺のみ等間隔で間引き、表示割合とSGD秒を併記する。stressは間引く前の全グラフで評価する。','']
        elif number in [2,3]:
            kind='full' if number==2 else 'sparse'
            for method in (METHODS[1:3] if number==2 else METHODS[4:]):
                grouped=defaultdict(list)
                for r in pairs:
                    if r['study']=='basic' and r['family']==kind and r['method']==method and catalog.get(r['graph'],{}).get('family')==('small' if kind=='full' else 'large'): grouped[r['graph']].append(r)
                if grouped:
                    speeds=[np.median([r['speedup'] for r in group]) for group in grouped.values()]
                    quality=[np.median([r['stress_ratio'] for r in group]) for group in grouped.values()]
                    text += [f'- {LABEL[method]}：対応比較できた {len(grouped)} グラフ中 {sum(s>1 for s in speeds)} グラフでCPUより高速。グラフ別速度比 {min(speeds):.2f}〜{max(speeds):.2f}倍、stress比 {min(quality):.3f}〜{max(quality):.3f}。']
            if number==3: text += ['', 'Sparse80 overviewにはRR GPUの代表描画4件を添える。seedは実験1と同じCPU stress中央値に近いものを選び、接続線は対応するグラフのRR速度比中央値を指す。描画seed単独の速度比を指す線ではない。取得・成功していない描画は表示しない。小規模20グラフのSparse補助比較も別図で表示する。大規模80グラフの集計とは混ぜない。時間内訳図の積み上げは計測できた成分の平均で、cold総時間との差には区間外の準備処理が含まれる。']
            text += ['', '上下2段は全グラフの共通x軸。背景は名前から推定したカテゴリで、分類内を頂点数順に並べる。graph-order.csvにID・名前を、counts.csvに対応seed数を保存。下端の×と数は欠測・CPU基準欠如による未対応seed数（5件中）で、0や1の測定値ではない。比は対応する成功seedのみで計算する。','']
        elif number==4:
            text += ['頂点数V・辺数E・次数の平均・最大値・変動係数の5指標を、Full/Sparseの上下2段で速度比と比較する。品質比も同じ2×5構成の補助図で示す。各グラフは対応seed比の中央値1点とし、Spearman CSVはmetric列で速度と品質を区別する。CV=0はsymlog軸で保持する。次数だけで実行時間は決まらず、頂点数・制約数・タイル数との交絡がある。相関係数はstructure-correlations.csvに保存し、因果関係として扱わない。点形状の構造カテゴリはグラフ名からの推定であり、確認済みの用途分類ではない。元のSuiteSparseグループはsource_groupとしてgraph-structure.csvに保存する。平均次数4/16とCV0.5/2で分けた図はグラフごとの速度比中央値を再集計し、グラフを等重みにする。同系列のサイズ曲線もグラフ名prefixからの推定系列で、規模以外の構造差を制御した実験ではない。','']
        elif number==5:
            text += ['座標保存を伴う診断実行から、反復0・1・2・3・5・8・10・15のstressを後計算する。GPUとCPUで同じ起点を使う。曲線はcheckpointごとの各手法の成功seed中央値であり、実際の単一実行の軌跡ではない。失敗により手法間の成功seed集合が異なる場合がある。',
                     '目標は同じseed・familyのCPU 15反復後stressの1.05倍と1.01倍。直前の未達checkpoint〜最初に到達を観測したcheckpointの時間幅を記録する。stressは非単調になり得るため、以前の観測点間で一度到達して再悪化した可能性までは除外できない。正確な初到達時刻や未到達条件の外挿は行わない。診断の同期・保存による影響があるため、図2・3の通常実行時間とは分けて読む。','']
            for factor in [1.05,1.01]:
                for method in METHODS:
                    a=[r for r in arrivals if r['method']==method and r['target_factor']==factor and r['evaluable']]
                    unavailable=sum(r['method']==method and r['target_factor']==factor and not r['evaluable'] for r in arrivals)
                    if a or unavailable: text += [f'- {method} / {factor:.2f}倍目標：必要checkpointとCPU基準が揃った {len(a)} 実行中 {sum(r["reached"] for r in a)} 実行で到達を観測。観測またはCPU基準の欠測 {unavailable} 実行は到達率の分母から除外。']
            text += ['', '計画上の全10seedを残し、観測なし・CPU基準なし・checkpoint不完全を未達と区別する。下表の欠測は到達率の分母に含めない。', '', '| Graph | Family | Method | 計画seed | 評価可能 | 1.05倍到達 | 1.01倍到達 | 欠測 |', '|---|---|---|---:|---:|---:|---:|---:|']
            coverage_file=base/'tables/target-arrival-coverage.csv'
            if coverage_file.exists():
                with coverage_file.open() as stream:
                    for r in csv.DictReader(stream):
                        text += [f'| {r["graph"]} | {r["family"]} | {r["method"]} | {r["planned_seeds"]} | {r["evaluable_seeds"]} | {r["reached_1_05"]} | {r["reached_1_01"]} | {r["missing_seeds"]} |']
            text += ['']
        else:
            study='tiling' if number==6 else 'pivots'; param='tile_size' if number==6 else 'pivots'
            for graph in sorted({r['graph'] for r in summary if r['study']==study}):
                timing='iteration_time_ms_median' if number==6 else 'algorithm_time_cold_ms_median'
                entries=[r for r in summary if r['study']==study and r['graph']==graph and finite(r.get(timing),positive=True)]
                if entries:
                    best=min(entries,key=lambda r:r[timing])
                    text += [f'- {graph}：今回測れた最短条件は {best["method"]}、{param}={best[param]}、SGD {best["iteration_time_ms_median"]/1000:.3f} 秒、stress {fmt(best["stress_value_median"],1)}。'+(f' cold総時間 {best[timing]/1000:.3f} 秒。' if number==7 else '')]
            text += ['', '時間だけで選ぶと品質を落とす可能性があるため、同じ条件のstressも併記する。最短条件は今回の測定範囲での値。' if number==6 else 'pivot増加は制約数・前処理・メモリと品質の交換になる。主図の時間は前処理・GPU初期化・SGD・readbackを含むcoldアルゴリズム総時間で、入力読込・stress評価・描画ファイル保存を含まない。summary.csvにはSGD単独・前処理・process wall・peak RSSも保存。Peak RSSはGPU専用VRAM量ではない。','']
    text += ['## 事故・欠測と調査','']
    failures=[r for r in rows if r.get('status')!='success']
    dump_csv(base/'tables/actual-attempts.csv',actual)
    dump_csv(base/'tables/warmup-attempts.csv',warmup_actual)
    dump_csv(base/'tables/warmup-failures.csv',warmup_failures)
    if failures or warmup_failures:
        actual_failures=[r for r in actual if r.get('status')!='success']
        causes=Counter((r.get('status','unknown'),str(r.get('error_message') or r.get('error') or r.get('reason') or '記録参照')) for r in actual_failures)
        for (state,why),count in causes.most_common(20):
            text += [f'- {state} {count} 件：{why[:600]}']
        if warmup_failures:
            text += [f'- warmupの実失敗 {len(warmup_failures)} 件：'+ '、'.join(f'{state} {count}' for state,count in Counter(r.get('status') for r in warmup_failures).items())+'。']
            for r in warmup_failures[:15]:
                text += [f'- warmup {r.get("graph",r.get("dataset",""))} / {r.get("method")}：{r.get("status")}、{str(r.get("error","記録参照"))[:400]}']
        if blocked:
            text += [f'- 未起動の原因別条件数：'+ '、'.join(f'{state} {count}' for state,count in Counter(r.get('failure_reason','unknown') for r in blocked).items())+f'。対応するwarmup_run_idは {len({r.get("warmup_run_id") for r in blocked if r.get("warmup_run_id")})} 件。']
            text += [f'- warmup_failed {len(blocked)} 条件はwarmupの失敗を受けて起動しなかった条件。warmup_run_idで失敗した実行に対応する。同じwarmupが5seedを止めても、実事故は1件として数える。']
        unavailable=sum(r.get('status')=='unavailable' for r in rows)
        if unavailable: text += [f'- 入力データ未取得による未起動 {unavailable} 条件。']
        text += ['', 'timeoutは計画した時間上限で停止した実行で、完了時間の実測値ではない。memory_limitは資源制限であり、そのグラフに対する性能やstressを0とみなさない。取得不能データの類似グラフへの差し替えは行わない。','']
    else: text += ['現在の実行記録に失敗はない。','']
    if excluded: text += [f'- CPU対応比較から {len(excluded)} 記録を除外。理由はpair-exclusions.csv。','']
    for r in incidents[:15]: text += [f'- 集計時 {r.get("graph", "")} / {r.get("stage", "")}：{r.get("error", "")}']
    text += ['', '## 保存先','', '- [全条件の集計](../tables/summary.csv)', '- [同一seedのCPU対応比](../tables/paired-ratios.csv)',
             '- [収束checkpoint](../tables/convergence.csv)', '- [目標到達](../tables/target-arrival.csv)', '- [失敗実行](../tables/failures.csv)',
             '- [データの規模・由来](../config/datasets.json)', '', '## 条件ごとの数値','',
             '| Study | Graph | Method | P | T | 成功 / 記録 | SGD秒 中央値 | stress 中央値 | RSS GiB |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        text += [f'| {r["study"]} | {r["graph"]} | {r["method"]} | {r["pivots"] or "—"} | {r["tile_size"]} | {r["success"]}/{r["records"]} | {fmt(r["iteration_time_ms_median"]/1000 if r["iteration_time_ms_median"] is not None else None)} | {fmt(r["stress_value_median"],1)} | {fmt(r["peak_rss_bytes_median"]/1024**3 if r["peak_rss_bytes_median"] is not None else None)} |']
    report='\n'.join(text)+'\n'
    (base/'report/RESULTS.md').write_text(report)
    (base/'report/esa-results.md').write_text(report.split('## 保存先')[0])
    (base/'README.md').write_text('# v3 実験\n\n[結果](report/RESULTS.md) · [集計CSV](tables/summary.csv)\n\n再集計：`python3 experiments/summarize_v3.py`\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=BASE)
    parser.add_argument('--skip-convergence',action='store_true',help='Reuse evaluated checkpoints without recomputing stress')
    args=parser.parse_args(argv); base=args.results_dir.resolve()
    for folder in ['figures','tables','report']: (base/folder).mkdir(parents=True,exist_ok=True)
    datasets,rows=load(base); summary=aggregate(rows); pairs,excluded=pair_rows(rows)
    incidents=[]; figures=[]
    layouts(base,rows,figures,incidents); comparisons(base,datasets,pairs,figures,rows); structure(base,datasets,pairs,figures)
    structure_supplement(base,datasets,rows,pairs,figures)
    checkpoints,arrivals=convergence(base,datasets,rows,figures,incidents,args.skip_convergence)
    sensitivity(base,rows,pairs,figures)
    iteration_timings(base,rows,figures); sparse_components(base,datasets,rows,figures)
    for name,data in [('runs',rows),('summary',summary),('paired-ratios',pairs),('pair-exclusions',excluded),('failures',[r for r in rows if r.get('status')!='success']),('summary-incidents',incidents)]:
        dump_csv(base/'tables'/f'{name}.csv',data)
    report(base,datasets,rows,summary,pairs,figures,arrivals,incidents,excluded)
    (base/'tables/provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=len(rows),figures=figures,
        source_results_sha256=hashlib.sha256((base/'raw/results.jsonl').read_bytes()).hexdigest() if (base/'raw/results.jsonl').exists() else None,
        commits=sorted({r['git_commit'] for r in rows if r.get('git_commit')})),indent=2)+'\n')
    print(f'{len(rows)} records; {len(pairs)} paired comparisons; {len(checkpoints)} checkpoints; {len(figures)} figures')

if __name__=='__main__': main()
