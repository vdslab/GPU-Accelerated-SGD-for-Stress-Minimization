#!/usr/bin/env python3
"""Render the seven v3 studies, preserving failed and unavailable conditions.

Run after run_v3.py. Safe to rerun while results are incomplete. Timing data comes
only from benchmark records; diagnostic coordinate snapshots supply convergence
quality. Large graph stress uses the same SplitMix64 sampled sources as Rust.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
try:
    from v3_recovery import load_effective_cohort
except ModuleNotFoundError:
    from experiments.v3_recovery import load_effective_cohort

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


def load(base, with_cohort=False):
    datasets,rows,cohort=load_effective_cohort(base)
    return (datasets,rows,cohort) if with_cohort else (datasets,rows)


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
                      'common_preprocess_time_ms', 'method_setup_time_ms', 'process_wall_seconds', 'peak_rss_bytes', 'archive_seconds']:
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
    path=base/'figures'/f'{name}.png'
    # esa accepts images below 10 MiB. Render only during postprocessing;
    # preserve an SVG separately if the PNG needs a lower export resolution.
    for dpi in [160,140,120,100,80,60]:
        fig.savefig(path,bbox_inches='tight',dpi=dpi,pil_kwargs={'compress_level':9})
        if path.stat().st_size<10*1024**2: break
    else:
        plt.close(fig)
        raise ValueError(f'PNG exceeds esa 10 MiB limit after resolution adjustment: {path}')
    fig.savefig(base/'figures'/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)
    figures.append(name)


def empty(ax, message='No successful measurement'):
    ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,color='#777777')
    ax.set_xticks([]); ax.set_yticks([])


def open_layout_text(path):
    """Stream the exact saved text, whether stored plain or losslessly gzipped."""
    path=Path(path)
    return gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open('r',encoding='utf-8')


def read_layout(path):
    positions, edges, section = [], [], None
    with open_layout_text(path) as f:
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


def layout_reference(rows, graph, cpu_method):
    """Select No.1's CPU-median seed, or fixed planned seed 0 without a CPU success.

    The fallback is a post-hoc display supplement only. It never searches GPU
    successes and never changes paired performance/quality comparisons.
    """
    cpu = [r for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')==cpu_method and r.get('status')=='success']
    eligible = [r for r in cpu if finite(r.get('stress_value'))]
    median = np.median([r['stress_value'] for r in eligible]) if eligible else None
    ref = min(eligible,key=lambda r:(abs(r['stress_value']-median),r['seed'])) if eligible else None
    fallback = not cpu
    rule = 'cpu_median_stress' if ref else 'planned_seed0_no_cpu_success' if fallback else 'cpu_stress_unavailable'
    return dict(selected_seed=ref['seed'] if ref else 0 if fallback else None,
                selection_rule=rule,selection_posthoc_display_supplement=fallback,
                cpu_successful_seed_count=len({r['seed'] for r in cpu}),
                cpu_finite_stress_seed_count=len({r['seed'] for r in eligible}),
                cpu_reference_available=ref is not None,cpu_reference_seed=ref['seed'] if ref else None,
                cpu_reference_stress=ref['stress_value'] if ref else None)


def layout_partial_supplement(base, rows, kind, graphs, methods, figures, incidents):
    """Show every available success for No.1 graphs without a CPU reference."""
    records = [r for r in rows if r['study']=='basic' and r['graph'] in graphs and r.get('method') in methods]
    available = sorted((r for r in records if r.get('status')=='success' and r.get('final_positions_path')),
                       key=lambda r:(graphs.index(r['graph']),r['seed'],methods.index(r['method'])))
    choices = [dict(graph=r['graph'],method=r['method'],seed=r['seed'],status=r.get('status'),
                    run_id=r.get('run_id'),study='basic',
                    failure_reason=r.get('failure_reason'),executed=r.get('executed'),path=r.get('final_positions_path'),
                    cpu_reference_available=False,selection_rule='all_available_successes_without_cpu_reference',
                    selection_posthoc_display_supplement=True,descriptive_only=True,
                    layout_status='pending' if r in available else 'unavailable') for r in records]
    if available:
        columns = min(3,len(available))
        fig, axes = plt.subplots(math.ceil(len(available)/columns),columns,figsize=(4*columns,3.8*math.ceil(len(available)/columns)),squeeze=False)
        bounds = defaultdict(list); graph_axes = defaultdict(list)
        for ax,row in zip(axes.flat,available):
            graph=row['graph']; graph_axes[graph].append(ax)
            ax.set_title(f"{graph} / {LABEL[row['method']]} / seed {row['seed']}")
            choice=next(c for c in choices if c['graph']==graph and c['method']==row['method'] and c['seed']==row['seed'])
            try:
                xy,edges=read_layout(path_of(row['final_positions_path'],base)); total=len(edges)
                if total>250000: edges=edges[np.linspace(0,total-1,250000,dtype=int)]
                ax.add_collection(LineCollection(xy[edges],colors='#222222',linewidths=.15,alpha=.35,rasterized=True))
                bounds[graph].append((xy.min(axis=0),xy.max(axis=0)))
                ax.autoscale(); ax.set_aspect('equal'); ax.set_axis_off()
                fraction=len(edges)/total if total else 1.0
                seconds=row['iteration_time_ms']/1000 if finite(row.get('iteration_time_ms')) else None
                stress_note='exact' if row.get('stress_kind')=='exact' else f"{row.get('stress_samples')} sources"
                ax.text(0,-.04,f"stress ({stress_note}) {row['stress_value']:,.1f}\nSGD {fmt(seconds)} s; edges shown {fraction:.1%}",transform=ax.transAxes,fontsize=8)
                choice.update(layout_status='shown',stress=row['stress_value'],stress_kind=row.get('stress_kind'),stress_samples=row.get('stress_samples'),stress_seed=row.get('stress_seed'),iteration_seconds=seconds,edges_shown_fraction=fraction)
            except (OSError, EOFError, ValueError, IndexError) as error:
                empty(ax,f'Layout unavailable\n{type(error).__name__}')
                choice.update(layout_status='read_error',missing_reason=str(error))
                incidents.append(dict(graph=graph,method=row['method'],seed=row['seed'],stage='layout_partial',error=str(error)))
        for ax in list(axes.flat)[len(available):]: ax.set_axis_off()
        for graph,extent in bounds.items():
            lo=np.min([b[0] for b in extent],axis=0); hi=np.max([b[1] for b in extent],axis=0)
            center=(hi+lo)/2; half=max(float(np.max(hi-lo))*.54,1e-9)
            for ax in graph_axes[graph]: ax.set_xlim(center[0]-half,center[0]+half); ax.set_ylim(center[1]-half,center[1]+half)
        fig.suptitle('No CPU reference: all available layouts (descriptive supplement)',fontsize=11)
        fig.text(.5,.015,'All successful saved layouts are included; no representative-seed selection or CPU comparison.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.075,1,.91)); save(fig,'01_layout_'+kind+'_partial',base,figures)
    dump_csv(base/'tables'/f'layout-{kind}-partial-availability.csv',choices)


def layouts(base, rows, figures, incidents):
    choices = []
    for kind, graphs, methods in [('full', FULL_REPS, METHODS[:3]),('sparse',SPARSE_REPS,METHODS[3:])]:
        fig, axes = plt.subplots(len(graphs),3,figsize=(12,3.4*len(graphs)),squeeze=False)
        fallbacks = []
        availability_notes = []
        for i, graph in enumerate(graphs):
            selection = layout_reference(rows,graph,methods[0])
            selected_seed = selection['selected_seed']
            available_seeds = {method:sorted({r['seed'] for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')==method and r.get('status')=='success' and r.get('final_positions_path')}) for method in methods}
            selection['graph_successful_layout_seeds_by_method'] = available_seeds
            if selection['selection_posthoc_display_supplement']:
                fallbacks.append(graph)
                availability_notes.extend(f"{graph}: {LABEL[method]} seeds "+','.join(map(str,seeds)) for method,seeds in available_seeds.items() if seeds)
            bounds=[]
            for j, method in enumerate(methods):
                ax = axes[i,j]
                ax.set_title(f'{graph} / {LABEL[method]}')
                candidates = [r for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')==method and selected_seed is not None and r.get('seed')==selected_seed]
                row = candidates[0] if candidates else None
                choice = dict(graph=graph,family=kind,method=method,seed=selected_seed,**selection,
                              run_id=row.get('run_id') if row else None,study='basic',
                              status=row.get('status') if row else 'unrecorded',
                              executed=row.get('executed') if row else None,
                              failure_reason=row.get('failure_reason') if row else None,
                              error_stage=row.get('error_stage') if row else None,
                              error_message=(row.get('error_message') or row.get('error')) if row else None,
                              warmup_run_id=row.get('warmup_run_id') if row else None)
                if not row or row.get('status')!='success' or not row.get('final_positions_path'):
                    if selected_seed is None: reason = 'CPU stress unavailable for seed selection'
                    elif not row: reason = 'Selected-seed record unavailable'
                    elif row.get('status')!='success':
                        reason = row.get('failure_reason') or row.get('error_stage') or row.get('error_message') or row.get('status','Unavailable')
                    else: reason = 'Final positions unavailable'
                    message = f"seed {selected_seed if selected_seed is not None else 'unselected'}\n{choice['status']}"
                    if reason != choice['status']: message += f'\n{reason}'
                    if row and row.get('executed') is False: message += '\nnot executed'
                    empty(ax,message)
                    choice.update(layout_status='unavailable',missing_reason=reason)
                    choices.append(choice); continue
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
                    stress_note='exact' if row.get('stress_kind')=='exact' else f"{row.get('stress_samples')} sources"
                    ax.text(0,-.04,f"seed {row['seed']}; stress ({stress_note}) {row['stress_value']:,.1f}\nSGD {fmt(seconds)} s; edges shown {fraction:.1%}",transform=ax.transAxes,fontsize=8)
                    choice.update(layout_status='shown',missing_reason=None,stress=row['stress_value'],stress_kind=row.get('stress_kind'),stress_samples=row.get('stress_samples'),stress_seed=row.get('stress_seed'),edges_sampled=sampled,edges_shown_fraction=fraction,iteration_seconds=seconds,path=row['final_positions_path'])
                    choices.append(choice)
                except (OSError, EOFError, ValueError, IndexError) as e:
                    empty(ax,f'seed {selected_seed}\nLayout unavailable\n{type(e).__name__}')
                    choice.update(layout_status='read_error',missing_reason=str(e))
                    choices.append(choice); incidents.append(dict(graph=graph,method=method,seed=selected_seed,stage='layout',error=str(e)))
            if bounds:
                lo=np.min([b[0] for b in bounds],axis=0); hi=np.max([b[1] for b in bounds],axis=0)
                center=(hi+lo)/2; half=max(float(np.max(hi-lo))*.54,1e-9)
                for ax in axes[i]:
                    ax.set_xlim(center[0]-half,center[0]+half); ax.set_ylim(center[1]-half,center[1]+half)
        if fallbacks:
            fig.text(.5,.015,'Post-hoc display supplement: '+', '.join(fallbacks)+' uses preplanned seed 0 because no CPU seed succeeded.\n'
                     'No successful-seed search; missing panels keep seed 0. No CPU ratio is inferred.\n'
                     +('Available separately: '+'; '.join(availability_notes)+'.' if availability_notes else 'No successful saved layouts are available for these graphs.'),ha='center',fontsize=8)
        fig.tight_layout(h_pad=2.8,rect=(0,.075 if fallbacks else 0,1,1))
        save(fig,'01_layout_'+kind,base,figures)
        if fallbacks: layout_partial_supplement(base,rows,kind,fallbacks,methods,figures,incidents)
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
        except (OSError,EOFError,ValueError,IndexError) as error:
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
        if info.get('family')!=('small' if r['family']=='full' else 'large'): continue
        if group and finite(info.get('nodes'),positive=True):
            size_rows.append(dict(series=group,graph=r['graph'],family=r['family'],method=r['method'],seed=r['seed'],nodes=info['nodes'],speedup=r['speedup']))
    panels=sorted({(r['series'],r['family']) for r in size_rows if len({x['graph'] for x in size_rows if x['series']==r['series'] and x['family']==r['family']})>=2})
    if panels:
        from matplotlib.ticker import NullFormatter, NullLocator
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
            node_values=sorted({r['nodes'] for r in size_rows if r['series']==group and r['family']==kind})
            ticks=node_values if len(node_values)<=3 else [node_values[0],node_values[len(node_values)//2],node_values[-1]]
            labels=[f'{value/1_000_000:.3g}M' if value>=1_000_000 else f'{value/1_000:.3g}k' if value>=1_000 else f'{value:g}' for value in ticks]
            ax.set_xticks(ticks,labels)
            ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='x',labelsize=8)
        fig.tight_layout(); save(fig,'04_size_series',base,figures)
    dump_csv(base/'tables/size-series.csv',size_rows)
    schedule=[dict(graph=r['graph'],dataset_size_group=catalog.get(r['graph'],{}).get('family'),method=r['method'],seed=r['seed'],pivots=r.get('pivots'),nodes=r.get('nodes'),constraints=r.get('constraints'),
                   constraints_per_vertex=r['constraints']/r['nodes'] if finite(r.get('nodes'),positive=True) and finite(r.get('constraints')) else None,
                   rr_rounds=r.get('rounds'),dispatches=r.get('dispatches'),
                   dispatches_per_iteration=r['dispatches']/r['iterations'] if finite(r.get('iterations'),positive=True) and finite(r.get('dispatches')) else None)
              for r in rows if r['study']=='basic' and r.get('status')=='success' and family(r)=='sparse']
    dump_csv(base/'tables/sparse-schedule.csv',schedule)


def sparse_write_load_rows(base):
    directory=base/'analysis/sparse-write-load'
    source=directory/'summary.csv'
    if not source.is_file(): return []
    try:
        outcomes=json.loads((directory/'outcomes.json').read_text())
        selection=json.loads((directory/'selection.json').read_text())
        if not isinstance(outcomes,dict) or not isinstance(selection,dict): return []
        if outcomes.get('analysis_status')!='complete': return []
        if outcomes.get('summary_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest(): return []
        tool=selection['diagnostic_provenance']
        requests={(r['graph'],r['seed'],r['request']['pivots']):r['request'] for r in selection['selected']}
        allowed=set()
        for outcome in outcomes['results']:
            key=(outcome.get('graph'),outcome.get('seed'),outcome.get('pivots'))
            if outcome.get('status')!='success' or key not in requests: continue
            try:
                sidecar=json.loads(Path(outcome['provenance_path']).read_text())
                if sidecar['provenance']!=dict(tool=tool,request=requests[key]): continue
                digest=hashlib.sha256(Path(outcome['result_path']).read_bytes()).hexdigest()
                if digest!=sidecar['result_sha256'] or digest!=outcome['result_sha256']: continue
            except (OSError,ValueError,KeyError,TypeError): continue
            allowed.add(key)
        with source.open(newline='',encoding='utf-8') as stream:
            return [r for r in csv.DictReader(stream) if r.get('seed')=='0' and r.get('pivots')=='200'
                    and (r.get('graph'),0,200) in allowed
                    and r.get('metric') in ['positive_weight_endpoint_updates','atomic_lock_target_touches']]
    except (OSError,ValueError,KeyError,TypeError): return []


def sparse_write_load_figure(base,figures):
    """Render existing independent diagnostics only; never run preprocessing."""
    rows=sparse_write_load_rows(base)
    if not rows: return
    graphs=list(dict.fromkeys(r['graph'] for r in rows))
    metrics=[('positive_weight_endpoint_updates','Positive-weight endpoint updates','#0072B2'),
             ('atomic_lock_target_touches','Atomic lock targets (including zero weight)','#D55E00')]
    indexed={(r['graph'],r['metric'],r['vertex_group']):r for r in rows}
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    specifications=[('coefficient_of_variation','all','All-vertex planned-count CV',1),
                    ('top_1pct_share','all','Top 1% vertices: share of planned count (%)',100),
                    ('share_of_all_vertex_total','pivot','Pivots: share of all planned count (%)',100)]
    for ax,(field,group,label,multiplier) in zip(axes,specifications):
        for mi,(metric,title,color) in enumerate(metrics):
            xx=[]; yy=[]
            for gi,graph in enumerate(graphs):
                row=indexed.get((graph,metric,group),{})
                try: value=float(row[field])
                except (KeyError,TypeError,ValueError): continue
                if not finite(value): continue
                xx.append(gi+(mi-.5)*.34); yy.append(value*multiplier)
            ax.bar(xx,yy,width=.3,color=color,label=title)
        ax.set_xticks(range(len(graphs)),graphs,rotation=35,ha='right',fontsize=8)
        ax.set_ylabel(label); ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    # Finish axes layout before placing the two header rows at fixed figure
    # coordinates; tight_layout must not reposition a suptitle over the legend.
    fig.tight_layout(rect=(0,0,1,.86))
    fig.text(.5,.985,'Static planned endpoint counts per Sparse iteration; seed 0 / P=200; no measured GPU stores or lock waits',
             ha='center',va='top',fontsize=10)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.92),
               bbox_transform=fig.transFigure,ncol=2,fontsize=8,frameon=False,borderaxespad=0)
    save(fig,'04_sparse_write_load',base,figures)


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
        e['elapsed_ms']=e['cumulative_algorithm_seconds']*1000 if e.get('cumulative_algorithm_seconds') is not None else e.get('cumulative_iteration_ms',e.get('cumulative_iteration_time_ms',e.get('host_elapsed_ms',e.get('elapsed_ms',e.get('iteration_time_ms')))))
        value=e.get('snapshot',e.get('positions_path',e.get('path',f"positions-{e['epoch']:04}.f64le")))
        if not value:
            e['snapshot']=None; continue
        p=Path(value)
        resolved=p if p.is_absolute() else directory/p
        if not resolved.exists() and Path(str(resolved)+'.gz').exists():
            resolved=Path(str(resolved)+'.gz')
        e['snapshot']=str(resolved)
    if not entries and directory.exists():
        candidates={}
        for file in sorted(directory.glob('positions-*.f64le*')):
            if not (file.name.endswith('.f64le') or file.name.endswith('.f64le.gz')): continue
            epoch=int(file.name.removeprefix('positions-').split('.')[0])
            if epoch not in candidates or file.suffix!='.gz': candidates[epoch]=file
        for epoch,file in sorted(candidates.items()):
            entries.append(dict(epoch=epoch,elapsed_ms=None,snapshot=str(file)))
    return entries


def read_snapshot(path):
    """Return interleaved little-endian f64 coordinates from plain or gzip bytes."""
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as source:
            values=np.frombuffer(source.read(),dtype='<f8')
    else:
        values=np.fromfile(path,dtype='<f8')
    return values.reshape(-1,2)


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
                    # Key the resolved storage file. Archiving a previously plain
                    # snapshot invalidates its old cache entry safely; stable gzip
                    # paths then reuse their evaluated values on later summaries.
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
                try:
                    xy=read_snapshot(e['snapshot'])
                except (OSError,EOFError,ValueError) as error:
                    incidents.append(dict(graph=graph,stage='trace',error=f'Unreadable snapshot {e["snapshot"]}: {error}')); continue
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
    planpath=base/'config/plan.json'
    planned=json.loads(planpath.read_text()) if planpath.exists() else rows
    if isinstance(planned,dict): planned=planned.get('jobs',planned.get('tasks',[]))
    planned=[r for r in planned if r.get('study')=='convergence']
    panels=sorted({(r['graph'],family(r)) for r in planned} | {(r['graph'],r['family']) for r in output})
    actual={(r['graph'],r['method'],r['seed']):r for r in rows if r['study']=='convergence'}
    fig,axes=plt.subplots(max(1,math.ceil(len(panels)/3)),3,figsize=(15,max(4,math.ceil(len(panels)/3)*3.5)),squeeze=False)
    arrivals=[]; panel_coverage=[]
    for ax in axes.flat: ax.set_visible(False)
    for ax,(graph,kind) in zip(axes.flat,panels):
        ax.set_visible(True); methods=METHODS[:3] if kind=='full' else METHODS[3:]
        subset=[r for r in output if r['graph']==graph and r['family']==kind]
        labels={}; usable={}
        for method in methods:
            points=[r for r in subset if r['method']==method and finite(r.get('elapsed_ms')) and finite(r.get('stress'))]
            usable[method]=points
            expected={r['seed'] for r in planned if r['graph']==graph and r['method']==method}
            if not expected: expected={r['seed'] for r in rows if r['study']=='convergence' and r['graph']==graph and r['method']==method}
            observed={r['seed'] for r in points}
            statuses=Counter(actual.get((graph,method,seed),{}).get('status','not_run') for seed in expected)
            missing=len(expected-observed)
            status_note=', '.join(f'{status}={count}' for status,count in sorted(statuses.items())) or 'no planned records'
            labels[method]=f'{LABEL[method]}: observed {len(observed)}/{len(expected)}, missing {missing}\n{status_note}'
            panel_coverage.append(dict(graph=graph,family=kind,method=method,planned_seeds=len(expected),
                                       observed_seeds=len(observed),missing_observation_seeds=missing,
                                       valid_checkpoint_observations=len(points),planned_seed_status_counts=dict(statuses)))
        if not any(usable.values()):
            empty(ax,'No valid checkpoint observations\n'+'\n'.join(labels[method] for method in methods))
            ax.set_title(f'{graph} / {kind}')
            continue
        for method in methods:
            points=usable[method]
            for epoch in sorted({r['epoch'] for r in points}):
                chosen=[r for r in points if r['epoch']==epoch]
                ax.scatter(np.median([r['elapsed_ms'] for r in chosen])/1000,np.median([r['stress'] for r in chosen]),color=COLOR[method],s=10)
            medians=[(np.median([r['elapsed_ms'] for r in points if r['epoch']==e])/1000,np.median([r['stress'] for r in points if r['epoch']==e])) for e in sorted({r['epoch'] for r in points})]
            if medians: ax.plot(*zip(*medians),color=COLOR[method],label=labels[method])
            else: ax.plot([],[],color=COLOR[method],label=labels[method])
        ax.set(title=f'{graph} / {kind}',xlabel='Cumulative SGD time (s)',ylabel='Full stress',yscale='log'); ax.legend(fontsize=6)
    if not panels: empty(axes[0,0],'No convergence snapshots evaluated'); axes[0,0].set_visible(True)
    fig.suptitle('All planned convergence panels retained; observed = seed with a usable checkpoint; missing = none.\nStatuses count planned conditions, not physical launch failures.',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.97)); save(fig,'05_convergence',base,figures)
    dump_csv(base/'tables/convergence-panel-coverage.csv',panel_coverage)
    convergence_stress_audit(base,rows,output)
    arrivals=build_arrivals(base,rows,output)
    dump_csv(base/'tables/target-arrival.csv',arrivals)
    return output,arrivals


def convergence_stress_audit(base,rows,checkpoints):
    """Compare saved epoch-15 stress to native evaluation without excluding runs."""
    finals={(r['graph'],r['method'],r['seed']):r for r in checkpoints if r['epoch']==15}
    audit=[]
    for row in rows:
        if row['study']!='convergence' or row.get('status')!='success': continue
        snapshot=finals.get((row['graph'],row['method'],row['seed']))
        native=row.get('stress_value'); offline=snapshot.get('stress') if snapshot else None
        native_kind=row.get('stress_kind'); offline_kind=snapshot.get('stress_kind') if snapshot else None
        native_samples=row.get('stress_samples')
        offline_samples=snapshot.get('stress_samples') if snapshot and offline_kind=='sampled' else None
        native_seed=row.get('stress_seed'); offline_seed=snapshot.get('stress_seed') if snapshot and offline_kind=='sampled' else None
        compatible=bool(snapshot) and (native_kind,native_samples,native_seed)==(offline_kind,offline_samples,offline_seed)
        relative=None; signed=None
        if finite(native) and finite(offline):
            if native>0:
                signed=(offline-native)/native*100; relative=abs(signed)
            elif offline==0: relative=0.; signed=0.
        if not snapshot: flag='missing_checkpoint'
        elif not compatible: flag='definition_mismatch'
        elif not finite(native) or not finite(offline): flag='invalid_value'
        elif relative is None or relative>=.1: flag='investigate'
        else: flag='within_0_1_percent'
        audit.append(dict(graph=row['graph'],method=row['method'],seed=row['seed'],run_status=row['status'],
                          native_stress_kind=native_kind,offline_stress_kind=offline_kind,
                          native_stress_samples=native_samples,offline_stress_samples=offline_samples,
                          offline_evaluation_sources=snapshot.get('stress_samples') if snapshot else None,
                          native_stress_seed=native_seed,offline_stress_seed=offline_seed,
                          native_stress_value=native,offline_epoch15_stress_value=offline,
                          absolute_relative_difference_percent=relative,signed_relative_difference_percent=signed,
                          evaluator_definition_compatible=compatible,flag=flag,
                          note='Snapshot precedes centering; native evaluation follows centering. Mathematical translation invariance does not imply bitwise equality after f32/f64 rounding. Audit flag does not exclude the successful run.'))
    dump_csv(base/'tables/convergence-stress-audit.csv',audit)


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
        cpu_status=actual.get((graph,cpu_method,seed),{}).get('status')
        cpu_final=next((r for r in traces[(graph,cpu_method,seed)] if r['epoch']==15),None) if cpu_status=='success' else None
        own_status=actual.get(key,{}).get('status','not_run')
        for factor in [1.05,1.01]:
            target=cpu_final['stress']*factor if cpu_final else None
            hit=next((r for r in seq if target is not None and r['stress']<=target),None)
            previous=seq[max(0,seq.index(hit)-1)] if hit else None
            rebound=any(r['epoch']>hit['epoch'] and r['stress']>target for r in seq) if hit else None
            final=seq[-1] if seq else None
            epoch15=next((r for r in seq if r['epoch']==15),None)
            final_below=final['stress']<=target if final and target is not None else None
            epoch15_below=epoch15['stress']<=target if epoch15 and target is not None else None
            evaluable=own_status=='success' and complete and target is not None
            if not seq or own_status!='success': state='measurement_unavailable'; reached=None
            elif target is None: state='reference_unavailable'; reached=None
            elif not complete: state='partial_observation'; reached=True if hit else None
            else: state='reached' if hit else 'not_observed'; reached=hit is not None
            arrivals.append(dict(graph=graph,family=kind,method=method,seed=seed,target_factor=factor,target_stress=target,
                                 first_observed_epoch=hit['epoch'] if hit else None,first_observed_ms=hit['elapsed_ms'] if hit else None,
                                 interval_lower_epoch=previous['epoch'] if previous else None,interval_lower_ms=previous['elapsed_ms'] if previous else None,
                                 interval_upper_epoch=hit['epoch'] if hit else None,interval_upper_ms=hit['elapsed_ms'] if hit else None,
                                 observed_rebound_after_hit=rebound,final_checkpoint_epoch=final['epoch'] if final else None,
                                 final_checkpoint_at_or_below_target=final_below,epoch15_at_or_below_target=epoch15_below,
                                 upper_time_has_target_satisfying_snapshot=True if hit else None,
                                 interval_semantics='Previous observation to first observed hit; upper time has a target-satisfying snapshot. This is not a guaranteed first-passage bracket: true first arrival may precede the lower observation time.',
                                 reached=reached,evaluable=evaluable,status=state,run_status=own_status,
                                 observed_epochs=sorted(observed),missing_epochs=sorted(required-observed)))
    coverage=[]
    for key in sorted({(r['graph'],r['family'],r['method']) for r in arrivals}):
        graph,kind,method=key
        a=[r for r in arrivals if (r['graph'],r['family'],r['method'])==key and r['target_factor']==1.05]
        b=[r for r in arrivals if (r['graph'],r['family'],r['method'])==key and r['target_factor']==1.01]
        item=dict(graph=graph,family=kind,method=method,planned_seeds=len(a),evaluable_seeds=sum(r['evaluable'] for r in a),
                  reached_1_05=sum(r['evaluable'] and r['reached'] is True for r in a),reached_1_01=sum(r['evaluable'] and r['reached'] is True for r in b),
                  missing_seeds=sum(not r['evaluable'] for r in a),measurement_unavailable=sum(r['status']=='measurement_unavailable' for r in a),
                  reference_unavailable=sum(r['status']=='reference_unavailable' for r in a),partial_observation=sum(r['status']=='partial_observation' for r in a))
        for factor,group in [('1_05',a),('1_01',b)]:
            item[f'observed_rebound_after_hit_{factor}']=sum(r['evaluable'] and r['observed_rebound_after_hit'] is True for r in group)
            item[f'epoch15_at_or_below_target_{factor}']=sum(r['evaluable'] and r['epoch15_at_or_below_target'] is True for r in group)
            observed=[r for r in group if r['evaluable'] and r['reached'] is True]
            for bound in ['lower','upper']:
                values=[r[f'interval_{bound}_ms'] for r in observed if finite(r.get(f'interval_{bound}_ms'))]
                item[f'interval_{bound}_ms_median_{factor}']=float(np.median(values)) if values else None
        coverage.append(item)
    dump_csv(base/'tables/target-arrival-coverage.csv',coverage)
    return arrivals


def iteration_timings(base,rows,figures):
    """Read clean per-iteration times from ordinary benchmark traces."""
    reps=FULL_REPS+SPARSE_REPS
    records=[]; validation=[]
    for row in rows:
        if row['study']!='basic' or row.get('status')!='success': continue
        trace=load_trace(row,base)
        actual_epochs=[e['epoch'] for e in trace]
        durations=[e.get('algorithm_seconds') for e in trace]
        cumulatives=[e.get('cumulative_algorithm_seconds') for e in trace]
        numeric=all(finite(v) for v in durations+cumulatives) and bool(trace)
        complete=actual_epochs==list(range(row.get('iterations',15)+1))
        consistent=numeric and all(math.isclose(sum(durations[:index+1]),value,rel_tol=1e-8,abs_tol=1e-9) for index,value in enumerate(cumulatives))
        agrees=numeric and finite(row.get('iteration_time_ms')) and math.isclose(cumulatives[-1]*1000,row['iteration_time_ms'],rel_tol=1e-8,abs_tol=1e-6)
        validation.append(dict(graph=row['graph'],method=row['method'],seed=row['seed'],complete=complete,cumulative_consistent=consistent,record_time_consistent=agrees,observed_epochs=actual_epochs,status='pass' if complete and consistent and agrees else 'check_required'))
        if not (complete and consistent and agrees): continue
        for entry in trace:
            if entry['epoch']==0 or entry.get('algorithm_seconds') is None: continue
            records.append(dict(graph=row['graph'],method=row['method'],family=family(row),seed=row['seed'],
                                iteration=entry['epoch'],algorithm_seconds=entry['algorithm_seconds'],
                                cumulative_algorithm_seconds=entry.get('cumulative_algorithm_seconds')))
    dump_csv(base/'tables/per-iteration-time.csv',records)
    dump_csv(base/'tables/iteration-time-validation.csv',validation)
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
            ax.plot(xx,yy,color=COLOR[method],label=f'{LABEL[method]} (n={len({r["seed"] for r in subset})}/5)')
            ax.fill_between(xx,lo,hi,color=COLOR[method],alpha=.15)
        ax.set(title=f'{graph} / {kind}',xlabel='Iteration',ylabel='SGD time (s)',yscale='log'); ax.legend()
    fig.suptitle('Ordinary basic benchmarks, seeds 0–4: successful valid traces; median and min–max\nTrace coordinate copy / IO excluded; separate from the 10-seed convergence diagnostics',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94)); save(fig,'05_iteration_times',base,figures)
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


def full_cpu_five_seed_reference(rows,graph):
    selected={r['seed']:r for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')=='sgd'
              and r.get('status')=='success' and r.get('seed') in range(5)
              and finite(r.get('stress_value')) and finite(r.get('iteration_time_ms'),positive=True)}
    if set(selected)!=set(range(5)): return None
    return dict(graph=graph,method='sgd',tile_size=None,status='cpu_reference',successful_seeds=5,expected_seeds=5,
                iteration_seconds_median=float(np.median([r['iteration_time_ms'] for r in selected.values()]))/1000,
                stress_median=float(np.median([r['stress_value'] for r in selected.values()])))


def sensitivity(base,rows,pairs,figures):
    """Keep every planned parameter on the axis; missing values break lines."""
    planpath=base/'config/plan.json'
    plan=json.loads(planpath.read_text()) if planpath.exists() else rows
    if isinstance(plan,dict): plan=plan.get('jobs',plan.get('tasks',[]))
    for study,param,name,grid,methods in [
            ('tiling','tile_size','06_tiles',[1,2,4,8,16,32,64,128,256,512,1024],['rr_sgd']),
            ('pivots','pivots','07_pivots',[50,100,200,400],METHODS[3:])]:
        selected=[r for r in rows if r['study']==study]
        graphs=sorted({r['graph'] for r in plan if r.get('study')==study} | {r['graph'] for r in selected})
        fields=['algorithm_time_cold_ms','stress_value','peak_rss_bytes'] if study=='pivots' else ['iteration_time_ms','stress_value']
        fig,axes=plt.subplots(max(1,len(graphs)),len(fields),figsize=(5*len(fields),max(3,len(graphs)*2.8)),squeeze=False)
        coverage=[]
        for gi,graph in enumerate(graphs):
            for col,field in enumerate(fields):
                ax=axes[gi,col]
                for mi,method in enumerate(methods):
                    raw=[r for r in selected if r['graph']==graph and r['method']==method and r.get('seed') in range(5)]
                    yy=[]; lower=[]; upper=[]
                    for value in grid:
                        condition={r['seed']:r for r in raw if r.get(param,r.get('rr_tile_size'))==value}
                        successes=[r for r in condition.values() if r.get('status')=='success']
                        valid=[r[field] for r in successes if finite(r.get(field),positive=True)]
                        median,low,high=statistics(valid)
                        divisor=1000 if field in ['iteration_time_ms','algorithm_time_cold_ms'] else 1024**3 if field=='peak_rss_bytes' else 1
                        yy.append(median/divisor if median is not None else np.nan)
                        lower.append((median-low)/divisor if median is not None else np.nan)
                        upper.append((high-median)/divisor if median is not None else np.nan)
                        count=len(valid)
                        if not count:
                            ax.plot([value],[.035+mi*.055],marker='x',color=COLOR[method],markersize=5,ls='',transform=ax.get_xaxis_transform())
                            ax.text(value,.065+mi*.055,'0/5',color=COLOR[method],ha='center',fontsize=6,transform=ax.get_xaxis_transform())
                        elif count<5:
                            ax.text(value,.065+mi*.055,f'{count}/5',color=COLOR[method],ha='center',fontsize=6,transform=ax.get_xaxis_transform())
                        coverage.append(dict(study=study,graph=graph,method=method,parameter=param,parameter_value=value,metric=field,
                                             expected_seeds=5,recorded_seeds=len(condition),successful_run_seeds=len(successes),valid_metric_seeds=count,
                                             missing_metric_seeds=5-count,not_recorded_seeds=5-len(condition),
                                             condition_status_counts=dict(Counter(r.get('status','unknown') for r in condition.values())),
                                             metric_median_native_units=median,status='complete' if count==5 else 'partial' if count else 'unavailable'))
                    # NaN entries preserve gaps and prevent drawing a trend across
                    # an entirely unobserved T or P. Never replace them with zero.
                    if np.isfinite(yy).any():
                        ax.errorbar(grid,yy,yerr=[lower,upper],fmt='o-',color=COLOR[method],markersize=3,capsize=2)
                    ax.plot([],[],marker='o',ls='-',color=COLOR[method],label=LABEL[method])
                if study=='tiling' and field=='stress_value':
                    cpu=full_cpu_five_seed_reference(rows,graph)
                    if cpu: ax.axhline(cpu['stress_median'],color='#555555',ls='--',lw=1,label='CPU reference (5 seeds)')
                if not np.isfinite(ax.dataLim.y0) or ax.dataLim.y1<=0: ax.set_ylim(.1,10)
                ax.set_xscale('log',base=2); ax.set_yscale('log')
                ax.set_xticks(grid,[str(value) for value in grid],fontsize=7)
                ax.set_xlim(grid[0]/1.2,grid[-1]*1.2)
                ax.set(title=graph,xlabel=param,ylabel={'iteration_time_ms':'SGD time (s)','algorithm_time_cold_ms':'Cold algorithm total (s)','stress_value':'Full stress','peak_rss_bytes':'Peak process RSS (GiB)'}[field])
                ax.legend(fontsize=7)
        if not graphs:
            for ax in axes.flat: empty(ax)
        fig.suptitle('All planned settings retained; × 0/5 = unavailable, n/5 = partial; missing settings break lines',fontsize=10,y=1.005)
        fig.tight_layout(); save(fig,name,base,figures)
        dump_csv(base/'tables'/f'{study}-parameter-coverage.csv',coverage)


def tile_pareto(base,rows,figures):
    """Plot time/quality for complete five-seed tile settings only."""
    representatives=FULL_REPS+['bcsstk16','wiki-Vote']
    fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
    table=[]
    for ax,graph in zip(axes.flat,representatives):
        all_rows=[r for r in rows if r['study']=='tiling' and r['graph']==graph and r.get('method')=='rr_sgd']
        points=[]
        for tile in [1,2,4,8,16,32,64,128,256,512,1024]:
            selected=[r for r in all_rows if r.get('tile_size',r.get('rr_tile_size'))==tile and r.get('status')=='success'
                      and finite(r.get('iteration_time_ms'),positive=True) and finite(r.get('stress_value'))]
            selected={r['seed']:r for r in selected if r.get('seed') in range(5)}
            complete=set(selected)==set(range(5))
            item=dict(graph=graph,method='rr_sgd',tile_size=tile,successful_seeds=len(selected),expected_seeds=5,
                      status='complete' if complete else 'partial' if selected else 'unavailable',iteration_seconds_median=None,stress_median=None)
            if complete:
                item['iteration_seconds_median']=float(np.median([r['iteration_time_ms'] for r in selected.values()]))/1000
                item['stress_median']=float(np.median([r['stress_value'] for r in selected.values()]))
                points.append(item)
            table.append(item)
        for point in points:
            xx=point['iteration_seconds_median']; yy=point['stress_median']
            point['nondominated']=not any(other['iteration_seconds_median']<=xx and other['stress_median']<=yy
                                         and (other['iteration_seconds_median']<xx or other['stress_median']<yy) for other in points)
            ax.scatter([xx],[yy],color=COLOR['rr_sgd'],s=28)
            ax.annotate(f'T={point["tile_size"]}',(xx,yy),xytext=(4,3),textcoords='offset points',fontsize=7)
        front=sorted((r['iteration_seconds_median'],r['stress_median']) for r in points if r['nondominated'])
        if len(front)>1: ax.plot(*zip(*front),color='#009E73',ls='--',lw=.8,label='Observed nondominated settings')
        cpu=full_cpu_five_seed_reference(rows,graph)
        if cpu:
            ax.scatter([cpu['iteration_seconds_median']],[cpu['stress_median']],marker='X',s=65,color='#444444',label='CPU reference (not a tile setting)')
            ax.annotate('CPU reference',(cpu['iteration_seconds_median'],cpu['stress_median']),xytext=(5,-10),textcoords='offset points',fontsize=7)
            table.append(cpu)
        if not points and not cpu: empty(ax,'No complete five-seed tile / CPU setting')
        else: ax.set_xscale('log')
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
        ax.set(title=graph,xlabel='Median SGD time (s)',ylabel='Median full stress')
    fig.suptitle('RR tile time–quality tradeoff: seeds 0–4 complete; CPU is a separate reference, not a tile measurement',fontsize=12)
    fig.tight_layout(); save(fig,'06_tiles_pareto',base,figures)
    dump_csv(base/'tables/tile-pareto.csv',table)


def fmt(value,digits=3):
    return f'{value:,.{digits}f}' if finite(value) else '—'


def dataset_exclusion_kind(info):
    distribution=str(info.get('source_distribution_status',info.get('source_status','')))
    if distribution.startswith('alternative_distribution'):
        return 'identity_unverified'
    if info.get('status')=='ready' and not info.get('eligible',True):
        return 'size_excluded' if info.get('family')=='large' and (info.get('nodes') or math.inf)<100000 else 'other_ineligible'
    if info.get('status')!='ready':
        return 'not_acquired' if distribution=='unavailable' else 'other_unavailable'
    return 'eligible'


def complete_five_seed_condition(rows,condition,timing_field):
    """Require all planned seeds and valid paired time/quality for best claims."""
    peers={r['seed']:r for r in rows if r['study']==condition['study'] and r['graph']==condition['graph']
           and r.get('method')==condition['method'] and r.get('pivots')==condition['pivots']
           and r.get('tile_size',r.get('rr_tile_size',1024))==condition['tile_size'] and r.get('seed') in range(5)}
    return set(peers)==set(range(5)) and all(r.get('status')=='success' and finite(r.get(timing_field),positive=True)
           and finite(r.get('iteration_time_ms'),positive=True) and finite(r.get('stress_value')) for r in peers.values())


def report(base,datasets,rows,summary,pairs,figures,arrivals,incidents,excluded,esa_artifact_base_url=None,cohort=None):
    cohort=cohort or {}
    catalog={r['name']:r for r in datasets}
    exclusion_counts=Counter(dataset_exclusion_kind(r) for r in datasets)
    excluded_conditions=Counter(dataset_exclusion_kind(catalog[r['graph']]) for r in rows if r['graph'] in catalog and dataset_exclusion_kind(catalog[r['graph']])!='eligible' and r.get('status')!='success')
    availability=[dict(graph=r['name'],status=r.get('status'),eligible=r.get('eligible'),source_distribution_status=r.get('source_distribution_status'),exclusion_kind=dataset_exclusion_kind(r),reason=r.get('reason',r.get('error'))) for r in datasets]
    dump_csv(base/'tables/dataset-availability.csv',availability)
    status=Counter(r.get('status','unknown') for r in rows)
    counts=Counter(r.get('status','unknown') for r in datasets)
    actual=[r for r in rows if not r.get('reused_from') and finite(r.get('process_wall_seconds')) and r.get('executed',True)]
    actual_status=Counter(r.get('status','unknown') for r in actual)
    reused=[r for r in rows if r.get('reused_from')]
    blocked=[r for r in rows if not r.get('reused_from') and r.get('status')=='warmup_failed' and dataset_exclusion_kind(catalog.get(r['graph'],{}))=='eligible']
    reused_failures=[r for r in reused if r.get('status')!='success']
    warmups=[]
    warmup_bases=[base]+([Path(cohort['recovery_base'])] if cohort.get('status')=='adopted' else [])
    for origin_base in warmup_bases:
        for p in sorted((origin_base/'warmups').glob('*/record.json')):
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
    write_load=sparse_write_load_rows(base)
    write_graphs=list(dict.fromkeys(r['graph'] for r in write_load))
    write_note=(f'Sparse全グラフの補足は制約数/頂点数・RRラウンド・dispatch数（sparse-schedule.csv）。さらに代表 {len(write_graphs)} グラフは同一input/preprocess/initial SHAを照合したseed0/P200の独立診断で、頂点別の厳密な予定回数分布を補足する。予定回数はGPU実store・lock retry・待ち時間の実測ではない。' if write_load else 'Sparse全グラフの補足は制約数/頂点数・RRラウンド・dispatch数のみsparse-schedule.csvに保存する。代表5グラフの厳密な頂点別予定回数診断は未取得のため、予定回数の補助図・表は表示しない。')
    text=['# v3 実験結果','',f'データ {len(datasets)} グラフ（'+ '、'.join(f'{k}: {v}' for k,v in counts.items())+'）。',
          f'実験適格 {exclusion_counts["eligible"]} グラフ。未取得 {exclusion_counts["not_acquired"]}、代替配布との同一性未確認 {exclusion_counts["identity_unverified"]}、LCC規模による除外 {exclusion_counts["size_excluded"]} グラフ。未取得・同一性未確認・規模除外は実行前の区別として扱う。',
          f'計画条件の記録 {len(rows):,} 件：'+ '、'.join(f'{k}: {v:,}' for k,v in status.items())+'。'+(f' 計画 {planned:,} 件。' if planned is not None else ''),'',
          f'条件レコードとして確定した本測定の実起動 {len(actual):,} 件（'+ '、'.join(f'{k}: {v:,}' for k,v in actual_status.items())+f'）。既存結果の再利用 {len(reused):,} 条件、warmup失敗による未起動 {len(blocked):,} 条件。未確定の中断attemptはこの実起動数に含めず、最終監査で別枠に数える。',
          f'再利用のうち失敗結果の参照は {len(reused_failures):,} 条件。新たな実起動やwarmup失敗による新規未起動には重複計上しない。',
          f'別枠のwarmupは実起動 {len(warmup_actual):,} 件（'+ '、'.join(f'{k}: {v:,}' for k,v in warmup_status.items())+'）。warmupは性能・stressの本集計に混ぜない。','',
          '速度は前処理を除いたSGD反復の実時間。速度比は同じグラフ・seed・family・pivot数のCPU時間÷各手法時間。stress比は各手法÷同じCPU。点はseedの中央値、範囲は最小〜最大。失敗・未取得の条件は0に置き換えない。',
          'FullとSparseの絶対stressは同じ評価定義を用いる。頂点数8,000以下は全点対の厳密値、8,000超は固定256起点（stress seed=0）による推定値とする。小中規模20件の中にも推定値のグラフがある。SGDの制約がFullであることと、比較用stressを厳密評価することは別である。',
          '座標等のテキストはlossless gzipで保管する。圧縮時間はnative process_wall_seconds・SGD・cold総時間の計測外で、archive_secondsとして別記録する。圧縮の有無で座標値やstressは変化しない。',
          'SGD時間は各反復の処理区間の合計で、CPUのshuffle、GPUへの投入と完了待ちを含む。前後の初期化やcheckpoint座標取得・保存は含まない。CPUの反復前ペアclone等はこの区間の外。cold総時間は別に保存する。反復別の図はtraceの反復番号・累積和・本測定record時間の一致を検査し、不一致をiteration-time-validation.csvに残して除外する。',
          write_note,
          '以下は現在保存されている記録の集計。実行中の再集計では未完了条件が欠ける。','']
    if cohort.get('status')=='adopted':
        text += [f'SNAP原配布復旧による追加測定は、core原本 {cohort["core_raw_records"]:,} 件と独立recovery原本 {cohort["recovery_raw_records"]:,} 件（計 {cohort["raw_record_count"]:,} 件）を保持し、未起動unavailableだった90条件だけを補完した。有効な計画条件は {cohort["effective_conditions"]:,} 件。元90件はsupersededとして追跡し、実起動済み条件を上書きしていない。',
                 '対象はcom-Amazon、com-DBLP、com-Youtube、com-LiveJournal、soc-Pokec、loc-Gowalla。公式SNAP原配布と公開ノード数・unique edge数・変換/LCCの証拠を検証した、同名の原配布からの補完である。SuiteSparse MTXとのbyte/edgewise一致は未検証で、既存Network Repository代替データを検証済み扱いに変えたものではない。power197kの未取得とG_n_pin_poutの規模除外は保持する。',
                 'coreとrecoveryはそれぞれのexecution.jsonのnative commitに照合した。commitは異なるが、native/sharedの67ファイル集合・SHA、runner 1ファイルのSHA、6実行binaryのSHA、実行制限は一致する。共通の15反復・seed・P200の元basic90 ID/設定を照合しており、新条件として二重計上しない。',
                 '[有効データ一覧](../tables/effective-datasets.json) · [補完検証provenance](../tables/cohort.json) · [原記録と有効/置換状態](../tables/cohort-source-records.csv) · [置換前の未起動90条件](../tables/cohort-superseded.csv) · [追加測定の固定設定](../snap-origin-recovery/config/execution.json)。','']
    elif cohort.get('status') in ('incomplete','rejected'):
        text += [f'SNAP原配布追加測定の採用状態：{cohort["status"]}。recoveryの一部だけを混ぜず、core cohortだけを集計した。検証エラーまたは未完了理由は[cohort provenance](../tables/cohort.json)に記録する。','']
    captions={1:'代表グラフの描画',2:'小規模20グラフ：Full SGD',3:'大規模80グラフ：Sparse SGD',4:'グラフ構造と速度',5:'収束曲線と目標stress到達',6:'タイルサイズ',7:'pivot数'}
    for number in range(1,8):
        text += [f'## {number}. {captions[number]}','']
        names=[name for name in figures if name.startswith(f'{number:02}_')]
        for name in names: text += [f'![{captions[number]}](../figures/{name}.png)','']
        if number==1:
            fallback_graphs = [f'{kind}: {graph}' for kind,graphs,cpu_method in [('Full',FULL_REPS,METHODS[0]),('Sparse',SPARSE_REPS,METHODS[3])]
                               for graph in graphs if layout_reference(rows,graph,cpu_method)['selection_posthoc_display_supplement']]
            text += ['成功CPUがある場合はCPUの最終stress中央値に最も近いseedを選び、GPUも同じseedを表示する。良好なGPU実行だけを選抜していない。CPU成功seedが0件の場合だけ、事前に計画されたseed 0を固定で使い、同seedで成功し座標が存在する手法だけを表示する。成功したGPU seedを探して置き換えない。このfallbackはNo.1の描画選択だけを対象とする事後補足で、CPU比や主結果の対応ペアには使わない。',
                     '今回のCPU基準欠如による固定seed 0の対象：'+('、'.join(fallback_graphs) if fallback_graphs else 'なし')+'。欠測枠には選択seedと失敗理由を表示し、warmup失敗による未起動も区別する。seed 0自身が失敗していれば、別seedの成功座標があっても描画しない。',
                     '[描画seed・選択規則・欠測理由](../tables/layout-selection.csv)に全24枠を記録する。CPU成功があるのに有効なstress値がない場合はseed 0へ移らず、選択不能として記録する。',
                     '同じグラフの全手法は座標のunion範囲から同じx/y縮尺で描画する。25万辺を超える描画は辺のみ等間隔で間引き、表示割合とSGD秒を併記する。stressは間引く前の全グラフで評価する。','']
            for kind,graphs,methods in [('full',FULL_REPS,METHODS[:3]),('sparse',SPARSE_REPS,METHODS[3:])]:
                if '01_layout_'+kind+'_partial' not in figures: continue
                availability = [f'{graph} / {LABEL[method]}: seed '+','.join(map(str,sorted({r['seed'] for r in rows if r['study']=='basic' and r['graph']==graph and r.get('method')==method and r.get('status')=='success' and r.get('final_positions_path')})))
                                for graph in graphs if layout_reference(rows,graph,methods[0])['selection_posthoc_display_supplement']
                                for method in methods if any(r['study']=='basic' and r['graph']==graph and r.get('method')==method and r.get('status')=='success' and r.get('final_positions_path') for r in rows)]
                text += ['別図 `01_layout_'+kind+'_partial` は **CPU基準なし・利用可能な全layoutを示す記述的補足**。本図の固定seed 0を成功seedで差し替えるものではなく、対応比較や中央値代表とは呼ばない。CPU成功のない対象グラフについて、成功かつ保存先のあるlayoutをseedや品質で絞らず全件表示する。今回の利用可能分：'+'；'.join(availability)+'。図を読めない保存先は欠測枠と事故記録に残す。',
                         f'[補足図の全条件・利用可否](../tables/layout-{kind}-partial-availability.csv)。本図が全欠測でも、別seedの成功layoutが存在する場合がある。','']
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
            if number==3: text += ['', 'Sparse80 overviewにはRR GPUの代表描画4枠を添える。seedは実験1と同じCPU stress中央値に近いものを選び、接続線は対応するグラフのRR速度比中央値を指す。描画seed単独の速度比を指す線ではない。取得・成功していない描画は欠測枠として残す。小規模20グラフのSparse補助比較も別図で表示する。大規模80グラフの集計とは混ぜない。時間内訳図の積み上げは計測できた成分の平均で、cold総時間との差には区間外の準備処理が含まれる。']
            if number==2:
                text += ['', '[Full20のID・名前対応](../tables/02_full-graph-order.csv) · [対応seed数](../tables/02_full-counts.csv)']
            else:
                text += ['', '[Sparse80のID・名前対応](../tables/03_sparse-graph-order.csv) · [対応seed数](../tables/03_sparse-counts.csv) · [Sparse小20補助のID対応](../tables/03_sparse_small_aux-graph-order.csv)']
            text += ['', '上下2段は全グラフの共通x軸。背景は名前から推定したカテゴリで、分類内を頂点数順に並べる。graph-order.csvにID・名前を、counts.csvに対応seed数を保存。下端の×と数は欠測・CPU基準欠如による未対応seed数（5件中）で、0や1の測定値ではない。比は対応する成功seedのみで計算する。','']
        elif number==4:
            text += ['頂点数V・辺数E・次数の平均・最大値・変動係数の5指標を、Full/Sparseの上下2段で速度比と比較する。品質比も同じ2×5構成の補助図で示す。各グラフは対応seed比の中央値1点とし、Spearman CSVはmetric列で速度と品質を区別する。CV=0はsymlog軸で保持する。次数だけで実行時間は決まらず、頂点数・制約数・タイル数との交絡がある。相関係数はstructure-correlations.csvに保存し、因果関係として扱わない。点形状の構造カテゴリはグラフ名からの推定であり、確認済みの用途分類ではない。元のSuiteSparseグループはsource_groupとしてgraph-structure.csvに保存する。平均次数4/16とCV0.5/2で分けた図はグラフごとの速度比中央値を再集計し、グラフを等重みにする。同系列のサイズ曲線もグラフ名prefixからの推定系列で、規模以外の構造差を制御した実験ではない。','']
            if write_load:
                text += ['[代表グラフの予定書込回数CSV](../analysis/sparse-write-load/summary.csv)・[選択とSHA照合の証拠](../analysis/sparse-write-load/selection.json)。Atomicは重み0の端点もlockするため、正重みの端点更新予定数と両端lock対象予定数を分ける。いずれも1反復あたりの静的回数で、実際のGPU store、競合率、retry、待ち時間を表さない。図は全頂点CV・上位1%頂点の割合・pivot群の総回数割合。表はall群の総数・最大値を併記する。', '', '| Graph | 予定回数の定義 | 総数 / 反復 | 最大 / 頂点 | CV | 上位1%割合 | pivot群割合 |', '|---|---|---:|---:|---:|---:|---:|']
                indexed={(r['graph'],r['metric'],r['vertex_group']):r for r in write_load}
                for graph in write_graphs:
                    for metric,title in [('positive_weight_endpoint_updates','正重み端点更新'),('atomic_lock_target_touches','Atomic lock対象')]:
                        all_row=indexed.get((graph,metric,'all')); pivot_row=indexed.get((graph,metric,'pivot'),{})
                        if not all_row: continue
                        def diagnostic_number(row,key,percent=False):
                            try: value=float(row[key])
                            except (KeyError,TypeError,ValueError): return '—'
                            if not finite(value): return '—'
                            return f'{100*value:.2f}%' if percent else fmt(value,3 if key=='coefficient_of_variation' else 0)
                        text += [f'| {graph} | {title} | {diagnostic_number(all_row,"total_per_iteration")} | {diagnostic_number(all_row,"max")} | {diagnostic_number(all_row,"coefficient_of_variation")} | {diagnostic_number(all_row,"top_1pct_share",True)} | {diagnostic_number(pivot_row,"share_of_all_vertex_total",True)} |']
                text += ['']
        elif number==5:
            text += ['`05_convergence` は座標保存を伴う収束診断の計画10seedから、反復0・1・2・3・5・8・10・15のstressを後計算した図。GPUとCPUで同じ起点を使う。曲線はcheckpointごとの各手法の成功seed中央値であり、実際の単一実行の軌跡ではない。失敗により手法間の成功seed集合が異なる場合がある。計画した全24 graph/family枠を残し、全観測欠測の枠も表示する。図のobservedは有効なcheckpointが1点以上あるseed数、missingはないseed数。併記statusは計画条件の分類で、warmup_failedを実起動の事故件数と数えない。[図の観測・欠測・status対応](../tables/convergence-panel-coverage.csv)。',
                     '`05_iteration_times` は通常の `study=basic` benchmarkのseed 0〜4のうち、成功し反復時間trace検査を通過した実行だけを使う別図。線は反復ごとの中央値、帯は最小〜最大、凡例n/5は利用できたseed数。座標コピー・trace IOはSGD時間から除外している。10seedの収束診断やその目標到達時間を示す図ではない。',
                     '目標は同じseed・familyのCPU 15反復後stressの1.05倍と1.01倍。最初に到達を観測したcheckpointと直前観測点の幅を記録する。これは実際の初到達を保証する区間ではない。stressが非単調になり、未観測の反復で到達後に再上昇する場合、真の初到達はこの幅より前の可能性がある。upper_ms時点には目標を満たす保存座標があるが、lower_msは真の初到達の下限と断定しない。正確な初到達時刻や未到達条件の外挿は行わない。診断の同期・保存による影響があるため、図2・3の通常実行時間とは分けて読む。','']
            text += ['[最終stress評価の整合監査](../tables/convergence-stress-audit.csv)は、各成功runのoffline反復15とnative stressを比較する。snapshotは重心移動前、nativeは移動後で、数学的には並進でstressは不変だがf32/f64の丸めによりbit一致は要求しない。stress定義・起点数・seedも照合し、相対差0.1%以上を要調査として記録する。監査flagだけで成功runを除外しない。','']
            for factor in [1.05,1.01]:
                for method in METHODS:
                    a=[r for r in arrivals if r['method']==method and r['target_factor']==factor and r['evaluable']]
                    unavailable=sum(r['method']==method and r['target_factor']==factor and not r['evaluable'] for r in arrivals)
                    rebound_count=sum(r['observed_rebound_after_hit'] is True for r in a)
                    final_below_count=sum(r['epoch15_at_or_below_target'] is True for r in a)
                    rebound_display=str(rebound_count) if a else '—（評価可能seedなし）'
                    final_display=f'{final_below_count}/{len(a)}' if a else '—（評価可能seedなし）'
                    if a or unavailable: text += [f'- {method} / {factor:.2f}倍目標：必要checkpointとCPU基準が揃った {len(a)} 実行中 {sum(r["reached"] for r in a)} 実行で到達を観測。観測またはCPU基準の欠測 {unavailable} 実行は到達率の分母から除外。到達観測後に再び目標超を観測したのは {rebound_display} 実行、最終反復15で目標以下だったのは {final_display} 実行。']
            text += ['', '計画上の全10seedを残し、観測なし・CPU基準なし・checkpoint不完全を未達と区別する。下表の欠測は到達率の分母に含めない。観測幅は評価可能で到達を観測したseedだけで、直前観測時刻と初到達を観測した時刻をそれぞれ中央値にした幅（秒）。seed間の隣接観測時刻の幅中央値であり、単一実行の初到達時刻でも、真の初到達を挟む保証区間でもない。CSVには到達観測後の再上昇と最終checkpoint・反復15で目標以下かのflagsも残す。[全seedの時刻幅](../tables/target-arrival.csv)・[集計CSV](../tables/target-arrival-coverage.csv)で確認できる。', '', '| Graph | Family | Method | 計画seed | 評価可能 | 1.05倍到達 | 1.01倍到達 | 欠測 | 1.05倍 観測幅中央値 s | 1.01倍 観測幅中央値 s |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
            coverage_file=base/'tables/target-arrival-coverage.csv'
            if coverage_file.exists():
                with coverage_file.open() as stream:
                    for r in csv.DictReader(stream):
                        intervals=[]
                        for factor in ['1_05','1_01']:
                            lower=r.get(f'interval_lower_ms_median_{factor}'); upper=r.get(f'interval_upper_ms_median_{factor}')
                            intervals.append(f'{float(lower)/1000:.3f}–{float(upper)/1000:.3f}' if lower and upper else '—')
                        text += [f'| {r["graph"]} | {r["family"]} | {r["method"]} | {r["planned_seeds"]} | {r["evaluable_seeds"]} | {r["reached_1_05"]} | {r["reached_1_01"]} | {r["missing_seeds"]} | {intervals[0]} | {intervals[1]} |']
            text += ['']
        else:
            study='tiling' if number==6 else 'pivots'; param='tile_size' if number==6 else 'pivots'
            for graph in sorted({r['graph'] for r in summary if r['study']==study}):
                timing='iteration_time_ms_median' if number==6 else 'algorithm_time_cold_ms_median'
                entries=[r for r in summary if r['study']==study and r['graph']==graph and finite(r.get(timing),positive=True)
                         and complete_five_seed_condition(rows,r,timing.removesuffix('_median'))]
                if entries:
                    best=min(entries,key=lambda r:r[timing])
                    text += [f'- {graph}：5seed完備条件のうち今回測れた最短は {best["method"]}（5/5成功）、{param}={best[param]}、SGD {best["iteration_time_ms_median"]/1000:.3f} 秒、stress {fmt(best["stress_value_median"],1)}。'+(f' cold総時間 {best[timing]/1000:.3f} 秒。' if number==7 else '')]
                else:
                    text += [f'- {graph}：5seed完備の比較なし。部分成功値は図・CSV・coverageに保存し、最短条件の選択には使わない。']
            if number==6:
                endpoints=[]
                for graph in sorted({r['graph'] for r in summary if r['study']=='tiling'}):
                    for tile in [1,1024]:
                        matched=[r for r in pairs if r['study']=='tiling' and r['graph']==graph and r['method']=='rr_sgd' and r['tile_size']==tile and r['seed'] in range(5)]
                        seeds={r['seed'] for r in matched}
                        timed=[r['iteration_time_ms']/1000 for r in rows if r['study']=='tiling' and r['graph']==graph and r.get('method')=='rr_sgd' and r.get('tile_size',r.get('rr_tile_size'))==tile and r.get('status')=='success' and r['seed'] in seeds and finite(r.get('iteration_time_ms'),positive=True)]
                        if matched and timed:
                            ratio=float(np.median([r['stress_ratio'] for r in matched])); seconds=float(np.median(timed))
                            text += [f'- {graph} / T={tile}：同seed CPU stress比の中央値 {ratio:.3f}、SGD {seconds:.3f} 秒（対応 {len(seeds)}/5 seed）。']
                            endpoints.append(dict(graph=graph,tile_size=tile,paired_seeds=len(seeds),cpu_stress_ratio_median=ratio,iteration_seconds_median=seconds))
                dump_csv(base/'tables/tile-endpoint-cpu-comparison.csv',endpoints)
            coverage_name='tiling' if number==6 else 'pivots'
            text += [f'全予定パラメータをx軸に残し、全欠測は軸下端の× 0/5、部分成功は手法色のn/5で示す。欠測はNaNとして線を切り、0の測定値へ置換しない。[条件別・指標別のcoverage](../tables/{coverage_name}-parameter-coverage.csv)で成功・欠測・未記録数を確認できる。','']
            text += ['', '時間だけで選ぶと品質を落とす可能性があるため、同じ条件のstressも併記する。時間–品質散布図はseed 0–4の5件すべてが成功したTだけを描き、Tラベルを付ける。CPU水平線と散布図のCPU点はbasic Full CPUの5seedが完備したときだけ表示する別枠の参照であり、T比較の測定点ではない。欠測・部分成功条件はtile-pareto.csvに残し、散布図に値を補わない。破線は測定した中央値に対して非劣位の条件を結ぶもので、未測定条件を含む最適性の証明ではない。最短条件は今回の測定範囲での値。' if number==6 else 'pivot増加は制約数・前処理・メモリと品質の交換になる。主図の時間は前処理・GPU初期化・SGD・readbackを含むcoldアルゴリズム総時間で、入力読込・stress評価・描画ファイル保存を含まない。summary.csvにはSGD単独・前処理・process wall・peak RSSも保存。Peak RSSはGPU専用VRAM量ではない。','']
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
        labels={'not_acquired':'入力データ未取得','identity_unverified':'代替配布との同一性未確認','size_excluded':'LCCがlarge基準未満のため規模除外','other_ineligible':'その他の適格性除外','other_unavailable':'その他の入力利用不能'}
        for kind,count in excluded_conditions.items():
            names=[r['name'] for r in datasets if dataset_exclusion_kind(r)==kind]
            text += [f'- {labels.get(kind,kind)}：{len(names)} グラフ（{", ".join(names)}）、未起動 {count} 計画条件。実行時のtimeoutやGPU事故とは区別する。']
        unavailable=sum(r.get('status')=='unavailable' and dataset_exclusion_kind(catalog.get(r['graph'],{}))=='eligible' for r in rows)
        if unavailable: text += [f'- 適格データだが利用不能と記録された条件 {unavailable} 件：個別のfailure記録を確認する。']
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
    esa=report.split('## 保存先')[0]
    def external_link(match):
        label,target=match.groups(); relative=target.removeprefix('../')
        return f'[{label}]({esa_artifact_base_url.rstrip("/")}/{relative})' if esa_artifact_base_url else f'{label}（保存先: {relative}）'
    # Local report links resolve from report/. Pasted esa content must not link
    # to esa.io/tables; image references remain for the uploader to replace.
    esa=re.sub(r'(?<!!)\[([^\]]+)\]\((\.\./[^)]+)\)',external_link,esa)
    (base/'report/esa-results.md').write_text(esa)
    (base/'README.md').write_text('# v3 実験\n\n[結果](report/RESULTS.md) · [集計CSV](tables/summary.csv)\n\n再集計：`python3 experiments/summarize_v3.py`\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=BASE)
    parser.add_argument('--esa-artifact-base-url',help='Published v3 artifact root for CSV links in esa draft; omit to show plain storage paths')
    parser.add_argument('--skip-convergence',action='store_true',help='Reuse evaluated checkpoints without recomputing stress')
    args=parser.parse_args(argv); base=args.results_dir.resolve()
    for folder in ['figures','tables','report']: (base/folder).mkdir(parents=True,exist_ok=True)
    datasets,rows,cohort=load(base,with_cohort=True); summary=aggregate(rows); pairs,excluded=pair_rows(rows)
    cohort_summary={key:value for key,value in cohort.items() if key not in ('record_provenance','superseded_records')}
    dump_csv(base/'tables/cohort-source-records.csv',cohort['record_provenance'])
    dump_csv(base/'tables/cohort-superseded.csv',cohort['superseded_records'])
    (base/'tables/cohort.json').write_text(json.dumps(cohort_summary,ensure_ascii=False,indent=2)+'\n')
    (base/'tables/effective-datasets.json').write_text(json.dumps(datasets,ensure_ascii=False,indent=2)+'\n')
    incidents=[]; figures=[]
    layouts(base,rows,figures,incidents); comparisons(base,datasets,pairs,figures,rows); structure(base,datasets,pairs,figures)
    structure_supplement(base,datasets,rows,pairs,figures)
    sparse_write_load_figure(base,figures)
    checkpoints,arrivals=convergence(base,datasets,rows,figures,incidents,args.skip_convergence)
    sensitivity(base,rows,pairs,figures)
    tile_pareto(base,rows,figures)
    iteration_timings(base,rows,figures); sparse_components(base,datasets,rows,figures)
    for name,data in [('runs',rows),('summary',summary),('paired-ratios',pairs),('pair-exclusions',excluded),('failures',[r for r in rows if r.get('status')!='success']),('summary-incidents',incidents)]:
        dump_csv(base/'tables'/f'{name}.csv',data)
    report(base,datasets,rows,summary,pairs,figures,arrivals,incidents,excluded,args.esa_artifact_base_url,cohort_summary)
    (base/'tables/provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=len(rows),figures=figures,
        source_results_sha256=hashlib.sha256((base/'raw/results.jsonl').read_bytes()).hexdigest() if (base/'raw/results.jsonl').exists() else None,
        commits=sorted({r['git_commit'] for r in rows if r.get('git_commit')}),
        cohort=cohort_summary,
        figure_png_bytes={name:(base/'figures'/f'{name}.png').stat().st_size for name in figures},
        esa_artifact_base_url=args.esa_artifact_base_url),indent=2)+'\n')
    print(f'{len(rows)} records; {len(pairs)} paired comparisons; {len(checkpoints)} checkpoints; {len(figures)} figures')

if __name__=='__main__': main()
