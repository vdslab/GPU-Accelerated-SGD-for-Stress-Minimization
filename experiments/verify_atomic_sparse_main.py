#!/usr/bin/env python3
"""Independently verify raw coverage, paired hierarchy and saved aggregate data."""
import collections,csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5.5/atomic-sparse-main-20261110'
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    jobs=load(BASE/'config/jobs.json');env=load(BASE/'config/environment.json')
    paths=sorted((BASE/'raw').glob('*/record.json'));rows=[load(p) for p in paths]
    jobs=[j for j in jobs if j['graph'] not in {'com-DBLP','web-NotreDame','com-Amazon','coPapersCiteseer'}]
    assert len(rows)==len(jobs)==780
    assert {r['run_id'] for r in rows}=={j['run_id'] for j in jobs}
    groups=collections.defaultdict(list)
    paired=[]
    for r in rows:
        assert r['status']=='success' and not r['warmup']
        assert r['git_commit']==env['merged_main_commit'] and not r['git_dirty']
        assert r['binary_sha256']==env['binary_sha256']
        assert r['iterations']==15 and r['epsilon']==.1 and r['pivots']==min(200,r['nodes'])
        assert r['attempted_updates']==r['completed_updates']==15*r['constraints']
        assert r['retry_failures']==0
        cpath=ROOT/'experiment_results/v5/raw/comparison'/r['previous_run_id'].replace('-atomic_sparse_sgd-', '-sparse_sgd-')/'record.json'
        c=load(cpath)
        for f in ['input_sha256','initial_positions_sha256','preprocess_sha256','stress_kind','stress_samples','stress_seed']:
            assert r[f]==c[f],(r['run_id'],f)
        manifest=load(Path(r['archive_manifest_path']))
        assert manifest['errors']==[] and manifest['measurement_excluded']
        for f in manifest['files']:
            assert Path(f['compressed_path']).exists()
            assert f['original_sha256']==f['decompressed_sha256'] and f['xz_crc_validated']
        d=dict(graph=r['graph'],seed=r['seed'],repetition=r['repetition'],
            sgd_speedup=(c['sgd_time_ms']+(c.get('readback_time_ms') or 0))/(r['sgd_time_ms']+(r.get('readback_time_ms') or 0)),
            total_speedup=c['total_compute_time_ms']/r['total_compute_time_ms'],
            stress_ratio=r['stress_value']/c['stress_value'])
        groups[r['graph']].append(d);paired.append(d)
    assert len(groups)==52
    independent={}
    for g,rs in groups.items():
        assert {(r['seed'],r['repetition']) for r in rs}=={(s,p) for s in range(5) for p in range(3)}
        independent[g]={}
        for f in ['sgd_speedup','total_speedup','stress_ratio']:
            medians=[statistics.median(r[f] for r in rs if r['seed']==s) for s in range(5)]
            independent[g][f]=dict(median=statistics.median(medians),min=min(medians),max=max(medians))
    summary=load(BASE/'tables/hierarchical.json')
    assert len(summary)==52
    for r in summary:
        for f in ['sgd_speedup','total_speedup','stress_ratio']:
            for stat in ['median','min','max']:
                assert math.isclose(r[f][stat],independent[r['graph']][f][stat],rel_tol=1e-12)
    for r in load(BASE/'config/v5-controls.json'):
        assert digest(Path(r['reference_record']))==r['reference_record_sha256']
    means={f:math.exp(statistics.mean(math.log(v[f]['median']) for v in independent.values())) for f in ['sgd_speedup','total_speedup','stress_ratio']}
    for f,v in means.items():assert math.isclose(v,load(BASE/'tables/geometric-means.json')[f],rel_tol=1e-12)
    output=dict(success=True,formal_runs=780,graphs=52,seeds=5,repetitions=3,failed=0,
        excluded_after_user_scope_change=60,main_commit=env['merged_main_commit'],binary_sha256=env['binary_sha256'],
        input_initial_preprocess_and_stress_protocol_matched=True,all_updates_completed=True,
        independent_paired_hierarchy_matches_report=True,source_records_unchanged=True,
        geometric_means=means)
    (BASE/'validation/independent-check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
