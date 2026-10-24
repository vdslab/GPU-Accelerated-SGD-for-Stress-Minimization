"""Read-only, all-or-nothing SNAP-origin recovery overlay for v3 analysis.

Original core/recovery configs and raw records are never rewritten. Large source,
mapping, coordinate and archive payloads are not rehashed here: acquisition and
native archive validation receipts supply those hashes. The small receipts and
frozen JSON configuration files are hashed again when an overlay is considered.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_DIRECTORY = 'snap-origin-recovery'
PREPARATION_POLICY = 'v3-nonzero-simple-undirected-largest-component-1'
CORE_NATIVE_COMMIT = '118d3ec22b350fcc7522527154f3d03131215a25'
IDENTITY_CODE = 'official_snap_origin_unverified_suitesparse_equivalence'
SNAP_ORIGINS = {
    'com-Amazon': (334863, 925872, 'bigdata/communities/com-amazon.ungraph.txt.gz'),
    'com-DBLP': (317080, 1049866, 'bigdata/communities/com-dblp.ungraph.txt.gz'),
    'com-Youtube': (1134890, 2987624, 'bigdata/communities/com-youtube.ungraph.txt.gz'),
    'com-LiveJournal': (3997962, 34681189, 'bigdata/communities/com-lj.ungraph.txt.gz'),
    'soc-Pokec': (1632803, 30622564, 'soc-pokec-relationships.txt.gz'),
    'loc-Gowalla': (196591, 950327, 'loc-gowalla_edges.txt.gz'),
}
METHODS = ('sparse_sgd', 'atomic_sparse_sgd', 'rr_sparse_sgd')
STATUSES = {'success', 'error', 'timeout', 'memory_limit', 'disk_limit', 'unavailable', 'warmup_failed'}
PATH_FIELDS = ('artifact_dir', 'trace_dir', 'archive_manifest_path', 'input_path',
               'initial_positions_path', 'final_positions_path', 'vertex_map_path',
               'native_initial_positions_path', 'native_final_positions_path',
               'native_vertex_map_path', 'stderr_log_path')
DATASET_PATH_FIELDS = ('path', 'vertex_map_path', 'source_path', 'origin_path',
                       'matrix_node_ids_path', 'lcc_origin_node_ids_path')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_path(value, base):
    path = Path(value)
    # macOS /var and /private/var (and other symlink aliases) can name the
    # same artifact. Compare canonical absolute paths in both branches.
    if path.is_absolute(): return path.resolve()
    for parent in (Path(base), ROOT):
        if (parent/path).exists(): return (parent/path).resolve()
    return (Path(base)/path).resolve()


def physical(row):
    value = row.get('process_wall_seconds')
    return (not row.get('reused_from') and row.get('executed', True)
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def blocking_warmup_issues(row, warmup):
    """Validate a blocking reference without treating input exclusions as runs."""
    if warmup is None:
        return [dict(kind='blocking_warmup_missing')]
    issues=[]
    for field in ('graph','method','pivots','tile_size'):
        if row.get(field)!=warmup.get(field):
            issues.append(dict(kind='blocking_warmup_parameter_mismatch',field=field))
    if warmup.get('status')=='success':
        issues.append(dict(kind='blocking_warmup_was_success'))
    if row.get('failure_reason')!=warmup.get('status'):
        issues.append(dict(kind='blocking_warmup_failure_reason_mismatch',
                           failure_reason=row.get('failure_reason'),warmup_status=warmup.get('status')))
    return issues


def read_records(path):
    rows, issues = [], []
    if not path.is_file(): return rows, issues
    for index,line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip(): continue
        try:
            row=json.loads(line)
            if not isinstance(row,dict) or not isinstance(row.get('run_id'),str):
                raise ValueError('record requires run_id')
            rows.append(row)
        except (json.JSONDecodeError,ValueError) as error:
            issues.append(dict(line=index,error=str(error)))
    return rows, issues


def normalized_record(row, base, scope):
    result=dict(row,analysis_origin_base=str(base),analysis_origin_scope=scope)
    result['graph']=result.get('graph',result.get('dataset'))
    result['study']=result.get('study',result.get('study_group','basic'))
    for field in PATH_FIELDS:
        if result.get(field): result[field]=str(resolve_path(result[field],base))
    return result


def normalized_dataset(row, base):
    result=dict(row,analysis_origin_base=str(base))
    for field in DATASET_PATH_FIELDS:
        if result.get(field): result[field]=str(resolve_path(result[field],base))
    return result


def source_configuration(base):
    config=base/'config'
    plan=read_json(config/'plan.json')
    if isinstance(plan,dict): plan=plan.get('jobs',plan.get('tasks',[]))
    return dict(base=base,plan=plan,datasets=read_json(config/'datasets.json'),
                execution=read_json(config/'execution.json'),limits=read_json(config/'limits.json'),
                hashes=read_json(config/'source-hashes.json'))


def active_runners(bases):
    import psutil
    found=[]
    for process in psutil.process_iter(['pid','cmdline','status'],ad_value=None):
        try:
            command=process.info.get('cmdline') or []
            if process.info.get('status')==psutil.STATUS_ZOMBIE or '--pilot' in command: continue
            if not any(Path(value).name=='run_v3.py' for value in command): continue
            configured=ROOT/'experiment_results/v3'
            if '--results-dir' in command:
                configured=Path(command[command.index('--results-dir')+1])
                if not configured.is_absolute(): configured=Path(process.cwd())/configured
            if configured.resolve() in bases: found.append(dict(pid=process.pid,base=str(configured.resolve())))
        except (psutil.Error,OSError,IndexError): continue
    return found


def validate_source_configuration(source, records, expected_count, errors):
    base=source['base']; execution=source['execution']; plan=source['plan']
    plans={row.get('run_id'):row for row in plan}
    ids=Counter(row.get('run_id') for row in records)
    prefix=str(base)
    def problem(kind, **detail): errors.append(dict(base=prefix,kind=kind,**detail))
    if len(plan)!=expected_count or len(plans)!=expected_count or None in plans:
        problem('plan_count_or_duplicate',expected=expected_count,actual=len(plan),unique=len(plans))
    if set(ids)!=set(plans) or any(count!=1 for count in ids.values()): problem('result_ids_not_exactly_complete')
    canonical=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if canonical!=execution.get('plan_sha256'): problem('frozen_plan_sha_mismatch')
    if file_sha256(base/'config/datasets.json')!=execution.get('datasets_sha256'): problem('frozen_datasets_sha_mismatch')
    if source['limits']!=execution.get('limits'): problem('frozen_limits_mismatch')
    if source['hashes'].get('experiments/run_v3.py')!=execution.get('runner_sha256'): problem('frozen_runner_sha_mismatch')
    if not isinstance(execution.get('git_commit'),str): problem('execution_commit_missing')
    catalog={row['name']:row for row in source['datasets']}
    if len(catalog)!=len(source['datasets']): problem('duplicate_dataset_names')
    warmups=[]
    for path in sorted((base/'warmups').glob('*/record.json')):
        try: warmups.append(read_json(path))
        except (OSError,ValueError) as error: problem('warmup_unreadable',path=str(path),error=str(error))
    warmup_ids={row.get('run_id'):row for row in warmups}
    for scope,group in [('formal',records),('warmup',warmups)]:
        for row in group:
            run_id=row.get('run_id'); info=catalog.get(row.get('graph',row.get('dataset')), {})
            if row.get('status') not in STATUSES: problem('unknown_status',run_id=run_id,scope=scope)
            if scope=='formal':
                planned=plans.get(run_id,{})
                for key in ('study','graph','method','seed','iterations','epsilon','pivots','tile_size','workgroup_size'):
                    if row.get(key)!=planned.get(key): problem('plan_parameter_mismatch',run_id=run_id,field=key)
            if physical(row):
                if row.get('graph',row.get('dataset')) not in catalog or row.get('method') not in execution.get('binary_sha256',{}):
                    problem('physical_graph_or_method_unknown',run_id=run_id,scope=scope)
                if row.get('binary_sha256')!=execution.get('binary_sha256',{}).get(row.get('method')):
                    problem('binary_record_sha_mismatch',run_id=run_id,scope=scope)
                if row.get('input_sha256')!=info.get('sha256'): problem('input_record_sha_mismatch',run_id=run_id,scope=scope)
                for field in ('nodes','edges'):
                    if row.get(field) is not None and row[field]!=info.get(field):
                        problem('native_prepared_graph_size_mismatch',run_id=run_id,scope=scope,field=field)
            if row.get('git_commit') is not None and row.get('git_commit')!=execution.get('git_commit'):
                problem('native_commit_mismatch',run_id=run_id,scope=scope)
            if row.get('status')=='success':
                if not physical(row) and not row.get('reused_from'): problem('success_without_physical_metadata',run_id=run_id,scope=scope)
                if row.get('git_commit')!=execution.get('git_commit'): problem('success_native_commit_mismatch',run_id=run_id,scope=scope)
            if scope=='formal' and row.get('status')=='warmup_failed':
                for issue in blocking_warmup_issues(row,warmup_ids.get(row.get('warmup_run_id'))):
                    problem(issue['kind'],run_id=run_id,**{k:v for k,v in issue.items() if k!='kind'})


def provenance_row(row, base, scope, superseded=False):
    fields=('run_id','study','graph','method','seed','status','executed','process_wall_seconds',
            'git_commit','binary_sha256','input_sha256','failure_reason','warmup_run_id')
    return dict(origin_base=str(base),origin_scope=scope,superseded=superseded,
                effective=not superseded,physical_execution=physical(row),**{key:row.get(key) for key in fields})


def load_effective_cohort(base):
    """Return datasets, effective rows, and provenance; incomplete recovery is ignored."""
    base=Path(base).resolve(); recovery=base/RECOVERY_DIRECTORY
    datasets=read_json(base/'config/datasets.json')
    core_rows,core_issues=read_records(base/'raw/results.jsonl')
    latest={row['run_id']:row for row in core_rows}
    normalized=[normalized_record(row,base,'core') for row in latest.values()]
    normalized_datasets=[normalized_dataset(row,base) for row in datasets]
    metadata=dict(status='absent',main_base=str(base),recovery_base=str(recovery),
                  loader_sha256=file_sha256(__file__),
                  core_raw_records=len(core_rows),recovery_raw_records=0,raw_record_count=len(core_rows),
                  effective_conditions=len(normalized),superseded_condition_count=0,errors=[],
                  record_provenance=[provenance_row(row,base,'core') for row in core_rows],
                  superseded_records=[],execution_commits_by_base={},
                  source_results_sha256_by_base={str(base):file_sha256(base/'raw/results.jsonl')} if (base/'raw/results.jsonl').is_file() else {},
                  source_equivalence='SNAP origin is the named original distribution; SuiteSparse MTX byte/edgewise equivalence is not verified.')
    if not recovery.exists(): return normalized_datasets,normalized,metadata
    metadata['status']='incomplete'
    recovery_rows,recovery_issues=read_records(recovery/'raw/results.jsonl')
    metadata.update(recovery_raw_records=len(recovery_rows),raw_record_count=len(core_rows)+len(recovery_rows))
    if (recovery/'raw/results.jsonl').is_file(): metadata['source_results_sha256_by_base'][str(recovery)]=file_sha256(recovery/'raw/results.jsonl')
    metadata['record_provenance'] += [dict(provenance_row(row,recovery,'snap_origin_recovery'),effective=False) for row in recovery_rows]
    recovery_ids=Counter(row['run_id'] for row in recovery_rows)
    if len(recovery_rows)>90 or any(count>1 for count in recovery_ids.values()):
        metadata['status']='rejected'; metadata['errors'].append(dict(kind='recovery_extra_or_duplicate_records'))
        return normalized_datasets,normalized,metadata
    if len(recovery_rows)!=90 or len(recovery_ids)!=90 or recovery_issues:
        metadata['incomplete_reason']='Recovery needs all 90 unique finalized records; no partial overlay is permitted.'
        return normalized_datasets,normalized,metadata
    errors=metadata['errors']
    try:
        if active_runners({base,recovery}):
            metadata['incomplete_reason']='A core or recovery native runner remains active.'
            return normalized_datasets,normalized,metadata
        core=source_configuration(base); extra=source_configuration(recovery)
        if core_issues: errors.append(dict(kind='core_invalid_result_json',issues=core_issues))
        validate_source_configuration(core,core_rows,3210,errors)
        validate_source_configuration(extra,recovery_rows,90,errors)
        if core['execution'].get('git_commit')!=CORE_NATIVE_COMMIT: errors.append(dict(kind='core_native_commit_not_frozen_118d3ec'))
        targets=set(SNAP_ORIGINS)
        expected={row['run_id']:row for row in core['plan'] if row.get('study')=='basic' and row.get('graph') in targets}
        if len(expected)!=90 or {row['run_id']:row for row in extra['plan']}!=expected:
            errors.append(dict(kind='recovery_plan_not_exact_original_90'))
        for graph in targets:
            subset=[r for r in expected.values() if r['graph']==graph]
            if {(r.get('method'),r.get('seed')) for r in subset}!={(method,seed) for method in METHODS for seed in range(5)}:
                errors.append(dict(kind='recovery_not_basic_3_methods_5_seeds',graph=graph))
        for run_id in expected:
            old=latest.get(run_id,{})
            old_artifacts=resolve_path(old['artifact_dir'],base) if old.get('artifact_dir') else None
            launch_evidence=bool(old_artifacts and (old_artifacts/'command.json').is_file())
            if (old.get('status')!='unavailable' or physical(old) or old.get('executed') is True
                    or old.get('reused_from') or old.get('binary_sha256') is not None
                    or old.get('input_sha256') is not None or old.get('git_commit') is not None or launch_evidence):
                errors.append(dict(kind='recovery_would_overwrite_attempted_or_available_condition',run_id=run_id))
        native_core={k:v for k,v in core['hashes'].items() if k.startswith(('implementations/','shared/'))}
        native_extra={k:v for k,v in extra['hashes'].items() if k.startswith(('implementations/','shared/'))}
        if len(native_core)!=67 or native_core!=native_extra: errors.append(dict(kind='native_shared_67_file_set_or_sha_mismatch'))
        if core['execution'].get('binary_sha256')!=extra['execution'].get('binary_sha256'): errors.append(dict(kind='binary_set_sha_mismatch'))
        if core['execution'].get('runner_sha256')!=extra['execution'].get('runner_sha256'): errors.append(dict(kind='runner_sha_mismatch'))
        if core['hashes'].get('experiments/prepare_v3.py')!=extra['hashes'].get('experiments/prepare_v3.py'):
            errors.append(dict(kind='preparation_script_sha_mismatch'))
        if core['limits']!=extra['limits']: errors.append(dict(kind='cross_base_limits_mismatch'))
        manifest=read_json(recovery/'config/recovery.json')
        if manifest.get('schema_version')!=1 or manifest.get('kind')!='official_snap_origin_recovery': errors.append(dict(kind='recovery_manifest_schema'))
        if len(manifest.get('target_graphs',[]))!=6 or set(manifest.get('target_graphs',[]))!=targets:
            errors.append(dict(kind='recovery_manifest_target_graphs'))
        if resolve_path(manifest.get('main_base',''),recovery)!=base: errors.append(dict(kind='recovery_manifest_main_base'))
        for field,relative in [('main_execution_sha256','config/execution.json'),('main_results_sha256','raw/results.jsonl'),
                               ('main_datasets_sha256','config/datasets.json'),('main_plan_sha256','config/plan.json'),
                               ('main_source_hashes_sha256','config/source-hashes.json')]:
            if manifest.get(field)!=file_sha256(base/relative): errors.append(dict(kind='recovery_main_snapshot_sha_mismatch',field=field))
        if manifest.get('preparation_policy')!=PREPARATION_POLICY: errors.append(dict(kind='recovery_manifest_preparation_policy'))
        if manifest.get('identity_code')!=IDENTITY_CODE or not isinstance(manifest.get('identity_claim'),str) or not manifest['identity_claim'].strip():
            errors.append(dict(kind='recovery_identity_claim_missing_or_changed'))
        sources={row.get('name'):row for row in manifest.get('sources',[])}
        extra_catalog={row['name']:row for row in extra['datasets']}; core_catalog={row['name']:row for row in datasets}
        if len(manifest.get('sources',[]))!=6 or set(sources)!=targets or set(extra_catalog)!=targets:
            errors.append(dict(kind='recovery_dataset_or_source_names'))
        for graph,(nodes,edges,origin_suffix) in SNAP_ORIGINS.items():
            source=sources.get(graph,{}); info=extra_catalog.get(graph,{})
            origin=urlparse(source.get('origin_url',''))
            if origin.scheme!='https' or origin.hostname!='snap.stanford.edu' or origin.path!='/data/'+origin_suffix:
                errors.append(dict(kind='unapproved_snap_origin_url',graph=graph))
            if (info.get('source_distribution_status')!='official_snap_origin_recovery'
                    or info.get('status')!='ready' or info.get('preparation_policy')!=PREPARATION_POLICY
                    or info.get('family')!='large' or info.get('source_nodes')!=nodes
                    or info.get('source_graph')!=core_catalog.get(graph,{}).get('source_graph')):
                errors.append(dict(kind='recovery_dataset_origin_or_preparation_metadata',graph=graph))
            if (not isinstance(info.get('nodes'),int) or not isinstance(info.get('edges'),int)
                    or info.get('nodes')!=nodes or info.get('edges',-1)<0
                    or info.get('eligible')!=(info.get('nodes',0)>=100000)):
                errors.append(dict(kind='recovery_prepared_size_or_eligibility',graph=graph))
            for path_key,sha_key in [('origin_path','origin_sha256'),('converted_mtx_path','converted_mtx_sha256'),
                                     ('matrix_node_ids_path','matrix_node_ids_sha256'),('prepared_path','prepared_sha256'),
                                     ('vertex_map_path','vertex_map_sha256'),('lcc_origin_node_ids_path','lcc_origin_node_ids_sha256'),
                                     ('validation_path','validation_sha256')]:
                value=source.get(path_key); digest=source.get(sha_key)
                if not value or not isinstance(digest,str) or len(digest)!=64 or not resolve_path(value,recovery).is_file():
                    errors.append(dict(kind='recovery_source_artifact_metadata_or_file_missing',graph=graph,field=path_key))
            validation_path=resolve_path(source.get('validation_path',''),recovery)
            if validation_path.is_file():
                if file_sha256(validation_path)!=source.get('validation_sha256'): errors.append(dict(kind='recovery_validation_receipt_sha_mismatch',graph=graph))
                validation=read_json(validation_path)
                observed_edge_field='observed_unique_directed_edges' if graph=='soc-Pokec' else 'observed_unique_edges'
                if (validation.get('schema_version')!=1 or validation.get('kind')!='official_snap_origin_validation'
                        or validation.get('identity_code')!=IDENTITY_CODE
                        or validation.get('name')!=graph or validation.get('metadata_matches_official') is not True
                        or validation.get('official_nodes')!=nodes or validation.get('official_edges')!=edges
                        or validation.get('observed_unique_nodes')!=nodes or validation.get(observed_edge_field)!=edges):
                    errors.append(dict(kind='official_source_observed_nodes_or_unique_edges_mismatch',graph=graph))
                # Bind the checked receipt to the same download/conversion/map
                # attempt advertised by the manifest. No large payload is read.
                for field in ('origin_url','origin_sha256','origin_bytes','converted_mtx_sha256',
                              'matrix_node_ids_sha256','vertex_map_sha256','lcc_origin_node_ids_sha256'):
                    if validation.get(field)!=source.get(field):
                        errors.append(dict(kind='recovery_validation_source_receipt_mismatch',graph=graph,field=field))
                for field in ('matrix_node_ids_path','vertex_map_path','lcc_origin_node_ids_path'):
                    if (not validation.get(field)
                            or resolve_path(validation[field],recovery)!=resolve_path(source.get(field,''),recovery)):
                        errors.append(dict(kind='recovery_validation_source_path_mismatch',graph=graph,field=field))
                expected_nnz=edges if graph=='soc-Pokec' else 2*edges
                if (validation.get('gzip_crc_validated') is not True
                        or validation.get('preparation_policy')!=PREPARATION_POLICY
                        or validation.get('official_expanded_nnz')!=expected_nnz
                        or validation.get('observed_expanded_nnz')!=expected_nnz
                        or validation.get('official_edges_are_directed') is not (graph=='soc-Pokec')
                        or validation.get('source_components')!=1):
                    errors.append(dict(kind='recovery_validation_crc_or_graph_invariants_mismatch',graph=graph))
                unordered=validation.get('observed_unique_undirected_edges')
                loops=validation.get('unique_self_loops')
                if (not isinstance(unordered,int) or isinstance(unordered,bool)
                        or not isinstance(loops,int) or isinstance(loops,bool)
                        or not 0<=loops<=unordered or unordered!=validation.get('observed_unique_edges')
                        or validation.get('prepared_edges')!=unordered-loops
                        or validation.get('observed_undirected_expanded_nnz')!=2*(unordered-loops)+loops):
                    errors.append(dict(kind='recovery_validation_undirected_or_prepared_edges_mismatch',graph=graph))
                if (not isinstance(source.get('origin_bytes'),int) or isinstance(source.get('origin_bytes'),bool)
                        or source['origin_bytes']<=0):
                    errors.append(dict(kind='recovery_origin_byte_count_missing_or_invalid',graph=graph))
                for key,expected_value in [('prepared_nodes',info.get('nodes')),('prepared_edges',info.get('edges')),('prepared_sha256',info.get('sha256'))]:
                    if validation.get(key)!=expected_value: errors.append(dict(kind='recovery_prepared_receipt_mismatch',graph=graph,field=key))
            if (info.get('sha256')!=source.get('prepared_sha256') or info.get('vertex_map_sha256')!=source.get('vertex_map_sha256')
                    or info.get('source_sha256')!=source.get('converted_mtx_sha256')):
                errors.append(dict(kind='recovery_dataset_receipt_sha_mismatch',graph=graph))
            for field in ('origin_url','origin_sha256','matrix_node_ids_sha256'):
                if info.get(field)!=source.get(field):
                    errors.append(dict(kind='recovery_dataset_source_receipt_mismatch',graph=graph,field=field))
            if info.get('recovery_source')!=source:
                errors.append(dict(kind='recovery_dataset_source_manifest_mismatch',graph=graph))
            for field,source_field in [('path','prepared_path'),('vertex_map_path','vertex_map_path'),
                                       ('source_path','converted_mtx_path'),('origin_path','origin_path'),
                                       ('matrix_node_ids_path','matrix_node_ids_path')]:
                if resolve_path(info.get(field,''),recovery)!=resolve_path(source.get(source_field,''),recovery):
                    errors.append(dict(kind='recovery_dataset_receipt_path_mismatch',graph=graph,field=field))
        metadata.update(native_shared_source_files=len(native_core),runner_files=1,
                        execution_commits_by_base={str(base):core['execution'].get('git_commit'),str(recovery):extra['execution'].get('git_commit')},
                        config_sha256_by_base={str(b):{name:file_sha256(b/'config'/name) for name in ('datasets.json','plan.json','execution.json','limits.json','source-hashes.json')} for b in (base,recovery)},
                        recovery_manifest_sha256=file_sha256(recovery/'config/recovery.json'))
        if errors:
            metadata['status']='rejected'
            return normalized_datasets,normalized,metadata
        replacements={row['run_id']:normalized_record(row,recovery,'snap_origin_recovery') for row in recovery_rows}
        effective=[replacements.get(row['run_id'],row) for row in normalized]
        combined_datasets=[normalized_dataset(extra_catalog[row['name']],recovery) if row['name'] in targets else normalized_dataset(row,base) for row in datasets]
        metadata.update(status='adopted',effective_conditions=len(effective),superseded_condition_count=90,
                        same_native_shared_source_hashes=True,same_runner_hash=True,same_binary_hashes=True,same_limits=True,
                        same_preparation_script_hash=True,
                        superseded_records=[dict(normalized_record(row,base,'core'),superseded_by_origin_base=str(recovery)) for row in core_rows if row['run_id'] in replacements],
                        record_provenance=[provenance_row(row,base,'core',row['run_id'] in replacements) for row in core_rows]
                                          +[provenance_row(row,recovery,'snap_origin_recovery') for row in recovery_rows])
        return combined_datasets,effective,metadata
    except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
        metadata['status']='rejected'; errors.append(dict(kind='recovery_validation_exception',error=f'{type(error).__name__}: {error}'))
        return normalized_datasets,normalized,metadata
