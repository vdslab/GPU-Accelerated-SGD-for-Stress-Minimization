#!/usr/bin/env python3
"""Audit final v3 plan/provenance/status JSON without reading large coordinates.

The audit does not rerun experiments, hash binaries, or decompress artifacts.
By default it refuses to run while the formal runner is alive or the plan is
incomplete. --allow-incomplete permits a lightweight provisional audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
try:
    from v3_recovery import RECOVERY_DIRECTORY, blocking_warmup_issues, load_effective_cohort
except ModuleNotFoundError:
    from experiments.v3_recovery import RECOVERY_DIRECTORY, blocking_warmup_issues, load_effective_cohort

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiment_results/v3'
EXPECTED_PLAN_COUNT = 3210
REBOOT_RECEIPT = Path('interruption-reboot-20261024/receipt.json')
STATUSES = {'success', 'error', 'timeout', 'memory_limit', 'disk_limit',
            'unavailable', 'warmup_failed'}


def read_json(path: Path):
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with path.open('w', newline='') as target:
        writer = csv.DictWriter(target, fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def resolve(value, base):
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path if (base / path).exists() else ROOT / path


def active_runners(base):
    import psutil
    found = []
    for process in psutil.process_iter(['pid', 'cmdline', 'status'], ad_value=None):
        try:
            command = process.info.get('cmdline') or []
            if process.info.get('status') == psutil.STATUS_ZOMBIE or '--pilot' in command:
                continue
            if not any(Path(value).name == 'run_v3.py' for value in command):
                continue
            configured = BASE
            if '--results-dir' in command:
                value = command[command.index('--results-dir') + 1]
                configured = Path(value)
                if not configured.is_absolute():
                    configured = Path(process.cwd()) / configured
            if configured.resolve() == base:
                found.append({'pid': process.pid, 'command': command})
        except (psutil.Error, OSError, IndexError):
            continue
    return found


def exclusion(info):
    if str(info.get('source_distribution_status', '')).startswith('alternative_distribution'):
        return 'identity_unverified'
    if info.get('status') == 'ready' and not info.get('eligible', True):
        return 'size_excluded' if info.get('family') == 'large' and (info.get('nodes') or math.inf) < 100000 else 'other_ineligible'
    return 'not_acquired' if info.get('status') != 'ready' else 'eligible'


def physical(row):
    return not row.get('reused_from') and row.get('executed', True) and finite(row.get('process_wall_seconds'))


def last_epoch(row, base, warnings):
    directory = row.get('trace_dir')
    if not directory and row.get('artifact_dir'):
        directory = str(Path(row['artifact_dir']) / 'trace')
    if not directory:
        return None
    path = resolve(directory, base) / 'trace.jsonl'
    if not path.exists():
        return None
    epochs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            epoch = value.get('epoch', value.get('iteration', value.get('checkpoint')))
            if isinstance(epoch, int):
                epochs.append(epoch)
        except json.JSONDecodeError:
            warnings.append({'kind': 'partial_trace', 'run_id': row.get('run_id'), 'path': str(path)})
    return max(epochs) if epochs else None


def stage(epoch):
    if epoch is None:
        return 'before_trace_or_trace_missing'
    if epoch == 0:
        return 'before_first_iteration_completed'
    if 1 <= epoch <= 14:
        return 'during_iterations'
    if epoch == 15:
        return 'after_iteration_15_completed'
    return 'unexpected_epoch'


def compact(row, scope):
    keys = ('run_id', 'graph', 'dataset', 'study', 'method', 'seed', 'pivots', 'tile_size',
            'status', 'failure_reason', 'warmup_run_id', 'process_wall_seconds',
            'peak_rss_bytes', 'archive_status', 'reused_from')
    return dict(scope=scope, **{key: row.get(key) for key in keys})


def status_statistics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['scope'], row['status'])].append(row)
    result = []
    for (scope, status), group in sorted(groups.items()):
        summary = {'scope': scope, 'status': status, 'physical_executions': len(group)}
        for field in ('process_wall_seconds', 'peak_rss_bytes'):
            values = [row[field] for row in group if finite(row.get(field))]
            summary.update({field + '_count': len(values), field + '_min': min(values) if values else None,
                            field + '_median': statistics.median(values) if values else None,
                            field + '_max': max(values) if values else None})
            if field == 'process_wall_seconds':
                summary[field + '_sum'] = sum(values) if values else None
        result.append(summary)
    return result


def interrupted_attempts(base, plans, confirmed, errors):
    """Keep the reboot receipt separate from one-result-per-condition records.

    Only the known formal-run receipt is inspected. Pilot and prebuffer-aborted
    directories are intentionally outside this source of additional attempts.
    """
    receipt_path = base / REBOOT_RECEIPT
    if not receipt_path.is_file():
        errors.append({'kind': 'reboot_interruption_receipt_missing', 'path': str(REBOOT_RECEIPT)})
        return []
    try:
        receipt = read_json(receipt_path)
    except (OSError, json.JSONDecodeError) as error:
        errors.append({'kind': 'reboot_interruption_receipt_unreadable',
                       'path': str(REBOOT_RECEIPT), 'error': str(error)})
        return []
    if not isinstance(receipt, dict):
        errors.append({'kind': 'reboot_interruption_receipt_invalid', 'path': str(REBOOT_RECEIPT)})
        return []
    run_id = receipt.get('interrupted_run_id')
    issue_count = len(errors)

    def problem(kind, **detail):
        errors.append(dict(kind=kind, run_id=run_id, receipt_path=str(REBOOT_RECEIPT), **detail))

    if not isinstance(run_id, str) or run_id not in plans:
        problem('interrupted_run_id_not_in_plan')
    elif plans[run_id].get('reuse'):
        problem('interrupted_run_id_is_reuse_condition')
    preserved_value = receipt.get('preserved_artifact_directory')
    preserved = resolve(preserved_value, base) if isinstance(preserved_value, str) else None
    artifact_exists = bool(preserved and preserved.is_dir())
    safe_directory = bool(preserved and preserved.resolve().is_relative_to(receipt_path.parent.resolve()))
    if not artifact_exists:
        problem('interrupted_artifact_directory_missing', path=preserved_value)
    if not safe_directory:
        problem('interrupted_artifact_outside_receipt_directory', path=preserved_value)
    evidence_files = receipt.get('files')
    if not isinstance(evidence_files, list) or not evidence_files:
        problem('interrupted_artifact_manifest_missing')
        evidence_files = []
    verified_files = 0
    if artifact_exists and safe_directory:
        for item in evidence_files:
            relative = item.get('relative_path') if isinstance(item, dict) else None
            candidate = preserved / relative if isinstance(relative, str) else None
            if (not candidate or not candidate.resolve().is_relative_to(preserved.resolve())
                    or not candidate.is_file()):
                problem('interrupted_evidence_file_missing_or_invalid', path=relative)
            elif candidate.stat().st_size != item.get('bytes'):
                problem('interrupted_evidence_file_size_mismatch', path=relative)
            else:
                verified_files += 1
        command_path = preserved / 'command.json'
        try:
            command = read_json(command_path)
            if not isinstance(command, dict):
                raise TypeError('command metadata must be an object')
            argv = command.get('argv', [])
            if not isinstance(argv, list):
                raise TypeError('command argv must be a list')
            command_run_id = argv[argv.index('--run-id') + 1] if '--run-id' in argv else None
            if command_run_id != run_id:
                problem('interrupted_command_run_id_mismatch', command_run_id=command_run_id)
        except (OSError, json.JSONDecodeError, IndexError, TypeError) as error:
            problem('interrupted_command_unreadable', error=str(error))
    relative_artifact = (preserved.resolve().relative_to(base.resolve()).as_posix()
                         if preserved and preserved.resolve().is_relative_to(base.resolve()) else preserved_value)
    plan = plans.get(run_id, {}) if isinstance(run_id, str) else {}
    return [{'scope': 'formal_interrupted_attempt', 'attempt_status': 'interrupted_host_reboot',
             'run_id': run_id, 'graph': plan.get('graph'), 'method': plan.get('method'), 'seed': plan.get('seed'),
             'observed_boot_time_jst': receipt.get('observed_boot_time_jst'),
             'incident': receipt.get('incident'), 'receipt_path': REBOOT_RECEIPT.as_posix(),
             'preserved_artifact_directory': relative_artifact, 'artifact_directory_exists': artifact_exists,
             'evidence_file_count': len(evidence_files), 'verified_evidence_files': verified_files,
             'evidence_valid': len(errors) == issue_count,
             'confirmed_condition_status': confirmed.get(run_id, {}).get('status') if isinstance(run_id, str) else None,
             'counting': 'additional native launch attempt; not an additional planned condition or final failure'}]


def audit(base, allow_incomplete=False, expected_plan_count=EXPECTED_PLAN_COUNT, require_reboot_receipt=True):
    runners = active_runners(base)
    if runners and not allow_incomplete:
        raise SystemExit('Formal v3 runner is active; use --allow-incomplete only for a provisional audit.')
    config = base / 'config'
    plan = read_json(config / 'plan.json')
    if isinstance(plan, dict):
        plan = plan.get('jobs', plan.get('tasks', []))
    execution = read_json(config / 'execution.json')
    datasets = read_json(config / 'datasets.json')
    catalog = {row['name']: row for row in datasets}
    errors, warnings = [], []
    results, partial_lines = [], []
    lines = (base / 'raw/results.jsonl').read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get('run_id'), str):
                errors.append({'kind': 'missing_or_invalid_result_run_id', 'line': index + 1})
                continue
            results.append(value)
        except json.JSONDecodeError:
            issue = {'kind': 'invalid_result_json', 'line': index + 1}
            (warnings if allow_incomplete and index == len(lines) - 1 else errors).append(issue)
            partial_lines.append(index + 1)
    plan_counts = Counter(row.get('run_id') for row in plan)
    result_counts = Counter(row.get('run_id') for row in results)
    expected, actual_ids = set(plan_counts), set(result_counts)
    missing, extra = sorted(expected - actual_ids), sorted(actual_ids - expected)
    duplicates_plan = {key: count for key, count in plan_counts.items() if count != 1}
    duplicates_result = {key: count for key, count in result_counts.items() if count != 1}
    complete = not runners and not missing and not partial_lines
    if not complete and not allow_incomplete:
        raise SystemExit(f'v3 is incomplete ({len(actual_ids)}/{len(expected)} IDs); no audit files written.')
    if len(plan) != expected_plan_count:
        errors.append({'kind': 'plan_count', 'expected': expected_plan_count, 'actual': len(plan)})
    for kind, value in [('duplicate_plan_ids', duplicates_plan), ('duplicate_result_ids', duplicates_result), ('extra_result_ids', extra)]:
        if value:
            errors.append({'kind': kind, 'value': value})
    if missing:
        warnings.append({'kind': 'missing_results', 'count': len(missing)})
    canonical = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if canonical != execution.get('plan_sha256'):
        errors.append({'kind': 'frozen_plan_sha_mismatch'})
    if hashlib.sha256((config / 'datasets.json').read_bytes()).hexdigest() != execution.get('datasets_sha256'):
        errors.append({'kind': 'frozen_datasets_sha_mismatch'})
    if read_json(config / 'limits.json') != execution.get('limits'):
        errors.append({'kind': 'frozen_limits_mismatch'})
    source_hashes = read_json(config / 'source-hashes.json')
    if source_hashes.get('experiments/run_v3.py') != execution.get('runner_sha256'):
        errors.append({'kind': 'frozen_runner_sha_mismatch'})
    # Keep duplicates visible above while counting one physical run per ID.
    unique = {row['run_id']: row for row in results}
    plans = {row['run_id']: row for row in plan}
    interruptions = interrupted_attempts(base, plans, unique, errors) if require_reboot_receipt or (base / REBOOT_RECEIPT).exists() else []
    warmups = []
    for path in sorted((base / 'warmups').glob('*/record.json')):
        try:
            warmups.append(read_json(path))
        except (OSError, json.JSONDecodeError) as error:
            errors.append({'kind': 'warmup_record_unreadable', 'path': str(path), 'error': str(error)})
    warmup_by_id = {row['run_id']: row for row in warmups}
    physical_rows, failed, rss_above, unexecuted, reused = [], [], [], [], []
    limit = execution['limits']['max_rss_bytes']
    for scope, group in [('formal', list(unique.values())), ('warmup', warmups)]:
        for row in group:
            run_id = row['run_id']
            def problem(kind, **detail):
                errors.append(dict(kind=kind, scope=scope, run_id=run_id, **detail))
            if row.get('status') not in STATUSES:
                problem('unknown_status', value=row.get('status'))
            if scope == 'formal' and run_id in plans:
                for key in ('study', 'graph', 'method', 'seed', 'iterations', 'epsilon', 'pivots', 'tile_size', 'workgroup_size'):
                    if row.get(key) != plans[run_id].get(key):
                        problem('plan_parameter_mismatch', field=key)
            if row.get('reused_from'):
                reused.append(compact(row, scope))
                source = unique.get(row['reused_from'])
                if not source or row['reused_from'] != plans.get(run_id, {}).get('reuse'):
                    problem('reuse_source_missing_or_unplanned')
                elif any(row.get(key) != source.get(key) for key in
                         ('graph', 'method', 'seed', 'pivots', 'tile_size', 'status', 'input_sha256',
                          'binary_sha256', 'stress_value', 'iteration_time_ms', 'final_positions_path')):
                    problem('reuse_result_mismatch')
                continue
            info = catalog.get(row.get('graph', row.get('dataset')), {})
            if not physical(row):
                if row.get('status') == 'success':
                    problem('success_without_physical_execution_metadata')
                entry = compact(row, scope)
                entry['input_exclusion'] = exclusion(info)
                if row.get('status') == 'warmup_failed':
                    warmup = warmup_by_id.get(row.get('warmup_run_id'))
                    issues = blocking_warmup_issues(row, warmup)
                    for issue in issues:
                        problem(issue['kind'], **{key:value for key,value in issue.items() if key != 'kind'})
                    if warmup is not None:
                        entry['warmup_physical_execution'] = physical(warmup)
                        entry['blocking_warmup_status'] = warmup.get('status')
                        entry['block_kind'] = ('invalid_blocking_warmup_reference' if issues else
                                               'physical_warmup_failure' if physical(warmup) else exclusion(info))
                unexecuted.append(entry)
                continue
            item = compact(row, scope)
            physical_rows.append(item)
            if row.get('input_sha256') != info.get('sha256'):
                problem('input_record_sha_mismatch')
            if row.get('binary_sha256') != execution['binary_sha256'].get(row.get('method')):
                problem('binary_record_sha_mismatch')
            if row.get('archive_status') != 'success':
                problem('archive_not_success', archive_status=row.get('archive_status'))
            manifest_value = row.get('archive_manifest_path')
            if not manifest_value or not resolve(manifest_value, base).is_file():
                problem('archive_manifest_missing')
            else:
                try:
                    manifest = read_json(resolve(manifest_value, base))
                    if manifest.get('errors'):
                        problem('archive_manifest_errors', errors=manifest['errors'])
                    for entry in manifest.get('files', []):
                        if (entry.get('original_sha256') != entry.get('decompressed_sha256')
                                or not entry.get('gzip_crc_validated')):
                            problem('archive_verification_metadata_invalid', path=entry.get('compressed_path'))
                        if not entry.get('compressed_path') or not resolve(entry['compressed_path'], base).is_file():
                            problem('archived_file_missing', path=entry.get('compressed_path'))
                except (OSError, json.JSONDecodeError) as error:
                    problem('archive_manifest_unreadable', error=str(error))
            if row.get('status') == 'success':
                if row.get('git_commit') != execution.get('git_commit'):
                    problem('native_commit_mismatch', native_commit=row.get('git_commit'))
                if row.get('git_dirty'):
                    warnings.append({'kind': 'native_git_dirty', 'scope': scope, 'run_id': run_id})
                if 'atomic' in row.get('method', ''):
                    expected_updates = row.get('constraints') * row.get('iterations') if (
                        isinstance(row.get('constraints'), int) and isinstance(row.get('iterations'), int)) else None
                    if (row.get('attempted_updates') is None or row.get('attempted_updates') != row.get('completed_updates')
                            or row.get('retry_failures') != 0 or expected_updates != row.get('completed_updates')):
                        problem('atomic_incomplete_updates', attempted=row.get('attempted_updates'),
                                completed=row.get('completed_updates'), expected=expected_updates,
                                final_unfinished=row.get('retry_failures'))
                if finite(row.get('peak_rss_bytes')) and row['peak_rss_bytes'] > limit:
                    rss_above.append(dict(item, peak_rss_gib=row['peak_rss_bytes'] / 1024 ** 3,
                                          threshold_rss_bytes=limit, interpretation='success; post-exit peak exceeded polling threshold'))
            else:
                epoch = last_epoch(row, base, warnings)
                failed.append(dict(item, last_epoch=epoch, stage=stage(epoch),
                                   error=str(row.get('error_message') or row.get('error') or '')[:2000]))
    status = Counter(row.get('status') for row in unique.values())
    warmup_actual = [row for row in physical_rows if row['scope'] == 'warmup']
    formal_actual = [row for row in physical_rows if row['scope'] == 'formal']
    formal_reused = sum(row['scope'] == 'formal' for row in reused)
    formal_unexecuted = sum(row['scope'] == 'formal' for row in unexecuted)
    accounted_conditions = len(formal_actual) + formal_reused + formal_unexecuted
    if accounted_conditions != len(unique):
        errors.append({'kind': 'confirmed_condition_accounting_mismatch',
                       'accounted_conditions': accounted_conditions, 'unique_conditions': len(unique)})
    report = {'audit_scope': 'final' if complete else 'provisional', 'complete_plan': complete,
              'status': 'errors' if errors else 'success' if complete else 'incomplete',
              'active_runners': runners, 'expected_plan_conditions': len(plan),
              'unique_result_conditions': len(unique), 'status_counts': dict(status),
              'missing_run_ids': missing, 'extra_run_ids': extra,
              'duplicate_plan_ids': duplicates_plan, 'duplicate_result_ids': duplicates_result,
              'physical_formal_executions': len(formal_actual), 'reused_conditions': formal_reused,
              'unexecuted_formal_conditions': formal_unexecuted,
              'additional_interrupted_formal_attempts': len(interruptions),
              'total_formal_native_launch_attempts': len(formal_actual) + len(interruptions),
              'interrupted_formal_attempts': interruptions,
              'condition_accounting': {
                  'confirmed_conditions': accounted_conditions, 'missing_conditions': len(missing),
                  'planned_condition_equation': 'confirmed physical executions + reused conditions + unexecuted conditions = planned conditions (when complete)',
                  'launch_attempt_equation': 'confirmed physical formal executions + additional interrupted formal attempts = total formal native launch attempts',
                  'physical_formal_executions_definition': 'native launches with a finalized results.jsonl record; excludes interruption receipts',
                  'interrupted_attempt_definition': 'formal native launch interrupted by host reboot, preserved in its receipt; kept separate from final condition results',
                  'excluded_attempt_scopes': ['warmups', 'pilots', 'prebuffer-aborted']},
              'physical_warmup_executions': len(warmup_actual),
              'physical_formal_failures': sum(row['scope'] == 'formal' for row in failed),
              'physical_warmup_failures': sum(row['scope'] == 'warmup' for row in failed),
              'blocking_warmup_kinds': dict(Counter(row.get('block_kind') for row in unexecuted if row.get('status') == 'warmup_failed')),
              'successful_rss_above_threshold': len(rss_above), 'rss_threshold_bytes': limit,
              'native_git_commit_expected': execution.get('git_commit'), 'errors': errors,
              'warning_counts': dict(Counter(row['kind'] for row in warnings)), 'warnings': warnings,
              'limitations': ['Record/config hashes checked; binaries and large inputs not rehashed.',
                             'Archive verification metadata and file existence checked; gzip payload not reread.',
                             'Reboot receipt evidence checked without adding an extra planned condition or final failure.',
                             'Last epoch locates an observed stage, not the exact interrupted operation.',
                             'retry_failures is final unfinished count; zero does not mean no lock contention.']}
    return report, physical_rows, failed, rss_above, unexecuted, reused


def tagged(rows, base, origin):
    return [dict(row,origin_base=str(base),origin_scope=origin) for row in rows]


def combine_recovery_audits(base, core, recovery, cohort, datasets, effective_rows):
    """Keep raw audits intact while accounting once for every effective condition."""
    extra_base=Path(cohort['recovery_base'])
    core_report,*core_groups=core; extra_report,*extra_groups=recovery
    replaced={row['run_id'] for row in cohort['superseded_records']}
    groups=[]
    for index,(old,new) in enumerate(zip(core_groups,extra_groups)):
        # Group 3 is unexecuted; the original 90 unavailable formal records are
        # retained in provenance, but are no longer effective missing conditions.
        kept=[row for row in old if not (index==3 and row.get('scope')=='formal' and row.get('run_id') in replaced)]
        groups.append(tagged(kept,base,'core')+tagged(new,extra_base,'snap_origin_recovery'))
    physical_rows,failed,rss_above,unexecuted,reused=groups
    report=dict(core_report)
    interruptions=tagged(core_report['interrupted_formal_attempts'],base,'core')+tagged(extra_report['interrupted_formal_attempts'],extra_base,'snap_origin_recovery')
    errors=tagged(core_report['errors'],base,'core')+tagged(extra_report['errors'],extra_base,'snap_origin_recovery')
    warnings=tagged(core_report['warnings'],base,'core')+tagged(extra_report['warnings'],extra_base,'snap_origin_recovery')
    formal_actual=sum(row['scope']=='formal' for row in physical_rows)
    formal_reused=sum(row['scope']=='formal' for row in reused)
    formal_unexecuted=sum(row['scope']=='formal' for row in unexecuted)
    accounted=formal_actual+formal_reused+formal_unexecuted
    if accounted!=len(effective_rows): errors.append(dict(kind='effective_cohort_condition_accounting_mismatch',accounted=accounted,effective=len(effective_rows)))
    report.update(audit_scope='effective_cohort',complete_plan=core_report['complete_plan'] and extra_report['complete_plan'],
                  status='errors' if errors else 'success',
                  unique_result_conditions=len(effective_rows),status_counts=dict(Counter(row.get('status') for row in effective_rows)),
                  raw_record_count=cohort['raw_record_count'],core_raw_records=cohort['core_raw_records'],
                  recovery_raw_records=cohort['recovery_raw_records'],superseded_unexecuted_conditions=len(replaced),
                  physical_formal_executions=formal_actual,reused_conditions=formal_reused,unexecuted_formal_conditions=formal_unexecuted,
                  additional_interrupted_formal_attempts=len(interruptions),total_formal_native_launch_attempts=formal_actual+len(interruptions),
                  interrupted_formal_attempts=interruptions,
                  physical_warmup_executions=sum(row['scope']=='warmup' for row in physical_rows),
                  physical_formal_failures=sum(row['scope']=='formal' for row in failed),
                  physical_warmup_failures=sum(row['scope']=='warmup' for row in failed),
                  blocking_warmup_kinds=dict(Counter(row.get('block_kind') for row in unexecuted if row.get('status')=='warmup_failed')),
                  successful_rss_above_threshold=len(rss_above),errors=errors,warnings=warnings,
                  warning_counts=dict(Counter(row['kind'] for row in warnings)),
                  effective_dataset_exclusion_counts=dict(Counter(exclusion(row) for row in datasets)),
                  native_git_commits_expected_by_origin=cohort['execution_commits_by_base'],
                  condition_accounting=dict(core_report['condition_accounting'],confirmed_conditions=accounted),
                  source_audits={'core':'audit-v3-core.json','snap_origin_recovery':'audit-v3-recovery.json'})
    report.pop('native_git_commit_expected',None)  # Each source audit retains its own exact expected commit.
    return report,*groups


def write_source_audit(tables, stem, result):
    report,physical_rows,failed,rss_above,unexecuted,reused=result
    (tables/(stem+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for suffix,rows in [('physical-executions',physical_rows),('failures',failed),('success-rss-exceedances',rss_above),
                        ('unexecuted',unexecuted),('reused',reused),('status',status_statistics(physical_rows))]:
        write_csv(tables/(stem+'-'+suffix+'.csv'),rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=BASE)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    base = args.results_dir.resolve()
    standalone_recovery=base.name==RECOVERY_DIRECTORY
    core_result = audit(base, args.allow_incomplete,expected_plan_count=90 if standalone_recovery else EXPECTED_PLAN_COUNT,
                        require_reboot_receipt=not standalone_recovery)
    tables = base / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    datasets,effective_rows,cohort=load_effective_cohort(base)
    write_source_audit(tables,'audit-v3-core',core_result)
    if cohort['status']=='adopted':
        recovery_result=audit(Path(cohort['recovery_base']),args.allow_incomplete,expected_plan_count=90,require_reboot_receipt=False)
        write_source_audit(tables,'audit-v3-recovery',recovery_result)
        report,physical_rows,failed,rss_above,unexecuted,reused=combine_recovery_audits(base,core_result,recovery_result,cohort,datasets,effective_rows)
    else:
        report,physical_rows,failed,rss_above,unexecuted,reused=core_result
        if cohort['status']=='rejected':
            report=dict(report,errors=report['errors']+cohort['errors'],status='errors')
    cohort_summary={key:value for key,value in cohort.items() if key not in ('record_provenance','superseded_records')}
    report=dict(report,cohort=cohort_summary)
    write_csv(tables/'cohort-source-records.csv',cohort['record_provenance'])
    write_csv(tables/'cohort-superseded.csv',cohort['superseded_records'])
    (tables/'cohort.json').write_text(json.dumps(cohort_summary,ensure_ascii=False,indent=2)+'\n')
    (tables/'effective-datasets.json').write_text(json.dumps(datasets,ensure_ascii=False,indent=2)+'\n')
    summaries = status_statistics(physical_rows)
    counts = [
        {'item': 'expected_plan_conditions', 'count': report['expected_plan_conditions'], 'unit': 'planned conditions'},
        {'item': 'unique_result_conditions', 'count': report['unique_result_conditions'], 'unit': 'finalized conditions'},
        {'item': 'confirmed_physical_formal_executions', 'count': report['physical_formal_executions'], 'unit': 'finalized native launches'},
        {'item': 'reused_conditions', 'count': report['reused_conditions'], 'unit': 'planned conditions'},
        {'item': 'unexecuted_formal_conditions', 'count': report['unexecuted_formal_conditions'], 'unit': 'planned conditions'},
        {'item': 'missing_conditions', 'count': len(report['missing_run_ids']), 'unit': 'planned conditions'},
        {'item': 'additional_interrupted_formal_attempts', 'count': report['additional_interrupted_formal_attempts'], 'unit': 'extra native launch attempts'},
        {'item': 'total_formal_native_launch_attempts', 'count': report['total_formal_native_launch_attempts'], 'unit': 'native launch attempts excluding warmups/pilots/prebuffer'},
        {'item': 'preserved_raw_result_records', 'count': cohort['raw_record_count'], 'unit': 'core plus recovery raw records; not unique conditions'},
        {'item': 'superseded_unexecuted_conditions', 'count': cohort['superseded_condition_count'], 'unit': 'original unavailable conditions replaced only in effective analysis'},
    ]
    for filename, rows in [('audit-v3-status.csv', summaries), ('audit-v3-failures.csv', failed),
                           ('audit-v3-success-rss-exceedances.csv', rss_above),
                           ('audit-v3-unexecuted.csv', unexecuted), ('audit-v3-reused.csv', reused),
                           ('audit-v3-physical-executions.csv', physical_rows),
                           ('audit-v3-interrupted-attempts.csv', report['interrupted_formal_attempts']),
                           ('audit-v3-execution-counts.csv', counts)]:
        write_csv(tables / filename, rows)
    (tables / 'audit-v3.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    prose = ['# v3 最終記録監査' if report['complete_plan'] else '# v3 暫定記録監査', '',
             f"監査状態：{report['status']}。計画 {report['expected_plan_conditions']} 条件、確定ID {report['unique_result_conditions']} 件。", '',
             f"確定recordを持つ正式起動 {report['physical_formal_executions']} 件、再利用 {report['reused_conditions']} 条件、未起動 {report['unexecuted_formal_conditions']} 条件。",
             '計画条件数 = 確定recordを持つ正式起動数 + 再利用条件数 + 未起動条件数（全条件の確定時）。暫定監査では未確定の条件数も別途残る。',
             f"追加の中断起動attempt {report['additional_interrupted_formal_attempts']} 件を加え、正式nativeの総起動attemptは {report['total_formal_native_launch_attempts']} 件。",
             '総起動attempt = 確定recordを持つ正式起動数 + 追加の中断attempt数。中断後に同じrun_idの条件が完了しても、途中attemptは別枠で残す。pilot・prebuffer-aborted・warmupはこの総数へ加算しない。',
             f"正式の実失敗 {report['physical_formal_failures']} 件。別枠warmupは実起動 {report['physical_warmup_executions']} 件、実失敗 {report['physical_warmup_failures']} 件。", '',
             '成否：`' + json.dumps(report['status_counts'], ensure_ascii=False) + '`。',
             'warmupによる未起動の理由：`' + json.dumps(report['blocking_warmup_kinds'], ensure_ascii=False) + '`。', '',
             f"記録整合性エラー {len(report['errors'])} 件。警告分類：`{json.dumps(report['warning_counts'], ensure_ascii=False)}`。",
             'git_dirtyはnative測定時の作業場所の状態を示す。固定snapshot/binary/recordの照合とは分けて記録する。', '',
             '## ホスト再起動による追加中断attempt', '',
             '中断receiptは計画条件や確定失敗の追加行として扱わない。receiptのrun_idがplan内にあること、退避artifactと記載ファイルの存在・サイズ、command.jsonのrun_idを照合する。', '',
             '| run_id | 確定条件の結果 | 中断証拠 | 退避artifact | 証拠整合 |', '|---|---|---|---|---|']
    if cohort['status']=='adopted':
        prose[2:2]=[f"公式SNAP原配布復旧の90条件を補完した有効cohortの監査。core原本{cohort['core_raw_records']}件＋recovery原本{cohort['recovery_raw_records']}件＝{cohort['raw_record_count']}記録を保持し、有効IDは{cohort['effective_conditions']}件。元90件は未起動unavailableで、supersededとして保持する。実起動済み条件の上書きはない。", '',
                    'coreとrecoveryをそれぞれ独立にplan/config/native commitへ照合した。67 native/sharedファイルの集合・SHA、runner 1ファイルのSHA、全binary SHA、limitsの一致を確認してから合算した。commitはbase別に区別する。正式起動・warmup・未起動・正式失敗・RSS超過は両baseを加算し、元の未起動90件だけは有効欠測から差し引いた。', '',
                    '原配布は同名の公式SNAP sourceと公開ノード数・unique edge数を検証したもの。SuiteSparse MTXとのbyte/edgewise同一性は未検証。power197k未取得・G_n_pin_pout規模除外は変更していない。', '',
                    '- [core単独監査](../tables/audit-v3-core.json) · [recovery単独監査](../tables/audit-v3-recovery.json)',
                    '- [採用検証provenance](../tables/cohort.json) · [全原記録の有効/置換状態](../tables/cohort-source-records.csv) · [元の未起動90条件](../tables/cohort-superseded.csv)', '']
    elif cohort['status'] in ('incomplete','rejected'):
        prose[2:2]=[f"SNAP追加測定の採用状態は{cohort['status']}。部分測定は合算せず、上記集計はcore単独cohort。理由・検証エラーは[cohort provenance](../tables/cohort.json)に保存した。", '']
    for row in report['interrupted_formal_attempts']:
        prose.append(f"| {row['run_id']} | {row['confirmed_condition_status'] or '未確定'} | [receipt](../{row['receipt_path']}) | [{row['preserved_artifact_directory']}](../{row['preserved_artifact_directory']}) | {row['evidence_valid']} |")
    prose += ['', '## 実失敗の観測段階', '', '| Scope | Stage | Count |', '|---|---|---:|']
    for (scope, phase), count in sorted(Counter((row['scope'], row['stage']) for row in failed).items()):
        prose.append(f'| {scope} | {phase} | {count} |')
    prose += ['', 'traceなしは前処理・buffer準備中などの可能性があり、詳細な停止命令は特定しない。反復0は最初の反復完了前、1〜14は反復途中、15は全15反復完了後の段階。反復15でもstress評価や保存の完了を意味しない。', '',
              '## 成功した実行のRSS閾値超過', '',
              f"成功した正式実行とwarmupのうち、終了後の最大RSSが閾値{report['rss_threshold_bytes'] / 1024 ** 3:g} GiBを超えるものは {len(rss_above)} 件。再利用条件は重複加算していない。",
              '監視は間隔観測であり、実行中の瞬間最大RSSが閾値以下だったことを保証しない。成功statusを閾値超過だけで書き換えない。RSSを専用VRAM量とは解釈しない。', '',
              '## 証拠と監査範囲', '',
              '- [監査JSON](../tables/audit-v3.json)', '- [status別時間・RSS](../tables/audit-v3-status.csv)',
              '- [起動数と条件数](../tables/audit-v3-execution-counts.csv)',
              '- [追加中断attemptの証拠](../tables/audit-v3-interrupted-attempts.csv)',
              '- [実失敗・trace段階](../tables/audit-v3-failures.csv)',
              '- [成功時RSS閾値超過](../tables/audit-v3-success-rss-exceedances.csv)',
              '- [未起動理由](../tables/audit-v3-unexecuted.csv)', '- [再利用条件](../tables/audit-v3-reused.csv)', '',
              'planの全ID・重複・余分・設定、configの固定hash、各実起動recordのinput/binary hash、成功native commit、Atomic更新完了、archive状態を照合する。座標の全再計算・gzip再読込・binary再hashは行わない。archiveは実行時の検証metadataと保存ファイルの存在を確認する。', '',
              'この文書はresource-incidents.mdの履歴を上書きせず、最終監査として別に保存する。']
    if report['errors']:
        prose += ['', '## 整合性エラー', ''] + ['- `' + json.dumps(error, ensure_ascii=False) + '`' for error in report['errors'][:30]]
    destination = base / 'report'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'final-audit.md').write_text('\n'.join(prose) + '\n')
    print(json.dumps({key: report[key] for key in ('audit_scope', 'status', 'unique_result_conditions',
          'physical_formal_executions', 'additional_interrupted_formal_attempts', 'total_formal_native_launch_attempts',
          'physical_warmup_failures', 'successful_rss_above_threshold')}, ensure_ascii=False))
    return 2 if report['errors'] else 0 if report['complete_plan'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
