"""Independent consistency checks for saved v5 large data and reports."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from run_v5 import ROOT, VAULT, now, save

BASE = ROOT / 'experiment_results/v5/large'
FIELDS = ['iteration_time_ms', 'total_compute_time_ms', 'stress_value', 'peak_rss_bytes']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def equal(actual, expected):
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    jobs = json.loads((BASE / 'config/jobs.json').read_text())
    records = [json.loads(p.read_text()) for p in (BASE / 'raw').glob('*/record.json')]
    assert len(records) == len(jobs) == 350
    assert {r['run_id'] for r in records} == {j['run_id'] for j in jobs}
    assert collections.Counter(r['status'] for r in records) == {'success': 130, 'preflight_unavailable': 220}
    csvrows = list(csv.DictReader((BASE / 'tables/runs.csv').open()))
    csvindex = {r['run_id']: r for r in csvrows}
    assert len(csvrows) == len(csvindex) == 350
    good = [r for r in records if r['status'] == 'success']
    checked_native = 0
    for record in records:
        saved = csvindex[record['run_id']]
        assert saved['status'] == record['status']
        if record['status'] != 'success':
            assert saved.get('stress_value', '') == ''
            continue
        native = json.loads((BASE / 'raw' / record['run_id'] / 'stdout.json').read_text())
        manifest = json.loads((BASE / 'raw' / record['run_id'] / 'artifact-archive.json').read_text())
        archived_paths = {entry['compressed_path'] for entry in manifest['files']}
        for field in ['initial_positions_path', 'final_positions_path', 'vertex_map_path']:
            if record.get(field):
                assert Path(record[field]).is_file() and record[field] in archived_paths, (record['run_id'], field)
        for key, value in native.items():
            if key not in ['initial_positions_path', 'final_positions_path', 'vertex_map_path']:
                equal(record[key], value)
        for key in FIELDS:
            equal(saved[key], record[key])
        checked_native += 1
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[(record['graph'], record['pivots'])].append(record)
    summary = json.loads((BASE / 'tables/summary.json').read_text())
    assert len(summary) == len(grouped) == 70
    summary_csv = {(r['graph'], int(r['pivots'])): r for r in csv.DictReader((BASE / 'tables/summary.csv').open())}
    for item in summary:
        group = grouped[(item['graph'], item['pivots'])]
        assert sorted(r['seed'] for r in group) == list(range(5))
        successful = [r for r in group if r['status'] == 'success']
        assert item['success'] == len(successful)
        for key in FIELDS:
            values = [r[key] for r in successful]
            for suffix, operation in [('median', statistics.median), ('min', min), ('max', max)]:
                field = key + '_' + suffix
                expected = operation(values) if values else None
                equal(item[field], expected)
                if expected is None:
                    assert summary_csv[(item['graph'], item['pivots'])][field] == ''
                else:
                    equal(summary_csv[(item['graph'], item['pivots'])][field], expected)
    index = {(r['graph'], r['pivots'], r['seed']): r for r in good}
    paired = list(csv.DictReader((BASE / 'tables/paired-pivot-ratios.csv').open()))
    assert len(paired) == 130
    ratios = collections.defaultdict(list)
    for row in paired:
        graph, pivots, seed = row['graph'], int(row['pivots']), int(row['seed'])
        largest = max(p for (g, p), group in grouped.items() if g == graph and all(r['status'] == 'success' for r in group))
        assert int(row['reference_pivots']) == largest
        expected = index[(graph, pivots, seed)]['stress_value'] / index[(graph, largest, seed)]['stress_value']
        equal(row['stress_ratio'], expected)
        ratios[(graph, pivots)].append(expected)
    for item in summary:
        values = ratios[(item['graph'], item['pivots'])]
        if values:
            equal(item['paired_stress_ratio_median'], statistics.median(values))
    certificate = json.loads((BASE / 'validation/report-completion.json').read_text())
    figures = certificate['figures']
    assert len(figures) == 19
    repo_text = (BASE / 'reports/large.md').read_text()
    vault_text = (VAULT / '4_大規模グラフ実験考察.md').read_text()
    for name in figures:
        png = BASE / 'figures' / (name + '.png')
        pdf = BASE / 'figures' / (name + '.pdf')
        assert png.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        with pdf.open('rb') as source:
            assert source.read(5) == b'%PDF-'
        assert sha(png) == sha(VAULT / 'v5_図表/large' / png.name)
        assert f'](../figures/{name}.png)' in repo_text
        assert f'![[v5_図表/large/{name}.png]]' in vault_text
    assert '![[v5_図表' not in repo_text
    sample = json.loads((BASE / 'tables/layout-sampling.json').read_text())
    assert len(sample) == 10
    for item in sample:
        assert item['seed'] == 0 and item['vertices_drawn'] <= 100000 and item['edges_drawn'] <= 50000
        assert item['pivots'] == max(r['pivots'] for r in good if r['graph'] == item['graph'])
    environment = json.loads((ROOT / 'experiment_results/v5/config/environment.json').read_text())
    from experiment_plan import METHODS
    for method, expected in environment['binary_sha256'].items():
        assert sha(ROOT / METHODS[method].binary) == expected
    save(BASE / 'validation/final-consistency-check.json', dict(
        at=now(), status='passed', planned_runs=350, native_records_preserved=checked_native,
        group_aggregates_verified=70, paired_ratios_verified=130, figures_and_vault_copies_verified=len(figures),
        layout_sampling_rules_verified=10, frozen_binaries_verified=len(environment['binary_sha256']),
        all_successful_record_artifact_pointers_verified=True,
        archive_validation='Reporter verifies manifest SHA agreement, CRC evidence and current file sizes; layout rendering decodes ten selected files fully. No redundant decode of all prior-verified archives.',
        visual_review_required=True))
    print('PASS: 350 records, 70 groups, 130 native outputs and paired ratios, 19 figure pairs/vault copies, frozen binaries')


if __name__ == '__main__':
    main()
