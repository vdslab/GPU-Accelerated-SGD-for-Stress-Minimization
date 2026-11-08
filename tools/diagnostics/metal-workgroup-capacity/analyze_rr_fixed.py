"""Summarize fixed-tile RR-like shared-memory/synchronization probes."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines()]
samples = collections.defaultdict(list)
hashes = collections.defaultdict(set)
work = collections.defaultdict(set)
for r in rows[1:]:
    assert r['kind'] == 'measurement'
    assert r['kernel'] in ('rr_uniform','rr_matrix')
    assert r['output_count'] == r['groups'] * r['tile']
    assert r['shared_bytes_per_group'] == r['tile'] * 16
    assert r['rr_rounds'] == 4 and r['dispatches'] == 4
    hashes[r['tile']].add(r['equivalence_sha256'])
    work[r['tile'],r['output_count']].add(r['work_items'])
    if not r['warmup']:
        samples[r['kernel'],r['tile'],r['output_count'],r['width']].append(r['gpu_ms']/r['dispatches'])
assert len(rows) == 1081
assert len(samples) == 180 and all(len(v) == 5 for v in samples.values())
assert len(hashes) == 5 and all(len(v) == 1 for v in hashes.values())
assert len(work) == 15 and all(len(v) == 1 for v in work.values())
records = []
for (kernel,t,n,w), times in sorted(samples.items()):
    records.append(dict(kernel=kernel,tile=t,output_count=n,workgroups=n//t,width=w,
                        simd_groups_per_wg=w//32,shared_bytes=t*16,
                        median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times)))
with (root/'summary.csv').open('w') as f:
    writer = csv.DictWriter(f,fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
lines = ['# RR-like shared-memory benchmark — M4 Pro, 20 GPU cores','',
         '180 configurations, one warmup plus five timed trials each (1,080 trials).',
         'Each trial contains four serial dispatches; GPU command-buffer elapsed time / 4, median of five, in milliseconds.',
         'The same pipeline is reused at all W values. No Instruments profiling ran concurrently.', '',
         '## Controlled comparison','',
         'At fixed (T,N), G=N/T workgroups each handle two coordinate tiles of T vertices.',
         'N counts update slots / output float4 elements, not all vertices in a real graph.',
         'Each WG uses 16T bytes of threadgroup memory; four cycles of T cross-tile matchings perform 4T² pair updates per dispatch.',
         'Each matching reads and updates both endpoints in shared memory and ends with a threadgroup barrier (4T+1 barriers including initialization).',
         'Only W changes within a row: six sizes W32/64/128/256/512/1024, corresponding to 1/2/4/8/16/32 SIMD groups.',
         'W<T makes threads process multiple disjoint pairs before the barrier. W>T leaves excess threads with no pair update, but participating in barriers.',
         'WG count, shared bytes, update count, input data, and synchronization count are identical across all six W in a row.',
         'All groups and all W values produce byte-identical output for each T, including both distance modes and all trials.', '',
         '## Relation to the actual SGD shader','',
         'This native Metal diagnostic reproduces the cross-tile pair-update formula, shared coordinate reads/writes and per-matching synchronization.',
         'It is not the production WGSL/wgpu pipeline. It omits self-tile handling, partial tiles, outer scheduling and randomized graph-specific data accesses.',
         'rr_uniform uses a runtime constant d=1, removing distance-buffer reads.',
         'rr_matrix reads an all-ones distance matrix with stride 8192; groups reuse the same read-only region. This tests strided reads but not the cache footprint of independent graph tiles.',
         'Four cycles reuse shared coordinates without returning them to global memory between cycles, unlike separately dispatched production tile jobs.',
         'The fixed initial input and distance values are synthetic. The result selects candidates for real-graph verification, not a universal SIMD-per-WG optimum.',
         'Compare W within each row. T changes both arithmetic and barrier counts, so times across T do not represent equal total work.', '']
for kernel in ['rr_uniform','rr_matrix']:
    lines += [f'## {kernel}', '', '| T | G | W32 | W64 | W128 | W256 | W512 | W1024 | W within 5% of best |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|:---|']
    for t in [64,128,256,512,1024]:
        for n in [8192,32768,131072]:
            vals = {w:statistics.median(samples[kernel,t,n,w]) for w in [32,64,128,256,512,1024]}
            near = ', '.join(str(w) for w,v in vals.items() if v <= min(vals.values())*1.05)
            lines.append(f'| {t} | {n//t} | '+' | '.join(f'{v:.4f}' for v in vals.values())+' | '+near+' |')
    lines.append('')
(root/'RESULTS.md').write_text('\n'.join(lines))
sources = [Path(__file__),Path(__file__).with_name('benchmark.swift'),Path(__file__).with_name('benchmark.metal'),Path(__file__).with_name('benchmark')]
provenance = {'metadata':rows[0], 'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
for p in sources:
    if p.suffix in ('.swift','.metal','.py'):
        (root/('source-'+p.name)).write_bytes(p.read_bytes())
print('Validated 180 configurations, 900 timed trials; identical outputs and work across W.')
print(root/'RESULTS.md')
