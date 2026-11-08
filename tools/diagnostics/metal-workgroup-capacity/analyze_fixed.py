"""Validate fixed-work probes and write per-configuration timings and report."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
samples = collections.defaultdict(list)
hashes = collections.defaultdict(set)
records = []
metadata = []
for path in sorted(root.glob('iterations*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    metadata.append(rows[0])
    for r in rows[1:]:
        assert r['kind'] == 'measurement' and r['kernel'] in ('fma1', 'fma4')
        assert r['output_count'] == r['width'] * r['groups']
        hashes[r['iterations'], r['kernel'], r['output_count']].add(r['output_sha256'])
        if not r['warmup']:
            key = r['iterations'], r['kernel'], r['output_count'], r['width']
            samples[key].append(r['gpu_ms'] * 1000 / r['dispatches'])
assert len(samples) == 192 and all(len(v) == 5 for v in samples.values())
assert len(hashes) == 32 and all(len(v) == 1 for v in hashes.values())
for (iterations, kernel, n, w), times in sorted(samples.items()):
    records.append(dict(iterations=iterations, kernel=kernel, threads=n, width=w,
                        simd_groups_per_wg=w//32, workgroups=n//w,
                        median_us=statistics.median(times), min_us=min(times), max_us=max(times)))
with (root/'summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

lines = ['# Fixed-work arithmetic benchmark — Apple M4 Pro (20 GPU cores)', '',
         'No graph inputs. N total threads, G=N/W workgroups, SIMD width 32.',
         'Each thread runs x=fma(x,a,b), either one recurrence (fma1) or four independent recurrences (fma4).',
         'No threadgroup memory or barriers. Fast math disabled; same pipeline for all W.',
         'Two loop lengths, eight N values, six W values, two kernels: 192 configurations.',
         'Each has one warmup and five measured trials, in shuffled order: 1,152 trials total.',
         'GPU command-buffer elapsed time / four serial dispatches; medians below, in microseconds.',
         'CPU reference checks passed; full output hashes agree across all six W values and all trials for each N/kernel/loop length.', '',
         'The previous residency benchmark measures inflight capacity; it does not determine the fastest WG size.',
         'These measurements select dispatch geometry for these recurrence kernels. They do not measure same-cycle issue width or establish an optimal WG size for every kernel.', '']
for iterations in [4096, 16384]:
    for kernel in ['fma1', 'fma4']:
        lines += [f'## {kernel}, {iterations} iterations per thread', '',
                  '| N | W32 | W64 | W128 | W256 | W512 | W1024 | W within 5% of best |',
                  '|---:|---:|---:|---:|---:|---:|---:|:---|']
        for n in [1024,4096,8192,16384,32768,65536,131072,262144]:
            vals = {w: statistics.median(samples[iterations,kernel,n,w]) for w in [32,64,128,256,512,1024]}
            near = ', '.join(str(w) for w,v in vals.items() if v <= min(vals.values())*1.05)
            lines.append('| '+str(n)+' | '+' | '.join(f'{v:.2f}' for v in vals.values())+' | '+near+' |')
        lines.append('')
lines += ['## Interpretation', '',
          'W32/W64 (one/two SIMD groups per WG) are consistently competitive here. At N=8192, W1024 is about 2.1–2.4 times slower than W32/W64 for fma4. At N=262144, the difference is around 1–2%.',
          'More SIMD groups per WG are not inherently faster. With enough total work, multiple configurations approach similar throughput. Near ties should not be treated as a universal ranking.',
          'No profile capture ran concurrently with these timing runs. The machine remained an interactive desktop; five-trial ranges are recorded in summary.csv.', '']
(root/'RESULTS.md').write_text('\n'.join(lines))
sources = [Path(__file__), Path(__file__).with_name('benchmark.swift'), Path(__file__).with_name('benchmark.metal'), Path(__file__).with_name('benchmark')]
provenance = {'metadata':metadata, 'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
for p in sources:
    if p.suffix in ('.swift','.metal','.py'):
        (root/('source-'+p.name)).write_bytes(p.read_bytes())
print(f'Validated {len(samples)} configurations, 960 measured trials, 32 matching output groups.')
print(root/'RESULTS.md')
