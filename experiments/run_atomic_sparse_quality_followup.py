"""Recheck the two largest observed stress increases against the v3 Atomic control."""
import hashlib
import json
import pathlib
import subprocess
import os
import signal

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT/'experiment_results/validation/atomic-sparse-20261106'
BINARY = ROOT/'tools/diagnostics/sparse-audit/target/release/speed-probe'

def main():
    statuses = [json.loads(s) for s in (BASE/'paired-status.jsonl').read_text().splitlines()]
    assert len(statuses) == 48, 'Run after the main sequential measurements finish'
    assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == json.loads((BASE/'paired-config.json').read_text())['binary_sha256']
    output = BASE/'quality-followup'
    output.mkdir(exist_ok=True)
    for graph in ['kron_g500-logn18', 'com-DBLP']:
        for seed in [0, 1]:
            prefix = output/f'{graph}-s{seed}'
            if prefix.with_suffix('.status.json').exists():
                continue
            cmd = [str(BINARY), str(ROOT/f'data/v4_lcc/{graph}.mtx'), str(seed),
                   'cpu,v3-each,production' if seed == 0 else 'production,v3-each,cpu', '256']
            with prefix.with_suffix('.jsonl').open('w') as out, prefix.with_suffix('.log').open('w') as err:
                child = subprocess.Popen(['/usr/bin/time', '-l', *cmd], cwd=ROOT, stdout=out,
                                         stderr=err, start_new_session=True)
                try:
                    child.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    raise
            rows = [json.loads(s) for s in prefix.with_suffix('.jsonl').read_text().splitlines()]
            assert child.returncode == 0 and len(rows) == 3
            assert {r['mode'] for r in rows} == {'cpu', 'v3-each', 'production'}
            assert all(r['stress_samples'] == 256 for r in rows)
            prefix.with_suffix('.status.json').write_text(json.dumps(dict(command=cmd, status='success'), indent=2))
            data = {r['mode']:r for r in rows}
            print(json.dumps(dict(graph=graph, seed=seed,
                new_speedup=data['cpu']['iteration_ms']/data['production']['iteration_ms'],
                old_stress_ratio=data['v3-each']['stress']/data['cpu']['stress'],
                new_stress_ratio=data['production']['stress']/data['cpu']['stress'])), flush=True)

if __name__ == '__main__': main()
