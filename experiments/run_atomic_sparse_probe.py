"""Sequential, resumable paired CPU/Atomic audit of the 38 graphs in the supplied figure."""
import hashlib
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiment_results/validation/atomic-sparse-20261106'
BINARY = ROOT / 'tools/diagnostics/sparse-audit/target/release/speed-probe'
GRAPHS = '144 598a Linux_call_graph amazon0302 auto bmw7st_1 cage12 cage13 caidaRouterLevel cnr-2000 co2010 coAuthorsCiteseer coPapersCiteseer com-Amazon com-DBLP delaunay_n17 delaunay_n18 delaunay_n19 email-EuAll fe_ocean ford2 hood kron_g500-logn17 kron_g500-logn18 language loc-Gowalla luxembourg_osm m14b mt2010 ny2010 preferentialAttachment rgg_n_2_17_s0 rgg_n_2_18_s0 rgg_n_2_19_s0 smallworld usroads web-NotreDame web-Stanford'.split()
EXTRA = ['luxembourg_osm', 'smallworld', 'kron_g500-logn17', 'web-NotreDame', 'coAuthorsCiteseer']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    BASE.mkdir(parents=True, exist_ok=True)
    jobs = [(g, 0, 64) for g in GRAPHS] + [(g, s, 256) for s in (1, 2) for g in EXTRA]
    config = dict(graphs=GRAPHS, extra_seeds=EXTRA, iterations=15, pivots=200, epsilon=0.1,
                  order='seed even: CPU then GPU; seed odd: GPU then CPU',
                  stress='same sampled full stress sources, evaluation seed 0',
                  timeout_seconds=240, rss_limit_bytes=12*1024**3,
                  binary_sha256=sha(BINARY), runner_sha256=sha(pathlib.Path(__file__)))
    smoke = [json.loads(line) for line in (BASE/'runner-smoke.jsonl').read_text().splitlines()]
    assert {r['mode'] for r in smoke} == {'cpu', 'production'}
    assert all(r['stress_samples'] == 64 for r in smoke)
    config_path = BASE/'paired-config.json' 
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    else:
        config_path.write_text(json.dumps(config, indent=2))
    status_file = BASE/'paired-status.jsonl'
    done = {tuple(r['key']) for r in map(json.loads, status_file.read_text().splitlines())} if status_file.exists() else set()
    raw = BASE/'paired-raw'
    raw.mkdir(exist_ok=True)
    for graph, seed, samples in jobs:
        key = (graph, seed, samples)
        if key in done:
            continue
        assert sha(BINARY) == config['binary_sha256'], 'Binary changed during measurements'
        path = ROOT/f'data/v4_lcc/{graph}.mtx'
        prefix = raw/f'{graph}-s{seed}-q{samples}'
        modes = 'production,cpu' if seed % 2 else 'cpu,production'
        command = [str(BINARY), str(path), str(seed), modes, str(samples)]
        start = time.monotonic()
        peak = 0
        reason = None
        with prefix.with_suffix('.jsonl').open('w') as out, prefix.with_suffix('.log').open('w') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
            while child.poll() is None:
                result = subprocess.run(['ps', '-o', 'rss=', '-p', str(child.pid)], capture_output=True, text=True)
                peak = max(peak, int(result.stdout.strip() or 0)*1024)
                if peak > config['rss_limit_bytes']:
                    reason = 'memory_limit'
                elif time.monotonic()-start > config['timeout_seconds']:
                    reason = 'timeout'
                if reason:
                    child.kill()
                    child.wait()
                    break
                time.sleep(0.5)
        rows = [json.loads(line) for line in prefix.with_suffix('.jsonl').read_text().splitlines()]
        valid = (len(rows) == 2 and {r['mode'] for r in rows} == {'cpu', 'production'}
                 and all(r.get('stress_samples') == samples for r in rows))
        status = reason or ('success' if child.returncode == 0 and valid else 'error')
        record = dict(key=key, status=status, returncode=child.returncode, peak_rss_bytes=peak,
                      wall_seconds=time.monotonic()-start, command=command, input_sha256=sha(path),
                      output=str(prefix.with_suffix('.jsonl').relative_to(ROOT)))
        with status_file.open('a') as f:
            f.write(json.dumps(record)+'\n')
        done.add(key)
        print(json.dumps(dict(completed=len(done), planned=len(jobs), **record)), flush=True)

if __name__ == '__main__':
    main()
