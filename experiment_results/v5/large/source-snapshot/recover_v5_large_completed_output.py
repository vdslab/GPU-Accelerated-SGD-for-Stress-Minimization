"""Recover a completed native measurement interrupted during post-run archival."""
import datetime
import json
import lzma
import re
import subprocess
from pathlib import Path

import psutil

from archive_v5_large import archive_text_artifacts, digest
from experiment_plan import Dataset, RunDefinition
from run_experiments import file_sha256, parse_and_validate_success
from run_v5 import now, save
from run_v5_large import BASE, BINARY, progress


def main():
    # This recovery is deliberately restricted to the inspected interrupted run.
    run_id = 'v5-large-com-LiveJournal-p100-s3-r0'
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        assert not any('run_v5_large.py' in arg or arg == str(BINARY) for arg in command), process.info
    folder = BASE / 'raw' / run_id
    assert not (folder / 'record.json').exists()
    job = next(j for j in json.loads((BASE / 'config/jobs.json').read_text()) if j['run_id'] == run_id)
    graph = next(g for g in json.loads((BASE / 'config/datasets.json').read_text()) if g['name'] == job['graph'])
    expected = json.loads((BASE / 'config/environment.json').read_text())['binary_sha256']
    assert file_sha256(BINARY) == expected
    command = json.loads((folder / 'command.json').read_text())
    assert command['environment'] == {'SGD_STRESS_SAMPLES': '64', 'SGD_GPU_TIMESTAMPS': '0'}
    stderr = (folder / 'stderr.log').read_text()
    assert 'terminated' not in stderr.lower() and 'signal' not in stderr.lower().replace('signals received', '')
    wall = float(re.search(r'(\d+\.\d+) real', stderr).group(1))
    rss = int(re.search(r'(\d+)\s+maximum resident set size', stderr).group(1))
    run = RunDefinition('v5-large', run_id, 'benchmark', 'rr_sparse_sgd', BINARY,
                        Dataset(graph['name'], Path(graph['path']), graph['sha256']), 3, 15, .1, 100, 0)
    # Child exit code was not persisted. Validate native fields/artifact paths
    # using the success schema; retain the missing exit code explicitly below.
    native = parse_and_validate_success(run, subprocess.CompletedProcess(command['argv'], 0, (folder / 'stdout.json').read_text(), stderr))
    assert native['rr_phase_chunks'] == 1 and native['stress_samples'] == 64 and native['stress_seed'] == 0
    assert native['nodes'] == graph['nodes'] and native['edges'] == graph['edges']
    entries = []
    redundant = []
    for source in folder.glob('*.txt'):
        target = source.with_suffix(source.suffix + '.xz')
        if not target.exists():
            continue
        expected_digest = digest(source)
        assert digest(target, lzma.open) == expected_digest
        compressed_sha, size = digest(target)
        entries.append(dict(original_path=str(source), compressed_path=str(target),
                            original_sha256=expected_digest[0], decompressed_sha256=expected_digest[0],
                            compressed_sha256=compressed_sha, original_bytes=expected_digest[1],
                            compressed_bytes=size, compression='xz', preset=0, xz_crc_validated=True))
        redundant.append(source)
    assert entries, 'Expected the completed final XZ file from interrupted archival'
    recovered_at = now()
    evidence = dict(at=recovered_at, run_id=run_id, cause='Unknown controller interruption during archival',
                    native_output_success=True, process_exit_code_available=False,
                    process_wall_source='/usr/bin/time -l real, precision 0.01 seconds',
                    original_native_fields_preserved=native, completed_xz_verified=entries,
                    no_layout_rerun=True, memory_after_unavailable=True)
    save(BASE / 'validation/completed-output-recovery.json', evidence)
    save(folder / 'artifact-archive.json', dict(format='xz', preset=0, measurement_excluded=True,
                                               files=entries, errors=[], recovery_partial_manifest=True))
    completed_at = datetime.datetime.fromtimestamp((folder / 'stderr.log').stat().st_mtime).astimezone().isoformat()
    row = dict(job, status='success', executed=True, measured=True, warmup=False,
               process_wall_seconds=wall, peak_rss_bytes=rss, returncode=None,
               completed_at=completed_at, memory_after=None, binary_sha256=expected,
               recovery_evidence=str(BASE / 'validation/completed-output-recovery.json'),
               process_wall_precision_seconds=.01, recovered_at=recovered_at)
    row.update(native)
    row.update(job)
    archive_text_artifacts(folder, row)
    for source in redundant:
        source.unlink()  # Manifest and complete decoded-byte verification persisted first.
    save(BASE / 'active-process.json', dict(pid=None, completed_at=now(), last_run=job, recovered=True))
    progress(recovered_completed_output=run_id)


if __name__ == '__main__':
    main()
