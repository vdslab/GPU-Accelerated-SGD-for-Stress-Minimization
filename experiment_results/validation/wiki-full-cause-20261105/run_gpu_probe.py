import hashlib
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[3]
out = pathlib.Path(__file__).resolve().parent / 'gpu-probe'
if len(sys.argv) > 1:
    out = pathlib.Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
out.mkdir(exist_ok=True)
binary = root / 'implementations/full/rr_gpu/target/release/examples/wiki_full_cause'
source = root / 'implementations/full/rr_gpu/examples/wiki_full_cause.rs'
input_path = pathlib.Path('/Users/kose/master/atomic-scheduling-investigation/data/v3_lcc/wiki-Vote.mtx')
cases = [('gpu',0,1024,1,'none'), ('gpu',0,1024,16,'none'),
         ('gpu',0,1024,4,'none'), ('gpu',0,1024,64,'none'),
         ('audit',0,1024,16,'none'), ('gpu-global',0,1024,1,'none')]
for seed in [1,2]:
    cases += [(mode,seed,1024,chunks,'none') for mode,chunks in
              [('cpu64',1),('gpu',1),('gpu',16),('rr64',1),('cpu32',1)]]

metadata = {'input':str(input_path), 'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'cases':cases, 'diagnostic_gpu_sha256':hashlib.sha256((source.parent/'wiki_full_cause/gpu.rs').read_bytes()).hexdigest(), 'diagnostic_shader_sha256':hashlib.sha256((source.parent/'wiki_full_cause/shader.wgsl').read_bytes()).hexdigest()}
pending = [case for case in cases if not (out/('-'.join(map(str,case))+'.json')).exists()]
if not pending:
    print('All cases already saved; preserving original metadata.')
    sys.exit(0)
metadata_path = out/'metadata.json'
if metadata_path.exists():
    if json.loads(metadata_path.read_text()) != metadata:
        raise SystemExit('Source or binary changed. Use a new output directory argument.')
else:
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
for case in cases:
    label = '-'.join(map(str,case))
    target = out/(label+'.json')
    if target.exists():
        continue
    command = [str(binary),str(input_path),*map(str,case)]
    print('RUN',label,flush=True)
    result = subprocess.run(command,text=True,capture_output=True,timeout=120)
    (out/(label+'.stderr')).write_text(result.stderr)
    result.check_returncode()
    record = json.loads(result.stdout)
    record['command']=command
    target.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
