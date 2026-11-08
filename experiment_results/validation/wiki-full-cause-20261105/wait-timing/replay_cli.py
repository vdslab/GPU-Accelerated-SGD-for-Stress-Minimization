import json,subprocess,os,pathlib
root=pathlib.Path('/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization')
out=pathlib.Path(__file__).resolve().parent
original=json.loads(pathlib.Path('/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/raw/artifacts/v3-basic-wiki-Vote-rr_sgd-s0-pNone-t1024/command.json').read_text())
for label,binary,trace in [('old-trace',original['argv'][0],True),('current-cli',str(root/'implementations/full/rr_gpu/target/release/rr-gpu'),False),('old-no-trace',original['argv'][0],False)]:
 dest=out/label;dest.mkdir(exist_ok=True)
 args=original['argv'].copy();args[0]=binary
 args[args.index('--input')+1]='/Users/kose/master/atomic-scheduling-investigation/data/v3_lcc/wiki-Vote.mtx'
 args[args.index('--output-dir')+1]=str(dest)
 args[args.index('--run-id')+1]=label
 env=os.environ.copy();env.pop('SGD_TRACE_DIR',None);env.pop('SGD_TRACE_CHECKPOINTS',None)
 env['SGD_STRESS_SAMPLES']='256'
 if trace:env.update(SGD_TRACE_DIR=str(dest/'trace'),SGD_TRACE_CHECKPOINTS='')
 result=subprocess.run(args,env=env,cwd=root,text=True,capture_output=True,timeout=120)
 (dest/'stdout.json').write_text(result.stdout);(dest/'stderr.log').write_text(result.stderr)
 result.check_returncode()
 print(label,result.stdout,flush=True)
