#!/usr/bin/env python3
"""Lossless Git object packing during v5's disk-only pause; never expire/prune history."""
import fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
from run_v5 import ROOT,now,save
BASE=ROOT/'experiment_results/v5/large';GIB=1024**3

def main():
 lock=(BASE/'maintenance.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 objects=ROOT/'.git/objects';initial_refs=subprocess.check_output(['git','show-ref'],cwd=ROOT,text=True)
 state=dict(pid=os.getpid(),started_at=now(),active=True,phase='git_object_compaction',policy='All refs, reflogs, checkpoints and unreachable objects retained. Only duplicate loose copies in verified packs removed.',packs=[])
 save(BASE/'maintenance.json',state)
 files=[p for d in objects.iterdir() if len(d.name)==2 and d.is_dir() for p in d.iterdir() if len(p.name)==38 and p.is_file() and p.stat().st_size<100*1024**2]
 files.sort(key=lambda p:p.stat().st_size,reverse=True)
 batch=[];size=0;batches=[]
 for p in files:
  n=p.stat().st_size
  if batch and size+n>512*1024**2:batches.append(batch);batch=[];size=0
  batch.append(p);size+=n
 if batch:batches.append(batch)
 for index,batch in enumerate(batches):
  if shutil.disk_usage(ROOT).free>=18*GIB:break
  if shutil.disk_usage(ROOT).free<2*GIB:raise RuntimeError('Insufficient safe space for one <=512MiB pack; originals retained')
  batch=[p for p in batch if p.exists()]
  if not batch:continue
  oids=[p.parent.name+p.name for p in batch];physical=sum(p.stat().st_size for p in batch);before=shutil.disk_usage(ROOT).free;started=time.monotonic()
  state.update(batch=index+1,batches_total=len(batches),input_bytes=physical,updated=now());save(BASE/'maintenance.json',state)
  command=['git','pack-objects','--delta-base-offset','--window=50','--window-memory=512m','--threads=2','--depth=50','--compression=3',str(objects/'pack/pack')]
  result=subprocess.run(command,input='\n'.join(oids)+'\n',cwd=ROOT,text=True,capture_output=True,check=True)
  packid=result.stdout.strip();idx=objects/'pack'/f'pack-{packid}.idx'
  verification=subprocess.run(['git','verify-pack','-v',str(idx)],cwd=ROOT,text=True,capture_output=True,check=True)
  present={line.split()[0] for line in verification.stdout.splitlines() if line and len(line.split()[0])==40}
  assert set(oids)<=present
  subprocess.run(['git','prune-packed'],cwd=ROOT,check=True)
  item=dict(pack=str(idx),objects=len(oids),loose_bytes=physical,packed_bytes=idx.with_suffix('.pack').stat().st_size,seconds=time.monotonic()-started,free_after=shutil.disk_usage(ROOT).free,verified=True)
  state['packs'].append(item);state['updated']=now();save(BASE/'maintenance.json',state)
  print(json.dumps(item),flush=True)
 final_refs=subprocess.check_output(['git','show-ref'],cwd=ROOT,text=True)
 state.update(refs_unchanged=initial_refs==final_refs,phase='wait_for_artifact_repack',updated=now());save(BASE/'maintenance.json',state)
 # A separately started lossless Flan rearchive must finish before measurement.
 info=json.loads((BASE/'validation/repack-process.json').read_text());pid=info['pid']
 while psutil.pid_exists(pid):
  try:
   p=psutil.Process(pid)
   if p.status()==psutil.STATUS_ZOMBIE or 'archive_v5_large.py' not in ' '.join(p.cmdline()):break
  except psutil.Error:break
  time.sleep(2)
 # Check every scheduled artifact rearchive completed rather than merely exited.
 planned=json.loads((BASE/'validation/repack-flan-plan.json').read_text())['folders']
 for folder in planned:
  d=json.loads((Path(folder)/'record.json').read_text());m=json.loads((Path(folder)/'artifact-archive.json').read_text())
  assert d['archive_status']=='success' and m['format']=='xz',folder
 state.update(phase='resume_measurement',updated=now(),free_bytes=shutil.disk_usage(ROOT).free);save(BASE/'maintenance.json',state)
 for source in [Path(__file__),ROOT/'experiments/run_v5_large.py',ROOT/'experiments/archive_v5_large.py']:shutil.copy2(source,BASE/'source-snapshot'/source.name)
 with (BASE/'logs/runner.log').open('a') as log:
  runner=subprocess.Popen([sys.executable,'-u',str(ROOT/'experiments/run_v5_large.py'),'--reuse-recent-warmups'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(active=False,completed_at=now(),phase='measurement_resumed',runner_pid=runner.pid);save(BASE/'maintenance.json',state)
 print('runner resumed',runner.pid,flush=True)

if __name__=='__main__':main()
