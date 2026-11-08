#!/usr/bin/env python3
"""Stop the already-loaded old job list after the user's retained graph completes."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/v5'
def main():
 original=json.loads((BASE/'process.json').read_text());pid=original['pid']
 while True:
  records=list((BASE/'raw/comparison').glob('v5-comparison-coPapersCiteseer-*/record.json'))
  if len(records)==45:
   assert all(json.loads(p.read_text())['status']=='success' for p in records)
   break
  if not psutil.pid_exists(pid):raise SystemExit('Runner stopped before retained graph completed; normal monitor must investigate')
  time.sleep(.2)
 cancelled=[]
 if psutil.pid_exists(pid):
  parent=psutil.Process(pid);os.kill(pid,signal.SIGSTOP)
  children=parent.children(recursive=True)
  # All retained runs have already persisted record.json; only cancelled scope
  # may have started during the race between the last save and SIGSTOP.
  allowed={'auto','NACA0015','Flan_1565'}
  for child in children:
   try:
    cmd=child.cmdline()
    if '--input' in cmd:
     name=Path(cmd[cmd.index('--input')+1]).stem
     if name not in allowed:
      os.kill(pid,signal.SIGCONT);raise RuntimeError(f'Refuse to stop retained child: {cmd}')
    cancelled.append(dict(pid=child.pid,command=cmd));child.terminate()
   except psutil.NoSuchProcess:pass
  _,alive=psutil.wait_procs(children,timeout=5)
  for child in alive:child.kill()
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  try:parent.wait(timeout=5)
  except psutil.TimeoutExpired:raise RuntimeError('Old runner did not exit; refusing duplicate start')
 event=dict(reason='user scope cutoff after coPapersCiteseer',retained_records=45,cancelled_out_of_scope_processes=cancelled,measurement_interrupted=False)
 (BASE/'validation/comparison-cutoff.json').write_text(json.dumps(event,ensure_ascii=False,indent=2)+'\n')
 with (BASE/'logs/runner-comparison.log').open('a') as log:
  proc=subprocess.Popen([sys.executable,'-u','experiments/run_v5.py','--stage','comparison'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (BASE/'process.json').write_text(json.dumps(dict(pid=proc.pid,stage='comparison',revision='user-cutoff-coPapersCiteseer',rr_full_tile_size=1024,log='experiment_results/v5/logs/runner-comparison.log'),indent=2)+'\n')
 result=proc.wait();event['finalization_exit_code']=result
 (BASE/'validation/comparison-cutoff.json').write_text(json.dumps(event,ensure_ascii=False,indent=2)+'\n')
 print('Retained scope measurement finished; final summary return code',result,flush=True)
if __name__=='__main__':main()
