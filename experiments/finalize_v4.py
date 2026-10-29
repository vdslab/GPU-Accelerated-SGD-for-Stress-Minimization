#!/usr/bin/env python3
"""Wait for the serial benchmark, then audit and create reviewable outputs."""
from pathlib import Path
import argparse,json,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/v4'
ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);args=ap.parse_args()
if args.wait_pid:
 try:p=psutil.Process(args.wait_pid)
 except psutil.NoSuchProcess:p=None
 while p is not None and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE:time.sleep(5)
checkpoint=json.loads((BASE/'checkpoint.json').read_text());assert checkpoint['completed']==checkpoint['planned'],checkpoint
for script in ['summarize_v4.py','plot_v4.py','report_v4.py','paper_v4.py','diagram_v4.py']:
 print('Starting',script,flush=True);subprocess.run([sys.executable,str(ROOT/'experiments'/script)],cwd=ROOT,check=True)
print('Analysis and local drafts ready for visual inspection; esa has not been changed.',flush=True)
