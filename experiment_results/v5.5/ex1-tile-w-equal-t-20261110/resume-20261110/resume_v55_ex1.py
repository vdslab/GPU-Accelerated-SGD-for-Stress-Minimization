#!/usr/bin/env python3
"""Resume the frozen optimized Full tile sweep without overwriting completed runs."""
import collections
import json
import shutil
import sys
import time
from pathlib import Path
import run_v55_ex1 as run
from run_experiments import file_sha256 as sha
from run_v5 import now, save, memory

BASE = run.BASE
SESSION = BASE/'resume-20261110'
last_note = 0


def progress(active=None, active_elapsed=0, state=None, force=False):
    global last_note
    rows = run._rows
    done = {r['run_id'] for r in rows}
    remaining = [j for j in run._jobs if j['run_id'] not in done]
    times = collections.defaultdict(list)
    for r in rows: times[r['graph'], r['tile_size']].append(r['process_wall_seconds'])
    import statistics
    known = sum(statistics.median(times[j['graph'], j['tile_size']]) for j in remaining if (j['graph'], j['tile_size']) in times)
    s = dict(updated=now(), version='v5.5', state=state or ('running' if active else 'in_progress'),
             planned=1650, finished=len(rows), success=sum(r['status']=='success' for r in rows),
             failed=sum(r['status']!='success' for r in rows), remaining=len(remaining),
             active=active, active_elapsed_seconds=active_elapsed,
             eta_known_conditions_seconds=known,
             eta_unmeasured_runs=sum((j['graph'], j['tile_size']) not in times for j in remaining),
             preserved_before_resume=742, new_since_resume=len(rows)-742,
             memory=memory(), free_disk_bytes=shutil.disk_usage(BASE).free,
             ex2_state='completed_with_capacity_supplement; unchanged')
    save(BASE/'progress.json', s)
    msg=f"{now()} 状態={s['state']}、成功{s['success']}/1650（今回追加{s['new_since_resume']}）、残り{len(remaining)}、失敗{s['failed']}。実行中：{active['run_id'] if active else 'なし'}（{active_elapsed:.0f}秒）。T=W、凍結Fullバイナリを継続。EX2は完了済みで再実行しない。"
    (BASE/'PROGRESS.md').write_text('# v5.5 EX1再開\n\n'+msg+'\n')
    if force or time.monotonic()-last_note>=60:
        last_note=time.monotonic(); print(msg, flush=True)
        run.NOTE.write_text('# EX1 v5.5：タイル実験（W=T）\n\n前回742/1650件で停止した実験を、同じ新Fullの凍結バイナリで再開。742件は上書きせず、残り908件を測定する。中断した1件の証跡はresume-20261110/interrupted/に保持。\n\n10グラフ×T=W=1,2,4,8,16,32,64,128,256,512,1024×seed10〜14×3回、15反復、epsilon0.1、shared、内外shuffle有効、phase_chunks1、auto BFS、GPU timestamp無効、全対象で厳密Full Stress（計時後）。各graph/Tのseed999ウォームアップは集計外。CPUは過去150件を参照し、CPU/Atomic・完了済みEX2は再実行しない。新規solverは逐次実行。\n\n## 現在の進捗\n\n'+msg+f'\n\n正本：`{BASE}`。結果：[[EX1_v5.5_タイル実験(W=T)_結果と考察]]。\n')
    return s


def main():
    run.integrity(True)
    SESSION.mkdir(exist_ok=True)
    manifest = SESSION/'manifest.json'
    if not manifest.exists():
        rows = run.records()
        assert len(rows)==742 and all(r['status']=='success' and r['archive_status']=='success' for r in rows)
        protected={str(p.relative_to(run.ROOT)):sha(p) for p in (BASE/'raw/tile').glob('*/record.json')}
        save(SESSION/'preserved-records-sha256.json', protected)
        for name in ['progress.json','STOPPED.md','process.json']:
            p=BASE/name
            if p.exists():shutil.copy2(p, SESSION/('before-resume-'+name))
        if run.NOTE.exists():shutil.copy2(run.NOTE, SESSION/'before-resume-obsidian.md')
        interrupted=[]
        for folder in (BASE/'raw/tile').iterdir():
            if folder.is_dir() and not (folder/'record.json').exists():
                assert folder.name=='v55-ex1-PGPgiantcompo-rr_sgd-t8-s12-r1'
                dest=SESSION/'interrupted'/folder.name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(folder),str(dest));interrupted.append(dict(original=str(folder),retained=str(dest)))
        save(manifest,dict(started=now(),preserved=742,remaining=908,interrupted_attempts=interrupted,
             binary_path=json.loads((BASE/'config/environment.json').read_text())['binary_paths']['rr_sgd'],
             binary_sha256=json.loads((BASE/'config/environment.json').read_text())['binary_sha256']['rr_sgd'],
             source_commit='70e945730d6bce7922de3174b556e2e3d03f0375',new_full_optimized_implementation=True,
             parameters_changed=False,ex2_already_completed=True,wrapper_sha256=sha(Path(__file__))))
        shutil.copy2(Path(__file__), SESSION/Path(__file__).name)
    run.progress=progress
    sys.argv=[str(run.__file__),'--run']
    run.main()
    for rel,h in json.loads((SESSION/'preserved-records-sha256.json').read_text()).items():assert sha(run.ROOT/rel)==h
    # The original frozen reporter predates the EX2-first scope change; clarify its status only.
    for p in [BASE/'reports/tile.md',run.VAULT/'EX1_v5.5_タイル実験(W=T)_結果と考察.md']:
        if p.exists():p.write_text(p.read_text().replace('EX2は未開始。','EX2は別途完了済み。今回の再開でEX2を再実行していない。'))
    for p in [BASE/'completion.json',BASE/'validation/completion-validation.json']:
        d=json.loads(p.read_text());d.update(resumed_from=742,new_after_resume=908,preserved_742_record_hashes_unchanged=True,ex2_already_completed=True);save(p,d)
    save(SESSION/'completion.json',dict(completed=now(),total_success=1650,new_after_resume=908,preserved_742_records_unchanged=True,ex2_unchanged=True))
    progress(state='completed',force=True)

if __name__=='__main__': main()
