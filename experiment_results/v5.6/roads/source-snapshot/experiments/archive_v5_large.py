"""Lossless XZ archival outside measurement; retain decompressed SHA256 evidence."""
import argparse
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import time

from run_v5 import save, now

PRESET=0


def digest(path, opener=open):
    h=hashlib.sha256();size=0
    with opener(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b);size+=len(b)
    return h.hexdigest(),size


def archive_text_artifacts(folder,row):
    folder=Path(folder).resolve()
    started=time.monotonic();manifest_path=folder/'artifact-archive.json'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old:
        save(folder/'archive-before-xz.json',old)
    entries=[];errors=[];removals=[]
    sources={}
    for entry in (old or {}).get('files',[]):
        p=Path(entry['compressed_path'])
        if p.exists():sources[entry['original_path']]=(p,entry)
    for path in folder.glob('*.txt'):
        sources.setdefault(str(path),(path,None))
    try:
        for original,(source,previous) in sources.items():
            if source.suffix=='.xz':
                entries.append(previous)
                # An interrupted archive can already have a verified XZ member
                # while the native record still points to its plaintext source.
                for field in ['initial_positions_path','final_positions_path','vertex_map_path']:
                    if row.get(field) and Path(row[field])==Path(original):
                        row.setdefault('native_'+field,original);row[field]=str(source)
                continue
            opener=gzip.open if source.suffix=='.gz' else open
            target=Path(original+'.xz');temporary=target.with_suffix('.xz.tmp')
            if target.exists():raise FileExistsError(target)
            sha=hashlib.sha256();length=0
            try:
                with opener(source,'rb') as f,lzma.open(temporary,'wb',preset=PRESET,check=lzma.CHECK_CRC64) as output:
                    for b in iter(lambda:f.read(1024*1024),b''):
                        sha.update(b);length+=len(b);output.write(b)
                expected=sha.hexdigest()
                if previous:
                    assert expected==previous['original_sha256'] and length==previous['original_bytes']
                assert digest(temporary,lzma.open)==(expected,length)
                compressed_sha,compressed_size=digest(temporary)
                os.link(temporary,target);temporary.unlink()
                entries.append(dict(original_path=original,compressed_path=str(target),
                    original_sha256=expected,decompressed_sha256=expected,
                    compressed_sha256=compressed_sha,original_bytes=length,
                    compressed_bytes=compressed_size,compression='xz',preset=PRESET,xz_crc_validated=True))
                removals.append((source,target))
            except BaseException:
                temporary.unlink(missing_ok=True);raise
        manifest=dict(format='xz',preset=PRESET,measurement_excluded=True,files=entries,errors=[],
                      converted_at=now(),previous_manifest='archive-before-xz.json' if old else None)
        save(manifest_path,manifest)
        # Persist redirects before removing any previous file. A crash leaves
        # redundant copies, never an untracked loss of the original artifact.
        for source,target in removals:
            for field in ['initial_positions_path','final_positions_path','vertex_map_path']:
                if row.get(field) and Path(row[field])==source:
                    row.setdefault('native_'+field,str(target)[:-3]);row[field]=str(target)
        row['archive_manifest_path']=str(manifest_path)
        row['archive_format']='xz';row['archive_status']='success';row.pop('archive_error',None)
        row['archive_seconds']=time.monotonic()-started
        save(folder/'record.json',row)
        for source,target in removals:source.unlink()
    except Exception as error:
        row.update(archive_status='partial_error',archive_error=repr(error),archive_seconds=time.monotonic()-started)
        save(folder/'record.json',row)
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+');args=parser.parse_args()
    for name in args.folders:
        folder=Path(name);path=folder/'record.json'
        row=json.loads(path.read_text());before=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        original_archive_seconds=row.get('archive_seconds')
        archive_text_artifacts(folder,row)
        row['xz_repack_seconds']=row['archive_seconds']
        row['archive_seconds']=original_archive_seconds
        save(path,row)
        after=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        print(folder.name,'saved MiB',round((before-after)/2**20,1),flush=True)


if __name__=='__main__':main()
