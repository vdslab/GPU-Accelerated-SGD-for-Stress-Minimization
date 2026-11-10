#!/usr/bin/env python3
"""Lossless, self-contained gzip layouts with APFS-shared immutable edge members.

Native measurements are unchanged. Each .gz decodes independently; APFS clones
share only its compressed edge member. Original archives survive until verified
replacement files, manifest and record have been persisted.
"""
import argparse,ctypes,fcntl,gzip,hashlib,io,json,lzma,os,shutil,struct,time,uuid,subprocess,sys
from pathlib import Path
from run_v5 import ROOT,now,save

BASE=ROOT/'experiment_results/v5/large';PREFIX=65536
EDGE=b'# Edges (source target)\n';POSITIONS=b'\n# Positions (x y)\n'


def sha(path,opener=open):
 h=hashlib.sha256();size=0
 with opener(path,'rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data);size+=len(data)
 return h.hexdigest(),size


def padded_header(data):
 raw=gzip.compress(data,compresslevel=1,mtime=0)
 extra_size=PREFIX-len(raw)-2
 assert 4<=extra_size<=65535
 extra=b'V5'+struct.pack('<H',extra_size-4)+bytes(extra_size-4)
 result=raw[:3]+bytes([raw[3]|4])+raw[4:10]+struct.pack('<H',extra_size)+extra+raw[10:]
 assert len(result)==PREFIX
 return result


def clone(source,target):
 libc=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
 fn=libc.clonefile;fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint32];fn.restype=ctypes.c_int
 if fn(os.fsencode(source),os.fsencode(target),0):
  code=ctypes.get_errno();raise OSError(code,os.strerror(code),str(target))


def split_layout(source,on_edge,on_position):
 opener=lzma.open if source.suffix=='.xz' else gzip.open if source.suffix=='.gz' else open
 original=hashlib.sha256();edge_hash=hashlib.sha256();total=0;edge_size=0;header=None;stage=0;buffer=b''
 with opener(source,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):
   original.update(chunk);total+=len(chunk)
   if stage==2:on_position(chunk);continue
   buffer+=chunk
   if stage==0:
    at=buffer.find(EDGE)
    if at<0:
     if len(buffer)>128*1024:raise ValueError('Native edge marker missing in header')
     continue
    end=at+len(EDGE);header=buffer[:end];buffer=buffer[end:];stage=1
   at=buffer.find(POSITIONS)
   if at<0:
    keep=len(POSITIONS)-1
    if len(buffer)>keep:
     data=buffer[:-keep];on_edge(data);edge_hash.update(data);edge_size+=len(data);buffer=buffer[-keep:]
   else:
    end=at+len(POSITIONS);data=buffer[:end];on_edge(data);edge_hash.update(data);edge_size+=len(data)
    on_position(buffer[end:]);buffer=b'';stage=2
 assert header is not None and stage==2,'Incomplete native layout; original retained'
 return header,original.hexdigest(),total,edge_hash.hexdigest(),edge_size


def fsync(path):
 with path.open('rb') as f:os.fsync(f.fileno())


def convert_layout(source,original,row,pool,expected):
 key=row['input_sha256'];pool.mkdir(parents=True,exist_ok=True)
 template=pool/(key+'.edges-template.gz');meta=pool/(key+'.edges.json')
 lock=(pool/(key+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX)
 token=uuid.uuid4().hex;positions_tmp=source.parent/('.shared-positions-'+token+'.gz')
 template_tmp=pool/('.shared-template-'+token+'.tmp');target=Path(original).with_suffix('.shared.txt.gz')
 target_tmp=source.parent/('.shared-target-'+token+'.gz')
 assert not target.exists(),f'Existing target must be inspected before retry: {target}'
 base=None;edge_writer=None;creating=not template.exists()
 try:
  if creating:
   assert not meta.exists();base=template_tmp.open('xb');base.write(padded_header(b''))
   edge_writer=gzip.GzipFile(filename='',mode='wb',fileobj=base,compresslevel=1,mtime=0)
  else:assert meta.exists(),'Template without manifest; originals retained'
  with positions_tmp.open('xb') as raw:
   with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as writer:
    header,digest,length,edge_digest,edge_length=split_layout(source,lambda data:edge_writer.write(data) if edge_writer else None,writer.write)
  assert digest==expected['original_sha256'] and length==expected['original_bytes']
  if edge_writer:edge_writer.close();edge_writer=None
  if base:base.close();base=None
  if creating:
   fsync(template_tmp);template_sha,template_size=sha(template_tmp)
   os.link(template_tmp,template);template_tmp.unlink()
   save(meta,dict(input_sha256=key,edge_sha256=edge_digest,edge_bytes=edge_length,template_sha256=template_sha,template_bytes=template_size,prefix_member_bytes=PREFIX,created_at=now()))
  info=json.loads(meta.read_text());assert info['edge_sha256']==edge_digest and info['edge_bytes']==edge_length,'Native edge order differs; original retained'
  clone(template,target_tmp)
  with target_tmp.open('r+b') as out:
   out.write(padded_header(header));out.seek(0,2)
   with positions_tmp.open('rb') as f:shutil.copyfileobj(f,out,1024*1024)
   out.flush();os.fsync(out.fileno())
  assert sha(target_tmp,gzip.open)==(digest,length),'Replacement does not recover native bytes'
  assert sha(template)[0]==info['template_sha256'],'Shared template changed during clone write'
  encoded,encoded_size=sha(target_tmp)
  os.link(target_tmp,target);target_tmp.unlink()
  return dict(original_path=str(original),compressed_path=str(target),original_sha256=digest,decompressed_sha256=digest,
              original_bytes=length,compressed_sha256=encoded,compressed_bytes=encoded_size,
              compression='gzip',gzip_crc_validated=True,storage='APFS clone; self-contained concatenated gzip members',
              shared_edge_template=str(template),shared_edge_sha256=edge_digest,prefix_member_bytes=PREFIX)
 finally:
  if edge_writer:edge_writer.close()
  if base:base.close()
  for path in [positions_tmp,template_tmp,target_tmp]:path.unlink(missing_ok=True)
  lock.close()


def repack(folder,pool=None):
 folder=Path(folder).resolve();pool=Path(pool or BASE/'shared-edges').resolve()
 record_path=folder/'record.json';manifest_path=folder/'artifact-archive.json'
 row=json.loads(record_path.read_text());assert row['status']=='success'
 manifest=json.loads(manifest_path.read_text())
 if manifest.get('format')=='shared-gzip':
  previous=json.loads((folder/'archive-before-shared-gzip.json').read_text())
  new_by_original={entry['original_path']:entry for entry in manifest['files']}
  cleanup=[];changed=False
  for old in previous['files']:
   if not old['original_path'].endswith(('.initial.txt','.final.txt')):continue
   entry=new_by_original[old['original_path']];target=Path(entry['compressed_path']);source=Path(old['compressed_path'])
   assert target.exists() and target.stat().st_size==entry['compressed_bytes']
   for field in ['initial_positions_path','final_positions_path']:
    if row.get(field) and Path(row[field])==source:row[field]=str(target);changed=True
   if source.exists() and source!=target:
    assert sha(target)[0]==entry['compressed_sha256'];cleanup.append(source)
  if changed or row.get('archive_format')!='shared-gzip':
   row['archive_format']='shared-gzip';row['archive_status']='success';save(record_path,row);fsync(record_path)
  for source in cleanup:source.unlink()
  return dict(folder=str(folder),already_complete=True,recovered_cleanup_files=len(cleanup))
 assert not manifest['errors']
 before_record=folder/'record-before-shared-gzip.json'
 if not before_record.exists():save(before_record,row)
 old=folder/'archive-before-shared-gzip.json'
 if not old.exists():save(old,manifest)
 pending_path=folder/'shared-gzip-pending.json';pending=json.loads(pending_path.read_text()) if pending_path.exists() else {}
 started=time.monotonic();before=shutil.disk_usage(folder).free;entries=[];removed=[]
 for entry in manifest['files']:
  source=Path(entry['compressed_path']);original=Path(entry['original_path'])
  if original.name.endswith(('.initial.txt','.final.txt')):
   new=pending.get(str(original))
   if new:
    assert sha(Path(new['compressed_path']),gzip.open)==(entry['original_sha256'],entry['original_bytes'])
   else:
    assert source.is_file() and source.stat().st_size==entry['compressed_bytes']
    new=convert_layout(source,original,row,pool,entry);pending[str(original)]=new;save(pending_path,pending)
   entries.append(new);removed.append((source,Path(new['compressed_path'])))
  else:entries.append(entry)
 assert len(removed)==2,'Exactly initial and final native layouts required'
 replacement=dict(format='shared-gzip',measurement_excluded=True,files=entries,errors=[],updated_at=now(),
                  storage_definition='Each layout is independently readable with ordinary gzip; APFS shares immutable edge-member blocks. Other artifacts retain prior encoding.',previous_manifest=old.name)
 save(manifest_path,replacement);fsync(manifest_path)
 for source,target in removed:
  for field in ['initial_positions_path','final_positions_path']:
   if row.get(field) and Path(row[field])==source:row[field]=str(target)
 row.update(archive_format='shared-gzip',archive_status='success',shared_repack_seconds=time.monotonic()-started)
 original_record=json.loads(before_record.read_text())
 allowed={'initial_positions_path','final_positions_path','archive_format','archive_status','shared_repack_seconds'}
 assert all(row.get(k)==value for k,value in original_record.items() if k not in allowed),'A measurement field changed; sources retained'
 save(record_path,row);fsync(record_path)
 for source,target in removed:source.unlink()
 result=dict(folder=str(folder),at=now(),seconds=time.monotonic()-started,free_before=before,free_after=shutil.disk_usage(folder).free,native_measurement_fields_changed=False)
 save(folder/'shared-gzip-storage.json',result)
 return result


def main():
 parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+');parser.add_argument('--pool');parser.add_argument('--resume-runner',action='store_true');args=parser.parse_args()
 state=dict(pid=os.getpid(),active=True,phase='lossless_shared_edge_repack',started_at=now(),planned=len(args.folders),finished=0,results=[])
 save(BASE/'shared-edges-maintenance.json',state)
 try:
  for folder in args.folders:
   result=repack(folder,args.pool);state['results'].append(result);state['finished']+=1;state['updated_at']=now()
   save(BASE/'shared-edges-maintenance.json',state);print(json.dumps(result),flush=True)
  if args.resume_runner:
   import psutil
   assert not any('sparse-sgd-gpu' in (p.info['name'] or '') for p in psutil.process_iter(['name']))
   with (BASE/'logs/runner.log').open('a') as log:
    runner=subprocess.Popen([sys.executable,'-u',str(ROOT/'experiments/run_v5_large.py'),'--reuse-recent-warmups'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state['runner_pid']=runner.pid;state['phase']='measurement_resumed'
  else:state['phase']='verified_repack_complete'
 except BaseException as error:
  state.update(active=False,phase='repack_error',error=repr(error),updated_at=now());save(BASE/'shared-edges-maintenance.json',state);raise
 state.update(active=False,completed_at=now(),free_bytes=shutil.disk_usage(BASE).free);save(BASE/'shared-edges-maintenance.json',state)

if __name__=='__main__':main()
