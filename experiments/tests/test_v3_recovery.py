"""Metadata-only recovery contract tests; run after all native measurements.

Only live process inventory is patched. Validation, checksums, ID matching,
overlay selection and audit bookkeeping run unchanged against temporary files.
Tiny placeholder payloads stand in for acquisition-validated large files; these
tests do not test SNAP parsing, coordinate contents or gzip payload validation.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments import audit_v3 as audit
from experiments import v3_recovery as recovery


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


class Fixture:
    def __init__(self, root):
        self.main=root/'v3'; self.extra=self.main/recovery.RECOVERY_DIRECTORY
        self.binaries={method:sha(method.encode()) for method in ('sgd','atomic_sgd','rr_sgd',*recovery.METHODS)}
        self.hashes={}
        for index in range(67):
            relative=f'implementations/fixture/source-{index}.rs'
            payload=f'// virtual frozen source {index}\n'.encode()
            path=root/'source-snapshot'/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(payload)
            self.hashes[relative]=sha(payload)
        self.hashes.update({'experiments/run_v3.py':sha(b'virtual runner'),
                            'experiments/prepare_v3.py':sha(b'virtual preparation')})
        for relative,payload in [('experiments/run_v3.py',b'virtual runner'),('experiments/prepare_v3.py',b'virtual preparation')]:
            path=root/'source-snapshot'/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(payload)
        self.limits={'timeout_seconds':120,'max_rss_bytes':8*1024**3,'min_free_disk_bytes':8*1024**3}
        self.datasets=[dict(name=name,family='large',source_graph='SNAP/'+name,source_nodes=spec[0],
                            status='unavailable',eligible=False,reason='Original source unavailable or unverified',
                            error='Legacy source error',source_path=str(root/'old-alternative.mtx'),
                            source_distribution_status='alternative_distribution_metadata_matched') for name,spec in recovery.SNAP_ORIGINS.items()]
        self.datasets += [dict(name='stub-'+str(index),family='small',status='unavailable',eligible=False) for index in range(92)]
        self.datasets += [dict(name='power197k',family='large',status='unavailable',eligible=False),
                          dict(name='G_n_pin_pout',family='large',status='ready',eligible=False,nodes=99995)]
        self.datasets[6].update(status='ready',eligible=True,nodes=2,edges=1,sha256=sha(b'core input'))
        core_input=self.main/'data/core.tiny'; core_input.parent.mkdir(parents=True,exist_ok=True); core_input.write_bytes(b'core input')
        self.datasets[6]['path']=str(core_input)
        self.plan=[self.job(f'core-{i}', 'stub-0' if i<2 else 'stub-1') for i in range(3120)]
        self.plan[1].update(study='convergence',reuse='core-0')
        self.targets=[self.job(f'v3-basic-{name}-{method}-s{seed}-p200-tNone',name,method,seed)
                      for name in recovery.SNAP_ORIGINS for method in recovery.METHODS for seed in range(5)]
        self.plan+=self.targets
        self.core_rows=[dict(job,status='unavailable') for job in self.plan]
        for row in self.core_rows:
            if row['graph'] in recovery.SNAP_ORIGINS: row['error']='Original source unavailable or unverified'
        self.core_rows[0]=self.success(self.plan[0],self.datasets[6],self.main,recovery.CORE_NATIVE_COMMIT)
        self.core_rows[1]=dict(self.core_rows[0],**self.plan[1],reused_from='core-0')
        self.config(self.main,self.datasets,self.plan,recovery.CORE_NATIVE_COMMIT)
        write_rows(self.main/'raw/results.jsonl',self.core_rows)
        interrupted=self.main/audit.REBOOT_RECEIPT.parent/'artifacts'/'core-0'
        write_json(interrupted/'command.json',{'argv':['virtual-native','--run-id','core-0']})
        (interrupted/'stdout.json').write_bytes(b''); (interrupted/'stderr.log').write_bytes(b'')
        write_json(self.main/audit.REBOOT_RECEIPT,dict(interrupted_run_id='core-0',
            incident='Host reboot interrupted an unconfirmed attempt',preserved_artifact_directory=str(interrupted),
            observed_boot_time_jst='2026-10-24T12:19:50+09:00',
            files=[dict(relative_path=p.name,bytes=p.stat().st_size) for p in interrupted.iterdir()]))
        sources=[]; prepared=[]
        for name,(nodes,edges,suffix) in recovery.SNAP_ORIGINS.items():
            source={'name':name,'origin_url':'https://snap.stanford.edu/data/'+suffix}
            for field in ('origin','converted_mtx','matrix_node_ids','prepared','vertex_map','lcc_origin_node_ids'):
                path=self.extra/'payloads'/name/(field+'.tiny'); path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((name+':'+field).encode()); source[field+'_path']=str(path); source[field+'_sha256']=recovery.file_sha256(path)
            source['origin_bytes']=Path(source['origin_path']).stat().st_size
            info=dict(name=name,family='large',source_graph='SNAP/'+name,source_nodes=nodes,status='ready',eligible=True,
                      nodes=nodes,edges=edges,path=source['prepared_path'],sha256=source['prepared_sha256'],
                      vertex_map_path=source['vertex_map_path'],vertex_map_sha256=source['vertex_map_sha256'],
                      source_path=source['converted_mtx_path'],source_sha256=source['converted_mtx_sha256'],
                      origin_url=source['origin_url'],origin_path=source['origin_path'],origin_sha256=source['origin_sha256'],
                      matrix_node_ids_path=source['matrix_node_ids_path'],matrix_node_ids_sha256=source['matrix_node_ids_sha256'],
                      source_distribution_status='official_snap_origin_recovery',preparation_policy=recovery.PREPARATION_POLICY)
            validation=dict(schema_version=1,kind='official_snap_origin_validation',name=name,identity_code=recovery.IDENTITY_CODE,
                            metadata_matches_official=True,official_nodes=nodes,official_edges=edges,observed_unique_nodes=nodes,
                            official_expanded_nnz=edges if name=='soc-Pokec' else 2*edges,
                            observed_expanded_nnz=edges if name=='soc-Pokec' else 2*edges,
                            observed_undirected_expanded_nnz=2*edges,official_edges_are_directed=name=='soc-Pokec',
                            observed_unique_edges=edges,observed_unique_undirected_edges=edges,
                            observed_unique_directed_edges=edges,unique_self_loops=0,source_components=1,
                            preparation_policy=recovery.PREPARATION_POLICY,gzip_crc_validated=True,
                            prepared_nodes=info['nodes'],prepared_edges=info['edges'],prepared_sha256=info['sha256'])
            for field in ('origin_url','origin_sha256','origin_bytes','converted_mtx_sha256','matrix_node_ids_path',
                          'matrix_node_ids_sha256','vertex_map_path','vertex_map_sha256',
                          'lcc_origin_node_ids_path','lcc_origin_node_ids_sha256'):
                validation[field]=source[field]
            path=self.extra/'validation'/(name+'.json'); write_json(path,validation)
            source.update(validation_path=str(path),validation_sha256=recovery.file_sha256(path))
            info['recovery_source']=dict(source)
            sources.append(source); prepared.append(info)
        self.extra_datasets=prepared
        self.config(self.extra,prepared,self.targets,'a'*40)
        catalog={row['name']:row for row in prepared}
        self.extra_rows=[self.success(job,catalog[job['graph']],self.extra,'a'*40) for job in self.targets]
        write_rows(self.extra/'raw/results.jsonl',self.extra_rows)
        self.manifest=dict(schema_version=1,kind='official_snap_origin_recovery',target_graphs=list(recovery.SNAP_ORIGINS),
                           main_base=str(self.main),preparation_policy=recovery.PREPARATION_POLICY,
                           identity_code=recovery.IDENTITY_CODE,identity_claim='公式原配布。SuiteSparseとの直接照合未実施。',sources=sources)
        self.refresh_manifest()

    @staticmethod
    def job(run_id, graph, method='sgd', seed=0):
        return dict(run_id=run_id,study='basic',graph=graph,method=method,seed=seed,iterations=15,epsilon=.1,
                    pivots=200 if method in recovery.METHODS else None,tile_size=None,workgroup_size=None,reuse=None)

    def success(self, job, info, base, commit):
        manifest=base/'artifacts'/job['run_id']/'artifact-archive.json'
        write_json(manifest,{'files':[],'errors':[]})
        return dict(job,status='success',process_wall_seconds=.1,iteration_time_ms=1,stress_value=1,
                    nodes=info['nodes'],edges=info['edges'],input_sha256=info['sha256'],
                    binary_sha256=self.binaries[job['method']],git_commit=commit,git_dirty=False,
                    constraints=1,attempted_updates=15,completed_updates=15,retry_failures=0,
                    archive_status='success',archive_manifest_path=str(manifest))

    def config(self, base, datasets, plan, commit):
        for name,value in [('datasets.json',datasets),('plan.json',plan),('source-hashes.json',self.hashes),('limits.json',self.limits)]:
            write_json(base/'config'/name,value)
        execution=dict(git_commit=commit,binary_sha256=self.binaries,limits=self.limits,
                       runner_sha256=self.hashes['experiments/run_v3.py'],
                       plan_sha256=sha(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()),
                       datasets_sha256=recovery.file_sha256(base/'config/datasets.json'))
        write_json(base/'config/execution.json',execution)

    def refresh_manifest(self):
        for field,path in [('execution','config/execution.json'),('results','raw/results.jsonl'),('datasets','config/datasets.json'),
                           ('plan','config/plan.json'),('source_hashes','config/source-hashes.json')]:
            self.manifest['main_'+field+'_sha256']=recovery.file_sha256(self.main/path)
        write_json(self.extra/'config/recovery.json',self.manifest)

    def refresh_sources(self):
        sources={source['name']:source for source in self.manifest['sources']}
        for info in self.extra_datasets:
            info['recovery_source']=dict(sources[info['name']])
        self.config(self.extra,self.extra_datasets,self.targets,'a'*40)
        self.refresh_manifest()

    def block_with_warmup(self, index, excluded=False):
        graph='G_n_pin_pout' if excluded else 'stub-0'
        job=self.job(self.plan[index]['run_id'],graph,'sparse_sgd' if excluded else 'sgd')
        self.plan[index]=job
        warmup_job=dict(job,study='warmup',run_id=f'warmup-{graph}-{job["method"]}',seed=0)
        if excluded:
            warmup=dict(warmup_job,status='unavailable')
        else:
            warmup=dict(self.success(warmup_job,self.datasets[6],self.main,recovery.CORE_NATIVE_COMMIT),
                        status='timeout',process_wall_seconds=120)
        path=self.main/'warmups'/warmup['run_id']/'record.json'
        write_json(path,warmup)
        self.core_rows[index]=dict(job,status='warmup_failed',executed=False,
                                  failure_reason=warmup['status'],warmup_run_id=warmup['run_id'])
        self.config(self.main,self.datasets,self.plan,recovery.CORE_NATIVE_COMMIT)
        write_rows(self.main/'raw/results.jsonl',self.core_rows)
        self.refresh_manifest()
        return path


class RecoveryContractTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.fixture=Fixture(Path(self.directory.name))
        inventory=patch.object(recovery,'active_runners',return_value=[])
        inventory.start(); self.addCleanup(inventory.stop)

    def test_absent_and_partial_leave_core_unchanged(self):
        fixture=self.fixture
        write_rows(fixture.extra/'raw/results.jsonl',fixture.extra_rows[:-1])
        for expected in ('incomplete','absent'):
            if expected=='absent': shutil.rmtree(fixture.extra)
            datasets,rows,meta=recovery.load_effective_cohort(fixture.main)
            self.assertEqual(meta['status'],expected); self.assertEqual(len(rows),3210)
            self.assertEqual(sum(r['status']=='unavailable' for r in rows if r['graph'] in recovery.SNAP_ORIGINS),90)
            self.assertTrue(all(r['analysis_origin_scope']=='core' for r in rows))
            self.assertEqual(sum(d['status']=='ready' for d in datasets if d['name'] in recovery.SNAP_ORIGINS),0)

    def test_complete_overlay_and_audit_accounting(self):
        fixture=self.fixture; before=recovery.file_sha256(fixture.main/'raw/results.jsonl')
        datasets,rows,meta=recovery.load_effective_cohort(fixture.main)
        self.assertEqual((meta['status'],meta['raw_record_count'],len(rows)),('adopted',3300,3210))
        self.assertEqual(len(meta['superseded_records']),90); self.assertEqual(len(meta['record_provenance']),3300)
        self.assertEqual(sum(r['analysis_origin_scope']=='snap_origin_recovery' for r in rows),90)
        self.assertEqual({r['run_id']:r['status'] for r in rows if r['graph'] not in recovery.SNAP_ORIGINS},
                         {r['run_id']:r['status'] for r in fixture.core_rows if r['graph'] not in recovery.SNAP_ORIGINS})
        self.assertEqual(next(r for r in rows if r['run_id']=='core-0')['git_commit'],recovery.CORE_NATIVE_COMMIT)
        self.assertEqual({r['git_commit'] for r in rows if r['graph'] in recovery.SNAP_ORIGINS},{'a'*40})
        self.assertEqual(sum(r['effective'] for r in meta['record_provenance']),3210)
        self.assertEqual(sum(d['analysis_origin_base']==str(fixture.extra.resolve()) for d in datasets),6)
        current=[d for d in datasets if d['name'] in recovery.SNAP_ORIGINS]
        self.assertTrue(all('reason' not in d and 'error' not in d for d in current))
        self.assertTrue(all(d['source_distribution_status']=='official_snap_origin_recovery' and 'old-alternative' not in d['source_path'] for d in current))
        self.assertTrue(all(r['error']=='Original source unavailable or unverified' for r in meta['superseded_records']))
        self.assertEqual(recovery.read_json(fixture.main/'config/datasets.json')[0]['error'],'Legacy source error')
        self.assertTrue(all(Path(d['path']).is_absolute() for d in datasets if d['name'] in recovery.SNAP_ORIGINS))
        self.assertFalse(next(d for d in datasets if d['name']=='G_n_pin_pout')['eligible'])
        self.assertEqual(next(d for d in datasets if d['name']=='power197k')['status'],'unavailable')
        self.assertEqual(recovery.file_sha256(fixture.main/'raw/results.jsonl'),before)
        with patch.object(audit,'active_runners',return_value=[]):
            core=audit.audit(fixture.main,require_reboot_receipt=False)
            extra=audit.audit(fixture.extra,expected_plan_count=90,require_reboot_receipt=False)
        self.assertEqual(core[0]['errors'],[]); self.assertEqual(extra[0]['errors'],[])
        result=audit.combine_recovery_audits(fixture.main,core,extra,meta,datasets,rows)[0]
        self.assertEqual(result['physical_formal_executions'],91)
        self.assertEqual(result['additional_interrupted_formal_attempts'],1)
        self.assertEqual(result['total_formal_native_launch_attempts'],92)
        self.assertEqual(len(result['interrupted_formal_attempts']),1)
        self.assertTrue(result['interrupted_formal_attempts'][0]['evidence_valid'])
        self.assertEqual(result['reused_conditions'],1); self.assertEqual(result['unexecuted_formal_conditions'],3118)
        self.assertEqual(sum(result[k] for k in ('physical_formal_executions','reused_conditions','unexecuted_formal_conditions')),3210)
        self.assertEqual(result['errors'],[])

    def test_invalid_contracts_are_rejected_without_partial_overlay(self):
        for mutation,kind in [('attempted','recovery_would_overwrite_attempted_or_available_condition'),
                              ('source','native_shared_67_file_set_or_sha_mismatch'),('binary','binary_set_sha_mismatch'),
                              ('metadata','official_source_observed_nodes_or_unique_edges_mismatch'),
                              ('id','result_ids_not_exactly_complete'),('manifest','recovery_manifest_target_graphs')]:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                fixture=Fixture(Path(temporary))
                if mutation=='attempted':
                    fixture.core_rows[-1].update(status='error',process_wall_seconds=.1,binary_sha256=fixture.binaries[fixture.core_rows[-1]['method']])
                    write_rows(fixture.main/'raw/results.jsonl',fixture.core_rows); fixture.refresh_manifest()
                elif mutation in ('source','binary'):
                    path=fixture.extra/'config'/('source-hashes.json' if mutation=='source' else 'execution.json')
                    value=recovery.read_json(path)
                    if mutation=='source': value[next(k for k in value if k.startswith('implementations/'))]='0'*64
                    else: value['binary_sha256']['sparse_sgd']='0'*64
                    write_json(path,value)
                elif mutation=='metadata':
                    source=fixture.manifest['sources'][0]; path=Path(source['validation_path'])
                    value=recovery.read_json(path); value['observed_unique_edges']-=1; write_json(path,value)
                    source['validation_sha256']=recovery.file_sha256(path); fixture.refresh_sources()
                elif mutation=='id':
                    fixture.extra_rows[-1]['run_id']='unplanned-id'; write_rows(fixture.extra/'raw/results.jsonl',fixture.extra_rows)
                else:
                    fixture.manifest['target_graphs'][0]='power197k'; fixture.refresh_manifest()
                _,rows,meta=recovery.load_effective_cohort(fixture.main)
                self.assertEqual(meta['status'],'rejected'); self.assertEqual(len(rows),3210)
                self.assertTrue(all(row['analysis_origin_scope']=='core' for row in rows))
                self.assertIn(kind,{error['kind'] for error in meta['errors']})

    def test_cross_attempt_receipts_and_missing_invariants_are_rejected(self):
        mutations=[('origin_url','https://snap.stanford.edu/data/wrong-attempt.txt.gz','recovery_validation_source_receipt_mismatch'),
                   ('origin_sha256','0'*64,'recovery_validation_source_receipt_mismatch'),
                   ('origin_bytes',123456,'recovery_validation_source_receipt_mismatch'),
                   ('converted_mtx_sha256','0'*64,'recovery_validation_source_receipt_mismatch'),
                   ('matrix_node_ids_sha256','0'*64,'recovery_validation_source_receipt_mismatch'),
                   ('vertex_map_sha256','0'*64,'recovery_validation_source_receipt_mismatch'),
                   ('lcc_origin_node_ids_sha256','0'*64,'recovery_validation_source_receipt_mismatch'),
                   ('matrix_node_ids_path','wrong-mapping.npy','recovery_validation_source_path_mismatch'),
                   ('vertex_map_path','wrong-mapping.npy','recovery_validation_source_path_mismatch'),
                   ('lcc_origin_node_ids_path','wrong-mapping.npy','recovery_validation_source_path_mismatch'),
                   ('gzip_crc_validated',False,'recovery_validation_crc_or_graph_invariants_mismatch'),
                   ('observed_expanded_nnz',1,'recovery_validation_crc_or_graph_invariants_mismatch'),
                   ('source_components',2,'recovery_validation_crc_or_graph_invariants_mismatch'),
                   ('unique_self_loops',1,'recovery_validation_undirected_or_prepared_edges_mismatch')]
        for field,value,kind in mutations:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temporary:
                fixture=Fixture(Path(temporary)); source=fixture.manifest['sources'][0]
                path=Path(source['validation_path']); validation=recovery.read_json(path)
                validation[field]=value; write_json(path,validation)
                source['validation_sha256']=recovery.file_sha256(path)
                fixture.refresh_sources()
                _,rows,meta=recovery.load_effective_cohort(fixture.main)
                self.assertEqual(meta['status'],'rejected')
                self.assertTrue(all(row['analysis_origin_scope']=='core' for row in rows))
                self.assertIn(kind,{error['kind'] for error in meta['errors']})

    def test_dataset_download_receipt_mismatch_is_rejected(self):
        for field,value in [('origin_url','https://snap.stanford.edu/data/wrong.txt.gz'),('origin_sha256','0'*64),
                            ('matrix_node_ids_sha256','0'*64),('origin_path','wrong-origin.gz')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temporary:
                fixture=Fixture(Path(temporary)); fixture.extra_datasets[0][field]=value
                fixture.config(fixture.extra,fixture.extra_datasets,fixture.targets,'a'*40)
                fixture.refresh_manifest()
                _,_,meta=recovery.load_effective_cohort(fixture.main)
                self.assertEqual(meta['status'],'rejected')
                self.assertTrue(any(error['kind'].startswith('recovery_dataset_') for error in meta['errors']))

    def test_physical_warmup_failure_and_input_exclusion_remain_distinct(self):
        fixture=self.fixture
        fixture.block_with_warmup(2)
        fixture.block_with_warmup(3,excluded=True)
        datasets,rows,meta=recovery.load_effective_cohort(fixture.main)
        self.assertEqual(meta['status'],'adopted')
        with patch.object(audit,'active_runners',return_value=[]):
            report,physical,failed,rss,unexecuted,reused=audit.audit(fixture.main,require_reboot_receipt=False)
        self.assertEqual(report['errors'],[])
        self.assertEqual(report['physical_warmup_executions'],1)
        self.assertEqual(report['physical_warmup_failures'],1)
        self.assertEqual(report['blocking_warmup_kinds'],{'physical_warmup_failure':1,'size_excluded':1})
        self.assertEqual(sum(row['scope']=='warmup' for row in failed),1)

    def test_inconsistent_blocking_warmup_is_rejected_by_loader_and_audit(self):
        mutations=[('graph','stub-1','blocking_warmup_parameter_mismatch'),
                   ('method','atomic_sgd','blocking_warmup_parameter_mismatch'),
                   ('pivots',200,'blocking_warmup_parameter_mismatch'),
                   ('tile_size',8,'blocking_warmup_parameter_mismatch'),
                   ('status','success','blocking_warmup_was_success'),
                   ('failure_reason','memory_limit','blocking_warmup_failure_reason_mismatch'),
                   ('missing',None,'blocking_warmup_missing')]
        for field,value,kind in mutations:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temporary:
                fixture=Fixture(Path(temporary)); path=fixture.block_with_warmup(2)
                if field=='missing':
                    path.unlink()
                elif field=='failure_reason':
                    fixture.core_rows[2][field]=value
                    write_rows(fixture.main/'raw/results.jsonl',fixture.core_rows); fixture.refresh_manifest()
                else:
                    warmup=recovery.read_json(path); warmup[field]=value; write_json(path,warmup)
                _,_,meta=recovery.load_effective_cohort(fixture.main)
                self.assertEqual(meta['status'],'rejected')
                self.assertIn(kind,{error['kind'] for error in meta['errors']})
                with patch.object(audit,'active_runners',return_value=[]):
                    report=audit.audit(fixture.main,require_reboot_receipt=False)[0]
                self.assertIn(kind,{error['kind'] for error in report['errors']})


if __name__=='__main__':
    unittest.main()
