"""Final audit: archived evidence, exact controls, bounded scope, synthetic tests."""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from analysis.argoverse_hd_phase2.common import archive, dump, load, sha
from .run import BASE, ROOT, BENCH, verify


def equal(left,right):
    a,b=archive(left),archive(right);assert a.keys()==b.keys()
    for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)


def main():
    if (ROOT/'COMPLETE.json').exists():raise FileExistsError('Audit complete')
    verify()
    assert (BASE/'docs/NEXT_RESEARCH_RESULTS.md').read_bytes()==(ROOT/'report.md').read_bytes()
    for p,h in load(ROOT/'diagnostic_source_sha256.json').items():assert sha(p)==h,p
    for stage in ['PREPARED','CONTROLS_COMPLETE','VARIANTS_COMPLETE','FINALISTS_COMPLETE','TIMING_COMPLETE','ANALYSIS_COMPLETE']:
        assert (ROOT/(stage+'.json')).exists(),stage
    plan=load(ROOT/'plan.json');specs=dict(plan['controls'])
    specs.update(load(ROOT/'variants_executed.json'));specs.update(load(ROOT/'finalist_specs.json'))
    actual={p.name for p in (ROOT/'arms').iterdir() if p.is_dir()}
    assert actual==set(specs) and len(specs)<=12
    frames=load(ROOT/'frameplan.json');assert len(frames)==3416
    dense=archive(BENCH/'baseline/preds.npz')
    dense_matches=archive(BASE/'runs/argoverse_hd_coco_phase1_v1/baseline/official_matches100.npz')
    assert dense_matches['n_pred'].tolist()==[len(dense[f['key']]) for f in frames]
    np.testing.assert_array_equal(dense_matches['conf'],np.concatenate([dense[f['key']][:,4].astype(float) for f in frames]))
    lookups=0
    for name,spec in specs.items():
        path=ROOT/'arms'/name;cfg=load(path/'config.json')
        assert cfg==spec['config'],name
        complete=load(path/'REPLAY_COMPLETE.json')
        assert complete['new_YOLO_inferences']==0 and complete['frames']==3416
        assert complete['predictions_sha256']==sha(path/'preds.npz')
        predictions=archive(path/'preds.npz');assert predictions.keys()==dense.keys()
        with (path/'frames.csv').open() as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==3416
        calls=0
        for f,row in zip(frames,rows):
            assert int(row['image_id'])==f['image_id'] and row['video']==f['video']
            assert row['decision'] in ['FULL','SKIP'] and int(row['n_crops'])==0 and float(row['t_detect_ms'])==0
            if spec.get('schedule'):
                assert (row['decision']=='FULL')==(f['frame_index'] in spec['schedule'][f['video']])
                if row['decision']=='FULL':assert int(row['flow_called'])==0
            if row['decision']=='FULL':
                calls+=1
                if not spec.get('filter_full_weak'):np.testing.assert_array_equal(predictions[f['key']],dense[f['key']])
            a=predictions[f['key']]
            assert a.ndim==2 and a.shape[1]==6 and np.isfinite(a).all()
            assert ((a[:,4]>=0)&(a[:,4]<=1)).all()
            assert set(a[:,5].astype(int)).issubset(set(range(8)))
        assert calls==load(path/'replay_summary.json')['simulated_FULL_requests']
        assert calls==(3416 if cfg['policy']=='always' else 1710)
        lookups+=calls
        reference=Path(spec['reference']) if spec.get('reference') else None
        if reference:
            for filename in ['preds.npz','prediction_details.npz']:equal(path/filename,reference/filename)
            assert load(path/'state_hashes.json')==load(reference/'state_hashes.json')
            equal(path/'scoring/official_tensors.npz',reference/'scoring/official_tensors.npz')
            for key in ['COCO','fixed_conf25','per_sequence']:
                assert load(path/'scoring/metrics.json')[key]==load(reference/'scoring/metrics.json')[key]
    gate=load(ROOT/'weak_evidence_gate.json')
    for group in ['birth','output']:
        for label in ['TP','FP','ignored','capped','ambiguous']:
            assert gate['counts'][group+'_all'][label]==gate['counts'][group+'_retained'][label]+gate['counts'][group+'_rejected'][label]
        for label in ['TP','FP','ignored','capped','ambiguous']:
            assert gate['counts'][group+'_all'][label]==sum(v[group+'_all'][label] for v in gate['per_sequence'].values())
    for name in load(ROOT/'variants_skipped.json'):
        condition=plan['variants'][name]['conditional']
        assert ('birth' in condition and not gate['gates']['birth']['passed']) or ('output' in condition and not gate['gates']['output']['passed']) or condition=='both gates'
        assert name not in actual
    timing=load(ROOT/'TIMING_COMPLETE.json');assert len(timing['runs'])==6
    assert timing['all_predictions_details_extended_tracker_states_exact']
    for run in timing['runs']:
        assert run['frames']==3416 and run['new_YOLO_inferences']==0
        lookups+=run['simulated_FULL_requests']
    selected=load(ROOT/'selection.json')['selected']
    before=load(ROOT/'arms'/selected/'config.json');after=load(ROOT/'arms/F_fastpath/config.json')
    assert {k for k in before if before[k]!=after[k]}=={'fixed_signal_fastpath'}
    assert after['fixed_signal_fastpath'] and not before['fixed_signal_fastpath']
    # Preserve the pre-existing dirty files and all other production files.
    changed=[]
    previous=ROOT/'pre_execution/adaptive_pipeline'
    for p in previous.rglob('*.py'):
        dest=BASE/'extracted/adaptive_pipeline'/p.relative_to(previous)
        if sha(p)!=sha(dest):changed.append(str(p.relative_to(previous)))
    assert sorted(changed)==['pipeline.py','shared/config.py'],changed
    for name in ['analysis/argoverse_hd_phase3/verify.py','analysis/argoverse_hd_phase3/report.py','docs/PHASE3_RESULTS.md']:
        assert sha(BASE/name)==sha(ROOT/'pre_execution'/name)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE)==(ROOT/'pre_execution/git_head.txt').read_bytes()
    original=(ROOT/'executed_source/analysis/argoverse_hd_next_research/run_v1_before_cohort_amendment.py').read_text()
    current=(BASE/'analysis/argoverse_hd_next_research/run.py').read_text()
    assert original.split('def evidence_gate():')[0]==current.split('def evidence_gate():')[0]
    assert original.split('\n\ndef variants():',1)[1]==current.split('\n\ndef variants():',1)[1]
    assert load(ROOT/'source_sha256_v1.json')[str(BASE/'analysis/argoverse_hd_next_research/run.py')]==sha(ROOT/'executed_source/analysis/argoverse_hd_next_research/run_v1_before_cohort_amendment.py')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-q'],cwd=BASE,
        env=dict(os.environ,PYTHONPATH='extracted:.'),capture_output=True,text=True)
    (ROOT/'synthetic_tests.txt').write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0,tests.stdout+tests.stderr
    snapshot=ROOT/'final_source';shutil.copytree(BASE/'analysis/argoverse_hd_next_research',snapshot/'analysis',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(BASE/'tests/test_next_research.py',snapshot/'test_next_research.py')
    dump(ROOT/'analysis_source_sha256.json',{str(p):sha(p) for p in (BASE/'analysis/argoverse_hd_next_research').glob('*.py')})
    for command,name in [(['git','status','--short'],'final_git_status.txt'),(['git','diff','--binary'],'final_working_tree.diff')]:
        (ROOT/name).write_bytes(subprocess.check_output(command,cwd=BASE))
    steps=(len(specs)+len(timing['runs']))*3416
    costs={'scored_replay_stage_seconds':sum(load(ROOT/'arms'/n/'replay_summary.json')['stage_wall_seconds_including_decode_hash_diagnostics_IO'] for n in specs),
        'scoring_stage_seconds':sum(load(ROOT/'arms'/n/'scoring/metrics.json')['scoring_wall_seconds'] for n in specs),
        'timing_stage_seconds':sum(r['stage_wall_seconds_with_decode_hash_validation_IO'] for r in timing['runs']),
        'note':'Stage accounting sums include stated I/O and instrumentation; exclude implementation, diagnostics, tests, analysis, guard hashing and reporting. Not task elapsed time or end-to-end runtime.'}
    dump(ROOT/'execution_costs.json',costs)
    result={'three_controls_bitwise_predictions_details_tracker_states_official_tensors':True,
        'fastpath_bitwise_predictions_details_tracker_states_official_tensors':True,
        'six_timing_replays_exact_scored_outputs_and_states':True,
        'prior_evidence_hashes_unchanged':len(load(ROOT/'prior_sha256.json')),
        'preserved_pre_existing_dirty_files':True,'unchanged_Git_HEAD':True,
        'changed_production_files':changed,'production_defaults_unchanged':True,
        'source_revision_confined_to_unexecuted_diagnostic_function':True,
        'all_frames_in_order_sequence_resets':True,'all_unfiltered_FULL_outputs_equal_bound_cache':True,
        'scored_replays':len(specs),'timing_replays':len(timing['runs']),'frame_steps':steps,
        'simulated_FULL_cache_lookups':lookups,'new_YOLO_inferences':0,
        'conditional_arms_skipped':load(ROOT/'variants_skipped.json'),
        'D_E_Phase4_training_commit_push_executed':False,'synthetic_tests_passed':True,
        'device':load(ROOT/'runtime.json'),'execution_costs':costs}
    dump(ROOT/'final_verification.json',result)
    dump(ROOT/'COMPLETE.json',{'completed_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scored_replays':len(specs),'timing_replays':6,'new_YOLO_inferences':0})
    hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file()}
    dump(ROOT/'artifact_sha256.json',hashes)
    print('FINAL AUDIT PASSED',len(specs),'scored + 6 timing;',steps,'steps; zero inference')


if __name__=='__main__':main()
