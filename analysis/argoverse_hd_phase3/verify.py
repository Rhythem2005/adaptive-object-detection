"""Audit approved Phase 3 results and preserved prior evidence; no replay."""
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from analysis.argoverse_hd_phase2.common import archive, dump, load, sha
from .run import BASE, ROOT, P2, BENCH, P1, REFS, verify_sources


def equal_archives(left,right):
    a,b=archive(left),archive(right)
    assert a.keys()==b.keys()
    for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)


def main():
    if (ROOT/'COMPLETE.json').exists():raise FileExistsError('Audit already complete')
    verify_sources()
    for stage in ['CONTROLS','VARIANTS','EFFICIENCY','TIMING','ANALYSIS']:
        assert (ROOT/(stage+'_COMPLETE.json')).exists(),stage
    assert (BASE/'docs/PHASE3_RESULTS.md').read_bytes()==(ROOT/'report.md').read_bytes()
    original=load(ROOT/'source_before_sha256.json')
    changed=[p for p,d in original.items() if sha(p)!=d]
    allowed=[str(BASE/'extracted/adaptive_pipeline'/p) for p in ['tracking/tracker.py','shared/config.py']]
    assert set(changed)==set(allowed),changed
    # Original Phase1/2 caches, reports and executed snapshots are immutable.
    prior_count=0
    for path,digest in load(ROOT/'prior_evidence_sha256.json').items():assert sha(path)==digest,path
    for rel,digest in load(P2/'artifact_sha256.json').items():
        assert sha(P2/rel)==digest,rel
        prior_count+=1
    for path,digest in load(BASE/'analysis/argoverse_hd_phase2_plan_v1/source_sha256.json').items():
        if path not in allowed:assert sha(path)==digest,path
    for name,ref in REFS.items():
        new,old=ROOT/'arms'/name,P2/'B_tracking_factors'/ref
        for filename in ['preds.npz','prediction_details.npz']:equal_archives(new/filename,old/filename)
        assert load(new/'state_hashes.json')==load(old/'state_hashes.json')
        a,b=load(new/'scoring/metrics.json'),load(old/'scoring/metrics.json')
        for key in ['COCO','fixed_conf25','per_sequence']:assert a[key]==b[key],(name,key)
        equal_archives(new/'scoring/official_tensors.npz',old/'scoring/official_tensors.npz')
    plan=load(ROOT/'execution_plan.json')
    specs=dict(plan['controls'],**plan['variants'],**load(ROOT/'efficiency_plan.json'))
    actual={p.name for p in (ROOT/'arms').iterdir() if p.is_dir()}
    assert actual==set(specs)
    timing=load(ROOT/'TIMING_COMPLETE.json');assert len(timing['runs'])==6
    assert len(specs)<=16 and len(specs)+len(timing['runs'])<=22
    frames=load(ROOT/'frameplan.json');assert len(frames)==3416
    dense=archive(BENCH/'baseline/preds.npz');full_lookups=0
    for name,spec in specs.items():
        directory=ROOT/'arms'/name
        cfg=load(directory/'config.json');assert cfg==spec['config']
        c=load(directory/'REPLAY_COMPLETE.json')
        assert c['new_YOLO_inferences']==0 and c['frames']==3416
        assert sha(directory/'preds.npz')==c['predictions_sha256']
        predictions=archive(directory/'preds.npz');assert predictions.keys()==dense.keys()
        with (directory/'frames.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==len(frames)
        for info,row in zip(frames,rows):
            assert row['video']==info['video'] and int(row['image_id'])==info['image_id']
            assert row['decision'] in ['FULL','SKIP'] and int(row['n_crops'])==0
            assert float(row['t_detect_ms'])==0
            if spec.get('schedule'):
                assert (row['decision']=='FULL')==(info['frame_index'] in spec['schedule'][info['video']])
                if row['decision']=='FULL':assert int(row['flow_called'])==0
            if row['decision']=='FULL':
                np.testing.assert_array_equal(predictions[info['key']],dense[info['key']])
                full_lookups+=1
        if not spec.get('schedule'):
            assert sum(row['decision']=='FULL' for row in rows)==1710
        assert (directory/'scoring/SCORE_COMPLETE.json').exists()
    for run in timing['runs']:
        assert run['new_YOLO_inferences']==0 and run['frames']==3416
        full_lookups+=run['simulated_FULL_requests']
    assert timing['bitwise_prediction_detail_and_state_parity']

    # Synthetic checks are the only executions in this final audit.
    import os
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-q'],cwd=BASE,
        env=dict(os.environ,PYTHONPATH='extracted:.'),capture_output=True,text=True)
    (ROOT/'synthetic_tests.txt').write_text(test.stdout+test.stderr)
    assert test.returncode==0,test.stdout+test.stderr
    snapshot=ROOT/'final_source';snapshot.mkdir()
    shutil.copytree(Path(__file__).parent,snapshot/'phase3',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(BASE/'tests/test_phase3_tracking.py',snapshot/'test_phase3_tracking.py')
    for path in allowed:
        rel=Path(path).relative_to(BASE/'extracted/adaptive_pipeline');dest=snapshot/'adaptive_pipeline'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    diff=[]
    for path in allowed:
        rel=Path(path).relative_to(BASE/'extracted/adaptive_pipeline')
        r=subprocess.run(['git','diff','--no-index','--',str(ROOT/'source_before/adaptive_pipeline'/rel),path],capture_output=True)
        assert r.returncode in [0,1];diff.append(r.stdout)
    (ROOT/'phase3_changes.patch').write_bytes(b'\n'.join(diff))
    for cmd,name in [(['git','status','--short'],'final_git_status.txt'),(['git','diff','--binary'],'final_working_tree.diff')]:
        (ROOT/name).write_bytes(subprocess.run(cmd,cwd=BASE,capture_output=True,check=True).stdout)
    verification={'three_controls_bitwise_predictions_details_states_official_tensors':True,
        'three_controls_aggregate_per_class_per_sequence_operating_metrics_exact':True,
        'prior_Phase2_artifact_hashes_unchanged':prior_count,
        'other_original_production_files_unchanged':True,'intentionally_changed_production_files':changed,
        'all_FULL_predictions_equal_requested_bound_cache':True,'all_forced_schedules_followed':True,
        'fixed_phase_FULL_flow_bypassed':True,'timing_repeats_bitwise_output_detail_state_parity':True,
        'scored_replays':len(specs),'timing_replays':len(timing['runs']),
        'frame_steps':(len(specs)+len(timing['runs']))*3416,
        'simulated_FULL_cache_lookups':full_lookups,'new_YOLO_inferences':0,
        'D_E_Phase4_training_executed':False,'commit_or_push':False,
        'device':load(ROOT/'runtime.json'),'synthetic_tests_passed':True}
    dump(ROOT/'final_verification.json',verification)
    dump(ROOT/'COMPLETE.json',{'approved_Phase3_complete':True,'new_YOLO_inferences':0,
        'completed_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
    dump(ROOT/'artifact_sha256.json',hashes)
    print(f'FINAL AUDIT PASSED:{len(specs)} scored+6 timing replays; {prior_count} prior hashes unchanged',flush=True)


if __name__=='__main__':main()
