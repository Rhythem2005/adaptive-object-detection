"""Resume E after an archive-schema validation stop; never repeats a timed pass.

The frozen runner expected identical NPZ key sets. Phase1 additionally stores
COCO axis metadata. Validate those metadata as well as every common tensor.
No scoring semantics, model settings or worker/timing code changes.
"""
import datetime
import os
import subprocess
import sys
import time

import numpy as np
from pycocotools.cocoeval import COCOeval

from analysis.argoverse_hd_phase2.common import archive, dump, load, sha
from analysis.argoverse_hd_phase2.scoring import score
from .run import BASE, ROOT, BLOCKS, REFS, assert_archive_equal, verify_sources, system_status


def tensor_fidelity(current, reference, frames):
    current=archive(current);reference=archive(reference)
    common={'precision','recall','scores'}
    assert set(current)==common
    extras=set(reference)-common
    assert extras in [set(),{'category_ids','image_ids','iou_thresholds','recall_thresholds'}]
    assert_archive_equal(current,{k:reference[k] for k in common},'official COCO result tensors')
    if extras:
        params=COCOeval().params
        expected=dict(category_ids=np.arange(8,dtype=np.int64),
            image_ids=np.asarray(sorted(r['image_id'] for r in frames),dtype=np.int64),
            iou_thresholds=params.iouThrs,recall_thresholds=params.recThrs)
        assert_archive_equal({k:reference[k] for k in extras},expected,'official COCO axis metadata')


def gate():
    frames=load(ROOT/'frames.json');gt=load(ROOT/'ground_truth.json')
    for m in 'ABCDE':
        out=ROOT/'accuracy'/m
        reference=BASE/'runs/argoverse_hd_coco_phase1_v1/baseline' if m=='A' else REFS[m]/'scoring'
        expected=load(reference/'metrics.json')
        metrics=expected['official_COCO_max100'] if m=='A' else expected['COCO']
        if (out/'SCORE_COMPLETE.json').exists():
            result=load(out/'metrics.json')
        else:
            result=score(archive(ROOT/'passes'/f'01_{m}'/'preds.npz'),gt,frames,out,expected=metrics)
        assert result['COCO']==metrics
        assert result['fixed_conf25']==expected['supplementary_fixed_conf25_IoU50' if m=='A' else 'fixed_conf25']
        assert result['per_sequence']==({k:v['metrics'] for k,v in expected['per_sequence'].items()} if m=='A' else expected['per_sequence'])
        tensor_fidelity(out/'official_tensors.npz',reference/('official_tensors100.npz' if m=='A' else 'official_tensors.npz'),frames)
        dump(out/'ARCHIVE_FIDELITY.json',dict(official_metrics_exact=True,official_tensors_bytes_exact=True,
            phase1_extra_axis_metadata_exact=(m=='A'),per_sequence_metrics_exact=True,
            fixed_conf25_exact=True,reference=str(reference)))
    dump(ROOT/'FIRST_BLOCK_ACCURACY_GATE.json',dict(all_five_pass=True,new_inferences=0,
        validation='All three result tensors exact, plus all four additional Phase1 axis metadata arrays exact.'))


def main():
    verify_sources()
    assert not (ROOT/'EXECUTION_COMPLETE.json').exists()
    assert not (ROOT/'RESUME_STARTED.json').exists(),'No silent retries or repeated passes'
    for m in 'ABCDE':assert (ROOT/'passes'/f'01_{m}'/'COMPLETE.json').exists()
    for b in range(2,6):
        for m in 'ABCDE':assert not (ROOT/'passes'/f'{b:02d}_{m}').exists()
    path=BASE/'analysis/argoverse_hd_runtime_E/resume.py'
    dump(ROOT/'RESUME_STARTED.json',dict(status=system_status(),source=str(path),sha256=sha(path),
        no_completed_pass_repeated=True,reason='Phase1 NPZ contains four axis-metadata fields; explicitly validate these, preserve exact common tensors.',
        frozen_worker_unchanged=True))
    before=time.perf_counter();gate()
    dump(ROOT/'scoring_cost.json',dict(remaining_gate_wall_s=time.perf_counter()-before,
        initial_A_scoring_wall_s=load(ROOT/'accuracy/A/metrics.json')['scoring_wall_seconds'],
        all_five_scoring_wall_s=sum(load(ROOT/'accuracy'/m/'metrics.json')['scoring_wall_seconds'] for m in 'ABCDE')))
    print('FIRST BLOCK OFFICIAL FIDELITY GATE PASSED INCLUDING PHASE1 AXIS METADATA',flush=True)
    for block,order in enumerate(BLOCKS[1:],2):
        before=time.perf_counter();time.sleep(30)
        dump(ROOT/f'pause_before_block{block}.json',dict(requested_s=30,actual_s=time.perf_counter()-before,status=system_status()))
        for m in order:
            out=ROOT/'passes'/f'{block:02d}_{m}'
            before=time.perf_counter()
            command=[sys.executable,'-m','analysis.argoverse_hd_runtime_E.run','worker','--block',str(block),'--method',m]
            with (out.parent/f'{block:02d}_{m}.log').open('x') as log:
                p=subprocess.run(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:
                dump(ROOT/'RESUME_FAILED.json',dict(block=block,method=m,returncode=p.returncode))
                raise RuntimeError('Stop dependent passes; preserve failed evidence')
            dump(out/'SUBPROCESS.json',dict(wall_including_interpreter_s=time.perf_counter()-before,command=command))
            print(f'COMPLETED BLOCK {block} ARM {m}',flush=True)
    verify_sources()
    assert sha(path)==load(ROOT/'RESUME_STARTED.json')['sha256']
    start=(ROOT/'EXECUTION_STARTED.json').stat().st_mtime_ns
    dump(ROOT/'EXECUTION_COMPLETE.json',dict(wall_s=(time.time_ns()-start)/1e9,
        wall_scope='Elapsed wall-clock from first execution-start marker, includes first-block evaluation-schema interruption; filesystem timestamp to completion.',
        passes=25,measured_requests=51280,warmup_requests=75,total_requests=51355,
        all_prediction_and_schedule_parity=True,no_timed_pass_repeated=True,status=system_status()))
    print('ALL 25 E PASSES COMPLETE',flush=True)


if __name__=='__main__':main()
