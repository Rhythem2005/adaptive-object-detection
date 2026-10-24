"""Approved next-stage cache-only experiments; bounded, resumable, immutable outputs."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

# Imports the same fail-closed torch/ultralytics import guard as Phase 3.
from analysis.argoverse_hd_phase3.run import NoModels
import cv2
import numpy as np
from analysis.argoverse_hd_phase2.common import archive, config_from_dict, dump, load, sha
from analysis.argoverse_hd_phase2.replay import replay
from analysis.argoverse_hd_phase2.scoring import score
from analysis.argoverse_hd_phase2.analyze import operating_points
from .methods import corroborated, replay_adapters

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'runs/argoverse_hd_next_research_v1'
P3=BASE/'runs/argoverse_hd_phase3_v1'
BENCH=BASE/'runs/argoverse_hd_benchmark_v1'
CONTROLS={'R_recovery':'R_T3_recovery','R_grid3':'R_T3_recovery__grid3','R_birth10':'V1_separate_birth10'}


def verify():
    for p,h in load(ROOT/'source_sha256.json').items():
        if sha(p)!=h:raise RuntimeError('Executed source changed: '+p)
    for p,h in load(ROOT/'prior_sha256.json').items():
        if sha(p)!=h:raise RuntimeError('Prior evidence changed: '+p)
    assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules


def prepare():
    if (ROOT/'plan.json').exists():raise FileExistsError('Already prepared')
    assert importlib.metadata.version('pycocotools')=='2.0.11'
    specs={n:{'config':config_from_dict(load(P3/'arms'/old/'config.json')).to_dict(),
        'reference':str(P3/'arms'/old)} for n,old in CONTROLS.items()}
    cfg=specs['R_grid3']['config']
    plan={'authorization':'Approved proposal: <=12 scored cached passes and <=6 timing passes. No inference, D/E, training, Phase4, commit or push.',
        'controls':specs,'frames':3416,'sequences':5,'max_scored_passes':12,'max_timing_passes':6,
        'gate':{'weak_band':[.1,.25],'IoU':.3,'history':'previous FULL only, maximum two frame indices; same-class one-to-one legacy assignment',
            'criteria':'>=50% current TP50 retention, >=5 percentage-point TP/(TP+FP) increase, >=50 eligible TP50 events. Engineering screen, not significance or out-of-sample evidence.',
            'birth':'previous unmatched weak birth proposals from unrestricted-birth control',
            'output':'previous raw FULL detections score>=.1; gate only current [.1,.25) FULL rows, all other scores unchanged; output-only, tracker sees raw cache'},
        'variants':{
            'V_strict_grid3':{'config':dict(cfg,low_association_iou=.5)},
            'V_birth_grid3':{'config':dict(cfg,track_birth_conf=.1),'conditional':'birth gate'},
            'V_confirmed_grid3':{'config':dict(cfg,track_birth_conf=.1),'confirmed_births':True,'conditional':'birth gate'},
            'V_output_grid3':{'config':dict(cfg),'filter_full_weak':True,'conditional':'output gate'},
            'V_confirmed_output_grid3':{'config':dict(cfg,track_birth_conf=.1),'confirmed_births':True,'filter_full_weak':True,'conditional':'both gates'},
            'V_dense_output':{'config':dict(cfg,policy='always'),'filter_full_weak':True,'conditional':'output gate'}},
        'selection':'Report Pareto frontier of AP,AR100,TP25,FP25,FP10,CPU/wall and requests. No claim of accepted accuracy loss or universal winner. Primary timing candidate highest AP among arms with TP25>=R_grid3,FP25<=R_grid3,FP10<=R_grid3; tie fewer flow points then name. If none retain R_grid3. Diagnostic gates may stop unsupported arms.',
        'finalists':'one alternate Fixed2 phase for selected candidate; one fastpath parity pass against candidate. No new scheduling strategy.',
        'timing':'AB,BA,AB three repetitions of selected candidate versus identical fastpath; outputs/details/extended tracker states exact; includes external step wrapper, excludes decode/hash/serialization/model; same lightweight subcomponent timer wrappers inside step timing; includes their overhead, no additional image or pipeline calls',
        'new_YOLO_inferences':0,'device':'CPU NumPy/SciPy/OpenCV; COCO CPU. Model device not executed.',
        'created_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(ROOT/'plan.json',plan)
    for name in ['frameplan.json','ground_truth.json','input_evidence.json']:
        shutil.copy2(P3/name,ROOT/name)
    runtime={'python':platform.python_version(),'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','opencv-python','pycocotools']},
        'cv_threads':cv2.getNumThreads(),'OpenCL':cv2.ocl.useOpenCL(),'model_import_guard':True,'actual_device':plan['device']}
    dump(ROOT/'runtime.json',runtime);(ROOT/'opencv_build.txt').write_text(cv2.getBuildInformation())
    sources={str(p):sha(p) for directory in ['extracted/adaptive_pipeline','analysis/argoverse_hd_phase2','analysis/argoverse_hd_phase3','analysis/argoverse_hd_next_research']
        for p in (BASE/directory).rglob('*.py')}
    sources[str(BASE/'tests/test_next_research.py')]=sha(BASE/'tests/test_next_research.py')
    dump(ROOT/'source_sha256.json',sources)
    prior={}
    for phase in ['argoverse_hd_coco_phase1_v1','argoverse_hd_phase2_diagnostics_v1','argoverse_hd_phase3_v1']:
        root=BASE/'runs'/phase
        for rel,digest in load(root/'artifact_sha256.json').items():
            path=root/rel
            assert sha(path)==digest,path
            prior[str(path)]=digest
    for p,h in load(ROOT/'input_evidence.json')['source_sha256'].items():
        assert sha(p)==h,p;prior[p]=h
    for p in ['docs/COCO_PHASE1_RESULTS.md','docs/PHASE2_RESULTS.md','docs/PHASE3_RESULTS.md',
              'analysis/argoverse_hd_phase3/verify.py','analysis/argoverse_hd_phase3/report.py']:
        prior[str(BASE/p)]=sha(BASE/p)
    dump(ROOT/'prior_sha256.json',prior)
    for directory in ['extracted/adaptive_pipeline','analysis/argoverse_hd_next_research']:
        shutil.copytree(BASE/directory,ROOT/'executed_source'/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    dump(ROOT/'PREPARED.json',{'frames':3416,'prior_hashes':len(prior),'new_YOLO_inferences':0})
    print('Prepared:',ROOT,flush=True)


def run_arm(name,spec):
    verify()
    destination=ROOT/'arms'/name
    config=config_from_dict(spec['config'])
    ref=Path(spec['reference']) if spec.get('reference') else None
    frames=load(ROOT/'frameplan.json');gt=load(ROOT/'ground_truth.json')
    dense=archive(BENCH/'baseline/preds.npz')
    forced={k:set(v) for k,v in spec.get('schedule',{}).items()} or None
    if not (destination/'REPLAY_COMPLETE.json').exists():
        count=sum(1 for p in (ROOT/'arms').glob('*/REPLAY_COMPLETE.json'))
        assert count<12
        with replay_adapters(spec):replay(config,frames,dense,destination,reference=ref,forced=forced)
    if not (destination/'scoring/SCORE_COMPLETE.json').exists():
        expected=load(ref/'scoring/metrics.json')['COCO'] if ref else None
        score(archive(destination/'preds.npz'),gt,frames,destination/'scoring',expected=expected)
    if ref:
        for key in ['COCO','fixed_conf25','per_sequence']:
            assert load(destination/'scoring/metrics.json')[key]==load(ref/'scoring/metrics.json')[key]
        assert load(destination/'state_hashes.json')==load(ref/'state_hashes.json')
        with np.load(destination/'scoring/official_tensors.npz') as a,np.load(ref/'scoring/official_tensors.npz') as b:
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    if not (destination/'operating_points.json').exists():
        dump(destination/'operating_points.json',operating_points(gt,frames,destination))
    print('COMPLETE',name,flush=True)


def rates(counts):
    a=counts.get('TP',0);b=counts.get('FP',0)
    return {'TP':a,'FP':b,'precision':a/(a+b) if a+b else None,
        'ignored':counts.get('ignored',0),'capped':counts.get('capped',0),'ambiguous':counts.get('ambiguous',0)}


def evidence_gate():
    # Recorded pre-execution cohort correction; preserve the original driver snapshot.
    from .diagnostic import evidence_gate as corrected_gate
    for p,h in load(ROOT/'diagnostic_source_sha256.json').items():
        if sha(p)!=h:raise RuntimeError('Diagnostic source changed: '+p)
    return corrected_gate()


def variants():
    gates=load(ROOT/'weak_evidence_gate.json')['gates'];specs=load(ROOT/'plan.json')['variants'];executed={};skipped={}
    for name,spec in specs.items():
        condition=spec.get('conditional','')
        okay=not condition or (condition=='birth gate' and gates['birth']['passed']) or (condition=='output gate' and gates['output']['passed']) or (condition=='both gates' and gates['birth']['passed'] and gates['output']['passed'])
        if okay:run_arm(name,spec);executed[name]=spec
        else:skipped[name]='Failed pre-specified saved-evidence '+condition
    dump(ROOT/'variants_executed.json',executed);dump(ROOT/'variants_skipped.json',skipped)
    dump(ROOT/'VARIANTS_COMPLETE.json',{'executed':list(executed),'skipped':skipped})


def select():
    assert (ROOT/'VARIANTS_COMPLETE.json').exists()
    specs={k:{'config':v['config']} for k,v in load(ROOT/'plan.json')['controls'].items()}
    specs.update(load(ROOT/'variants_executed.json'))
    ref=load(ROOT/'arms/R_grid3/operating_points.json')['all']['thresholds']
    gates={};eligible=[]
    for name,spec in specs.items():
        if spec['config']['policy']!='fixed':continue
        p=load(ROOT/'arms'/name/'operating_points.json')['all']['thresholds']
        gates[name]={'TP25':p['0.25']['50']['TP']>=ref['0.25']['50']['TP'],
            'FP25':p['0.25']['50']['FP']<=ref['0.25']['50']['FP'],
            'FP10':p['0.1']['50']['FP']<=ref['0.1']['50']['FP']}
        if all(gates[name].values()):eligible.append(name)
    eligible.sort(key=lambda n:(-load(ROOT/'arms'/n/'scoring/metrics.json')['COCO']['AP'],
        load(ROOT/'arms'/n/'replay_summary.json')['total_flow_point_evaluations_before_FB'],n))
    selected=eligible[0] if eligible else 'R_grid3';base=specs[selected]
    frames=load(ROOT/'frameplan.json');schedule={}
    for seq in sorted({f['video'] for f in frames}):
        n=sum(f['video']==seq for f in frames);budget=(n+1)//2
        schedule[seq]=[0]+[1+2*j for j in range(budget-1)]
        assert len(set(schedule[seq]))==budget and max(schedule[seq])<n
    phase=dict(base,config=dict(base['config'],fixed_k=len(frames)+1),schedule=schedule)
    fast=dict(base,config=dict(base['config'],fixed_signal_fastpath=True),reference=str(ROOT/'arms'/selected))
    final={'F_phase1':phase,'F_fastpath':fast}
    dump(ROOT/'selection.json',{'gates':gates,'eligible':eligible,'selected':selected,'reason':'Pre-specified TP/FP gate then AP/flow points; not a general all-metric winner.'})
    dump(ROOT/'finalist_specs.json',final)
    for name,spec in final.items():run_arm(name,spec)
    dump(ROOT/'FINALISTS_COMPLETE.json',{'arms':list(final),'new_YOLO_inferences':0})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,
        choices=['prepare','controls','gate','variants','finalists','timing'])
    stage=parser.parse_args().stage
    if stage=='prepare':prepare();return
    verify()
    if stage=='controls':
        if (ROOT/'CONTROLS_COMPLETE.json').exists():raise FileExistsError('Controls complete')
        for name,spec in load(ROOT/'plan.json')['controls'].items():run_arm(name,spec)
        dump(ROOT/'CONTROLS_COMPLETE.json',{'three_controls_exact':True,'new_YOLO_inferences':0})
    elif stage=='gate':evidence_gate()
    elif stage=='variants':variants()
    elif stage=='finalists':select()
    else:
        from .timing import run_timing
        run_timing(ROOT)
    verify()


if __name__=='__main__':main()
