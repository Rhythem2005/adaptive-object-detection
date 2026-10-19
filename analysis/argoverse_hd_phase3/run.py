"""Approved Phase 3 execution: cached observations only, at most22 replays."""
import argparse
import csv
import importlib.abc
import importlib.metadata
from pathlib import Path
import platform
import shutil
import sys
import time


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'ultralytics'}:
            raise RuntimeError('New model inference/import is prohibited in Phase3')
        return None


sys.meta_path.insert(0, NoModels())
import cv2
import numpy as np
from adaptive_pipeline.evaluation.score_saved_coco import prepare_inputs
from analysis.argoverse_hd_phase2.common import archive, config_from_dict, dump, load, sha
from analysis.argoverse_hd_phase2.replay import replay
from analysis.argoverse_hd_phase2.scoring import score
from analysis.argoverse_hd_phase2.analyze import operating_points
from .observe import observations

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE/'runs/argoverse_hd_phase3_v1'
P2 = BASE/'runs/argoverse_hd_phase2_diagnostics_v1'
BENCH = BASE/'runs/argoverse_hd_benchmark_v1'
P1 = BASE/'runs/argoverse_hd_coco_phase1_v1'
REFS = {'R_T1_anchor':'T1_fixed2_anchor', 'R_T3_recovery':'T3_existing_track_recovery',
        'R_T4_shared10':'T4_low_birth_threshold'}
VARIANTS = {
    'V1_separate_birth10':{'track_birth_conf':.10},
    'V2_active_recovery':{'low_association_visible_only':True},
    'V3_low_iou50':{'low_association_iou':.50},
    'V4_active_low_iou50':{'low_association_visible_only':True, 'low_association_iou':.50},
    'V5_threshold_assignment':{'threshold_aware_assignment':True},
}


def source_hashes():
    roots=[BASE/'extracted/adaptive_pipeline', Path(__file__).parent,
           BASE/'analysis/argoverse_hd_phase2']
    return {str(p):sha(p) for root in roots for p in root.rglob('*.py') if '__pycache__' not in p.parts}


def verify_sources():
    for path,digest in load(ROOT/'execution_sources.json').items():
        if sha(path)!=digest:raise ValueError(f'Executed source changed: {path}')
    for path,digest in load(ROOT/'prior_evidence_sha256.json').items():
        if sha(path)!=digest:raise ValueError(f'Prior evidence changed: {path}')


def prepare():
    if (ROOT/'execution_plan.json').exists():raise FileExistsError('Preparation already complete')
    if importlib.metadata.version('pycocotools')!='2.0.11':raise ValueError('COCO dependency changed')
    gt, frames, _, evidence=prepare_inputs(BASE/'data/argoverse_hd', BENCH)
    original=load(BENCH/'baseline/input_manifest.json')['frames']
    for f,row in zip(frames,original):f['image_sha256']=row['image_sha256']
    controls={name:{'config':config_from_dict(load(P2/'B_tracking_factors'/ref/'config.json')).to_dict(),
                    'reference':str(P2/'B_tracking_factors'/ref)} for name,ref in REFS.items()}
    base=controls['R_T3_recovery']['config']
    variants={name:{'config':dict(base,**delta),'overrides':delta} for name,delta in VARIANTS.items()}
    plan={'authorization':'User approved bounded Phase3:3 controls+5 variants; conditional efficiency8+timing6, max22. No new inference/D/E/Phase4/training/commit/push.',
        'created_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'frames':len(frames), 'controls':controls, 'variants':variants,
        'efficiency_variants':{'grid3':{'flow_grid':3,'flow_min_points':4},
            'fixed2_phase1':{'interval':2,'phase':1},'fixed3_phase0':{'interval':3,'phase':0},
            'fixed3_phase1':{'interval':3,'phase':1}},
        'selection':'Conservative candidate must retain T3 AP,AR100,macroP25 and TP25; not increase FP25 or FP10 or mean tracks. Highest AP among eligible; then fewer tracks; deterministic name tie-break. If none, only T3 efficiency controls run.',
        'no_accepted_accuracy_loss_margin':True,
        'timing':'3 interleaved runs per finalist; no observation wrappers; validate bitwise outputs against scored arm; no model or model warmup.',
        'initial_scored_replays':8,'max_scored_replays':16,'max_total_replays':22,
        'detector':'original mapped stock YOLOv8n cache; only requested FULL may access it',
        'new_YOLO_inferences':0,'device':'CPU NumPy/SciPy/OpenCV; COCO CPU; no acceleration port'}
    dump(ROOT/'execution_plan.json',plan);dump(ROOT/'frameplan.json',frames)
    dump(ROOT/'ground_truth.json',gt);dump(ROOT/'input_evidence.json',evidence)
    dump(ROOT/'runtime.json',{'python':platform.python_version(),'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','opencv-python','pycocotools']},
        'cv_threads':cv2.getNumThreads(),'OpenCL':cv2.ocl.useOpenCL(), 'device':plan['device'],
        'model_import_guard':True})
    (ROOT/'opencv_build.txt').write_text(cv2.getBuildInformation())
    shutil.copytree(BASE/'extracted/adaptive_pipeline',ROOT/'executed_source/adaptive_pipeline',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(Path(__file__).parent,ROOT/'executed_source/phase3',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    dump(ROOT/'execution_sources.json',source_hashes())
    dump(ROOT/'PREPARED.json',{'frames':len(frames),'new_YOLO_inferences':0})
    print('PREPARED:8 initial full-sequence replays; no model inference',flush=True)


def run_arm(name, specification, gt, frames, dense):
    directory=ROOT/'arms'/name
    cfg=config_from_dict(specification['config'])
    reference=Path(specification['reference']) if specification.get('reference') else None
    forced={seq:set(indices) for seq,indices in specification.get('schedule',{}).items()} or None
    if not (directory/'REPLAY_COMPLETE.json').exists():
        with observations():replay(cfg,frames,dense,directory,reference=reference,forced=forced)
    if not (directory/'scoring/SCORE_COMPLETE.json').exists():
        expected=load(reference/'scoring/metrics.json')['COCO'] if reference else None
        score(archive(directory/'preds.npz'),gt,frames,directory/'scoring',expected=expected)
    result=load(directory/'scoring/metrics.json')
    if reference:
        old=load(reference/'scoring/metrics.json')
        assert result['COCO']==old['COCO'] and result['per_sequence']==old['per_sequence']
        assert load(directory/'state_hashes.json')==load(reference/'state_hashes.json')
        with np.load(reference/'scoring/official_tensors.npz') as a, np.load(directory/'scoring/official_tensors.npz') as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    if not (directory/'operating_points.json').exists():
        dump(directory/'operating_points.json',operating_points(gt,frames,directory))
    return result


def result(name):
    p=ROOT/'arms'/name
    return {'metrics':load(p/'scoring/metrics.json'),'cost':load(p/'replay_summary.json'),
            'points':load(p/'operating_points.json')}


def conservative_gate(candidate, reference):
    m,r=candidate['metrics'],reference['metrics']
    cp,rp=candidate['points']['all']['thresholds'],reference['points']['all']['thresholds']
    return {'AP':m['COCO']['AP']+1e-14>=r['COCO']['AP'],
        'AR100':m['COCO']['AR100']+1e-14>=r['COCO']['AR100'],
        'macro_precision25':m['fixed_conf25']['macro_precision']+1e-14>=r['fixed_conf25']['macro_precision'],
        'TP25':cp['0.25']['50']['TP']>=rp['0.25']['50']['TP'],
        'FP25':cp['0.25']['50']['FP']<=rp['0.25']['50']['FP'],
        'FP10':cp['0.1']['50']['FP']<=rp['0.1']['50']['FP'],
        'mean_tracks':candidate['cost']['mean_post_frame_tracks']<=reference['cost']['mean_post_frame_tracks']+1e-14}


def select():
    if not (ROOT/'VARIANTS_COMPLETE.json').exists():raise RuntimeError('Initial variants incomplete')
    ref=result('R_T3_recovery');gates={n:conservative_gate(result(n),ref) for n in VARIANTS}
    eligible=[n for n,g in gates.items() if all(g.values())]
    eligible.sort(key=lambda n:(-result(n)['metrics']['COCO']['AP'],result(n)['cost']['mean_post_frame_tracks'],n))
    selected=eligible[0] if eligible else None
    dump(ROOT/'selection.json',{'gates':gates,'eligible':eligible,'selected_confidence_candidate':selected,
        'note':'No AP-loss or precision-loss tolerance assumed. This is a development screen, not a significance/generalization claim.'})
    frames=load(ROOT/'frameplan.json');specs={}
    for engine in ['R_T3_recovery']+([selected] if selected else []):
        base=load(ROOT/'arms'/engine/'config.json')
        for label,settings in load(ROOT/'execution_plan.json')['efficiency_variants'].items():
            cfg=dict(base);spec={'parent':engine,'config':cfg}
            if label=='grid3':cfg.update(settings)
            else:
                k,phase=settings['interval'],settings['phase'];schedule={}
                for seq in sorted({f['video'] for f in frames}):
                    n=sum(f['video']==seq for f in frames);budget=(n+k-1)//k
                    schedule[seq]=list(range(0,n,k)) if phase==0 else [0]+[phase+j*k for j in range(budget-1)]
                    assert len(schedule[seq])==budget and len(set(schedule[seq]))==budget and max(schedule[seq])<n
                spec['schedule']=schedule
                # The harness knows the prescribed FULL frames. A large native
                # interval prevents an extra implicit refresh near the end;
                # the harness early-FULL hook preserves fixed-path behaviour.
                cfg['fixed_k']=len(frames)+1
                spec['interval']=k;spec['phase']=phase
            specs[engine+'__'+label]=spec
    dump(ROOT/'efficiency_plan.json',specs)
    print('Confidence candidate:',selected,'; conditional efficiency replays:',len(specs),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['prepare','controls','variants','select','efficiency','timing'])
    args=p.parse_args()
    if args.stage=='prepare':prepare();return
    verify_sources()
    if args.stage=='select':select();return
    if args.stage=='timing':
        from .timing import run_timing
        run_timing(ROOT);return
    gt=load(ROOT/'ground_truth.json');frames=load(ROOT/'frameplan.json');dense=archive(BENCH/'baseline/preds.npz')
    plan=load(ROOT/'execution_plan.json')
    if args.stage=='controls':specs=plan['controls'];marker='CONTROLS_COMPLETE.json'
    elif args.stage=='variants':
        if not (ROOT/'CONTROLS_COMPLETE.json').exists():raise RuntimeError('Controls must reproduce before variants')
        specs=plan['variants'];marker='VARIANTS_COMPLETE.json'
    else:
        if not (ROOT/'selection.json').exists():raise RuntimeError('Initial screen required before efficiency checks')
        specs=load(ROOT/'efficiency_plan.json');marker='EFFICIENCY_COMPLETE.json'
    if (ROOT/marker).exists():raise FileExistsError('Completed stage cannot be rerun')
    for name,spec in specs.items():run_arm(name,spec,gt,frames,dense)
    verify_sources()
    assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
    dump(ROOT/marker,{'arms':list(specs),'new_YOLO_inferences':0})
    print(args.stage,'COMPLETE',flush=True)


if __name__=='__main__':main()
