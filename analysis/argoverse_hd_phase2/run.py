"""Approved A/R/B/C only. No model inference, training, D/E, or algorithm edits."""
import argparse
import csv
import importlib.abc
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import time


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','ultralytics'}:
            raise RuntimeError(f'Model library import prohibited: {fullname}')
        return None


sys.meta_path.insert(0, NoModels())
import cv2
import numpy as np
from adaptive_pipeline.evaluation.score_saved_coco import prepare_inputs
from .common import archive, config_from_dict, dump, load, sha
from .replay import replay
from .scoring import score

BASE = Path(__file__).resolve().parents[2]
PLANROOT = BASE/'analysis/argoverse_hd_phase2_plan_v1'
METHODS = ['fixed2_093','C_fullfallback','adaptive_observed','adaptive_quality']


def verify_evidence():
    for file, digest in load(PLANROOT/'source_sha256.json').items():
        if sha(file) != digest:
            raise ValueError(f'Approved evidence/source changed: {file}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['R','A','B','C'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    proposed = load(PLANROOT/'plan.json')
    out = Path(args.out).resolve()
    if out != Path(proposed['fresh_execution_root']):
        raise ValueError('Output must match the approved fresh output root')
    verify_evidence()
    if importlib.metadata.version('pycocotools') != '2.0.11':
        raise ValueError('Phase1 scoring dependency changed')
    benchmark = Path(proposed['source_benchmark']); phase1 = Path(proposed['verified_scoring'])
    gt, frameplan, _, evidence = prepare_inputs(proposed['dataset'], benchmark)
    source_manifest = load(benchmark/'baseline/input_manifest.json')['frames']
    for frame, original in zip(frameplan, source_manifest):
        frame['image_sha256'] = original['image_sha256']
    if not out.exists():
        if args.stage != 'R':
            raise RuntimeError('Experiment R must initialize the run')
        out.mkdir()
        dump(out/'execution_plan.json', {'approved_plan':proposed, 'authorization':'User approved A,R,B,C; R mandatory prerequisite; no D/E/inference/training/Phase3',
            'backend_policy':'original CPU NumPy/OpenCV/SciPy; GPU proposal cancelled; no acceleration port',
            'argv':sys.argv,'created_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        dump(out/'evidence.json', evidence)
        dump(out/'frameplan.json', frameplan)
        dump(out/'runtime.json', {'python':platform.python_version(),'platform':platform.platform(),
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','opencv-python','pycocotools']},
            'cv_threads':cv2.getNumThreads(), 'OpenCL_enabled':cv2.ocl.useOpenCL(),
            'model_import_guard':True, 'device':'CPU, inference not executed'})
        (out/'opencv_build.txt').write_text(cv2.getBuildInformation())
        shutil.copytree(BASE/'extracted/adaptive_pipeline', out/'source_snapshot/adaptive_pipeline', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(Path(__file__).parent, out/'source_snapshot/phase2_harness', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(PLANROOT, out/'approved_plan_snapshot')
    elif load(out/'execution_plan.json')['approved_plan'] != proposed:
        raise RuntimeError('Existing output belongs to another plan')
    if args.stage != 'R' and not (out/'R_COMPLETE.json').exists():
        raise RuntimeError('Successful four-method R is required before dependent experiments')
    if (out/(args.stage+'_COMPLETE.json')).exists():
        raise FileExistsError('Stage already complete; do not rerun')
    dense = archive(benchmark/'baseline/preds.npz')
    if args.stage == 'R':
        results = {}
        for method in METHODS:
            directory = out/'R_reference_parity'/method
            cfg = config_from_dict(load(PLANROOT/f'configs/reproduce_{method}.json'))
            if not (directory/'REPLAY_COMPLETE.json').exists():
                replay(cfg, frameplan, dense, directory, reference=benchmark/method)
            else:
                if not load(directory/'REPLAY_COMPLETE.json')['reference_bitwise_parity']:
                    raise RuntimeError('Invalid reference replay')
            expected = load(phase1/method/'metrics.json')['official_COCO_max100']
            if not (directory/'scoring/SCORE_COMPLETE.json').exists():
                score(archive(directory/'preds.npz'), gt, frameplan, directory/'scoring', expected=expected)
            results[method] = load(directory/'replay_summary.json')
        dump(out/'R_COMPLETE.json', {'all_four_bitwise_prediction_decision_detail_parity':True,
            'all_four_COCO_metrics_parity':True, 'results':results, 'new_YOLO_inferences':0})
    elif args.stage == 'B':
        results = {'T0_fixed2_original':{'reuse':str(out/'R_reference_parity/fixed2_093')}}
        for name, arm in proposed['requested_initial_batch']['B_tracking_factors']['arms'].items():
            if name=='T0_fixed2_original':
                continue
            directory = out/'B_tracking_factors'/name
            if not (directory/'REPLAY_COMPLETE.json').exists():
                replay(config_from_dict(load(arm['full_config'])), frameplan, dense, directory)
            if not (directory/'scoring/SCORE_COMPLETE.json').exists():
                score(archive(directory/'preds.npz'), gt, frameplan, directory/'scoring')
            results[name] = load(directory/'scoring/metrics.json')['COCO']
        # The score-only control must leave all states and geometry unchanged.
        anchor = out/'B_tracking_factors/T1_fixed2_anchor'
        no_decay = out/'B_tracking_factors/T5_no_score_decay'
        if load(anchor/'state_hashes.json') != load(no_decay/'state_hashes.json'):
            raise AssertionError('Score-only arm changed internal tracker states')
        a,b = archive(anchor/'preds.npz'),archive(no_decay/'preds.npz')
        for key in a:
            np.testing.assert_array_equal(a[key][:,[0,1,2,3,5]], b[key][:,[0,1,2,3,5]])
        dump(out/'B_COMPLETE.json', {'results':results,'score_only_state_geometry_parity':True,'new_YOLO_inferences':0})
    elif args.stage == 'C':
        results = {}
        for cell in proposed['requested_initial_batch']['C_schedule_factors']['cells']:
            name=cell['name']; directory=out/'C_schedule_factors'/name
            forced={seq:set(v) for seq,v in load(cell['schedule'])['full_frame_indices'].items()}
            reference = benchmark/'adaptive_observed' if name=='S2_anchor_archived_anchor' else benchmark/'adaptive_quality' if name=='S7_quality_archived_quality' else None
            if not (directory/'REPLAY_COMPLETE.json').exists():
                replay(config_from_dict(load(cell['engine_config'])), frameplan, dense, directory, forced=forced, reference=reference)
            if not (directory/'scoring/SCORE_COMPLETE.json').exists():
                expected=load(phase1/reference.name/'metrics.json')['official_COCO_max100'] if reference else None
                score(archive(directory/'preds.npz'), gt, frameplan, directory/'scoring', expected=expected)
            results[name]=load(directory/'scoring/metrics.json')['COCO']
        dump(out/'C_COMPLETE.json', {'results':results,'new_YOLO_inferences':0})
    elif args.stage == 'A':
        from .paired import saved_pairing
        saved_pairing(gt, frameplan, dense, benchmark, phase1, out)
    verify_evidence()
    if 'torch' in sys.modules or 'ultralytics' in sys.modules:
        raise AssertionError('Forbidden model module loaded')
    print(f'{args.stage} COMPLETE: no model inference; original sources and evidence unchanged', flush=True)


if __name__=='__main__':
    main()
