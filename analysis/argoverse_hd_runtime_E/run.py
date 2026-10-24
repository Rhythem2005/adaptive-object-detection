"""Approved E: sequential live CPU timing, immutable references, no algorithm changes.

Run as a module with PYTHONPATH=extracted:. .venv/bin/python -m ...run STAGE.
Every worker is a fresh process. Accuracy scoring never overlaps timing.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter

import cv2
import numpy as np

from analysis.argoverse_hd_phase2.common import archive, config_from_dict, dump, load, sha

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / 'runs/argoverse_hd_real_runtime_E_v4'
LATEST = BASE / 'runs/argoverse_hd_next_research_v1'
BLOCKS = ['ABDCE', 'BCEDA', 'CDAEB', 'DEBAC', 'EACBD']
REFS = {
    'A': BASE/'runs/argoverse_hd_benchmark_v1/baseline',
    'B': LATEST/'arms/F_fastpath',
    'C': LATEST/'arms/R_grid3',
    'D': BASE/'runs/argoverse_hd_phase3_v1/arms/V1_separate_birth10',
    'E': LATEST/'arms/V_strict_grid3',
}


def assert_archive_equal(actual, expected, label):
    if set(actual) != set(expected):
        raise AssertionError(f'{label}: frame/key mismatch')
    for key in expected:
        a, b = actual[key], expected[key]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise AssertionError(f'{label}: byte/dtype/shape mismatch at {key}')


def hash_inputs(frames):
    for row in frames:
        if sha(row['image']) != row['image_sha256']:
            raise AssertionError('Image hash mismatch: ' + row['key'])


def verify_sources():
    for name in ['execution_sources.json', 'prior_evidence.json']:
        for path, digest in load(ROOT/name).items():
            if sha(path) != digest:
                raise AssertionError('Changed protected file: ' + path)


def system_status():
    result = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    for args, key in [(['pmset', '-g', 'batt'], 'power'), (['pmset', '-g', 'therm'], 'thermal')]:
        p = subprocess.run(args, capture_output=True, text=True)
        result[key] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    return result


def peak_mib():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**2 if sys.platform == 'darwin' else 1024)


class ForwardCounter:
    """Passive root hook; closure stays attached to the counter across deepcopy."""
    def __init__(self, model):
        self.count = 0
        self.shapes = Counter()
        def capture(module, args):
            value = args[0]
            self.count += 1
            self.shapes[(tuple(value.shape), str(value.dtype), str(value.device))] += 1
        self.handle = model.register_forward_pre_hook(capture)

    def summary(self):
        return [dict(shape=list(s), dtype=d, device=v, count=n)
                for (s, d, v), n in self.shapes.items()]


def full_frame_warmup(detector, frame):
    # The production warmup also calls detect_batch on ROIs: deliberately unused.
    for _ in range(3):
        detector.detect(frame)


def validate_predictor(predictor):
    assert str(predictor.device)=='cpu' and not predictor.model.fp16
    # Installed 8.4.149 replaced args.half with quantize. Actual backend/tensor
    # dtype is checked separately; do not require a removed configuration key.
    for key, value in dict(imgsz=640, conf=.001, iou=.7, max_det=300,
                           rect=True, augment=False, agnostic_nms=False,
                           classes=None, quantize=None).items():
        assert getattr(predictor.args,key)==value,(key,getattr(predictor.args,key))


def timed_step(pipe, row, read=cv2.imread):
    """Nested scopes stop at output readiness, before buffering/serialization."""
    w0, c0 = time.perf_counter_ns(), time.process_time_ns()
    frame = read(row['image'])
    if frame is None or frame.shape != (1200, 1920, 3) or frame.dtype != np.uint8:
        raise ValueError('Invalid decoded image: ' + row['key'])
    w1, c1 = time.perf_counter_ns(), time.process_time_ns()
    pred, rec = pipe.step(frame, row['frame_index'])
    c2, w2 = time.process_time_ns(), time.perf_counter_ns()
    rec = dict(rec)
    rec.update(key=row['key'], video=row['video'], image_id=row['image_id'],
        decode_ms=(w1-w0)/1e6, decode_cpu_ms=(c1-c0)/1e6,
        step_wall_ms=(w2-w1)/1e6, step_cpu_ms=(c2-c1)/1e6,
        decode_inclusive_ms=(w2-w0)/1e6, decode_inclusive_cpu_ms=(c2-c0)/1e6)
    return pred, pipe.last_prediction_details, rec


def prepare():
    ROOT.mkdir(exist_ok=False)
    snapshot = ROOT/'pre_execution'
    snapshot.mkdir()
    for cmd, name in [(['git','status','--short'], 'git_status.txt'),
                       (['git','diff','--binary'], 'git_diff.patch'),
                       (['git','rev-parse','HEAD'], 'git_head.txt')]:
        (snapshot/name).write_bytes(subprocess.check_output(cmd, cwd=BASE))
    for folder in ['extracted/adaptive_pipeline', 'analysis/argoverse_hd_runtime_E']:
        shutil.copytree(BASE/folder, snapshot/folder,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    frames = load(LATEST/'frameplan.json')
    assert len(frames) == 3416 and len({r['key'] for r in frames}) == 3416
    counts = Counter(r['video'] for r in frames)
    assert list(counts.values()) == [901, 901, 469, 475, 670]
    for seq in counts:
        assert [r['frame_index'] for r in frames if r['video']==seq] == list(range(counts[seq]))
    hash_inputs(frames)
    dump(ROOT/'frames.json', frames)
    shutil.copyfile(LATEST/'ground_truth.json', ROOT/'ground_truth.json')
    configs = {}
    (ROOT/'configs').mkdir()
    for method, ref in REFS.items():
        cfg = load(ref/'metrics.json')['config'] if method=='A' else load(ref/'config.json')
        if method == 'C':
            cfg = load(LATEST/'candidate_config.json'); cfg['fixed_signal_fastpath'] = False
        if method in 'DE':
            cfg['fixed_signal_fastpath'] = True
        cfg.update(device='cpu', half=False)
        assert (cfg['imgsz'], cfg['det_conf'], cfg['det_iou'], cfg['max_det']) == (640,.001,.7,300)
        assert cfg['class_map'] == {'0':0,'1':1,'2':2,'3':3,'5':4,'7':5,'9':6,'11':7}
        assert not cfg['use_clahe'] and not cfg['keyframe_refine']
        config_from_dict(cfg)  # Production validation, including fastpath eligibility.
        configs[method] = cfg
        dump(ROOT/'configs'/f'{method}.json', cfg)
    checkpoint = Path(configs['A']['model_path'])
    assert sha(checkpoint) == 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
    prior = {}
    # Preserve complete previous results, including sidecars and source evidence.
    for dirname in ['argoverse_hd_coco_phase1_v1','argoverse_hd_phase2_diagnostics_v1',
                    'argoverse_hd_phase3_v1','argoverse_hd_next_research_v1',
                    'argoverse_hd_detector_diagnosis_D_v1']:
        folder = BASE/'runs'/dirname
        assert folder.is_dir(), folder
        manifest = load(folder/'artifact_sha256.json')
        for rel, digest in manifest.items():
            path = folder/rel
            assert sha(path)==digest, path
            prior[str(path)] = digest
        prior[str(folder/'artifact_sha256.json')] = sha(folder/'artifact_sha256.json')
    for ref in REFS.values():
        for name in ['preds.npz','prediction_details.npz','frames.csv','config.json','state_hashes.json']:
            path=ref/name
            if path.exists(): prior[str(path)]=sha(path)
    prior[str(checkpoint)] = sha(checkpoint)
    for path in (BASE/'docs').glob('*.md'):
        prior[str(path)] = sha(path)
    dump(ROOT/'prior_evidence.json', prior)
    sources = {str(p):sha(p) for dirname in ['extracted/adaptive_pipeline',
        'analysis/argoverse_hd_phase2','analysis/argoverse_hd_runtime_E']
        for p in (BASE/dirname).rglob('*.py')}
    sources[str(BASE/'tests/test_runtime_E.py')] = sha(BASE/'tests/test_runtime_E.py')
    dump(ROOT/'execution_sources.json', sources)
    dump(ROOT/'execution_plan.json', dict(status='APPROVED_E_640_CPU_ONLY',
        blocks=BLOCKS, references={k:str(v) for k,v in REFS.items()},
        frames_per_pass=3416, measured_requests=51280, warmup_requests=75,
        total_requests=51355, torch_intra=8, torch_interop=1, opencv_threads=10,
        interblock_pause_seconds=30, dtype='float32', device='cpu',
        validation='Bitwise prediction/detail parity; exact schedules; sequence-end tracker hashes; first-pass official tensors and metrics exact; later scores reused only after exact parity.',
        no_phase4=True, no_algorithms_changed=True,
        command='PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.run run'))
    dump(ROOT/'PREPARED.json', dict(inputs_verified=True, prior_files=len(prior),
        new_inference_so_far=0, hardware=platform.platform(), status=system_status()))
    print('E prepared: 25 fresh processes, 51355 total requests.', flush=True)


def worker(block, method):
    # Import order mirrors D/native wrapper. No NMS instrumentation or alternate implementation.
    import torch
    from ultralytics import YOLO  # Complete lazy imports before common thread controls.
    from adaptive_pipeline.detection.detector import YOLODetector
    from adaptive_pipeline.pipeline import AdaptivePipeline
    from analysis.argoverse_hd_phase2.replay import state_digest
    invocation_wall, invocation_cpu = time.perf_counter(), time.process_time()
    out = ROOT/'passes'/f'{block:02d}_{method}'
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'STARTED.json', dict(block=block, method=method, pid=os.getpid(), status=system_status()))
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    cv2.setNumThreads(10); cv2.ocl.setUseOpenCL(False)
    frames=load(ROOT/'frames.json')
    cfg=config_from_dict(load(ROOT/'configs'/f'{method}.json'))
    prehash=time.perf_counter(); hash_inputs(frames)
    prehash=time.perf_counter()-prehash
    startup_peak=peak_mib()

    class MeasuredDetector(YOLODetector):
        def __init__(self, config):
            super().__init__(config)
            self.requests=0; self.last_cpu_ms=0.
        def detect(self, frame):
            self.requests+=1
            before=time.process_time_ns()
            answer=super().detect(frame)
            self.last_cpu_ms=(time.process_time_ns()-before)/1e6
            return answer
        def detect_batch(self, crops, imgsz):
            raise RuntimeError('ROI/batched inference is outside E scope')

    setup_wall, setup_cpu=time.perf_counter(),time.process_time()
    detector=MeasuredDetector(cfg)
    counter=ForwardCounter(detector.model.model)
    setup_cpu=time.process_time()-setup_cpu;setup_wall=time.perf_counter()-setup_wall
    warm_image=cv2.imread(frames[0]['image'])
    assert warm_image.shape==(1200,1920,3)
    warm_wall,warm_cpu=time.perf_counter(),time.process_time()
    full_frame_warmup(detector,warm_image)
    warm_cpu=time.process_time()-warm_cpu;warm_wall=time.perf_counter()-warm_wall
    assert detector.requests==3 and counter.count==3
    dump(out/'WARMUP_COMPLETE.json',dict(requests=detector.requests,root_forwards=counter.count,
        shapes=counter.summary(),torch_intra=torch.get_num_threads(),
        torch_interop=torch.get_num_interop_threads(),opencv_threads=cv2.getNumThreads()))
    del warm_image
    assert (torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads())==(8,1,10)
    predictor=detector.model.predictor
    validate_predictor(predictor)
    runtime=dict(python=sys.version, platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ['torch','torchvision','ultralytics',
            'numpy','scipy','opencv-python','pycocotools']},
        torch_intra=torch.get_num_threads(),torch_interop=torch.get_num_interop_threads(),
        opencv_threads=cv2.getNumThreads(),opencv_opencl=cv2.ocl.useOpenCL(),
        thread_environment={k:v for k,v in os.environ.items() if any(s in k for s in ['THREAD','OMP','VECLIB','BLAS'])},
        actual_device=str(predictor.device),model_fp16=predictor.model.fp16,
        model_format=predictor.model.format,
        model_backend_class=type(predictor.model.backend).__module__+'.'+type(predictor.model.backend).__name__,
        model_fused=bool(predictor.model.model.is_fused()),
        resolved_predict_args={k:v if isinstance(v,(str,int,float,bool,type(None),list,tuple)) else repr(v)
            for k,v in vars(predictor.args).items()},
        nms_backend='torchvision.ops.nms' if 'torchvision' in sys.modules else 'ultralytics.TorchNMS.nms',
        checkpoint_sha256=detector.checkpoint_sha256,
        component_devices={'detector':'torch CPU FP32','decode':'OpenCV CPU',
            'flow':'OpenCV CPU pyramidal LK','tracker':'NumPy/SciPy CPU',
            'scheduler':'CPU Python/NumPy','evaluation':'CPU pycocotools (offline)'})
    dump(out/'runtime.json',runtime)
    postwarm_peak=peak_mib()
    pipe=AdaptivePipeline(cfg,detector)
    predictions={}; details={}; rows=[]; boundaries={}; previous=None
    loop_wall,loop_cpu=time.perf_counter(),time.process_time()
    for i,row in enumerate(frames):
        if row['video']!=previous:
            pipe.reset();previous=row['video']
        before=detector.requests
        pred, detail, rec=timed_step(pipe,row)
        calls=detector.requests-before
        assert calls==int(rec['decision']=='FULL') and rec['decision'] in ['FULL','SKIP']
        rec['detector_calls']=calls
        rec['detector_cpu_ms']=detector.last_cpu_ms if calls else 0.
        predictions[row['key']]=pred
        details[row['key']]=detail
        rows.append(rec)
        if i+1==len(frames) or frames[i+1]['video']!=row['video']:
            boundaries[row['key']]=state_digest(pipe.tracker)
            print(f'block {block} {method}: {i+1}/3416 frames, requests {detector.requests-3}',flush=True)
    loop_cpu=time.process_time()-loop_cpu;loop_wall=time.perf_counter()-loop_wall
    measured_peak=peak_mib()
    expected_calls=3416 if method=='A' else 1710
    assert detector.requests==expected_calls+3 and counter.count==detector.requests
    assert counter.summary()==[dict(shape=[1,3,416,640],dtype='torch.float32',device='cpu',count=detector.requests)]
    validate_wall,validate_cpu=time.perf_counter(),time.process_time()
    hash_inputs(frames)
    ref=REFS[method]
    assert_archive_equal(predictions,archive(ref/'preds.npz'),'archived predictions')
    assert_archive_equal(details,archive(ref/'prediction_details.npz'),'archived prediction details')
    with (ref/'frames.csv').open() as f: expected_rows=list(csv.DictReader(f))
    assert len(expected_rows)==len(rows)
    for a,b in zip(rows,expected_rows):
        for name in ['frame','video','decision','reasons','since_full','n_out','n_tracks','n_crops']:
            assert str(a[name])==b[name],(a['key'],name,a[name],b[name])
    if method!='A':
        reference_states=load(ref/'state_hashes.json')
        assert all(digest==reference_states[key] for key,digest in boundaries.items())
    validate_cpu=time.process_time()-validate_cpu;validate_wall=time.perf_counter()-validate_wall
    serial_wall,serial_cpu=time.perf_counter(),time.process_time()
    np.savez_compressed(out/'preds.npz',**predictions)
    np.savez_compressed(out/'prediction_details.npz',**details)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'frames.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    dump(out/'sequence_end_state_hashes.json',boundaries)
    serial_cpu=time.process_time()-serial_cpu;serial_wall=time.perf_counter()-serial_wall
    dump(out/'COMPLETE.json',dict(block=block,method=method,frames=len(rows),
        measured_requests=expected_calls,warmup_requests=3,total_requests=detector.requests,
        root_forwards=counter.count,forward_shapes=counter.summary(),
        prediction_bytes_exact=True,detail_bytes_exact=True,schedule_exact=True,
        sequence_end_states_exact=(method!='A'),all_frame_state_hashes_checked=False,
        prehash_wall_s=prehash,setup_wall_s=setup_wall,setup_cpu_s=setup_cpu,
        warmup_wall_s=warm_wall,warmup_cpu_s=warm_cpu,loop_wall_s=loop_wall,loop_cpu_s=loop_cpu,
        validation_wall_s=validate_wall,validation_cpu_s=validate_cpu,
        serialization_wall_s=serial_wall,serialization_cpu_s=serial_cpu,
        startup_peak_mib=startup_peak,postwarmup_peak_mib=postwarm_peak,
        measured_loop_peak_mib=measured_peak,entire_worker_peak_mib=peak_mib(),
        invocation_wall_s=time.perf_counter()-invocation_wall,
        invocation_cpu_s=time.process_time()-invocation_cpu,end_status=system_status()))
    print(f'PASS COMPLETE {block} {method}: exact archive parity; loop {loop_wall:.2f}s',flush=True)


def scoring_gate():
    from analysis.argoverse_hd_phase2.scoring import score
    frames=load(ROOT/'frames.json');gt=load(ROOT/'ground_truth.json')
    for method in 'ABCDE':
        out=ROOT/'accuracy'/method
        reference=(BASE/'runs/argoverse_hd_coco_phase1_v1/baseline') if method=='A' else REFS[method]/'scoring'
        expected=load(reference/'metrics.json')
        metrics=expected['official_COCO_max100'] if method=='A' else expected['COCO']
        result=score(archive(ROOT/'passes'/f'01_{method}'/'preds.npz'),gt,frames,out,expected=metrics)
        assert result['COCO']==metrics
        if method=='A':
            assert result['fixed_conf25']==expected['supplementary_fixed_conf25_IoU50']
            assert result['per_sequence']=={k:v['metrics'] for k,v in expected['per_sequence'].items()}
        else:
            assert result['fixed_conf25']==expected['fixed_conf25']
            assert result['per_sequence']==expected['per_sequence']
        tensor_name='official_tensors100.npz' if method=='A' else 'official_tensors.npz'
        assert_archive_equal(archive(out/'official_tensors.npz'),archive(reference/tensor_name),'official COCO tensors')
        dump(out/'ARCHIVE_FIDELITY.json',dict(official_metrics_exact=True,official_tensors_bytes_exact=True,
            per_sequence_metrics_exact=True,fixed_conf25_exact=True,reference=str(reference)))
    dump(ROOT/'FIRST_BLOCK_ACCURACY_GATE.json',dict(all_five_pass=True,new_inferences=0))


def execute():
    verify_sources()
    assert not (ROOT/'EXECUTION_STARTED.json').exists(),'Do not silently resume/repeat interrupted passes'
    start=time.perf_counter()
    dump(ROOT/'EXECUTION_STARTED.json',dict(status=system_status(),blocks=BLOCKS))
    for block,order in enumerate(BLOCKS,1):
        if block>1:
            before=time.perf_counter();time.sleep(30)
            dump(ROOT/f'pause_before_block{block}.json',dict(requested_s=30,actual_s=time.perf_counter()-before,status=system_status()))
        for method in order:
            out=ROOT/'passes'/f'{block:02d}_{method}'
            out.parent.mkdir(exist_ok=True)
            before=time.perf_counter()
            command=[sys.executable,'-m','analysis.argoverse_hd_runtime_E.run','worker','--block',str(block),'--method',method]
            with (out.parent/f'{block:02d}_{method}.log').open('x') as log:
                p=subprocess.run(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:
                dump(ROOT/'EXECUTION_FAILED.json',dict(block=block,method=method,returncode=p.returncode,
                    elapsed_s=time.perf_counter()-start,log=str(out.parent/f'{block:02d}_{method}.log')))
                raise RuntimeError(f'Pass {block} {method} failed: stop, preserve, investigate log')
            dump(out/'SUBPROCESS.json',dict(wall_including_interpreter_s=time.perf_counter()-before,command=command))
            print(f'COMPLETED BLOCK {block} ARM {method}',flush=True)
        if block==1:
            score_start=time.perf_counter();scoring_gate()
            dump(ROOT/'scoring_cost.json',dict(wall_s=time.perf_counter()-score_start))
            print('FIRST BLOCK OFFICIAL FIDELITY GATE PASSED',flush=True)
    verify_sources()
    dump(ROOT/'EXECUTION_COMPLETE.json',dict(wall_s=time.perf_counter()-start,
        passes=25,measured_requests=51280,warmup_requests=75,total_requests=51355,
        all_prediction_and_schedule_parity=True,status=system_status()))
    print('ALL 25 E PASSES COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','worker'])
    p.add_argument('--block',type=int);p.add_argument('--method',choices=list('ABCDE'))
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':worker(a.block,a.method)
    else:execute()


if __name__=='__main__':main()
