"""Approved D only: paired stock detector diagnosis; E requires separate approval."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from adaptive_pipeline.detection.detector import YOLODetector
from analysis.argoverse_hd_phase2.common import archive, config_from_dict, dump, load, sha
from analysis.argoverse_hd_phase2.scoring import score
from .observer import NativeNMSObserver, map_rows

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'runs/argoverse_hd_detector_diagnosis_D_v1'
LATEST=BASE/'runs/argoverse_hd_next_research_v1'
BENCH=BASE/'runs/argoverse_hd_benchmark_v1'


def verify():
    for filename in ['execution_sources.json','prior_evidence.json']:
        for path,digest in load(ROOT/filename).items():
            if sha(path)!=digest:raise RuntimeError('Evidence/source changed: '+path)
    if (BASE/'runs/argoverse_hd_real_runtime_E_v1').exists():
        raise RuntimeError('E destination exists: investigate separate authorization/state')


def prepare():
    ROOT.mkdir(exist_ok=False)
    before=ROOT/'pre_execution';before.mkdir()
    for cmd,name in [(['git','status','--short'],'git_status.txt'),(['git','diff','--binary'],'git_diff.patch'),(['git','rev-parse','HEAD'],'git_head.txt')]:
        (before/name).write_bytes(subprocess.check_output(cmd,cwd=BASE))
    for folder in ['extracted/adaptive_pipeline','analysis/argoverse_hd_detector_D']:
        shutil.copytree(BASE/folder,before/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    proposal=load(BASE/'analysis/argoverse_hd_DE_plan_v1/plan.json')
    for path,digest in load(BASE/'analysis/argoverse_hd_DE_plan_v1/source_sha256.json').items():assert sha(path)==digest,path
    original=load(proposal['D']['manifest'])['frames']
    allframes=load(LATEST/'frameplan.json');index={(f['video'],f['frame_index']):f for f in allframes}
    frames=[]
    for f in original:
        row=index[(f['video'],f['frame_index'])]
        assert row['image_id']==f['image_id'] and row['image_sha256']==f['image_sha256']
        assert sha(row['image'])==row['image_sha256']
        frames.append(row)
    assert len(frames)==320 and len({f['image_id'] for f in frames})==320
    gt=load(LATEST/'ground_truth.json');ids={f['image_id'] for f in frames}
    gt['images']=[im for im in gt['images'] if im['id'] in ids]
    gt['annotations']=[a for a in gt['annotations'] if a['image_id'] in ids]
    validation=[]
    for seq in sorted({f['video'] for f in frames}):
        seqframes=[f for f in frames if f['video']==seq]
        validation.extend([seqframes[0]['key'],seqframes[-1]['key']])
    config=load(LATEST/'candidate_config.json')
    specifications={str(size):dict(config,imgsz=size,policy='always',fixed_signal_fastpath=False) for size in [640,960]}
    plan={'authorization':'User approved D first, only. Report findings/costs and wait for separate E approval; no Phase4/training/commit/push.',
        'created_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'frames':320,'primary_predictions':640,
        'canonical_extra_requests':20,'explicit_full_frame_warmups':6,'planned_application_requests':666,
        'threads':proposal['threads'],'validation_keys':validation,'configs':specifications,
        'observer':'Native backend/index/output preserved; raw head cloned before mutation; pre-cap rows must reconstruct all80 canonical post-NMS outputs exactly.',
        'bridge':'Canonical640 validation and all observed640 mapped rows must exactly match saved Dense. No score/tolerance relaxation.',
        'shadow':'One standard-COCO640 global-cap-removal diagnostic from already-computed native NMS survivors, conditional on exact fidelity; no extra NMS/inference.',
        'score_reuse':'If live640 and saved320-subset rows are bitwise identical, score once and use the identical official tensors for both; no redundant scoring.',
        'timing':'D diagnostic inference/capture timings, decoding/serialization/validation separately. No E/uninstrumented/streaming performance claim.',
        'E_status':'NOT_AUTHORIZED_TO_START. Future E must report FULL/SKIP latency, decode-inclusive throughput, block drift/possible thermal effects, prioritizing Dense vs practical reference.'}
    dump(ROOT/'plan.json',plan);dump(ROOT/'frames.json',frames);dump(ROOT/'ground_truth.json',gt)
    prior={}
    for rel,digest in load(LATEST/'artifact_sha256.json').items():
        path=LATEST/rel;assert sha(path)==digest,path;prior[str(path)]=digest
    for p,h in load(LATEST/'input_evidence.json')['source_sha256'].items():assert sha(p)==h,p;prior[p]=h
    for p in ['analysis/argoverse_hd_DE_plan_v1/plan.json','analysis/argoverse_hd_DE_plan_v1/source_sha256.json','docs/DE_EXPERIMENT_PLAN.md','docs/NEXT_RESEARCH_RESULTS.md',
        'analysis/argoverse_hd_phase3/verify.py','analysis/argoverse_hd_phase3/report.py','docs/PHASE3_RESULTS.md']:
        prior[str(BASE/p)]=sha(BASE/p)
    dump(ROOT/'prior_evidence.json',prior)
    sources={str(p):sha(p) for folder in ['extracted/adaptive_pipeline','analysis/argoverse_hd_detector_D','analysis/argoverse_hd_phase2'] for p in (BASE/folder).rglob('*.py')}
    sources[str(BASE/'tests/test_detector_D.py')]=sha(BASE/'tests/test_detector_D.py')
    import ultralytics.utils.nms as nms
    import ultralytics.models.yolo.detect.predict as predict
    for module in [nms,predict]:sources[module.__file__]=sha(module.__file__)
    dump(ROOT/'execution_sources.json',sources)
    dump(ROOT/'PREPARED.json',{'frames':320,'regular_annotations':sum(not a.get('iscrowd') for a in gt['annotations']),
        'crowds':sum(bool(a.get('iscrowd')) for a in gt['annotations']),'new_inference_so_far':0})
    print('D prepared; E remains pending',flush=True)


class ForwardCounter:
    def __init__(self,model):
        self.count=0;self.shapes=[]
        # Functions retain their closure on deepcopy; a bound method would copy
        # the counter object when Ultralytics clones the model for prediction.
        def capture(module,args):
            self.capture(module,args)
        self.handle=model.register_forward_pre_hook(capture)

    def capture(self,module,args):
        self.count+=1
        value=args[0]
        self.shapes.append({'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device)})


def infer():
    verify()
    if (ROOT/'INFERENCE_COMPLETE.json').exists():raise FileExistsError('Inference already complete')
    if (ROOT/'requests.jsonl').exists():raise FileExistsError('Partial inference exists: investigate before any restart')
    torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(10)
    assert not cv2.ocl.useOpenCL()
    runtime={'python':platform.python_version(),'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','ultralytics','numpy','scipy','opencv-python','pycocotools']},
        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'opencv_threads':cv2.getNumThreads(),
        'device':'CPU','precision':'FP32','GPU_used':False}
    dump(ROOT/'runtime.json',runtime)
    plan=load(ROOT/'plan.json');frames=load(ROOT/'frames.json');dense=archive(BENCH/'baseline/preds.npz')
    amendment=ROOT/'counter_instrumentation_amendment.json'
    prior_failed_count=load(amendment)['failed_attempt_completed_application_requests'] if amendment.exists() else 0
    if prior_failed_count not in (0,1):raise RuntimeError('Unexpected failure accounting')
    bykey={f['key']:f for f in frames};validation=plan['validation_keys'];requests=[];model_counts=[]
    stage_wall=time.perf_counter();stage_cpu=time.process_time()
    ledger=(ROOT/'requests.jsonl').open('x',buffering=1)
    try:
        for size in [640,960]:
            destination=ROOT/f'imgsz{size}';destination.mkdir(exist_ok=False)
            (destination/'raw').mkdir()
            cfg=config_from_dict(plan['configs'][str(size)])
            assert cfg.device=='cpu' and not cfg.half and cfg.det_conf==.001 and cfg.det_iou==.7 and cfg.max_det==300
            before=time.perf_counter();detector=YOLODetector(cfg);load_seconds=time.perf_counter()-before
            if detector.checkpoint_sha256!=load(BASE/'analysis/argoverse_hd_DE_plan_v1/plan.json')['weights_sha256']:
                raise AssertionError('Checkpoint mismatch')
            categories={c['id']:c['name'].replace(' ','_') for c in load(ROOT/'ground_truth.json')['categories']}
            for src,dst in cfg.class_map.items():assert detector.model.names[src].replace(' ','_')==categories[dst]
            counter=ForwardCounter(detector.model.model)
            predictions={};all80={};shadow={};rows=[];canonical={};canonical80={}

            def read(key):
                f=bykey[key]
                if sha(f['image'])!=f['image_sha256']:raise AssertionError('Image changed: '+key)
                start=time.perf_counter();image=cv2.imread(f['image']);dt=time.perf_counter()-start
                if image is None or image.shape!=(1200,1920,3):raise AssertionError('Image shape/decode')
                return image,dt

            def request(image,key,kind):
                number=len(requests)+1+prior_failed_count
                if number>666+prior_failed_count:raise RuntimeError('D request budget including recorded failed warmup exhausted')
                ledger.write(json.dumps({'request':number,'size':size,'key':key,'kind':kind,'status':'started'})+'\n')
                start,cpu=time.perf_counter(),time.process_time();f0=counter.count
                result=detector._predict(image,size)
                wall,process=time.perf_counter()-start,time.process_time()-cpu
                if len(result)!=1:raise AssertionError('Batch must be one')
                native=result[0].boxes.data.cpu().numpy().copy()
                mapped=detector._to_array(result[0])
                if native.shape[1]!=6 or mapped.shape[1]!=6:raise AssertionError('Detection shape')
                record={'request':number,'size':size,'key':key,'kind':kind,'predict_wall_seconds':wall,
                    'predict_process_seconds':process,'observed_root_forwards':counter.count-f0,
                    'last_forward':counter.shapes[-1],'native80_rows':len(native),'mapped_rows':len(mapped),'status':'complete'}
                if counter.count-f0!=1:raise AssertionError('Unexpected implicit forward(s); inspect accounting')
                if counter.shapes[-1]['device']!='cpu' or counter.shapes[-1]['dtype']!='torch.float32':raise AssertionError('Actual backend/dtype differs')
                requests.append(record);ledger.write(json.dumps(record)+'\n')
                return mapped,native,record

            first,_=read(frames[0]['key'])
            for _ in range(3):request(first,frames[0]['key'],'excluded_full_frame_warmup')
            for key in validation:
                image,decode=read(key);mapped,native,record=request(image,key,'canonical_validation')
                canonical[key]=mapped;canonical80[key]=native
                if size==640 and not np.array_equal(mapped,dense[key]):
                    np.savez_compressed(destination/'BRIDGE_FAILURE.npz',new=mapped,old=dense[key])
                    dump(destination/'BRIDGE_FAILURE.json',{'key':key,'new_shape':mapped.shape,'old_shape':dense[key].shape})
                    raise AssertionError('New canonical640 differs from saved Dense; dependent work prohibited')
            np.savez_compressed(destination/'canonical_validation.npz',**canonical)
            np.savez_compressed(destination/'canonical_validation80.npz',**canonical80)
            order=validation+[f['key'] for f in frames if f['key'] not in set(validation)]
            for index,key in enumerate(order):
                image,decode=read(key)
                with NativeNMSObserver() as obs:
                    mapped,native,record=request(image,key,'observed_primary')
                shape=counter.shapes[-1]['shape'][2:]
                survivors=obs.original_coordinates(shape,image.shape)
                np.testing.assert_array_equal(survivors[:300],native,err_msg='Restored all80 top300 reconstruction')
                np.testing.assert_array_equal(map_rows(native,cfg.class_map),mapped)
                if key in canonical:
                    np.testing.assert_array_equal(mapped,canonical[key],err_msg='Observed/canonical mapped')
                    np.testing.assert_array_equal(native,canonical80[key],err_msg='Observed/canonical80')
                if size==640:np.testing.assert_array_equal(mapped,dense[key],err_msg='Saved/live640 bridge '+key)
                predictions[key]=mapped;all80[key]=native
                shadow[key]=map_rows(survivors,cfg.class_map)
                f=bykey[key]
                serial=time.perf_counter()
                np.savez_compressed(destination/'raw'/f"{f['image_id']:06d}.npz",raw_head=obs.raw.numpy(),
                    pre_nms_rows=obs.rows.numpy(),native_survivor_indices=obs.indices.numpy(),native_survivors_original=survivors,
                    capped_letterbox=obs.capped_letterbox.numpy(),input_tensor_shape=shape)
                rows.append(dict(record,decode_seconds=decode,serialization_seconds=time.perf_counter()-serial,
                    backend=obs.backend,pre_nms_rows=len(obs.rows),native_survivors=len(survivors),
                    discarded_after_global300=max(0,len(survivors)-300),discarded_supported_after_global300=len(map_rows(survivors[300:],cfg.class_map)),
                    exact_all80_reconstruction=True,exact_observed_canonical=key in canonical,
                    exact_saved640_bridge=size==640))
                if index%64==0:print(f'D imgsz{size}: {index+1}/320 observed primary; native survivors {len(survivors)}',flush=True)
            # Official prediction order remains the frozen manifest order.
            np.savez_compressed(destination/'preds.npz',**{f['key']:predictions[f['key']] for f in frames})
            np.savez_compressed(destination/'native80.npz',**{f['key']:all80[f['key']] for f in frames})
            np.savez_compressed(destination/'precap_mapped.npz',**{f['key']:shadow[f['key']] for f in frames})
            with (destination/'frames.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            args=vars(detector.model.predictor.args)
            dump(destination/'resolved_predictor_args.json',{k:v for k,v in args.items() if isinstance(v,(str,int,float,bool,type(None),list,tuple))})
            dump(destination/'VALIDATED_COMPLETE.json',{'primary_frames':320,'canonical_validation_images':10,
                'all80_pre_cap_reconstruction_exact':True,'observed_canonical_exact':True,'saved640_bridge_exact':size==640,
                'root_forward_calls':counter.count,'model_load_seconds':load_seconds,'backend':rows[0]['backend']})
            model_counts.append({'size':size,'root_forwards':counter.count,'model_load_seconds':load_seconds})
            counter.handle.remove()
        verify()
        assert len(requests)==666 and sum(x['root_forwards'] for x in model_counts)==666
        dump(ROOT/'INFERENCE_COMPLETE.json',{'planned_validated_application_requests':666,'validated_application_requests':len(requests),
            'prior_failed_counter_warmup_application_requests':prior_failed_count,'actual_application_requests_including_failure':len(requests)+prior_failed_count,
            'prior_failed_warmup_root_counter_unavailable':bool(prior_failed_count),
            'primary_requests':640,'canonical_validation_requests':20,'explicit_warmup_requests':6,'root_forwards':model_counts,
            'stage_wall_seconds_with_load_decode_hash_validation_capture_serialization':time.perf_counter()-stage_wall,
            'stage_process_seconds_with_load_decode_hash_validation_capture_serialization':time.process_time()-stage_cpu,
            'E_started':False})
    except Exception as error:
        dump(ROOT/'INFERENCE_FAILURE.json',{'exception':repr(error),'completed_application_requests':len(requests),
            'incomplete_started_request_may_exist_in_ledger':True,'model_counts_completed':model_counts,'E_started':False})
        raise
    finally:ledger.close()


def scoring():
    verify();assert (ROOT/'INFERENCE_COMPLETE.json').exists()
    if (ROOT/'SCORING_COMPLETE.json').exists():raise FileExistsError('Scoring already complete')
    if importlib.metadata.version('pycocotools')!='2.0.11':raise AssertionError('COCO version')
    frames=load(ROOT/'frames.json');gt=load(ROOT/'ground_truth.json')
    dense=archive(BENCH/'baseline/preds.npz');live=archive(ROOT/'imgsz640/preds.npz')
    for f in frames:np.testing.assert_array_equal(live[f['key']],dense[f['key']])
    cases={'imgsz640':ROOT/'imgsz640/preds.npz','imgsz960':ROOT/'imgsz960/preds.npz','shadow640_no_global300':ROOT/'imgsz640/precap_mapped.npz'}
    for name,path in cases.items():
        destination=ROOT/'scores'/name
        if not (destination/'SCORE_COMPLETE.json').exists():score(archive(path),gt,frames,destination)
    dump(ROOT/'SCORING_COMPLETE.json',{'cases':list(cases),'saved_subset_and_live640_share_identical_scored_arrays':True,
        'new_detector_requests_in_scoring':0,'shadow_is_diagnostic_only':True,'E_started':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','infer','score'],required=True)
    stage=p.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='infer':infer()
    else:scoring()


if __name__=='__main__':main()
