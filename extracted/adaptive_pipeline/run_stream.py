"""Timestamp replay: Dense640 by default, opt-in streaming Practical Fixed-2.

Frozen offline run.py and algorithms are untouched. All clocks use perf_counter_ns.
"""
import argparse
import csv
import json
from pathlib import Path
import platform
import resource
import sys
import time

import cv2
import numpy as np

from .adaptive_framing.capture import TimestampReplayCapture
from .adaptive_framing.frame_manager import LatestFrameBuffer


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform=='darwin' else 1024)


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(p50=float(np.percentile(a,50)), p95=float(np.percentile(a,95)),
                max=float(a.max()), mean=float(a.mean())) if len(a) else None


def check_dense_config(data):
    expected = dict(device='cpu', half=False, imgsz=640, det_conf=.001, det_iou=.7,
                    max_det=300, policy='always', use_clahe=False, keyframe_refine=False)
    if any(data.get(k) != v for k,v in expected.items()):
        raise ValueError('Stream validation requires unchanged frozen CPU Dense640 settings')
    if data['class_map'] != {'0':0,'1':1,'2':2,'3':3,'5':4,'7':5,'9':6,'11':7}:
        raise ValueError('Frozen Argoverse class mapping required')


def check_practical_config(data):
    check_dense_config(dict(data,policy='always'))
    expected = dict(policy='fixed',fixed_k=2,propagation='flow',flow_grid=3,
        observation_anchoring=True,low_score_association=True,association_low_conf=.1,
        track_min_conf=.25,conf_decay=.93,fixed_signal_fastpath=True,
        quality_aware_flow=False,motion_compensation=False,appearance_verification=False,
        track_birth_conf=None,threshold_aware_assignment=False)
    if any(data.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen Practical Fixed-2 settings required')


def consume(capture, pipe, *, deadline_ns, memory_mib=4096):
    """Single model owner; producer runs throughout inference. Drain normal EOF.

    selected_ns is the atomic buffer selection time. inference_start_ns is after
    selection. A newer arrival in that tiny interval never cancels active work.
    """
    predictions, details, rows = {}, {}, []
    previous_id = -1
    wall0, cpu0 = time.perf_counter_ns(), time.process_time_ns()
    try:
        capture.start()
        while True:
            if time.perf_counter_ns() >= deadline_ns:
                raise TimeoutError('Approved experiment wall-time limit reached')
            observed_peak = peak_mib()
            if observed_peak > memory_mib:
                raise MemoryError('Approved process peak-memory limit reached')
            selection = capture.buffer.take(timeout=.1)
            if selection is None:
                if capture.is_stopped():
                    # take rechecks closure under the queue lock: no final-frame race.
                    selection = capture.buffer.take(timeout=0)
                    if selection is None:
                        break
                else:
                    continue
            p = selection.packet
            if p.frame_id <= previous_id:
                raise AssertionError('Source IDs must strictly increase')
            source_gap = p.frame_id-previous_id if previous_id>=0 else 1
            previous_id = p.frame_id
            start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
            if getattr(type(pipe),'step_packet',None) is not None:
                pred, rec = pipe.step_packet(p)
            else:
                pred, rec = pipe.step(p.frame, p.frame_id)
            cpu_end, ready = time.process_time_ns(), time.perf_counter_ns()
            key = f'{p.source_id}/{p.frame_id}'
            predictions[key] = pred
            details[key] = pipe.last_prediction_details.copy()
            rows.append(dict(key=key, frame_id=p.frame_id, source_id=p.source_id,
                source_timestamp_ns=p.source_timestamp_ns, capture_ns=p.capture_ns,
                decode_start_ns=p.decode_start_ns, decode_end_ns=p.decode_end_ns,
                published_ns=p.published_ns, selected_ns=selection.selected_ns,
                newest_published_id_at_selection=selection.newest_published_id,
                pending_after_selection=selection.pending_after,
                inference_start_ns=start, result_ready_ns=ready,
                frame_age_at_inference_ms=(start-p.capture_ns)/1e6,
                capture_to_result_ms=(ready-p.capture_ns)/1e6,
                queue_wait_ms=(selection.selected_ns-p.published_ns)/1e6,
                decode_ms=(p.decode_end_ns-p.decode_start_ns)/1e6,
                service_ms=(ready-start)/1e6, service_cpu_ms=(cpu_end-cpu_start)/1e6,
                peak_rss_mib=observed_peak, decision=rec['decision'], n_out=len(pred),
                source_gap=source_gap,consumed_index=len(rows),nominal_decision='FULL',
                forced_full=0,gap_triggered_full=0,state_reset=0,flow_attempted=0,
                flow_accepted=0,fallback_reasons='',state_reset_reasons=''))
            # Scalar export only, outside result-ready timing. Dense's processing
            # call and prediction/detail construction are unchanged.
            for name in ['reasons','since_full','source_elapsed_ms','temporal_dt_source_frames',
                'nominal_decision','forced_full','gap_triggered_full','state_reset','flow_attempted',
                'flow_accepted','flow_failed_visible','flow_valid_visible','flow_scale_clipped',
                'fallback_reasons','state_reset_reasons','source_frames_since_full','max_track_source_age',
                't_signal_ms','t_propagate_ms','t_decide_ms','t_detect_ms','t_track_ms','n_tracks']:
                if name in rec:
                    rows[-1][name]=rec[name]
            # Drop this packet reference; neither result rows nor predictions hold
            # decoded source images. The model may retain only its current input.
            del p, selection
        capture.join(timeout=2)
    finally:
        capture.stop()
        capture.join(timeout=2)
    wall1, cpu1 = time.perf_counter_ns(), time.process_time_ns()
    stats = capture.buffer.stats()
    if not rows or not stats['closed'] or stats['failed'] or stats['pending']:
        raise AssertionError('Incomplete or failed stream')
    if stats['captured'] != len(rows)+stats['replaced'] or stats['consumed'] != len(rows):
        raise AssertionError('Capture/dequeue/completion accounting mismatch')
    return predictions, details, rows, dict(buffer=stats, loop_wall_s=(wall1-wall0)/1e9,
        loop_cpu_s=(cpu1-cpu0)/1e9, cpu_percent_one_core_100=100*(cpu1-cpu0)/(wall1-wall0),
        peak_rss_mib=peak_mib(), epoch_ns=capture.epoch_ns, capture_finished_ns=capture.finished_ns)


def run(args):
    # Complete lazy imports before common CPU controls, as in archived E.
    import torch
    from ultralytics import YOLO
    from .detection.detector import YOLODetector
    from .pipeline import AdaptivePipeline
    from .shared.config import PipelineConfig
    data = json.loads(Path(args.config).read_text())
    processor=getattr(args,'processor','dense')
    if processor=='practical-fixed2':
        check_practical_config(data)
        if args.policy!='latest':
            raise ValueError('Combined mode requires latest-frame buffering')
    else:
        check_dense_config(data)
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    cv2.setNumThreads(10); cv2.ocl.setUseOpenCL(False)
    data['class_map'] = {int(k):v for k,v in data['class_map'].items()}
    cfg = PipelineConfig(**data)
    rows = json.loads(Path(args.manifest).read_text())
    if len(rows) != 300:
        raise ValueError('Bounded experiment requires exactly 300 source frames')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    write_json(out/'STARTED.json', dict(policy=args.policy, fps=args.fps, pid=__import__('os').getpid()))
    forwards = []
    detector = YOLODetector(cfg)
    def count_forward(module, inputs):
        tensor = inputs[0]
        forwards.append(dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device)))
    handle = detector.model.model.register_forward_pre_hook(count_forward)
    first = cv2.imread(rows[0]['image'])
    if first is None:
        raise IOError(rows[0]['image'])
    for _ in range(3):
        detector.detect(first)
    del first
    expected_shape = dict(shape=[1,3,416,640], dtype='torch.float32', device='cpu')
    if forwards != [expected_shape]*3:
        raise AssertionError('Warmup device/shape/forward count mismatch')
    predictor = detector.model.predictor
    for k,v in dict(imgsz=640,conf=.001,iou=.7,max_det=300,rect=True,augment=False,
                    agnostic_nms=False,classes=None,quantize=None).items():
        if getattr(predictor.args,k) != v:
            raise AssertionError((k,getattr(predictor.args,k)))
    if str(predictor.device) != 'cpu' or predictor.model.fp16:
        raise AssertionError('CPU FP32 required')
    if (torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()) != (8,1,10):
        raise AssertionError('Common thread settings changed during model initialization')
    write_json(out/'runtime.json', dict(python=sys.version,platform=platform.platform(),
        torch=torch.__version__,opencv=cv2.__version__,actual_device=str(predictor.device),
        model_fp16=predictor.model.fp16,checkpoint_sha256=detector.checkpoint_sha256,
        torch_intra=8,torch_interop=1,opencv_threads=10,opencv_opencl=cv2.ocl.useOpenCL(),
        backend=type(predictor.model.backend).__module__+'.'+type(predictor.model.backend).__name__,
        nms_backend='torchvision.ops.nms' if 'torchvision' in sys.modules else 'ultralytics.TorchNMS.nms',
        component_devices={'capture_decode':'OpenCV CPU','buffer':'CPU Python',
            'inference':'PyTorch CPU FP32','flow':'OpenCV CPU' if processor=='practical-fixed2' else 'not used',
            'tracking':'NumPy/SciPy CPU' if processor=='practical-fixed2' else 'not used'},
        timestamp_scope='mapped replay source deadlines, not camera sensor timestamps'))
    capture = TimestampReplayCapture(rows, LatestFrameBuffer(fifo=args.policy=='fifo'), args.fps)
    if processor=='practical-fixed2':
        from .streaming_fixed2 import StreamingFixed2,SourceCadence
        pipe=StreamingFixed2(cfg,detector,SourceCadence.from_rows(rows))
    else:
        pipe = AdaptivePipeline(cfg,detector)
    try:
        pred, detail, records, summary = consume(capture, pipe, deadline_ns=args.deadline_ns)
        full_keys={r['key'] for r in records if r['decision']=='FULL'}
        assert len(forwards)==len(full_keys)+3 and all(x==expected_shape for x in forwards)
        assert all(r['decision'] in ['FULL','SKIP'] for r in records)
        if processor=='dense':
            assert len(full_keys)==len(records)
        assert len(records)<=300
        if args.policy=='latest':
            assert all(r['frame_id']==r['newest_published_id_at_selection'] for r in records)
        ref = np.load(args.reference,allow_pickle=False)
        ref_details = np.load(args.reference_details,allow_pickle=False)
        for result, archive in [(pred,ref),(detail,ref_details)]:
            for k,v in result.items():
                if k not in full_keys:
                    continue
                expected = archive[k]
                if v.shape!=expected.shape or v.dtype!=expected.dtype or v.tobytes()!=expected.tobytes():
                    raise AssertionError('Archived Dense fidelity failed: '+k)
        ref.close();ref_details.close()
        np.savez_compressed(out/'preds.npz',**pred)
        np.savez_compressed(out/'prediction_details.npz',**detail)
        with (out/'results.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        write_json(out/'capture_events.json',capture.events)
        for key in ['capture_to_result_ms','frame_age_at_inference_ms','queue_wait_ms','service_ms','decode_ms']:
            summary[key] = distribution([r[key] for r in records])
        summary.update(processed=len(records),warmup_requests=3,total_requests=len(forwards),
            prediction_bytes_exact=processor=='dense',prediction_details_bytes_exact=processor=='dense',
            full_prediction_bytes_exact=True,full_detail_bytes_exact=True,processor=processor,
            measured_yolo_calls=len(full_keys),flow_steps=sum(r['flow_attempted'] for r in records),
            accepted_flow_steps=sum(r['flow_accepted'] for r in records),
            forced_full=sum(r['forced_full'] for r in records),
            gap_triggered_full=sum(r['gap_triggered_full'] for r in records),
            state_resets=sum(r['state_reset'] for r in records),
            newest_at_selection_exact=args.policy=='latest',fidelity_reference=args.reference,
            source_backend=capture.backend,policy=args.policy,target_fps=args.fps,
            actual_source_span_fps=(len(capture.events)-1)*1e9/(capture.events[-1]['capture_ns']-capture.events[0]['capture_ns']),
            actual_publication_span_fps=(len(capture.events)-1)*1e9/(capture.events[-1]['published_ns']-capture.events[0]['published_ns']),
            processed_fps_through_drain=len(records)*1e9/(records[-1]['result_ready_ns']-capture.epoch_ns),
            post_eof_drain_ms=max(0,(records[-1]['result_ready_ns']-capture.finished_ns)/1e6),
            publish_lateness_ms=distribution([e['publish_lateness_ms'] for e in capture.events]))
        write_json(out/'COMPLETE.json',summary)
        print(json.dumps(dict(out=str(out),processed=len(records),replaced=summary['buffer']['replaced'],
                              p95_ms=summary['capture_to_result_ms']['p95'])),flush=True)
    except BaseException as exc:
        write_json(out/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),
            total_requests=len(forwards),buffer=capture.buffer.stats(),capture_error=repr(capture.error)))
        # Preserve timestamps of a partial/failed attempt; never rerun it silently.
        if not (out/'capture_events.json').exists():
            write_json(out/'capture_events.json',capture.events)
        raise
    finally:
        handle.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--config',required=True)
    p.add_argument('--processor',choices=['dense','practical-fixed2'],default='dense')
    p.add_argument('--policy',choices=['fifo','latest'],required=True)
    p.add_argument('--fps',type=float,choices=[15,90],required=True)
    p.add_argument('--reference',required=True);p.add_argument('--reference-details',required=True)
    p.add_argument('--deadline-ns',type=int,required=True);p.add_argument('--out',required=True)
    run(p.parse_args())


if __name__=='__main__':
    main()
