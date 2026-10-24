"""Fresh-process complete-sequence workers using unchanged production algorithms."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from analysis.argoverse_hd_phase2.common import load, dump, config_from_dict, sha
from adaptive_pipeline.run_stream import consume, distribution, peak_mib


def save_csv(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['fifo','latest','merged','offline'],required=True)
    p.add_argument('--fps',type=int,choices=[15,30,60,90],default=15)
    p.add_argument('--out',required=True);p.add_argument('--reference')
    p.add_argument('--reference-details');p.add_argument('--memory-mib',type=int,default=4096)
    p.add_argument('--max-seconds',type=int,default=180)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    dump(out/'STARTED.json',dict(command=sys.argv,pid=os.getpid(),unix_time=time.time()))
    # Same import order and thread controls as archived streaming experiments.
    import torch
    from ultralytics import YOLO
    from adaptive_pipeline.detection.detector import YOLODetector
    from adaptive_pipeline.pipeline import AdaptivePipeline
    from adaptive_pipeline.adaptive_framing.capture import TimestampReplayCapture
    from adaptive_pipeline.adaptive_framing.frame_manager import LatestFrameBuffer
    from adaptive_pipeline.streaming_fixed2 import StreamingFixed2,SourceCadence
    from analysis.argoverse_hd_runtime_E.run import ForwardCounter,validate_predictor,timed_step
    rows=load(a.manifest);cfg=config_from_dict(load(a.config))
    assert len(rows)>=2 and len({r['source_id'] for r in rows})==1
    assert [r['frame_id'] for r in rows]==list(range(len(rows)))
    for r in rows:assert sha(r['image'])==r['image_sha256'],r['image']
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    cv2.setNumThreads(10);cv2.ocl.setUseOpenCL(False)
    assert sha(cfg.model_path)=='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
    detector=YOLODetector(cfg);counter=ForwardCounter(detector.model.model)
    first=cv2.imread(rows[0]['image']);assert first.shape==(1200,1920,3)
    for _ in range(3):detector.detect(first)
    del first
    validate_predictor(detector.model.predictor)
    assert (torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads())==(8,1,10)
    dump(out/'runtime.json',dict(python=sys.version,torch=torch.__version__,opencv=cv2.__version__,
        checkpoint_sha256=detector.checkpoint_sha256,threads=[8,1,10],device='cpu',dtype='float32',
        config=load(a.config),manifest_sha256=sha(a.manifest)))
    capture=None;pred={};detail={};records=[]
    deadline=time.perf_counter_ns()+a.max_seconds*1_000_000_000
    try:
        if a.mode=='offline':
            pipe=AdaptivePipeline(cfg,detector);pipe.reset()
            w,c=time.perf_counter(),time.process_time()
            for row in rows:
                if time.perf_counter_ns()>deadline:raise TimeoutError('Worker deadline')
                if peak_mib()>a.memory_mib:raise MemoryError('RSS guard')
                value,dt,rec=timed_step(pipe,row)
                pred[row['key']]=value;detail[row['key']]=dt.copy();records.append(rec)
            summary=dict(loop_wall_s=time.perf_counter()-w,loop_cpu_s=time.process_time()-c,
                processed=len(records),peak_rss_mib=peak_mib())
        else:
            # FIFO capacity covers the complete finite population; no drop/backpressure.
            capture=TimestampReplayCapture(rows,LatestFrameBuffer(fifo=a.mode=='fifo',max_pending=len(rows)),a.fps)
            pipe=StreamingFixed2(cfg,detector,SourceCadence.from_rows(rows)) if a.mode=='merged' else AdaptivePipeline(cfg,detector)
            pred,detail,records,summary=consume(capture,pipe,deadline_ns=deadline,memory_mib=a.memory_mib)
            events=capture.events;assert len(events)==len(rows)
            assert records[-1]['frame_id']==len(rows)-1
            if a.mode=='fifo':
                assert [r['frame_id'] for r in records]==list(range(len(rows)))
                assert summary['buffer']['replaced']==0
            else:
                assert summary['buffer']['peak_pending']==1
                publications=np.array([e['published_ns'] for e in events],np.int64)
                for r in records:
                    j=np.searchsorted(publications,r['selected_ns'],side='right')-1
                    assert r['frame_id']==events[j]['frame_id']==r['newest_published_id_at_selection']
            begin,end=events[0]['capture_ns'],events[-1]['capture_ns']
            lateness=[(e['decode_start_ns']-e['capture_ns'])/1e6 for e in events]
            pubfps=(len(events)-1)*1e9/(events[-1]['published_ns']-events[0]['published_ns'])
            late=distribution(lateness);period=1000/a.fps
            validity=dict(publication_rate=abs(pubfps/a.fps-1)<=.03,
                p95_release_lateness=late['p95']<=period,max_release_lateness=late['max']<=2*period)
            summary.update(source_validity=validity,valid=all(validity.values()),
                source_duration_s=(end-begin)/1e9,actual_publication_span_fps=pubfps,
                decode_start_lateness_ms=late,active_output_count=sum(r['result_ready_ns']<=end for r in records),
                active_output_fps=sum(r['result_ready_ns']<=end for r in records)*1e9/(end-begin),
                output_fps_through_drain=len(records)*1e9/(records[-1]['result_ready_ns']-begin),
                post_eof_drain_ms=max(0,(records[-1]['result_ready_ns']-capture.finished_ns)/1e6))
            for k in ['capture_to_result_ms','frame_age_at_inference_ms','queue_wait_ms','service_ms','decode_ms']:
                summary[k]=distribution([r[k] for r in records])
            dump(out/'capture_events.json',events)
        full_keys={r['key'] for r in records if r['decision']=='FULL'}
        assert counter.count==len(full_keys)+3
        assert counter.summary()==[dict(shape=[1,3,416,640],dtype='torch.float32',device='cpu',count=counter.count)]
        fidelity=None
        if a.reference:
            fidelity=True
            for actual,filename in [(pred,a.reference),(detail,a.reference_details)]:
                if not filename:continue
                with np.load(filename,allow_pickle=False) as expected:
                    for key in full_keys:
                        x,y=actual[key],expected[key]
                        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),key
        np.savez_compressed(out/'preds.npz',**pred);np.savez_compressed(out/'prediction_details.npz',**detail)
        save_csv(out/'results.csv',records)
        summary.update(mode=a.mode,target_fps=a.fps,source_frames=len(rows),processed=len(records),
            measured_yolo_calls=len(full_keys),warmup_calls=3,total_forwards=counter.count,
            full_prediction_fidelity=fidelity,flow_steps=sum(int(r.get('flow_attempted',0)) for r in records),
            accepted_flow_steps=sum(r['decision']=='SKIP' for r in records),
            forced_full=sum(int(r.get('forced_full',0)) for r in records),
            gap_triggered_full=sum(int(r.get('gap_triggered_full',0)) for r in records),
            fallback_reasons={reason:sum(reason in str(r.get('fallback_reasons','')) for r in records)
                for reason in sorted({str(r.get('fallback_reasons','')) for r in records if r.get('fallback_reasons')})})
        dump(out/'COMPLETE.json',summary)
        print(json.dumps(dict(out=str(out),processed=len(records),calls=len(full_keys),valid=summary.get('valid',True))),flush=True)
    except BaseException as exc:
        # A failed consume does not return partial predictions; retain capture trace and error.
        dump(out/'INCOMPLETE.json',dict(error=repr(exc),peak_rss_mib=peak_mib(),total_forwards=counter.count,
            buffer=capture.buffer.stats() if capture else None,partial_predictions_available=False))
        if capture and not (out/'capture_events.json').exists():dump(out/'capture_events.json',capture.events)
        raise


if __name__=='__main__':main()
