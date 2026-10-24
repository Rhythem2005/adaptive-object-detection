"""ONE approved Branch B FIFO/latest replay experiment, immutable plan and outputs.

prepare inspects/hashes inputs but performs no inference. run executes eight fresh
workers exactly once; scoring/verification never overlaps a timed worker. No
sweep, trial, automatic retry or tracker experiment is provided.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from adaptive_pipeline.run_stream import distribution, write_json

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'runs/branch_b_latest_frame_v1'
E=BASE/'runs/argoverse_hd_real_runtime_E_v4'
REFERENCE=BASE/'runs/argoverse_hd_benchmark_v1/baseline'
ORDERS=[(1,15,'fifo'),(1,15,'latest'),(1,90,'latest'),(1,90,'fifo'),
        (2,90,'fifo'),(2,90,'latest'),(2,15,'latest'),(2,15,'fifo')]


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def protected():
    exempt={'extracted/adaptive_pipeline/adaptive_framing/capture.py',
            'extracted/adaptive_pipeline/adaptive_framing/frame_manager.py'}
    initial=load(ROOT/'preexisting_sha256.json')
    for path,digest in initial.items():
        if path not in exempt and sha(BASE/path)!=digest:
            raise AssertionError('Protected preexisting file changed: '+path)
    return dict(checked=len(initial)-len(exempt),permitted_stream_component_changes=sorted(exempt))


def prepare():
    assert ROOT.is_dir() and not (ROOT/'execution_plan.json').exists()
    protected()
    frames=load(E/'frames.json')[:300]
    assert all(r['video']=='sequence_00' and r['frame_index']==i for i,r in enumerate(frames))
    source=load(BASE/'data/argoverse_hd/annotations.json')
    byid={im['id']:im for im in source['images']}
    manifest=[]
    for r in frames:
        assert sha(r['image'])==r['image_sha256']
        im=byid[r['image_id']]
        assert im['fid']==r['frame_index'] and im['sid']==0
        pts=int(re.search(r'_(\d+)\.jpg$',im['original_name']).group(1))
        manifest.append(dict(r,source_id=r['video'],frame_id=r['frame_index'],
                             source_timestamp_ns=pts,shape=[1200,1920,3]))
    assert all(b['source_timestamp_ns']>a['source_timestamp_ns'] for a,b in zip(manifest,manifest[1:]))
    write_json(ROOT/'manifest.json',manifest)
    config=load(E/'configs/A.json')
    write_json(ROOT/'config.json',config)
    gt=load(E/'ground_truth.json');ids={r['image_id'] for r in frames}
    gt['images']=[im for im in gt['images'] if im['id'] in ids]
    gt['annotations']=[a for a in gt['annotations'] if a['image_id'] in ids]
    write_json(ROOT/'ground_truth.json',gt)
    assert sha(config['model_path'])=='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
    write_json(ROOT/'execution_plan.json',dict(status='APPROVED_SINGLE_BOUNDED_EXPERIMENT',
        frames=300,sequence='sequence_00',source_frame_ids=[0,299],native_timestamp_unit='ns',
        rates=[15,90],rates_are_scaled_replay_not_native_camera=True,orders=ORDERS,
        maximum_measured_requests=2400,warmups_per_pass=3,maximum_total_requests=2424,
        maximum_wall_seconds=600,maximum_process_rss_mib=4096,maximum_output_bytes=250_000_000,
        common_cpu=dict(torch_intra=8,torch_interop=1,opencv=10,opencl=False),
        success_criteria=dict(exact_prediction_and_detail_fidelity=True,exact_accounting=True,
            latest_always_newest_at_atomic_selection=True,latest_max_pending=1,
            normal_latest_overwrites=0,normal_latest_p95_at_most_fifo_plus_ms=5,
            overload_p95_latency_reduction_at_least=.5),
        source_validity=dict(publication_span_fps_relative_error_at_most=.03,
            p95_decode_start_lateness_at_most_periods=1,max_decode_start_lateness_at_most_periods=2),
        accuracy='Only common source-frame populations compare detector AP. Separate replay-time-aligned held-output AP is a latency-aware diagnostic, not official streaming AP.',
        entrypoint='adaptive_pipeline.run_stream',
        command='PYTHONPATH=extracted:. .venv/bin/python -m analysis.latest_frame_validation run',
        no_retry=True,tracker_not_used=True,no_960_experiment=True))
    paths=list((BASE/'extracted/adaptive_pipeline').rglob('*.py'))
    paths += [Path(__file__).resolve(),BASE/'tests/test_streaming.py',ROOT/'manifest.json',ROOT/'config.json',ROOT/'execution_plan.json',ROOT/'ground_truth.json']
    write_json(ROOT/'execution_sources.json',{str(p):sha(p) for p in paths})
    write_json(ROOT/'PREPARED.json',dict(preservation=protected(),inputs_verified=True,new_inference=0))
    print('Prepared: 8 passes; 300 frames each; <=2424 requests, 600 seconds, 4 GiB RSS.',flush=True)


def verify_execution_sources():
    for path,digest in load(ROOT/'execution_sources.json').items():
        assert sha(path)==digest,path


def score_arrays(pred,gt,ids,out):
    from adaptive_pipeline.evaluation.coco import prediction_records,make_coco,run_coco,coco_metrics,fixed_operating_point
    records,_=prediction_records(pred,{i:i for i in range(8)})
    cg,cd=make_coco(gt,records)
    ev,log=run_coco(cg,cd,image_ids=ids)
    result=dict(COCO=coco_metrics(ev),fixed_conf25=fixed_operating_point(ev))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'metrics.json',result)
    (out/'coco_stdout.txt').write_text(log)
    np.savez_compressed(out/'official_tensors.npz',precision=ev.eval['precision'],recall=ev.eval['recall'],scores=ev.eval['scores'])
    return result


def summarize_pass(out):
    result=load(out/'COMPLETE.json');events=load(out/'capture_events.json')
    with (out/'results.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for key in ['frame_id','capture_ns','decode_start_ns','decode_end_ns','published_ns','selected_ns',
                    'newest_published_id_at_selection','pending_after_selection','inference_start_ns','result_ready_ns']:
            r[key]=int(r[key])
        for key in ['capture_to_result_ms','frame_age_at_inference_ms','queue_wait_ms','service_ms','decode_ms']:
            r[key]=float(r[key])
    assert len(events)==300 and [e['frame_id'] for e in events]==list(range(300))
    assert result['buffer']['captured']==300
    assert result['buffer']['consumed']+result['buffer']['replaced']==300
    assert result['buffer']['pending']==0 and result['buffer']['pending_bytes']==0
    assert result['buffer']['consumed']==len(rows)
    assert rows[-1]['frame_id']==299
    if result['policy']=='fifo':
        assert result['buffer']['replaced']==0 and [r['frame_id'] for r in rows]==list(range(300))
    else:
        assert result['buffer']['peak_pending']==1
        assert all(r['frame_id']==r['newest_published_id_at_selection'] for r in rows)
    for e in events:
        assert e['capture_ns']<=e['decode_start_ns']<=e['decode_end_ns']<=e['published_ns']
    eventmap={e['frame_id']:e for e in events}
    for r in rows:
        e=eventmap[r['frame_id']]
        for k in ['capture_ns','decode_start_ns','decode_end_ns','published_ns']:assert r[k]==e[k]
        assert r['published_ns']<=r['selected_ns']<=r['inference_start_ns']<=r['result_ready_ns']
        assert r['capture_to_result_ms']==(r['result_ready_ns']-r['capture_ns'])/1e6
    begin=events[0]['capture_ns'];end=events[-1]['capture_ns']
    # Common active source window, separately from EOF drain.
    active=[r for r in rows if r['result_ready_ns']<=end]
    result['active_output_count']=len(active)
    result['active_output_fps']=len(active)*1e9/(end-begin)
    result['drain_output_count']=len(rows)-len(active)
    result['source_duration_s']=(end-begin)/1e9
    result['decode_start_lateness_ms']=distribution([(e['decode_start_ns']-e['capture_ns'])/1e6 for e in events])
    result['publication_interval_ms']=distribution([(b['published_ns']-a['published_ns'])/1e6 for a,b in zip(events,events[1:])])
    deadlines=np.array([e['capture_ns'] for e in events],dtype=np.int64)
    result['source_frames_behind_at_delivery']=distribution([
        max(0,int(np.searchsorted(deadlines,r['result_ready_ns'],side='right'))-1-r['frame_id']) for r in rows])
    result['published_frames_behind_at_delivery']=distribution([
        max(0,sum(e['published_ns']<=r['result_ready_ns'] for e in events)-1-r['frame_id']) for r in rows])
    period_ms=1000/result['target_fps']
    validity=dict(publication_rate=abs(result['actual_publication_span_fps']/result['target_fps']-1)<=.03,
        p95_release_lateness=result['decode_start_lateness_ms']['p95']<=period_ms,
        max_release_lateness=result['decode_start_lateness_ms']['max']<=2*period_ms,
        peak_rss=result['peak_rss_mib']<=4096)
    result['source_validity']=validity
    result['valid']=all(validity.values())
    return result,rows,events


def report(deadline_ns):
    started=time.perf_counter();cpu=time.process_time()
    gt=load(ROOT/'ground_truth.json');manifest=load(ROOT/'manifest.json')
    results={};data={}
    for rep,fps,policy in ORDERS:
        name=f'r{rep}_{fps}_{policy}';s,r,e=summarize_pass(ROOT/'passes'/name)
        results[name]=s;data[name]=(r,e)
    reference=np.load(REFERENCE/'preds.npz',allow_pickle=False)
    dense={i:reference[f'sequence_00/{i}'] for i in range(300)}
    reference.close()
    dense_metrics=score_arrays(dense,gt,list(range(300)),ROOT/'accuracy/dense_same300')
    pairs={};aligned={}
    for rep in [1,2]:
        for fps in [15,90]:
            if time.perf_counter_ns()>deadline_ns:raise TimeoutError('Deadline during scoring')
            fk=f'r{rep}_{fps}_fifo';lk=f'r{rep}_{fps}_latest'
            f,l=results[fk],results[lk]
            fr,_=data[fk];lr,_=data[lk]
            common=sorted({r['frame_id'] for r in fr}&{r['frame_id'] for r in lr})
            # Exact parity already gates every emitted prediction/detail. On the
            # same image IDs, both methods and Dense therefore have identical AP.
            m=dense_metrics if len(common)==300 else score_arrays({i:dense[i] for i in common},gt,common,ROOT/f'accuracy/common_r{rep}_{fps}')
            write_json(ROOT/f'common_ids_r{rep}_{fps}.json',common)
            reduction=1-l['capture_to_result_ms']['p95']/f['capture_to_result_ms']['p95']
            passed=(f['valid'] and l['valid'] and
                (reduction>=.5 if fps==90 else l['buffer']['replaced']==0 and
                 l['capture_to_result_ms']['p95']<=f['capture_to_result_ms']['p95']+5))
            pairs[f'r{rep}_{fps}']=dict(source_valid=f['valid'] and l['valid'],criteria_passed=passed,
                p95_latency_reduction=reduction,common_frames=len(common),identical_common_frame_COCO=m,
                fifo_active_fps=f['active_output_fps'],latest_active_fps=l['active_output_fps'])
    # At each SOURCE arrival timestamp use the last completed output, unchanged.
    # Empty before the first result. Same 300 query images for all policies; this
    # measures stale/absent delivery, not intrinsic quality on differing subsets.
    for name,(rows,events) in data.items():
        if time.perf_counter_ns()>deadline_ns:raise TimeoutError('Deadline during aligned scoring')
        ready=np.array([r['result_ready_ns'] for r in rows],dtype=np.int64)
        arrays={};ages=[];empty=0;assignments=[]
        for e in events:
            j=int(np.searchsorted(ready,e['capture_ns'],side='right'))-1
            frame=e['frame_id']
            if j<0:
                arrays[frame]=np.zeros((0,6),np.float32);empty+=1
                assignments.append(dict(query_frame_id=frame,result_source_frame_id=None))
            else:
                old=rows[j]['frame_id'];arrays[frame]=dense[old]
                ages.append((e['capture_ns']-rows[j]['capture_ns'])/1e6)
                assignments.append(dict(query_frame_id=frame,result_source_frame_id=old))
        m=score_arrays(arrays,gt,list(range(300)),ROOT/f'accuracy/replay_time_aligned/{name}')
        aligned[name]=dict(metrics=m,no_result_queries=empty,queries=300,
            visible_source_age_ms=distribution(ages),
            protocol='Unchanged last completed boxes at common source-arrival query times; COCO bbox scorer; NOT official streaming AP')
        write_json(ROOT/f'accuracy/replay_time_aligned/{name}/assignments.json',assignments)
    write_json(ROOT/'summary.json',dict(passes=results,pairs=pairs,replay_time_aligned=aligned,
        dense_same300=dense_metrics,all_pair_criteria_passed=all(x['criteria_passed'] for x in pairs.values()),
        scoring_wall_s=time.perf_counter()-started,scoring_cpu_s=time.process_time()-cpu))


def run():
    assert not (ROOT/'EXECUTION_STARTED.json').exists(),'No retries or repeated experiment runs'
    verify_execution_sources();protected()
    for r in load(ROOT/'manifest.json'):assert sha(r['image'])==r['image_sha256']
    begin=time.perf_counter_ns();deadline_ns=begin+600_000_000_000
    write_json(ROOT/'EXECUTION_STARTED.json',dict(start_ns=begin,deadline_ns=deadline_ns,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    total=0
    for rep,fps,policy in ORDERS:
        name=f'r{rep}_{fps}_{policy}';out=ROOT/'passes'/name
        cmd=[str(BASE/'.venv/bin/python'),'-m','adaptive_pipeline.run_stream',
            '--manifest',str(ROOT/'manifest.json'),'--config',str(ROOT/'config.json'),
            '--policy',policy,'--fps',str(fps),'--reference',str(REFERENCE/'preds.npz'),
            '--reference-details',str(REFERENCE/'prediction_details.npz'),
            '--deadline-ns',str(deadline_ns),'--out',str(out)]
        verify_execution_sources()
        env=dict(os.environ,PYTHONPATH=str(BASE/'extracted')+':'+str(BASE))
        print('START',name,flush=True)
        t=time.perf_counter();remaining=(deadline_ns-time.perf_counter_ns())/1e9
        if remaining<=0:raise TimeoutError('Global 10-minute limit')
        proc=subprocess.run(cmd,cwd=BASE,env=env,capture_output=True,text=True,timeout=remaining)
        # Worker creates out; startup failures without out are preserved as well.
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'SUBPROCESS.json',dict(command=cmd,returncode=proc.returncode,
            wall_s=time.perf_counter()-t,stdout=proc.stdout,stderr=proc.stderr))
        print(proc.stdout,flush=True)
        if proc.returncode:raise RuntimeError('Worker failed; no retry: '+name+'\n'+proc.stderr)
        total+=load(out/'COMPLETE.json')['total_requests']
        assert total<=2424
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=250_000_000
        interim,_,_=summarize_pass(out)
        if not interim['valid']:
            raise RuntimeError('Source pacing invalid; no performance claim or retry: '+name+' '+str(interim['source_validity']))
    write_json(ROOT/'INFERENCE_COMPLETE.json',dict(actual_requests=total,warmup_requests=24,
        measured_requests=total-24,elapsed_s=(time.perf_counter_ns()-begin)/1e9))
    report(deadline_ns)
    verify_execution_sources()
    audit=protected()
    assert (time.perf_counter_ns()-begin)/1e9<=600
    write_json(ROOT/'COMPLETE.json',dict(actual_requests=total,elapsed_s=(time.perf_counter_ns()-begin)/1e9,
        protected=audit,all_pair_criteria_passed=load(ROOT/'summary.json')['all_pair_criteria_passed']))
    write_json(ROOT/'artifact_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='execution.log'})
    print('COMPLETE',load(ROOT/'COMPLETE.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'])
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        try:run()
        except BaseException as exc:
            if not (ROOT/'FAILED.json').exists():write_json(ROOT/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
            raise


if __name__=='__main__':main()
