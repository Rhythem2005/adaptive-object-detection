"""One approved integration experiment. No tuning, retry or extra inference arms."""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from adaptive_pipeline.run_stream import write_json,distribution
from analysis.argoverse_hd_phase2.common import sha,load,config_from_dict,dump
from analysis.latest_frame_validation import summarize_pass

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'runs/branch_b_latest_frame_fixed2_v1'
OLD=BASE/'runs/branch_b_latest_frame_v1'
E=BASE/'runs/argoverse_hd_real_runtime_E_v4'
REFERENCE=BASE/'runs/argoverse_hd_benchmark_v1/baseline'
PRACTICAL=BASE/'runs/argoverse_hd_next_research_v1/arms/F_fastpath'
BLOCKS=[(1,15,'ABC'),(1,90,'BCA'),(2,90,'ACB'),(2,15,'CBA')]
ARMS={'A':dict(policy='fifo',processor='dense',config='A'),
      'B':dict(policy='latest',processor='dense',config='A'),
      'C':dict(policy='latest',processor='practical-fixed2',config='C')}


def protected():
    initial=load(ROOT/'preexisting_sha256.json')
    allowed={'extracted/adaptive_pipeline/run_stream.py'}
    for path,digest in initial.items():
        if path not in allowed and sha(BASE/path)!=digest:
            raise AssertionError('Protected file changed: '+path)
    return dict(files_checked=len(initial)-len(allowed),allowed_source_changes=sorted(allowed))


def verify_sources():
    for path,digest in load(ROOT/'execution_sources.json').items():
        assert sha(path)==digest,path


def cached_regression():
    """Original offline Practical, 300 consecutive frames, archived detections.

    No model construction, no new inference, no new candidate experiment.
    """
    import cv2
    from adaptive_pipeline.detection.detector import YOLODetector
    from adaptive_pipeline.pipeline import AdaptivePipeline
    from analysis.argoverse_hd_phase2.replay import state_digest
    assert not (ROOT/'CACHED_REGRESSION.json').exists()
    begin=time.perf_counter();cpu=time.process_time()
    cv2.setNumThreads(10);cv2.ocl.setUseOpenCL(False)
    cfg=config_from_dict(load(E/'configs/B.json'))
    dense=np.load(REFERENCE/'preds.npz',allow_pickle=False)
    expected=np.load(PRACTICAL/'preds.npz',allow_pickle=False)
    details=np.load(PRACTICAL/'prediction_details.npz',allow_pickle=False)
    states=load(PRACTICAL/'state_hashes.json')
    class Cache:
        def __init__(self):self.cfg=cfg;self.key=None;self.calls=0
        def detect(self,frame):self.calls+=1;return dense[self.key].copy(),0.
        full_gflops=YOLODetector.full_gflops
    detector=Cache();pipe=AdaptivePipeline(cfg,detector)
    rows=load(OLD/'manifest.json')
    for r in rows:
        assert sha(r['image'])==r['image_sha256']
        detector.key=r['key'];frame=cv2.imread(r['image'])
        actual,rec=pipe.step(frame,r['frame_id'])
        for a,b in [(actual,expected[r['key']]),(pipe.last_prediction_details,details[r['key']])]:
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),r['key']
        assert state_digest(pipe.tracker)==states[r['key']],r['key']
        assert rec['decision']==('FULL' if r['frame_id']%2==0 else 'SKIP')
    dense.close();expected.close();details.close()
    assert detector.calls==150
    write_json(ROOT/'CACHED_REGRESSION.json',dict(frames=300,predictions_details_and_state_hashes_byte_exact=True,
        detector_cache_lookups=150,new_YOLO_forwards=0,wall_s=time.perf_counter()-begin,cpu_s=time.process_time()-cpu))
    print('Cached offline Practical regression: 300 exact predictions/details/state hashes; zero inference.',flush=True)


def prepare():
    assert not (ROOT/'execution_plan.json').exists()
    protected()
    rows=load(OLD/'manifest.json')
    assert len(rows)==300 and [r['frame_id'] for r in rows]==list(range(300))
    assert {r['source_id'] for r in rows}=={'sequence_00'}
    for r in rows:assert sha(r['image'])==r['image_sha256']
    shutil.copyfile(OLD/'manifest.json',ROOT/'manifest.json')
    shutil.copyfile(OLD/'ground_truth.json',ROOT/'ground_truth.json')
    (ROOT/'configs').mkdir()
    for arm,path in [('A',E/'configs/A.json'),('C',E/'configs/B.json')]:
        shutil.copyfile(path,ROOT/f'configs/{arm}.json')
    cfg=load(ROOT/'configs/C.json')
    assert sha(cfg['model_path'])=='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
    intervals=[b['source_timestamp_ns']-a['source_timestamp_ns'] for a,b in zip(rows,rows[1:])]
    write_json(ROOT/'execution_plan.json',dict(status='APPROVED_ONE_INTEGRATION_CANDIDATE',
        blocks=BLOCKS,arms=ARMS,source='sequence_00/0..299',frames_per_pass=300,
        maximum_YOLO_forwards=3636,warmups_per_pass=3,warmups_total=36,
        maximum_wall_seconds=600,maximum_rss_mib=4096,maximum_output_bytes=250000000,
        interblock_pause_seconds=10,torch_intra=8,torch_interop=1,opencv_threads=10,
        device='cpu',dtype='float32',no_tuning=True,no_retry=True,
        source_validity=dict(publication_fps_relative_error=.03,p95_release_lateness_periods=1,max_release_lateness_periods=2),
        source_cadence_ns=dict(minimum=min(intervals),maximum=max(intervals),median=float(np.median(intervals))),
        safety=dict(source_age_limit=cfg['k_max'],limit_is_existing_policy_not_LK_guarantee=True,
            reject_any_visible_flow_failure=True,reject_invalid_geometry=True,reject_scale_saturation=True,
            schedule_counts_consumed_steps=True,tracker_units='verified original source frames'),
        criteria=dict(normal_calls_reduction=.30,normal_cpu_reduction=.20,
            overload_calls_reduction=.10,overload_cpu_reduction=.05,max_paired_cpu_increase=.05,
            p95_allowed_extra_ms=5,p95_allowed_factor=1.15,maximum_latency_ms=150,
            aligned_AP_mean_loss_relative=.10,aligned_AP_mean_loss_absolute=.01,
            aligned_AP_max_paired_loss_relative=.25,aligned_AP_max_paired_loss_absolute=.01,
            FP25_mean_factor=1.10,recall25_mean_factor=.90),
        command='PYTHONPATH=extracted:. .venv/bin/python -m analysis.latest_frame_fixed2_validation run'))
    paths=list((BASE/'extracted/adaptive_pipeline').rglob('*.py'))
    paths += [Path(__file__).resolve(),BASE/'analysis/latest_frame_validation.py',BASE/'tests/test_streaming_fixed2.py',
        ROOT/'manifest.json',ROOT/'ground_truth.json',ROOT/'execution_plan.json',ROOT/'configs/A.json',ROOT/'configs/C.json']
    write_json(ROOT/'execution_sources.json',{str(p):sha(p) for p in paths})
    write_json(ROOT/'PREPARED.json',dict(protected=protected(),new_inference=0))
    print('Prepared: 12 fresh workers; <=3636 forwards, 600 s, 4096 MiB/process.',flush=True)


def pass_name(rep,fps,arm):return f'r{rep}_{fps}_{arm}'


def validate_pass(out):
    summary,records,events=summarize_pass(out)
    for row in records:
        for key in ['consumed_index','source_gap','forced_full','gap_triggered_full','state_reset','flow_attempted','flow_accepted']:
            row[key]=int(row[key])
    assert [r['consumed_index'] for r in records]==list(range(len(records)))
    assert [r['source_gap'] for r in records]==[1]+[b['frame_id']-a['frame_id'] for a,b in zip(records,records[1:])]
    full=sum(r['decision']=='FULL' for r in records)
    assert full==summary['measured_yolo_calls'] and full+3==summary['total_requests']
    assert summary['flow_steps']==sum(r['flow_attempted'] for r in records)
    assert summary['accepted_flow_steps']==sum(r['decision']=='SKIP' for r in records)
    assert summary['full_prediction_bytes_exact'] and summary['full_detail_bytes_exact']
    if summary['processor']=='practical-fixed2':
        since=None
        for r in records:
            nominal='FULL' if since is None or since+1>=2 else 'SKIP'
            assert r['nominal_decision']==nominal
            assert int(r['forced_full'])==int(r['decision']=='FULL' and nominal=='SKIP')
            assert int(r['gap_triggered_full'])==int(r['forced_full'] and r['source_gap']>1)
            if r['decision']=='SKIP':
                assert not r['fallback_reasons'] and r['flow_accepted']==1
            since=0 if r['decision']=='FULL' else since+1
    summary.update(cpu_seconds_per_source=summary['loop_cpu_s']/300,
        cpu_seconds_per_consumed=summary['loop_cpu_s']/len(records),
        yolo_fraction_consumed=full/len(records),source_gap_counts=dict(collections.Counter(r['source_gap'] for r in records)),
        fallback_reason_counts=dict(collections.Counter(reason for r in records for reason in r['fallback_reasons'].split('|') if reason)),
        reset_reason_counts=dict(collections.Counter(reason for r in records for reason in r['state_reset_reasons'].split('|') if reason)),
        by_decision={decision:dict(count=len(part),service_ms=distribution([r['service_ms'] for r in part]),
            capture_to_result_ms=distribution([r['capture_to_result_ms'] for r in part]))
            for decision in ['FULL','SKIP'] if (part:=[r for r in records if r['decision']==decision])})
    deadlines=np.array([e['published_ns'] for e in events],np.int64)
    for r in records:
        newest=int(np.searchsorted(deadlines,r['selected_ns'],side='right'))-1
        if summary['policy']=='latest':assert r['frame_id']==events[newest]['frame_id']
    return summary,records,events


def held_outputs(records,events,predictions):
    ready=np.array([r['result_ready_ns'] for r in records],dtype=np.int64)
    arrays={};assign=[];last_full=[];full_id=None
    for row in records:
        if row['decision']=='FULL':full_id=row['frame_id']
        last_full.append(full_id)
    for event in events:
        j=int(np.searchsorted(ready,event['capture_ns'],side='right'))-1
        if j<0:
            arrays[event['frame_id']]=np.zeros((0,6),np.float32)
            assign.append(dict(query_frame_id=event['frame_id'],result_source_frame_id=None,
                output_decision=None,last_full_source_frame_id=None,visible_age_ms=None))
        else:
            row=records[j]
            arrays[event['frame_id']]=predictions[row['key']]  # Actual flow output, NEVER Dense substitution.
            assign.append(dict(query_frame_id=event['frame_id'],result_source_frame_id=row['frame_id'],
                output_decision=row['decision'],last_full_source_frame_id=last_full[j],
                visible_age_ms=(event['capture_ns']-row['capture_ns'])/1e6))
    return arrays,assign


def score_arrays(predictions,gt,frame_ids,out):
    from adaptive_pipeline.evaluation.coco import prediction_records,make_coco,run_coco,coco_metrics,fixed_operating_point
    from analysis.argoverse_hd_phase2.common import json_gzip
    imap={im['fid']:im['id'] for im in gt['images']}
    records,_=prediction_records({imap[i]:predictions[i] for i in frame_ids},{i:i for i in range(8)})
    cg,cd=make_coco(gt,records);ev,log=run_coco(cg,cd,[imap[i] for i in frame_ids])
    metrics=dict(COCO=coco_metrics(ev),fixed_conf25=fixed_operating_point(ev),fixed_conf10=fixed_operating_point(ev,.1))
    out.mkdir(parents=True,exist_ok=False);dump(out/'metrics.json',metrics)
    (out/'coco_stdout.txt').write_text(log)
    np.savez_compressed(out/'official_tensors.npz',precision=ev.eval['precision'],recall=ev.eval['recall'],scores=ev.eval['scores'])
    matched={i:dict(tp25=0,fp25=0,matched_gt25=[],matched_gt75_25=[]) for i in frame_ids}
    inverse={v:k for k,v in imap.items()}
    for r in ev.evalImgs:
        if r is None or r['aRng']!=ev.params.areaRng[0]:continue
        keep=np.asarray(r['dtScores'])>=.25
        for ti,key in [(0,'matched_gt25'),(5,'matched_gt75_25')]:
            selected=keep & ~r['dtIgnore'][ti].astype(bool)
            good=selected & (r['dtMatches'][ti]>0)
            matched[inverse[r['image_id']]][key].extend(r['dtMatches'][ti,good].astype(int).tolist())
        point=matched[inverse[r['image_id']]]
        point['tp25']+=int((keep & ~r['dtIgnore'][0].astype(bool) & (r['dtMatches'][0]>0)).sum())
        point['fp25']+=int((keep & ~r['dtIgnore'][0].astype(bool) & (r['dtMatches'][0]==0)).sum())
    json_gzip(out/'per_image_matches.json.gz',matched)
    return metrics,matched


def acceptance(passes,aligned):
    rates={}
    for fps in [15,90]:
        b=[passes[pass_name(rep,fps,'B')] for rep in [1,2]]
        c=[passes[pass_name(rep,fps,'C')] for rep in [1,2]]
        ba=[aligned[pass_name(rep,fps,'B')]['metrics'] for rep in [1,2]]
        ca=[aligned[pass_name(rep,fps,'C')]['metrics'] for rep in [1,2]]
        avg=lambda rows,key:float(np.mean([r[key] for r in rows]))
        cpu=1-avg(c,'loop_cpu_s')/avg(b,'loop_cpu_s')
        calls=1-avg(c,'measured_yolo_calls')/avg(b,'measured_yolo_calls')
        bap=np.mean([r['COCO']['AP'] for r in ba]);cap=np.mean([r['COCO']['AP'] for r in ca])
        point=lambda rows,k:float(np.mean([r['fixed_conf25'][k] for r in rows]))
        fp=lambda rows:float(np.mean([r['fixed_conf25']['counts']['FP'] for r in rows]))
        flags=dict(all_sources_valid=all(r['valid'] for r in b+c),
            detector_savings=calls>=(.30 if fps==15 else .10),
            cpu_savings=cpu>=(.20 if fps==15 else .05),
            no_large_cpu_regression=all(x['loop_cpu_s']<=1.05*y['loop_cpu_s'] for x,y in zip(c,b)),
            paired_p95=all(x['capture_to_result_ms']['p95']<=max(y['capture_to_result_ms']['p95']+5,1.15*y['capture_to_result_ms']['p95']) for x,y in zip(c,b)),
            maximum_latency=all(x['capture_to_result_ms']['max']<=150 for x in c),
            mean_aligned_AP=cap>=bap-min(.01,.10*bap),
            no_large_paired_AP_loss=all(x['COCO']['AP']>=y['COCO']['AP']-max(.01,.25*y['COCO']['AP']) for x,y in zip(ca,ba)),
            FP25=fp(ca)<=1.10*fp(ba),
            recall25=point(ca,'micro_recall')>=.90*point(ba,'micro_recall'))
        if fps==15:flags['no_normal_overwrites']=all(x['processed']==300 and x['buffer']['replaced']==0 for x in c)
        rates[str(fps)]=dict(flags=flags,all_passed=all(flags.values()),cpu_reduction=cpu,
            yolo_reduction=calls,mean_B_aligned_AP=float(bap),mean_C_aligned_AP=float(cap))
    return rates


def diagnosis(data,aligned,aligned_matches,gt,dense):
    # Metadata compatibility is audited before interpreting annotation track IDs.
    regular=[a for a in gt['annotations'] if not a.get('iscrowd')]
    byid={a['id']:a for a in regular};imfid={im['id']:im['fid'] for im in gt['images']}
    identities={};seen=set();issues=[]
    for a in regular:
        key=(a['image_id'],a.get('track'));cls=a['category_id'];track=a.get('track')
        if not isinstance(track,int) or track<0 or key in seen:issues.append(a['id'])
        seen.add(key)
        if track in identities and identities[track]!=cls:issues.append(a['id'])
        identities[track]=cls
    first={}
    if not issues:
        for a in regular:first[a['track']]=min(first.get(a['track'],10**9),imfid[a['image_id']])
    findings=dict(annotation_identity_check=dict(valid=not issues,issues=issues,
        meaning='First annotation within this prefix, not physical object arrival'),passes={})
    for name,(rows,events,predictions) in data.items():
        if not name.endswith('_C'):continue
        ids=[r['frame_id'] for r in rows];actual={r['frame_id']:predictions[r['key']] for r in rows}
        cm,cp=score_arrays(actual,gt,ids,ROOT/f'accuracy/consumed_diagnostic/{name}/actual')
        dm,dp=score_arrays({i:dense[i] for i in ids},gt,ids,ROOT/f'accuracy/consumed_diagnostic/{name}/dense_same_ids')
        buckets={}
        for r in rows:
            fid=r['frame_id'];bucket=r['decision']+'/gap'+str(r['source_gap'])
            b=buckets.setdefault(bucket,dict(frames=0,dense_TP25=0,actual_TP25=0,dense_FP25=0,actual_FP25=0,
                dense_TP_lost=0,dense_FN_also_missed=0,dense_FN_recovered=0,TP75_lost=0))
            b['frames']+=1;d=set(dp[fid]['matched_gt25']);a=set(cp[fid]['matched_gt25'])
            targets={x['id'] for x in regular if imfid[x['image_id']]==fid}
            b['dense_TP25']+=len(d);b['actual_TP25']+=len(a)
            b['dense_FP25']+=dp[fid]['fp25'];b['actual_FP25']+=cp[fid]['fp25']
            b['dense_TP_lost']+=len(d-a);b['dense_FN_also_missed']+=len(targets-d-a)
            b['dense_FN_recovered']+=len(a-d)
            b['TP75_lost']+=len(set(dp[fid]['matched_gt75_25'])-set(cp[fid]['matched_gt75_25']))
        appearance=dict(supported=not issues,newly_annotated_targets_since_last_FULL=0,missed_newly_annotated_targets=0)
        if not issues:
            assigns=aligned[name]['assignments']
            for assignment in assigns:
                fid=assignment['query_frame_id'];last=assignment['last_full_source_frame_id']
                if last is None:continue
                targets=[a for a in regular if imfid[a['image_id']]==fid and first[a['track']]>last and first[a['track']]>0]
                matches=set(aligned_matches[name][fid]['matched_gt25'])
                appearance['newly_annotated_targets_since_last_FULL']+=len(targets)
                appearance['missed_newly_annotated_targets']+=sum(a['id'] not in matches for a in targets)
        findings['passes'][name]=dict(same_consumed_population_actual=cm,same_consumed_population_dense=dm,
            error_buckets=buckets,appearance=appearance,
            limitation='Flow-frame localization/retention losses are propagation-associated; matching does not establish physical causality or isolate camera motion.')
    return findings


def report(deadline_ns):
    begin=time.perf_counter();cpu=time.process_time()
    gt=load(ROOT/'ground_truth.json');passes={};data={};aligned={};matches={}
    dense_npz=np.load(REFERENCE/'preds.npz',allow_pickle=False)
    dense={i:dense_npz[f'sequence_00/{i}'] for i in range(300)};dense_npz.close()
    for rep,fps,order in BLOCKS:
        for arm in order:
            if time.perf_counter_ns()>deadline_ns:raise TimeoutError('Deadline while scoring')
            name=pass_name(rep,fps,arm);out=ROOT/'passes'/name
            summary,rows,events=validate_pass(out);passes[name]=summary
            with np.load(out/'preds.npz',allow_pickle=False) as f:predictions={k:f[k] for k in f.files}
            data[name]=(rows,events,predictions)
            arrays,assignments=held_outputs(rows,events,predictions)
            metrics,point=score_arrays(arrays,gt,list(range(300)),ROOT/f'accuracy/aligned/{name}')
            matches[name]=point
            ages=[a['visible_age_ms'] for a in assignments if a['visible_age_ms'] is not None]
            aligned[name]=dict(metrics=metrics,visible_age_ms=distribution(ages),
                no_result_queries=sum(a['result_source_frame_id'] is None for a in assignments),
                nonempty_queries=sum(bool(len(arr)) for arr in arrays.values()),assignments=assignments)
    common={}
    for rep in [1,2]:
        for fps in [15,90]:
            names=[pass_name(rep,fps,a) for a in 'ABC']
            ids=sorted(set.intersection(*[{r['frame_id'] for r in data[n][0]} for n in names]))
            scores={}
            for name in names:
                predictions=data[name][2]
                scores[name]=score_arrays({i:predictions[f'sequence_00/{i}'] for i in ids},gt,ids,ROOT/f'accuracy/common/r{rep}_{fps}/{name[-1]}')[0]
            common[f'r{rep}_{fps}']=dict(frame_ids=ids,metrics=scores)
    diag=diagnosis(data,aligned,matches,gt,dense)
    dump(ROOT/'diagnosis.json',diag)
    for value in aligned.values():value.pop('assignments')
    # Assignments remain available as separate immutable JSON documents.
    for name,(rows,events,predictions) in data.items():
        _,assignments=held_outputs(rows,events,predictions)
        dump(ROOT/f'accuracy/aligned/{name}/assignments.json',assignments)
    gates=acceptance(passes,aligned)
    write_json(ROOT/'summary.json',dict(passes=passes,aligned=aligned,common=common,criteria=gates,
        all_criteria_passed=all(r['all_passed'] for r in gates.values()),
        scoring_wall_s=time.perf_counter()-begin,scoring_cpu_s=time.process_time()-cpu))


def system_status():
    data={}
    for key,args in [('power',['pmset','-g','batt']),('thermal',['pmset','-g','therm'])]:
        p=subprocess.run(args,capture_output=True,text=True)
        data[key]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return data


def run():
    assert not (ROOT/'EXECUTION_STARTED.json').exists(),'This experiment must not be repeated'
    assert load(ROOT/'CACHED_REGRESSION.json')['predictions_details_and_state_hashes_byte_exact']
    verify_sources();protected()
    for r in load(ROOT/'manifest.json'):assert sha(r['image'])==r['image_sha256']
    begin=time.perf_counter_ns();deadline=begin+600_000_000_000
    write_json(ROOT/'EXECUTION_STARTED.json',dict(start_ns=begin,deadline_ns=deadline,system=system_status()))
    total=0
    for block,(rep,fps,order) in enumerate(BLOCKS):
        if block:
            print('Interblock pause 10 s',flush=True);time.sleep(10)
        for arm in order:
            verify_sources();name=pass_name(rep,fps,arm);out=ROOT/'passes'/name;spec=ARMS[arm]
            cmd=[str(BASE/'.venv/bin/python'),'-m','adaptive_pipeline.run_stream',
                '--manifest',str(ROOT/'manifest.json'),'--config',str(ROOT/f'configs/{spec["config"]}.json'),
                '--policy',spec['policy'],'--processor',spec['processor'],'--fps',str(fps),
                '--reference',str(REFERENCE/'preds.npz'),'--reference-details',str(REFERENCE/'prediction_details.npz'),
                '--deadline-ns',str(deadline),'--out',str(out)]
            print('START',name,flush=True);start=time.perf_counter();status=system_status()
            remaining=(deadline-time.perf_counter_ns())/1e9
            if remaining<=0:raise TimeoutError('Global validation deadline')
            proc=subprocess.run(cmd,cwd=BASE,env=dict(os.environ,PYTHONPATH=str(BASE/'extracted')+':'+str(BASE)),
                capture_output=True,text=True,timeout=remaining)
            out.mkdir(parents=True,exist_ok=True)
            write_json(out/'SUBPROCESS.json',dict(command=cmd,returncode=proc.returncode,stdout=proc.stdout,
                stderr=proc.stderr,wall_s=time.perf_counter()-start,before=status,after=system_status()))
            print(proc.stdout,flush=True)
            if proc.returncode:raise RuntimeError('Failed worker, no retry: '+name+'\n'+proc.stderr)
            summary,_,_=validate_pass(out);total+=summary['total_requests'];assert total<=3636
            assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=250000000
            if not summary['valid']:raise RuntimeError('Invalid source pacing, no retry: '+name+' '+str(summary['source_validity']))
    write_json(ROOT/'INFERENCE_COMPLETE.json',dict(total_requests=total,warmups=36,measured=total-36,wall_s=(time.perf_counter_ns()-begin)/1e9))
    report(deadline)
    verify_sources();audit=protected()
    assert time.perf_counter_ns()<deadline
    write_json(ROOT/'COMPLETE.json',dict(actual_requests=total,elapsed_s=(time.perf_counter_ns()-begin)/1e9,
        all_criteria_passed=load(ROOT/'summary.json')['all_criteria_passed'],protected=audit,system=system_status()))
    write_json(ROOT/'artifact_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='execution.log'})
    print('COMPLETE',load(ROOT/'COMPLETE.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['regression','prepare','run'])
    a=p.parse_args()
    if a.stage=='regression':cached_regression()
    elif a.stage=='prepare':prepare()
    else:
        try:run()
        except BaseException as exc:
            if not (ROOT/'FAILED.json').exists():write_json(ROOT/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
            raise


if __name__=='__main__':main()
