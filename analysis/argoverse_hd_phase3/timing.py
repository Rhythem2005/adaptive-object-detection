"""Repeated cache-only processing costs, never real-inference Experiment E."""
import csv
from pathlib import Path
import time
from types import MethodType

import cv2
import numpy as np
from adaptive_pipeline.pipeline import AdaptivePipeline
from analysis.argoverse_hd_phase2.common import archive, config_from_dict, dump, load, sha
from analysis.argoverse_hd_phase2.replay import CacheDetector, Recorder, state_digest


def force_fixed(pipe, schedule, current):
    def early(self,lum):
        f=current[0]
        return ['forced_schedule'] if f['frame_index'] in schedule[f['video']] else []
    original=pipe.controller.decide
    def decide(*args,**kwargs):
        natural=original(*args,**kwargs);f=current[0]
        action='FULL' if f['frame_index'] in schedule[f['video']] else 'SKIP'
        return action,['forced_schedule'] if action=='FULL' else [],[],natural[3]
    pipe.controller.early_full=MethodType(early,pipe.controller)
    pipe.controller.decide=decide


def stats(values):
    a=np.asarray(values,float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99))}


def run_timing(root):
    from .run import conservative_gate, result
    if not (root/'EFFICIENCY_COMPLETE.json').exists():raise RuntimeError('Efficiency scoring must finish first')
    if (root/'TIMING_COMPLETE.json').exists():raise FileExistsError('Timing already complete')
    plan=load(root/'execution_plan.json');specs=dict(plan['controls'],**plan['variants'],**load(root/'efficiency_plan.json'))
    baseline='R_T3_recovery';reference=result(baseline)
    gates={n:conservative_gate(result(n),reference) for n in specs if n!=baseline}
    eligible=[n for n,g in gates.items() if all(g.values()) and
              result(n)['cost']['simulated_FULL_requests']<=reference['cost']['simulated_FULL_requests']]
    eligible.sort(key=lambda n:(result(n)['cost']['simulated_FULL_requests'],
        result(n)['cost']['total_flow_point_evaluations_before_FB'], -result(n)['metrics']['COCO']['AP'],n))
    challenger=eligible[0] if eligible else baseline+'__grid3'
    dump(root/'timing_selection.json',{'baseline':baseline,'challenger':challenger,
        'conservative_gates':gates,'eligible':eligible,'challenger_passed_accuracy_precision_screen':bool(eligible),
        'rule':'Among no-degradation candidates: fewer requests, then fewer flow points, then AP. If none, measure pre-specified T3 grid3 only as a cost challenger, not as an accepted improvement.'})
    frames=load(root/'frameplan.json');dense=archive(root.parent/'argoverse_hd_benchmark_v1/baseline/preds.npz')
    destination=root/'timing';destination.mkdir(exist_ok=False)
    summaries=[]
    for round_index in range(3):
        order=[baseline,challenger] if round_index%2==0 else [challenger,baseline]
        for position,name in enumerate(order):
            spec=specs[name];cfg=config_from_dict(spec['config'])
            reference_preds=archive(root/'arms'/name/'preds.npz')
            reference_details=archive(root/'arms'/name/'prediction_details.npz')
            digests=load(root/'arms'/name/'state_hashes.json')
            cache=CacheDetector(cfg,dense,Recorder());pipe=AdaptivePipeline(cfg,cache)
            current=[None];schedule={seq:set(v) for seq,v in spec.get('schedule',{}).items()}
            if schedule:force_fixed(pipe,schedule,current)
            original_flow=pipe.flow.propagate;points=[0]
            def flow(prev,gray,boxes,scale):
                points[0]+=len(boxes)*cfg.flow_grid**2
                return original_flow(prev,gray,boxes,scale)
            pipe.flow.propagate=flow
            rows=[];last=None;stage_start=time.perf_counter()
            print(f'Timing round{round_index+1}, position{position+1}: {name}',flush=True)
            for info in frames:
                if info['video']!=last:pipe.reset();last=info['video']
                assert sha(info['image'])==info['image_sha256']
                image=cv2.imread(info['image']);assert image.shape==(1200,1920,3)
                cache.bind(info['key']);current[0]=info
                wall,cpu=time.perf_counter_ns(),time.process_time_ns()
                pred,row=pipe.step(image,info['frame_index'])
                wall_ms=(time.perf_counter_ns()-wall)/1e6;cpu_ms=(time.process_time_ns()-cpu)/1e6
                # Validation/hashing/serialization is outside timed step.
                np.testing.assert_array_equal(pred,reference_preds[info['key']])
                np.testing.assert_array_equal(pipe.last_prediction_details,reference_details[info['key']])
                assert state_digest(pipe.tracker)==digests[info['key']]
                rows.append({'key':info['key'],'decision':row['decision'],'step_wall_ms':wall_ms,
                    'step_cpu_ms':cpu_ms,'track_update_ms':row['t_track_ms'],
                    'propagation_ms':row['t_propagate_ms'],'signal_ms':row['t_signal_ms'],
                    'n_tracks':len(pipe.tracker),'since_full':row['since_full']})
            file=destination/f'round{round_index+1}_{name}.csv'
            with file.open('x',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            summaries.append({'round':round_index+1,'position':position+1,'arm':name,
                'frames':len(rows),'simulated_FULL_requests':cache.calls,'new_YOLO_inferences':0,
                'initial_flow_points':points[0],'stage_wall_seconds_including_validation_decode_IO':time.perf_counter()-stage_start,
                'step_wall_ms':stats([r['step_wall_ms'] for r in rows]),
                'step_cpu_ms':stats([r['step_cpu_ms'] for r in rows]),
                'by_decision':{d:{'wall_ms':stats([r['step_wall_ms'] for r in rows if r['decision']==d]),
                                  'cpu_ms':stats([r['step_cpu_ms'] for r in rows if r['decision']==d])} for d in ['FULL','SKIP']}})
    dump(root/'TIMING_COMPLETE.json',{'runs':summaries,'bitwise_prediction_detail_and_state_parity':True,
        'scope':'CPU cache-only step including minimal wrapper and prediction-detail construction; excludes decode/hash/validation/IO/model. No deployed FPS, GPU or energy claim. All frames including init retained.',
        'order':'AB,BA,AB; no real-model warmup','new_YOLO_inferences':0})
    print('Timing COMPLETE:6 cache-only replays, all match scored arms',flush=True)
