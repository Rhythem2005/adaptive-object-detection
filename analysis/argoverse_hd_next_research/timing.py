"""Six interleaved cached timing passes, with exact scored-output controls."""
import csv
from contextlib import contextmanager
import time
from unittest.mock import patch

import cv2
import numpy as np
import adaptive_pipeline.pipeline as pipeline_module
from analysis.argoverse_hd_phase2.common import archive, config_from_dict, dump, load, sha
from analysis.argoverse_hd_phase2.replay import CacheDetector, Recorder
from analysis.argoverse_hd_phase3.timing import stats
from .methods import make_pipeline, state_digest


@contextmanager
def profile_signals(pipe, values):
    """Same lightweight wrappers in both arms; no extra image/pipeline calls."""
    resize=cv2.resize;convert=cv2.cvtColor;blur=cv2.GaussianBlur
    lum=pipeline_module.mean_lab_l;prepare=pipe.flow.prepare
    def timed(name, fn, *args, **kwargs):
        start=time.perf_counter_ns();result=fn(*args,**kwargs)
        values[name]+=(time.perf_counter_ns()-start)/1e6
        return result
    def resize_wrapper(src,size,*args,**kwargs):
        if size[0]==pipe.cfg.thumb_w:return timed('thumbnail_resize_ms',resize,src,size,*args,**kwargs)
        return resize(src,size,*args,**kwargs)
    def convert_wrapper(src,code,*args,**kwargs):
        if src.shape[1]==pipe.cfg.thumb_w and code==cv2.COLOR_BGR2GRAY:
            return timed('thumbnail_gray_ms',convert,src,code,*args,**kwargs)
        return convert(src,code,*args,**kwargs)
    def blur_wrapper(src,*args,**kwargs):
        if src.shape[1]==pipe.cfg.thumb_w:return timed('thumbnail_blur_ms',blur,src,*args,**kwargs)
        return blur(src,*args,**kwargs)
    def lum_wrapper(src):return timed('thumbnail_luminance_ms',lum,src)
    def flow_wrapper(frame):return timed('required_flow_prepare_ms',prepare,frame)
    pipe.flow.prepare=flow_wrapper
    with patch.object(cv2,'resize',resize_wrapper),patch.object(cv2,'cvtColor',convert_wrapper), \
         patch.object(cv2,'GaussianBlur',blur_wrapper),patch.object(pipeline_module,'mean_lab_l',lum_wrapper):
        yield
    pipe.flow.prepare=prepare


def run_timing(root):
    assert (root/'FINALISTS_COMPLETE.json').exists()
    if (root/'TIMING_COMPLETE.json').exists():raise FileExistsError('Timing complete')
    selected=load(root/'selection.json')['selected']
    specs={k:{'config':v['config']} for k,v in load(root/'plan.json')['controls'].items()}
    specs.update(load(root/'variants_executed.json'));specs.update(load(root/'finalist_specs.json'))
    names=[selected,'F_fastpath'];frames=load(root/'frameplan.json')
    dense=archive(root.parent/'argoverse_hd_benchmark_v1/baseline/preds.npz')
    destination=root/'timing';destination.mkdir(exist_ok=False)
    runs=[]
    for repetition in range(3):
        for name in names if repetition%2==0 else names[::-1]:
            spec=specs[name];cfg=config_from_dict(spec['config'])
            preds=archive(root/'arms'/name/'preds.npz');details=archive(root/'arms'/name/'prediction_details.npz')
            digests=load(root/'arms'/name/'state_hashes.json')
            cache=CacheDetector(cfg,dense,Recorder());pipe=make_pipeline(cfg,cache,spec)
            points=[0];propagate=pipe.flow.propagate
            def flow(*args):
                points[0]+=len(args[2])*cfg.flow_grid**2
                return propagate(*args)
            pipe.flow.propagate=flow
            components={k:0. for k in ['thumbnail_resize_ms','thumbnail_gray_ms','thumbnail_blur_ms',
                'thumbnail_luminance_ms','required_flow_prepare_ms']}
            rows=[];last=None;start=time.perf_counter()
            print('Timing',repetition+1,name,flush=True)
            with profile_signals(pipe,components):
                for f in frames:
                    if f['video']!=last:pipe.reset();last=f['video']
                    assert sha(f['image'])==f['image_sha256']
                    image=cv2.imread(f['image']);assert image.shape==(1200,1920,3)
                    cache.bind(f['key'])
                    for k in components:components[k]=0.
                    wall,cpu=time.perf_counter_ns(),time.process_time_ns()
                    pred,rec=pipe.step(image,f['frame_index'])
                    wall=(time.perf_counter_ns()-wall)/1e6;cpu=(time.process_time_ns()-cpu)/1e6
                    np.testing.assert_array_equal(pred,preds[f['key']])
                    np.testing.assert_array_equal(pipe.last_prediction_details,details[f['key']])
                    assert state_digest(pipe.tracker)==digests[f['key']]
                    rows.append({'key':f['key'],'decision':rec['decision'],'step_wall_ms':wall,'step_cpu_ms':cpu,
                        'signal_ms':rec['t_signal_ms'],'propagation_ms':rec['t_propagate_ms'],
                        'track_update_ms':rec['t_track_ms'],'n_tracks':rec['n_tracks'],**components})
            path=destination/f'round{repetition+1}_{name}.csv'
            with path.open('x',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            runs.append({'name':name,'round':repetition+1,'frames':len(rows),'simulated_FULL_requests':cache.calls,
                'flow_points':points[0],'wall_ms':stats([r['step_wall_ms'] for r in rows]),
                'cpu_ms':stats([r['step_cpu_ms'] for r in rows]),'components':{k:stats([r[k] for r in rows]) for k in components},
                'stage_wall_seconds_with_decode_hash_validation_IO':time.perf_counter()-start,'new_YOLO_inferences':0})
    summary={}
    for name in names:
        selected_runs=[r for r in runs if r['name']==name];allrows=[]
        for r in selected_runs:
            with (destination/f"round{r['round']}_{name}.csv").open() as stream:allrows.extend(csv.DictReader(stream))
        summary[name]={k:stats([float(r[k]) for r in allrows]) for k in ['step_wall_ms','step_cpu_ms','signal_ms',
            'propagation_ms','track_update_ms',*components]}
        summary[name]['round_wall_means_ms']=[r['wall_ms']['mean'] for r in selected_runs]
        summary[name]['round_cpu_means_ms']=[r['cpu_ms']['mean'] for r in selected_runs]
    dump(root/'timing_summary.json',summary)
    dump(root/'TIMING_COMPLETE.json',{'runs':runs,'all_predictions_details_extended_tracker_states_exact':True,
        'scope':'Cached external step including common lightweight component-timer wrappers, no observers. Excludes decode, hashing, validation, serialization and all inference. Not end-to-end performance or E.',
        'new_YOLO_inferences':0})
