"""Independent arithmetic, telemetry, tensor and graph-value audit. No inference."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analysis.argoverse_hd_phase2.common import load,dump,sha
from analysis.branch_b_expanded_v1.continue_local import BASE,ROOT


def require_close(x,y,label):
    np.testing.assert_allclose(float(x),float(y),atol=1e-10,rtol=1e-10,err_msg=label)


def main():
    with (ROOT/'analysis/streaming.csv').open() as f:rows=list(csv.DictReader(f))
    lookup={r['name']:r for r in rows};results=load(ROOT/'analysis/results.json')
    assert len(results['records'])==180
    assert len({(r['sequence'],r['fps'],r['trial'],r['mode']) for r in results['records']})==180
    telemetry_checked=0;forward_total=0;full_checked=0;runtimes=set()
    for r in results['records']:
        out=Path(r['path'])
        if r['status']!='complete':
            assert (out/'INCOMPLETE.json').exists();continue
        s=load(out/'COMPLETE.json');events=load(out/'capture_events.json')
        runtime=load(out/'runtime.json')
        assert runtime['threads']==[8,1,10] and runtime['device']=='cpu' and runtime['dtype']=='float32'
        assert runtime['checkpoint_sha256']=='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
        runtimes.add((runtime['python'],runtime['torch'],runtime['opencv']))
        with (out/'results.csv').open() as f:raw=list(csv.DictReader(f))
        assert len(events)==s['source_frames']
        assert [e['frame_id'] for e in events]==list(range(len(events)))
        assert s['processed']+s['buffer']['replaced']==len(events)
        assert s['buffer']['pending']==0 and s['buffer']['consumed']==len(raw)
        lateness=np.array([(e['decode_start_ns']-e['capture_ns'])/1e6 for e in events])
        pubfps=(len(events)-1)*1e9/(events[-1]['published_ns']-events[0]['published_ns'])
        flags=dict(publication_rate=abs(pubfps/s['target_fps']-1)<=.03,
            p95_release_lateness=np.percentile(lateness,95)<=1000/s['target_fps'],
            max_release_lateness=lateness.max()<=2000/s['target_fps'])
        assert flags==s['source_validity'] and all(flags.values())==s['valid']
        csvrow=lookup[r['name']]
        lat=np.array([(int(v['result_ready_ns'])-int(v['capture_ns']))/1e6 for v in raw])
        require_close(csvrow['latency_p50_ms'],np.percentile(lat,50),'p50')
        require_close(csvrow['latency_p95_ms'],np.percentile(lat,95),'p95')
        require_close(csvrow['latency_max_ms'],lat.max(),'max')
        require_close(csvrow['frame_coverage'],len(raw)/len(events),'coverage')
        require_close(csvrow['cpu_ms_source'],1000*s['loop_cpu_s']/len(events),'CPU/source')
        require_close(csvrow['active_output_fps'],sum(int(v['result_ready_ns'])<=events[-1]['capture_ns'] for v in raw)*1e9/(events[-1]['capture_ns']-events[0]['capture_ns']),'active FPS')
        full=sum(v['decision']=='FULL' for v in raw)
        assert full==int(csvrow['calls'])==s['measured_yolo_calls']
        assert full+3==s['total_forwards'];forward_total+=full+3
        seq=r['sequence'];manifest=load(ROOT/'manifests'/(seq+'.json'))
        assert len(manifest)==len(events)
        assert [e['source_timestamp_ns'] for e in events]==[m['source_timestamp_ns'] for m in manifest]
        if r['mode']=='fifo':assert [int(v['frame_id']) for v in raw]==list(range(len(events))) and s['buffer']['replaced']==0
        else:
            pubs=[e['published_ns'] for e in events]
            for v in raw:
                j=int(np.searchsorted(pubs,int(v['selected_ns']),side='right'))-1
                assert int(v['frame_id'])==events[j]['frame_id']
        # Independently check every held-output assignment using a linear clock scan.
        assignments=load(ROOT/'analysis'/(r['name']+'_assignments.json'));j=-1
        for e,a in zip(events,assignments):
            while j+1<len(raw) and int(raw[j+1]['result_ready_ns'])<=e['capture_ns']:j+=1
            expected=None if j<0 else int(raw[j]['frame_id'])
            assert a['result_source_frame_id']==expected
            if j>=0:require_close(a['visible_age_ms'],(e['capture_ns']-int(raw[j]['capture_ns']))/1e6,'query age')
        assert len(assignments)==len(events)
        telemetry_checked+=len(raw);full_checked+=full
    scores=0
    for score in list((ROOT/'analysis/scores').iterdir())+list((ROOT/'analysis/pooled_scores').iterdir()):
        if not score.is_dir():continue
        m=load(score/'metrics.json')
        with np.load(score/'official_tensors.npz') as tensors:
            p=tensors['precision']
            for name,a in [('AP',p[:,:,:,0,2]),('AP50',p[0,:,:,0,2]),('AP75',p[5,:,:,0,2]),
                ('AP_small',p[:,:,:,1,2]),('AP_medium',p[:,:,:,2,2]),('AP_large',p[:,:,:,3,2])]:
                good=a[a>-1]
                if len(good):require_close(m['COCO'][name],good.mean(),name)
                else:assert m['COCO'][name] is None
        point=m['fixed_conf25'];tp=point['counts']['TP'];fp=point['counts']['FP'];nt=point['counts']['targets']
        for name,value in [('micro_precision',tp/(tp+fp) if tp+fp else 0),('micro_recall',tp/nt if nt else 0),
                           ('micro_F1',2*tp/(tp+fp+nt) if tp+fp+nt else 0)]:
            if point[name] is not None:require_close(point[name],value,name)
        scores+=1
    figures=load(ROOT/'figures/manifest.json');plotted=0
    observed={(r['sequence'],r['trial'],int(r['rate']),r['mode']) for r in rows if r['pacing_valid']=='True'}
    expected_panel=[]
    for seq in [f'sequence_{i:02d}' for i in range(5)]:
        for trial in ['1','2','3']:
            if all((seq,trial,rate,mode) in observed for rate in [15,30,60,90] for mode in ['fifo','latest','merged']):expected_panel.append([seq,trial])
    assert figures['balanced_scene_trial_panel']==expected_panel
    for entry in figures['plotted_values']:
        if entry['kind']=='json':
            value=load(entry['file'])
            for k in entry['keys']:value=value[k]
            value=None if value is None else value*entry['scale']
        elif entry['kind']=='csv_pooled':
            with Path(entry['file']).open() as f:pp=list(csv.DictReader(f))
            wanted={tuple(c) for c in entry['cells']}
            chosen=[r for r in pp if (r['mode'],r['rate'],r['trial']) in wanted]
            value=np.mean([float(r[entry['field']])*entry['scale'] for r in chosen]) if chosen else None
        else:
            chosen=[lookup[name] for name in entry['names']]
            value=np.mean([float(r[entry['field']])*entry['scale'] for r in chosen]) if chosen else None
        if value is None:assert entry['value'] is None
        else:require_close(entry['value'],value,'plotted value')
        plotted+=1
    for path,h in figures['source_sha256'].items():assert sha(path)==h
    for rel,h in load(ROOT/'frozen_sha256.json').items():assert sha(BASE/rel)==h,rel
    for rel,h in load(ROOT/'analysis/scoring_sources.json').items():assert sha(BASE/rel)==h,rel
    protected=load(ROOT/'protected_sha256.json')
    changed=[rel for rel,h in protected.items() if not (BASE/rel).exists() or sha(BASE/rel)!=h]
    assert not changed,changed
    # Original inputs are re-hashed after execution, not merely assumed present.
    frames=load(ROOT/'manifests/frames.json')
    for r in frames:assert sha(r['image'])==r['image_sha256']
    source_gt=load(BASE/'data/argoverse_hd/annotations.json');evaluation_gt=load(ROOT/'manifests/ground_truth.json')
    original_annotations={a['id']:a for a in source_gt['annotations']}
    assert len(evaluation_gt['annotations'])==len(original_annotations)==57711
    assert len({a['id'] for a in evaluation_gt['annotations']})==57711
    for annotation in evaluation_gt['annotations']:
        original=original_annotations[annotation['source_annotation_id']]
        assert annotation['id']>0
        for field in ['image_id','bbox','category_id','area','track']:
            assert annotation[field]==original[field],field
        for field in ['iscrowd','ignore']:assert annotation.get(field,False)==original.get(field,False)
    offline=load(ROOT/'analysis/offline.json')
    assert len(runtimes)==1,'Runtime versions differ across selected workers'
    for name,v in offline.items():
        require_close(v['fps'],1000/v['wall_ms_frame'],'offline FPS')
        assert v['calls']==(3416 if name=='dense' else 1710)
        expected=load(BASE/v['source'])
        if name=='original_fixed2':expected=expected['official_COCO_max100']['AP']
        else:expected=expected['methods'][{'dense':'A','practical':'B','high_ap':'D'}[name]]['accuracy']['COCO']['AP']
        require_close(v['COCO']['AP'],expected,'offline AP')
    # Recompute paired effects without the report generator's aggregation code.
    pairs=load(ROOT/'analysis/paired_effects.json')
    for pair in pairs:
        b,c=lookup[pair['B_name']],lookup[pair['C_name']]
        assert b['pacing_valid']==c['pacing_valid']=='True'
        require_close(pair['aligned_AP_pp'],100*(float(c['aligned_AP'])-float(b['aligned_AP'])),'AP pp')
        require_close(pair['cpu_reduction_percent'],100*(1-float(c['loop_cpu_s'])/float(b['loop_cpu_s'])),'CPU %')
        require_close(pair['p95_reduction_percent'],100*(1-float(c['latency_p95_ms'])/float(b['latency_p95_ms'])),'latency %')
    out=dict(matrix_cells=180,complete=len(rows),incomplete=180-len(rows),
        pacing_valid=sum(r['pacing_valid']=='True' for r in rows),
        measured_records_independently_recomputed=telemetry_checked,
        full_prediction_fidelity_verified_by_prior_independent_pass_audit=full_checked,
        forwards_in_selected_attempts_including_reused=forward_total,
        score_tensor_metrics_verified=scores,plotted_values_verified=plotted,figures=len(figures['figures']),
        historical_files_verified=len(protected),historical_changes=changed,original_images_verified=len(frames),
        original_annotations_verified=57711,
        frozen_production_and_configs_unchanged=True,paired_effects_verified=len(pairs),
        no_statistical_significance_claim=True,no_unseen_data_claim=True)
    dump(ROOT/'FINAL_VERIFICATION.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
