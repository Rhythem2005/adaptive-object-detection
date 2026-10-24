"""User-authorized local-only continuation. Four offline methods, three stream arms.

Historical attempts are immutable. Two complete attempts are independently
audited for reuse. The interrupted attempt is restarted in this new run root.
"""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from analysis.argoverse_hd_phase2.common import load,dump,sha
from adaptive_pipeline.run_stream import distribution

BASE=Path(__file__).resolve().parents[2]
OLD=BASE/'runs/branch_b_expanded_v1'
ROOT=BASE/'runs/branch_b_expanded_local_v2'


def read_rows(out):
    with (out/'results.csv').open() as f:rows=list(csv.DictReader(f))
    ints=['frame_id','capture_ns','decode_start_ns','decode_end_ns','published_ns','selected_ns',
          'inference_start_ns','result_ready_ns','newest_published_id_at_selection','source_timestamp_ns']
    for r in rows:
        for k in ints:r[k]=int(r[k])
        for k in ['capture_to_result_ms','service_ms','queue_wait_ms']:r[k]=float(r[k])
    return rows


def verify_pass(out,manifest,reference,details):
    s=load(out/'COMPLETE.json');events=load(out/'capture_events.json');rows=read_rows(out)
    n=len(manifest)
    assert len(events)==n and [e['frame_id'] for e in events]==list(range(n))
    assert s['buffer']['captured']==n==s['buffer']['consumed']+s['buffer']['replaced']
    assert len(rows)==s['buffer']['consumed']==s['processed']
    assert s['buffer']['pending']==s['buffer']['pending_bytes']==0
    assert rows[-1]['frame_id']==n-1
    assert [r['frame_id'] for r in rows]==sorted(set(r['frame_id'] for r in rows))
    pubs=np.array([e['published_ns'] for e in events],np.int64)
    for r in rows:
        e=events[r['frame_id']];m=manifest[r['frame_id']]
        assert r['source_timestamp_ns']==m['source_timestamp_ns']==e['source_timestamp_ns']
        for k in ['capture_ns','decode_start_ns','decode_end_ns','published_ns']:assert r[k]==e[k]
        assert r['capture_ns']<=r['decode_start_ns']<=r['decode_end_ns']<=r['published_ns']<=r['selected_ns']<=r['inference_start_ns']<=r['result_ready_ns']
        assert r['capture_to_result_ms']==(r['result_ready_ns']-r['capture_ns'])/1e6
        if s['mode']!='fifo':
            j=np.searchsorted(pubs,r['selected_ns'],side='right')-1
            assert r['frame_id']==events[j]['frame_id']==r['newest_published_id_at_selection']
    if s['mode']=='fifo':
        assert [r['frame_id'] for r in rows]==list(range(n)) and s['buffer']['replaced']==0
    else:assert s['buffer']['peak_pending']==1
    assert s['capture_to_result_ms']==distribution([r['capture_to_result_ms'] for r in rows])
    full=[r for r in rows if r['decision']=='FULL']
    assert s['measured_yolo_calls']==len(full) and s['total_forwards']==len(full)+3
    for archive,ref in [('preds.npz',reference),('prediction_details.npz',details)]:
        with np.load(out/archive,allow_pickle=False) as actual,np.load(ref,allow_pickle=False) as expected:
            assert set(actual.files)=={r['key'] for r in rows}
            for r in full:
                x,y=actual[r['key']],expected[r['key']]
                assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
    return dict(source_frames=n,processed=len(rows),full_arrays_verified=len(full),
        clocks_accounting_selection_and_fidelity=True,pacing_valid=s['valid'])


def prepare():
    ROOT.mkdir(exist_ok=False)
    for rel in ['configs','manifests','passes','analysis','figures','historical_handoff']:(ROOT/rel).mkdir()
    for mode in ['dense','practical','high_ap']:shutil.copy2(OLD/f'configs/{mode}.json',ROOT/f'configs/{mode}.json')
    original=load(BASE/'runs/argoverse_hd_benchmark_v1/fixed2_093/metrics.json')['config']
    dump(ROOT/'configs/original_fixed2.json',original)
    for p in (OLD/'datasets/development').glob('*.json'):shutil.copy2(p,ROOT/'manifests'/p.name)
    for p in (BASE/'research_handoff').glob('Branch_B_*'):shutil.copy2(p,ROOT/'historical_handoff'/p.name)
    protected=load(OLD/'protected_sha256.json')
    assert all(sha(BASE/p)==h for p,h in protected.items())
    for p in OLD.rglob('*'):
        if p.is_file():protected[str(p.relative_to(BASE))]=sha(p)
    dump(ROOT/'protected_sha256.json',protected)
    frozen={str(p.relative_to(BASE)):sha(p) for p in (BASE/'extracted/adaptive_pipeline').rglob('*.py')}
    frozen[str(BASE.relative_to(BASE)/'analysis/branch_b_expanded_v1/worker.py')]=sha(BASE/'analysis/branch_b_expanded_v1/worker.py')
    for p in (ROOT/'configs').glob('*.json'):frozen[str(p.relative_to(BASE))]=sha(p)
    for p in (ROOT/'manifests').glob('*.json'):frozen[str(p.relative_to(BASE))]=sha(p)
    frozen['data/argoverse_hd/annotations.json']=sha(BASE/'data/argoverse_hd/annotations.json')
    dump(ROOT/'frozen_sha256.json',frozen)
    plan=load(OLD/'development_plan.json')
    ref=BASE/'runs/argoverse_hd_benchmark_v1/baseline'
    reused={}
    for item in plan['matrix']:
        item['manifest']=str(ROOT/'manifests'/(item['sequence']+'.json'))
        old=OLD/'passes'/item['name']
        if (old/'COMPLETE.json').exists():
            check=verify_pass(old,load(item['manifest']),ref/'preds.npz',ref/'prediction_details.npz')
            assert check['pacing_valid']
            reused[item['name']]=dict(path=str(old),verification=check)
            item['reuse']=str(old)
        else:item['reuse']=None
    assert len(reused)==2
    plan.update(scope='LOCAL FIVE SEQUENCES ONLY; four offline approaches, three streaming arms',
        offline=['dense','original_fixed2','practical','high_ap'],no_external_dataset_access=True,
        original_interrupted_attempt=str(OLD/'passes/development_sequence_00_r1_15_merged'),
        restart_authorization='User explicitly requested restart of interrupted merged run',
        reused=reused,source_frames=3416,source_presentations=122976,
        source_pacing_policy='Keep original acceptance limits; invalid timing stays visible; no automatic reruns',
        safety='4096 MiB worker guard; 180s loop/240s process; >=12GB free; <=5GB output; <=4h execution')
    dump(ROOT/'execution_plan.json',plan)
    dump(ROOT/'REUSE_VERIFICATION.json',reused)


def execute():
    assert not (ROOT/'EXECUTION_STARTED.json').exists()
    dump(ROOT/'EXECUTION_STARTED.json',dict(unix_time=time.time(),no_network=True))
    plan=load(ROOT/'execution_plan.json');frozen=load(ROOT/'frozen_sha256.json');outcomes=[]
    env=dict(os.environ,PYTHONPATH='extracted:.',PYTHONDONTWRITEBYTECODE='1')
    start=time.perf_counter()
    for i,item in enumerate(plan['matrix']):
        assert all(sha(BASE/p)==h for p,h in frozen.items()),'Frozen source/config changed'
        if shutil.disk_usage(BASE).free<12_000_000_000:raise RuntimeError('Free disk guard')
        if time.perf_counter()-start>4*3600:raise RuntimeError('Four-hour guard')
        if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>5_000_000_000:raise RuntimeError('Output guard')
        if item['reuse']:
            record=dict(item,status='complete',pacing_valid=True,path=item['reuse'],reused=True)
        else:
            out=ROOT/'passes'/item['name'];ref=BASE/'runs/argoverse_hd_benchmark_v1/baseline'
            cmd=[sys.executable,'-m','analysis.branch_b_expanded_v1.worker','--manifest',item['manifest'],
                '--config',str(ROOT/'configs'/('practical.json' if item['mode']=='merged' else 'dense.json')),
                '--mode',item['mode'],'--fps',str(item['fps']),'--out',str(out),
                '--reference',str(ref/'preds.npz'),'--reference-details',str(ref/'prediction_details.npz')]
            before=time.perf_counter()
            with (ROOT/'passes'/(item['name']+'.log')).open('x') as log:
                proc=subprocess.Popen(cmd,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:rc=proc.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    rc=-124
            complete=(out/'COMPLETE.json').exists()
            if not complete and not (out/'INCOMPLETE.json').exists():
                out.mkdir(exist_ok=True);dump(out/'INCOMPLETE.json',dict(error='Worker start failure or supervisor timeout',returncode=rc))
            record=dict(item,status='complete' if complete else 'incomplete',
                pacing_valid=load(out/'COMPLETE.json')['valid'] if complete else False,
                returncode=rc,path=str(out),reused=False,wall_s=time.perf_counter()-before)
            dump(out/'SUPERVISOR.json',record)
        outcomes.append(record)
        print(f'{i+1}/180 {item["name"]}: {record["status"]}, valid={record["pacing_valid"]}, reused={record["reused"]}',flush=True)
        if (i+1)%3==0:time.sleep(3)
    dump(ROOT/'EXECUTION_COMPLETE.json',dict(outcomes=outcomes,wall_s=time.perf_counter()-start))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','execute']);a=p.parse_args();globals()[a.stage]()
