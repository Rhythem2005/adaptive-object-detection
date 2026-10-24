"""Prepare and execute declared complete-sequence replay matrix. No automatic retries."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from analysis.argoverse_hd_phase2.common import load,dump,sha

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'runs/branch_b_expanded_v1'


def prepare(dataset):
    folder=ROOT/'datasets'/dataset
    frames=load(folder/'frames.json')
    source=load(BASE/('data/argoverse_hd/annotations.json' if dataset=='development' else 'data/argoverse_hd_unseen/annotations.json'))
    byid={im['id']:im for im in source['images']}
    streams=[]
    for r in frames:
        im=byid[r['image_id']]
        pts=int(re.search(r'_(\d+)\.jpg$',im['original_name']).group(1))
        streams.append(dict(r,source_id=r['video'],frame_id=r['frame_index'],source_timestamp_ns=pts,shape=[1200,1920,3]))
    for seq in sorted({r['video'] for r in streams}):
        part=[r for r in streams if r['video']==seq]
        assert [r['frame_id'] for r in part]==list(range(len(part)))
        assert all(b['source_timestamp_ns']>a['source_timestamp_ns'] for a,b in zip(part,part[1:]))
        dump(folder/(seq+'.json'),part)
    matrix=[]
    modes=['fifo','latest','merged']
    for trial in [1,2,3]:
        sequences=sorted({r['video'] for r in streams})
        if trial==2:sequences.reverse()
        for si,seq in enumerate(sequences):
            rates=[15,30,60,90] if trial%2 else [90,60,30,15]
            for ri,fps in enumerate(rates):
                shift=(trial-1+si+ri)%3
                for mode in modes[shift:]+modes[:shift]:
                    matrix.append(dict(dataset=dataset,sequence=seq,trial=trial,fps=fps,mode=mode,
                        name=f'{dataset}_{seq}_r{trial}_{fps}_{mode}',manifest=str(folder/(seq+'.json'))))
    dump(ROOT/(dataset+'_plan.json'),dict(matrix=matrix,frames=len(frames),
        policies='complete FIFO, capacity-one latest, unchanged consumed-step Fixed-2',
        repetitions=3,rates=[15,30,60,90],rss_limit_mib=4096,worker_limit_seconds=180,
        output_limit_bytes=5_000_000_000,minimum_free_bytes=12_000_000_000,
        predeclared='No tuning; no automatic retries; invalid pacing reported, not silently replaced',
        source_pacing_criteria=dict(relative_publication_rate=.03,p95_release_periods=1,max_release_periods=2),
        analysis='Per sequence and repetition paired effects; no frame-independent significance. Original engineering gates remain historical.',
        command=f'PYTHONPATH=extracted:. .venv/bin/python -m analysis.branch_b_expanded_v1.run execute --dataset {dataset}'))


def execute(dataset):
    matrix=load(ROOT/(dataset+'_plan.json'))['matrix']
    marker=ROOT/(dataset+'_EXECUTION_STARTED.json')
    assert not marker.exists(),'Explicit continuation required; do not silently repeat runs'
    dump(marker,dict(unix_time=time.time(),passes=len(matrix)))
    (ROOT/'passes').mkdir(exist_ok=True)
    sources={str(p.relative_to(BASE)):sha(p) for f in ['extracted/adaptive_pipeline','analysis/branch_b_expanded_v1'] for p in (BASE/f).rglob('*.py')}
    dump(ROOT/(dataset+'_execution_sources.json'),sources)
    outcomes=[];start=time.perf_counter()
    for i,item in enumerate(matrix):
        for path,digest in sources.items():assert sha(BASE/path)==digest,path
        if shutil.disk_usage(BASE).free<12_000_000_000:raise RuntimeError('Free storage guard')
        if time.perf_counter()-start>4*3600:raise RuntimeError('Four-hour execution guard')
        out=ROOT/'passes'/item['name']
        config=ROOT/'configs'/('practical.json' if item['mode']=='merged' else 'dense.json')
        ref=BASE/'runs/argoverse_hd_benchmark_v1/baseline' if dataset=='development' else ROOT/'offline'/dataset/'dense'
        cmd=[sys.executable,'-m','analysis.branch_b_expanded_v1.worker','--manifest',item['manifest'],
            '--config',str(config),'--mode',item['mode'],'--fps',str(item['fps']),'--out',str(out),
            '--reference',str(ref/'preds.npz'),'--reference-details',str(ref/'prediction_details.npz')]
        before=time.perf_counter()
        with (ROOT/'passes'/(item['name']+'.log')).open('x') as log:
            proc=subprocess.Popen(cmd,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH='extracted:.',PYTHONDONTWRITEBYTECODE='1'))
            # Hard supervisor deadline is outside imports, hashing and orderly worker cleanup.
            try:rc=proc.wait(timeout=240)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                rc=-124
        if not (out/'COMPLETE.json').exists() and not (out/'INCOMPLETE.json').exists():
            out.mkdir(exist_ok=True);dump(out/'INCOMPLETE.json',dict(returncode=rc,error='Worker failed before output or supervisor timeout'))
        status='complete' if (out/'COMPLETE.json').exists() else 'incomplete'
        valid=load(out/'COMPLETE.json').get('valid',True) if status=='complete' else False
        record=dict(item,status=status,pacing_valid=valid,returncode=rc,wall_s=time.perf_counter()-before)
        dump(out/'SUPERVISOR.json',record);outcomes.append(record)
        print(f'{i+1}/{len(matrix)} {item["name"]}: {status}, pacing_valid={valid}, {record["wall_s"]:.1f}s',flush=True)
        if (i+1)%3==0:time.sleep(3)
    dump(ROOT/(dataset+'_EXECUTION_COMPLETE.json'),dict(outcomes=outcomes,wall_s=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','execute']);p.add_argument('--dataset',default='development')
    a=p.parse_args();globals()[a.stage](a.dataset)
