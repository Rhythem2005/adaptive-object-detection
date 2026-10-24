"""Corrected pre-execution gate cohort; all original gate criteria unchanged."""
from collections import Counter, defaultdict
import gzip
import json
import numpy as np
from analysis.argoverse_hd_phase2.common import archive, dump, load, sha
from .methods import corroborated
from .run import BASE, ROOT, BENCH, verify, rates

def evidence_gate():
    assert (ROOT/'CONTROLS_COMPLETE.json').exists()
    frames=load(ROOT/'frameplan.json');pred=archive(BENCH/'baseline/preds.npz')
    directory=ROOT/'arms/R_grid3'
    matches=archive(BASE/'runs/argoverse_hd_coco_phase1_v1/baseline/official_matches100.npz')
    offsets=np.r_[0,np.cumsum(matches['n_pred'])]
    info={r['key']:(i,r) for i,r in enumerate(frames)}
    counts=defaultdict(Counter);byseq=defaultdict(lambda:defaultdict(Counter));byclass=defaultdict(lambda:defaultdict(Counter))
    last_video=None;birth_prev=np.zeros((0,6));raw_prev=np.zeros((0,6));last_frame=None
    rows=[]
    with gzip.open(directory/'events.jsonl.gz','rt') as stream:
        for line in stream:
            event=json.loads(line);i,f=info[event['key']]
            if f['video']!=last_video:
                birth_prev=np.zeros((0,6));raw_prev=np.zeros((0,6));last_frame=None;last_video=f['video']
            if event['decision']!='FULL':continue
            raw=pred[f['key']]
            if last_frame is None or f['frame_index']-last_frame>2:
                birth_prev=np.zeros((0,6));raw_prev=np.zeros((0,6))
            weak_rows=np.flatnonzero((raw[:,4]>=.1)&(raw[:,4]<.25))
            used={int(r) for e in event['events'] if e['type']=='confirm' for r in e['source_rows']}
            birth_rows=np.array([r for r in weak_rows if int(r) not in used],dtype=int)
            birth=raw[birth_rows]
            sources=[[int(r)] for r in birth_rows]
            output_rows=weak_rows
            for group,values,source,previous in [('birth',birth,sources,birth_prev),
                ('output',raw[output_rows],[[int(k)] for k in output_rows],raw_prev)]:
                mask=corroborated(values,previous,.3)
                for j,indices in enumerate(source):
                    if len(indices)!=1:label='ambiguous'
                    else:
                        k=offsets[i]+indices[0]
                        label='capped' if not matches['considered'][k] else 'ignored' if matches['ignored'][k,0] else 'TP' if matches['tp'][k,0] else 'FP'
                    for cell in ['all']+(['retained'] if mask[j] else ['rejected']):
                        counts[group+'_'+cell][label]+=1
                        byseq[f['video']][group+'_'+cell][label]+=1
                        byclass[str(int(values[j,5]))][group+'_'+cell][label]+=1
                    rows.append({'group':group,'key':f['key'],'source_rows':indices,'retained':bool(mask[j]),'label':label,'class':int(values[j,5]),'initial_FULL':last_frame is None})
            birth_prev=birth;raw_prev=raw[raw[:,4]>=.1].copy();last_frame=f['frame_index']
    result={'counts':{k:rates(v) for k,v in counts.items()},'per_sequence':{k:{g:rates(c) for g,c in v.items()} for k,v in byseq.items()},
        'per_class':{k:{g:rates(c) for g,c in v.items()} for k,v in byclass.items()},
        'note':'Recovery-history unmatched weak-observation descriptive screen, not causal AP or unique physical objects. TP/FP from official IoU50 source matches. No candidate sees GT at runtime. Pool membership can change after intervention. Using immediate-birth events here would condition away successful weak continuations; protocol_amendment.json records this pre-execution correction.'}
    result['gates']={}
    for g in ['birth','output']:
        a=result['counts'][g+'_all'];b=result['counts'][g+'_retained']
        checks={'TP_support':a['TP']>=50,'TP_retention':b['TP']>=.5*a['TP'],
            'precision_gain':b['precision'] is not None and b['precision']>=a['precision']+.05}
        result['gates'][g]={'checks':checks,'passed':all(checks.values()),'TP_retention':b['TP']/max(a['TP'],1)}
    dump(ROOT/'weak_evidence_gate.json',result)
    with gzip.open(ROOT/'weak_evidence_rows.jsonl.gz','xt') as out:
        for row in rows:out.write(json.dumps(row)+'\n')
    print(json.dumps(result['gates']),flush=True)


if __name__=='__main__':
    verify()
    for p,h in load(ROOT/'diagnostic_source_sha256.json').items():assert sha(p)==h,p
    evidence_gate()
