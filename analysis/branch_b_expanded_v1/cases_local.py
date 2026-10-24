"""Deterministic adverse-query diagnostics from official saved matching only."""
import collections
import gzip
from analysis.argoverse_hd_phase2.common import load,dump
from analysis.branch_b_expanded_v1.continue_local import ROOT


def main():
    results=load(ROOT/'analysis/results.json');cells=collections.defaultdict(dict)
    for name,s in results['passes'].items():
        if s['valid']:cells[(s['sequence'],s['fps'],s['trial'])][s['mode']]=name
    totals=collections.defaultdict(lambda:collections.Counter());examples=[]
    def matches(name):
        folder=results['aligned'][name]['score_directory']
        with gzip.open(folder+'/per_image_matches.json.gz','rt') as f:
            import json
            return json.load(f)
    for cell,names in cells.items():
        if not {'latest','merged'}<=set(names):continue
        bn,cn=names['latest'],names['merged'];b,c=matches(bn),matches(cn)
        ba=load(ROOT/'analysis'/(bn+'_assignments.json'));ca=load(ROOT/'analysis'/(cn+'_assignments.json'))
        candidates=[];seq,fps,trial=cell
        for fid in b:
            dtp=c[fid]['tp25']-b[fid]['tp25'];dfp=c[fid]['fp25']-b[fid]['fp25'];t=totals[str(fps)]
            t['queries']+=1;t['queries_with_TP_loss']+=dtp<0;t['queries_with_TP_gain']+=dtp>0
            t['queries_with_more_FP']+=dfp>0;t['queries_with_fewer_FP']+=dfp<0
            t['TP_change']+=dtp;t['FP_change']+=dfp
            if dtp<0 or dfp>0:
                i=int(fid);candidates.append(dict(sequence=seq,rate=fps,trial=trial,query_frame_id=i,
                    B_TP=b[fid]['tp25'],C_TP=c[fid]['tp25'],B_FP=b[fid]['fp25'],C_FP=c[fid]['fp25'],
                    TP_change=dtp,FP_change=dfp,B_assignment=ba[i],C_assignment=ca[i]))
        if candidates:
            # At most one adverse example per scene/rate/trial; deterministic ranking.
            candidates.sort(key=lambda v:(v['TP_change'],-v['FP_change'],v['query_frame_id']))
            examples.append(candidates[0])
    dump(ROOT/'analysis/failure_cases.json',dict(query_event_counts=totals,
        adverse_examples=examples,selection='One worst TP-loss/FP-increase query per valid B/C cell; ties by source frame ID',
        interpretation='Official match-event discrepancies, not unique physical objects or causal attribution. Repeated queries/scenes are correlated.'))


if __name__=='__main__':main()
