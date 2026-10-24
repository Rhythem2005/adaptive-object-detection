"""Repair final JSON assembly ONLY from completed immutable integration artifacts.

No model, pipeline replay or COCO evaluation is invoked. The original partial
JSON and failure remain preserved; worker/scoring source and criteria stay frozen.
"""
import time
from pathlib import Path

import numpy as np

from analysis.latest_frame_fixed2_validation import (ROOT,BLOCKS,pass_name,validate_pass,
    acceptance,verify_sources,protected)
from analysis.argoverse_hd_phase2.common import load,dump,sha
from adaptive_pipeline.run_stream import distribution


def main():
    assert (ROOT/'FAILED.json').exists() and not (ROOT/'COMPLETE.json').exists()
    verify_sources();protected()
    partial=ROOT/'summary.json';assert partial.exists()
    partial.rename(ROOT/'summary_partial_before_report_recovery.json')
    passes={};aligned={};data={};common={}
    for rep,fps,order in BLOCKS:
        for arm in order:
            name=pass_name(rep,fps,arm);out=ROOT/'passes'/name
            summary,rows,events=validate_pass(out);passes[name]=summary;data[name]=rows
            score=ROOT/f'accuracy/aligned/{name}'
            assignments=load(score/'assignments.json');ages=[a['visible_age_ms'] for a in assignments if a['visible_age_ms'] is not None]
            with np.load(out/'preds.npz',allow_pickle=False) as pred:
                nonempty=sum(a['result_source_frame_id'] is not None and
                    len(pred['sequence_00/'+str(a['result_source_frame_id'])])>0 for a in assignments)
            aligned[name]=dict(metrics=load(score/'metrics.json'),visible_age_ms=distribution(ages),
                no_result_queries=sum(a['result_source_frame_id'] is None for a in assignments),nonempty_queries=nonempty)
    for rep in [1,2]:
        for fps in [15,90]:
            names=[pass_name(rep,fps,arm) for arm in 'ABC']
            ids=sorted(set.intersection(*[{r['frame_id'] for r in data[n]} for n in names]))
            common[f'r{rep}_{fps}']=dict(frame_ids=ids,
                metrics={n:load(ROOT/f'accuracy/common/r{rep}_{fps}/{n[-1]}/metrics.json') for n in names})
    gates=acceptance(passes,aligned)
    dump(ROOT/'summary.json',dict(passes=passes,aligned=aligned,common=common,criteria=gates,
        all_criteria_passed=all(r['all_passed'] for r in gates.values()),
        scoring_wall_s=None,scoring_cpu_s=None,
        scoring_cost_note='Scoring completed before JSON failure; its standalone timer was lost. No scoring was repeated.'))
    elapsed=(time.perf_counter_ns()-load(ROOT/'EXECUTION_STARTED.json')['start_ns'])/1e9
    assert elapsed<600
    dump(ROOT/'POSTPROCESS_RECOVERY.json',dict(reason='np.bool_ JSON encoding failure in final report only',
        original_failure_preserved=True,partial_summary_preserved=True,scoring_runs_repeated=0,
        inference_runs_repeated=0,candidate_configs_and_frozen_sources_unchanged=True,
        recovery_source_sha256=sha(Path(__file__)),elapsed_including_recovery_s=elapsed))
    dump(ROOT/'COMPLETE.json',dict(actual_requests=load(ROOT/'INFERENCE_COMPLETE.json')['total_requests'],
        elapsed_s=elapsed,all_criteria_passed=all(r['all_passed'] for r in gates.values()),protected=protected(),
        report_recovered=True,original_report_failure_retained=True))
    dump(ROOT/'artifact_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='execution.log'})
    print('Recovered without inference or scoring. Criteria:',gates)


if __name__=='__main__':main()
