"""Write the Phase 3 measured report from completed scored/timed artifacts."""
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/adaptive_phase3_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.argoverse_hd_phase2.common import dump, load
from .run import BASE, ROOT, P1, result, verify_sources


LABELS={
    'R_T1_anchor':'Anchor control', 'R_T3_recovery':'Recovery control',
    'R_T4_shared10':'Shared threshold 0.10', 'V1_separate_birth10':'V1 separate births 0.10',
    'V2_active_recovery':'V2 active recovery', 'V3_low_iou50':'V3 low-stage IoU 0.50',
    'V4_active_low_iou50':'V4 active + IoU 0.50', 'V5_threshold_assignment':'V5 gated assignment',
    'R_T3_recovery__grid3':'Recovery + 3x3 flow',
    'R_T3_recovery__fixed2_phase1':'Fixed-2 phase U1',
    'R_T3_recovery__fixed3_phase0':'Fixed-3 phase U0',
    'R_T3_recovery__fixed3_phase1':'Fixed-3 phase U1',
}


def table(header,rows):
    return '| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)


def pct(v):return f'{100*v:.4f}'


def main():
    verify_sources()
    summary=load(ROOT/'diagnostics/summary.json');rows={r['arm']:r for r in summary['arms']}
    details=summary['details'];timing=load(ROOT/'TIMING_COMPLETE.json')
    chosen=load(ROOT/'timing_selection.json');base=rows['R_T3_recovery'];grid=rows['R_T3_recovery__grid3']
    dense=load(P1/'baseline/metrics.json')['official_COCO_max100']
    initial=['R_T1_anchor','R_T3_recovery','R_T4_shared10']+[n for n in LABELS if n.startswith('V')]
    efficiency=[n for n in rows if '__' in n]
    def link(label,relative,line=None):return f'[{label}]({BASE/relative}'+(f':{line}' if line else '')+')'
    step={}
    for name in [chosen['baseline'],chosen['challenger']]:
        vals=[]
        for file in sorted((ROOT/'timing').glob('round*_'+name+'.csv')):
            with file.open() as f:vals.extend(list(csv.DictReader(f)))
        wall=np.array([float(r['step_wall_ms']) for r in vals]);cpu=np.array([float(r['step_cpu_ms']) for r in vals])
        rounds=[r for r in timing['runs'] if r['arm']==name]
        step[name]={'wall_mean_ms':float(wall.mean()),'wall_median_ms':float(np.median(wall)),
            'wall_p95_ms':float(np.percentile(wall,95)),'wall_p99_ms':float(np.percentile(wall,99)),
            'cpu_mean_ms':float(cpu.mean()),'round_wall_means_ms':[r['step_wall_ms']['mean'] for r in rounds],
            'round_cpu_means_ms':[r['step_cpu_ms']['mean'] for r in rounds],'samples':len(wall)}
    dump(ROOT/'timing_summary.json',step)
    btime=step[chosen['baseline']];ctime=step[chosen['challenger']]
    cost_totals={'scored_replay_wall_seconds_including_decode_hash_diagnostics_IO':sum(result(n)['cost']['stage_wall_seconds_including_decode_hash_diagnostics_IO'] for n in rows),
        'official_scoring_wall_seconds':sum(result(n)['metrics']['scoring_wall_seconds'] for n in rows),
        'timing_stage_wall_seconds_including_decode_validation_IO':sum(r['stage_wall_seconds_including_validation_decode_IO'] for r in timing['runs']),
        'scope':'Sums of recorded stage times only; excludes setup, analysis, reporting, audit and interruption. Not elapsed task time or deployed processing throughput.'}
    dump(ROOT/'cost_summary.json',cost_totals)
    wall_change=100*(ctime['wall_mean_ms']/btime['wall_mean_ms']-1)
    cpu_change=100*(ctime['cpu_mean_ms']/btime['cpu_mean_ms']-1)
    pairs=[]
    for round_index in [1,2,3]:
        rr={r['arm']:r for r in timing['runs'] if r['round']==round_index}
        pairs.append(100*(rr[chosen['challenger']]['step_wall_ms']['mean']/rr[chosen['baseline']]['step_wall_ms']['mean']-1))
    text='''# Phase 3: accuracy, precision and computation trade-offs

Completed20 October2026. All approved initial experiments and the applicable conditional checks are complete: **12 scored full-sequence replays and6 cache-only timing replays**, or61,488 frame steps. Zero new YOLO inference, training, D/E experiments or Phase4 work occurred. No algorithm defaults were changed and no commit/push was performed by this agent.

The best practical candidate from this batch is **the existing anchored Fixed-2 recovery configuration with a3x3 flow grid**, subject to the limitations below. Its gains are small and descriptive, not a generalization or statistical-significance claim. No tested configuration improves every possible metric: low-score FP count, precision ratios, AP and total tracking work must be reported together.

## Validity and controls

Before new variants, the anchoring, recovery and shared-threshold controls reproduced Phase2 predictions, prediction details, per-frame tracker-state hashes and complete official COCO precision/recall/score tensors bitwise. Aggregate, per-class, per-sequence and exact-confidence metrics agree exactly. Production source changes are confined to opt-in tracker settings and association/birth handling; detector weights, inference settings, flow estimator, scheduling algorithms and evaluator are unchanged.

All five sequences and3,416 frames are retained, including all provided classes/crowds. The same1920x1200 images are hashed before each replay; each sequence resets state. FULL frames return only the original, image-bound Dense cache; ROI/warmup/model calls are prohibited. Stock YOLOv8n's historical settings remain CPU FP32, imgsz640, confidence0.001, NMS0.7 and global max_det300. These settings describe the cache source; no YOLO device executed in this phase. Replay uses NumPy/SciPy/OpenCV CPU; scoring uses pinned pycocotools2.0.11, normalized positive annotation IDs, original pixel coordinates, all eight categories and official maxDets=[1,10,100].

COCO AP means AP50–95. AP/AR tables are percentages; differences in AP are percentage points. Precision/recall/F1 at an exact confidence and IoU0.50 are supplementary operating-point statistics, not COCO summary slots. AP is not averaged across sequences. All results use the same official matching/crowd/cap semantics. No score was optimized for best F1, and no outputs were promoted to higher confidence.

## Initial eight controlled replays

All eight replay arms use identical1,710 FULL requests and1,706 SKIPs:49.94% fewer requests than Dense. The previously scored Dense reference uses all3,416 detector frames. Birth/association variants change tracking only; FULL outputs are identical to Dense on each selected image.

'''
    text+=table(['Method','AP50','AP50–95','AP75','AR100'],[['Dense existing reference',*[pct(dense[k]) for k in ['AP50','AP','AP75','AR100']]]]+[[LABELS[n],*[pct(rows[n][k]) for k in ['AP50','AP','AP75','AR100']]] for n in initial])
    text+='\nAll-frame exact-confidence0.25, class-macro operating points:\n\n'
    text+=table(['Method','Precision %','Recall %','F1 %'],[[LABELS[n],*[pct(rows[n][k]) for k in ['macro_P25','macro_R25','macro_F1_25']]] for n in initial])
    text+='\nOn the same1,706 SKIP frames, official IoU0.50 matches show the low-score trade-offs:\n\n'
    text+=table(['Method','TP≥0.10','FP≥0.10','Precision≥0.10 %','TP≥0.25','FP≥0.25','Mean tracks'],[
        [LABELS[n],rows[n]['SKIP_TP_conf0.1'],rows[n]['SKIP_FP_conf0.1'],pct(rows[n]['SKIP_TP_conf0.1']/(rows[n]['SKIP_TP_conf0.1']+rows[n]['SKIP_FP_conf0.1'])),
         rows[n]['SKIP_TP_conf0.25'],rows[n]['SKIP_FP_conf0.25'],f"{rows[n]['mean_tracks']:.4f}"] for n in initial])
    v3=rows['V3_low_iou50'];v1=rows['V1_separate_birth10']
    text+=f'''
**V1 resolves the Phase2 confound on this subset.** It keeps high-score association at0.25, performs ordinary existing-track low-score recovery, then independently admits unmatched births down to0.10. Relative to recovery-only it adds {100*(v1['AP']-base['AP']):.6f} AP points, but adds {v1['SKIP_FP_conf0.1']-base['SKIP_FP_conf0.1']:,} SKIP FPs for {v1['SKIP_TP_conf0.1']-base['SKIP_TP_conf0.1']:,} extra TPs at0.10. It reproduces the shared-threshold control's aggregate COCO/operating metrics exactly, although4 frame prediction arrays differ. This supports low-score births as the conditional source of additional AP here; it does not establish universal equivalence of one/two-stage association or make births reliable.

**V2–V4 reduce low-score FP, but also remove true detections.** V3 is the most conservative precision-oriented option: AP changes by {100*(v3['AP']-base['AP']):+.6f} points; SKIP FPs at0.10 fall {100*(1-v3['SKIP_FP_conf0.1']/base['SKIP_FP_conf0.1']):.2f}% and TPs fall by {base['SKIP_TP_conf0.1']-v3['SKIP_TP_conf0.1']}. Mean tracks fall {100*(1-v3['mean_tracks']/base['mean_tracks']):.2f}%. Its confidence0.25 outputs have identical TP/FP/macro P/R/F1 counts to recovery-only. Therefore it is a precision/low-score-recall trade-off, not an established accuracy-preserving improvement. No noninferiority margin or deployment miss/false-alarm cost was supplied. V3 did not receive repeated timing or a grid-combination experiment.

The pre-specified conservative screen required no degradation in AP, AR100, macro P25 or TP25, and no increase in FP25, FP10 or mean tracks. **None of the five new variants passed.** The screen was not relaxed after seeing V3. Thus the optional second confidence engine was not promoted: only four efficiency controls on the existing recovery baseline were run. All V2–V5 results at0.25 remain identical to recovery-only; their effects are primarily in lower-score retained evidence and subsequent history. V3 reduces low-score confirmations from3,647 to2,967: source TP50 confirmation events fall1,232→1,088 and source FP50 events fall2,258→1,744. This supports removal of both valid and invalid continuation evidence, not a new detection capability or a flow improvement.

### What did threshold-aware assignment actually change?

The old implementation maximizes summed IoU before rejecting invalid pairs. A synthetic matrix [[0.30,0.29],[0.29,0]] at threshold0.30 selects two invalid pairs even though one valid match exists. V5 prioritizes valid-match cardinality, then summed IoU, within the same class and confidence stages. This alternative can also replace one very strong pair with two weaker valid pairs. It is an explicit objective change, not a universal correctness guarantee.

'''
    text+=table(['History','Matrices','Different pair sets','Matrices with more valid pairs','Extra valid pairs'],[[LABELS[n],
        sum(d['matrices'] for d in details[n]['event_summary']['same_input_assignment_comparisons'].values()),
        sum(d['different_pairs'] for d in details[n]['event_summary']['same_input_assignment_comparisons'].values()),
        sum(d['gated_more_valid_pairs_matrices'] for d in details[n]['event_summary']['same_input_assignment_comparisons'].values()),
        sum(d['gated_extra_valid_pairs_total'] for d in details[n]['event_summary']['same_input_assignment_comparisons'].values())] for n in ['R_T3_recovery','V5_threshold_assignment']])
    text+=f"\nV5's AP changes by {100*(rows['V5_threshold_assignment']['AP']-base['AP']):+.6f} points, with two more all-frame TPs and three more FPs at0.10. The theoretical issue is real, but it is not a material bottleneck demonstrated by this benchmark. Do not adopt a more complicated association objective on this evidence. Same-matrix alternatives are diagnostic counterfactuals, not whole-run causal AP decompositions.\n\n### Where the extra low-score seeds come from\n\n"
    text+=table(['Method','Low birth TP50 events','Low birth FP50 events','Low-confirm TP50 events','Low-confirm FP50 events'],[[LABELS[n]]+[
        details[n]['event_summary']['source_observation_events'].get(k,0) for k in ['birth_low__official_TP50','birth_low__official_FP50','confirm_low__official_TP50','confirm_low__official_FP50']] for n in initial])
    text+='\nThese are observation/update events, not unique physical objects. Crowd-ignored, capped and ambiguous duplicated source rows are retained separately in diagnostic JSON. Source TP50 establishes the current detector box/class match; it does not prove physical identity continuity. The paired target and linked-geometry diagnostics retain gained/lost GT identities, hidden states and all denominator definitions. Provided track IDs can fragment.\n\n## Approved efficiency and phase controls\n\n'
    text+=table(['Method','AP50–95','AP50','AR100','FULL requests','Call savings %','Flow points'],[[LABELS[n],pct(rows[n]['AP']),pct(rows[n]['AP50']),pct(rows[n]['AR100']),rows[n]['simulated_FULL_requests'],f"{100*rows[n]['call_savings']:.2f}",rows[n]['flow_points']] for n in ['R_T3_recovery']+efficiency])
    text+=f'''
The3x3 grid changes only existing flow_grid from4 to3; flow width640, minimum4 valid points, LK window/pyramid/FB settings, anchoring, association and score decay remain fixed. It uses {100*(1-grid['flow_points']/base['flow_points']):.2f}% fewer initial flow points ({base['flow_points']:,}→{grid['flow_points']:,}). This changes spatial sampling and the valid-point fraction required by the fixed four-point rule; it is not a numerical-equivalence optimization. Both forward and backward flow still run. Pyramid preparation and other fixed work remain, so point count is not a latency or energy estimate.

Grid3 AP changes by {100*(grid['AP']-base['AP']):+.6f} points, AP75 by {100*(grid['AP75']-base['AP75']):+.6f}, and AR100 by {100*(grid['AR100']-base['AR100']):+.6f}. The accuracy changes are tiny. It does not solve the stock detector's small-object weakness.

'''
    text+='All-frame IoU0.50 counts at the same FULL/SKIP frames:\n\n'
    text+=table(['Confidence','Recovery TP','Grid3 TP','Recovery FP','Grid3 FP','Recovery micro P %','Grid3 micro P %'],[
        [conf,base['TP_conf'+conf],grid['TP_conf'+conf],base['FP_conf'+conf],grid['FP_conf'+conf],pct(base['micro_precision_conf'+conf]),pct(grid['micro_precision_conf'+conf])] for conf in ['0.001','0.1','0.25']])
    text+=f'''
At0.25, grid3 adds9 TPs and removes9 FPs. At0.10, it adds12 TPs and3 FPs. It slightly improves micro precision at all three reported thresholds, but **does not reduce the absolute FP count at every threshold**. Mean tracks increase by {grid['mean_tracks']-base['mean_tracks']:.6f}. Consequently it also fails the original strict all-criteria screen (FP10/track-count proxies); it was timed as the pre-specified cost challenger, not secretly passed by relaxing that screen. The practical recommendation below weighs its measured point/time reduction and explicit precision ratios rather than calling it a strict all-metric Pareto win.

The second Fixed-2 phase scores below the original recovery phase. Fixed-3 uses1,142 requests (66.57% savings) but loses accuracy under both phases. All count-matched phases share the anchored fixed early-FULL path; propagation is skipped before scheduled FULL. U0 is every k-th frame from0. U1 retains frame0 and shifts the remaining budget to1+j*k. Budgets are equal per sequence, and end-of-sequence withholding can lengthen the final gap. **U1 is an offline phase/budget control, not a newly validated online scheduler.** The engine's large native fixed_k disables extra implicit refreshes; the complete prescribed schedule is the source of truth. No GT/future detection content enters decisions. U0 corresponds to ordinary periodic scheduling. These tests do not rehabilitate the original adaptive scheduler or establish a better closed-loop policy.

![Phase3 measured trade-offs]({ROOT/'figures/tradeoffs.png'})

## Repeated cache-only processing costs

Six runs used AB,BA,AB order, three per arm, with all3,416 frames including initialization. Every output, detail array and tracker-state hash matches its corresponding scored replay bitwise. No deep observation/event wrapper executes in the timed stage; only a common lightweight flow-point counter and bound cache are present. Timing excludes image decoding, input hashing, output validation, serialization and model execution. It includes Python orchestration and construction of prediction details. CPU time includes library worker threads and can exceed wall time.

'''
    text+=table(['Arm','Samples','Mean wall ms','Median wall ms','P95 wall ms','P99 wall ms','Mean CPU ms'],[[LABELS[n],d['samples']]+[f"{d[k]:.4f}" for k in ['wall_mean_ms','wall_median_ms','wall_p95_ms','wall_p99_ms','cpu_mean_ms']] for n,d in step.items()])
    text+='\n'+table(['Round','Recovery mean wall ms','Grid3 mean wall ms','Grid3 change %'],[[i+1,f"{btime['round_wall_means_ms'][i]:.4f}",f"{ctime['round_wall_means_ms'][i]:.4f}",f'{pairs[i]:+.2f}'] for i in range(3)])
    text+=f'''
Pooled mean cache-step wall time changes {wall_change:+.2f}%; CPU time changes {cpu_change:+.2f}%. These are measured cache-processing differences on this machine/runtime, not deployed YOLO+tracking speedups, real-time FPS or energy savings. Only three interleaved repetitions were run; system/thermal noise and order effects remain. No GPU/MPS speed claim is made. Original historical CPU-inference FPS is not combined with these timings to manufacture a new end-to-end number. D/E remain deferred.

Recorded stage wall-time sums were {cost_totals['scored_replay_wall_seconds_including_decode_hash_diagnostics_IO']:.1f}s for12 scored replays (including decode/hash/diagnostics/I/O), {cost_totals['official_scoring_wall_seconds']:.1f}s for official scoring, and {cost_totals['timing_stage_wall_seconds_including_decode_validation_IO']:.1f}s for six timing stages including decode/validation/I/O. These are execution-accounting sums, not elapsed task time or deployable throughput; setup, analysis, reporting, audit and the interruption are excluded.

## Per-sequence and class/size accuracy

All sequence cells below are independently evaluated COCO AP50–95 (%). Categories without regular GT in a sequence are excluded by COCO, so sequence class support differs.

'''
    sequences=list(result('R_T3_recovery')['metrics']['per_sequence'])
    text+=table(['Method']+sequences,[[LABELS[n]]+[pct(details[n]['per_sequence_AP'][seq]) for seq in sequences] for n in initial+efficiency])
    text+='\nGrid3 improves AP in four sequences and decreases it slightly in sequence_04. The small pooled gain is not uniform across every class/size/sequence. Selected per-class AP50–95 (%); complete class metrics for every arm remain in scoring/metrics.json:\n\n'
    cls_names=['R_T3_recovery','R_T3_recovery__grid3','V3_low_iou50']
    text+=table(['Class','Dense','Recovery','Grid3','V3'],[[cls,pct(dense['per_class'][cls]['AP'])]+[pct(result(n)['metrics']['COCO']['per_class'][cls]['AP']) for n in cls_names] for cls in dense['per_class']])
    text+='\n'
    text+=table(['Method','AP small','AP medium','AP large'],[['Dense',*[pct(dense[k]) for k in ['AP_small','AP_medium','AP_large']]]]+[[LABELS[n],*[pct(rows[n][k]) for k in ['AP_small','AP_medium','AP_large']]] for n in cls_names])
    text+='\nFor the common strong-seed cohort, previous FULL and current Dense must both match the provided GT identity at TP75 and confidence≥0.25. This cohort is identical across the original Fixed-2-frame arms; it does not represent all objects.\n\n'
    text+=table(['Arm','Eligible GT instances','TP50','TP75'],[[LABELS[n]]+[details[n]['strong_seed_cohort'][k] for k in ['GT_instances','adaptive_TP_IoU50','adaptive_TP_IoU75']] for n in ['R_T3_recovery','R_T3_recovery__grid3','V3_low_iou50']])
    text+='''
Interpretation: confidence/birth changes add or remove low-score evidence; source TP/FP event counts expose their cost. Stricter recovery also loses valid low-score confirmations, not just background. The sparse-grid change modifies box sampling; paired identities and the strong-seed cohort show whether localisation/retention changes rather than a score-only effect. Flow rejection and centre/width/height/IoU summaries are retained for every arm in diagnostics. These conditional geometry statistics are not an unbiased dataset-wide error decomposition, and AP contributions are not additive.

Grid3 rejects286 of15,095 flow track updates versus212 of15,082 with4x4. It also changes later association/retention (3,665 versus3,647 low-score confirmations). Thus this is not a pure reduction in arithmetic with unchanged decisions: spatial sampling, the relative valid-point requirement and downstream history all change. The common strong-seed TP75 gain supports a small geometry/retention benefit, but does not isolate which sampling/acceptance mechanism caused it. No extra minimum-point sweep was run to force an explanation.

## Recommended final candidate and limits

**Practical development candidate:** anchored Fixed-2 + existing-track recovery, flow_grid=3, flow_min_points=4. Keep track_min_conf=0.25, association_low_conf=0.10, low_score_association=True, assoc_iou=0.30 and conf_decay=0.93. Keep stock YOLOv8n and all historical inference settings. New track_birth_conf remains None; low_association_visible_only=False, low_association_iou=None and threshold_aware_assignment=False. Quality-aware flow, score boosting, camera/appearance additions, ROI inference and new scheduling are not enabled. The complete candidate_config.json is saved alongside this report; no existing preset/default is silently switched.

This candidate has the best measured practical combination of approximately retained/slightly improved accuracy, better confidence0.25 TP/FP, unchanged detector requests and much less flow-point work. Repeated cache timings above price the observed processing benefit and its variability. It is **not** a strict all-metric winner because of the small low-threshold FP-count increase, nor a proven end-to-end acceleration. The absolute detector accuracy remains weak. Do not deploy or claim generalization based only on these five already-analysed sequences.

**Precision-oriented alternative:** V3 low-stage IoU0.50 reduces low-score FPs and internal tracking work, with a measured AP/AR and low-score-recall loss. If the application explicitly accepts that trade-off it deserves consideration. Its actual repeated runtime and its interaction with grid3 remain unverified. A V3+grid3 combination would be a new conditional experiment requiring approval; it was not run and no additive gain is predicted.

Do not adopt unrestricted low-score births, active-only recovery, or cardinality-first assignment merely because they are plausible research transfers. Do not introduce a learned scheduler, detector replacement, retraining, score promotion or extra geometry machinery now. No such complexity earned a measured unqualified benefit here.

Research context: [ByteTrack's paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) and [reference implementation](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py) motivate separating high-score births from lower-score continuation and restricting its second-stage association to active tracks. Our negative active-only result shows that this restriction does not automatically transfer from MOT to frame-aligned YOLOv8n detection AP. [OC-SORT](https://arxiv.org/abs/2203.14360) supports preserving detector observations; the existing simpler anchoring was retained. [Detect or Track](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/02/AAAI-LuoH.pdf) motivates judging refresh value, but does not justify a trained scheduler here. [Towards Streaming Perception](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470460.pdf) explicitly includes output latency; our COCO AP/cache costs are not streaming AP. These papers motivate mechanisms, not expected numerical gains in this experiment.

Remaining uncertainty includes statistical stability, new-sequence generalization, annotation identity fragmentation, original detector truncation/confidence/class errors, input-resolution causality, actual model+tracking contention, deployed latency/throughput and energy. D/E and Phase4 remain unexecuted. No new data was downloaded.

## Implementation, provenance and reproduction

'''
    text+=f'''Only {link('PipelineConfig opt-ins','extracted/adaptive_pipeline/shared/config.py')} and {link('Tracker association/birth logic','extracted/adaptive_pipeline/tracking/tracker.py')} changed relative to the pre-Phase3 production snapshot. Historical defaults and controls reproduce. New {link('synthetic tests','tests/test_phase3_tracking.py')} cover independent admission, no duplicate births, hidden-track miss bookkeeping, low/high-stage gating, class/empty/rectangular assignment, objective counterexamples and native/forced fixed fast-path parity. All84 tests pass. Experimental opt-ins remain disabled by default; the recommended configuration uses the original association algorithm.

The {link('execution driver','analysis/argoverse_hd_phase3/run.py')}, observational adapter, cache-only timing, paired analysis/report and final-verification helpers are isolated under analysis/argoverse_hd_phase3. The executed source snapshot, original production copies, full pre-existing working-tree diff, all config/schedule specifications and file hashes are preserved in {ROOT}. Production algorithm source stayed fixed across all scored and timed runs. A stage cannot silently overwrite completed work.

An interruption exposed an external repository update: three earlier Markdown reports changed only absolute/relative links and repository-root wording. The timing guard stopped before any replay. Exact inverse normalization reconstructed each original document hash; all executed code and635 archived Phase2 artifact hashes remained unchanged. Both document versions and the explicit manifest revision are preserved in {link('document_portability_revision.json','runs/argoverse_hd_phase3_v1/document_portability_revision.json')}. The changed documentation and external Git commit were preserved. No completed experiment was repeated, and no commit/push was made by this agent. A denied read-only process-list query was unnecessary; completion markers resolved execution state.

Commands executed from `{BASE}`:

```sh
PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage prepare
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage controls
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage variants
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage select
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage efficiency
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage timing
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.analyze
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.report
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.verify
```

The first timing invocation was stopped by the documented report-hash guard; the successful invocation performed six new cache-only repetitions. No scoring/replay stage was restarted. Scored archive resumability was available but unnecessary. The run completed18 of the approved maximum22 replays because no new confidence arm passed the pre-specified screen. Timings/outputs are fresh; original dataset and experimental results remain untouched.

Artifacts: {link('summary.csv','runs/argoverse_hd_phase3_v1/diagnostics/summary.csv')}, {link('full paired/event diagnostics','runs/argoverse_hd_phase3_v1/diagnostics/summary.json')}, {link('selection gates','runs/argoverse_hd_phase3_v1/selection.json')}, {link('full candidate config','runs/argoverse_hd_phase3_v1/candidate_config.json')}, {link('timing_summary.json','runs/argoverse_hd_phase3_v1/timing_summary.json')}, {link('final audit','runs/argoverse_hd_phase3_v1/final_verification.json')} and {link('artifact hashes','runs/argoverse_hd_phase3_v1/artifact_sha256.json')}. Per-arm scoring/metrics.json contains aggregate/class/size/sequence scores; official match identities and tensors, predictions, details, tracker states, source links, events and per-frame timing are retained. Full source and test snapshots accompany the audit. Reproduction requires a newly specified fresh destination/plan; completed stages refuse overwrites.
'''
    # Keep compact source literals readable in the delivered document.
    for old,new in [('Completed20','Completed 20'),('October2026','October 2026'),('and6','and 6'),('or61,488','or 61,488'),
        ('Phase4','Phase 4'),('Phase2','Phase 2'),('Phase3','Phase 3'),('a3x3','a 3x3'),('same1920','same 1920'),('and3,416','and 3,416'),
        ('imgsz640','imgsz 640'),('confidence0.001','confidence 0.001'),('NMS0.7','NMS 0.7'),('max_det300','max_det 300'),
        ('pycocotools2.0.11','pycocotools 2.0.11'),('identical1,710','identical 1,710'),('and1,706','and 1,706'),(':49.94',': 49.94'),
        ('at0.','at 0.'),('to0.','to 0.'),('the0.','the 0.'),('all3,416','all 3,416'),('flow_grid from4 to3','flow_grid from 4 to 3'),
        ('width640','width 640'),('minimum4','minimum 4'),('uses1,142','uses 1,142'),('frame0','frame 0'),('to1+j','to 1+j'),
        ('PhaseU','phase U'),('confidence0.25','confidence 0.25'),('exact-confidence0.25','exact-confidence 0.25'),('same1,706','same 1,706'),
        ('although4','although 4'),('threshold0.30','threshold 0.30'),('The3x3','The 3x3'),('At0.','At 0.'),('adds9','adds 9'),
        ('removes9','removes 9'),('adds12','adds 12'),('and3 FPs','and 3 FPs'),('from0.','from 0.'),('All84','All 84'),
        ('all84','all 84'),('and635','and 635'),('completed18','completed 18'),('maximum22','maximum 22'),('for12','for 12'),
        ('from3,647','from 3,647'),('to2,967','to 2,967'),('fall1,232','fall 1,232'),('fall2,258','fall 2,258'),
        ('rejects286','rejects 286'),('of15,095','of 15,095'),('versus212','versus 212'),('of15,082','of 15,082'),
        ('with4x4','with 4x4'),('versus3,647','versus 3,647')]:text=text.replace(old,new)
    dump(ROOT/'candidate_config.json',load(ROOT/'arms/R_T3_recovery__grid3/config.json'))
    dump(ROOT/'recommendation.json',{'practical_candidate':'R_T3_recovery__grid3','strict_all_metric_screen_passed':False,
        'accuracy_difference_pp':100*(grid['AP']-base['AP']),
        'low_confidence_FP_count_increase_explicit':True,'precision_alternative':'V3_low_iou50',
        'precision_alternative_has_AP_AR_recall_loss':True,'combined_variant_run':False,
        'cached_wall_mean_change_percent':wall_change,'cached_cpu_mean_change_percent':cpu_change,
        'deployed_runtime_validated':False,'new_inference':0})
    report=BASE/'docs/PHASE3_RESULTS.md'
    with report.open('x') as f:f.write(text)
    with (ROOT/'report.md').open('x') as f:f.write(text)
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=False)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    selected=['R_T1_anchor','R_T3_recovery','R_T4_shared10','V2_active_recovery','V3_low_iou50','V4_active_low_iou50','R_T3_recovery__grid3']
    for i,n in enumerate(selected):
        ax.scatter(rows[n]['AP']*100,rows[n]['SKIP_FP_conf0.1'],s=60,label=LABELS[n],marker=['o','o','o','v','s','D','^'][i])
    ax.legend(loc='upper left',fontsize=8,framealpha=.95)
    ax.set_xlabel('COCO AP50–95 (%)');ax.set_ylabel('SKIP false positives, confidence ≥0.10');ax.set_title('Accuracy and low-score FP trade-off');ax.grid(alpha=.2);ax.margins(x=.3,y=.12)
    bx.bar(['Recovery 4x4','Recovery 3x3'],[btime['wall_mean_ms'],ctime['wall_mean_ms']],color=['#647d98','#229979'])
    bx.set_ylabel('Mean cache-only step wall time (ms)');bx.set_title('3 interleaved repeats per method')
    for i,d in enumerate([btime,ctime]):bx.scatter([i]*3,d['round_wall_means_ms'],color='black',s=20,zorder=3)
    bx.set_xlabel('Dots: repetition means; excludes decoding, model inference and I/O',fontsize=8)
    for suffix in ['png','svg']:fig.savefig(figdir/f'tradeoffs.{suffix}',dpi=160)
    plt.close(fig)
    print('Report:',report,flush=True)


if __name__=='__main__':main()
