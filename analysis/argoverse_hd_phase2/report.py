"""Generate the Phase 2 report from completed approved experiments only."""
import argparse
import csv
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/adaptive_phase2_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
from .common import dump, load
from .analyze import catalog


def table(head, rows):
    return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)


def percent(value):return 'undefined' if value is None else f'{value*100:.2f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out).resolve();base=out.parents[1]
    if not (out/'ANALYSIS_COMPLETE.json').exists():raise RuntimeError('Analysis incomplete')
    summary=load(out/'combined_diagnostics/summary.json');arms=catalog(out)
    points=load(out/'combined_diagnostics/operating_points_by_decision_age.json')
    A=load(out/'A_saved_pairing/summary.json');dense=load(out/'A_saved_pairing/Dense_scoring/metrics.json')['COCO']
    phase1=Path(load(out/'execution_plan.json')['approved_plan']['verified_scoring'])
    names=['fixed2_093']+[f'T{i}_'+suffix for i,suffix in [(1,'fixed2_anchor'),(2,'fixed2_quality'),(3,'existing_track_recovery'),(4,'low_birth_threshold'),(5,'no_score_decay'),(6,'hold_boxes')]]
    labels=['T0 Fixed-2 original','T1 + anchoring','T2 + quality gates','T3 + existing-track recovery','T4 shared threshold 0.10','T5 no score decay','T6 hold boxes']
    def metric(name):return arms[name]['metrics']['COCO']
    def link(label,relative,line=None):return f'[{label}]({base/relative}'+(f':{line}' if line else '')+')'
    text='''# Phase 2: measured performance bottlenecks

Completed on 19 October 2026 using the approved A/R/B/C scope. No new YOLO inference, model loading, training, D/E run, acceleration port or Phase 3 implementation occurred. The original detector, tracker, flow and scheduling source files remain unchanged.

The results support **confidence lifecycle handling on top of an anchored fixed schedule** as the next research direction. They do not support a more complicated scheduler or treating flow quality as box correctness. Lower-score observations recover genuine AP, but also propagate more false positives; the preferred deployment threshold and accuracy/cost tolerance remain unspecified.

## Validity and scope

Experiment R passed before dependent experiments: every prediction array, decision and prediction-detail array reproduces bitwise for all four original temporal runs on all 3,416 frames. Track/output counts and ages also agree. Official COCO aggregate metrics reproduce Phase 1 to absolute tolerance1e-14. Frozen-schedule C controls reproduce the corresponding natural anchoring and quality-aware runs as well.

There were 21 complete cached replays (71,736 frame steps), plus saved-output scoring/analysis and visual review. Every FULL request returned the original stock-YOLOv8n Dense prediction for that validated image. These are simulated detector requests, not new detector executions. Torch/Ultralytics imports are blocked in the execution driver, crop/warmup calls raise errors, and source/image/cache hashes are checked. State resets occur at each of the five original sequence boundaries. No new data or pseudo-labels were produced.

The scorer reuses pinned pycocotools2.0.11 and Phase 1's positive annotation-ID conversion, original-image coordinates, eight-class taxonomy, crowd policy and standard maxDets=[1,10,100]. AP means COCO AP50–95; AR100 is averaged over classes and IoUs. Per-sequence scores are evaluated independently. All score entries below are percentages; deltas are percentage points. Aggregate AP is not an average of sequence APs.

All replay components execute on the original CPU NumPy/SciPy/OpenCV path. Detector inference device is **not applicable** to cached execution. This phase does not implement the cancelled GPU proposal or make a GPU speed claim.

## B: controlled Fixed-2 results

All seven arms use exactly 1,710 FULL requests and 1,706 SKIPs, saving49.94% of requests relative to Dense. FULL output is the same cached detector output in every arm; only the declared tracking configuration changes. Dense's prior measured AP is included as a reference.

'''
    text+=table(['Arm','AP50','AP50–95','AP75','AR100'],[['Dense reference',*[percent(dense[k]) for k in ['AP50','AP','AP75','AR100']]]]+[[label,*[percent(metric(n)[k]) for k in ['AP50','AP','AP75','AR100']]] for label,n in zip(labels,names)])
    text+=f"\n![Accuracy and false-positive trade-off]({out/'figures/confidence_tradeoff.png'})\n"
    t1,t3,t4=[metric(n)['AP'] for n in [names[1],names[3],names[4]]]
    text+=f'''\nT1 improves over the original Fixed-2 by {100*(t1-metric(names[0])['AP']):.4f} points at identical detection times. T3 adds {100*(t3-t1):.4f} points over T1; T4 adds {100*(t4-t1):.4f}. T4 reaches {100*t4/dense['AP']:.2f}% of Dense AP with half the requested detector calls, leaving an absolute AP gap of {100*(dense['AP']-t4):.4f} points. This is retention of a weak absolute baseline, not high absolute detection accuracy.

T3 is a two-stage confirmation experiment: high-confidence births remain0.25, and observations in [0.10,0.25) can confirm existing unmatched tracks. **T4 is not a pure birth-only intervention:** track_min_conf is shared by first-stage association and spawning, so lowering it to0.10 changes both. Its additional gain cannot be attributed exclusively to new births. No extra unapproved arm was added to remove that confound.

T2's gain over T1 is only {100*(metric(names[2])['AP']-t1):.4f} points. T5 changes AP by {100*(metric(names[5])['AP']-t1):+.4f} points with identical internal tracker states and output geometry/classes. Its different fixed-threshold counts are score calibration effects, not newly located objects. T6 reduces tracking work but loses {100*(t1-metric(names[6])['AP']):.4f} AP points: optical flow is useful here.

### Reliability at stated confidence thresholds

Below are official TP/FP counts on the same1,706 SKIP frames at IoU0.50. Ignored/capped predictions are excluded according to COCO. These counts describe operating points, not AP, and all classes are pooled.

'''
    text+=table(['Arm','TP ≥0.001','FP ≥0.001','Precision ≥0.001 %','TP ≥0.25','FP ≥0.25'],[[lab,points[n]['SKIP']['thresholds']['0.001']['50']['TP'],points[n]['SKIP']['thresholds']['0.001']['50']['FP'],percent(points[n]['SKIP']['thresholds']['0.001']['50']['micro_precision']),points[n]['SKIP']['thresholds']['0.25']['50']['TP'],points[n]['SKIP']['thresholds']['0.25']['50']['FP']] for lab,n in zip(labels,names)])
    text+='''
T3 and T4 improve low-score recall/AP, while the0.25 operating point is essentially unchanged. T4 more than quadruples SKIP false positives relative to T1 at the retained-score floor. That is a real trade-off, even though Dense itself emits many low-score false positives. Lowering a threshold is not evidence that the additional outputs are uniformly reliable. Do not promote their confidence merely to make more boxes pass0.25.

Across all frames, fixed-confidence0.25 class-macro statistics are:

'''
    text+=table(['Arm','Macro precision %','Macro recall %','Macro F1 %'],[[lab,*[percent(arms[n]['metrics']['fixed_conf25'][k]) for k in ['macro_precision','macro_recall','macro_F1']]] for lab,n in zip(labels,names)])
    text+='\nT5 raises thresholded recall and FP together using unchanged geometry. This distinguishes a metric/operating-point effect from recovery of missing detections.\n\n### Per-sequence and object-size results\n\n'
    text+=table(['Sequence']+['T0','T1','T2','T3','T4','T5','T6'],[[seq]+[percent(arms[n]['metrics']['per_sequence'][seq]['AP']) for n in names] for seq in arms[names[0]]['metrics']['per_sequence']])
    text+='\nT3 and T4 improve AP over T1 on all five sequences in this diagnostic set. This is descriptive consistency, not a held-out or statistical-significance claim. Class support differs across sequence APs.\n\n'
    selected=[names[i] for i in [1,3,4]]
    text+=table(['Arm','AP small','AP medium','AP large'],[['Dense',*[percent(dense[k]) for k in ['AP_small','AP_medium','AP_large']]]]+[[n,*[percent(metric(n)[k]) for k in ['AP_small','AP_medium','AP_large']]] for n in selected])
    text+='\nSmall-object detection remains poor even with T4. The experiments do not isolate resolution or original inference truncation; D was not run. Full class AP is:\n\n'
    text+=table(['Class','Dense','T1','T3','T4'],[[cls,percent(dense['per_class'][cls]['AP'])]+[percent(metric(n)['per_class'][cls]['AP']) for n in selected] for cls in dense['per_class']])
    text+='\n## A: separate baseline misses from SKIP losses\n\nOn the1,627 quality-aware SKIP frames there are26,635 regular targets. Pairing official GT match identities gives:\n\n'
    aq=A['methods']['adaptive_quality']
    text+=table(['Threshold','Both match','Neither matches','Dense only','Adaptive only'],[[key]+[aq['paired_gt_matches'][key].get(k,0) for k in ['both','neither','Dense_only','adaptive_only']] for key in ['IoU50_conf0.001','IoU50_conf0.25','IoU75_conf0.001','IoU75_conf0.25']])
    text+='\nThe all-score gap is substantial, but most missing detector evidence has low confidence. Among the7,184 Dense-only TP50 targets, previous-FULL evidence was:\n\n'
    text+=table(['Previous-FULL evidence','Target instances'],[[k,v] for k,v in aq['previous_FULL_candidate_evidence'].items()])
    text+='''
Actual tracker event logs show no correct previous detector-row link for7,145 of these cases; the39 with such a link still have an internal track, without a correct current official output. The no-link group also includes previous detector misses and absent annotation identities. This is not proof that every missing target could be recovered by tracking, nor an additive AP decomposition. The controlled B interventions establish that confidence handling does contribute to the AP deficit; they do not uniquely allocate the entire deficit among mechanisms.

Twenty-four predetermined diagnostic cases across all five sequences were visually reviewed with immediate-neighbour context (72 panels). They include small/occluded objects, low-score FULL observations absent on SKIP, shifted propagated extents, bus/car/truck and sign/light disagreements, and background light-like responses. These are illustrations, not prevalence estimates or a new annotation audit. No boxes/classes were relabelled. The images show nearest-overlap boxes for clarity, which need not be the internally admitted source row; runtime logs carry that linkage.

## Geometry and confidence lifecycle diagnostics

'''
    cohorts=summary['strong_seed_cohort']
    text+='The following uses one identical GT cohort for every fixed-schedule arm: both previous FULL and current Dense have an official TP75 with confidence≥0.25, joined using the provided annotation identity. It isolates performance where a strong seed and a strong current detector reference exist, rather than averaging over objects never admitted.\n\n'
    text+=table(['Arm','Eligible GT instances','Adaptive TP50','Adaptive TP75'],[[lab,cohorts[n].get('GT_instances',0),cohorts[n].get('adaptive_TP_IoU50',0),cohorts[n].get('adaptive_TP_IoU75',0)] for lab,n in zip(labels,names)])
    text+='\nRuntime summaries separately retain hidden/deleted/unemitted states and geometry of uniquely linked emitted tracks. They are conditional identity/geometry diagnostics; annotation IDs may fragment, and unlinked detector FPs are not silently treated as correct geometry. Key event counts are:\n\n'
    diagnostic={n:load(out/'combined_diagnostics'/f'{n}_runtime.json') for n in names}
    text+=table(['Arm','Spawn events','Births <0.25','Low-score confirmations','Hidden-track recoveries','Flow rejections'],[[lab]+[diagnostic[n]['event_counts'].get(k,0) for k in ['spawn','births_below_25','low_score_confirmations','hidden_track_recoveries','flow_rejected_updates']] for lab,n in zip(labels,names)])
    text+='\nCounts are track-update events across sequences, not unique physical objects. Detailed centre, width/height, IoU, age, visibility and flow-quality distributions are in combined_diagnostics. Confidence decay does not delete tracks in this implementation; the score-only control confirms that distinction.\n\n## C: does scheduling improve the trade-off?\n\nAll cells below use the same anchored tracker and legacy adaptive execution path, including propagation before FULL. Archived schedules are forced by the analysis harness, while tracker updates remain normal. Uniform U0 and phase-shifted U1 have identical FULL budgets per sequence. Both phases were pre-specified.\n\n'
    schedule_pairs=[('Original',1733,'S0_anchor_archived_original','S1_anchor_uniform_original','S_phase_anchor_uniform_C_fullfallback'),('Anchoring',1729,'S2_anchor_archived_anchor','S3_anchor_uniform_anchor','S_phase_anchor_uniform_adaptive_observed'),('Quality-aware',1789,'S4_anchor_archived_quality','S5_anchor_uniform_quality','S_phase_anchor_uniform_adaptive_quality')]
    text+=table(['Budget source','FULL requests','Archived AP %','Uniform U0 AP %','Uniform U1 AP %','Archived−U0 pp','Archived−U1 pp'],[[lab,count,percent(metric(a)['AP']),percent(metric(u)['AP']),percent(metric(v)['AP']),f"{100*(metric(a)['AP']-metric(u)['AP']):+.4f}",f"{100*(metric(a)['AP']-metric(v)['AP']):+.4f}"] for lab,count,a,u,v in schedule_pairs])
    text+='''
These controls test allocation given an archived per-sequence budget and a common engine. Their total budgets are known offline; they are not newly validated online schedulers. A small gain over one phase is insufficient evidence of a superior adaptive policy. The original native Fixed-2+anchoring control is also competitive with fewer calls. Its early-FULL path differs from C, so comparing it with C is a practical system comparison, not a pure schedule intervention.

The schedule × quality-flow factorial is:

'''
    text+=table(['Engine','Archived anchoring schedule (1729)','Archived quality schedule (1789)'],[['Anchoring only',percent(metric('S2_anchor_archived_anchor')['AP']),percent(metric('S4_anchor_archived_quality')['AP'])],['Anchoring + quality gates',percent(metric('S6_quality_archived_anchor')['AP']),percent(metric('S7_quality_archived_quality')['AP'])]])
    q0=metric('S2_anchor_archived_anchor')['AP'];q1=metric('S4_anchor_archived_quality')['AP'];q2=metric('S6_quality_archived_anchor')['AP'];q3=metric('S7_quality_archived_quality')['AP']
    text+=f'''\nWith the quality schedule fixed, adding quality gates changes AP by {100*(q3-q1):+.6f} points. With the anchoring schedule fixed, the change is {100*(q2-q0):+.6f}. Changing the schedule without quality gates adds {100*(q1-q0):+.6f} points and60 calls. The interaction is {100*(q3-q1-q2+q0):+.6f} points. These effects depend on the reference schedule; AP is nonlinear. The historical incremental quality-aware result is therefore not evidence of a generally improved accepted-box estimator.

Same-history pre-FULL shadow outputs are compared with Dense, with initial FULL frames excluded. Correlations use **pre-decision** states, not post-FULL track counts that would leak detector information. Immediate TP/FP changes, including negative benefits and both selected FULL/SKIP cohorts, are retained in per-frame utility CSVs. These are not downstream causal effects or streaming accuracy. Neither motion/flow consistency nor a current detection miss alone establishes the value of another detector call.

'''
    utilities=summary['schedule_utilities']
    text+='At confidence 0.25 and IoU0.50, the immediate mean changes per frame are:\n\n'
    text+=table(['Schedule','Decision','Frames','Dense − propagated TP/frame','Dense − propagated FP/frame'],[
        [label,decision,utilities[n]['by_decision'][decision]['frames'],
         f"{utilities[n]['by_decision'][decision]['Dense_minus_propagation_TP50_conf0.25']['mean']:+.3f}",
         f"{utilities[n]['by_decision'][decision]['Dense_minus_propagation_FP50_conf0.25']['mean']:+.3f}"]
        for label,n in [('Original allocation','S0_anchor_archived_original'),('Quality allocation','S7_quality_archived_quality')]
        for decision in ['FULL','SKIP']])
    text+='\nSelected FULL frames have greater immediate mean TP benefit, but also introduce more FP. Median TP benefit is zero in all four cohorts. This shows some targeting, not a validated optimal allocation or a downstream accuracy gain. For the quality allocation, Spearman correlations with immediate TP benefit are age +0.114, novelty +0.024, failure fraction −0.037, flow quality +0.006 and uncertainty +0.016. These weak marginal associations do not rule out nonlinear signals; temporal dependence and selection remain.\n\n## Computational costs and measurement limits\n\n'
    text+=table(['Arm','Mean internal tracks','Initial LK points total','Replay step wall ms','Replay step CPU ms','Diagnostic callback wall ms'],[[lab,f"{arms[n]['cost']['mean_post_frame_tracks']:.2f}",arms[n]['cost']['total_flow_point_evaluations_before_FB'],f"{arms[n]['cost']['mean_corrected_step_wall_ms']:.3f}",f"{arms[n]['cost']['mean_corrected_step_cpu_ms']:.3f}",f"{arms[n]['cost']['mean_diagnostic_wall_ms']:.3f}"] for lab,n in zip(labels,names)])
    total_replay=sum(a['cost']['stage_wall_seconds_including_decode_hash_diagnostics_IO'] for a in arms.values())
    total_score=sum(a['metrics']['scoring_wall_seconds'] for a in arms.values())
    text+=f'''\nThere were {summary['cache_FULL_lookups']:,} simulated FULL cache lookups and **zero** new YOLO executions. Summed recorded replay-stage wall time across21 arms was {total_replay:.1f}s, including image decoding/hashing, diagnostic serialization and archive writing. Summed per-arm scoring wall time was {total_score:.1f}s, including aggregate and independent per-sequence COCO evaluation. These sums exclude preparation, A's Dense rescoring, manual rendering and subsequent combined analysis, and are not an end-to-end benchmark runtime.

Replay-step times subtract measured diagnostic callback time; residual wrapper/cache overhead remains. They exclude decode, hashing, event serialization, scoring and model execution. This is one sequential replay per arm, without timing counterbalancing or live detector contention. CPU time can exceed wall time due to library worker threads. Small timing differences are not proven speedups. Initial LK point totals count grid points passed to forward flow; the backward check also runs.

T3/T4 increase state and point work while preserving the same detector-request budget. The high-confidence admission baseline deliberately tracks far fewer boxes than Dense emits. Lowering admission can narrow the accuracy gap but weakens that tracking-efficiency advantage. Actual combined YOLO+tracking FPS, hardware energy and MPS performance were not remeasured; E was not run. The existing GFLOPs formula covers detector arithmetic only and cannot price this additional tracking work.

## Evidence-based Phase 3 recommendation

1. **Use anchored Fixed-2 as the required working control.** It improves localisation at the same requests; a more complicated scheduler has not established a robust advantage over the pre-specified uniform controls or this practical reference. Preserve stock YOLOv8n.
2. **Investigate continuation separately from births.** T3 is the smaller existing mechanism to carry forward for validation: high-confidence births with lower-score confirmation. T4 demonstrates further headroom but changes association too, increases false outputs and more than doubles internal tracks. It is not a justified blanket deployment threshold. A future birth-only intervention must freeze the high-stage association rule, with an explicit added-track/FP budget.
3. **Treat score calibration and detection reliability separately.** T5 can move boxes across0.25 without improving geometry. Agree on an application operating point and acceptable miss/false-alarm costs; do not select a policy by AP or thresholded F1 alone. Validate confidence handling on new approved data before deployment.
4. **Keep flow and anchoring; do not replace the tracker or add a scheduler network now.** Holding boxes costs accuracy, and quality gates add little conditional AP. Only pursue a more complicated flow/scale/occlusion model if the linked residual errors justify its cost. No camera/appearance/ROI/CLAHE sweep is supported by these experiments.
5. **A scheduler must predict useful refresh benefit at matched total cost.** Cheap signals should be validated against unseen-object/retention/localisation risk, including invisible low-score evidence, rather than only health of already admitted tracks. Frozen-schedule results are diagnostic; a new prospective closed-loop controller requires separate approval and comparison with both periodic phases. Do not claim superiority from extra calls.

These priorities follow mechanisms rather than transplanting complete systems. [ByteTrack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) motivates continuation from lower scores while preserving high-score births; its MOT results do not promise detection AP gains here. [OC-SORT](https://arxiv.org/html/2203.14360v3) supports observation-centric state correction, while our anchoring is a simpler existing mechanism. [Detect or Track](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/02/AAAI-LuoH.pdf) motivates measuring actual tracking quality and refresh utility; its learned scheduler/tracker is a larger trained system and was not adopted. [Towards Streaming Perception](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470460.pdf) separates frame-aligned AP from latency-aware streaming AP; no streaming result is claimed in this phase.

Remaining uncertainty: original pre-LUT/global top300 losses, causal resolution effects, detailed annotation quality, physical identity through annotation fragmentation, true live-inference total cost, and generalization beyond five already-analysed sequences. No held-out accuracy, statistical significance, calibration guarantee, energy saving or expected Phase 3 AP gain is claimed. D/E and Phase 3 implementation remain unexecuted.

## Root causes ranked by evidence and impact

1. **Weak detector evidence limits absolute performance (confirmed).** Dense AP is 15.73%, with small-object AP 1.01% versus large-object AP 53.46%. Adaptive tracking cannot reliably create new class evidence for objects the detector never identifies. These size strata establish a disparity, not the causal effect of resolution; no resolution sweep was approved. Dense is a reference, not a mathematical upper bound on temporal AP.
2. **Confidence admission/continuation loses usable evidence (confirmed intervention).** T3/T4 recover AP at unchanged detector requests; this is larger than either quality gating or score decay. Their added false positives and tracking work constrain useful gains. T4 cannot separate births from first-stage association.
3. **Refresh allocation has no robust advantage over periodic controls (confirmed on this subset).** Each archived adaptive schedule beats one budget-matched phase and loses to the other. Weak marginal signal/benefit correlations support diagnosing the objective before adding controller complexity; they do not prove all adaptive scheduling is ineffective.
4. **Propagation geometry matters, but is not the only or dominant recoverable issue (supported).** Anchoring improves the common strong-seed cohort at TP75; holding boxes hurts. Quality gates add only small conditional gains. This does not exclude geometry failures on difficult/low-score objects outside the cohort.
5. **Score decay has negligible AP impact here (confirmed control).** Removing it leaves state/geometry unchanged and adds 0.0053 AP points. It changes a chosen confidence operating point, so promoting scores is not a substitute for recovering detections.

## Changes and reproducibility

'''
    text+=f'''New files are confined to {link('the analysis harness','analysis/argoverse_hd_phase2/run.py')}, its cache/observation, scoring, pairing, summary/report and rendering helpers, {link('synthetic harness tests','tests/test_phase2_harness.py')}, and this report. The frozen plan, existing dataset, original benchmark, Phase 1 outputs and production algorithm code remain intact. All73 tests passed, including5 new cache/wrapper/official-match tests. Two test-development failures (an incorrect unittest module invocation and an assertion looking for a birth on the final SKIP frame) were corrected before any dataset replay; they did not alter algorithms or benchmark results.

Implementation evidence: {link('FULL admission, two-stage association and spawning','extracted/adaptive_pipeline/tracking/tracker.py',224)} establishes the shared threshold and T3 semantics; {link('SKIP outputs','extracted/adaptive_pipeline/tracking/tracker.py',264)} applies decay to scores without deleting state. {link('fixed anchored early FULL','extracted/adaptive_pipeline/pipeline.py',99)} explains its reduced flow work relative to the legacy adaptive path. {link('legacy controller','extracted/adaptive_pipeline/adaptive_framing/controller.py',93)} computes health/refresh eligibility on visible admitted tracks; its uncertainty trigger uses detector confidence and age. {link('cache and observation wrappers','analysis/argoverse_hd_phase2/replay.py',59)}, {link('official scoring adapter','analysis/argoverse_hd_phase2/scoring.py',11)} and {link('same-history utility analysis','analysis/argoverse_hd_phase2/analyze.py',141)} provide the diagnostic implementation. Exact source snapshots accompany the run.

The approved plan is {link('plan.json','analysis/argoverse_hd_phase2_plan_v1/plan.json')}. Exact full configs, archived/uniform schedules and source hashes are retained there and copied into the new run. The execution driver enforces R completion before A/B/C. The primary output is {link('summary.csv','runs/argoverse_hd_phase2_diagnostics_v1/combined_diagnostics/summary.csv')}, with per-arm COCO metrics, per-sequence/class/size scores, official GT match identities, prediction archives, tracker states/hashes, event logs and per-frame timing/utility data. The manual review is documented in {link('review_notes.json','runs/argoverse_hd_phase2_diagnostics_v1/A_saved_pairing/manual_visuals_v2/review_notes.json')}.

The independent {link('final archive audit','analysis/argoverse_hd_phase2/verify.py')} checks saved arrays, official tensors, aggregate/class/sequence scores, exact forced schedules, every FULL cache output, production snapshots and all 71 frozen source-evidence hashes. It reruns synthetic tests only. The machine-readable audit is {link('final_verification.json','runs/argoverse_hd_phase2_diagnostics_v1/final_verification.json')}; artifact hashes are in {link('artifact_sha256.json','runs/argoverse_hd_phase2_diagnostics_v1/artifact_sha256.json')}.

Commands executed from `{base}`:

```sh
PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.run --stage R --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.run --stage A --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.run --stage B --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.run --stage C --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.manual_review --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.analyze --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.report --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.verify --out runs/argoverse_hd_phase2_diagnostics_v1
```

The initial visual render was retained separately and re-rendered with clipped labels for readability; no experiment was rerun. Fresh directories are required. The one-off driver intentionally binds to the approved plan/output path and refuses completed stages; reproduction in another destination requires a newly specified plan/output, not overwriting these results. Runtime/backend, source snapshots and final verification records accompany the outputs. No commit or push was made.
'''
    # Normalize a few compact numeric phrases for readability.
    for a,b in [('tolerance1e-14','tolerance 1e-14'),('pycocotools2.0.11','pycocotools 2.0.11'),('saving49.94','saving 49.94'),('remain0.25','remain 0.25'),('to0.10','to 0.10'),('same1,706','same 1,706'),('the0.25','the 0.25'),('pass0.25','pass 0.25'),('the1,627','the 1,627'),('are26,635','are 26,635'),('the7,184','the 7,184'),('for7,145','for 7,145'),('the39','the 39'),('and60','and 60'),('across21','across 21'),('across0.25','across 0.25'),('All73','All 73'),('including5','including 5')]:text=text.replace(a,b)
    report=base/'docs/PHASE2_RESULTS.md'
    with report.open('x') as f:f.write(text)
    with (out/'report.md').open('x') as f:f.write(text)
    figures=out/'figures';figures.mkdir(exist_ok=False)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
    values=[metric(n)['AP']*100 for n in names]
    ax.scatter(values,np.arange(7),s=55,color=['#777777','#2463a6','#5488a8','#17906d','#c0771b','#9178a8','#a45c57'])
    ax.set_yticks(np.arange(7),labels);ax.invert_yaxis();ax.axvline(dense['AP']*100,color='black',linestyle='--',label='Dense AP reference')
    ax.set_xlabel('COCO AP50–95 (%)');ax.set_title('Identical Fixed-2 detector requests');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
    ss=[names[i] for i in [1,3,4]];tp=[points[n]['SKIP']['thresholds']['0.001']['50']['TP'] for n in ss];fp=[points[n]['SKIP']['thresholds']['0.001']['50']['FP'] for n in ss]
    bx.barh(np.arange(3),tp,label='TP50',color='#17906d');bx.barh(np.arange(3),fp,left=tp,label='FP50',color='#c87763')
    bx.set_yticks(np.arange(3),['T1 anchor','T3 recovery','T4 threshold 0.10']);bx.invert_yaxis();bx.set_xlabel('SKIP prediction matches, confidence ≥0.001');bx.set_title('Low-score recall has a false-positive cost');bx.legend(fontsize=8)
    bx.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    bx.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    for suffix in ['png','svg']:fig.savefig(figures/f'confidence_tradeoff.{suffix}',dpi=160)
    plt.close(fig)
    print('Report written:',report,flush=True)


if __name__=='__main__':main()
