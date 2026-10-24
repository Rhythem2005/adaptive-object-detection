"""Build the focused six-approach research handoff after local validation."""
import collections
import csv
import json
from pathlib import Path
import statistics

from analysis.argoverse_hd_phase2.common import load,dump,sha
from analysis.branch_b_expanded_v1.continue_local import BASE,ROOT

NAMES={'dense':'Dense640','original_fixed2':'Original Fixed-2','practical':'Practical Fixed-2','high_ap':'High-AP Fixed-2',
    'fifo':'FIFO Dense640','latest':'Latest-frame Dense640','merged':'Latest-frame Practical Fixed-2'}


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(x) for x in row)+'|' for row in rows])


def fmt(value,d=3,percent=False):
    if value is None:return 'N/A'
    return f'{float(value)*(100 if percent else 1):.{d}f}'


def mean(rows,field):
    return statistics.mean(float(r[field]) for r in rows) if rows else None


def main():
    result=load(ROOT/'analysis/results.json');offline=load(ROOT/'analysis/offline.json')
    if not (ROOT/'analysis/failure_cases.json').exists():
        from analysis.branch_b_expanded_v1.cases_local import main as derive_cases
        derive_cases()
    supplement=load(ROOT/'analysis/offline_supplement.json');figs=load(ROOT/'figures/manifest.json')['figures']
    with (ROOT/'analysis/streaming.csv').open() as f:rows=list(csv.DictReader(f))
    valid=[r for r in rows if r['pacing_valid']=='True'];allrecords=result['records']
    complete=len(rows);invalid=len(rows)-len(valid);incomplete=len(allrecords)-complete
    bycell={}
    for r in valid:bycell.setdefault((r['sequence'],r['rate'],r['trial']),{})[r['mode']]=r
    pairs=[]
    for cell,d in bycell.items():
        if {'latest','merged'}<=set(d):
            b,c=d['latest'],d['merged']
            pairs.append(dict(sequence=cell[0],rate=int(cell[1]),trial=int(cell[2]),
                aligned_AP_pp=100*(float(c['aligned_AP'])-float(b['aligned_AP'])),
                cpu_reduction_percent=100*(1-float(c['loop_cpu_s'])/float(b['loop_cpu_s'])),
                p95_reduction_percent=100*(1-float(c['latency_p95_ms'])/float(b['latency_p95_ms'])),
                call_reduction_percent=100*(1-float(c['calls'])/float(b['calls'])),
                coverage_pp=100*(float(c['frame_coverage'])-float(b['frame_coverage'])),
                common_AP_pp=100*(float(c['common_AP'])-float(b['common_AP'])),
                latency_allowance_pass=float(c['latency_p95_ms'])<=max(float(b['latency_p95_ms'])+5,1.15*float(b['latency_p95_ms'])),
                maximum_latency_pass=float(c['latency_max_ms'])<=150,
                B_name=b['name'],C_name=c['name']))
    dump(ROOT/'analysis/paired_effects.json',pairs)
    pages=[]
    def page(title,body):pages.append('# '+title+'\n\n'+body)
    page('Branch B Research Handoff',
        '## Expanded local-sequence validation\n\n'
        '**Six completed approaches plus FIFO streaming control.**\n\n'
        'Research: Practical YOLO-Based Object Detection Under Challenging Visual Conditions: Low-Light Failure Analysis and Adaptive Real-Time Video Processing. This dossier covers Branch B only.\n\n'
        'The four offline configurations reuse verified complete-dataset results. New streaming validation uses all five local Argoverse-HD sequences at 15, 30, 60 and 90 FPS with three intended trials. No training, tuning, new algorithm, external dataset access or download was performed during this resumed phase.\n\n'
        f'Execution: **{complete}/180 complete**, **{invalid} complete but pacing-invalid**, **{incomplete} incomplete**. Two completed attempts were independently verified and reused. One earlier user-interrupted merged attempt is preserved separately.\n\n'
        'Official frame-aligned COCO AP, custom replay-time-aligned AP, unpaced throughput and paced output FPS answer different questions. All AP values in this dossier are percentages; AP differences are percentage points.\n\n'
        'Evidence root: `runs/branch_b_expanded_local_v2/`. The earlier handoff is preserved under historical_handoff/.')
    statusrows=[]
    for mode in ['fifo','latest','merged']:
        for fps in [15,30,60,90]:
            allpart=[r for r in allrecords if r['mode']==mode and r['fps']==fps]
            done=[r for r in rows if r['mode']==mode and int(r['rate'])==fps]
            statusrows.append([NAMES[mode],fps,len(allpart),len(done),sum(r['pacing_valid']=='True' for r in done),len(allpart)-len(done)])
    page('1. Scope and execution status',
        'The scope is Dense640, Original Fixed-2, Practical Fixed-2, High-AP Fixed-2, latest-frame Dense640 (original adaptive framing), and latest-frame Practical Fixed-2 (integration). FIFO Dense640 is the streaming control. Fixed-3, rejected adaptive schedulers and ablations were not run.\n\n'+
        table(['Streaming arm','FPS','Intended','Complete','Pacing-valid','Incomplete'],statusrows)+'\n\n'
        'Complete means the entire source stream was received and normal EOF accounting passed. A pacing-invalid run can be complete while failing to reproduce the intended arrival process. It remains in the full tables and is excluded from primary timing-valid comparisons. Missing runs are never assigned zero AP.\n\n'
        'Full execution rows: analysis/execution_matrix.csv. Per-run measurements: analysis/streaming.csv. Valid-pair effects: analysis/paired_effects.json.')
    data=[]
    for seq,s in supplement['dataset'].items():
        data.append([seq,s['frames'],s['regular_targets'],s['crowds'],fmt(s['mean_targets_per_frame'],2),s['small_targets'],s['medium_targets'],s['large_targets']])
    page('2. Dataset and independence boundary',
        table(['Sequence','Frames','Regular GT','Crowds','GT/frame','Small','Medium','Large'],data)+'\n\n'
        'Total: 3,416 unique 1920 x 1200 frames, 55,412 regular targets and 2,299 crowd annotations. All original source IDs, ordering, timestamps, boxes and eight-category mappings are retained. Small/medium/large counts use area intervals <32^2, [32^2,96^2), and >=96^2; official COCO boundary conventions remain in the unchanged evaluator.\n\n'
        'These are the same five scenes used during development. Full-sequence streaming tests increase temporal and scene coverage relative to the old 300-frame sequence_00 prefix, but they are not unseen-scene validation. Repeated presentations do not create independent scenes. No generalization beyond this local collection is established.\n\n'
        'Classes: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign. COCO indices 0,1,2,3,5,7,9,11 map to dataset indices 0-7. No pseudo-label accuracy is substituted for ground truth. Preservation checks establish consistency with the provided export, not an independent label-quality or upstream-authenticity audit.')
    cfgrows=[]
    for n in ['dense','original_fixed2','practical','high_ap']:
        c=offline[n]['config']
        cfgrows.append([NAMES[n],c['policy'],c['fixed_k'] if c['policy']=='fixed' else 'N/A',c['flow_grid'] if c['policy']=='fixed' else 'N/A',
            c.get('observation_anchoring',False),c.get('low_score_association',False),c.get('track_birth_conf'),c.get('fixed_signal_fastpath',False)])
    page('3. Frozen configurations and provenance',
        'Shared detector: unchanged stock COCO-pretrained YOLOv8n; CPU FP32; batch 1; imgsz 640; prediction confidence .001; NMS IoU .7; global max_det 300. Actual tensor 1 x 3 x 416 x 640. Checkpoint SHA256: f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36.\n\n'+
        table(['Offline configuration','Policy','K','Grid','Anchor','Low recovery','Birth conf','Fastpath'],cfgrows)+'\n\n'
        'All Fixed-2 configurations use flow propagation and confidence decay .93. Practical uses existing-track recovery at .10 and retains .25 track admission. High-AP uses weak births at .10 and a 4 x 4 grid; it is a separate configuration. Full dictionaries and hashes are saved in configs/ and frozen_sha256.json. Inactive Dense tracking/scheduling fields have no effect because its always-detect path bypasses them.\n\n'
        'Streaming A/B use the frozen Dense configuration; C uses Practical plus the existing isolated streaming safety adapter. No production source file was changed. Historical source hashes and all previous results are audited separately.')
    page('4. The two contributions and the integration',
        '**Original adaptive framing:** independent capture -> capacity-one pending buffer -> atomic newest-frame selection -> Dense YOLO -> output. New arrivals overwrite obsolete unconsumed frames while processing continues. An in-flight frame is not cancelled.\n\n'
        '**Fixed-2 selective inference:** after consuming a frame, either run fresh YOLO (FULL) or propagate state with optical flow (SKIP). SKIP means no YOLO inference; it does not mean no image processing.\n\n'
        '**Integrated mode:** capture -> latest buffer -> select newest packet -> consumed-step Fixed-2 -> output. Original source-frame IDs and timestamps remain available for motion/aging. Unsafe propagation triggers FULL on the same frame and restarts the consumed-step cycle. Forced-FULL counts therefore cannot simply be added to half the consumed count. No second pending queue is introduced.\n\n'
        '**FIFO control:** independent capture -> FIFO of decoded packets -> consume in source order -> Dense YOLO. Capacity equals the complete finite sequence size rather than the former 300-frame experiment bound. This changes the safety capacity for the expanded population, not the arrival-order/no-dropping policy. Resource limits may abort a run; aborted runs are explicitly incomplete.')
    page('5. Measurement and evaluation definitions',
        'Official COCO bbox AP uses the unchanged pycocotools implementation, eight categories, IoU .50:.05:.95, maxDets [1,10,100], preserved crowds and positive copied annotation IDs. Supplementary precision, recall and F1 are micro values at confidence >=.25 and IoU .50. They are not COCO AR100.\n\n'
        'For each streaming sequence/rate/trial cell, frame-aligned AP is scored on the intersection of frames consumed by all complete arms in that cell. This conditional population is listed explicitly. It can vary with rate and trial, and is not an unbiased estimate of full-stream detector AP.\n\n'
        'Custom replay-time-aligned held-output AP evaluates all source-arrival queries. At each query it uses the actual most recently completed output, including flow outputs; before first completion, the output is empty. The output is scored against that query frame. This is not an official streaming benchmark.\n\n'
        'Outputs after the last source query contribute to frame-aligned AP and completion/drain metrics, but not to time-aligned query AP. No future-completed output is used at an earlier query.\n\n'
        'Offline FPS includes JPEG decode and pipeline processing where stated. Stream service throughput excludes decode/queue wait. Active output FPS counts completions inside the common first-to-last source-arrival window. Drain-inclusive output FPS includes post-EOF processing. These quantities are not interchangeable.')
    page('6. Replay protocol, reuse and safety',
        'Each complete sequence is replayed independently at 15, 30, 60 and 90 FPS, three intended trials per arm. Pacing uses original nanosecond timestamp differences scaled by the median native interval. Every arm receives the same complete source population and timestamp schedule; there is no preload, manual frame selection, or source backpressure.\n\n'
        'Fresh interpreter/model/processor state is used per new pass. Common thread controls: Torch intra/inter-op 8/1, OpenCV 10, OpenCL off. Three full-frame warmups precede the replay epoch. Source hashing occurs outside measurement and warms the file cache; cold-storage performance is not tested.\n\n'
        'The matrix rotates mode order within cells and reverses sequence/rate ordering in trial 2. This reduces some ordering effects but does not eliminate background, thermal or frequency variability. The two reused 15 FPS passes predate the user pause, so their session/order context is an additional limitation.\n\n'
        'Guards: 4,096 MiB process RSS at processing-loop checks, 180 s loop deadline, 240 s supervisor deadline, 12 GB minimum free disk, 5 GB output budget, four-hour execution budget. Guards are checked discretely; they are not hard real-time resource guarantees. FIFO never silently drops or throttles on a limit.\n\n'
        'Pacing validity retains the prior limits: publication span within 3% of intended rate; P95 decode-start lateness <=1 period; maximum lateness <=2 periods. Invalid trials are retained and excluded from primary comparisons without automatic replacement.\n\n'
        'Three-arm overview graphs use a fixed balanced scene/trial panel valid for every arm and rate, chosen solely from completion/pacing status. This prevents silently changing populations across plotted means. Full per-scene tables and paired B/C analysis retain all applicable data; the balanced graphs are explicitly a smaller subset.')
    offline_rows=[]
    for n,v in offline.items():
        p=v['fixed_conf25'];offline_rows.append([NAMES[n],fmt(v['COCO']['AP'],6,True),fmt(v['COCO']['AP50'],4,True),fmt(v['COCO']['AP75'],4,True),fmt(p['micro_precision'],3,True),fmt(p['micro_recall'],3,True),fmt(p['micro_F1'],3,True)])
    page('7. Reused whole-dataset offline accuracy',table(['Method','AP %','AP50 %','AP75 %','P25 %','R25 %','F1 %'],offline_rows)+'\n\n'
        'No new offline inference was necessary: these configurations and the 3,416-frame populations match the verified archived experiments. Original Fixed-2 official scores come from Phase 1 rescoring of its saved predictions. Dense/Practical/High-AP use Experiment E output-and-tensor fidelity evidence.\n\n'
        'Practical AP is 0.672230 percentage points above Original Fixed-2 and 1.040467 points below Dense. High-AP is 0.398819 points above Practical and remains 0.641648 points below Dense. These are multi-setting configuration comparisons, not single-factor causal effects.\n\n'
        'The full area-range metrics and per-sequence P/R/F1 evidence are in analysis/offline.json, offline.csv and offline_supplement.json.')
    offline_cost=[]
    for n,v in offline.items():offline_cost.append([NAMES[n],fmt(v['fps']),fmt(v['wall_ms_frame']),fmt(v['cpu_ms_frame']),v['calls'],v['timing_repetitions']])
    page('8. Offline efficiency and scope limits',table(['Method','FPS','Wall ms/frame','CPU ms/frame','Calls','Timing trials'],offline_cost)+'\n\n'
        'Dense, Practical and High-AP share the matched five-block E campaign. Practical saves 49.9415% of detector calls and 36.5029% of decode-inclusive CPU, with 49.3506% greater throughput than Dense. High-AP has higher AP and slightly lower throughput than Practical. Do not combine their best metrics.\n\n'
        'Original Fixed-2 timing is from the older benchmark: FPS here is recomputed as 1000 / mean(decode_ms + t_total_ms), rather than its historically named pipeline-only fps_end_to_end. Its CPU metric covers a broader evaluation process. It is reported for completeness but excluded from matched CPU comparison plots. No per-sequence CPU value is invented for it.\n\n'
        'Original timing instrumentation/campaign differs from E. Identical image populations make accuracy comparisons valid, but they do not make these runtime measurements paired. GFLOPs estimates are not used as measured CPU or energy savings.\n\n'
        'At confidence .10, Practical has 15,239 false-positive events versus High-AP 20,680. At .25 the corresponding counts are 4,373 and 4,383. The low-score trade-off must remain visible; High-AP accuracy and Practical FP counts belong to different configurations. Exact .10 operating points are in analysis/offline_conf10.json.')
    for fig in figs[:4]:
        rel=Path(fig['png']).relative_to(BASE/'research_handoff') if str(fig['png']).startswith(str(BASE/'research_handoff')) else Path('../')/Path(fig['png']).relative_to(BASE)
        page(fig['title'],f'![{fig["title"]}]({rel})\n\n{fig["scope"]}\n\n**Scientific interpretation:** {fig["interpretation"]}')
    for seq in [f'sequence_{i:02d}' for i in range(5)]:
        data=[]
        for fps in [15,30,60,90]:
            for mode in ['fifo','latest','merged']:
                part=[r for r in rows if r['sequence']==seq and r['mode']==mode and int(r['rate'])==fps]
                good=[r for r in part if r['pacing_valid']=='True']
                data.append([fps,NAMES[mode].replace('640',''),f'{len(part)}/3',len(good),fmt(mean(good,'frame_coverage'),2,True),fmt(mean(good,'latency_p95_ms'),2),fmt(mean(good,'cpu_ms_source'),2),fmt(mean(good,'aligned_AP'),3,True)])
        page(f'9. Streaming results: {seq}',table(['FPS','Arm','Complete','Valid','Coverage %','P95 ms','CPU ms/src','Aligned AP %'],data)+'\n\n'
            'Numerical means use only complete pacing-valid trials for this sequence. Coverage is consumed/source frames. AP is custom replay-time-aligned AP over the complete source query population. Complete but invalid runs remain in streaming.csv. The Complete and Valid columns make unavailable evidence explicit.\n\n'
            'These per-sequence means do not pool predictions across scenes. Individual trials, official common-population AP, precision, recall, F1, FP, calls, RSS, pending queues, draining, and fallback counts are in the complete machine-readable table.')
    pairrows=[]
    for fps in [15,30,60,90]:
        p=[r for r in pairs if r['rate']==fps]
        pairrows.append([fps,len(p),fmt(mean(p,'aligned_AP_pp')),fmt(mean(p,'common_AP_pp')),fmt(mean(p,'cpu_reduction_percent'),2),fmt(mean(p,'call_reduction_percent'),2),fmt(mean(p,'p95_reduction_percent'),2)])
    page('10. Paired integrated-versus-latest comparison',table(['FPS','Pairs','Aligned AP pp','Frame AP pp','CPU reduction %','Calls reduction %','P95 reduction %'],pairrows)+'\n\n'
        'Each pair has the same sequence, rate and trial, and both arms passed pacing validity. Values are arithmetic means of within-pair effects; they are not ratios of pooled means, pooled COCO AP, or significance estimates. Positive AP changes are gains; positive reductions are savings.\n\n'
        'The frame-aligned column uses the declared common-consumed population within each cell. The aligned column scores every query from that sequence. Their signs can differ because delivery timing, selected frames and propagation all contribute. No causal decomposition of those effects was run.')
    with (ROOT/'analysis/pooled_aligned.csv').open() as f:pooled=list(csv.DictReader(f))
    pooledrows=[]
    for fps in [15,30,60,90]:
        for mode in ['fifo','latest','merged']:
            part=[r for r in pooled if r['mode']==mode and int(r['rate'])==fps]
            good=[r for r in part if r['pacing_valid']=='True']
            pooledrows.append([fps,NAMES[mode].replace('640',''),len(part),len(good),fmt(mean(good,'AP'),4,True),
                fmt(mean(good,'precision'),2,True),fmt(mean(good,'recall'),2,True),fmt(mean(good,'F1'),2,True)])
    page('10a. Joint full-population held-output accuracy',table(['FPS','Arm','Complete cohorts','Valid cohorts','AP %','P25 %','R25 %','F1 %'],pooledrows)+'\n\n'
        'Each cohort jointly scores all 3,416 source queries from five independently reset complete streams for one arm/rate/trial. Metrics above average only cohorts whose five source replays passed pacing validity. Full cohort rows, including invalid timing, are in analysis/pooled_aligned.csv.\n\n'
        'This joint score is not the same as the descriptive mean of per-sequence AP. It remains a custom held-output diagnostic, not conventional per-frame detector AP or an official streaming leaderboard score. No pooled result is assigned to an incomplete cohort.')
    for fig in figs[4:]:
        rel=Path('../')/Path(fig['png']).relative_to(BASE)
        page(fig['title'],f'![{fig["title"]}]({rel})\n\n{fig["scope"]}\n\n**Scientific interpretation:** {fig["interpretation"]}')
    bad=[]
    for r in allrecords:
        if r['status']!='complete':
            p=Path(r['path'])/'INCOMPLETE.json';reason=load(p) if p.exists() else dict(error='No completion record')
            bad.append([r['sequence'],r['fps'],r['trial'],r['mode'],str(reason.get('error'))])
        else:
            s=result['passes'][r['name']]
            if not s['valid']:bad.append([r['sequence'],r['fps'],r['trial'],r['mode'],'Pacing invalid: '+','.join(k for k,v in s['source_validity'].items() if not v)])
    page('11. Failures, invalid timing and interrupted history',
        (table(['Sequence','FPS','Trial','Arm','Reason'],bad) if bad else 'All 180 selected passes completed and met the declared source-pacing criteria.')+'\n\n'
        'The earlier merged sequence_00/15 FPS trial was stopped by the user after 625 source publications and is preserved in runs/branch_b_expanded_v1/. It is not an additional successful trial. Its explicitly authorized restart is stored separately in the new root.\n\n'
        'Historical 300-frame integration failures remain: 90 FPS mean CPU reduction 4.8731% missed a 5% gate; 15 FPS trial 2 P95 46.5723 ms exceeded 43.2769 ms. New evidence does not retroactively change those outcomes.\n\n'
        'The old gates were predeclared for the prefix experiment. This expanded phase reports their latency allowance descriptively, without silently redefining 30/60 FPS adoption criteria or claiming retrospective confirmatory significance.')
    incomplete_file=ROOT/'analysis/incomplete.csv'
    if incomplete_file.exists():
        with incomplete_file.open() as f:partial=list(csv.DictReader(f))
        page('11a. Resource-aborted attempt accounting',table(['Sequence','FPS','Trial','Arm','Received','Dequeued','Pending','RSS MiB'],
            [[r['sequence'],r['rate'],r['trial'],r['mode'],r['received'],r['dequeued_not_verified_completed'],r['remaining_pending'],fmt(r['peak_rss_mib'],2)] for r in partial])+'\n\n'
            'These attempts did not receive/process the full intended stream. Dequeued counts are not substituted for a verified completed-prediction population because partial prediction arrays were not returned by the stopped consumer. No AP, CPU-per-full-stream saving, or completed-trial result is fabricated.\n\n'
            'FIFO retained its no-replacement policy. The capture thread was stopped and joined by the existing cleanup path. RSS is checked discretely and may slightly exceed the configured threshold before the next check. Full error, buffer and forward-counter records remain in each INCOMPLETE.json and analysis/incomplete.csv.')
    gate=[]
    for fps in [15,30,60,90]:
        p=[r for r in pairs if r['rate']==fps]
        gate.append([fps,len(p),sum(r['latency_allowance_pass'] for r in p),sum(r['maximum_latency_pass'] for r in p),
            sum(r['aligned_AP_pp']<0 for r in p),sum(r['cpu_reduction_percent']<0 for r in p)])
    page('12. Adverse outcomes and experimental variability',table(['FPS','Valid pairs','P95 allowance pass','Max <=150ms','Aligned AP losses','CPU increases'],gate)+'\n\n'
        'P95 allowance is the historical max(B+5 ms, 1.15 x B), applied descriptively per pair. Maximum latency is an empirical observation, not a guaranteed deadline. AP losses and CPU increases are retained rather than optimized away.\n\n'
        'Three repeats characterize runtime variability within each scene. Consecutive frames and repeated scenes are temporally correlated. Five scenes are a small development collection; no frame-level independent-sample significance test, universal performance claim, or unseen-scene confidence bound is made.\n\n'
        'Other applications and uncontrolled CPU scheduling can affect service spikes. Component process-CPU attribution is unavailable while capture runs concurrently. Peak RSS is the process lifetime high-water mark, not incremental buffer allocation. Source cadence, safety fallbacks and long-run optical-flow reliability remain distinct questions.')
    cases=load(ROOT/'analysis/failure_cases.json')
    casecounts=[]
    for rate,c in cases['query_event_counts'].items():
        casecounts.append([rate,c['queries'],c['queries_with_TP_loss'],c['queries_with_TP_gain'],c['queries_with_more_FP'],c['queries_with_fewer_FP']])
    example_rows=[]
    for seq in [f'sequence_{i:02d}' for i in range(5)]:
        chosen=[c for c in cases['adverse_examples'] if c['sequence']==seq and c['rate']==90]
        if chosen:
            c=sorted(chosen,key=lambda x:(x['TP_change'],-x['FP_change'],x['trial']))[0]
            example_rows.append([seq,c['trial'],c['query_frame_id'],c['B_TP'],c['C_TP'],c['B_FP'],c['C_FP'],c['C_assignment']['output_decision']])
    page('12a. Query-level adverse evidence',table(['FPS','Queries','TP loss queries','TP gain queries','More FP queries','Fewer FP queries'],casecounts)+'\n\n'+
        table(['Seq','Trial','Query','B TP','C TP','B FP','C FP','C output'],example_rows)+'\n\n'
        'B is latest-frame Dense; C is integrated Practical Fixed-2. The examples use a declared deterministic worst-TP-loss/FP-increase ranking and show adverse 90 FPS queries where available. They are diagnostic examples, not a representative sample. All cell examples and source-output assignments are saved in analysis/failure_cases.json.\n\n'
        'Counts are query/object-observation matching events, not independent objects or proven motion failures. Source selection, completion timing, confidence and box geometry can all affect these differences. No tuning or causal claim follows from them.')
    conclusions=[]
    for fps in [15,30,60,90]:
        p=[r for r in pairs if r['rate']==fps]
        if p:conclusions.append(f'At {fps} FPS, {len(p)} valid paired comparisons show mean aligned AP change {fmt(mean(p,"aligned_AP_pp"))} pp, CPU reduction {fmt(mean(p,"cpu_reduction_percent"),2)}%, and P95 reduction {fmt(mean(p,"p95_reduction_percent"),2)}%. Individual effects and reversals remain visible.')
    overload=[p for p in pairs if p['rate']==90]
    if overload:
        direction='supports the earlier overload advantage on average within this development collection' if mean(overload,'aligned_AP_pp')>0 else 'does not reproduce the earlier overload advantage on average in this expanded comparison'
        conclusions.append(f'The mean sequence-paired aligned-AP result {direction}. {sum(p["aligned_AP_pp"]<0 for p in overload)}/{len(overload)} valid pairs have an AP loss. This is descriptive replication within reused scenes, not unseen-data generalization.')
    b90=[r for r in pooled if r['mode']=='latest' and int(r['rate'])==90 and r['pacing_valid']=='True']
    c90=[r for r in pooled if r['mode']=='merged' and int(r['rate'])==90 and r['pacing_valid']=='True']
    if len(b90)==len(c90)==3:
        conclusions.append('At 90 FPS, the mean of three joint 3,416-query cohort scores gives latest Dense versus integrated AP '
            f'{fmt(mean(b90,"AP"),4,True)}% versus {fmt(mean(c90,"AP"),4,True)}%. '
            'At confidence .25 and IoU .50, the corresponding precision/recall/F1 values are '
            f'{fmt(mean(b90,"precision"),2,True)}/{fmt(mean(b90,"recall"),2,True)}/{fmt(mean(b90,"F1"),2,True)}% versus '
            f'{fmt(mean(c90,"precision"),2,True)}/{fmt(mean(c90,"recall"),2,True)}/{fmt(mean(c90,"F1"),2,True)}%. '
            'Thus the integrated fixed operating point improves despite lower AP. This does not establish an AP gain or a uniformly better precision-recall curve.')
    normal=[p for p in pairs if p['rate']==15]
    if normal:
        conclusions.append(f'At 15 FPS, the historical paired P95 allowance is exceeded in {sum(not p["latency_allowance_pass"] for p in normal)}/{len(normal)} valid pairs. The low-rate latency concern remains visible regardless of average CPU savings.')
    freshness=[]
    for cell,d in bycell.items():
        if int(cell[1])==90 and {'fifo','latest'}<=set(d):freshness.append(100*(1-float(d['latest']['latency_p95_ms'])/float(d['fifo']['latency_p95_ms'])))
    if freshness:
        conclusions.append(f'On {len(freshness)} valid matched 90 FPS FIFO/latest-Dense pairs, mean P95 reduction is {fmt(statistics.mean(freshness),2)}%. Unsafe and timing-invalid FIFO attempts are excluded from this descriptive effect and remain reported separately.')
    page('13. Conclusions supported by the expanded evidence',
        '\n\n'.join(conclusions)+'\n\n'
        'The offline finding remains: selective inference saves work with an official AP penalty. Latest-frame buffering remains the original freshness mechanism; Fixed-2 is a separate detector-work mechanism. Integrated gains, if present, concern the conditional system trade-off rather than a more accurate detector.\n\n'
        'The new evidence extends streaming coverage from one 300-frame prefix to five complete development sequences and broadens the arrival-rate comparison. It does not establish generalization to unseen scenes or another dataset, real-camera sensor-to-display performance, arbitrary-gap flow correctness, energy savings, or stability over long continuous deployment.\n\n'
        'Keep latest-frame Dense640 as the streaming reference and integrated Practical Fixed-2 as an optional mode. Interpret the rate/scene/trial evidence explicitly; do not claim universal replacement or erase historical failed criteria. No further optimization was performed.')
    page('14. Reproduction, graphs and evidence',
        'Run root: runs/branch_b_expanded_local_v2/. execution_plan.json specifies all 180 cells and the reused paths. EXECUTION_COMPLETE.json lists every outcome. Frozen configs and manifests are immutable. Protected hashes cover historical sources and artifacts.\n\n'
        'Infrastructure: analysis/branch_b_expanded_v1/continue_local.py and worker.py. ACTUAL_INVOCATION.json records the actual continuation commands and corrects the inherited older command string without changing the saved plan. Production capture, pending buffer, pipeline, flow, tracker and streaming adapter are unchanged. analyze_local.py reads saved outputs and invokes the unchanged official scorer and held-output definition only after timing completes.\n\n'
        'Tables: analysis/offline.csv, offline_per_sequence.csv, offline_supplement.json, streaming.csv, execution_matrix.csv, paired_effects.json, results.json. Score directories preserve official precision/recall/score tensors, match diagnostics, exact populations and metrics. Per-query assignments preserve held-output provenance.\n\n'
        'Figures: figures/ contains 300 DPI PNGs, editable SVGs and manifest.json with exact plotted values, source paths, row identifiers and hashes. An independent verification pass re-reads raw telemetry, source metrics, tables and figure values before delivery.\n\n'
        'The prior handoff and historical experiment reports remain in their preserved archive. This focused report supersedes their final-validation scope without discarding their bounded experimental findings.')
    page('15. Technical foundations and manuscript positioning',
        'Li, Wang and Ramanan, Towards Streaming Perception (ECCV 2020), establishes why completed outputs must be evaluated against a changing scene. This motivates the timing-aware question here, but our source-query held-output diagnostic is not their official benchmark protocol.\n\n'
        'Observation-centric tracking and low-score association are established ideas. OC-SORT motivates preserving observations, and ByteTrack motivates using weak detections for continuation. Our implementation is neither a full OC-SORT nor a full ByteTrack reproduction. Lucas-Kanade flow and SORT-style state estimation are standard components; this work does not claim to invent them.\n\n'
        'The defensible contribution is a controlled systems investigation: separate detector-work reduction from obsolete-work avoidance, integrate them without losing source-time semantics, verify prediction and state behavior, and expose conditional benefits and failures across rates and scenes. Latest-frame buffering and Fixed-2 must remain distinct contributions in the manuscript.\n\n'
        'The stronger validation scope is five complete development sequences with three trials per selected condition. The remaining publication boundary is unseen-scene and real-camera validation, not a reason to reopen algorithm optimization.\n\n'
        'Primary references below are carried forward from the previously reviewed local handoff; no external source was accessed during this resumed phase.')
    previous=(ROOT/'historical_handoff/Branch_B_Research_Handoff.md').read_text()
    references=[]
    for line in previous.splitlines():
        if line.startswith('**[L'):references.append(line)
    for start in range(0,len(references),7):
        page('Primary references retained from the prior review','\n\n'.join(references[start:start+7]))
    md='\n\n<!-- PAGE BREAK -->\n\n'.join(pages)+'\n'
    for dash in ['\u2011','\u2013','\u2014']:md=md.replace(dash,'-')
    (ROOT/'analysis/expanded_handoff.md').write_text(md)
    # Current handoff replacement is authorized; immutable copies already exist.
    (BASE/'research_handoff/Branch_B_Research_Handoff.md').write_text(md)
    evidence=['# Branch B Expanded Local Validation Evidence Index\n',
        'Scope: four offline configurations and three streaming arms only. Five local development sequences; no unseen-data or real-camera claim.\n',
        'The prior evidence index is preserved under runs/branch_b_expanded_local_v2/historical_handoff/.\n',
        table(['Claim / artifact','Primary evidence'],[
            ['Full matrix and exact policies','runs/branch_b_expanded_local_v2/execution_plan.json'],
            ['Two verified reused passes','runs/branch_b_expanded_local_v2/REUSE_VERIFICATION.json'],
            ['All completed/incomplete outcomes','runs/branch_b_expanded_local_v2/EXECUTION_COMPLETE.json'],
            ['Frozen implementation, config, manifest hashes','runs/branch_b_expanded_local_v2/frozen_sha256.json'],
            ['Historical protection baseline','runs/branch_b_expanded_local_v2/protected_sha256.json'],
            ['Offline accuracy/runtime and scope','runs/branch_b_expanded_local_v2/analysis/offline.json'],
            ['All four offline per-sequence operating points','runs/branch_b_expanded_local_v2/analysis/offline_per_sequence_fixed25.csv'],
            ['Offline confidence .10 operating points','runs/branch_b_expanded_local_v2/analysis/offline_conf10.json'],
            ['Original per-scene P/R/F1; target distributions','runs/branch_b_expanded_local_v2/analysis/offline_supplement.json'],
            ['Every stream measurement','runs/branch_b_expanded_local_v2/analysis/streaming.csv'],
            ['Joint 3,416-query cohort accuracy and validity','runs/branch_b_expanded_local_v2/analysis/pooled_aligned.csv'],
            ['Exact pooled evaluator tensors','runs/branch_b_expanded_local_v2/analysis/pooled_scores/'],
            ['Resource-aborted partial accounting','runs/branch_b_expanded_local_v2/analysis/incomplete.csv'],
            ['Common frame populations, aligned metrics','runs/branch_b_expanded_local_v2/analysis/results.json'],
            ['Raw clocks, actual predictions, errors','runs/branch_b_expanded_local_v2/passes/'],
            ['Exact scorer tensors and match diagnostics','runs/branch_b_expanded_local_v2/analysis/scores/'],
            ['Source-query held-output assignments','runs/branch_b_expanded_local_v2/analysis/*_assignments.json'],
            ['Paired effects and descriptive latency checks','runs/branch_b_expanded_local_v2/analysis/paired_effects.json'],
            ['Deterministically selected adverse queries','runs/branch_b_expanded_local_v2/analysis/failure_cases.json'],
            ['Actual continuation commands','runs/branch_b_expanded_local_v2/ACTUAL_INVOCATION.json'],
            ['Scoring source snapshots and hashes','runs/branch_b_expanded_local_v2/analysis/scoring_sources.json'],
            ['Two new evaluation-contract checks','runs/branch_b_expanded_local_v2/analysis/postprocess_stage_0.json'],
            ['Independent calculations and figure audit','runs/branch_b_expanded_local_v2/FINAL_VERIFICATION.json']])]
    for fig in figs:evidence.append(f'## {fig["id"]}: {fig["title"]}\n\nScope: {fig["scope"]}\n\nInterpretation: {fig["interpretation"]}\n\nPNG: {fig["png"]}\n\nSVG: {fig["svg"]}\n\nExact inputs and sources: runs/branch_b_expanded_local_v2/figures/manifest.json.\n')
    evidence.append('## Metric safeguards\n\nAP percentages and AP percentage-point differences are distinct. Marginal per-sequence AP means are not pooled COCO AP. Original Fixed-2 runtime is historical and not paired with E. Invalid/incomplete runs remain visible. Original 300-frame acceptance failures remain failures. No new model inference is performed by report generation.\n')
    (BASE/'research_handoff/Branch_B_Evidence_Index.md').write_text('\n\n'.join(evidence))
    dump(ROOT/'analysis/REPORT_GENERATED.json',dict(pages_declared=len(pages),complete=complete,pacing_invalid=invalid,incomplete=incomplete,
        mean_paired_effects={str(fps):{k:mean([p for p in pairs if p['rate']==fps],k) for k in ['aligned_AP_pp','cpu_reduction_percent','p95_reduction_percent','call_reduction_percent']} for fps in [15,30,60,90]}))
    print('Markdown handoff pages',len(pages))


if __name__=='__main__':main()
