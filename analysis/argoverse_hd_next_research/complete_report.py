"""Supplement the fresh report with interpretation and historical controls."""
import shutil

from analysis.argoverse_hd_phase2.common import dump, load
from .run import BASE, ROOT, verify
from .summarize import points_from_arrays


def main():
    verify()
    if (ROOT/'REPORT_COMPLETE.json').exists():raise FileExistsError('Report complete')
    destination=BASE/'docs/NEXT_RESEARCH_RESULTS.md'
    if destination.exists():raise FileExistsError(destination)
    x=load(ROOT/'summary.json');timing=load(ROOT/'timing_summary.json')
    historical={}
    for name in ['baseline','fixed2_093','C_fullfallback','adaptive_observed','adaptive_quality']:
        path=BASE/'runs/argoverse_hd_coco_phase1_v1'/name;m=load(path/'metrics.json')
        historical[name]={'COCO':m['official_COCO_max100'],'fixed_conf25':m['supplementary_fixed_conf25_IoU50'],
            'operating_points':points_from_arrays(path/'official_matches100.npz'),
            'detector_requests':m['source_efficiency_unchanged_not_remeasured']['full_detector_calls'],
            'per_sequence':m['per_sequence'],'source':str(path),
            'note':'Previously scored output reused, not a new replay; historical inference timing not combined with current cache timing.'}
    dump(ROOT/'historical_comparison.json',historical)
    cfg=load(ROOT/'arms/F_fastpath/config.json');dump(ROOT/'candidate_config.json',cfg)
    lead=['# Evidence-driven optimization: findings and recommendation','',
        '**Recommended candidate: anchored Fixed-2, low-score continuation, 3x3 flow and the guarded fixed-signal fastpath.** It preserves the prior 14.692246% COCO AP exactly while reducing measured cache-step overhead by 55.29% and CPU time by 36.77%. Detector requests remain 1,710 (49.94% fewer than Dense). This stage establishes a processing-efficiency improvement, not a new accuracy gain or an end-to-end acceleration.','',
        'The detector remains stock YOLOv8n. The only production changes are an opt-in fixed_signal_fastpath flag, validation guards and omission of unused thumbnail/luminance/illumination updates for supported fixed configurations. The flag defaults off. Tracking, flow, detector settings and evaluator semantics are unchanged. Tentative-birth and output-filter prototypes stay isolated in analysis and were not promoted or run on the dataset after their diagnostic gates failed.','',
        'The controlled timing comparison is against the identical recovery+3x3 configuration. The unused 160-pixel INTER_AREA thumbnail resize costs 1.392818 ms/frame in this measurement, compared with 0.056593 ms for thumbnail luminance. Required flow preparation remains enabled. The main removed cost is resizing for unused signals, not a replacement flow algorithm or GPU transfer. Timer wrappers are common to both arms and included in the measured scope.','',
        '## What improved, and what did not','',
        '- Fastpath: every output, prediction detail, tracker-state hash and official COCO tensor matches its control; there is no additional frame-buffering delay. Mean cached step 2.759414 -> 1.233757 ms; P95 3.514942 -> 1.970289 ms; CPU 4.546395 -> 2.874584 ms. All three paired wall-time changes are between -55.43% and -55.11%. These repeated measurements remain specific to this hardware, OpenCV build and cache-processing scope.',
        '- Stricter low-score association plus 3x3: AP 14.682026%, a loss of 0.010221 percentage points versus recovery+3x3. At confidence0.10/IoU50, FP falls 15,239 -> 14,755 (-3.18%) while TP falls 19,561 -> 19,420 (-141). Precision rises 56.2098% -> 56.8252%; recall falls 35.3010% -> 35.0466%. Counts at confidence0.25 are identical. Mean tracks fall 8.85246 -> 8.44965 and flow points 135,855 -> 129,663. This is a precision/recall/work-proxy trade-off; its latency was not separately priced.',
        '- The alternate Fixed-2 phase scores 14.616755% AP versus 14.692246%, with identical request counts. It increases TP25 but also FP25. This does not establish a new online scheduler; its budget uses known sequence lengths.',
        '- Simple two-observation confirmation fails the predeclared precision-gain screen. Birth-source precision improves only 3.30 percentage points while rejecting 41.52% of source true detections; FULL-output precision improves 2.29 points while rejecting 23.46%. Consequently five conditional arms, including Dense postprocessing, are skipped. No AP outcome is claimed for these unrun variants, and the screen does not rule out all temporal confirmation approaches.','',
        '## Why persistence was insufficient','',
        'Of 2,954 retained weak birth-source FP50 events, descriptive GT-overlap checks identify 1,500 with maximum regular-GT IoU in [0.10,0.50), 655 with same-class overlap>=0.50 (duplicate/matching-conflict candidates), 212 with only other-class overlap>=0.50, and 587 with lower overlap. Cars account for 2,246 retained FP events. These are errors in the FULL detector source observations, before propagation. Persistence cannot by itself establish correct class, uniqueness or box extent.','',
        'These categories are overlap diagnostics, not verified physical root-cause labels: coarse overlap does not prove a particular object identity, and low overlap does not prove background or annotation error. Counts are observation events, not unique objects. They nevertheless challenge an explanation centred only on unstable background lights or optical-flow drift. Any later candidate-quality work should explicitly distinguish localisation, duplication and classification from object absence.','',
        '## Historical official controls','',
        '| Previously scored method | AP % | Requests | P25 micro % | R25 micro % | FP25 |',
        '|---|---:|---:|---:|---:|---:|']
    for n,a in historical.items():
        p=a['operating_points']['0.25']
        lead.append(f"| {n} | {100*a['COCO']['AP']:.6f} | {a['detector_requests']} | {100*p['micro_precision']:.4f} | {100*p['micro_recall']:.4f} | {p['FP']} |")
    lead+=['',
        'Phase2/3 anchoring, recovery, unrestricted weak births, stricter association, Fixed-2 phase and Fixed-3 references are additionally retained in summary.json/summary.csv. The high-AP weak-birth control reproduces 15.091065% AP, but has 20,680 all-frame FP10 versus 15,239 for recovery+3x3 and 428,144 versus 135,855 flow points. Historical flow-point differences are work proxies; no unmeasured latency is assigned to that arm.','',
        '## Interpretation and next decision','',
        'Use the exported candidate_config.json for the strongest measured practical efficiency configuration. If fewer low-score false positives are worth the explicit recall/AP loss, retain V_strict_grid3 as a separate precision-oriented option. No deployment miss/false-alarm cost or accepted noninferiority margin was supplied, so this report does not silently select that trade-off.','',
        'The selected candidate remains 1.040467 AP percentage points below Dense. This stage supplies no new evidence that it can surpass Dense at half the requests. The most defensible next research question is reliable box/candidate quality, with explicit duplicate/localisation/classification diagnostics and cost limits, rather than stronger persistence thresholds or a more elaborate scheduler. A harder NMS threshold, appearance model, confidence promotion or new propagation algorithm is not justified merely by these overlaps and has not been implemented.','',
        '[ByteTrack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) supports lower-score continuation conditioned on established tracks; its tracking results do not validate weak births here. [OC-SORT](https://arxiv.org/abs/2203.14360) motivates preserving observations but does not establish a need for more tracker complexity. [Towards Streaming Perception](https://arxiv.org/abs/2005.10420) explains why cached processing cost and offline COCO AP cannot establish deployed streaming accuracy.','',
        'All five sequences are development evidence already reviewed in earlier phases. Results are descriptive; independent generalization, real-inference throughput, energy, and any resolution/raw-detector hypothesis remain unresolved. No new inference or D/E work was performed. Phase4 is not begun. Further algorithm experiments require separate approval.','',
        '## Detailed measurements and reproducibility','']
    draft=(ROOT/'report.md').read_text();shutil.copy2(ROOT/'report.md',ROOT/'report_before_conclusions.md')
    text='\n'.join(lead)+'\n'+draft
    text+='\nFinal reporting/audit commands:\n\n```text\nPYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.complete_report\nPYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.verify\n```\n\nThe first successful corrected gate used the diagnostic module directly, as recorded in protocol_amendment.json; the canonical `--stage gate` command now delegates to the same corrected implementation. Original manifests/source and failed development sources are retained. Final verification and execution_costs.json record synthetic checks, immutable prior hashes, completed pass counts and stage costs.\n'
    (ROOT/'report.md').write_text(text)
    destination.write_text(text)
    dump(ROOT/'REPORT_COMPLETE.json',{'report':str(destination),'candidate_config':str(ROOT/'candidate_config.json')})
    print(destination)


if __name__=='__main__':main()
