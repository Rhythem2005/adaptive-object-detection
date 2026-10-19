"""Arbitrary-arm official COCO scoring, retaining row and GT identity matches."""
import time
from pathlib import Path

import numpy as np
from adaptive_pipeline.evaluation.coco import (prediction_records, make_coco, run_coco,
    coco_metrics, fixed_operating_point, official_match_arrays)
from .common import dump, json_gzip, load


def match_arrays(scorer, records):
    tp, ignored, considered = official_match_arrays(scorer, len(records))
    matched_gt = np.zeros(tp.shape, np.int32)
    gt_scores = np.full((max(scorer.cocoGt.anns)+1, len(scorer.params.iouThrs)), -1., np.float32)
    all_area = scorer.params.areaRng[0]
    for row in scorer.evalImgs:
        if row is None or row['aRng'] != all_area:
            continue
        idx = np.asarray(row['dtIds'], int)-1
        matched_gt[idx] = np.asarray(row['dtMatches'], np.int32).T
        for j in range(len(scorer.params.iouThrs)):
            for gi, did, ignore in zip(row['gtIds'], row['gtMatches'][j], row['gtIgnore']):
                if did and not ignore:
                    gt_scores[gi, j] = records[int(did)-1]['score']
    return dict(tp=tp, ignored=ignored, considered=considered, matched_gt=matched_gt, gt_scores=gt_scores)


def frame_counts(arrays, plan, predictions):
    offsets = np.r_[0, np.cumsum([len(predictions[r['key']]) for r in plan])]
    conf = np.concatenate([predictions[r['key']][:, 4] for r in plan])
    counts = {}
    for threshold in [.001, .1, .25]:
        selected = (conf >= threshold) & arrays['considered']
        counts[str(threshold)] = np.array([[int((arrays['tp'][a:b,j] & selected[a:b]).sum()),
                    int((~arrays['tp'][a:b,j] & ~arrays['ignored'][a:b,j] & selected[a:b]).sum())]
                  for a,b in zip(offsets[:-1],offsets[1:]) for j in [0,5]], dtype=np.int32).reshape(len(plan),2,2)
    return counts


def score(predictions, gtdata, plan, out, expected=None, per_sequence=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    arrays = {r['image_id']:predictions[r['key']] for r in plan}
    records, conversion = prediction_records(arrays, {i:i for i in range(8)})
    json_gzip(out/'predictions.coco.json.gz', records)
    gt, dt = make_coco(gtdata, records)
    ev, log = run_coco(gt, dt)
    metrics = coco_metrics(ev)
    if expected is not None:
        for k,v in expected.items():
            if k != 'per_class':
                np.testing.assert_allclose(metrics[k], v, atol=1e-14, rtol=0, err_msg=k)
    (out/'coco_stdout.txt').write_text(log)
    matches = match_arrays(ev, records)
    counts = frame_counts(matches, plan, predictions)
    np.savez_compressed(out/'official_matches.npz', **matches,
        n_pred=[len(predictions[r['key']]) for r in plan], image_ids=[r['image_id'] for r in plan],
        conf=np.array([r['score'] for r in records]), **{'counts_conf'+k:v for k,v in counts.items()})
    np.savez_compressed(out/'official_tensors.npz', precision=ev.eval['precision'],
        recall=ev.eval['recall'], scores=ev.eval['scores'])
    point = fixed_operating_point(ev)
    del ev, matches
    sequences = {}
    if per_sequence:
        for video in sorted({r['video'] for r in plan}):
            ids = [r['image_id'] for r in plan if r['video']==video]
            sub, sublog = run_coco(gt, dt, ids)
            sequences[video] = coco_metrics(sub)
            (out/(video+'_coco_stdout.txt')).write_text(sublog)
            del sub
    result = {'COCO': metrics, 'fixed_conf25': point, 'per_sequence':sequences,
        'conversion':conversion, 'scoring_wall_seconds':time.perf_counter()-wall,
        'scoring_process_seconds':time.process_time()-cpu, 'device':'CPU pycocotools2.0.11',
        'new_YOLO_inferences':0}
    dump(out/'metrics.json', result)
    print(f"  COCO AP {out.parent.name}: {metrics['AP']*100:.6f}%", flush=True)
    dump(out/'SCORE_COMPLETE.json', {'AP':metrics['AP'], 'expected_metrics_parity':expected is not None})
    return result
