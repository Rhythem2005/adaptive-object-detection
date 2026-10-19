"""COCO bbox scoring of saved detections. Does not import a detector or pipeline.

The official maintained pycocotools implementation performs matching and AP/AR.
Helpers below validate conversion, expose its tensors, and audit legacy matches.
"""
import contextlib
import copy
import io
import json
from collections import Counter

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def unique_records(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if type(value) is not int or value < 0 or value in result:
            raise ValueError(f"Invalid/duplicate {key}: {value}")
        result[value] = row
    return result


def prepare_ground_truth(source):
    """Preserve all image/category IDs and boxes; use positive annotation IDs.

    COCO's dtMatches uses zero as 'unmatched', so GT annotation ID 0 is unsafe.
    General ignore-only labels need an explicit policy: pycocotools bbox _prepare
    overwrites ignore with iscrowd. The actual export has no ignore-only labels.
    """
    data = copy.deepcopy(source)
    images = unique_records(data['images'], 'id')
    categories = unique_records(data['categories'], 'id')
    unique_records(data['annotations'], 'id')
    if not images or not categories:
        raise ValueError('Nonempty image and category lists are required')
    for im in images.values():
        if any(type(im[k]) is not int or im[k] <= 0 for k in ('width', 'height')):
            raise ValueError(f"Invalid image dimensions: {im['id']}")
    id_map, outside, crowds = [], [], 0
    for new_id, ann in enumerate(data['annotations'], 1):
        if ann['image_id'] not in images or ann['category_id'] not in categories:
            raise ValueError(f"Orphan image/category: annotation {ann['id']}")
        box = np.asarray(ann['bbox'], dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
            raise ValueError(f"Invalid annotation bbox: {ann['id']}")
        if not np.isfinite(ann['area']) or not np.isclose(ann['area'], box[2]*box[3], rtol=1e-8):
            raise ValueError(f"Inconsistent annotation area: {ann['id']}")
        for flag in ('iscrowd', 'ignore'):
            if ann.get(flag, False) not in (False, True, 0, 1):
                raise ValueError(f"Invalid {flag}: {ann['id']}")
        if ann.get('ignore', False) and not ann.get('iscrowd', False):
            raise ValueError('Ignore-only annotations need an explicit policy before COCO scoring')
        old_id = ann['id']
        id_map.append({'source_id': old_id, 'coco_id': new_id})
        ann['source_annotation_id'] = old_id
        ann['id'] = new_id
        ann['iscrowd'] = int(bool(ann.get('iscrowd', False)))
        crowds += ann['iscrowd']
        im = images[ann['image_id']]
        x, y, w, h = box
        if x < 0 or y < 0 or x+w > im['width'] or y+h > im['height']:
            outside.append(old_id)
    data.setdefault('info', {})
    return data, {'annotation_id_map': id_map, 'crowds': crowds,
                  'regular_targets': len(data['annotations'])-crowds,
                  'out_of_bounds_annotation_ids_retained': outside,
                  'source_zero_annotation_ids': sum(r['source_id'] == 0 for r in id_map)}


def prediction_records(frame_arrays, category_map):
    """Map image_id -> Nx6 original-coordinate xyxy,conf,class to COCO records.

    Input row/frame insertion order is retained for deterministic score ties.
    Zero-area predictions are retained as emitted; negative extents fail.
    """
    records, degenerate = [], []
    for image_id, arr in frame_arrays.items():
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 6 or not np.isfinite(arr).all():
            raise ValueError(f'Invalid predictions for image {image_id}')
        extents = arr[:, 2:4].astype(float)-arr[:, :2].astype(float)
        if (extents < 0).any() or ((arr[:, 4] < 0) | (arr[:, 4] > 1)).any():
            raise ValueError(f'Negative extent/invalid confidence for image {image_id}')
        if (arr[:, 5] != arr[:, 5].astype(int)).any():
            raise ValueError(f'Noninteger prediction class for image {image_id}')
        for index, row in enumerate(arr):
            cls = int(row[5])
            if cls not in category_map:
                raise ValueError(f'Unknown prediction class: {cls}')
            x1,y1,x2,y2,score,_ = map(float,row)
            records.append({'image_id': int(image_id), 'category_id': category_map[cls],
                            'bbox': [x1,y1,x2-x1,y2-y1], 'score': score})
            if x1 == x2 or y1 == y2:
                degenerate.append({'image_id': int(image_id), 'row': index,
                                   'flat_index': len(records)-1})
    return records, {'predictions': len(records), 'zero_area_predictions_retained': degenerate}


def make_coco(ground_truth, records):
    gt = COCO()
    gt.dataset = copy.deepcopy(ground_truth)
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        if records:
            dt = gt.loadRes(copy.deepcopy(records))
        else:
            # Official loadRes dereferences anns[0]; an empty result is valid.
            dt = COCO()
            dt.dataset = {'images': copy.deepcopy(gt.dataset['images']),
                          'categories': copy.deepcopy(gt.dataset['categories']), 'annotations': []}
            dt.createIndex()
    return gt, dt


def run_coco(gt, dt, image_ids=None, max_dets=(1,10,100)):
    scorer = COCOeval(gt, dt, 'bbox')
    scorer.params.imgIds = sorted(gt.getImgIds() if image_ids is None else image_ids)
    scorer.params.catIds = sorted(gt.getCatIds())
    scorer.params.maxDets = list(max_dets)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        scorer.evaluate()
        scorer.accumulate()
        if tuple(max_dets) == (1,10,100):
            scorer.summarize()
    return scorer, log.getvalue()


def mean_valid(a):
    a = np.asarray(a)
    valid = a[a > -1]
    return float(valid.mean()) if len(valid) else None


def coco_metrics(scorer, max_det=100):
    """Read official tensors, including arbitrary caps without summarize's 100 default."""
    p = scorer.params
    mi = p.maxDets.index(max_det)
    prec, rec = scorer.eval['precision'], scorer.eval['recall']
    out = {}
    for label, area in [('','all'),('_small','small'),('_medium','medium'),('_large','large')]:
        ai = p.areaRngLbl.index(area)
        out['AP'+label] = mean_valid(prec[:,:,:,ai,mi])
        out['AR'+label] = mean_valid(rec[:,:,ai,mi])
    ai = p.areaRngLbl.index('all')
    for value in (.5,.75):
        ti = int(np.flatnonzero(np.isclose(p.iouThrs,value))[0])
        out[f'AP{int(value*100)}'] = mean_valid(prec[ti,:,:,ai,mi])
    for limit in p.maxDets:
        out[f'AR{limit}'] = mean_valid(rec[:,:,ai,p.maxDets.index(limit)])
    names = {c['id']:c['name'] for c in scorer.cocoGt.dataset['categories']}
    out['per_class'] = {names[c]: {'AP':mean_valid(prec[:,:,ki,ai,mi]),
                                   'AP50':mean_valid(prec[0,:,ki,ai,mi]),
                                   'AR':mean_valid(rec[:,ki,ai,mi])} for ki,c in enumerate(p.catIds)}
    if tuple(p.maxDets) == (1,10,100):
        expected = [out[k] for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']]
        np.testing.assert_allclose([v if v is not None else -1 for v in expected],scorer.stats,atol=1e-14,rtol=0)
    return out


def official_match_arrays(scorer, n_predictions):
    """Return original-row-aligned official TP/ignore flags for all-area scoring."""
    shape = (n_predictions,len(scorer.params.iouThrs))
    tp, ignored, considered = np.zeros(shape,bool),np.zeros(shape,bool),np.zeros(n_predictions,bool)
    all_area = scorer.params.areaRng[scorer.params.areaRngLbl.index('all')]
    for r in scorer.evalImgs:
        if r is None or r['aRng'] != all_area:
            continue
        idx = np.asarray(r['dtIds'],int)-1
        considered[idx] = True
        ignored[idx] = r['dtIgnore'].T
        tp[idx] = (r['dtMatches'].T > 0) & ~ignored[idx]
    return tp, ignored, considered


def cap_mask(conf, pcls, n_pred, limit):
    mask = np.zeros(len(conf),bool)
    offset = 0
    for n in n_pred:
        for cls in np.unique(pcls[offset:offset+n]):
            idx = offset+np.flatnonzero(pcls[offset:offset+n] == cls)
            chosen = idx[np.argsort(-conf[idx],kind='stable')[:limit]]
            mask[chosen] = True
        offset += n
    return mask


def sampled_ap(tp, conf, pcls, tcls, ignored, *, stable=True, keep=None):
    """COCO 101-recall accumulation holding supplied match flags fixed.

    This is diagnostic, NOT an alternate official matcher. It is verified against
    official all-area precision tensors when supplied official matching flags.
    """
    keep = np.ones(len(conf),bool) if keep is None else keep
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-conf[idx],kind='stable' if stable else 'quicksort')]
    thresholds = np.linspace(0,1,101)
    ap = []
    for cls in np.unique(tcls):
        ii = idx[pcls[idx] == cls]
        nt = int((tcls == cls).sum())
        values = []
        for j in range(tp.shape[1]):
            t = (tp[ii,j] & ~ignored[ii,j]).cumsum()
            f = (~tp[ii,j] & ~ignored[ii,j]).cumsum()
            recall = t/nt
            precision = t/np.maximum(t+f,np.spacing(1))
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            pos = np.searchsorted(recall,thresholds,side='left')
            q = np.zeros(101)
            valid = pos < len(envelope)
            q[valid] = envelope[pos[valid]]
            values.append(float(q.mean()))
        ap.append(values)
    ap = np.asarray(ap)
    return {'AP':float(ap.mean()),'AP50':float(ap[:,0].mean()),'per_class_AP_matrix':ap.tolist()}


def fixed_operating_point(scorer, confidence=.25):
    """Supplementary class-macro/micro P/R/F1 at IoU .50, never called COCO AR."""
    counts = {c:Counter() for c in scorer.params.catIds}
    all_area = scorer.params.areaRng[scorer.params.areaRngLbl.index('all')]
    for r in scorer.evalImgs:
        if r is None or r['aRng'] != all_area:
            continue
        c = counts[r['category_id']]
        c['targets'] += int((~r['gtIgnore'].astype(bool)).sum())
        selected = np.asarray(r['dtScores']) >= confidence
        c['TP'] += int(((r['dtMatches'][0] > 0) & ~r['dtIgnore'][0] & selected).sum())
        c['FP'] += int(((r['dtMatches'][0] == 0) & ~r['dtIgnore'][0] & selected).sum())
    def pr(c):
        precision = c['TP']/max(c['TP']+c['FP'],1)
        recall = c['TP']/max(c['targets'],1)
        return [precision,recall,2*precision*recall/max(precision+recall,1e-16)]
    supported = [pr(c) for c in counts.values() if c['targets']]
    macro = np.mean(supported,axis=0).tolist() if supported else [None]*3
    total = sum(counts.values(),Counter())
    return {'confidence':confidence,'IoU':.5,'macro_precision':macro[0],'macro_recall':macro[1],'macro_F1':macro[2],
            'micro_precision':pr(total)[0],'micro_recall':pr(total)[1],'micro_F1':pr(total)[2],
            'counts':dict(total),'per_category_counts':{str(k):dict(v) for k,v in counts.items()}}
