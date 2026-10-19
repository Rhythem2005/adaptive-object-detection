"""Independent serialized-export audit and direct COCO API parity check.

Uses only existing annotations/predictions and the fresh scoring output.
Does not import the integration helpers, models, tracker, or pipeline.
"""
import argparse
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoring', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--benchmark', required=True)
    args = parser.parse_args()
    out, data, benchmark = map(Path, [args.scoring, args.dataset, args.benchmark])
    complete = json.loads((out/'COMPLETE.json').read_text())
    assert complete['status'] == 'complete'
    evidence = json.loads((out/'evidence.json').read_text())
    for filename, digest in evidence['source_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
    original = json.loads((data/'annotations.json').read_text())
    with gzip.open(out/'annotations.coco.json.gz', 'rt') as f:
        normalized = json.load(f)
    assert normalized['images'] == original['images']
    assert normalized['categories'] == original['categories']
    assert len(normalized['annotations']) == len(original['annotations'])
    for i, (old, new) in enumerate(zip(original['annotations'], normalized['annotations']), 1):
        expected = dict(old, id=i, source_annotation_id=old['id'], iscrowd=int(bool(old.get('iscrowd', 0))))
        assert new == expected
    images = sorted(original['images'], key=lambda im: im['id'])
    categories = {row['name']: row['id'] for row in original['categories']}
    counts = {}
    for method in complete['methods']:
        previous = json.loads((benchmark/method/'metrics.json').read_text())
        names = previous['dataset']['names']
        with gzip.open(out/method/'predictions.coco.json.gz', 'rt') as f:
            records = json.load(f)
        index = 0
        with np.load(benchmark/method/'preds.npz') as archive:
            for im in images:
                rows = archive[f"sequence_{im['sid']:02d}/{im['fid']}"]
                for row in rows:
                    rec = records[index]
                    x1, y1, x2, y2, score, cls = map(float, row)
                    assert rec == {'image_id': im['id'], 'category_id': categories[names[int(cls)]],
                                   'bbox': [x1, y1, x2-x1, y2-y1], 'score': score}
                    np.testing.assert_array_equal([rec['bbox'][0], rec['bbox'][1],
                        rec['bbox'][0]+rec['bbox'][2], rec['bbox'][1]+rec['bbox'][3]], row[:4])
                    index += 1
        assert index == len(records)
        counts[method] = index
    # Bypass the new adapters and summary helpers, reloading the serialized Dense
    # export directly into COCO. Compare complete precision/recall/score tensors.
    with gzip.open(out/'baseline'/'predictions.coco.json.gz', 'rt') as f:
        records = json.load(f)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        gt = COCO()
        gt.dataset = normalized
        gt.createIndex()
        dt = gt.loadRes(records)
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    with np.load(out/'baseline'/'official_tensors100.npz') as tensors:
        for key in ['precision', 'recall', 'scores']:
            np.testing.assert_array_equal(ev.eval[key], tensors[key])
    result = {'source_hashes_unchanged': True, 'annotation_copy_exact_except_documented_ID_normalization': True,
              'all_prediction_exports_lossless_rowwise': True, 'prediction_counts': counts,
              'direct_API_baseline_export_tensor_parity': True, 'direct_API_stats': ev.stats.tolist()}
    with (out/'independent_export_verification.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    (out/'independent_export_verification_stdout.txt').write_text(log.getvalue())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
