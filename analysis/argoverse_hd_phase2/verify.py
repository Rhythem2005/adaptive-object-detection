"""Final read-only result/provenance audit; no replay, scoring or inference."""
import argparse
import ast
from collections import Counter
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from .common import archive, dump, load, sha


def equal_archives(left, right, columns=None):
    a, b = archive(left), archive(right)
    assert a.keys() == b.keys(), (left, right)
    for key in a:
        aa, bb = (a[key], b[key]) if columns is None else (a[key][:, columns], b[key][:, columns])
        np.testing.assert_array_equal(aa, bb, err_msg=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    base = out.parents[1]
    if (out/'COMPLETE.json').exists():
        raise FileExistsError('Final audit already complete')
    frozen = base/'analysis/argoverse_hd_phase2_plan_v1'
    plan = load(frozen/'plan.json')
    assert out == Path(plan['fresh_execution_root'])
    for stage in ['R', 'A', 'B', 'C', 'ANALYSIS']:
        assert (out/(stage+'_COMPLETE.json')).exists(), stage
    source_hashes = load(frozen/'source_sha256.json')
    for path, digest in source_hashes.items():
        assert sha(path) == digest, path
    assert load(out/'execution_plan.json')['approved_plan'] == plan

    # Compare both the original benchmark's Python source subset and the full
    # production snapshot made immediately before R. Existing changes survive.
    snapshot_counts = {}
    for label, source in [
        ('benchmark', Path(plan['source_benchmark'])/'source_snapshot/adaptive_pipeline'),
        ('pre_R', out/'source_snapshot/adaptive_pipeline'),
    ]:
        count = 0
        for path in source.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if label == 'benchmark' and path.suffix != '.py':
                continue
            current = base/'extracted/adaptive_pipeline'/path.relative_to(source)
            assert current.is_file() and sha(current) == sha(path), current
            count += 1
        snapshot_counts[label] = count

    benchmark = Path(plan['source_benchmark'])
    phase1 = Path(plan['verified_scoring'])
    frames = load(out/'frameplan.json')
    assert len(frames) == 3416
    sequence_counts = Counter(r['video'] for r in frames)
    assert len(sequence_counts) == 5
    for seq, count in sequence_counts.items():
        assert [r['frame_index'] for r in frames if r['video'] == seq] == list(range(count))

    dense = archive(benchmark/'baseline/preds.npz')
    methods = ['fixed2_093', 'C_fullfallback', 'adaptive_observed', 'adaptive_quality']
    for method in methods:
        directory = out/'R_reference_parity'/method
        for filename in ['preds.npz', 'prediction_details.npz']:
            equal_archives(directory/filename, benchmark/method/filename)
        old = load(phase1/method/'metrics.json')
        new = load(directory/'scoring/metrics.json')
        assert new['COCO'] == old['official_COCO_max100'], method
        assert new['per_sequence'] == {seq: value['metrics'] for seq, value in old['per_sequence'].items()}, method
        with np.load(phase1/method/'official_tensors100.npz') as a, np.load(directory/'scoring/official_tensors.npz') as b:
            for key in ['precision', 'recall', 'scores']:
                np.testing.assert_array_equal(a[key], b[key], err_msg=method+'/'+key)

    b_arms = plan['requested_initial_batch']['B_tracking_factors']['arms']
    c_cells = plan['requested_initial_batch']['C_schedule_factors']['cells']
    expected = {out/'R_reference_parity'/m for m in methods}
    expected.update(out/'B_tracking_factors'/n for n in b_arms if n != 'T0_fixed2_original')
    expected.update(out/'C_schedule_factors'/cell['name'] for cell in c_cells)
    actual = {p.parent for p in out.glob('*/*/REPLAY_COMPLETE.json')}
    assert actual == expected and len(actual) == 21
    steps = calls = 0
    for directory in sorted(actual):
        complete = load(directory/'REPLAY_COMPLETE.json')
        assert sha(directory/'preds.npz') == complete['predictions_sha256']
        assert complete['new_YOLO_inferences'] == 0
        assert (directory/'scoring/SCORE_COMPLETE.json').exists()
        with (directory/'frames.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(frames)
        predictions = archive(directory/'preds.npz')
        assert predictions.keys() == dense.keys()
        for frame, row in zip(frames, rows):
            assert row['video'] == frame['video'] and int(row['image_id']) == frame['image_id']
            assert row['decision'] in {'FULL', 'SKIP'}
            assert int(row['n_crops']) == 0 and float(row['t_detect_ms']) == 0
            if row['decision'] == 'FULL':
                np.testing.assert_array_equal(predictions[frame['key']], dense[frame['key']])
                calls += 1
        steps += len(rows)
        if directory.parent.name == 'B_tracking_factors':
            approved = load(b_arms[directory.name]['full_config'])
            assert load(directory/'config.json') == approved
            assert sum(row['decision'] == 'FULL' for row in rows) == 1710
        if directory.parent.name == 'R_reference_parity':
            assert load(directory/'config.json') == load(frozen/f'configs/reproduce_{directory.name}.json')
        if directory.parent.name == 'C_schedule_factors':
            cell = next(c for c in c_cells if c['name'] == directory.name)
            assert load(directory/'config.json') == load(cell['engine_config'])
            schedule = load(cell['schedule'])['full_frame_indices']
            for seq in sequence_counts:
                indices = [int(row['frame']) for row in rows if row['video'] == seq and row['decision'] == 'FULL']
                assert indices == schedule[seq], (directory.name, seq)
    assert steps == 71736 and calls == 36492
    for method, control in [('adaptive_observed', 'S2_anchor_archived_anchor'), ('adaptive_quality', 'S7_quality_archived_quality')]:
        for filename in ['preds.npz', 'prediction_details.npz']:
            equal_archives(out/'C_schedule_factors'/control/filename, benchmark/method/filename)
    t1 = out/'B_tracking_factors/T1_fixed2_anchor'
    t5 = out/'B_tracking_factors/T5_no_score_decay'
    assert load(t1/'state_hashes.json') == load(t5/'state_hashes.json')
    equal_archives(t1/'preds.npz', t5/'preds.npz', [0, 1, 2, 3, 5])

    # Synthetic validation only. No real-data experiments are invoked here.
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q']
    env = dict(os.environ, PYTHONPATH='extracted:.')
    tests = subprocess.run(command, cwd=base, env=env, capture_output=True, text=True)
    log = tests.stdout + tests.stderr
    (out/'synthetic_tests.txt').write_text(log)
    assert tests.returncode == 0, log
    count = int(re.search(r'Ran (\d+) tests', log).group(1))
    assert count == 73, count
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    assert (out/'report.md').read_bytes() == (base/'docs/PHASE2_RESULTS.md').read_bytes()

    snapshot = out/'final_source_snapshot'
    snapshot.mkdir()
    shutil.copytree(Path(__file__).parent, snapshot/'phase2_harness', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(base/'tests/test_phase2_harness.py', snapshot/'test_phase2_harness.py')
    for command, filename in [(['git', 'status', '--short'], 'git_status.txt'), (['git', 'diff', '--binary'], 'working_tree.diff')]:
        result = subprocess.run(command, cwd=base, capture_output=True, check=True)
        (out/filename).write_bytes(result.stdout)
    audit = {
        'source_evidence_hashes_unchanged': len(source_hashes),
        'production_snapshot_files_identical': snapshot_counts,
        'R_all_four_bitwise_predictions_details_and_official_tensors': True,
        'R_all_four_aggregate_per_class_per_sequence_metrics_exact': True,
        'C_natural_reference_prediction_detail_parity': True,
        'C_all_eleven_exact_per_sequence_forced_schedules': True,
        'all_FULL_outputs_equal_bound_Dense_cache': True,
        'score_only_state_geometry_class_parity': True,
        'approved_replays': len(actual), 'processed_frame_steps': steps,
        'simulated_FULL_cache_lookups': calls, 'new_YOLO_inferences': 0,
        'sequence_frames': dict(sequence_counts), 'synthetic_tests_passed': count,
        'device': load(out/'runtime.json'), 'D_E_executed': False,
        'Phase3_implemented': False, 'committed_or_pushed': False,
        'scope': 'Final archive/source audit and synthetic tests; no replay/scoring rerun',
    }
    dump(out/'final_verification.json', audit)
    dump(out/'COMPLETE.json', {'stages': ['R', 'A', 'B', 'C'], 'analysis': True, 'final_audit': True,
        'completed_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'new_YOLO_inferences': 0})
    hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out/'artifact_sha256.json', hashes)
    print(f'Final audit passed: {len(actual)} replays, {steps:,} steps, {count} tests; {len(hashes)} artifact hashes.', flush=True)


if __name__ == '__main__':
    main()
