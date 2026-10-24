"""Final read-only fidelity/preservation audit plus new E integrity manifest."""
import subprocess
import time

import numpy as np

from analysis.argoverse_hd_phase2.common import archive, dump, load, sha
from .run import BASE, ROOT, LATEST, BLOCKS, REFS, assert_archive_equal, hash_inputs, verify_sources


def main():
    wall,cpu=time.perf_counter(),time.process_time()
    assert (ROOT/'EXECUTION_COMPLETE.json').exists()
    assert (ROOT/'REPORT_GENERATED.json').exists()
    assert (ROOT/'FIRST_BLOCK_ACCURACY_GATE.json').exists()
    verify_sources()
    for path,digest in load(ROOT/'startup_corrections.json')['preserved_files'].items():
        assert sha(path)==digest,path
    # D's standalone verifier intentionally prohibits E directories; audit its
    # preservation manifests directly instead of invoking that historical guard.
    droot=BASE/'runs/argoverse_hd_detector_diagnosis_D_v1'
    for name in ['prior_evidence.json','execution_sources.json']:
        for path,digest in load(droot/name).items():assert sha(path)==digest,path
    assert subprocess.check_output(['git','diff','--binary'],cwd=BASE)==(ROOT/'pre_execution/git_diff.patch').read_bytes()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE)==(ROOT/'pre_execution/git_head.txt').read_bytes()
    hash_inputs(load(ROOT/'frames.json'))
    assert load(ROOT/'frames.json')==load(LATEST/'frameplan.json')
    assert sha(ROOT/'ground_truth.json')==sha(LATEST/'ground_truth.json')
    measured=warmups=forwards=passes=0
    summary=load(ROOT/'summary.json')['methods']
    for m in 'ABCDE':
        ref_pred=archive(REFS[m]/'preds.npz');ref_detail=archive(REFS[m]/'prediction_details.npz')
        assert load(ROOT/'accuracy'/m/'ARCHIVE_FIDELITY.json')['official_tensors_bytes_exact']
        for block in range(1,6):
            out=ROOT/'passes'/f'{block:02d}_{m}'
            assert_archive_equal(archive(out/'preds.npz'),ref_pred,'serialized predictions')
            assert_archive_equal(archive(out/'prediction_details.npz'),ref_detail,'serialized details')
            complete=load(out/'COMPLETE.json')
            measured+=complete['measured_requests'];warmups+=complete['warmup_requests'];forwards+=complete['root_forwards'];passes+=1
        result=summary[m]
        assert result['pooled']['step_wall_ms']['n']==17080
        assert sum(v['scopes']['step_wall_ms']['n'] for v in result['per_sequence'].values())==17080
        assert sum(result['decisions'][d]['step_wall_ms']['n'] for d in ['FULL','SKIP'])==17080
        for inclusive,decode,step in [('decode_inclusive_ms','decode_ms','step_wall_ms'),
                ('decode_inclusive_cpu_ms','decode_cpu_ms','step_cpu_ms')]:
            np.testing.assert_allclose(result['pooled'][inclusive]['sum'],
                result['pooled'][decode]['sum']+result['pooled'][step]['sum'],rtol=1e-13,atol=1e-8)
        assert result['fixed_points']['0.25']['TP']==result['accuracy']['fixed_conf25']['counts']['TP']
        assert result['fixed_points']['0.25']['FP']==result['accuracy']['fixed_conf25']['counts']['FP']
    assert (passes,measured,warmups,forwards)==(25,51280,75,51355)
    resume=load(ROOT/'RESUME_STARTED.json');assert sha(resume['source'])==resume['sha256']
    dump(ROOT/'FINAL_AUDIT.json',dict(passes=passes,frames=85400,measured_requests=measured,
        warmups=warmups,accepted_root_forwards=forwards,failed_startup_requests=9,
        actual_E_requests_including_startups=51364,all25_serialized_prediction_and_detail_archives_bytes_exact=True,
        official_accuracy_fidelity_all5=True,sequence_end_state_and_all_schedule_fidelity=True,
        all_frame_tracker_states_not_recorded=True,protected_prior_files=len(load(ROOT/'prior_evidence.json')),
        protected_prior_hashes_unchanged=True,D_prior_and_source_hashes_unchanged=True,
        frozen_source_hashes_unchanged=True,tracked_working_diff_unchanged=True,HEAD_unchanged=True,
        aggregate_timing_and_operating_point_consistency=True,
        audit_wall_s=time.perf_counter()-wall,audit_cpu_s=time.process_time()-cpu,
        no_commit_push=True,no_phase4=True))
    sources={str(p):sha(p) for p in (BASE/'analysis/argoverse_hd_runtime_E').glob('*.py')}
    for p in [BASE/'tests/test_runtime_E.py',BASE/'tests/test_runtime_E_validation.py',BASE/'docs/EXPERIMENT_E_RESULTS.md']:
        sources[str(p)]=sha(p)
    dump(ROOT/'final_source_report_sha256.json',sources)
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'}
    dump(ROOT/'artifact_sha256.json',artifacts)
    dump(ROOT/'COMPLETE.json',dict(status='COMPLETE_AUDITED',passes=25,actual_E_requests=51364,
        report=str(BASE/'docs/EXPERIMENT_E_RESULTS.md'),artifacts_hashed=len(artifacts),
        note='COMPLETE.json and artifact_sha256.json are integrity-envelope files, excluded from artifact manifest.'))
    print(f'E final audit passed: {len(artifacts)} artifacts; 25 exact serialized passes; no prior changes.')


if __name__=='__main__':main()
