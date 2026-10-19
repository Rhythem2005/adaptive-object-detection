# Adaptive pipeline improvements

Implementation is ready for controlled evaluation. **No dataset experiment has
been run for these changes. No accuracy, FPS or computation gain is claimed.**
The user requires explicit approval before dataset or real-video execution.

## What changed

The implementation keeps stock YOLOv8n, the FULL/LOCAL/SKIP actions, video-disjoint
evaluation and the existing AP protocol. The original presets remain controls.
New settings are opt-in; their thresholds and priors are unvalidated.

1. **Observation anchoring.** Fresh FULL/LOCAL detector boxes are preserved for
   output and subsequent propagation. A separate Kalman state estimates motion.
   Velocity is updated from successive detector observations in frame units.
2. **Quality-aware propagation.** Flow exposes surviving-point fraction,
   forward-backward consistency, spatial spread and displacement coherence.
   Weak measurements have greater covariance; opt-in filtering uses Joseph-form
   covariance updates. A corner-based point-selection variant is separate.
3. **Risk-aware scheduling.** Geometry uncertainty, flow quality, motion
   disagreement and detection age drive refreshes. Low detector confidence
   alone no longer forces a refresh in the new mode. Hard deadlines, unreliable
   scene alignment and motion failures override the soft detector budget.
4. **Cost-aware LOCAL.** Crop count, modeled cost, observed inference times and
   an online confirmation estimate govern admission. Initial presets allow one
   crop. Failed confirmation still escalates to FULL on the same frame, with
   both costs charged. Dynamic crop margins and bounded new-track admission are
   independently configurable.
5. **Confidence separation.** An optional reliability-dependent score age
   reduces uniform confidence penalties. Low-score association can recover
   existing tracks during detector passes; it never creates low-score births.
6. **Optional verification.** Background alignment corrects novelty measurement,
   not box motion a second time. A bounded normalized patch comparison can flag
   changed appearances. These variants require separate runtime/quality ablations.
7. **Evaluation integrity.** New runs save input manifests, label hashes, weight
   hashes, environment versions and resolved device information. Image byte
   hashing is optional. Missing/malformed labels fail preflight. Existing runs
   are not overwritten. Paired comparison verifies settings, provenance when
   available, and ordered saved target statistics.

The `baseline_refine` fast-path bug is fixed: requested refinement now actually
executes. The ordinary dense baseline does not change its predictions. macOS RSS
conversion is corrected for future telemetry; saved historical files are unchanged.

## Decision order

```text
Read frame and compute inexpensive scene signals
    |
    +-- definite initialization / hard refresh --> FULL
    |
    v
Predict motion and measure its reliability
    |
    v
Evaluate novelty, geometric risk, age and soft computation allowance
    |
    +-- global failure / deadline / broad risk --> FULL
    +-- localized risk --> LOCAL cost gate --> confirm, or FULL fallback
    +-- reliable predictions --> SKIP with propagated output
```

`controller_mode=legacy` retains the original scheduler behavior, including its
historical `use_local=False` behavior. In `controller_mode=risk`, disabling LOCAL
means refresh by FULL when required; it does not disable reliability checks.

A detector budget is a **soft rolling GFLOP target**, not a hard real-time
limit. A necessary FULL may exceed it. Overruns and deferred soft refreshes are
logged. Estimated detector GFLOPs do not include flow, tracking or verification.
Measured stage times do include those processing stages.

## Experimental presets

| Preset | Isolated purpose |
|---|---|
| `fixed2_observed`, `adaptive_observed` | Observation anchoring with reference scheduling |
| `fixed2_quality`, `adaptive_quality` | Anchoring plus quality-aware flow |
| `adaptive_safe` | Quality-aware FULL/SKIP with risk scheduling; LOCAL off |
| `fixed2_quality_scores`, `adaptive_safe_scores` | Reliability-dependent confidence decay |
| `adaptive_safe_local` | One-crop cost-aware LOCAL and dynamic context |
| `adaptive_safe_camera` | Background-aligned novelty |
| `adaptive_safe_verify` | Bounded appearance verification |
| `adaptive_safe_corners` | Corner-selected flow points |
| `adaptive_safe_association` | Existing-track low-score recovery |
| `adaptive_safe_births` | Bounded high-confidence LOCAL births |
| `adaptive_hybrid` | Core improvements plus cost-aware LOCAL and score decay |

`adaptive_hybrid` is a candidate combination, not an experimentally selected
winner. Do not enable every optional component merely because it exists.

Ablation suites are `observation`, `quality`, `safe`, `local_cost`, `verification`,
`confidence` and `hybrid`. Each compares a specific intervention with its controls.

## Important new settings

| Setting | Initial choice | Meaning |
|---|---:|---|
| `observation_anchoring` | Off in old presets | Preserve detector/flow observation geometry |
| `quality_aware_flow` | Off in old presets | Variable trust and stronger geometric checks |
| `flow_min_valid_fraction` | 0.50 | Minimum valid fraction for opt-in flow acceptance |
| `flow_min_spread` | 0.10 | Required point-span area relative to the box |
| `risk_uncertainty` | 0.15 | Normalize box-relative KF standard deviation |
| `risk_innovation` | 0.35 | Normalize measurement/prediction disagreement |
| `adaptive_min_interval` | 2 | Shortest nominal adaptive deadline |
| `detector_budget_fraction` | 0.55 | Soft rolling cost target in risk presets |
| `budget_window` | 60 | Frames in the cost window, reset per video |
| `local_max_crops` | 1 | Initial admitted LOCAL crop budget |
| `local_success_prior` | 0.65 | Unvalidated prior, not a measured success rate |
| `local_cost_margin` | 0.10 | Margin above crop/full break-even requirement |
| `reliable_decay_floor` | 0.25 | Reliable boxes still receive some score decay |

Risk scores and KF covariance are not calibrated probabilities. Appearance
correlation is not semantic confirmation. Camera alignment can fail with parallax,
low texture and darkness; the optional mode responds conservatively with FULL.
LOCAL confirmation statistics describe avoiding FULL, not ground-truth accuracy.
All optional state and online cost estimates reset at every source-video boundary.

## Synthetic verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=extracted \
  python -m unittest discover -s tests -v
```

Tests use generated arrays and a fake detector. They do not open the prepared
dataset, raw videos, model checkpoints or invoke real YOLO. They check correctness,
not research performance.

## Proposed first dataset experiment - approval required

The first evaluation should isolate observation anchoring before controller or
LOCAL changes. Proposed scope:

- The same first 15 sorted validation sequences: 4,680 daytime frames.
- Presets: `baseline`, `fixed2_093`, `fixed2_observed`, `C_fullfallback`,
  `adaptive_observed` (suite `observation`).
- Stock `yolov8n.pt`, current AP protocol, unchanged detector settings.
- Fresh outputs under `runs/improvements_observation_v1/`.
- Save predictions for geometry/error analysis. No training and no test-set tuning.

**Do not execute the command below until that dataset run is explicitly approved.**

```bash
PYTHONPATH=extracted python -m adaptive_pipeline.experiments.run_ablations \
  --suite observation \
  --images Prep-Data/images/val \
  --seq-regex '^(?P<video>.+?)_f(?P<frame>\d+)$' \
  --max-videos 15 \
  --save-preds \
  --ap-protocol current \
  --out runs/improvements_observation_v1
```

The suite command saves geometry exports with `--save-preds`. Individual presets
can also use `adaptive_pipeline.run` with the same input selection. Optional
`--hash-inputs` additionally hashes image bytes outside processing-time measurement.

After this first experiment, compare identical Fixed-2 and adaptive propagation
changes before claiming a scheduling benefit. Add quality, risk, confidence and
LOCAL changes independently. Preserve useful negative results. A component that
adds cost without a measured quality benefit should remain disabled.

## New output fields

- `macro_f1` and `macro_f1@0.25`: mean of per-class F1 values, not the harmonic mean
  of aggregate precision and recall. AP and existing P/R definitions are unchanged.
- `input_manifest.json` and `metrics.json.provenance`: settings/input identity,
  annotation hashes, checkpoint hash, environment and actual recorded device.
- `prediction_details.npz` alongside optional `preds.npz`: row-aligned track ID,
  detector-observation age, flow quality and normalized innovation. FULL outputs
  use ID `-1`, age zero, and quality/innovation zero as unavailable sentinels.
- Frame telemetry: prediction risk, flow quality, geometric uncertainty, chosen
  deadline, budget fraction/overrun/defer, LOCAL admission estimates, verification
  time, appearance checks and motion inlier fraction.
- Aggregate telemetry: verification overhead, LOCAL cost rejections and budget
  events. Verification time is a subset of propagation time, not added twice.

Pipeline timing excludes image decoding, scoring, provenance I/O, diagnostic
export construction and prediction-file serialization. Report complete wall-clock
execution separately when measuring application performance.
