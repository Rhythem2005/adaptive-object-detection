"""Controlled evaluation of the existing methods on the local five-sequence export.

See docs/ARGOVERSE_HD.md for protocol and exact commands. Never writes to data/.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

from ..evaluation.argoverse import load_argoverse
from ..evaluation.provenance import sha256_file
from ..detection.detector import YOLODetector
from ..evaluation.argoverse import COCO_NAMES
from ..run import build_config, run_sequences
from .run_ablations import COLS, row

REFERENCES = {
    "baseline": "runs/improvements_observation_v1/baseline/metrics.json",
    "fixed2_093": "runs/improvements_observation_v1/fixed2_093/metrics.json",
    "C_fullfallback": "runs/improvements_observation_v1/C_fullfallback/metrics.json",
    "adaptive_observed": "runs/improvements_quality_v1/adaptive_observed/metrics.json",
    "adaptive_quality": "runs/improvements_quality_v1/adaptive_quality/metrics.json",
}


def recovered_configs(repo, class_map):
    """Require current presets to reproduce saved settings before dataset overrides."""
    configs, references = {}, {}
    for name, relative in REFERENCES.items():
        path = repo / relative
        saved = json.loads(path.read_text())
        cfg = build_config(name, [])
        if json.loads(json.dumps(cfg.to_dict())) != saved["config"] or saved["ap_protocol"] != "current":
            raise ValueError(f"Preset {name} differs from its saved reference {path}")
        weights = (repo / cfg.model_path).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Local weights required (no download): {weights}")
        if sha256_file(weights) != saved["provenance"]["weights_sha256"]:
            raise ValueError(f"Checkpoint differs from saved reference: {weights}")
        # Only dataset taxonomy and explicit hardware/checkpoint paths change.
        configs[name] = cfg.with_overrides(class_map=class_map, device="cpu", model_path=str(weights))
        references[name] = {"path": str(path), "sha256": sha256_file(path),
                            "overrides": ["class_map", "device=cpu", "absolute model_path"]}
    return configs, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validate-only", action="store_true", help="decode/audit only; no model load")
    parser.add_argument("--max-frames", type=int, help="development check: first N frames of EACH sequence")
    parser.add_argument("--presets", nargs="+", choices=REFERENCES, default=list(REFERENCES))
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    sequences, names, class_map, audit = load_argoverse(args.dataset)
    repo = Path(__file__).resolve().parents[3]
    configs, references = recovered_configs(repo, class_map)
    out.mkdir(parents=True)
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
    plan = {"argv": sys.argv, "kind": "validation_only" if args.validate_only else
            ("development_check" if args.max_frames else "benchmark"),
            "max_frames_per_sequence": args.max_frames, "ap_protocol": "current",
            "configurations": {n: configs[n].to_dict() for n in args.presets},
            "recovered_references": references,
            "timing_scope": "pipeline processing; excludes image decoding, scoring, warmup, model loading and provenance I/O",
            "ordering": "sequential methods; sequence order by sid; frame order by validated manifest frame_index"}
    (out / "execution_plan.json").write_text(json.dumps(plan, indent=2))
    source = out / "source_snapshot"
    shutil.copytree(repo / "extracted/adaptive_pipeline", source / "adaptive_pipeline",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(repo / "tests", source / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "working_tree.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=repo))
    (out / "git_status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=repo))
    print(json.dumps({"images": audit["images"], "annotations": audit["annotations"],
                      "regular_instances": audit["regular_instances"], "crowd_instances": audit["crowd_instances"],
                      "sequences": audit["sequences"], "output": str(out)}, indent=2), flush=True)
    if args.validate_only:
        return
    detector = YOLODetector(configs[args.presets[0]])
    for name, coco_id in COCO_NAMES.items():
        if detector.model.names[coco_id].replace(" ", "_") != name:
            raise ValueError(f"Checkpoint taxonomy disagrees at COCO index {coco_id}: expected {name}")
    rows = []
    for name in args.presets:
        print(f"Running {name}", flush=True)
        result, detector = run_sequences(configs[name], sequences, str(out / name), detector=detector,
                                         max_frames=args.max_frames, save_preds=True, log_every=1,
                                         hash_inputs=True, names=names)
        result["dataset"] = {"annotations_sha256": audit["annotations_sha256"],
                             "manifest_sha256": audit["manifest_sha256"], "names": names,
                             "metric_protocol": audit["metric_protocol"], "ignore_policy": audit["ignore_policy"]}
        (out / name / "metrics.json").write_text(json.dumps(result, indent=2))
        rows.append(row(name, result))
        with (out / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLS)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(rows[-1]), flush=True)
    manifests = [json.loads((out / n / "input_manifest.json").read_text())["sha256"] for n in args.presets]
    if len(set(manifests)) != 1:
        raise ValueError("Input/target hashes changed between methods; benchmark invalid")


if __name__ == "__main__":
    main()
