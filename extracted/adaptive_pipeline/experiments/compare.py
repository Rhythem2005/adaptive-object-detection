"""Paired per-video bootstrap between two finished runs.

  python -m adaptive_pipeline.experiments.compare runs/test/fixed4_flow runs/test/adaptive_full

Reports B - A for mAP50 and mAP50-95 with 95% CIs. Use it for every claim
of the form "adaptive is better/no worse than X at matched compute".
"""
import argparse
import json
import os
from pathlib import Path

from ..evaluation import evaluator


def verify_comparison(run_a, run_b, protocol):
    runs = [json.loads((Path(d) / "metrics.json").read_text()) for d in (run_a, run_b)]
    for run in runs:
        if run.get("ap_protocol") != protocol:
            raise ValueError("Saved AP protocol differs from the requested comparison protocol")
    for field in ("model_path", "imgsz", "det_conf", "det_iou", "max_det", "half", "class_map"):
        if runs[0]["config"].get(field) != runs[1]["config"].get(field):
            raise ValueError(f"Detector settings differ: {field}")
    provenance = [r.get("provenance", {}) for r in runs]
    verified = {}
    for field in ("weights_sha256", "input_manifest_sha256"):
        a, b = [p.get(field) for p in provenance]
        if a is not None and b is not None and a != b:
            raise ValueError(f"Run provenance differs: {field}")
        verified[field] = a is not None and b is not None
    verified["recorded_devices"] = [p.get("resolved_device") for p in provenance]
    return verified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--ap-protocol", choices=["current", "legacy"], default="current")
    a = ap.parse_args()
    evaluator.AP_PROTOCOL = a.ap_protocol
    verification = verify_comparison(a.run_a, a.run_b, a.ap_protocol)
    res = evaluator.paired_bootstrap(os.path.join(a.run_a, "eval_stats.npz"),
                                     os.path.join(a.run_b, "eval_stats.npz"), a.n_boot)
    eff = {}
    for name, d in (("A", a.run_a), ("B", a.run_b)):
        e = json.loads((Path(d) / "metrics.json").read_text())["efficiency"]
        eff[name] = {k: e.get(k) for k in ("detector_skip_rate", "gflops_fraction_of_baseline", "fps_end_to_end")}
    res["efficiency"] = eff
    res["comparison_verification"] = verification
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
