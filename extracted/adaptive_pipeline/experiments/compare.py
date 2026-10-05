"""Paired per-video bootstrap between two finished runs.

  python -m adaptive_pipeline.experiments.compare runs/test/fixed4_flow runs/test/adaptive_full

Reports B - A for mAP50 and mAP50-95 with 95% CIs. Use it for every claim
of the form "adaptive is better/no worse than X at matched compute".
"""
import argparse
import json
import os

from ..evaluation import evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--ap-protocol", choices=["current", "legacy"], default="current")
    a = ap.parse_args()
    evaluator.AP_PROTOCOL = a.ap_protocol
    res = evaluator.paired_bootstrap(os.path.join(a.run_a, "eval_stats.npz"),
                                     os.path.join(a.run_b, "eval_stats.npz"), a.n_boot)
    eff = {}
    for name, d in (("A", a.run_a), ("B", a.run_b)):
        e = json.load(open(os.path.join(d, "metrics.json")))["efficiency"]
        eff[name] = {k: e.get(k) for k in ("detector_skip_rate", "gflops_fraction_of_baseline", "fps_end_to_end")}
    res["efficiency"] = eff
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
