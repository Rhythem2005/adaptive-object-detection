"""Named configurations. Each is a set of overrides on PipelineConfig()."""

NO_ADAPT = dict(use_novelty=False, use_track_health=False, use_local=False,
                use_uncertainty=False, use_illum_trigger=False)

PRESETS = {
    # Reference: detector on every frame, in this harness. Must reproduce the frozen baseline.
    "baseline": dict(policy="always"),

    # Adaptive system (default config) and its build-up
    "adaptive_full": dict(),
    "adaptive_sched_only": dict(**{**NO_ADAPT, "use_novelty": True, "use_illum_trigger": True}),
    "adaptive_plus_health": dict(**{**NO_ADAPT, "use_novelty": True, "use_illum_trigger": True,
                                    "use_track_health": True}),
    "adaptive_plus_local": dict(use_uncertainty=False),
    "adaptive_no_novelty": dict(use_novelty=False),
    "adaptive_kalman_only": dict(propagation="kalman"),  # no flow -> no health signal

    # Environmental adaptation / original selective re-detection (score on night + human labels)
    "baseline_clahe": dict(policy="always", use_clahe=True),
    "adaptive_full_clahe": dict(use_clahe=True),
    "baseline_refine": dict(policy="always", keyframe_refine=True),
    "adaptive_full_refine": dict(keyframe_refine=True),
}

# Non-adaptive controls: fixed interval with three propagation strengths
for k in (2, 3, 4, 6, 8):
    PRESETS[f"fixed{k}_hold"] = dict(policy="fixed", fixed_k=k, propagation="hold")
    PRESETS[f"fixed{k}_kalman"] = dict(policy="fixed", fixed_k=k, propagation="kalman")
    PRESETS[f"fixed{k}_flow"] = dict(policy="fixed", fixed_k=k, propagation="flow")

# Operating-point sweep for the adaptive Pareto curve (tune on VAL)
for nov in (0.10, 0.20, 0.30, 0.45):
    for km in (4, 8, 12):
        PRESETS[f"adaptive_nov{nov:.2f}_k{km}"] = dict(novelty_thr=nov, k_max=km)

SUITES = {
    "sanity": ["baseline"],
    "main": ["baseline", "adaptive_full"] + [f"fixed{k}_flow" for k in (2, 3, 4, 6, 8)],
    "propagation": [f"fixed{k}_{p}" for k in (2, 4, 8) for p in ("hold", "kalman", "flow")],
    "components": ["adaptive_sched_only", "adaptive_plus_health", "adaptive_plus_local",
                   "adaptive_full", "adaptive_no_novelty", "adaptive_kalman_only"],
    "pareto": [k for k in PRESETS if k.startswith("adaptive_nov")] + [f"fixed{k}_flow" for k in (2, 3, 4, 6, 8)],
    "illumination": ["baseline", "baseline_clahe", "adaptive_full", "adaptive_full_clahe"],
    "refine": ["baseline", "baseline_refine", "adaptive_full", "adaptive_full_refine"],
}

# ── Final targeted validation (appended, existing presets untouched) ──
PRESETS["adaptive_C"] = dict(
    k_max=4, novelty_thr=0.10, conf_decay=0.93, k_uncertain=2,
    use_local=True, max_local_rois=5, fail_frac_thr=0.20,
)
PRESETS["C_fullfallback"] = dict(
    k_max=4, novelty_thr=0.10, conf_decay=0.93, k_uncertain=2,
    use_local=True, max_local_rois=0, fail_frac_thr=0.20,
)
PRESETS["C_k3_no_local"] = dict(
    k_max=3, novelty_thr=0.10, conf_decay=0.93, k_uncertain=2,
    use_local=True, max_local_rois=0, fail_frac_thr=0.20,
)
PRESETS["C_k5_no_local"] = dict(
    k_max=5, novelty_thr=0.10, conf_decay=0.93, k_uncertain=2,
    use_local=True, max_local_rois=0, fail_frac_thr=0.20,
)
PRESETS["fixed2_093"] = dict(policy="fixed", fixed_k=2, propagation="flow", conf_decay=0.93)
PRESETS["fixed3_093"] = dict(policy="fixed", fixed_k=3, propagation="flow", conf_decay=0.93)
PRESETS["fixed4_093"] = dict(policy="fixed", fixed_k=4, propagation="flow", conf_decay=0.93)
