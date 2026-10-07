| Draft Claim | Status/Implementation |
|---|---|
| "Reduces redundant YOLO invocations by 30-50%" | [MEASURED] results/metrics.csv -> See baseline FLOP vs best configs |
| "Local ROI tracking bridges full-frame gaps" | [MEASURED] local_redetect_rate=0 across the board, so this claim is DEBUNKED / NOT VERIFIED. Config C_fullfallback (local=False) actually outperformed C. |
| "mAP50-95 parity with k_max=2 or k_max=3" | [MEASURED] We observed that none of the adaptive configs significantly beat Fixed-K. Adaptive methods achieved near parity, but Fixed-K strictly won out on trade-offs. |
| "Controller runs in <1ms overhead" | [MEASURED] results/timing.csv -> overhead |
| "End-to-end processing exceeds 60 FPS" | [MEASURED] results/timing.csv -> yes, most achieve >60 FPS |
