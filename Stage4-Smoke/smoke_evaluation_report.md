# YOLOv8n Smoke Test Evaluation Report

## 1. Exact Models Evaluated
- **Stock Model:** `yolov8n.pt` (COCO pretrained)
- **Fine-Tuned Model:** `Stage3-Smoke/runs/smoke/weights/best.pt` (2 epochs)

## 2. Exact Validation Subset
- **Images:** 2,000 images located in `Stage3-Smoke/images/val/`
- **Labels:** 2,000 label files located in `Stage3-Smoke/labels/val/`
- **Verification:** Both models were evaluated on the EXACT same 2,000 pairs. No training images or full test-set images were used.

## 3. Verification Checks
- **Dataset Count:** 2,000 images and 2,000 labels strictly verified.
- **Protocol:** Identical evaluator script (`evaluate_smoke.py`) was used. IoU matching thresholds, confidence pools (0.001 for AP, 0.25 for P/R), device (MPS), resolution (640), and metric implementations were exactly the same.
- **Class Mapping:** Handled accurately; stock model used COCO-mapping while FT model directly predicted the 7 Custom classes.

---

## 4. Overall Stock vs Fine-Tuned Comparison

| Metric | Stock | Fine-tuned (2-epoch) | Delta (FT - Stock) | % Change |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | 0.9773 | 0.7482 | -0.2291 | -23.4% |
| **Recall** | 0.9737 | 0.8409 | -0.1328 | -13.6% |
| **mAP50** | 0.9860 | 0.6012 | -0.3848 | -39.0% |
| **mAP50-95** | 0.9758 | 0.4547 | -0.5211 | -53.4% |
| **TP** | 17,163 | 14,822 | -2,341 | -13.6% |
| **FP** | 399 | 4,987 | +4,588 | +1149.8% |
| **FN** | 463 | 2,804 | +2,341 | +505.6% |
| **GT Count** | 17,626 | 17,626 | 0 | 0.0% |
| **Op Pred Count** | 17,562 | 19,809 | +2,247 | +12.8% |
| **AP Pool Count** | 390,843 | 231,513 | -159,330 | -40.8% |

---

## 5. Latency Comparison (MPS)
*Positive improvement means lower latency (faster inference).*

| Metric | Stock | Fine-tuned | Delta | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Latency** | 9.25 ms | 7.47 ms | -1.78 ms | 19.2% faster |
| **Median Latency** | 7.66 ms | 7.16 ms | -0.50 ms | 6.5% faster |
| **P95 Latency** | 21.50 ms | 7.99 ms | -13.51 ms | 62.8% faster |

> [!NOTE]
> The latency improvement in the fine-tuned model is a side effect of it predicting significantly fewer boxes with confidence > 0.001 (AP Pool dropped from 390k to 231k), which drastically reduces the computational overhead during Non-Maximum Suppression (NMS).

---

## 6. Class-by-Class Comparison

| Class | GT Count | Stock AP50 | FT AP50 | Delta AP50 | Stock AP50-95 | FT AP50-95 | Delta AP50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 13,330 | 0.9878 | 0.9045 | -0.0833 | 0.9761 | 0.7252 | -0.2509 |
| **bus** | 318 | 0.9866 | 0.6291 | -0.3575 | 0.9822 | 0.5175 | -0.4647 |
| **truck** | 859 | 0.9780 | 0.7085 | -0.2695 | 0.9728 | 0.6023 | -0.3705 |
| **pedestrian** | 1,943 | 0.9863 | 0.7670 | -0.2193 | 0.9673 | 0.5439 | -0.4234 |
| **rider** | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| **bicycle** | 31 | 0.9908 | 0.4819 | -0.5089 | 0.9800 | 0.2556 | -0.7244 |
| **traffic light** | 1,145 | 0.9863 | 0.7173 | -0.2690 | 0.9766 | 0.5387 | -0.4379 |

*Note: `rider` had 0 instances in this 2,000-image subset.*

**Material Changes:**
- **ALL classes showed a material decrease** in accuracy.
- `bicycle` suffered the worst degradation (AP50 dropped by -0.5089).
- `car` was the most resilient but still lost -0.2509 in AP50-95.

---

## 7. Interpretation & Conclusion

**"Does the 2-epoch fine-tuned YOLOv8n show improvement over stock YOLOv8n on the same 2,000-image validation subset?"**

**Conclusion: NO.** 
The 2-epoch fine-tuned model performs significantly worse across every single accuracy metric (Precision, Recall, mAP50, and mAP50-95) and for every single class compared to the stock model. 

- **Trade-offs:** There is an artificial gain in inference latency (p95 dropped significantly from 21.5ms to 7.99ms). However, this is strictly a consequence of catastrophic forgetting resulting in fewer total predictions passing the initial low-confidence filter, easing the burden on NMS computation. It is not an architectural speedup.
- **Why this happens:** When the classification heads are replaced/mapped to a new 7-class configuration, the pre-trained weights in the head are lost or misaligned. 2 epochs on 11,680 images is not nearly enough to converge the new heads, leading to high False Positives (+1149.8%) and False Negatives (+505.6%).

**Limitations:**
1. This is a strict **SMOKE-TEST comparison**, validating only the plumbing. 
2. Only **2 training epochs** were completed. We cannot claim fine-tuning is definitively beneficial or harmful based on this alone, as the model has not had time to converge.
3. This subset does not represent the full 43,772-image test set. 
4. You should proceed with a full training run (e.g., 30+ epochs) as planned to see genuine convergence and evaluate if it actually surpasses the stock baseline.

---
## Output Files Created
- `evaluate_smoke.py`: Authoritative identical evaluation script bridging custom DS2 and COCO labels for both models.
- `results_smoke_eval.json`: The raw extracted inference results from the evaluation.
