# Experiment B Report: Targeted Fine-Tuning of YOLOv8n

## A. Audit Findings

*   **Python Version:** 3.14.6 [VERIFIED] (ran `.venv/bin/python -c "import platform; print(platform.python_version())"`)
*   **PyTorch Version:** 2.14.0 [VERIFIED] (ran `.venv/bin/python -c "import torch; print(torch.__version__)"`)
*   **MPS Status:** Available and built (ran `torch.randn` matrix multiplication successfully) [VERIFIED]
*   **Ultralytics Version:** 8.4.149 [VERIFIED]
*   **Albumentations:** NOT INSTALLED [VERIFIED]
*   **Project Root:** `/Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT` (contains `Prep-Data/` and `yolov8n.pt`) [VERIFIED]
*   **Notebook:** `Model/Phase1-Tune-Improved-Stage3-Fixed.ipynb` [VERIFIED]
*   **Train/Val/Test Original File Counts:** 206,304 / 43,350 / 43,772 [VERIFIED] (counted `Prep-Data/images/{split}`)
*   **Targeted Dataset Manifest:** Found at `Prep-Data-Sampled-Targeted/sampling_manifest.csv`. It contains 11,680 rows, exactly 584 unique videos, 20 frames each, matching the required columns. [VERIFIED]

## B. Critique of My Approach and Proposed Better Approach

1.  **Albumentations Pipeline:** Passing `augmentations=[]` properly overrides the default list of Albumentations transforms (Blur, MedianBlur, ToGray, CLAHE, etc.) which sets `T = []`. Additionally, since Albumentations is not even installed in the `.venv`, it fails gracefully and disables itself entirely. **Approach is sound, but passing `augmentations=[]` is a good explicit safeguard.** [VERIFIED]
2.  **`close_mosaic=10` when `epochs=5`:** This setting disables mosaic augmentation in the last `close_mosaic` epochs. If `epochs=5` and `close_mosaic=10`, the dataloader would attempt to disable mosaic at epoch `-5`. Because training starts at epoch 0, this condition is never met, meaning mosaic is **never closed**.
    *   **Proposed TYPE 2 Change:** Adjust `close_mosaic` to `0` if you want mosaic for all 5 epochs, or something smaller like `2` (closes mosaic in the last 2 epochs, which is standard practice for YOLOv8 fine-tuning to let the network learn natural scales).
3.  **`optimizer="auto"`:** For this schedule (`epochs=5` and `11680` images with `batch=16`), the iterations equal 3,650. Ultralytics' `auto` optimizer falls back to `AdamW` when iterations `< 10,000`, using a learning rate of `round(0.002 * 5 / (4 + 7), 6) = 0.000909`. [VERIFIED]
    *   **Conclusion:** The auto optimizer is correct and predictable.
4.  **`patience=5` when `epochs=5`:** Early stopping needs to observe `patience` epochs of no improvement before stopping. Since `epochs=5`, it is impossible to trigger early stopping.
    *   **Conclusion:** Harmless, but essentially ineffective. No change needed unless training is extended.
5.  **`deterministic=True` on MPS:** Setting `torch.use_deterministic_algorithms(True, warn_only=True)` works without errors on MPS, though it falls back to non-deterministic versions of `scatter_reduce_mps` and `index_put_with_accumulate_mps` (producing warnings). [VERIFIED]
    *   **Proposed TYPE 1 Change:** Enable `PYTORCH_ENABLE_MPS_FALLBACK="1"` explicitly before importing Torch to prevent crashes on unsupported MPS operations.
6.  **Batch Size and Workers:** `batch=16` and `workers=8` are standard defaults. M-series Macs usually handle this easily, but `workers=0` or `workers=4` might be safer to avoid memory pressure on Apple Silicon dataloaders depending on RAM. However, the original smoke test ran with `workers=8` (which Ultralytics scaled to 0 in its output). [INFERRED]
    *   **Conclusion:** Will leave as specified.

## C. TYPE 1 Changes Applied and TYPE 2 Proposals Awaiting Approval

**TYPE 1 Changes Applied (in `Experiment-B-Targeted-Finetune.ipynb`):**
*   Rewrote the notebook using `json` generation (avoiding `nbformat` due to missing dependency).
*   Added strict preflight checks that assert every count, overlap check, manifest path, and YAML configuration before allowing training.
*   Added `PYTORCH_ENABLE_MPS_FALLBACK="1"` before importing Torch.
*   Gated the `model.train()` call behind `RUN_TRAINING = True` and `preflight_passed = True`.
*   Saved a pre-flight state record to `Prep-Data-Sampled-Targeted/experiment_b_preflight.json`.

**TYPE 2 Proposals Awaiting Approval (NOT APPLIED YET):**
*   **`close_mosaic`**: The user specified `close_mosaic=10` for `epochs=5`, which means mosaic never closes. Recommend setting `close_mosaic=2` or `0`.
*   **`workers`**: Consider setting `workers=4` (or `workers=0` if dataloader crashes on MacOS).

## D. Exact Targeted Dataset Path

`/Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/Prep-Data-Sampled-Targeted` [VERIFIED]

## E. Exact Training Config

```python
{
    "data":            "/Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/Prep-Data-Sampled-Targeted/experiment_b.yaml",
    "epochs":          5,
    "patience":        5,
    "imgsz":           640,
    "batch":           16,
    "workers":         8,
    "device":          "mps",
    "seed":            0,
    "deterministic":   True,
    "pretrained":      True,
    "resume":          False,
    "cache":           False,
    "optimizer":       "auto",
    "mosaic":          1.0,
    "close_mosaic":    10,
    "val":             True,
    "save":            True,
    "plots":           True,
    "project":         "/Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/Experiment-B/runs",
    "name":            "targeted_finetune",
    "exist_ok":        False,
    "augmentations":   []
}
```
[VERIFIED] - Included precisely as this dictionary in the notebook.

## F. Validation Dataset

Path: `/Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/Prep-Data/images/val`
Count: 43,350 images [VERIFIED]

## G. Starting Checkpoint

Path: `/Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/yolov8n.pt` [VERIFIED]

## H. Preflight Table

| Check | Status | Measured Value |
| :--- | :---: | :--- |
| train images == 11,680 | PASS | 11,680 |
| train labels == 11,680 | PASS | 11,680 |
| train videos == 584 | PASS | 584 |
| 20 frames per video | PASS | all 584 have 20 |
| val images == 43,350 | PASS | 43,350 |
| test images == 43,772 | PASS | 43,772 |
| YAML train points to targeted dataset | PASS | `images/train` |
| YAML val points to original Prep-Data | PASS | `/.../Prep-Data/images/val` |
| YAML has no test entry | PASS | (True) |
| YAML names == 7 expected classes | PASS | (True) |
| no train↔val video overlap | PASS | (True) |
| no train↔test video overlap | PASS | (True) |
| start checkpoint is root yolov8n.pt | PASS | `/.../yolov8n.pt` |
| no Stage3-Smoke references in config | PASS | notebook built cleanly |
| resume=False, pretrained=True | PASS | set in config |
| MPS available | PASS | built=True, available=True |
| output dir non-colliding | PASS | `/.../Experiment-B/runs/targeted_finetune` does not exist |

## I. READY for the 5-epoch run?

**YES**. The notebook has been cleanly regenerated, the data is verified, no leaks exist, the configuration matches the strict specifications, and the gating prevents accidental runs. (You just need to decide if you want to leave `close_mosaic=10` as is).

## J. Remaining Unverified Items or Blockers

None. All constraints from the prompt have been met.

## K. Confirmation that training was not started

**Confirmed.** The notebook script explicitly sets `RUN_TRAINING = False`, and the cell is completely gated by this flag. No `model.train()` or `yolo train` has been executed. [VERIFIED]
