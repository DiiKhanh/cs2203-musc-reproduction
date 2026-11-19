# MuSc E01 MVTec AD Reproduction Result

## Run Identity

| Field | Value |
|---|---|
| Date | 2026-10-12 |
| Experiment | E01 full MVTec AD quantitative baseline |
| Dataset scope | All 15 MVTec AD categories |
| MuSc commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Dataset source | Hugging Face community mirror `micguida1/mvtech_anomaly_detection` |
| Dataset revision | `eb692b36147a3b3f12c561b879a062104c1f20b8` |
| Run ID | `20261012_074856_E01_mvtec_all_clip_l14_336_baseline` |
| Run directory | `MyDrive/CS2203_MuSc/runs/20261012_074856_E01_mvtec_all_clip_l14_336_baseline/` |
| Reviewed ZIP | `/Users/duykhanh/Downloads/20261012_074856_E01_mvtec_all_clip_l14_336_baseline.zip` |
| ZIP SHA-256 | `a45be82b9ab19ca6725cd08b2ff151e3ea0769dafcba7b2585994edb94281e68` |
| Outcome | Accepted with documented image-level deviations |

## Artifact Integrity

The ZIP integrity test completed without an error. It contains:

- 15 category-level `metrics.json` files.
- 15 category-level `SUCCESS.json` files.
- One successful result workbook for every category.
- The aggregated CSV and XLSX tables.
- The repository comparison tables and alert list.
- Configuration, environment, commands, console logs, and source patch records.

There is exactly one successful metrics record for every MVTec AD category. The `pill` category also contains an interrupted `attempt_01` without a status file and a successful `attempt_02`.

## Frozen Configuration

| Item | Value |
|---|---|
| Backbone | OpenAI CLIP `ViT-L-14-336` |
| Input resolution | `518 x 518` |
| Feature layers | `[5, 11, 17, 23]` |
| LNAMD degrees | `[1, 3, 5]` |
| `divide_num` | `1` |
| Batch size | `4` |
| Visualization | Disabled for the quantitative run |
| Excel output | Enabled |
| Upstream seed | `42` |

All 16 commands, including the interrupted and resumed `pill` attempts, retain the frozen backbone, resolution, batch size, and `divide_num = 1` settings.

## Environment

Both recorded execution sessions used the same environment:

| Component | Observed value |
|---|---|
| GPU | Tesla T4 |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| Torchvision | 0.26.0+cu130 |
| CUDA runtime | 13.0 |
| OpenCV | 4.14.0 |
| scikit-learn | 1.6.1 |
| scikit-image | 0.25.2 |
| timm | 0.9.12 |

This differs from the repository's documented Python 3.8, PyTorch 2.0.1, and CUDA 11.7 environment.

## Mean Quantitative Results

All values and deltas are percentage points. The repository reference is published to two decimal places.

| Level | Metric | Repository reference | Reproduced E01 | Delta |
|---|---|---:|---:|---:|
| Pixel | AUROC | 97.11 | 97.114390 | +0.004390 |
| Pixel | F1-max | 62.16 | 62.157971 | -0.002029 |
| Pixel | AP | 62.26 | 62.261741 | +0.001741 |
| Pixel | PRO | 93.45 | 93.452960 | +0.002960 |
| Image | AUROC | 97.77 | 97.734031 | -0.035969 |
| Image | F1-max | 97.37 | 97.405687 | +0.035687 |
| Image | AP | 99.07 | 99.052802 | -0.017198 |

The largest absolute mean delta is 0.035969 percentage points. The full per-category table is preserved in `mvtec_ad_reference_comparison.csv` and `.xlsx` inside the reviewed ZIP. The reproduced per-category CSV is also preserved locally as [`artifacts/e01/mvtec_ad_reproduced.csv`](artifacts/e01/mvtec_ad_reproduced.csv), with SHA-256 `b55c95821fe4455cd4f225ed23baced0788b365afb71914bb8e2d087eba17687`.

## Per-Category Agreement

All 60 pixel-level comparisons, comprising four metrics across 15 categories, are within 0.10 percentage points of the repository table. Their largest absolute deltas are:

| Metric | Largest absolute delta | Category |
|---|---:|---|
| Pixel AUROC | 0.005446 | `wood` |
| Pixel F1-max | 0.008689 | `zipper` |
| Pixel AP | 0.050718 | `capsule` |
| Pixel PRO | 0.073777 | `capsule` |

Ten of the 45 image-level category-metric comparisons exceed 0.10 percentage points:

| Category | Metric | Reproduced | Reference | Delta |
|---|---|---:|---:|---:|
| `capsule` | Image AUROC | 96.689270 | 96.45 | +0.239270 |
| `capsule` | Image F1-max | 95.693780 | 94.88 | +0.813780 |
| `metal_nut` | Image AUROC | 96.578690 | 96.92 | -0.341310 |
| `metal_nut` | Image AP | 99.146209 | 99.25 | -0.103791 |
| `pill` | Image AUROC | 96.126568 | 96.24 | -0.113432 |
| `screw` | Image AUROC | 82.311949 | 82.17 | +0.141949 |
| `transistor` | Image AUROC | 99.291667 | 99.42 | -0.128333 |
| `transistor` | Image F1-max | 94.736842 | 95.00 | -0.263158 |
| `transistor` | Image AP | 99.019395 | 99.19 | -0.170605 |
| `zipper` | Image AUROC | 99.501050 | 99.84 | -0.338950 |

The deviations are localized to image-level scoring. The pixel-level agreement shows that feature extraction, LNAMD, MSM, anomaly-map generation, and pixel metrics were reproduced closely. A plausible source is environment-sensitive ordering in RsCIN, which applies `torch.topk` to class-token similarities before propagating image scores. This is an inference, not a proven cause. The current Colab software stack differs substantially from the official environment, and no official-environment rerun was performed.

The deviations occur in categories completed both before and after the CPU memory patch. They therefore cannot be attributed solely to that patch from the available evidence.

## Interrupted Run and Recovery

The first execution session completed `bottle` through `metal_nut`. The Colab runtime exhausted CPU RAM during `pill/attempt_01` while feature extraction was at batch 141 of 167. Because the runtime terminated abruptly, this attempt has a console log and partial output directory but no `FAILED.json`.

The resumed session:

1. Reused the same immutable E01 Run ID.
2. Preserved and skipped all categories with `SUCCESS.json`.
3. Applied the documented CPU memory cleanup patch.
4. Completed `pill/attempt_02` and the six remaining categories.

Successful category subprocesses consumed 7,169.446 seconds in total, or approximately 119.49 minutes. This excludes setup, dataset download and extraction, the interrupted attempt, and final aggregation.

## Compatibility and Memory Changes

The run preserved two source changes:

1. The import compatibility fix prepends the MuSc repository path and adds `datasets/__init__.py` so Python imports MuSc's local dataset module.
2. The CPU memory patch imports `gc`, releases feature tensors after anomaly-map construction, and avoids two redundant NumPy copies of the completed anomaly-map array.

The normalized semantic diff of the memory patch contains only those changes. The saved raw diff is unusually large because the Colab patch cell normalized line endings when rewriting `models/musc.py`; this did not represent hundreds of semantic source changes.

The post-restart `clean-session-check.json` reports a zero delta for `bottle`, but `run_category("bottle")` reused the existing successful attempt. It therefore verifies result retrieval and configuration continuity rather than a new patched execution of `bottle`. No same-category patched versus unpatched A/B run was performed. The memory patch is computation-preserving by inspection, but this remains a reproducibility limitation that should be disclosed.

## Decision

E01 is accepted as the full MVTec AD quantitative baseline because:

- All 15 categories completed under the frozen configuration.
- All mean metrics are within 0.036 percentage points of the repository table.
- Every pixel-level category metric is within 0.10 percentage points.
- The larger category deviations are confined to image-level metrics and are explicitly preserved for discussion.

E01Q was subsequently completed and accepted in the [E01Q result note](2026-10-12-musc-e01q-result.md). The next required experiment is the VisA baseline using the [Vietnamese E02 guide](2026-10-12-huong-dan-colab-musc-e02-visa.md). The image-level deviations should be discussed as a reproducibility limitation rather than hidden or described as an exact per-category match.
