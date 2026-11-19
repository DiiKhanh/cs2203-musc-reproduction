# MuSc E01Q MVTec AD Qualitative Result

## Run Identity

| Field | Value |
|---|---|
| Date | 2026-10-12 |
| Experiment | E01Q MVTec AD qualitative analysis |
| Source quantitative run | `20261012_074856_E01_mvtec_all_clip_l14_336_baseline` |
| Categories | `leather`, `transistor`, `screw` |
| MuSc commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Run ID | `20261012_151018_E01Q_mvtec_qualitative` |
| Run directory | `MyDrive/CS2203_MuSc/runs/20261012_151018_E01Q_mvtec_qualitative/` |
| Reviewed ZIP | `/Users/duykhanh/Downloads/20261012_151018_E01Q_mvtec_qualitative.zip` |
| ZIP SHA-256 | `3d985777d766257a5cc3b24e8b2dde905ac84f51db91fe45f6d3f0d8a7ced066` |
| Outcome | Accepted |

## Artifact Integrity

The ZIP integrity test completed without an error. It contains:

- Three category-level `metrics.json` and `SUCCESS.json` records.
- 384 saved heatmaps: 124 for `leather`, 100 for `transistor`, and 160 for `screw`.
- Scores for every test image in the three categories.
- 18 selected cases, comprising two best-localization anomalies, two worst-localization anomalies, and two high-response normal candidates per category.
- Original images, ground-truth masks, heatmaps, overlays, and four-panel figures for every selected case.
- Environment, command, compatibility-fix, and CPU-memory-patch records.

## Frozen Configuration

E01Q uses the same baseline as E01: OpenAI CLIP `ViT-L-14-336`, image size 518, feature layers `[5, 11, 17, 23]`, LNAMD degrees `[1, 3, 5]`, `divide_num = 1`, batch size 4, and upstream seed 42. The only intended execution change is `vis = true` with `vis_type = whole_norm` so MuSc saves anomaly maps.

The run used a Tesla T4 with Python 3.13.15, PyTorch 2.11.0+cu130, Torchvision 0.26.0+cu130, and CUDA 13.0. It used the same MuSc commit and VisA-independent MVTec source revision as E01.

## Quantitative Continuity with E01

| Category | Pixel AUROC | Pixel F1-max | Pixel AP | PRO | Image AUROC | Image F1-max | Image AP | Time, min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `leather` | 99.7170 | 62.8450 | 64.4725 | 98.7345 | 100.0000 | 100.0000 | 100.0000 | 8.40 |
| `transistor` | 91.3814 | 59.2420 | 58.4081 | 77.2087 | 99.2917 | 94.7368 | 99.0194 | 6.14 |
| `screw` | 98.7674 | 41.8771 | 36.1135 | 94.4013 | 82.3119 | 88.8889 | 90.9512 | 12.16 |

The maximum absolute metric delta from E01 is approximately `1.42e-14` percentage points for every category. This is floating-point noise. Enabling visualization and running with the documented CPU-memory patch did not materially change the seven reported metrics for these categories.

The three category subprocesses consumed 1,602.713 seconds in total, or approximately 26.71 minutes. This excludes runtime setup, dataset extraction, case ranking, panel creation, and ZIP export.

## Case Selection Protocol

For each category, the run selected:

1. The two anomalous images with the highest per-image pixel AUROC.
2. The two anomalous images with the lowest per-image pixel AUROC.
3. The two normal images with the highest mean response over the top one percent of heatmap pixels.

The per-image pixel AUROC used for ranking was reconstructed from the saved 8-bit JET heatmap. It is suitable for case selection but is not the official raw-score metric reported by MuSc. The normal selections are high-response candidates. They are not proven false positives because no classification or segmentation threshold was applied to them.

## Qualitative Findings

### Leather

`leather` is the strongest of the three categories. Small `glue` and `poke` defects produce compact peaks that overlap the masks closely. Even the lowest-ranked selected anomaly has a saved-map pixel AUROC above 0.9878. The `color/015` case also shows an unrelated strong response near the upper-left boundary, but the actual defect remains the dominant central response. Normal leather images have much lower top-one-percent responses than anomalous images, with a normal maximum of 0.4157 and an anomalous median of 0.7024.

### Transistor

MuSc localizes a damaged transistor case well when the defect is a compact region on the package. It performs poorly on the selected `cut_lead` cases. For `cut_lead/008`, the response is diffuse across the package and nearby structure while the long missing-lead region is only weakly aligned with the heatmap. Its saved-map pixel AUROC is 0.5957. This category also contains substantial responses on normal package edges, leads, and background patterns. These observations agree with the low category PRO of 77.21 despite high image-level AUROC.

### Screw

The selected `scratch_neck` case contains a strong local peak at the defect, but the heatmap also activates across most of the screw body. The `manipulated_front` examples are difficult because the ground-truth defect is confined to a small tip region while the anomaly map remains object-wide. Normal screw images show a similar object-shaped response. The median top-one-percent response is 0.7486 for anomalous images and 0.7070 for normal images, which provides a visual explanation for the relatively weak image AUROC of 82.31 and pixel AP of 36.11.

## Interpretation Limits

`whole_norm` creates a common visualization scale within one category run, but heatmap colors alone are not classification decisions. Object-wide activation in a normal image does not prove a false positive without applying the score threshold used for classification or segmentation.

The selected cases are deliberately extreme. They illustrate strengths and weaknesses but do not estimate how frequently each behavior occurs. Aggregate metrics remain the primary performance evidence, while these panels explain representative behavior behind those numbers.

## Decision

E01Q is accepted because all three categories completed, every rerun metric matches E01 within floating-point noise, all 384 expected heatmaps were produced, and the artifact contains balanced successful, difficult, and normal examples. MVTec AD quantitative and qualitative reproduction is now complete.

The next required experiment is E02 on VisA with the same frozen baseline configuration. The E01Q panels and findings should be reused in the final report and presentation, especially the compact leather detections, the transistor `cut_lead` failure, and the object-wide screw response.
