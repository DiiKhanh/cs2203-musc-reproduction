# MuSc E02Q VisA Qualitative Result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-15
- Verification Status: ANALYZED
- Version Label: validation_v1

## Run Identity

| Field | Value |
|---|---|
| Execution date | 2026-10-14 UTC |
| Experiment | E02Q VisA qualitative analysis |
| Source quantitative run | `20261012_171400_E02_visa_all_clip_l14_336_baseline` |
| Dataset scope | VisA `pcb2`, `pcb3`, `pcb4`, and `fryum`, 752 test images |
| MuSc commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Dataset source | Official Amazon S3 `VisA_20220922.tar` |
| Dataset SHA-256 | `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| Run ID | `20261014_050040_E02Q_visa_qualitative` |
| Run directory | `MyDrive/CS2203_MuSc/runs/20261014_050040_E02Q_visa_qualitative/` |
| Reviewed ZIP | `/Users/duykhanh/Downloads/20261014_050040_E02Q_visa_qualitative.zip` |
| ZIP SHA-256 | `cc4b656835129eaaf113a8b143da06ccde0aa80d4d2deb5d1f1ff7a9a630337b` |
| Outcome | Accepted |

## Artifact Integrity

The ZIP integrity test completed without an error. The structured audit found no material inconsistency. The artifact records:

- Four successful category attempts with return code 0 and no traceback, exception, or error marker in the console logs.
- Per-image records for all 752 test images: 200 `pcb2`, 201 `pcb3`, 201 `pcb4`, and 150 `fryum`, with no duplicate image path.
- A recorded heatmap count equal to the per-image count for every category. The compact ZIP intentionally excludes the full intermediate `output` directories, so the 752 heatmap files remain in the immutable Drive run directory rather than in the reviewed archive.
- Exactly 24 distinct selected cases, with six cases per category and the required balance of two low-scoring anomaly candidates, two high-scoring normal candidates, one best-localization anomaly, and one worst-localization anomaly.
- Original, ground-truth, heatmap, overlay, and panel PNG files for every selected case. All component images are `518 x 518`; all panels are `2767 x 820`.
- Non-empty ground-truth masks for all 16 selected anomalous images and empty masks for all eight selected normal images.
- Per-category score CSVs that agree with the combined case table within at most `1.11e-16`, which is CSV floating-point serialization noise.
- Exact conformance between the selected rows and the documented stable ranking procedure, including deterministic path-based tie breaking.
- Environment, command, compatibility-fix, CPU-memory-patch, and observational instrumentation records.

The four successful category subprocesses consumed 3,851.499 seconds, or 64.19 minutes. This excludes environment setup, dataset restoration, panel construction, and ZIP export.

## Frozen Configuration and Environment

E02Q retained the E02 baseline configuration: OpenAI CLIP `ViT-L-14-336`, image size 518, feature layers `[5, 11, 17, 23]`, LNAMD degrees `[1, 3, 5]`, `divide_num = 1`, batch size 4, and upstream seed 42. The intended additions were `vis = true`, `vis_type = whole_norm`, and an observational per-image score export inserted after anomaly-map and RsCIN computation but before the unchanged metric call.

The run used a Tesla T4 with Python 3.13.15, PyTorch 2.11.0+cu130, Torchvision 0.26.0+cu130, CUDA 13.0, scikit-learn 1.6.1, scikit-image 0.25.2, OpenCV 4.14.0, timm 0.9.12, ftfy 6.1.2, and openpyxl 3.1.2. As in E02, the environment record reports ftfy 6.1.2 although the guide requests 6.1.3; this remains a reproducibility note rather than evidence of a metric failure.

## Quantitative Continuity with E02

All values are percentages. Every rerun metric matches E02 within `1.42e-14` percentage points, which is floating-point noise.

| Category | Pixel AUROC | Pixel F1-max | Pixel AP | PRO | Image AUROC | Image F1-max | Image AP | Time, min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pcb2` | 97.3897 | 34.3808 | 21.8547 | 86.0489 | 92.0100 | 86.1538 | 93.5731 | 17.52 |
| `pcb3` | 98.0528 | 40.2299 | 41.0304 | 92.3246 | 92.9703 | 85.9813 | 93.2793 | 17.74 |
| `pcb4` | 98.7035 | 46.3819 | 44.7236 | 92.6549 | 98.6040 | 93.8776 | 98.6431 | 17.81 |
| `fryum` | 97.9309 | 58.1301 | 50.4290 | 94.3762 | 98.5800 | 97.4359 | 99.4038 | 11.13 |

Enabling visualization and observational score export therefore did not materially change the E02 metrics.

## Observed Effect of RsCIN

The exported pre-RsCIN score is the per-image maximum anomaly-map response, min-max normalized within each category. Recomputing the three image-level metrics from the exported labels and scores reproduces the recorded post-RsCIN metrics. This enables a paired descriptive comparison without modifying the model execution.

| Category | Initial AUROC | RsCIN AUROC | Delta | Initial F1-max | RsCIN F1-max | Delta | Initial AP | RsCIN AP | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pcb2` | 87.0600 | 92.0100 | +4.9500 | 84.8168 | 86.1538 | +1.3371 | 88.6538 | 93.5731 | +4.9193 |
| `pcb3` | 94.9109 | 92.9703 | -1.9406 | 87.3239 | 85.9813 | -1.3426 | 95.2467 | 93.2793 | -1.9673 |
| `pcb4` | 92.4059 | 98.6040 | +6.1980 | 83.7438 | 93.8776 | +10.1337 | 92.7985 | 98.6431 | +5.8446 |
| `fryum` | 97.4600 | 98.5800 | +1.1200 | 94.0594 | 97.4359 | +3.3765 | 98.8284 | 99.4038 | +0.5754 |

`pcb3` is the only selected category for which RsCIN reduces all three image-level metrics. Its score movement is also directionally unfavorable: 85 of 100 anomalous images receive lower scores, while 72 of 101 normal images receive higher scores. The mean score change is -0.1096 for anomalous images and +0.0232 for normal images. This is direct evidence that RsCIN weakens separation for this category in this run.

The other three categories improve after RsCIN, most strongly `pcb4`. The two lowest-scoring selected `pcb4` anomalies both receive increases of approximately 0.104 to 0.107, consistent with RsCIN rescuing difficult examples even though they remain the category's lowest-ranked anomalies.

This paired comparison explains category-specific behavior within E02Q. It does not by itself identify why E02 differs from the official repository table, because no matched official-environment score export is available.

## Case Selection Summary

The labels below retain the word `candidate`. They identify score extremes and are not threshold-confirmed classification errors.

| Category | Low anomaly-score candidates | High normal-score candidates | Best localization | Worst localization |
|---|---|---|---|---|
| `pcb2` | `083`, `063` | `0024`, `0011` | `020` (0.9997) | `047` (0.7486) |
| `pcb3` | `079`, `042` | `0027`, `0977` | `005` (0.9996) | `043` (0.8049) |
| `pcb4` | `046`, `047` | `0320`, `0294` | `032` (0.9997) | `061` (0.9320) |
| `fryum` | `095`, `094` | `239`, `384` | `024` (0.9999) | `010` (0.9250) |

Values in parentheses are per-image raw pixel AUROC for the selected best and worst localization cases.

## Qualitative Findings

### PCB2

The two low anomaly-score candidates contain small, localized defects, while their heatmaps respond broadly to the entire circuit board. This produces low image scores even when per-image pixel AUROC is high. The two high normal-score candidates both contain a dark mark immediately to the right of the board, and the strongest heatmap response occurs on that off-object mark. This is a plausible scene-nuisance pattern rather than evidence of an intrinsic board defect. Case `020` has a compact response aligned with the central defect, while case `047` responds most strongly to the connector and multiple components rather than the small ground-truth regions.

### PCB3

The low anomaly-score candidates involve small component-level changes that are weak inside object-wide heatmaps. Normal candidates activate strongly around LEDs, reflective components, and board structure. Case `005` contains a compact, visually salient LED-region anomaly and receives a well-aligned local peak. In contrast, case `043` has ground-truth regions extending to missing or displaced structures beside the board, while the heatmap remains concentrated on the board body. Its raw per-image pixel AUROC falls to 0.8049. Together with the unfavorable RsCIN score movement, these panels provide a concrete explanation for `pcb3`'s weaker image-level behavior.

### PCB4

`pcb4` shows the clearest image-level separation of the three PCB categories. RsCIN raises all three image-level metrics substantially. Case `032` is a strong localization example: a compact heatmap peak aligns with the circular defect on the connector. The low anomaly-score cases and case `061` demonstrate the remaining limitation: small component changes can be submerged in a diffuse board-shaped response. The normal candidates also show isolated activation on components or background marks, but their post-RsCIN ranking remains well separated from most anomalies.

### Fryum

The `fryum` panels explain how a high aggregate Pixel AUROC can coexist with coarse localization. Most heatmaps separate the snack object from the background well, but activate much of the object rather than only the defect. The two low anomaly-score candidates contain thin surface marks that receive little distinct emphasis. Case `024` has a small central defect aligned with a local peak, whereas the thin edge defect in case `010` is weak relative to the object-wide response.

These observations make the reproduced `fryum` Pixel AUROC of 97.9309 plausible. They do not prove that the official table's correct value is exactly 97.89. The official-row inconsistency documented in E02 remains a table-level inference.

## Interpretation Limits

- Per-image pixel AUROC ranks all image pixels. With a large normal background, it can be high even when the heatmap is diffuse across the object and the defect boundary is poorly localized.
- The 24 cases are deliberately selected extremes. They illustrate mechanisms and failure modes but do not estimate their prevalence.
- `whole_norm` supports a consistent display scale within a category run, but heatmap color is not a classification decision.
- A `false_negative_candidate` or `false_positive_candidate` is not a confirmed error unless a decision threshold is explicitly applied.
- The compact ZIP contains all selected visual evidence and score tables, but not the 752 intermediate heatmap files. Their existence is supported by the recorded counts and remains auditable in the Drive run directory.

## Statistical Validation

- Overall Confidence: CAUTION
- Statistical content: fixed-benchmark point metrics and paired deterministic score transformations; no p-values, confidence intervals, or inferential hypothesis tests are present.
- Multiple-comparison correction: not applicable because all four planned categories and all planned metrics are reported descriptively.
- Reproducibility method: internal artifact audit and continuity comparison with E02; no independent matched-environment rerun.
- Reproducibility verdict: CANNOT_VERIFY under the experiment-agent rerun protocol. The artifact is internally consistent and adequate for the course qualitative analysis, but this audit did not execute MuSc again.

### Fallacy Scan

- Coverage: 11/11 statistical fallacy types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Category-specific initial and RsCIN results are retained rather than replaced by one pooled score. |
| Ecological fallacy | NOTE | Aggregate metrics are not used to claim that every image behaves identically. |
| Berkson's paradox | NOTE | No correlation claim is made from a conditionally selected sample. |
| Collider bias | NOTE | No covariate adjustment is performed. |
| Base-rate neglect | CAUTION | F1-max and AP depend on each category's normal-to-anomaly composition; comparisons remain tied to the fixed VisA split. |
| Regression to the mean | NOTE | The extreme cases are selected for illustration, not used to estimate an intervention effect. |
| Survivorship bias | NOTE | All four planned categories completed and all 752 per-image records are retained. |
| Look-elsewhere effect | NOTE | All planned categories, metrics, and selected case types are reported, including unfavorable results. |
| Garden of forking paths | CAUTION | Selection rules were fixed in the guide before this run, but the experiment was not formally preregistered. |
| Correlation and causation | NOTE | The report describes score movement and visual alignment without a causal claim about the official-result deviation. |
| Reverse causality | NOTE | No directional causal relationship is estimated. |

## Decision

E02Q is accepted because all four categories completed, all 752 test images have unique score records and recorded heatmaps, every metric matches E02 within floating-point noise, the selection protocol was followed exactly, and all 24 selected cases contain complete visual evidence.

The most useful final-report findings are:

1. Small component or surface defects are often embedded in diffuse object-wide responses.
2. Off-object marks, reflective components, and repeated board structure can create high responses in normal candidates.
3. RsCIN helps `pcb2`, `pcb4`, and `fryum`, but harms `pcb3` across all three image-level metrics in this run.
4. High pixel AUROC, especially for `fryum`, does not imply tight defect-boundary localization.

The required MVTec AD and VisA quantitative and qualitative reproduction is now complete. E03 and E04 remain optional focused experiments and should be run only if they serve the final report's central analysis.
