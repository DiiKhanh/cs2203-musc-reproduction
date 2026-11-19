# MuSc E04Q MVTec AD Pooling Qualitative Result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-17
- Verification Status: VERIFIED
- Version Label: validation_v1

## Run Identity

| Field | Value |
|---|---|
| Experiment | E04Q qualitative review of top-1-percent pooling |
| Dataset | MVTec AD `carpet`, `transistor`, and `screw` |
| MuSc commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Run ID | `20261017_082619_E04Q_mvtec_pooling_qualitative` |
| Source E04 run | `20261016_102025_E04_mvtec_top1_pooling` |
| Run directory | `MyDrive/CS2203_MuSc/runs/20261017_082619_E04Q_mvtec_pooling_qualitative/` |
| Reviewed archive | [E04Q qualitative ZIP](artifacts/e04q/20261017_082619_E04Q_mvtec_pooling_qualitative.zip) |
| ZIP SHA-256 | `66c09ec2988334e37a2ada4834515ff90fda733e46efd0ee8e1bd8f0ac3c70a6` |
| Outcome | Accepted as the qualitative explanation of the E04 trade-off; the E04 hypothesis remains unsupported overall |

## Artifact Integrity and Continuity

The compact ZIP passed `testzip()` with no damaged member. The review also confirmed:

- All three categories completed successfully on the first attempt with return code `0` on a Tesla T4.
- The pinned MuSc commit, E04 source patch, configuration, source E04 run ID, and fixed case plan are present.
- The E04 source patch is byte-identical to the patch preserved in the accepted E04 archive.
- The category manifests record 117 `carpet`, 100 `transistor`, and 160 `screw` heatmaps, for 377 total.
- The compact archive intentionally excludes the full heatmap directories. It contains the five required PNG assets for each of the nine fixed cases: original, ground truth, heatmap, overlay, and panel.
- All 45 selected PNG assets decode successfully, and all nine panels have dimensions `2781 x 855`.
- The nine selected rows and their score values match the accepted E04 per-image table.
- Direct comparison with E04 gives a maximum metric delta of `0.0` and a maximum per-image score delta of `1.11e-16`. The E04Q manifest reports `1.42e-14` and `0.0`, respectively, because the Colab continuity check used pandas floating-point parsing. Both are machine-precision effects and are far below the preregistered limits.

The three subprocesses recorded 26.87 minutes of inference time.

## Reproducibility Verdict

- Method: deterministic subset rerun using the archived E04 source patch and the same dataset, backbone, layers, image size, LNAMD degrees, pooling fraction, and score variants.
- Scope: the three qualitative categories, not a second rerun of all 15 E04 categories.
- Verdict: `REPRODUCIBLE` for the E04Q subset. All four per-image score variants agree with E04 within floating-point precision.

## Fixed Case Review

Each transition below uses the separately optimized E04 F1-max threshold for its own score variant. Therefore, a lower absolute top-1-percent score can still become a false positive when its variant-specific threshold is also lower.

| Category and case | Raw transition | RsCIN transition | Visual observation |
|---|---|---|---|
| `carpet/good/019.png` | FP to TN | FP to TN | The normal texture has diffuse low-level activation and a stronger border response. Top-1-percent pooling suppresses the maximum-driven false positive. |
| `carpet/metal_contamination/013.png` | TP to TP | TP to TP | The main hotspot overlaps the small contamination mask. Pooling lowers its rank but preserves the correct decision. |
| `carpet/color/013.png` | TP to TP | TP to TP | The strongest response lies at the left boundary and overlaps the color-defect mask, with some spatial spread. The anomaly remains detected after its rank drops. |
| `transistor/good/022.png` | FP to TN | TN to TN | The map responds diffusely to normal object structure and image boundaries. Top-1-percent pooling removes the raw false positive. |
| `transistor/good/001.png` | TN to FP | TN to FP | Broad activation covers normal leads and parts of the package. Relative to the lower top-1-percent thresholds, this becomes a new false positive in both contexts. |
| `transistor/cut_lead/001.png` | TP to FN | TP to TP | The response is diffuse, with a local increase near the damaged right lead. The raw top-1-percent score loses enough rank to cross its threshold. |
| `screw/good/008.png` | TN to FP | FP to FP | The whole normal screw is hot rather than a small isolated region. The raw score remains high relative to its top-1-percent threshold. |
| `screw/good/003.png` | TN to FP | TN to FP | Object-wide activation follows the complete normal screw. Top-1-percent pooling does not isolate a nuisance pixel and creates false positives in both contexts. |
| `screw/scratch_neck/000.png` | TP to TP | TP to TP | The map highlights most of the screw, with a local maximum near the neck defect. The case stays positive, but its anomaly rank drops by `0.4750` raw and `0.6125` after RsCIN. |

## Qualitative Interpretation

The positive examples provide partial support for the proposed mechanism. Maximum pooling can be driven by a stronger local or boundary response, and top-1-percent pooling can suppress that response enough to resolve a false positive. However, the resolved examples are not evidence that every false positive comes from one isolated noisy patch because their maps also contain broad, structured activation.

The `screw` counterexamples explain why the same aggregation is not consistently beneficial. MuSc assigns high responses across much of the object, including normal screws. Averaging the top region therefore does not simply remove one spurious peak. It changes score ordering and the optimized operating threshold, producing new false positives while substantially lowering the rank of a real `scratch_neck` anomaly.

Maximum and top-1-percent pooling use the same anomaly map, and RsCIN only changes the image-level score. The panels explain the spatial context behind the scores, but they do not show a localization improvement caused by the pooling change.

## Statistical and Methodological Warnings

- Overall confidence remains `CAUTION` for generalization. E04Q explains selected E04 behavior but does not overturn the negative full-benchmark result.
- The nine cases are diagnostic examples selected from E04 score changes. They are not a representative sample for estimating how often each mechanism occurs.
- The cases were fixed before E04Q heatmap review, which limits visual cherry-picking, but selection still targets extreme or decision-changing examples.
- Each variant uses its own post hoc F1-max threshold. Decision transitions are descriptive benchmark operating points, not deployment error rates.
- `whole_norm` visualizations should be interpreted spatially within each panel. Color intensity is not a calibrated score comparison across different images.
- E04Q introduces no new pixel-level metric or independent statistical test.

## Fallacy Scan

Coverage: 11/11 statistical fallacy types checked.

| Fallacy | Assessment |
|---|---|
| Simpson's paradox | The conclusion remains tied to the full E04 macro and category tables; the three selected categories are not substituted for the benchmark aggregate. |
| Ecological fallacy | Image-level transitions are described from per-image scores rather than inferred only from category means. |
| Berkson's paradox | CAUTION: cases are deliberately selected for diagnostic roles, so their visual pattern must not be treated as representative prevalence. |
| Collider bias | No additional covariate is conditioned on in the scoring comparison. |
| Base-rate neglect | F1-max transitions use prevalence-sensitive, variant-specific thresholds and are reported as descriptive only. |
| Regression to the mean | Not applicable because E04Q is a deterministic rerun, not a pre-post intervention selected on noisy measurements. |
| Survivorship bias | All three planned categories and all nine fixed cases are present; there are no failed or omitted attempts. |
| Look-elsewhere effect | The case set was fixed before reviewing E04Q heatmaps. No new significance claim is derived from the panels. |
| Garden of forking paths | Categories, cases, pooling fraction, variants, and continuity rules were frozen before the qualitative rerun. |
| Correlation versus causation | Visual alignment is used only as a mechanism explanation. It is not treated as proof that pooling caused better localization. |
| Reverse causality | Not applicable to this controlled score-aggregation comparison. |

## Decision

E04Q is accepted as complete qualitative evidence for the focused extension. It confirms that top-1-percent pooling can suppress some maximum-driven normal responses, but it also shows that broad object-wide activation can make the alternative aggregation harmful. The qualitative evidence is consistent with the E04 quantitative decision: top-1-percent mean pooling is not an overall improvement over maximum pooling under the frozen MVTec AD configuration.

The report and presentation should frame E04 as a negative but informative controlled experiment. Use a small number of the fixed panels to illustrate the benefit-counterexample pair, and retain the full quantitative result as the basis for the conclusion.
