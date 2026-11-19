# MuSc Classroom Demo App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab-hosted Gradio app with an immediate prepared-results mode and a scientifically guarded live MuSc inference mode for one unlabeled industrial product category.

**Architecture:** A small `musc_demo` package isolates secure upload validation, generic image loading, MuSc inference, visualization/export, prepared-artifact loading, and Gradio presentation. The live engine imports the official pinned MuSc checkout at runtime, preserves CLIP, LNAMD, MSM, and RsCIN, and omits benchmark-only labels, masks, and metrics. The UI depends on typed services so unit and browser tests can use a fake engine without loading CLIP.

**Tech Stack:** Python 3, PyTorch and torchvision from Google Colab, official MuSc commit `b76b93da8bd3096a99964a96ae29d46f197a0651`, Gradio `6.30.0`, Pillow, NumPy, pandas, pytest, Playwright for browser smoke tests.

**Spec:** [MuSc classroom demo app design](../../plans/2026-10-18-musc-demo-app-design.md)

**Implementation repository:** `https://github.com/DiiKhanh/cs2203-musc-reproduction`

## Global Constraints

- Run the final app on Google Colab with a Tesla T4 GPU and expose it through a temporary Gradio share link.
- Keep PyTorch and torchvision from the Colab runtime. Do not install replacement CUDA wheels.
- Pin the official MuSc checkout to `b76b93da8bd3096a99964a96ae29d46f197a0651`.
- Preserve the upstream CLIP, LNAMD, MSM, and RsCIN algorithm with `ViT-L-14-336`, image size `518`, feature layers `[5, 11, 17, 23]`, LNAMD degrees `[1, 3, 5]`, MSM interval `[0, 0.3]`, batch size `4`, and custom-category RsCIN neighbors `[1, 2, 3]`.
- Block fewer than 5 and more than 200 valid images, warn for 5 to 39, and recommend 40 to 200.
- Require explicit confirmation that the batch represents one industrial product category under a reasonably consistent inspection setting.
- Never display a definitive normal/defect label or benchmark metric for an unlabeled custom upload.
- Use one batch-wide min-max normalization for live heatmaps.
- If all raw scores are equal, preserve them as adjusted scores and warn instead of returning `NaN`.
- Use Vietnamese interface copy while retaining the technical terms `category`, `anomaly score`, `heatmap`, `ground truth`, and `live inference` in English.
- Do not redistribute datasets, model weights, secrets, or private Google Drive paths in the repository.
- Use Conventional Commits and do not add an agent co-author.

## Review Focus

- A ZIP containing `../` traversal entries or symbolic links must be rejected without writing outside the run workspace. Covered by Task 1 validation tests.
- Grayscale, RGBA, CMYK, and EXIF-rotated images must become correctly oriented RGB tensors. Covered by Task 2 adapter tests.
- Files with identical basenames in different folders must receive stable unique image IDs and must not overwrite one another. Covered by Tasks 1, 2, and 4 tests.
- Constant anomaly maps and constant raw scores must produce finite outputs, zero-valued heatmaps, preserved adjusted scores, and visible warnings. Covered by Tasks 3 and 4 tests.
- Missing, corrupt, or partially available prepared artifacts must disable only the affected category or asset, not crash the whole app. Covered by Tasks 5 and 6 tests.

---

### Task 1: Secure Batch Validation and Shared Data Contracts

**Files:**
- Create: `musc_demo/__init__.py`
- Create: `musc_demo/models.py`
- Create: `musc_demo/validation.py`
- Create: `tests/test_validation.py`
- Create: `requirements-demo.txt`
- Create: `requirements-test.txt`

**Interfaces:**
- Consumes: uploaded file paths supplied by Gradio.
- Produces: `ValidationIssue`, `ValidatedImage`, `BatchValidationReport`, `validate_uploads(inputs: Sequence[Path], workspace: Path) -> BatchValidationReport`, and `confirmation_allows_run(report: BatchValidationReport, confirmed: bool) -> bool`.

- [ ] **Step 1: Write failing validation tests**

Add tests named:

- `test_rejects_one_image`
- `test_warns_for_five_images`
- `test_accepts_forty_images`
- `test_rejects_more_than_two_hundred_images`
- `test_reports_corrupt_unsupported_and_exact_duplicate_files`
- `test_rejects_zip_path_traversal_and_symlink_entries`
- `test_assigns_unique_ids_to_duplicate_basenames`
- `test_requires_user_confirmation_after_valid_batch`

Assert the exact batch boundaries, issue codes, stable image IDs, and `can_run` behavior from the spec.

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `python -m pytest tests/test_validation.py -v`
Expected: FAIL during import because `musc_demo.validation` does not exist.

- [ ] **Step 3: Define the shared immutable data models**

In `musc_demo/models.py`, implement:

```python
Severity = Literal["info", "warning", "error"]
BatchStatus = Literal["blocked", "experimental", "recommended"]

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message_vi: str
    filenames: tuple[str, ...] = ()

@dataclass(frozen=True)
class ValidatedImage:
    image_id: str
    source_name: str
    path: Path
    sha256: str

@dataclass(frozen=True)
class BatchValidationReport:
    images: tuple[ValidatedImage, ...]
    issues: tuple[ValidationIssue, ...]
    status: BatchStatus
    can_run: bool
```

Use an SHA-256 prefix plus a collision-safe ordinal for `image_id`; retain the original relative name separately.

- [ ] **Step 4: Implement secure upload validation**

Implement `validate_uploads()` and `confirmation_allows_run()` in `musc_demo/validation.py`. Accept `.jpg`, `.jpeg`, and `.png` case-insensitively, inspect images with Pillow, hash source file bytes for exact duplicates, keep the first occurrence, exclude later byte-identical copies from the valid count, and extract ZIP members only after resolving each target under `workspace` and rejecting traversal or symlink entries.

- [ ] **Step 5: Add the pinned demo and test dependencies**

Set `requirements-demo.txt` to include `-r requirements-colab.txt`, `gradio==6.30.0`, `Pillow>=11,<14`, and `pandas>=2.2,<4`. Set `requirements-test.txt` to include `-r requirements-demo.txt`, `pytest>=8,<10`, and `playwright==1.64.0`.

- [ ] **Step 6: Run the validation tests**

Run: `python -m pytest tests/test_validation.py -v`
Expected: all Task 1 tests PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add musc_demo requirements-demo.txt requirements-test.txt tests/test_validation.py
git commit -m "feat(demo): validate uploaded image batches"
```

### Task 2: Generic Unlabeled Image Adapter

**Files:**
- Create: `musc_demo/dataset_adapter.py`
- Create: `tests/test_dataset_adapter.py`

**Interfaces:**
- Consumes: `tuple[ValidatedImage, ...]` and the CLIP preprocessing callable returned by the upstream model.
- Produces: `UnlabeledImageDataset(images: Sequence[ValidatedImage], transform: Callable)`, where `__getitem__(index: int) -> dict[str, object]` returns `image`, `image_id`, `source_name`, and `path`.

- [ ] **Step 1: Write failing adapter tests**

Add tests named:

- `test_dataset_preserves_stable_order_and_metadata`
- `test_converts_grayscale_rgba_and_cmyk_to_rgb`
- `test_applies_exif_orientation_before_transform`
- `test_duplicate_basenames_remain_distinct`
- `test_transform_receives_a_pillow_rgb_image`

Use small generated image fixtures and a transform spy. Do not require torch or the upstream checkout for these tests.

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `python -m pytest tests/test_dataset_adapter.py -v`
Expected: FAIL during import because `musc_demo.dataset_adapter` does not exist.

- [ ] **Step 3: Implement `UnlabeledImageDataset`**

Open each image lazily, apply `ImageOps.exif_transpose()`, convert to RGB, call the injected transform, and return stable metadata. Raise a typed `ImageDecodeError(image_id: str, source_name: str)` if a file changes or becomes unreadable after validation.

- [ ] **Step 4: Run adapter and validation tests**

Run: `python -m pytest tests/test_dataset_adapter.py tests/test_validation.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add musc_demo/dataset_adapter.py tests/test_dataset_adapter.py
git commit -m "feat(demo): load generic unlabeled image sets"
```

### Task 3: MuSc Live Inference Engine and Upstream Equivalence Harness

**Files:**
- Create: `musc_demo/upstream.py`
- Create: `musc_demo/inference.py`
- Create: `tests/test_inference.py`
- Create: `tests/integration/test_upstream_equivalence.py`
- Create: `configs/demo.json`

**Interfaces:**
- Consumes: `UnlabeledImageDataset`, pinned upstream root, demo configuration, and `ProgressCallback = Callable[[str, float, str], None]`.
- Produces: `InferenceItem`, `InferenceResult`, `MuScInferenceEngine.initialize() -> None`, `MuScInferenceEngine.run(dataset: UnlabeledImageDataset, progress: ProgressCallback) -> InferenceResult`, and `get_cached_engine(musc_root: Path, config_path: Path) -> MuScInferenceEngine`.

- [ ] **Step 1: Write failing engine tests with injected fakes**

Add tests named:

- `test_run_emits_inference_stages_in_order`
- `test_run_returns_raw_adjusted_scores_and_maps_in_input_order`
- `test_custom_category_uses_rscin_neighbors_one_two_three`
- `test_constant_scores_preserve_adjusted_scores_and_emit_warning`
- `test_result_contains_no_ground_truth_labels_or_metrics`
- `test_cached_engine_initializes_the_backbone_once`
- `test_initialize_without_cuda_reports_gpu_unavailable`
- `test_cuda_oom_clears_cache_and_returns_recovery_code`
- `test_failure_identifies_the_current_pipeline_stage`

Inject fake backbone, LNAMD, MSM, and RsCIN callables. Assert finite NumPy outputs and exact inference stage names: `normalize`, `clip`, `lnamd`, `msm`, and `rscin`. Task 4 owns the later `render` stage.

- [ ] **Step 2: Run the unit tests and verify the expected failure**

Run: `python -m pytest tests/test_inference.py -v`
Expected: FAIL during import because `musc_demo.inference` does not exist.

- [ ] **Step 3: Implement the pinned upstream bridge**

In `musc_demo/upstream.py`, implement `load_upstream_modules(musc_root: Path) -> UpstreamModules`. Verify the checkout commit with `git -C <root> rev-parse HEAD`, prepend the checkout to `sys.path`, and import the upstream OpenCLIP factory, `LNAMD`, `MSM`, and `RsCIN`. Raise `UpstreamVersionError` before model loading when the commit differs.

- [ ] **Step 4: Implement model initialization and caching**

Implement `MuScInferenceEngine.initialize()` and `get_cached_engine()`. Load the CLIP backbone once, move it to the selected CUDA device, retain its preprocessing transform, and expose a read-only state of `uninitialized`, `ready`, or `failed`.

- [ ] **Step 5: Implement unlabeled MuSc inference**

Port only the inference stages from upstream `MuSc.make_category_data()` into `MuScInferenceEngine.run()`: batched CLIP feature extraction, LNAMD for `[1, 3, 5]`, MSM for `[0, 0.3]`, layer and degree averaging, bilinear resize to `518 x 518`, maximum raw image score, and RsCIN with `[1, 2, 3]`. Do not import dataset loaders, masks, labels, metric computation, Excel output, or upstream visualization.

- [ ] **Step 6: Add the constant-score guard and typed failures**

Before calling upstream RsCIN, detect `max(raw_scores) - min(raw_scores) <= np.finfo(np.float32).eps`. Return a copy of the raw scores plus warning code `RSCIN_CONSTANT_SCORES`. Wrap each stage failure in `InferenceStageError(stage: str, code: str, message: str)` while preserving the original exception for the Colab log. Map CUDA out-of-memory failures to code `CUDA_OOM`, release failed-run tensors, call `torch.cuda.empty_cache()`, and provide the UI recovery action to reduce the batch.

- [ ] **Step 7: Run inference unit tests**

Run: `python -m pytest tests/test_inference.py -v`
Expected: all unit tests PASS without a GPU or model download.

- [ ] **Step 8: Write the opt-in upstream equivalence test**

Implement `test_demo_engine_matches_upstream_on_mvtec_bottle()` behind environment variables `MUSC_ROOT`, `MVTEC_ROOT`, and `RUN_MUSC_GPU_TESTS=1`. Instrument the upstream path in the test to capture the arrays passed into metric computation, derive raw scores from the captured anomaly maps, and run the upstream and demo engines sequentially so only one CLIP model occupies the GPU. Use the same complete MVTec AD `bottle` test set. Assert identical image ordering, maximum absolute raw-score difference `<= 1e-6`, adjusted-score difference `<= 1e-6`, and anomaly-map difference `<= 1e-5`.

- [ ] **Step 9: Run the Colab GPU equivalence test**

Run: `RUN_MUSC_GPU_TESTS=1 MUSC_ROOT=/content/MuSc MVTEC_ROOT=/content/mvtec python -m pytest tests/integration/test_upstream_equivalence.py -v`
Expected: PASS on a Tesla T4. Preserve the command, environment versions, and numeric deltas in the demo verification log.

- [ ] **Step 10: Commit Task 3**

```bash
git add configs/demo.json musc_demo/inference.py musc_demo/upstream.py tests/test_inference.py tests/integration/test_upstream_equivalence.py
git commit -m "feat(demo): run MuSc on unlabeled image batches"
```

### Task 4: Batch-Wide Visualization and Download Exports

**Files:**
- Create: `musc_demo/visualization.py`
- Create: `musc_demo/export.py`
- Create: `tests/test_visualization.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Consumes: `InferenceResult`, original RGB images, and a per-run output directory.
- Produces: `normalize_batch_maps(maps: np.ndarray) -> NormalizedMaps`, `render_result_images(result: InferenceResult, output_dir: Path, progress: ProgressCallback) -> tuple[RenderedItem, ...]`, and `export_run(result: InferenceResult, rendered: Sequence[RenderedItem], output_dir: Path) -> ExportBundle`.

- [ ] **Step 1: Write failing visualization and export tests**

Add tests named:

- `test_batch_wide_normalization_uses_one_global_range`
- `test_constant_maps_render_as_zero_and_warn`
- `test_render_emits_the_render_progress_stage`
- `test_overlay_preserves_original_dimensions`
- `test_unique_ids_prevent_filename_overwrite`
- `test_export_csv_matches_adjusted_score_order`
- `test_export_zip_contains_manifest_csv_heatmaps_and_overlays`
- `test_export_manifest_records_config_commit_and_normalization_policy`
- `test_export_contains_no_labels_or_metrics_for_custom_upload`

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `python -m pytest tests/test_visualization.py tests/test_export.py -v`
Expected: FAIL during import because the modules do not exist.

- [ ] **Step 3: Implement batch-wide heatmaps and overlays**

Use one min and max over the complete run. Return zero-valued normalized maps plus warning `CONSTANT_ANOMALY_MAPS` when the range is zero. Render PNG heatmaps and alpha-blended overlays at each original image size, keyed by `image_id` rather than basename.

- [ ] **Step 4: Implement deterministic CSV and ZIP export**

Sort by adjusted score descending with `image_id` as the tie-breaker. Export `results.csv`, `run-manifest.json`, `heatmaps/<image_id>.png`, and `overlays/<image_id>.png`. Record source names, raw and adjusted scores, rank, configuration, MuSc commit, RsCIN neighbors, warnings, and `batch_global_minmax` normalization.

- [ ] **Step 5: Run visualization and export tests**

Run: `python -m pytest tests/test_visualization.py tests/test_export.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add musc_demo/visualization.py musc_demo/export.py tests/test_visualization.py tests/test_export.py
git commit -m "feat(demo): render and export anomaly results"
```

### Task 5: Prepared Benchmark Catalog

**Files:**
- Create: `musc_demo/prepared_results.py`
- Create: `prepared/README.md`
- Create: `prepared/manifest.example.json`
- Create: `tests/test_prepared_results.py`
- Create: `tests/fixtures/prepared/manifest.json`

**Interfaces:**
- Consumes: a manifest path and an external prepared-artifact root, usually mounted from Google Drive.
- Produces: `PreparedCatalog.load(manifest_path: Path, artifact_root: Path) -> PreparedCatalog`, `datasets() -> tuple[str, ...]`, `categories(dataset: str) -> tuple[str, ...]`, `runs(dataset: str, category: str) -> tuple[str, ...]`, and `get_example(run_id: str, image_id: str) -> PreparedExample`.

- [ ] **Step 1: Write failing prepared-catalog tests**

Add tests named:

- `test_catalog_lists_only_manifest_categories`
- `test_catalog_keeps_metrics_scores_and_assets_under_one_run_id`
- `test_catalog_never_implicitly_combines_e01_metrics_with_e04q_assets`
- `test_metrics_preserve_recorded_names_and_values`
- `test_example_resolves_original_ground_truth_heatmap_and_overlay`
- `test_missing_optional_asset_marks_only_that_asset_unavailable`
- `test_missing_category_artifacts_disable_only_that_category`
- `test_corrupt_manifest_reports_a_catalog_error_without_crashing_app_state`
- `test_prepared_entries_are_labeled_not_live_inference`

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `python -m pytest tests/test_prepared_results.py -v`
Expected: FAIL during import because `musc_demo.prepared_results` does not exist.

- [ ] **Step 3: Implement the manifest-driven catalog**

Use standard-library JSON parsing and dataclasses. Resolve all artifact paths under `artifact_root`, reject traversal, validate metric values as finite numbers, and retain unavailable optional assets as `None` with a user-visible notice.

- [ ] **Step 4: Document the external prepared bundle**

In `prepared/README.md`, describe a bundle containing the accepted E01 MVTec metric CSV and the accepted E04Q selected qualitative assets as separate run entries. Keep the bundle outside Git because it includes dataset-derived images. Document `MUSC_PREPARED_ROOT` and the manifest field meanings without committing private Drive paths.

- [ ] **Step 5: Run prepared-catalog tests**

Run: `python -m pytest tests/test_prepared_results.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add musc_demo/prepared_results.py prepared tests/test_prepared_results.py tests/fixtures/prepared
git commit -m "feat(demo): load prepared benchmark artifacts"
```

### Task 6: Gradio Research-First Interface

**Files:**
- Create: `musc_demo/presenters.py`
- Create: `musc_demo/app.py`
- Create: `musc_demo/theme.css`
- Create: `tests/test_presenters.py`
- Create: `tests/test_app.py`
- Create: `tests/e2e/test_app_browser.py`

**Interfaces:**
- Consumes: `BatchValidationReport`, `PreparedCatalog`, `MuScInferenceEngine`, rendered results, and export bundles.
- Produces: `DemoServices(engine: MuScInferenceEngine, prepared: PreparedCatalog, workspace_root: Path)`, `build_app(services: DemoServices) -> gr.Blocks`, `launch_app() -> None`, and pure presenter functions that return Vietnamese UI state without importing model code.

- [ ] **Step 1: Write failing presenter and app-structure tests**

Add tests named:

- `test_scope_card_states_input_output_and_non_goals`
- `test_environment_card_reports_gpu_model_and_prepared_status`
- `test_run_button_requires_valid_batch_confirmation_and_ready_model`
- `test_experimental_batch_shows_red_warning`
- `test_custom_results_do_not_contain_labels_metrics_or_ground_truth`
- `test_prepared_result_has_explicit_not_live_badge`
- `test_gpu_failure_keeps_prepared_tab_available`
- `test_pipeline_error_hides_raw_traceback_and_names_failed_stage`
- `test_build_app_contains_prepared_and_live_tabs`
- `test_each_live_run_uses_an_isolated_workspace`
- `test_gpu_queue_allows_only_one_inference_job`

- [ ] **Step 2: Run presenter and app tests and verify the expected failure**

Run: `python -m pytest tests/test_presenters.py tests/test_app.py -v`
Expected: FAIL during import because the UI modules do not exist.

- [ ] **Step 3: Implement pure UI presenters**

Implement functions that map validation, runtime, prepared-catalog, progress, result, and error states to Vietnamese Markdown, component visibility, interactivity, gallery rows, and download paths. Keep all scope text in one constant so it is identical in the initial and post-error states.

- [ ] **Step 4: Build the one-page Gradio app**

Create the persistent scope and environment cards, `Kết quả chuẩn bị sẵn` tab, and `Thử nghiệm tập ảnh mới` tab. Wire dataset, category, and experiment/run selectors for prepared artifacts. Wire upload validation, attestation, model initialization, a single-concurrency GPU queue, progress stages, ranked gallery selection, prepared four-panel detail, live three-panel detail, and CSV/ZIP downloads. Allocate one child directory under `workspace_root` per live run and never reuse a failed run directory. Use `gradio==6.30.0` APIs only.

- [ ] **Step 5: Apply projector-oriented styling**

In `theme.css`, set readable minimum font sizes, restrained status colors, two-column desktop sections that collapse on narrow screens, fixed image aspect behavior, and no horizontal overflow at `1366 x 768` and `1920 x 1080`.

- [ ] **Step 6: Run unit-level UI tests**

Run: `python -m pytest tests/test_presenters.py tests/test_app.py -v`
Expected: all tests PASS without loading CLIP.

- [ ] **Step 7: Write and run a browser smoke test with fake services**

Launch `build_app()` on localhost with a fake prepared catalog and fake inference engine. In `test_app_browser.py`, use Playwright to assert the scope card is visible before interaction, both tabs open, a one-image upload is blocked, a valid fake batch requires confirmation, the prepared example renders, and no horizontal scroll exists at `1366 x 768`.

Run: `python -m playwright install chromium && python -m pytest tests/e2e/test_app_browser.py -v`
Expected: PASS and save screenshots for the default, prepared-result, validation-error, live-progress, and live-result states under `test-output/ui/`.

- [ ] **Step 8: Visually inspect the screenshots**

Open every screenshot at original resolution. Fix clipped text, inconsistent alignment, weak contrast, oversized empty areas, distorted images, or controls that are hard to identify. Rerun Step 7 until the inspected screenshots are presentation-ready.

- [ ] **Step 9: Commit Task 6**

```bash
git add musc_demo/app.py musc_demo/presenters.py musc_demo/theme.css tests/test_presenters.py tests/test_app.py tests/e2e/test_app_browser.py
git commit -m "feat(demo): add the Gradio classroom interface"
```

### Task 7: Colab Launch Notebook and Operational Runbook

**Files:**
- Create: `docs/musc_demo_colab_runbook.md`
- Create: `notebooks/MuSc_classroom_demo.ipynb`
- Modify: `tools/build_notebook.py`
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `tests/test_build_notebook.py`

**Interfaces:**
- Consumes: the private reproduction repository, pinned upstream repository, optional Google Drive prepared bundle, and a Colab T4 runtime.
- Produces: a generated Colab notebook that installs dependencies, verifies the upstream commit, mounts optional prepared artifacts, initializes the model, launches the app, and prints the temporary share URL.

- [ ] **Step 1: Write failing notebook-builder tests**

Add tests named:

- `test_builder_uses_output_stem_as_colab_notebook_name`
- `test_demo_runbook_produces_markdown_and_executable_cells`
- `test_demo_notebook_contains_commit_verification_before_app_launch`
- `test_demo_notebook_does_not_embed_credentials_or_private_drive_paths`

- [ ] **Step 2: Run builder tests and verify the expected failure**

Run: `python -m pytest tests/test_build_notebook.py -v`
Expected: at least the notebook-name and demo-runbook tests FAIL.

- [ ] **Step 3: Generalize the existing notebook builder**

Change `convert(markdown: str)` to `convert(markdown: str, notebook_name: str)` and have `main()` pass `args.output.name`. Preserve the existing E00 notebook output except for deriving its already matching filename.

- [ ] **Step 4: Write the Colab runbook**

Use fenced Python and Bash cells to:

1. Confirm a CUDA GPU and print environment versions.
2. Clone or update the private reproduction repository.
3. Clone MuSc and verify the exact pinned commit.
4. Apply the documented import compatibility patch.
5. Install `requirements-demo.txt` without replacing PyTorch or torchvision.
6. Optionally mount Google Drive and set `MUSC_PREPARED_ROOT` without embedding a user-specific path.
7. Initialize `get_cached_engine()` before class starts.
8. Launch `python -m musc_demo.app` with Gradio sharing enabled.
9. Print the fallback instructions when GPU or prepared artifacts are unavailable.

- [ ] **Step 5: Generate the notebook and verify determinism**

Run: `python tools/build_notebook.py docs/musc_demo_colab_runbook.md notebooks/MuSc_classroom_demo.ipynb` twice.
Expected: the second run produces no Git diff.

- [ ] **Step 6: Update repository documentation and ignores**

Add the app scope, input contract, Colab badge, launch steps, prepared-bundle requirement, and medical-domain non-goal to `README.md`. Ignore live uploads, temporary runs, exports, model weights, and prepared binary artifacts while retaining example manifests.

- [ ] **Step 7: Run notebook and regression tests**

Run: `python -m pytest tests/test_build_notebook.py -v && python tools/build_notebook.py docs/E00_colab_runbook.md /tmp/E00.ipynb && cmp notebooks/E00_mvtec_bottle_smoke_test.ipynb /tmp/E00.ipynb`
Expected: tests PASS and `cmp` exits `0`.

- [ ] **Step 8: Commit Task 7**

```bash
git add .gitignore README.md docs/musc_demo_colab_runbook.md notebooks/MuSc_classroom_demo.ipynb tools/build_notebook.py tests/test_build_notebook.py
git commit -m "docs(demo): add the Colab launch workflow"
```

### Task 8: Full Colab E2E Verification and Presentation Handoff

**Files:**
- Create: `docs/musc_demo_verification.md`
- Create: `docs/musc_demo_presentation_checklist.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the complete app, authorized MVTec prepared artifacts, one prevalidated 40-image industrial batch, and a fresh Tesla T4 Colab runtime.
- Produces: recorded verification evidence and a presentation-day checklist.

- [ ] **Step 1: Run the complete automated test suite locally**

Run: `python -m pytest -v`
Expected: all non-GPU tests PASS with GPU integration tests explicitly SKIPPED unless enabled.

- [ ] **Step 2: Execute the generated notebook from a fresh T4 runtime**

Run every cell in order without relying on state from an earlier session.
Expected: exact commit verification passes, the model initializes once, the app launches, and the share URL opens in a separate browser session.

- [ ] **Step 3: Reproduce the invalid-input E2E cases**

Through the browser, test one image, five images, more than 200 files, a corrupt image, an unsupported file, an exact duplicate, and a ZIP traversal fixture.
Expected: each state matches the batch contract and no raw traceback appears.

- [ ] **Step 4: Verify Prepared Benchmark mode**

Load at least one MVTec AD category from the accepted prepared bundle. Compare every displayed metric with its source CSV and inspect original, ground-truth, heatmap, and overlay assets.
Expected: values and asset identities match exactly, and the page displays `Prepared result - not live inference`.

- [ ] **Step 5: Verify live inference on 40 images**

Run a prevalidated same-category industrial batch, observe every progress stage, inspect the ranked gallery, and download CSV and ZIP.
Expected: all 40 images appear once, ranks match adjusted-score order, all values are finite, generated assets decode, and no unsupported metrics or labels appear.

- [ ] **Step 6: Verify degraded operation**

Restart without a GPU and once without the prepared bundle.
Expected: the unavailable capability is clearly disabled while the independent mode remains usable.

- [ ] **Step 7: Record evidence and presentation timings**

In `docs/musc_demo_verification.md`, record the repository commit, upstream commit, Colab environment, test commands and outcomes, equivalence deltas, model initialization time, 40-image inference time, screenshots inspected, and any remaining limitations. Do not commit private paths or credentials.

- [ ] **Step 8: Write the presentation-day checklist**

Include preloading the model, opening the prepared example, checking the share URL from an incognito window, staging a validated live batch, retaining CSV/ZIP fallback outputs, and rehearsing the exact input/output/non-goal explanation.

- [ ] **Step 9: Run final repository checks**

Run: `git diff --check && python -m pytest -v && git status --short`
Expected: no whitespace errors, all non-GPU tests PASS, GPU tests show their intended pass/skip state, and only planned files remain modified.

- [ ] **Step 10: Commit Task 8**

```bash
git add README.md docs/musc_demo_verification.md docs/musc_demo_presentation_checklist.md
git commit -m "test(demo): verify the classroom workflow"
```
