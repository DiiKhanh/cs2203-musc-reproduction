# MuSc Classroom Demo Verification Record

## Verification status

This record separates completed local evidence from checks that require a
fresh Google Colab Tesla T4 runtime and external artifacts. Pending checks must
not be described as passed during the presentation.

| Gate | Status | Evidence |
|---|---|---|
| Complete local automated suite | Passed | 65 passed; 1 GPU equivalence test skipped by its explicit environment guard; 4 non-failing Gradio/Pandas warnings |
| Browser flow with fake services | Passed | Included in the complete suite; Playwright Chromium at 1366 x 768 |
| Notebook generation | Passed | Two builds produced SHA-256 `bf2592f48fe13160a83ff2e1f9241c4119a4392248585cc4e07f5c378db7509f` |
| E00 notebook regression | Passed | Rebuilt notebook compared byte-for-byte with the committed E00 notebook when the output filename matched |
| Fresh Colab T4 launch | Pending | Requires a fresh authorized Colab runtime |
| Upstream numerical equivalence | Pending | Requires T4, the pinned MuSc checkout, and MVTec AD `bottle` |
| Prepared artifact audit | Pending | Requires the accepted external prepared bundle and source CSV |
| Live 40-image timing and artifact audit | Pending | Requires a prevalidated same-category industrial batch on T4 |
| No-GPU degraded mode in deployed app | Pending | Unit behavior is covered; deployment behavior still requires manual verification |

## Frozen identifiers

- Reproduction commit tested locally: `dd4aee29e94fcbdc00f8d491d8301f60f039c06f`
- Official MuSc commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`
- Demo configuration: `configs/demo.json`
- Local verification date: 2026-10-19

The final Colab verification must record the reproduction commit printed by the
runtime. If it differs from the commit above, rerun the complete verification
instead of reusing this local evidence.

## Completed local evidence

The local environment used macOS 15.7.9 on arm64, Python 3.13.12, Gradio
6.30.0, NumPy 2.5.4, pandas 3.0.6, and Pillow 12.3.0. The browser test used
Playwright Chromium. The complete suite was run with localhost access:

```bash
/tmp/cs2203-musc-demo-venv/bin/python -m pytest -v
```

Result: 65 passed, 1 GPU equivalence test skipped, and 4 non-failing
Gradio/Pandas compatibility warnings. The browser flow covered the initial
scope card, both modes, invalid upload feedback, confirmation gating, live
progress, ranked output, gallery selection, and horizontal overflow at 1366 x
768.

The following generated UI states were inspected at original resolution:

- Default state
- Prepared result
- Validation error
- Live progress
- Live result

The screenshots are test artifacts under `test-output/ui/` and are intentionally
ignored by Git.

Notebook checks used:

```bash
python tools/build_notebook.py docs/musc_demo_colab_runbook.md notebooks/MuSc_classroom_demo.ipynb
python tools/build_notebook.py docs/E00_colab_runbook.md /tmp/E00_mvtec_bottle_smoke_test.ipynb
cmp notebooks/E00_mvtec_bottle_smoke_test.ipynb /tmp/E00_mvtec_bottle_smoke_test.ipynb
```

## Required Colab T4 verification

Use a new runtime with no state retained from an earlier session. Run
`notebooks/MuSc_classroom_demo.ipynb` from top to bottom and record the fields
below without including secrets or private Drive paths.

| Field | Required value |
|---|---|
| Verification date | Pending |
| Reproduction commit | Pending |
| Python version | Pending |
| PyTorch version | Pending |
| torchvision version | Pending |
| CUDA runtime | Pending |
| GPU name | Pending, expected Tesla T4 |
| Gradio version | Pending |
| Model initialization time | Pending |
| 40-image live inference time | Pending |
| Temporary share URL opened in a separate session | Pending, record pass/fail only |

### Upstream equivalence gate

Run the opt-in test with the complete MVTec AD `bottle` category:

```bash
RUN_MUSC_GPU_TESTS=1 \
MUSC_ROOT=/content/MuSc \
MVTEC_ROOT=/content/mvtec \
python -m pytest tests/integration/test_upstream_equivalence.py -v
```

Record the measured deltas here:

| Comparison | Required threshold | Measured |
|---|---:|---:|
| Raw image scores | max absolute difference `<= 1e-6` | Pending |
| RsCIN-adjusted scores | max absolute difference `<= 1e-6` | Pending |
| Anomaly maps | max absolute difference `<= 1e-5` | Pending |

### Input and degraded-mode gate

Verify in the deployed browser:

- One image is blocked.
- Five valid images are allowed with an experimental warning.
- More than 200 valid images are blocked.
- Corrupt and unsupported files are reported without a raw traceback.
- Exact duplicates are excluded from the valid count.
- A ZIP traversal entry is rejected without writing outside the run workspace.
- Without the prepared bundle, Live Inference remains available when the model
  is ready.
- Without CUDA, Prepared Benchmark remains available when its bundle is
  mounted.

### Prepared Benchmark gate

For at least one accepted MVTec AD run, compare every displayed metric with its
source CSV and verify the identity of the original, ground-truth, heatmap, and
overlay assets. Confirm that the page says `Prepared result - not live
inference` and that metrics and images come from the same recorded run ID.

### Live 40-image gate

Use a prevalidated unlabeled set from one industrial product category. Confirm
that all 40 images appear exactly once, adjusted-score ranks are descending,
all numeric values are finite, downloaded PNG files decode, and the CSV and ZIP
contain no ground-truth labels or unsupported benchmark metrics.

## Known limitations

- Numerical equivalence with the official path has not yet been measured on a
  Tesla T4.
- Model startup time and 40-image runtime have not yet been measured.
- Prepared results depend on an external artifact bundle that is not stored in
  Git.
- Live scores are relative to the uploaded set and are not calibrated defect
  probabilities or definitive labels.
- The workflow is scoped to industrial visual inspection and has not been
  validated for medical or dental use.
- The Gradio share link is temporary and must not be treated as a deployed
  production service.
