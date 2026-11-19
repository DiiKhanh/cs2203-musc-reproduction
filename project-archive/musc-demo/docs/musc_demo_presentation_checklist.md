# MuSc Classroom Demo Presentation Checklist

## One day before the presentation

- [ ] Use a fresh Google Colab runtime with a Tesla T4.
- [ ] Run `notebooks/MuSc_classroom_demo.ipynb` from the first cell without
  relying on cached notebook state.
- [ ] Confirm that the reproduction and upstream commit checks pass.
- [ ] Complete every pending gate in `docs/musc_demo_verification.md`.
- [ ] Mount the accepted prepared bundle and compare at least one category with
  its source CSV and assets.
- [ ] Stage one prevalidated batch of 40 images from one industrial product
  category. Do not mix categories or inspection settings.
- [ ] Download and retain the successful CSV and ZIP as an offline fallback.
- [ ] Retain the prepared screenshots needed to explain the method if live
  inference becomes unavailable.

## Thirty minutes before presenting

- [ ] Start a fresh app process with `MUSC_EAGER_INIT=1` and wait until the
  environment card reports that the model is ready.
- [ ] Open one Prepared Benchmark example and verify all four panels.
- [ ] Open the temporary Gradio share URL in an incognito or signed-out browser
  window.
- [ ] Confirm that the scope card is visible before any interaction.
- [ ] Upload the staged 40-image batch, validate it, and stop before starting a
  second live run unless that run is part of the planned demonstration.
- [ ] Keep the Colab tab open and prevent the computer from sleeping.
- [ ] Keep the successful CSV, ZIP, and prepared screenshots locally available.

## Scope statement to rehearse

State the boundary before opening the live tab:

1. **Problem:** detect and localize unusual visual regions during industrial
   surface inspection without task-specific training examples.
2. **Input:** one unlabeled set of 5-200 images from one industrial product
   category under a reasonably consistent inspection setup; 40-200 images are
   recommended.
3. **Output:** a relative anomaly score and ranking within that set, plus a
   heatmap and overlay for each image.
4. **Method:** CLIP features, LNAMD, Mutual Scoring Mechanism, RsCIN score
   adjustment, and batch-wide heatmap normalization.
5. **Non-goals:** the live app does not return definitive normal/defect labels,
   benchmark metrics, calibrated probabilities, or medical conclusions.

Clarify that MVTec AD and VisA are industrial visual anomaly detection
benchmarks. Medical or dental images are outside the validated scope and would
require a separate dataset, protocol, baselines, metrics, and domain review.

## Planned demonstration order

- [ ] Show the scope and environment cards first.
- [ ] Open Prepared Benchmark and identify it as a completed artifact, not live
  inference.
- [ ] Explain one example with original image, ground truth, heatmap, and
  overlay.
- [ ] Open Live Inference and repeat the input contract before uploading.
- [ ] Confirm that the batch is one industrial category only after inspecting
  the files supplied for the demonstration.
- [ ] Run the model once, point out relative ranking, and select examples from
  the gallery.
- [ ] Download the CSV and ZIP to demonstrate reproducible outputs.
- [ ] Return to the scope card when discussing limitations.

## If the instructor supplies a new batch

Accept it only when all answers below are yes:

- [ ] Does it contain 5-200 JPEG/PNG images, preferably at least 40?
- [ ] Do all images represent the same industrial product category?
- [ ] Are viewpoint, scale, background, and lighting reasonably consistent?
- [ ] Is the instructor comfortable treating the result as a relative ranking,
  not a definitive defect decision?

If any answer is no, explain that the batch is outside the declared input
contract. Do not broaden the claim during the demonstration. A new category is
allowed when it still satisfies the same industrial, same-category batch
contract; it does not require training, but its output remains experimental.

## Recovery actions

- **Model initialization fails:** keep Prepared Benchmark open, show the saved
  artifacts, and retain the initialization log for later diagnosis.
- **CUDA out of memory:** do not reuse the failed run directory; reduce the
  batch and retry once, or use the prepared fallback.
- **Prepared bundle is unavailable:** state that only Live Inference is
  available; never copy unverified metrics from the paper into the app.
- **Share URL fails:** relaunch the final notebook cell and test the new URL in
  an incognito window. If it still fails, use the local CSV, ZIP, and prepared
  screenshots.
- **Uploaded files violate the contract:** do not override validation or the
  confirmation step. Explain the boundary and use the staged batch.
- **Unexpected output:** describe it as a model result, not ground truth, and
  use it to discuss limitations rather than inventing a label.

## After the presentation

- [ ] Stop the notebook cell to close the temporary share link.
- [ ] Remove presentation uploads and temporary artifacts from the runtime and
  any shared location where they are no longer needed.
- [ ] Save only non-sensitive timing, version, and verification evidence.
- [ ] Record any instructor question that suggests a future experiment without
  changing the demonstrated scope retroactively.
