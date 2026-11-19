# MuSc Live-Inference Demo Batch Plan

## Purpose

Prepare prerecorded backup demonstrations of the Gradio Live Inference mode.
The app may be mentioned during the presentation and opened on request, while
the videos provide a reliable fallback when a live Colab run would take too
long or encounter connectivity problems.

The proposed demonstration uses two datasets that were not part of the MVTec
AD and VisA reproduction experiments:

1. An industrial batch from MPDD to probe transfer within the intended domain.
2. A non-industrial batch from PlantVillage to observe behavior under domain
   shift.

These are qualitative demonstration batches, not benchmark evaluations or
evidence of generalization.

## Shared Input Rules

- Run each dataset as a separate batch. Never mix industrial and
  non-industrial images in one MuSc input set.
- Use exactly 40 images from one semantic category and a reasonably consistent
  acquisition setting.
- Use 32 visually normal images and 8 anomalous images so that anomalies remain
  a minority of the batch.
- Rename uploaded files with neutral identifiers that do not reveal labels.
- Keep a private mapping from each neutral filename to its source path and
  ground-truth label.
- Fix the image list and random seed before recording because MuSc scores are
  transductive and can change when the batch composition changes.
- Do not compare numeric anomaly scores across the two batches. Scores are
  relative within each uploaded set.

## Batch A: Industrial Transfer Probe

### Dataset and category

- Dataset: Metal Parts Defect Detection Dataset (MPDD)
- Category: `metal_plate`
- Intended observation: whether MuSc produces a plausible ranking and useful
  heatmaps on an unseen industrial anomaly-detection dataset.

MPDD contains more than 1,000 images of manufactured metal parts, with
anomaly-free samples, anomalous samples, and pixel-precise defect masks.

### Proposed 40-image composition

| Source group | Count |
|---|---:|
| `good` | 32 |
| `scratches` | 3 |
| `major_rust` | 3 |
| `total_rust` | 2 |
| **Total** | **40** |

Prefer normal images from the official test or validation split. If that split
does not contain 32 suitable normal images, use additional samples from
`train/good` and explicitly describe the result as a custom demo batch rather
than an MPDD benchmark run.

Suggested neutral upload names:

```text
mpdd_plate_001.png
mpdd_plate_002.png
...
mpdd_plate_040.png
```

### Interpretation boundary

A visually plausible result supports only the statement that MuSc can execute
on this selected industrial batch. One curated batch is not an external
validation result and must not be reported as MPDD benchmark performance.

### Sources and license

- Dataset repository: <https://github.com/stepanje/MPDD>
- Associated paper: Stepan Jezek, Martin Jonak, Radim Burget, Pavel Dvorak, and
  Milos Skotak, "Deep learning-based defect detection of metal parts:
  evaluating current methods in complex conditions," ICUMT, 2021.
  <https://doi.org/10.1109/ICUMT54235.2021.9631567>
- License: CC BY-NC-SA 4.0.
  <https://raw.githubusercontent.com/stepanje/MPDD/main/LICENSE>

## Batch B: Non-Industrial Domain-Shift Probe

### Dataset and category

- Dataset: PlantVillage
- Plant species: tomato
- Normal group: `Tomato___healthy`
- Anomalous group: `Tomato___Bacterial_spot`
- Intended observation: whether MuSc responds to leaf lesions, leaf shape, or
  background artifacts when used outside industrial visual inspection.

PlantVillage contains more than 54,000 healthy and diseased leaf images from
14 crop species. Use the original RGB `color` configuration rather than the
segmented configuration, whose artificial boundaries may create misleading
heatmap responses.

### Proposed 40-image composition

| Source group | Count |
|---|---:|
| `Tomato___healthy` | 32 |
| `Tomato___Bacterial_spot` | 8 |
| **Total** | **40** |

Select 40 distinct `leaf_id` values to avoid near-duplicate views of the same
physical leaf. Prefer one official split and a fixed random seed.

Suggested neutral upload names:

```text
tomato_001.jpg
tomato_002.jpg
...
tomato_040.jpg
```

The Hugging Face version can be loaded with:

```python
from datasets import load_dataset

dataset = load_dataset("mohanty/PlantVillage", "color")
```

### Interpretation boundary

PlantVillage is primarily an image-classification dataset and does not provide
the same pixel-mask evaluation protocol as MVTec AD, VisA, or MPDD. Treat the
output as a qualitative out-of-domain probe only. Do not describe the anomaly
score as a disease probability, a diagnosis, or evidence that MuSc generalizes
to agricultural disease detection.

Suggested narration:

> This is a qualitative out-of-domain probe used to observe MuSc behavior
> outside industrial inspection. It is not a plant-disease diagnosis and does
> not establish generalization.

### Sources and license

- Dataset repository: <https://github.com/spMohanty/PlantVillage-Dataset>
- Hugging Face dataset card: <https://huggingface.co/datasets/mohanty/PlantVillage>
- Associated paper: Sharada P. Mohanty, David P. Hughes, and Marcel Salathe,
  "Using Deep Learning for Image-Based Plant Disease Detection," Frontiers in
  Plant Science, 2016. <https://doi.org/10.3389/fpls.2016.01419>
- The referenced Hugging Face package identifies its license as CC BY-SA 3.0.
  Preserve attribution and follow the terms attached to the exact downloaded
  distribution.

## Recording Plan

Record one short video per batch rather than combining both runs into one long
video.

1. Show the scope and environment cards.
2. State that the input contains 40 unlabeled images from one category.
3. Upload the neutralized image batch.
4. Validate and confirm the batch.
5. Start Live Inference. Waiting time may be accelerated in editing if the
   video clearly indicates that it was shortened.
6. Show the ranked result table.
7. Inspect one strong result, one low-ranked normal image, and one imperfect or
   diffuse heatmap if available.
8. Download the CSV and ZIP outputs.
9. End with the scope limitation appropriate to the batch.

## Evidence to Retain

For each recorded run, preserve:

- The exact 40-image manifest and private label mapping.
- Dataset source, license, category, and selection seed.
- Reproduction repository commit and upstream MuSc commit.
- Colab environment versions and GPU name.
- Model initialization time and inference time.
- Downloaded CSV and ZIP output.
- The final video and a small set of fallback screenshots.
- A note describing any failed or misleading heatmaps without hiding them.

## Decision Summary

- Recommended industrial batch: MPDD `metal_plate`.
- Recommended non-industrial batch: PlantVillage tomato healthy versus
  bacterial spot.
- If there is time for only one external batch, prioritize MPDD because it
  remains within MuSc's intended industrial visual-inspection domain.
- Use the prerecorded videos as presentation fallbacks. Run the live app during
  the presentation only when requested and when time and Colab connectivity
  permit.
