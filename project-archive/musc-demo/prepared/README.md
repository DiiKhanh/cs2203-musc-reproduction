# Prepared Benchmark Bundle

The classroom app reads prepared benchmark results from an external artifact
bundle. Binary artifacts stay outside Git because selected qualitative panels
contain dataset-derived images and the complete experiment outputs are large.

Set `MUSC_PREPARED_ROOT` to the mounted bundle directory before starting the
app. The repository contains only a manifest example that describes the
expected structure. Do not commit a personal Google Drive path.

The current presentation bundle has two separate provenance entries:

- E01 contains the accepted complete MVTec AD baseline metrics.
- E04Q contains the accepted selected qualitative assets for the top-1-percent
  pooling analysis.

The catalog never attaches E04Q visualizations to E01 metrics implicitly. A
cross-run comparison must name both run IDs and explain why the artifacts are
comparable.

Each run in the manifest records:

- `run_id`, `experiment_id`, dataset, category, timestamp, and source commit.
- Human-readable description and dataset summary.
- Required external artifacts that determine whether the run is available.
- Only metrics actually computed for that run.
- Optional examples and their original, ground-truth, heatmap, and overlay
  paths.

Missing optional assets disable only the corresponding panel. Missing required
artifacts disable only the affected run. The rest of the prepared catalog and
the live-inference mode remain usable.
