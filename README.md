# CardioNet

CardioNet is a Python cardiac MRI analysis project built around a local
fine-tuned CineMA short-axis segmentation model plus downstream geometric,
wall-thickness, radial-strain, and endocardial-excursion code.

The final pipeline is intended to allow the user to run segmentation of SAX
cineMRI images using any arbitrary segmentation model, and intelligently
extract geometric and physiological features in accordance with AHA standards
of cardiac segmentation. Per segment features will include wall thickness,
normalised wall thickness, timing, and range of motion, which will be both
graphically and numerically presented to the end user. Finally, an optional
classifier will be created to produce clinician-ready diagnostic results
using the data exposed by the feature extraction.

The workflow currently in use is:

1. Load preprocessed ACDC short-axis cine data.
2. Run framewise SAX mask inference with the local CineMA `ConvUNetR` model.
3. Save predicted label arrays and QC outputs.
4. Derive AHA-aligned wall-thickness, normalized-wall-thickness, radial-strain,
   and endocardial-excursion outputs from the predicted masks.

## Current repo layout

```text
cardionet_config.yaml     # Compatibility entrypoint that includes config/*.yaml
config/                   # Split user-editable runtime config by concern
data/                     # Local dataset mirror, including ACDC raw/processed data
model/                    # Local model mirror, including fine-tuned CineMA weights
scratch/                  # Ad hoc smoke-test scripts and local experiment outputs
scripts/                  # User-facing and developer-facing runnable scripts
src/cardionet/
|-- config/               # Shared config loading and runtime path/device helpers
|-- geometry/             # AHA reference geometry and contour helpers
|-- io/                   # Shared label / IO-facing helpers
|-- segmentation/         # Model loading, inference, transforms, and mask IO
|-- features/             # AHA sectors, wall thickness, radial strain, endocardial excursion
|-- visualization/        # Segmentation QC and AHA QC plotting
`-- pipelines/            # Reserved package namespace
tests/                    # Test suite
```

## Active modules

`cardionet.config`

Loads the split files under `config/` and resolves shared runtime settings such
as dataset roots, output directories, and device / dtype selection.

`cardionet.segmentation`

Contains the current CineMA integration:
- local fine-tuned model path resolution
- strict local model loading
- SAX preprocessing transforms
- framewise 4D cine inference
- saving inferred arrays

`cardionet.geometry`

Contains geometry helpers used by the current AHA workflow, including RV/LV
reference handling and direct RV-MYO contact anchors for clinician-selected
short-axis slices.

`cardionet.features`

Contains the current downstream feature logic for explicit AHA slice metadata,
canonical AHA segment numbering/names, AHA sector construction, wall thickness,
normalized wall thickness, tracked segment centroids, radial strain, and raycast
endocardial excursion.

`cardionet.visualization`

Contains the currently used QC outputs. New smoke tests and pipelines should
prefer `cardionet.visualization.canonical` for standardized physician-facing
bullseyes, per-ring time-series plots, mask overlay GIFs, and LV volume / EF
plots. Existing specialized QC helpers remain available for stage-specific
debugging.

Canonical outputs include:
- per-slice segmentation GIFs
- direct voxel-summation physical mask-volume plots with EF estimates from
  NIfTI spacing and optional LV slice-quality dropping
- raw per-ray NWT heatmap frames/GIFs with AHA segment boundary rays clipped
  to the myocardial wall
- AHA bullseye heatmaps with standard segment numbering and labels
- per-ring time-series plots with stable segment colors, ED/ES markers, and peaks

## Canonical segmentation labels

The repository uses the CineMA-aligned label convention throughout the active
pipeline:

- `0` = background
- `1` = RV
- `2` = MYO
- `3` = LV

## Config-first workflow

The main control surface is split by concern under `config/`. The compatibility
`cardionet_config.yaml` file only lists those fragments for older commands that
still pass a single config path.

- `config/paths.yaml`: roots, output locations, naming, and label conventions
- `config/segmentation.yaml`: runtime, model, inference, and segmentation scripts
- `config/features.yaml`: AHA geometry, shared raycast defaults, and feature scripts
- `config/classification.yaml`: CardioGPT export and threshold-classifier defaults
- `config/visualisation.yaml`: segmentation and volume QC settings
- `config/smoke_test.yaml`: scratch/debug patient selections and smoke reruns
- `config/full_dataset.yaml`: dataset selectors, validation, and dev switches

AHA feature scripts no longer infer basal/mid/apical slice type from the stack.
Before running AHA, radial-strain, endocardial-excursion, or stack smoke-test
outputs, fill in the clinician-selected `slice_index` and `slice_type` fields
or the smoke-test `selected_slices` placeholders.

The most relevant script sections are:

- `scripts.run_segmentation`
- `scripts.run_feature_extraction`
- `scripts.run_threshold_classifier`
- `scripts.run_visualisations`
- `scripts.run_full_pipeline`

## Current runnable scripts

## Modular pipeline runners

New work should prefer the staged runners. Each expensive stage writes durable
artifacts that later stages consume.

`scripts/run_segmentation.py`

Runs the configured segmentation model and saves reusable image and mask arrays.

`scripts/run_feature_extraction.py`

Consumes saved segmentation arrays and selected-slice metadata, then writes:

- `segment_metrics.csv`
- `metric_time_series.csv`
- `segment_metrics_labelled.csv` when labels are available
- `features.json`
- `feature_manifest.json`

`scripts/run_threshold_classifier.py`

Consumes `segment_metrics_labelled.csv` and writes threshold-search results,
including confusion matrices and per-class / macro F1, to
`threshold_sweep_summary.json`.

`scripts/run_visualisations.py`

Consumes saved feature tables and optional classification results to render
bullseyes and time-series plots without recomputing mask-derived features.

`scripts/run_full_pipeline.py`

Runs the staged scripts in order. Examples:

```bash
python scripts/run_full_pipeline.py --config config/smoke_test.yaml --skip-segmentation
python scripts/run_full_pipeline.py --start-at features --stop-after classification
```

The older one-shot scripts were moved to ignored local archive
`scripts/legacy/`; see `scripts/DEPRECATED.md`.

## Cluster execution

The cluster layer is configured in `config/cluster.yaml`. It keeps scheduler,
environment, resource, manifest, status, log, and artifact-root assumptions out
of the cardiac processing code. Slurm is the first supported backend.

Typical flow:

```bash
python scripts/cluster_make_manifest.py --config config/full_dataset.yaml --data-index-path selected_slices.csv
python scripts/cluster_submit.py --config config/full_dataset.yaml --stage segmentation --dry-run
python scripts/cluster_submit.py --config config/full_dataset.yaml --stage segmentation
python scripts/cluster_submit.py --config config/full_dataset.yaml --stage features
python scripts/cluster_collect_status.py --config config/full_dataset.yaml --merge-features
python scripts/cluster_submit.py --config config/full_dataset.yaml --stage classification
python scripts/cluster_submit.py --config config/full_dataset.yaml --stage visualisation
```

`cluster_make_manifest.py` writes one row per configured patient/case with
resolved input paths, expected segmentation outputs, per-case feature output
directories, and selected-slice metadata. `cluster_run_stage.py` is the worker
called by Slurm array jobs. It skips completed outputs unless `--force` is set
and writes JSON status files under `cluster.artifacts.status_root`.

Use `cluster_submit.py --dry-run` first on a new cluster. For MareNostrum or any
other cluster, adjust `config/cluster.yaml` for the real partition names, GPU
request syntax expectations, Conda/module activation command, memory, wall time,
and log/output roots.

`scratch/smoke_test_acdc_inference.py`

Developer smoke test used to validate the current local workflow on
`patient001` and `patient021`.

`scratch/smoke_test_acdc_stack_aha.py`

Runs a single-patient end-to-end smoke test that saves segmentation outputs,
clinician-selected AHA slice/chunk metrics, per-slice overlays, and a
nested-ring bullseye summary across basal, mid, and apical groups.

## Data and model assumptions

The current working setup expects:

- local ACDC data under `data/acdc/`
- preprocessed SAX cine volumes under `data/acdc/processed/<split>/<patient_id>/`
- a local mirrored fine-tuned CineMA model under `model/hf_mirror/mathpluscode_CineMA/`

The active segmentation path assumes preprocessed 4D SAX cine arrays in
canonical CardioNet order `(x, y, z, t)` after loading.

## Example usage

PowerShell:

```powershell
$env:PYTHONPATH = "D:/CardioNet/src"
python D:/CardioNet/scripts/example_acdc_pipeline.py
```

To change which patients are processed or where outputs are written, edit the
`scripts.example_acdc_pipeline` section in `config/smoke_test.yaml`.
