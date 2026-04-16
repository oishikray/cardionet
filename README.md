# CardioNet

CardioNet is a Python cardiac MRI analysis project built around a local
fine-tuned CineMA short-axis segmentation model plus downstream geometric and
feature extraction code.

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
4. Derive AHA-aligned wall-thickness and normalized-wall-thickness outputs from
   the predicted masks.

## Current repo layout

```text
cardionet_config.yaml     # Central user-editable runtime config
data/                     # Local dataset mirror, including ACDC raw/processed data
model/                    # Local model mirror, including fine-tuned CineMA weights
scratch/                  # Ad hoc smoke-test scripts and local experiment outputs
scripts/                  # User-facing and developer-facing runnable scripts
src/cardionet/
|-- config/               # Shared config loading and runtime path/device helpers
|-- geometry/             # AHA reference geometry and contour helpers
|-- io/                   # Shared label / IO-facing helpers
|-- segmentation/         # Model loading, inference, transforms, and mask IO
|-- features/             # AHA sectors and wall-thickness feature extraction
|-- visualization/        # Segmentation QC and AHA QC plotting
`-- pipelines/            # Reserved package namespace
tests/                    # Test suite
```

## Active modules

`cardionet.config`

Loads `cardionet_config.yaml` and resolves shared runtime settings such as
dataset roots, output directories, and device / dtype selection.

`cardionet.segmentation`

Contains the current CineMA integration:
- local fine-tuned model path resolution
- strict local model loading
- SAX preprocessing transforms
- framewise 4D cine inference
- saving inferred arrays

`cardionet.geometry`

Contains geometry helpers used by the current AHA workflow, including RV/LV
reference handling and anchor-angle construction.

`cardionet.features`

Contains the current downstream feature logic for AHA sector construction, wall
thickness, and normalized wall thickness.

`cardionet.visualization`

Contains the currently used QC outputs:
- per-slice segmentation GIFs
- mask-volume plots
- AHA / wall-thickness QC plots

## Canonical segmentation labels

The repository uses the CineMA-aligned label convention throughout the active
pipeline:

- `0` = background
- `1` = RV
- `2` = MYO
- `3` = LV

## Config-first workflow

`cardionet_config.yaml` is the main control surface for the current pipeline.
The active scripts read their model paths, dataset roots, patient selection,
output roots, and runtime settings from that file.

The most relevant script sections are:

- `scripts.infer_acdc_cine`
- `scripts.extract_aha_wt_nwt`
- `scripts.scratch_smoke_acdc_inference`
- `scripts.example_acdc_pipeline`

## Current runnable scripts

`scripts/infer_acdc_cine.py`

Runs config-driven SAX segmentation inference on the configured ACDC split and
patient selection, then writes arrays and segmentation QC outputs.

`scripts/extract_aha_wt_nwt.py`

Loads saved segmentation labels and derives AHA-aligned wall-thickness outputs
and related QC artifacts.

`scripts/example_acdc_pipeline.py`

A simplified, heavily commented example intended for non-technical users. It
shows how to run the currently working pipeline on configured preprocessed ACDC
patients and save all smoke-tested outputs.

`scratch/smoke_test_acdc_inference.py`

Developer smoke test used to validate the current local workflow on
`patient001` and `patient021`.

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
`scripts.example_acdc_pipeline` section in `cardionet_config.yaml`.
