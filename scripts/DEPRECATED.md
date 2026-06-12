# Deprecated Script Entry Points

The scripts below have been archived locally under `scripts/legacy/` while the
modular pipeline runners are adopted. The legacy directory is intentionally
ignored by Git.

- `infer_acdc_cine.py`
- `segmentation_gifs_acdc_selection.py`
- `extract_aha_wt_nwt.py`
- `extract_radial_strain.py`
- `extract_endocardial_excursion.py`
- `example_acdc_pipeline.py`
- `smoke_test_saved_segmentation_visualizations.py`
- `export_cardiogpt_segment_timeseries.py`
- `classify_segment_metrics_thresholds.py`
- `generate_delta_t_graphs_cardiogpt.py`

Prefer the staged runners for new work:

- `run_segmentation.py`
- `run_feature_extraction.py`
- `run_threshold_classifier.py`
- `run_visualisations.py`
- `run_full_pipeline.py`

Do not add new workflow logic to archived scripts.
