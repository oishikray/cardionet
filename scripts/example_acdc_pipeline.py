from __future__ import annotations

"""
Small end-user example for the CardioNet ACDC pipeline.

What this script does
---------------------
1. Reads the central settings from ``cardionet_config.yaml``.
2. Loads the local fine-tuned CineMA SAX segmentation model.
3. Loads preprocessed ACDC cine data for the patient IDs listed in the config.
4. Runs frame-by-frame mask inference.
5. Saves the predicted masks and input arrays to disk.
6. Generates the same QC outputs used in the smoke test:
   - per-slice GIF overlays
   - mask-volume plot
   - AHA / wall-thickness outputs

How to use
----------
1. Open ``cardionet_config.yaml``.
2. Edit the ``scripts.example_acdc_pipeline`` section:
   - choose the patient IDs you want to run
   - choose where you want outputs saved
3. Run this script from the project root with ``PYTHONPATH`` pointing at ``src``.

PowerShell example
------------------
$env:PYTHONPATH = "D:/CardioNet/src"
python D:/CardioNet/scripts/example_acdc_pipeline.py

This example only uses the workflow that has already been smoke-tested locally.
"""

from pathlib import Path

import numpy as np

from cardionet.config import (
    load_cardionet_config,
    resolve_dataset_root,
    resolve_output_basename,
    resolve_runtime_device_and_dtype,
    resolve_script_output_dir,
)
from cardionet.features.aha_segments import (
    build_sector_map,
    get_segment_names,
    sector_boundaries_from_anchor,
)
from cardionet.features.wall_thickness import (
    choose_ed_frame_from_lv_area,
    choose_slice,
    compute_aha_binned_wt,
    compute_wt_matrix_and_sector_ids,
    normalize_by_ed,
)
from cardionet.geometry.aha_reference import (
    compute_anchor_angle_from_rv_contact,
    get_masks,
)
from cardionet.segmentation.inference import infer_cine_frames, subsample_time_axis
from cardionet.segmentation.io import load_cine_nifti, save_inference_arrays
from cardionet.segmentation.model_loader import load_convunetr_from_local
from cardionet.segmentation.model_paths import resolve_finetuned_model_paths_from_config
from cardionet.segmentation.transforms import build_sax_inference_transform
from cardionet.visualization.aha_qc import (
    moving_average_nan,
    save_debug_wt_plot,
    save_structure_and_sector_overlay,
    save_time_series_plots,
)
from cardionet.visualization.segmentation_qc import (
    plot_segmentations_per_slice,
    plot_volume_changes,
)

SCRIPT_NAME = "example_acdc_pipeline"


def get_patient_output_dir(config, *, patient_id: str, view: str) -> Path:
    """
    Resolve the output directory for one patient.

    This keeps the path logic in one place and makes the script follow the
    same config-driven convention as the rest of the repo.
    """
    script_cfg = config.scripts[SCRIPT_NAME]
    return resolve_script_output_dir(
        config,
        script_name=SCRIPT_NAME,
        patient_id=patient_id,
        view=view,
        output_dir_template=str(script_cfg.output_dir_template),
    )


def get_cine_filename(config, *, dataset_name: str, patient_id: str) -> str:
    """
    Build the cine filename from the dataset naming rule in the config.

    For the current ACDC setup this resolves to ``patientXXX_sax_t.nii.gz``.
    """
    dataset_cfg = config.datasets[dataset_name]
    return str(dataset_cfg.file_patterns.cine).format(patient_id=patient_id)


def get_qc_label_colors(config) -> dict[int, np.ndarray]:
    """
    Resolve the overlay colors for RV, MYO, and LV.

    The label IDs come from the canonical CardioNet convention:
    0=background, 1=RV, 2=MYO, 3=LV.
    """
    colors_cfg = config.qc.segmentation_overlays.colors_rgba
    return {
        int(config.conventions.labels.rv): np.array(colors_cfg.rv, dtype=np.float32),
        int(config.conventions.labels.myocardium): np.array(
            colors_cfg.myocardium,
            dtype=np.float32,
        ),
        int(config.conventions.labels.lv): np.array(colors_cfg.lv, dtype=np.float32),
    }


def save_aha_outputs(config, *, labels: np.ndarray, output_dir: Path) -> None:
    """
    Save the same AHA / wall-thickness outputs used in the smoke test.

    The logic here is intentionally direct:
    - choose a representative SAX slice
    - choose the ED frame from LV area
    - compute a sector anchor from RV/LV geometry
    - measure wall thickness over time
    - save arrays and QC plots
    """
    aha_script_cfg = config.scripts.extract_aha_wt_nwt
    strategy = str(config.geometry.slice_selection.default_mid_slice_strategy)

    slice_index = choose_slice(
        labels,
        aha_script_cfg.slice_index,
        strategy=strategy,
    )
    ed_frame = choose_ed_frame_from_lv_area(labels, slice_index)
    overlay_frame = (
        ed_frame
        if aha_script_cfg.frame_index_for_overlay is None
        else int(aha_script_cfg.frame_index_for_overlay)
    )

    label_slice_overlay = labels[:, :, slice_index, overlay_frame]
    anchor_angle, lv_centroid, contact_centroid, _ = compute_anchor_angle_from_rv_contact(
        label_slice_overlay
    )

    ring_type = str(aha_script_cfg.ring_type)
    segment_names = get_segment_names(ring_type)
    bounds = sector_boundaries_from_anchor(anchor_angle, len(segment_names))

    _, myo, _ = get_masks(label_slice_overlay)
    sector_map = build_sector_map(lv_centroid[0], lv_centroid[1], myo, bounds)

    angles, wt_matrix, sector_ids, centroids, lv_areas = compute_wt_matrix_and_sector_ids(
        labels_4d=labels,
        slice_index=slice_index,
        bounds=bounds,
        n_rays=int(aha_script_cfg.n_rays),
        ray_step=float(aha_script_cfg.ray_step),
        max_radius=float(aha_script_cfg.max_radius),
    )
    binned_wt = compute_aha_binned_wt(
        wt_matrix,
        sector_ids,
        n_sectors=len(segment_names),
    )
    nwt = normalize_by_ed(binned_wt, ed_frame=ed_frame)

    aha_dir = output_dir / "aha"
    aha_dir.mkdir(parents=True, exist_ok=True)

    # Save the numeric outputs first so they can be reused directly in analysis.
    np.save(aha_dir / "angles.npy", angles)
    np.save(aha_dir / "wt_matrix.npy", wt_matrix)
    np.save(aha_dir / "sector_ids_per_ray.npy", sector_ids)
    np.save(aha_dir / "centroids.npy", centroids)
    np.save(aha_dir / "lv_areas.npy", lv_areas)
    np.save(aha_dir / "aha_binned_wt.npy", binned_wt)
    np.save(aha_dir / "aha_nwt.npy", nwt)
    np.save(aha_dir / "sector_bounds_rad.npy", bounds)
    np.save(aha_dir / "sector_map.npy", sector_map)

    # Then save the human-readable QC plots for quick inspection.
    wt_frame = wt_matrix[:, ed_frame]
    wt_frame_smooth = moving_average_nan(
        wt_frame,
        window=int(aha_script_cfg.smooth_window),
    )

    save_debug_wt_plot(wt_frame, wt_frame_smooth, ed_frame, aha_dir)
    save_time_series_plots(lv_areas, binned_wt, nwt, ed_frame, segment_names, aha_dir)
    save_structure_and_sector_overlay(
        label_slice=label_slice_overlay,
        sector_map=sector_map,
        bounds=bounds,
        lv_centroid=lv_centroid,
        contact_centroid=contact_centroid,
        segment_names=segment_names,
        binned_wt=binned_wt,
        nwt=nwt,
        ed_frame=ed_frame,
        outdir=aha_dir,
        line_radius=float(aha_script_cfg.line_radius),
    )


def main() -> None:
    """
    Run the example pipeline from start to finish.

    The steps are deliberately explicit so the script can also be read as a
    guide:
    - load config
    - load model
    - build preprocessing transform
    - loop over patients
    - run inference
    - save arrays
    - save QC outputs
    """
    config = load_cardionet_config()
    script_cfg = config.scripts[SCRIPT_NAME]
    dataset_name = str(script_cfg.dataset_name)
    split = str(script_cfg.split)
    view = str(config.segmentation.model.view)

    # Resolve the device and dtype from the central config.
    # On a machine with CUDA available, this can move inference to the GPU.
    device, dtype = resolve_runtime_device_and_dtype(config)
    print("Using device:", device)
    print("Using dtype:", dtype)

    # Resolve the local mirrored model files from the config and load the model.
    weights_path, config_path = resolve_finetuned_model_paths_from_config(config)
    model = load_convunetr_from_local(
        weights_path=weights_path,
        config_path=config_path,
        device=device,
        eval_mode=bool(config.segmentation.model.eval_mode),
    )

    # Build the same preprocessing transform used in the smoke test.
    transform = build_sax_inference_transform(
        view=str(config.segmentation.inference.input_key),
        scale_intensity=bool(config.segmentation.inference.preprocessing.scale_intensity),
        pad_spatial=bool(config.segmentation.inference.preprocessing.pad_spatial),
        spatial_size=tuple(config.segmentation.inference.preprocessing.pad_spatial_size),
        pad_method=str(config.segmentation.inference.preprocessing.pad_method),
    )

    # Resolve the input dataset directory once, then iterate over the patient IDs
    # listed in the config section for this script.
    dataset_root = resolve_dataset_root(
        config,
        dataset_name=dataset_name,
        split=split,
    )
    patient_ids = [str(patient_id) for patient_id in script_cfg.patient_ids]
    t_step = int(config.segmentation.inference.temporal_subsampling_step)

    for patient_id in patient_ids:
        print("\n" + "=" * 60)
        print(f"Running example pipeline for: {patient_id}")
        print("=" * 60)

        # Build the path to the preprocessed cine file, for example:
        # D:/CardioNet/data/acdc/processed/train/patient001/patient001_sax_t.nii.gz
        input_nifti_path = dataset_root / patient_id / get_cine_filename(
            config,
            dataset_name=dataset_name,
            patient_id=patient_id,
        )
        if not input_nifti_path.exists():
            raise FileNotFoundError(f"Missing preprocessed cine file: {input_nifti_path}")

        # Create the output directory for this patient and decide the common
        # filename prefix used for arrays and QC images.
        output_dir = get_patient_output_dir(config, patient_id=patient_id, view=view)
        output_dir.mkdir(parents=True, exist_ok=True)
        basename = resolve_output_basename(config, patient_id=patient_id, view=view)

        # Load the 4D cine data into CardioNet's canonical numpy axis order:
        # (x, y, z, t)
        images = load_cine_nifti(input_nifti_path)

        # Optional temporal subsampling lives in the config. With the current
        # smoke-tested setup the default is 1, so every frame is used.
        images = subsample_time_axis(images, t_step=t_step)
        print("Input shape:", images.shape)

        # Run framewise segmentation to get a label volume with the same shape
        # as the input cine. The output labels use the canonical mapping:
        # 0=background, 1=RV, 2=MYO, 3=LV.
        labels = infer_cine_frames(
            model=model,
            images=images,
            transform=transform,
            view=str(config.segmentation.inference.input_key),
            device=device,
            dtype=dtype,
            show_progress=bool(config.logging.tqdm_enabled),
        )
        print("Predicted labels shape:", labels.shape)
        print("Unique labels:", np.unique(labels))

        # Save the image array and the predicted label array so they can be used
        # by downstream analysis or reloaded later without repeating inference.
        save_inference_arrays(
            images=images,
            labels=labels,
            output_dir=output_dir,
            basename=basename,
            save_inputs=bool(config.outputs.save_inputs),
            save_predictions=bool(config.outputs.save_predictions),
            image_suffix=str(config.conventions.file_naming.saved_images_suffix),
            labels_suffix=str(config.conventions.file_naming.predicted_labels_suffix),
        )

        # Save one GIF per SAX slice to make it easy to inspect the segmentation
        # over the cardiac cycle.
        if bool(script_cfg.save_gif):
            gif_paths = plot_segmentations_per_slice(
                images=images,
                labels=labels,
                t_step=t_step,
                output_dir=output_dir,
                basename=basename,
                figure_size=tuple(config.qc.segmentation_overlays.figure_size),
                dpi=int(config.qc.segmentation_overlays.dpi),
                cmap=str(config.qc.segmentation_overlays.cmap),
                gif_loop=int(config.qc.segmentation_overlays.gif_loop),
                gif_duration_base_ms=int(config.qc.segmentation_overlays.gif_duration_base_ms),
                label_colors=get_qc_label_colors(config),
            )
            print(f"Saved {len(gif_paths)} GIF(s)")

        # Save the LV/RV volume plot derived from the predicted masks.
        if bool(script_cfg.save_volume_plot):
            volume_plot_path = output_dir / (
                f"{basename}{config.conventions.file_naming.volume_plot_suffix}"
            )
            plot_volume_changes(
                labels=labels,
                t_step=t_step,
                filepath=volume_plot_path,
                voxel_volume_ml=float(config.qc.volume_plot.voxel_volume_ml),
                figsize=tuple(config.qc.volume_plot.figsize),
                dpi_screen=int(config.qc.volume_plot.dpi_screen),
                dpi_save=int(config.qc.volume_plot.dpi_save),
                ylabel=str(config.qc.volume_plot.ylabel),
            )
            print("Saved volume plot:", volume_plot_path)

        # Save the AHA and wall-thickness outputs under an "aha" subdirectory.
        if bool(script_cfg.save_aha_outputs):
            save_aha_outputs(config, labels=labels, output_dir=output_dir)
            print("Saved AHA outputs under:", output_dir / "aha")


if __name__ == "__main__":
    main()
