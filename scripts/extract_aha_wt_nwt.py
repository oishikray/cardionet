from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from cardionet.config import (
    load_cardionet_config,
    resolve_prediction_labels_path,
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
from cardionet.io.labels import load_label_volume
from cardionet.visualization.aha_qc import (
    moving_average_nan,
    save_debug_wt_plot,
    save_structure_and_sector_overlay,
    save_time_series_plots,
)

SCRIPT_NAME = "extract_aha_wt_nwt"


def resolve_runtime_paths(config: DictConfig) -> tuple[Path, Path]:
    """Resolve the configured label input and AHA output directories."""
    script_cfg = config.scripts[SCRIPT_NAME]
    patient_id = str(script_cfg.patient_id)
    view = str(config.segmentation.model.view)

    labels_path = resolve_prediction_labels_path(
        config,
        patient_id=patient_id,
        view=view,
        source_script_name=str(script_cfg.source_script),
    )
    output_dir = resolve_script_output_dir(
        config,
        script_name=SCRIPT_NAME,
        patient_id=patient_id,
        view=view,
        output_dir_template=str(script_cfg.output_dir_template),
    )
    return labels_path, output_dir


def main() -> None:
    config = load_cardionet_config()
    script_cfg = config.scripts[SCRIPT_NAME]
    patient_id = str(script_cfg.patient_id)

    labels_path, outdir = resolve_runtime_paths(config)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = load_label_volume(labels_path)
    print("Patient:", patient_id)
    print("Loaded labels from:", labels_path)
    print("Loaded labels shape:", labels.shape)
    print("Unique labels:", np.unique(labels))

    slice_index = choose_slice(
        labels,
        script_cfg.slice_index,
        strategy=str(config.geometry.slice_selection.default_mid_slice_strategy),
    )
    print("Using slice index:", slice_index)

    ed_frame = choose_ed_frame_from_lv_area(labels, slice_index)
    overlay_frame = (
        ed_frame
        if script_cfg.frame_index_for_overlay is None
        else int(script_cfg.frame_index_for_overlay)
    )
    if overlay_frame < 0 or overlay_frame >= labels.shape[-1]:
        raise IndexError(
            f"frame_index_for_overlay {overlay_frame} is out of bounds for "
            f"{labels.shape[-1]} frames"
        )

    print("Chosen ED frame (max LV area in slice):", ed_frame)
    print("Overlay frame:", overlay_frame)

    label_slice_overlay = labels[:, :, slice_index, overlay_frame]

    anchor_angle, lv_centroid, contact_centroid, contact_mask = (
        compute_anchor_angle_from_rv_contact(label_slice_overlay)
    )

    print("LV centroid:", lv_centroid)
    print("RV-MYO contact pixels:", int(contact_mask.sum()))
    print("RV-contact centroid:", contact_centroid)
    print("Anchor angle (deg):", np.degrees(anchor_angle))

    ring_type = str(script_cfg.ring_type)
    segment_names = get_segment_names(ring_type)
    expected_sectors = int(config.geometry.aha.segment_counts[ring_type])
    if len(segment_names) != expected_sectors:
        raise RuntimeError(
            f"Configured segment count for {ring_type} is {expected_sectors}, "
            f"but get_segment_names returned {len(segment_names)} names."
        )

    bounds = sector_boundaries_from_anchor(anchor_angle, len(segment_names))
    print("Sector boundary angles (deg):", np.degrees(bounds))

    _, myo, _ = get_masks(label_slice_overlay)
    sector_map = build_sector_map(lv_centroid[0], lv_centroid[1], myo, bounds)
    print("Assigned pixel sectors:", np.unique(sector_map[sector_map >= 0]))

    angles, wt_matrix, sector_ids, centroids, lv_areas = compute_wt_matrix_and_sector_ids(
        labels_4d=labels,
        slice_index=slice_index,
        bounds=bounds,
        n_rays=int(script_cfg.n_rays),
        ray_step=float(script_cfg.ray_step),
        max_radius=float(script_cfg.max_radius),
    )

    print("WT matrix shape:", wt_matrix.shape)
    print("Centroids shape:", centroids.shape)
    print("LV areas shape:", lv_areas.shape)

    finite_mask = np.isfinite(wt_matrix)
    print("Finite WT values:", finite_mask.sum(), "/", wt_matrix.size)
    print("Mean WT:", np.nanmean(wt_matrix))
    print("Min WT:", np.nanmin(wt_matrix))
    print("Max WT:", np.nanmax(wt_matrix))

    binned_wt = compute_aha_binned_wt(
        wt_matrix,
        sector_ids,
        n_sectors=len(segment_names),
    )
    nwt = normalize_by_ed(binned_wt, ed_frame=ed_frame)

    print("AHA binned WT shape:", binned_wt.shape)
    print("AHA NWT shape:", nwt.shape)

    np.save(outdir / "angles.npy", angles)
    np.save(outdir / "wt_matrix.npy", wt_matrix)
    np.save(outdir / "sector_ids_per_ray.npy", sector_ids)
    np.save(outdir / "centroids.npy", centroids)
    np.save(outdir / "lv_areas.npy", lv_areas)
    np.save(outdir / "aha_binned_wt.npy", binned_wt)
    np.save(outdir / "aha_nwt.npy", nwt)
    np.save(outdir / "sector_bounds_rad.npy", bounds)
    np.save(outdir / "sector_map.npy", sector_map)

    wt_frame = wt_matrix[:, ed_frame]
    wt_frame_smooth = moving_average_nan(
        wt_frame,
        window=int(script_cfg.smooth_window),
    )

    save_debug_wt_plot(wt_frame, wt_frame_smooth, ed_frame, outdir)
    save_time_series_plots(lv_areas, binned_wt, nwt, ed_frame, segment_names, outdir)

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
        outdir=outdir,
        line_radius=float(script_cfg.line_radius),
    )

    print("Saved outputs to:", outdir)


if __name__ == "__main__":
    main()
