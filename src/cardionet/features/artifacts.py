from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cardionet.features.delta_t import compute_delta_t
from cardionet.features.endocardial_excursion import (
    compute_aha_binned_endocardial_radius,
    compute_endocardial_excursion,
    compute_endocardial_radius_matrix_and_sector_ids,
    compute_finite_ray_counts_by_sector,
)
from cardionet.features.radial_strain import compute_radial_strain
from cardionet.features.stack_aha import AHAStackFeatureSet, analyze_stack_aha
from cardionet.io.artifacts import SegmentationPair
from cardionet.io.common import normalize_patient_id, normalize_slice_type
from cardionet.io.selection import SelectedSlice, SelectedSlicesByPatient


@dataclass(frozen=True, slots=True)
class RingMetricBundle:
    """Ring-level metric time series and peak values."""

    wall_thickness: dict[str, np.ndarray]
    nwt: dict[str, np.ndarray]
    radial_strain: dict[str, np.ndarray]
    endocardial_excursion: dict[str, np.ndarray]
    peak_wall_thickness: dict[str, np.ndarray]
    wall_thickness_std: dict[str, np.ndarray]
    peak_nwt: dict[str, np.ndarray]
    nwt_std: dict[str, np.ndarray]
    peak_radial_strain: dict[str, np.ndarray]
    peak_endocardial_excursion: dict[str, np.ndarray]


SEGMENT_METRIC_FIELDNAMES = [
    "patient_id",
    "patient_folder",
    "slice_index",
    "slice_type",
    "aha_sector_number",
    "aha_sector_index",
    "aha_sector_name",
    "ed_frame",
    "es_frame",
    "frame_count",
    "frame_interval_ms",
    "wall_thickness_ed",
    "wall_thickness_es",
    "peak_wall_thickness",
    "peak_wall_thickness_frame",
    "mean_wall_thickness",
    "wall_thickness_std",
    "nwt_ed",
    "nwt_es",
    "peak_nwt",
    "peak_nwt_frame",
    "mean_nwt",
    "nwt_std",
    "radial_strain_es",
    "peak_radial_strain",
    "peak_radial_strain_frame",
    "endocardial_excursion_es",
    "peak_endocardial_excursion",
    "peak_endocardial_excursion_frame",
    "delta_t_frame",
    "abs_delta_t_frame",
    "delta_t_ms",
    "wall_thickness_time_series",
    "nwt_time_series",
    "radial_strain_time_series",
    "endocardial_excursion_time_series",
]
LABELLED_SEGMENT_METRIC_FIELDNAMES = SEGMENT_METRIC_FIELDNAMES + ["diagnosis"]
TIME_SERIES_FIELDNAMES = [
    "patient_id",
    "patient_folder",
    "slice_index",
    "slice_type",
    "aha_sector_number",
    "aha_sector_index",
    "aha_sector_name",
    "frame",
    "metric",
    "value",
]


def weighted_average_slices(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """NaN-aware weighted average over the slice axis."""
    expanded_weights = weights[:, :, None]
    finite = np.isfinite(values)
    weighted = np.where(finite, values * expanded_weights, 0.0)
    weight_sum = np.where(finite, expanded_weights, 0.0).sum(axis=0)
    return np.divide(
        weighted.sum(axis=0),
        weight_sum,
        out=np.full(values.shape[1:], np.nan, dtype=float),
        where=weight_sum > 0,
    )


def safe_rowwise_nanmax(values: np.ndarray) -> np.ndarray:
    """Return row-wise nanmax while tolerating all-NaN rows."""
    peaks = np.full(values.shape[0], np.nan, dtype=float)
    for index, row in enumerate(values):
        if np.any(np.isfinite(row)):
            peaks[index] = float(np.nanmax(row))
    return peaks


def safe_rowwise_nanstd(values: np.ndarray) -> np.ndarray:
    """Return row-wise nanstd while tolerating all-NaN rows."""
    std_values = np.full(values.shape[0], np.nan, dtype=float)
    for index, row in enumerate(values):
        if np.any(np.isfinite(row)):
            std_values[index] = float(np.nanstd(row))
    return std_values


def finite_or_blank(value: float | int | None) -> float | int | str:
    """Return CSV-friendly scalar values with blanks for missing finite data."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value) if isinstance(value, (float, np.floating)) else value


def finite_float(value: float | int | None) -> float | None:
    """Return a finite float or None for JSON-friendly missing values."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def finite_json_list(values: np.ndarray) -> str:
    """Serialize a numeric series as JSON, preserving NaNs as nulls."""
    return json.dumps([finite_float(value) for value in np.asarray(values, dtype=float)])


def rowwise_peak_frames(values: np.ndarray) -> np.ndarray:
    """Return row-wise nanargmax frame indices, using -1 for all-NaN rows."""
    frames = np.full(values.shape[0], -1, dtype=int)
    for index, row in enumerate(values):
        if np.any(np.isfinite(row)):
            frames[index] = int(np.nanargmax(row))
    return frames


def scalar_at_frame(values: np.ndarray, frame: int) -> np.ndarray:
    """Return per-segment values at a frame, or NaNs if the frame is unavailable."""
    if frame < 0 or frame >= values.shape[1]:
        return np.full(values.shape[0], np.nan, dtype=float)
    return values[:, frame]


def choose_es_frame_from_stack_lv_volume(labels: np.ndarray) -> int:
    """Choose ES as the populated frame with minimum stack LV volume in pixels."""
    lv_volume = np.sum(labels == 3, axis=(0, 1, 2))
    populated = np.where(lv_volume > 0)[0]
    if populated.size == 0:
        raise ValueError("No LV pixels found in label volume.")
    return int(populated[np.argmin(lv_volume[populated])])


def compute_endocardial_excursion_by_ring(
    labels: np.ndarray,
    stack_features: AHAStackFeatureSet,
    *,
    n_rays: int,
    ray_step: float,
    max_radius: float,
) -> dict[str, np.ndarray]:
    """Compute ring-aggregated fractional inward endocardial excursion."""
    ed_frame = int(stack_features.global_ed_frame)
    values_by_ring: dict[str, list[np.ndarray]] = {}
    weights_by_ring: dict[str, list[np.ndarray]] = {}

    for feature in stack_features.slice_features:
        _, radius_matrix, sector_ids, _, _ = compute_endocardial_radius_matrix_and_sector_ids(
            labels,
            feature.slice_index,
            feature.bounds,
            n_rays=n_rays,
            ray_step=ray_step,
            max_radius=max_radius,
            reference_frame=ed_frame,
        )
        binned_radius = compute_aha_binned_endocardial_radius(
            radius_matrix,
            sector_ids,
            n_sectors=len(feature.segment_names),
        )
        _, fractional_excursion = compute_endocardial_excursion(
            binned_radius,
            ed_frame=ed_frame,
        )
        finite_ray_counts = compute_finite_ray_counts_by_sector(
            radius_matrix,
            sector_ids,
            n_sectors=len(feature.segment_names),
        )
        values_by_ring.setdefault(feature.slice_type, []).append(fractional_excursion)
        weights_by_ring.setdefault(feature.slice_type, []).append(
            finite_ray_counts[:, ed_frame].astype(float)
        )

    aggregated: dict[str, np.ndarray] = {}
    for ring, ring_values in values_by_ring.items():
        aggregated[ring] = weighted_average_slices(
            np.stack(ring_values, axis=0),
            np.stack(weights_by_ring[ring], axis=0),
        )
    return aggregated


def compute_metric_bundle(
    labels: np.ndarray,
    *,
    slice_types: dict[int, str],
    n_rays: int,
    ray_step: float,
    max_radius: float,
) -> tuple[AHAStackFeatureSet, RingMetricBundle, int]:
    """Compute NWT, radial-strain proxy, and endocardial-excursion ring metrics."""
    stack_features = analyze_stack_aha(
        labels,
        slice_types=slice_types,
        n_rays=n_rays,
        ray_step=ray_step,
        max_radius=max_radius,
    )
    es_frame = choose_es_frame_from_stack_lv_volume(labels)
    wt_by_ring = {
        ring: chunk.aggregated_wt
        for ring, chunk in stack_features.chunk_features.items()
    }
    nwt_by_ring = {
        ring: chunk.aggregated_nwt
        for ring, chunk in stack_features.chunk_features.items()
    }
    radial_strain_by_ring = {
        ring: compute_radial_strain(chunk.aggregated_wt, ed_frame=stack_features.global_ed_frame)
        for ring, chunk in stack_features.chunk_features.items()
    }
    endocardial_by_ring = compute_endocardial_excursion_by_ring(
        labels,
        stack_features,
        n_rays=n_rays,
        ray_step=ray_step,
        max_radius=max_radius,
    )
    bundle = RingMetricBundle(
        wall_thickness=wt_by_ring,
        nwt=nwt_by_ring,
        radial_strain=radial_strain_by_ring,
        endocardial_excursion=endocardial_by_ring,
        peak_wall_thickness={ring: safe_rowwise_nanmax(values) for ring, values in wt_by_ring.items()},
        wall_thickness_std={ring: safe_rowwise_nanstd(values) for ring, values in wt_by_ring.items()},
        peak_nwt={ring: safe_rowwise_nanmax(values) for ring, values in nwt_by_ring.items()},
        nwt_std={ring: safe_rowwise_nanstd(values) for ring, values in nwt_by_ring.items()},
        peak_radial_strain={
            ring: safe_rowwise_nanmax(values) for ring, values in radial_strain_by_ring.items()
        },
        peak_endocardial_excursion={
            ring: safe_rowwise_nanmax(values) for ring, values in endocardial_by_ring.items()
        },
    )
    return stack_features, bundle, es_frame


def build_segment_metric_rows(
    *,
    pair: SegmentationPair,
    selected_slice: SelectedSlice,
    stack_features: AHAStackFeatureSet,
    bundle: RingMetricBundle,
    es_frame: int,
) -> list[dict[str, Any]]:
    """Build one analysis row per patient/slice/AHA-sector combination."""
    ed_frame = int(stack_features.global_ed_frame)
    frame_interval_ms = (
        pair.frame_timing.frame_interval_ms if pair.frame_timing is not None else None
    )
    delta_t_by_ring = (
        compute_delta_t(
            bundle.nwt,
            es_frame=es_frame,
            frame_interval_ms=pair.frame_timing.frame_interval_ms,
        )
        if pair.frame_timing is not None
        else {}
    )

    rows: list[dict[str, Any]] = []
    for ring, chunk in stack_features.chunk_features.items():
        wt = np.asarray(bundle.wall_thickness[ring], dtype=float)
        nwt = np.asarray(bundle.nwt[ring], dtype=float)
        radial = np.asarray(bundle.radial_strain[ring], dtype=float)
        endocardial = np.asarray(bundle.endocardial_excursion[ring], dtype=float)
        wt_peak_frames = rowwise_peak_frames(wt)
        nwt_peak_frames = rowwise_peak_frames(nwt)
        radial_peak_frames = rowwise_peak_frames(radial)
        endocardial_peak_frames = rowwise_peak_frames(endocardial)
        wt_ed = scalar_at_frame(wt, ed_frame)
        wt_es = scalar_at_frame(wt, es_frame)
        nwt_ed = scalar_at_frame(nwt, ed_frame)
        nwt_es = scalar_at_frame(nwt, es_frame)
        radial_es = scalar_at_frame(radial, es_frame)
        endocardial_es = scalar_at_frame(endocardial, es_frame)
        delta_metrics = delta_t_by_ring.get(ring)

        for index, segment_number in enumerate(chunk.segment_numbers):
            wt_row = wt[index]
            nwt_row = nwt[index]
            rows.append({
                "patient_id": pair.patient_id,
                "patient_folder": pair.patient_folder,
                "slice_index": selected_slice.slice_index,
                "slice_type": ring,
                "aha_sector_number": segment_number,
                "aha_sector_index": index,
                "aha_sector_name": chunk.segment_names[index],
                "ed_frame": ed_frame,
                "es_frame": es_frame,
                "frame_count": wt.shape[1],
                "frame_interval_ms": finite_or_blank(frame_interval_ms),
                "wall_thickness_ed": finite_or_blank(wt_ed[index]),
                "wall_thickness_es": finite_or_blank(wt_es[index]),
                "peak_wall_thickness": finite_or_blank(bundle.peak_wall_thickness[ring][index]),
                "peak_wall_thickness_frame": (
                    finite_or_blank(wt_peak_frames[index]) if wt_peak_frames[index] >= 0 else ""
                ),
                "mean_wall_thickness": (
                    finite_or_blank(np.nanmean(wt_row)) if np.any(np.isfinite(wt_row)) else ""
                ),
                "wall_thickness_std": finite_or_blank(bundle.wall_thickness_std[ring][index]),
                "nwt_ed": finite_or_blank(nwt_ed[index]),
                "nwt_es": finite_or_blank(nwt_es[index]),
                "peak_nwt": finite_or_blank(bundle.peak_nwt[ring][index]),
                "peak_nwt_frame": (
                    finite_or_blank(nwt_peak_frames[index]) if nwt_peak_frames[index] >= 0 else ""
                ),
                "mean_nwt": (
                    finite_or_blank(np.nanmean(nwt_row)) if np.any(np.isfinite(nwt_row)) else ""
                ),
                "nwt_std": finite_or_blank(bundle.nwt_std[ring][index]),
                "radial_strain_es": finite_or_blank(radial_es[index]),
                "peak_radial_strain": finite_or_blank(bundle.peak_radial_strain[ring][index]),
                "peak_radial_strain_frame": (
                    finite_or_blank(radial_peak_frames[index]) if radial_peak_frames[index] >= 0 else ""
                ),
                "endocardial_excursion_es": finite_or_blank(endocardial_es[index]),
                "peak_endocardial_excursion": finite_or_blank(
                    bundle.peak_endocardial_excursion[ring][index]
                ),
                "peak_endocardial_excursion_frame": (
                    finite_or_blank(endocardial_peak_frames[index])
                    if endocardial_peak_frames[index] >= 0
                    else ""
                ),
                "delta_t_frame": finite_or_blank(delta_metrics.delta_t[index]) if delta_metrics else "",
                "abs_delta_t_frame": (
                    finite_or_blank(delta_metrics.abs_delta_t[index]) if delta_metrics else ""
                ),
                "delta_t_ms": finite_or_blank(delta_metrics.delta_t_ms[index]) if delta_metrics else "",
                "wall_thickness_time_series": finite_json_list(wt_row),
                "nwt_time_series": finite_json_list(nwt_row),
                "radial_strain_time_series": finite_json_list(radial[index]),
                "endocardial_excursion_time_series": finite_json_list(endocardial[index]),
            })
    return rows


def build_time_series_metric_rows(
    *,
    pair: SegmentationPair,
    selected_slice: SelectedSlice,
    stack_features: AHAStackFeatureSet,
    bundle: RingMetricBundle,
) -> list[dict[str, Any]]:
    """Build long-format raw metric rows for framewise downstream analysis."""
    metric_arrays = {
        "wall_thickness": bundle.wall_thickness,
        "nwt": bundle.nwt,
        "radial_strain": bundle.radial_strain,
        "endocardial_excursion": bundle.endocardial_excursion,
    }
    rows: list[dict[str, Any]] = []
    for ring, chunk in stack_features.chunk_features.items():
        for metric_name, values_by_ring in metric_arrays.items():
            values = np.asarray(values_by_ring[ring], dtype=float)
            for index, segment_number in enumerate(chunk.segment_numbers):
                for frame, value in enumerate(values[index]):
                    rows.append({
                        "patient_id": pair.patient_id,
                        "patient_folder": pair.patient_folder,
                        "slice_index": selected_slice.slice_index,
                        "slice_type": ring,
                        "aha_sector_number": segment_number,
                        "aha_sector_index": index,
                        "aha_sector_name": chunk.segment_names[index],
                        "frame": frame,
                        "metric": metric_name,
                        "value": finite_or_blank(value),
                    })
    return rows


def compute_pair_metric_rows(
    pair: SegmentationPair,
    *,
    selected_slice: SelectedSlice,
    n_rays: int,
    ray_step: float,
    max_radius: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute export rows for one saved segmentation without rerunning inference."""
    labels = np.load(pair.labels_path)
    if selected_slice.slice_index < 0 or selected_slice.slice_index >= labels.shape[2]:
        raise IndexError(
            f"{pair.patient_id} selected slice_index {selected_slice.slice_index} "
            f"is out of bounds for {labels.shape[2]} slices"
        )

    stack_features, bundle, es_frame = compute_metric_bundle(
        labels,
        slice_types={selected_slice.slice_index: selected_slice.slice_type},
        n_rays=n_rays,
        ray_step=ray_step,
        max_radius=max_radius,
    )
    return (
        build_segment_metric_rows(
            pair=pair,
            selected_slice=selected_slice,
            stack_features=stack_features,
            bundle=bundle,
            es_frame=es_frame,
        ),
        build_time_series_metric_rows(
            pair=pair,
            selected_slice=selected_slice,
            stack_features=stack_features,
            bundle=bundle,
        ),
    )


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write dictionaries to CSV with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_ground_truth_labels(path: Path) -> dict[tuple[str, int, str, int], str]:
    """Load diagnosis labels keyed by patient, slice, slice type, and sector."""
    labels: dict[tuple[str, int, str, int], str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "slice_index", "slice_type", "aha_sector_number", "diagnosis"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Ground-truth label CSV is missing columns: {sorted(missing)}")
        for row in reader:
            diagnosis = str(row.get("diagnosis", "")).strip()
            if not diagnosis:
                continue
            key = (
                normalize_patient_id(str(row["patient_id"])),
                int(row["slice_index"]),
                normalize_slice_type(str(row["slice_type"])),
                int(row["aha_sector_number"]),
            )
            labels[key] = diagnosis
    return labels


def attach_diagnosis_labels(
    rows: list[dict[str, Any]],
    labels: dict[tuple[str, int, str, int], str],
) -> list[dict[str, Any]]:
    """Return metric rows with a diagnosis column added when available."""
    labelled_rows = []
    for row in rows:
        key = (
            str(row["patient_id"]),
            int(row["slice_index"]),
            str(row["slice_type"]),
            int(row["aha_sector_number"]),
        )
        labelled = dict(row)
        labelled["diagnosis"] = labels.get(key, "")
        labelled_rows.append(labelled)
    return labelled_rows


def write_ground_truth_template(path: Path, rows: list[dict[str, Any]]) -> None:
    """Create a blank label-entry CSV matching the computed segment keys."""
    fieldnames = [
        "patient_id",
        "slice_index",
        "slice_type",
        "aha_sector_number",
        "aha_sector_name",
        "diagnosis",
        "notes",
    ]
    template_rows = [
        {
            "patient_id": row["patient_id"],
            "slice_index": row["slice_index"],
            "slice_type": row["slice_type"],
            "aha_sector_number": row["aha_sector_number"],
            "aha_sector_name": row["aha_sector_name"],
            "diagnosis": "",
            "notes": "",
        }
        for row in rows
    ]
    write_csv_rows(path, template_rows, fieldnames)


def run_feature_table_export(
    *,
    pairs_by_patient: dict[str, SegmentationPair],
    selected_slices: SelectedSlicesByPatient,
    output_dir: Path,
    ground_truth_labels_path: Path | None,
    n_rays: int,
    ray_step: float,
    max_radius: float,
) -> dict[str, Any]:
    """Compute all segment metrics from saved masks and save durable artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_rows: list[dict[str, Any]] = []
    time_series_rows: list[dict[str, Any]] = []
    selected_slice_count = 0

    for patient_id, patient_slices in selected_slices.items():
        if patient_id not in pairs_by_patient:
            raise FileNotFoundError(f"No segmentation arrays found for selected patient: {patient_id}")
        pair = pairs_by_patient[patient_id]
        for selected_slice in patient_slices:
            selected_slice_count += 1
            patient_segment_rows, patient_time_series_rows = compute_pair_metric_rows(
                pair,
                selected_slice=selected_slice,
                n_rays=n_rays,
                ray_step=ray_step,
                max_radius=max_radius,
            )
            segment_rows.extend(patient_segment_rows)
            time_series_rows.extend(patient_time_series_rows)

    segment_metrics_path = output_dir / "segment_metrics.csv"
    time_series_path = output_dir / "metric_time_series.csv"
    write_csv_rows(segment_metrics_path, segment_rows, SEGMENT_METRIC_FIELDNAMES)
    write_csv_rows(time_series_path, time_series_rows, TIME_SERIES_FIELDNAMES)

    labelled_path = None
    template_path = None
    if ground_truth_labels_path is not None and ground_truth_labels_path.exists():
        diagnosis_labels = load_ground_truth_labels(ground_truth_labels_path)
        labelled_rows = attach_diagnosis_labels(segment_rows, diagnosis_labels)
        labelled_path = output_dir / "segment_metrics_labelled.csv"
        write_csv_rows(labelled_path, labelled_rows, LABELLED_SEGMENT_METRIC_FIELDNAMES)
    else:
        template_path = output_dir / "diagnosis_template.csv"
        write_ground_truth_template(template_path, segment_rows)

    feature_json_path = output_dir / "features.json"
    feature_json_path.write_text(
        json.dumps(
            {
                "segment_metrics_csv": str(segment_metrics_path),
                "metric_time_series_csv": str(time_series_path),
                "labelled_segment_metrics_csv": str(labelled_path) if labelled_path else None,
                "records": segment_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "output_dir": str(output_dir),
        "segment_metrics_path": str(segment_metrics_path),
        "labelled_segment_metrics_path": str(labelled_path) if labelled_path else None,
        "metric_time_series_path": str(time_series_path),
        "features_json_path": str(feature_json_path),
        "diagnosis_template_path": str(template_path) if template_path else None,
        "ground_truth_labels_path": str(ground_truth_labels_path) if ground_truth_labels_path else None,
        "ground_truth_labels_found": bool(ground_truth_labels_path and ground_truth_labels_path.exists()),
        "patient_count": len(selected_slices),
        "selected_slice_count": selected_slice_count,
        "segment_row_count": len(segment_rows),
        "time_series_row_count": len(time_series_rows),
        "n_rays": n_rays,
        "ray_step": ray_step,
        "max_radius": max_radius,
    }
    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
