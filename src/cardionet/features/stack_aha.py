from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cardionet.features.aha_segments import (
    AHASliceReference,
    AHASegment,
    build_sector_map,
    compute_sector_areas,
    get_segments,
    resolve_aha_slice_references,
    sector_boundaries_from_anchor,
)
from cardionet.features.slices import label_slice_yx
from cardionet.features.wall_thickness import (
    compute_aha_binned_wt,
    compute_wt_matrix_and_sector_ids,
    mean_initial_epicardial_radius,
    normalize_by_initial_epicardial_radius,
)
from cardionet.geometry.aha_reference import (
    get_masks,
    point_from_angle,
)


@dataclass(slots=True)
class AHASliceFeatureSet:
    """Per-slice AHA geometry and wall-thickness features."""

    slice_index: int
    slice_type: str
    frame_index: int
    anchor_angle: float
    anchor_point: tuple[float, float]
    lv_centroid: tuple[float, float]
    anchor_source: str
    segments: tuple[AHASegment, ...]
    segment_names: list[str]
    segment_numbers: list[int]
    bounds: np.ndarray
    sector_map: np.ndarray
    sector_areas: np.ndarray
    angles: np.ndarray
    sector_ids: np.ndarray
    wt_matrix: np.ndarray
    epicardial_radius_matrix: np.ndarray
    binned_wt: np.ndarray
    nwt: np.ndarray
    initial_epicardial_radius: float
    centroids: np.ndarray
    lv_areas: np.ndarray


@dataclass(slots=True)
class AHAChunkFeatureSet:
    """Aggregated AHA features for one stack chunk."""

    chunk_name: str
    slice_indices: tuple[int, ...]
    segments: tuple[AHASegment, ...]
    segment_names: list[str]
    segment_numbers: list[int]
    per_slice_weights: np.ndarray
    aggregated_sector_areas: np.ndarray
    aggregated_wt: np.ndarray
    aggregated_nwt: np.ndarray
    weighted_mean_ed_wt: float
    weighted_mean_peak_nwt: float
    weighted_mean_nwt_curve: np.ndarray


@dataclass(slots=True)
class AHAStackFeatureSet:
    """Full-stack AHA feature output for one cine volume."""

    global_ed_frame: int
    overlay_frame: int
    slice_references: list[AHASliceReference | None]
    slice_features: list[AHASliceFeatureSet]
    chunk_features: dict[str, AHAChunkFeatureSet]


CHUNK_ORDER = ("basal", "mid", "apical")


def choose_global_ed_frame_from_lv_volume(labels_4d: np.ndarray) -> int:
    """
    Choose one stack-wide ED frame from total LV volume over all slices.
    """
    if labels_4d.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels_4d.shape}")

    lv_volume = np.sum(labels_4d == 3, axis=(0, 1, 2))
    if not np.any(lv_volume > 0):
        raise ValueError("No LV pixels found anywhere in the label volume.")

    return int(np.argmax(lv_volume))


def _weighted_average(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute a NaN-aware weighted average over axis 0."""
    finite = np.isfinite(values)
    weighted = np.where(finite, values * weights, 0.0)
    weight_sum = np.where(finite, weights, 0.0).sum(axis=0)

    return np.divide(
        weighted.sum(axis=0),
        weight_sum,
        out=np.full(values.shape[1:], np.nan, dtype=float),
        where=weight_sum > 0,
    )


def _weighted_average_over_first_axis(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute a NaN-aware weighted average over axis 1."""
    finite = np.isfinite(values)
    weighted = np.where(finite, values * weights, 0.0)
    weight_sum = np.where(finite, weights, 0.0).sum(axis=1)

    return np.divide(
        weighted.sum(axis=1),
        weight_sum,
        out=np.full(values.shape[0], np.nan, dtype=float),
        where=weight_sum > 0,
    )


def _safe_rowwise_nanmax(values: np.ndarray) -> np.ndarray:
    """Return the row-wise nanmax while tolerating all-NaN rows."""
    maxima = np.full(values.shape[0], np.nan, dtype=float)
    for index, row in enumerate(values):
        if np.any(np.isfinite(row)):
            maxima[index] = float(np.nanmax(row))
    return maxima


def aggregate_chunk_features(
    chunk_name: str,
    slice_features: list[AHASliceFeatureSet],
    *,
    ed_frame: int,
) -> AHAChunkFeatureSet:
    """
    Aggregate multiple slices from the same stack chunk into one summary.
    """
    if not slice_features:
        raise ValueError(f"Cannot aggregate empty slice feature list for {chunk_name}.")

    segment_names = slice_features[0].segment_names
    segments = slice_features[0].segments
    segment_numbers = slice_features[0].segment_numbers
    per_slice_weights = np.stack([feature.sector_areas for feature in slice_features], axis=0)
    aggregated_sector_areas = per_slice_weights.sum(axis=0)

    wt_stack = np.stack([feature.binned_wt for feature in slice_features], axis=0)
    nwt_stack = np.stack([feature.nwt for feature in slice_features], axis=0)
    expanded_weights = per_slice_weights[:, :, None]

    aggregated_wt = _weighted_average(wt_stack, expanded_weights)
    aggregated_nwt = _weighted_average(nwt_stack, expanded_weights)
    peak_nwt = _safe_rowwise_nanmax(aggregated_nwt)

    weighted_mean_nwt_curve = _weighted_average_over_first_axis(
        aggregated_nwt.T,
        aggregated_sector_areas[None, :].repeat(aggregated_nwt.shape[1], axis=0),
    )
    weighted_mean_ed_wt = float(_weighted_average_over_first_axis(
        aggregated_wt[:, ed_frame][None, :],
        aggregated_sector_areas[None, :],
    )[0])
    weighted_mean_peak_nwt = float(_weighted_average_over_first_axis(
        peak_nwt[None, :],
        aggregated_sector_areas[None, :],
    )[0])

    return AHAChunkFeatureSet(
        chunk_name=chunk_name,
        slice_indices=tuple(feature.slice_index for feature in slice_features),
        segments=segments,
        segment_names=segment_names,
        segment_numbers=segment_numbers,
        per_slice_weights=per_slice_weights,
        aggregated_sector_areas=aggregated_sector_areas,
        aggregated_wt=aggregated_wt,
        aggregated_nwt=aggregated_nwt,
        weighted_mean_ed_wt=weighted_mean_ed_wt,
        weighted_mean_peak_nwt=weighted_mean_peak_nwt,
        weighted_mean_nwt_curve=weighted_mean_nwt_curve,
    )


def compute_slice_feature_set(
    labels_4d: np.ndarray,
    slice_reference: AHASliceReference,
    *,
    global_ed_frame: int,
    overlay_frame: int,
    n_rays: int,
    ray_step: float,
    max_radius: float,
) -> AHASliceFeatureSet:
    """Compute AHA wall-thickness features for one resolved slice reference."""
    label_slice_reference = label_slice_yx(
        labels_4d,
        slice_reference.slice_index,
        global_ed_frame,
    )
    lv_centroid = slice_reference.lv_centroid
    anchor_radius = float(np.hypot(
        slice_reference.anchor_point[0] - lv_centroid[0],
        slice_reference.anchor_point[1] - lv_centroid[1],
    ))

    anchor_angle = float(slice_reference.anchor_angle)
    anchor_point = point_from_angle(
        lv_centroid[0],
        lv_centroid[1],
        anchor_angle,
        anchor_radius,
    )

    segments = get_segments(slice_reference.slice_type)
    segment_names = [segment.name for segment in segments]
    segment_numbers = [segment.number for segment in segments]
    bounds = sector_boundaries_from_anchor(anchor_angle, len(segment_names))
    _, myo, _ = get_masks(label_slice_reference)
    sector_map = build_sector_map(lv_centroid[0], lv_centroid[1], myo, bounds)
    sector_areas = compute_sector_areas(sector_map, len(segment_names))

    (
        angles,
        wt_matrix,
        epicardial_radius_matrix,
        sector_ids,
        centroids,
        lv_areas,
    ) = compute_wt_matrix_and_sector_ids(
        labels_4d=labels_4d,
        slice_index=slice_reference.slice_index,
        bounds=bounds,
        n_rays=n_rays,
        ray_step=ray_step,
        max_radius=max_radius,
        reference_frame=global_ed_frame,
        return_epicardial_radius_matrix=True,
    )
    binned_wt = compute_aha_binned_wt(
        wt_matrix,
        sector_ids,
        n_sectors=len(segment_names),
    )
    initial_epicardial_radius = mean_initial_epicardial_radius(
        epicardial_radius_matrix,
        initial_frame=global_ed_frame,
    )
    nwt = normalize_by_initial_epicardial_radius(
        binned_wt,
        epicardial_radius_matrix,
        initial_frame=global_ed_frame,
    )

    return AHASliceFeatureSet(
        slice_index=slice_reference.slice_index,
        slice_type=slice_reference.slice_type,
        frame_index=slice_reference.frame_index,
        anchor_angle=anchor_angle,
        anchor_point=anchor_point,
        lv_centroid=lv_centroid,
        anchor_source=slice_reference.anchor_source,
        segments=segments,
        segment_names=segment_names,
        segment_numbers=segment_numbers,
        bounds=bounds,
        sector_map=sector_map,
        sector_areas=sector_areas,
        angles=angles,
        sector_ids=sector_ids,
        wt_matrix=wt_matrix,
        epicardial_radius_matrix=epicardial_radius_matrix,
        binned_wt=binned_wt,
        nwt=nwt,
        initial_epicardial_radius=initial_epicardial_radius,
        centroids=centroids,
        lv_areas=lv_areas,
    )


def analyze_stack_aha(
    labels_4d: np.ndarray,
    *,
    slice_types: dict[int, str] | list[str | None] | tuple[str | None, ...],
    overlay_frame: int | None = None,
    n_rays: int = 360,
    ray_step: float = 0.25,
    max_radius: float = 250.0,
) -> AHAStackFeatureSet:
    """
    Compute AHA wall-thickness features for explicitly typed selected slices.
    """
    global_ed_frame = choose_global_ed_frame_from_lv_volume(labels_4d)
    resolved_overlay_frame = global_ed_frame if overlay_frame is None else int(overlay_frame)
    if resolved_overlay_frame < 0 or resolved_overlay_frame >= labels_4d.shape[-1]:
        raise IndexError(
            f"overlay_frame {resolved_overlay_frame} is out of bounds for "
            f"{labels_4d.shape[-1]} frames"
        )

    slice_references = resolve_aha_slice_references(
        labels_4d,
        slice_types=slice_types,
        frame_index=global_ed_frame,
    )

    slice_features: list[AHASliceFeatureSet] = []
    for slice_reference in slice_references:
        if slice_reference is None:
            continue

        slice_features.append(
            compute_slice_feature_set(
                labels_4d,
                slice_reference,
                global_ed_frame=global_ed_frame,
                overlay_frame=resolved_overlay_frame,
                n_rays=n_rays,
                ray_step=ray_step,
                max_radius=max_radius,
            )
        )

    chunk_features: dict[str, AHAChunkFeatureSet] = {}
    for chunk_name in CHUNK_ORDER:
        chunk_slices = [
            feature for feature in slice_features if feature.slice_type == chunk_name
        ]
        if not chunk_slices:
            continue

        chunk_features[chunk_name] = aggregate_chunk_features(
            chunk_name,
            chunk_slices,
            ed_frame=global_ed_frame,
        )

    return AHAStackFeatureSet(
        global_ed_frame=global_ed_frame,
        overlay_frame=resolved_overlay_frame,
        slice_references=slice_references,
        slice_features=slice_features,
        chunk_features=chunk_features,
    )
