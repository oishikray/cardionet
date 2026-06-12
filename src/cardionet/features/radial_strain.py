from __future__ import annotations

import numpy as np

from cardionet.features.aha_segments import build_sector_map
from cardionet.features.slices import label_slice_yx
from cardionet.geometry.aha_reference import compute_lv_centroid, get_masks


def compute_sector_centroids(
    sector_map: np.ndarray,
    n_sectors: int,
) -> np.ndarray:
    """
    Compute one centroid per sector in ``(y, x)`` coordinates.

    Missing sectors are returned as ``(nan, nan)``.
    """
    centroids = np.full((n_sectors, 2), np.nan, dtype=float)

    for sector_index in range(n_sectors):
        coords = np.argwhere(sector_map == sector_index)
        if coords.size == 0:
            continue

        centroids[sector_index, 0] = float(coords[:, 0].mean())
        centroids[sector_index, 1] = float(coords[:, 1].mean())

    return centroids


def compute_frame_segment_centroids(
    label_slice: np.ndarray,
    bounds: np.ndarray,
    *,
    n_sectors: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float]:
    """
    Compute per-segment centroids for one frame of one short-axis slice.
    """
    lv_centroid = compute_lv_centroid(label_slice)
    sector_map = np.full(label_slice.shape, -1, dtype=int)
    segment_centroids = np.full((n_sectors, 2), np.nan, dtype=float)

    _, myo, lv = get_masks(label_slice)
    lv_area = float(np.sum(lv))

    if not np.isfinite(lv_centroid[0]) or not np.isfinite(lv_centroid[1]) or not np.any(myo):
        return sector_map, segment_centroids, lv_centroid, lv_area

    sector_map = build_sector_map(lv_centroid[0], lv_centroid[1], myo, bounds)
    segment_centroids = compute_sector_centroids(sector_map, n_sectors)
    return sector_map, segment_centroids, lv_centroid, lv_area


def compute_segment_centroid_trajectories(
    labels_4d: np.ndarray,
    slice_index: int,
    bounds: np.ndarray,
    *,
    n_sectors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Track per-segment MYO centroids over time for one short-axis slice.

    Returns
    -------
    tuple
        (
            segment_centroids_yx,
            sector_areas,
            lv_centroids_yx,
            lv_areas,
        )
    """
    if labels_4d.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels_4d.shape}")

    _, _, num_slices, num_frames = labels_4d.shape
    if slice_index < 0 or slice_index >= num_slices:
        raise IndexError(f"slice_index {slice_index} is out of bounds for {num_slices} slices")

    segment_centroids = np.full((n_sectors, num_frames, 2), np.nan, dtype=float)
    sector_areas = np.zeros((n_sectors, num_frames), dtype=int)
    lv_centroids = np.full((num_frames, 2), np.nan, dtype=float)
    lv_areas = np.zeros(num_frames, dtype=float)

    for frame_index in range(num_frames):
        label_slice = label_slice_yx(labels_4d, slice_index, frame_index)
        sector_map, frame_centroids, lv_centroid, lv_area = compute_frame_segment_centroids(
            label_slice,
            bounds,
            n_sectors=n_sectors,
        )

        segment_centroids[:, frame_index, :] = frame_centroids
        lv_centroids[frame_index] = lv_centroid
        lv_areas[frame_index] = lv_area

        for sector_index in range(n_sectors):
            sector_areas[sector_index, frame_index] = int(np.sum(sector_map == sector_index))

    return segment_centroids, sector_areas, lv_centroids, lv_areas


def compute_segment_radial_distances(
    segment_centroids: np.ndarray,
    lv_centroids: np.ndarray,
) -> np.ndarray:
    """
    Compute LV-centroid-to-segment-centroid radial distances over time.
    """
    if segment_centroids.ndim != 3 or segment_centroids.shape[-1] != 2:
        raise ValueError(
            "Expected segment_centroids shaped (num_sectors, num_frames, 2), "
            f"got {segment_centroids.shape}"
        )

    if lv_centroids.ndim != 2 or lv_centroids.shape[-1] != 2:
        raise ValueError(f"Expected lv_centroids shaped (num_frames, 2), got {lv_centroids.shape}")

    if segment_centroids.shape[1] != lv_centroids.shape[0]:
        raise ValueError(
            "Segment centroid frames and LV centroid frames must match, got "
            f"{segment_centroids.shape[1]} and {lv_centroids.shape[0]}"
        )

    dy = segment_centroids[:, :, 0] - lv_centroids[None, :, 0]
    dx = segment_centroids[:, :, 1] - lv_centroids[None, :, 1]
    return np.hypot(dy, dx)


def compute_radial_strain(
    binned_wt: np.ndarray,
    *,
    ed_frame: int,
) -> np.ndarray:
    """
    Compute segmental radial strain from AHA-binned wall thickness.

    Radial strain is defined here as the relative change in thickness from ED:

    ``(WT(t) - WT(ED)) / WT(ED)``
    """
    baseline = binned_wt[:, ed_frame][:, None]
    return np.divide(
        binned_wt - baseline,
        baseline,
        out=np.full_like(binned_wt, np.nan, dtype=float),
        where=np.isfinite(baseline) & (baseline != 0),
    )


def compute_weighted_global_curve(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute a NaN-aware weighted mean curve over sectors.
    """
    if values.ndim != 2:
        raise ValueError(f"Expected values shaped (num_sectors, num_frames), got {values.shape}")

    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim == 1:
        if weight_array.shape[0] != values.shape[0]:
            raise ValueError(
                f"Expected {values.shape[0]} weights, got {weight_array.shape[0]}"
            )
        weight_array = weight_array[:, None]
    elif weight_array.shape != values.shape:
        raise ValueError(
            "weights must be shaped (num_sectors,) or match values, got "
            f"{weight_array.shape} for values {values.shape}"
        )

    finite = np.isfinite(values)
    weighted = np.where(finite, values * weight_array, 0.0)
    weight_sum = np.where(finite, weight_array, 0.0).sum(axis=0)

    return np.divide(
        weighted.sum(axis=0),
        weight_sum,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=weight_sum > 0,
    )


def choose_es_frame_from_lv_areas(lv_areas: np.ndarray) -> int:
    """
    Choose ES as the populated frame with the smallest LV cavity area.
    """
    populated_frames = np.where(lv_areas > 0)[0]
    if populated_frames.size == 0:
        raise ValueError("No LV pixels found in the selected slice.")

    return int(populated_frames[np.argmin(lv_areas[populated_frames])])
