from __future__ import annotations

import numpy as np

from cardionet.features.aha_segments import assign_sector_index
from cardionet.features.slices import label_slice_yx
from cardionet.features.wall_thickness import sample_ray_labels, thickness_from_transitions
from cardionet.geometry.aha_reference import compute_lv_centroid, get_masks


def compute_frame_endocardial_profile_and_sectors(
    label_slice: np.ndarray,
    bounds: np.ndarray,
    *,
    n_rays: int,
    ray_step: float,
    max_radius: float = 250.0,
    lv_centroid: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], float]:
    """Compute ray-wise LV endocardial radii for one frame/slice."""
    yc, xc = compute_lv_centroid(label_slice) if lv_centroid is None else lv_centroid
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    endocardial_radii = np.full(n_rays, np.nan, dtype=float)
    sector_ids = np.full(n_rays, -1, dtype=int)
    _, _, lv = get_masks(label_slice)
    lv_area = float(np.sum(lv))

    if not np.isfinite(yc) or not np.isfinite(xc):
        return angles, endocardial_radii, sector_ids, (yc, xc), lv_area

    for ray_index, theta in enumerate(angles):
        samples = sample_ray_labels(
            label_slice,
            yc,
            xc,
            theta,
            step=ray_step,
            max_radius=max_radius,
        )
        r_endo, _, _ = thickness_from_transitions(samples)
        endocardial_radii[ray_index] = r_endo
        sector_ids[ray_index] = assign_sector_index(theta, bounds)

    return angles, endocardial_radii, sector_ids, (yc, xc), lv_area


def compute_endocardial_radius_matrix_and_sector_ids(
    labels_4d: np.ndarray,
    slice_index: int,
    bounds: np.ndarray,
    *,
    n_rays: int,
    ray_step: float,
    max_radius: float = 250.0,
    reference_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ray-wise LV endocardial radii over time for one slice.

    When ``reference_frame`` is supplied, its LV centroid is used as the fixed
    ray origin for every frame. Pass the ED frame to measure endocardial motion
    against a stable cardiac-cycle reference.
    """
    if labels_4d.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels_4d.shape}")

    _, _, num_slices, num_frames = labels_4d.shape
    if slice_index < 0 or slice_index >= num_slices:
        raise IndexError(f"slice_index {slice_index} is out of bounds for {num_slices} slices")

    radius_matrix = np.full((n_rays, num_frames), np.nan, dtype=float)
    centroids = np.full((num_frames, 2), np.nan, dtype=float)
    lv_areas = np.zeros(num_frames, dtype=float)

    reference_centroid: tuple[float, float] | None = None
    if reference_frame is not None:
        if reference_frame < 0 or reference_frame >= num_frames:
            raise IndexError(
                f"reference_frame {reference_frame} is out of bounds for {num_frames} frames"
            )
        reference_slice = label_slice_yx(labels_4d, slice_index, reference_frame)
        reference_centroid = compute_lv_centroid(reference_slice)
        if not np.isfinite(reference_centroid[0]) or not np.isfinite(reference_centroid[1]):
            raise ValueError(
                f"Reference frame {reference_frame} does not have a finite LV centroid."
            )

    angles = None
    sector_ids = None

    for frame_index in range(num_frames):
        label_slice = label_slice_yx(labels_4d, slice_index, frame_index)
        angles, radii, frame_sector_ids, centroid, lv_area = (
            compute_frame_endocardial_profile_and_sectors(
                label_slice=label_slice,
                bounds=bounds,
                n_rays=n_rays,
                ray_step=ray_step,
                max_radius=max_radius,
                lv_centroid=reference_centroid,
            )
        )

        radius_matrix[:, frame_index] = radii
        centroids[frame_index] = centroid
        lv_areas[frame_index] = lv_area

        if sector_ids is None:
            sector_ids = frame_sector_ids

    if angles is None or sector_ids is None:
        raise RuntimeError("Failed to compute endocardial radius profiles.")

    return angles, radius_matrix, sector_ids, centroids, lv_areas


def compute_aha_binned_endocardial_radius(
    radius_matrix: np.ndarray,
    sector_ids: np.ndarray,
    *,
    n_sectors: int,
) -> np.ndarray:
    """Bin ray-wise endocardial radii into AHA sectors using per-sector median."""
    _, num_frames = radius_matrix.shape
    binned = np.full((n_sectors, num_frames), np.nan, dtype=float)

    for sector_index in range(n_sectors):
        ray_indices = np.where(sector_ids == sector_index)[0]
        if len(ray_indices) == 0:
            continue
        binned[sector_index] = np.nanmedian(radius_matrix[ray_indices, :], axis=0)

    return binned


def compute_endocardial_excursion(
    binned_endocardial_radius: np.ndarray,
    *,
    ed_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute inward LV endocardial excursion from ED.

    Absolute excursion is in pixels and is positive when the endocardial radius
    decreases relative to ED. Fractional excursion divides that absolute motion
    by the ED radius for each sector.
    """
    baseline = binned_endocardial_radius[:, ed_frame][:, None]
    absolute = baseline - binned_endocardial_radius
    fractional = np.divide(
        absolute,
        baseline,
        out=np.full_like(binned_endocardial_radius, np.nan, dtype=float),
        where=np.isfinite(baseline) & (baseline != 0),
    )
    return absolute, fractional


def compute_finite_ray_counts_by_sector(
    radius_matrix: np.ndarray,
    sector_ids: np.ndarray,
    *,
    n_sectors: int,
) -> np.ndarray:
    """Count finite endocardial-radius rays per sector and frame."""
    counts = np.zeros((n_sectors, radius_matrix.shape[1]), dtype=int)
    finite = np.isfinite(radius_matrix)

    for sector_index in range(n_sectors):
        ray_indices = np.where(sector_ids == sector_index)[0]
        if len(ray_indices) == 0:
            continue
        counts[sector_index] = np.sum(finite[ray_indices, :], axis=0)

    return counts
