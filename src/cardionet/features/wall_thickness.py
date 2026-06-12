from __future__ import annotations

from typing import Literal, overload

import numpy as np

from cardionet.features.aha_segments import assign_sector_index
from cardionet.features.slices import label_slice_yx
from cardionet.geometry.aha_reference import compute_lv_centroid, get_masks


def sample_ray_labels(
    label_slice: np.ndarray,
    yc: float,
    xc: float,
    theta: float,
    *,
    step: float = 0.25,
    max_radius: float = 250.0,
) -> list[tuple[float, int, int, int]]:
    """
    Sample labels along a ray starting at the LV centroid.

    Returns
    -------
    list of tuples
        Each tuple is (radius, iy, ix, label).
    """
    samples: list[tuple[float, int, int, int]] = []
    r = 0.0

    while r <= max_radius:
        y = yc + r * np.sin(theta)
        x = xc + r * np.cos(theta)

        iy = int(round(y))
        ix = int(round(x))

        if iy < 0 or ix < 0 or iy >= label_slice.shape[0] or ix >= label_slice.shape[1]:
            break

        label = int(label_slice[iy, ix])
        samples.append((r, iy, ix, label))
        r += step

    return samples


def thickness_from_transitions(
    samples: list[tuple[float, int, int, int]],
) -> tuple[float, float, float]:
    """
    Estimate endocardial radius, epicardial radius, and wall thickness
    from label transitions along one ray.

    Current transition logic expects:
    LV cavity (3) -> MYO (2) -> not-MYO
    """
    if not samples:
        return np.nan, np.nan, np.nan

    r_endo = None
    r_epi = None
    prev_label = samples[0][3]

    for r, _, _, label in samples[1:]:
        if r_endo is None and prev_label == 3 and label == 2:
            r_endo = r
        elif r_endo is not None and r_epi is None and prev_label == 2 and label != 2:
            r_epi = r
            break

        prev_label = label

    if r_endo is None or r_epi is None:
        return np.nan, np.nan, np.nan

    return r_endo, r_epi, r_epi - r_endo


def compute_frame_profile_and_sectors(
    label_slice: np.ndarray,
    bounds: np.ndarray,
    *,
    n_rays: int,
    ray_step: float,
    max_radius: float = 250.0,
    lv_centroid: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Compute ray-wise wall thickness, epicardial radii, and sector IDs.
    """
    yc, xc = compute_lv_centroid(label_slice) if lv_centroid is None else lv_centroid
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    wt = np.full(n_rays, np.nan, dtype=float)
    epicardial_radii = np.full(n_rays, np.nan, dtype=float)
    sector_ids = np.full(n_rays, -1, dtype=int)

    if not np.isfinite(yc) or not np.isfinite(xc):
        return angles, wt, epicardial_radii, sector_ids, (yc, xc)

    for i, theta in enumerate(angles):
        samples = sample_ray_labels(
            label_slice,
            yc,
            xc,
            theta,
            step=ray_step,
            max_radius=max_radius,
        )
        _, r_epi, th = thickness_from_transitions(samples)
        wt[i] = th
        epicardial_radii[i] = r_epi
        sector_ids[i] = assign_sector_index(theta, bounds)

    return angles, wt, epicardial_radii, sector_ids, (yc, xc)


@overload
def compute_wt_matrix_and_sector_ids(
    labels_4d: np.ndarray,
    slice_index: int,
    bounds: np.ndarray,
    *,
    n_rays: int,
    ray_step: float,
    max_radius: float = 250.0,
    reference_frame: int | None = None,
    return_epicardial_radius_matrix: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def compute_wt_matrix_and_sector_ids(
    labels_4d: np.ndarray,
    slice_index: int,
    bounds: np.ndarray,
    *,
    n_rays: int,
    ray_step: float,
    max_radius: float = 250.0,
    reference_frame: int | None = None,
    return_epicardial_radius_matrix: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def compute_wt_matrix_and_sector_ids(
    labels_4d: np.ndarray,
    slice_index: int,
    bounds: np.ndarray,
    *,
    n_rays: int,
    ray_step: float,
    max_radius: float = 250.0,
    reference_frame: int | None = None,
    return_epicardial_radius_matrix: bool = False,
) -> tuple[np.ndarray, ...]:
    """
    Compute ray-wise wall thickness over time for one slice.

    Parameters
    ----------
    labels_4d
        Label volume shaped (x, y, z, t).
    slice_index
        Slice index to analyze.
    bounds
        AHA sector boundaries in radians.
    reference_frame
        Optional frame whose LV centroid is used as the fixed ray origin for
        every frame. Pass the ED frame to keep raycasts spatially anchored
        through the cardiac cycle.

    Returns
    -------
    tuple
        (
            angles,
            wt_matrix,
            sector_ids,
            centroids,
            lv_areas,
        )

        If ``return_epicardial_radius_matrix`` is true, the returned tuple
        includes ``epicardial_radius_matrix`` between ``wt_matrix`` and
        ``sector_ids``.
    """
    _, _, _, t = labels_4d.shape

    wt_matrix = np.full((n_rays, t), np.nan, dtype=float)
    epicardial_radius_matrix = np.full((n_rays, t), np.nan, dtype=float)
    centroids = np.full((t, 2), np.nan, dtype=float)
    lv_areas = np.zeros(t, dtype=float)

    reference_centroid: tuple[float, float] | None = None
    if reference_frame is not None:
        if reference_frame < 0 or reference_frame >= t:
            raise IndexError(f"reference_frame {reference_frame} is out of bounds for {t} frames")
        reference_slice = label_slice_yx(labels_4d, slice_index, reference_frame)
        reference_centroid = compute_lv_centroid(reference_slice)
        if not np.isfinite(reference_centroid[0]) or not np.isfinite(reference_centroid[1]):
            raise ValueError(
                f"Reference frame {reference_frame} does not have a finite LV centroid."
            )

    angles = None
    sector_ids = None

    for frame in range(t):
        label_slice = label_slice_yx(labels_4d, slice_index, frame)
        _, _, lv = get_masks(label_slice)
        lv_areas[frame] = np.sum(lv)

        angles, wt, epicardial_radii, frame_sector_ids, centroid = (
            compute_frame_profile_and_sectors(
                label_slice=label_slice,
                bounds=bounds,
                n_rays=n_rays,
                ray_step=ray_step,
                max_radius=max_radius,
                lv_centroid=reference_centroid,
            )
        )

        wt_matrix[:, frame] = wt
        epicardial_radius_matrix[:, frame] = epicardial_radii
        centroids[frame] = centroid

        if sector_ids is None:
            sector_ids = frame_sector_ids

    if angles is None or sector_ids is None:
        raise RuntimeError("Failed to compute wall-thickness profiles.")

    if return_epicardial_radius_matrix:
        return angles, wt_matrix, epicardial_radius_matrix, sector_ids, centroids, lv_areas

    return angles, wt_matrix, sector_ids, centroids, lv_areas


def compute_aha_binned_wt(
    wt_matrix: np.ndarray,
    sector_ids: np.ndarray,
    *,
    n_sectors: int,
) -> np.ndarray:
    """
    Bin ray-wise thickness into AHA sectors using per-sector median.
    """
    _, t = wt_matrix.shape
    binned = np.full((n_sectors, t), np.nan, dtype=float)

    for s in range(n_sectors):
        idx = np.where(sector_ids == s)[0]
        if len(idx) == 0:
            continue

        binned[s] = np.nanmedian(wt_matrix[idx, :], axis=0)

    return binned


def normalize_by_ed(
    binned_wt: np.ndarray,
    ed_frame: int,
) -> np.ndarray:
    """
    Normalize sector-wise thickness by the ED frame thickness.
    """
    baseline = binned_wt[:, ed_frame][:, None]

    nwt = np.divide(
        binned_wt,
        baseline,
        out=np.full_like(binned_wt, np.nan, dtype=float),
        where=np.isfinite(baseline) & (baseline != 0),
    )

    return nwt


def mean_initial_epicardial_radius(
    epicardial_radius_matrix: np.ndarray,
    initial_frame: int,
) -> float:
    """
    Compute the mean epicardial radius at the normalization frame.

    The paper-style NWT denominator is one scalar per analyzed slice: the mean
    distance from the LV centroid to the epicardium at the initial timeframe.
    """
    if epicardial_radius_matrix.ndim != 2:
        raise ValueError(
            "Expected epicardial_radius_matrix shaped (num_rays, num_frames), "
            f"got {epicardial_radius_matrix.shape}"
        )
    if initial_frame < 0 or initial_frame >= epicardial_radius_matrix.shape[1]:
        raise IndexError(
            f"initial_frame {initial_frame} is out of bounds for "
            f"{epicardial_radius_matrix.shape[1]} frames"
        )

    frame_radii = epicardial_radius_matrix[:, initial_frame]
    finite = frame_radii[np.isfinite(frame_radii)]
    if finite.size == 0:
        raise ValueError(
            f"No finite epicardial radii found at initial frame {initial_frame}."
        )

    return float(np.mean(finite))


def normalize_by_initial_epicardial_radius(
    binned_wt: np.ndarray,
    epicardial_radius_matrix: np.ndarray,
    initial_frame: int,
) -> np.ndarray:
    """
    Normalize sector-wise thickness by mean initial epicardial radius.

    This follows the method described for NWT as:
    ``WT(t) / mean_epicardial_radius(initial_frame)``.
    """
    baseline = mean_initial_epicardial_radius(
        epicardial_radius_matrix,
        initial_frame=initial_frame,
    )

    return np.divide(
        binned_wt,
        baseline,
        out=np.full_like(binned_wt, np.nan, dtype=float),
        where=np.isfinite(binned_wt) & np.isfinite(baseline) & (baseline != 0),
    )


def choose_slice(
    labels: np.ndarray,
    slice_index: int | None,
    *,
    strategy: str = "center_nonempty",
) -> int:
    """
    Select a slice index.

    If no explicit slice is provided, use the requested strategy.
    """
    if labels.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels.shape}")

    _, _, s, _ = labels.shape

    if slice_index is not None:
        resolved_slice = int(slice_index)
        if resolved_slice < 0 or resolved_slice >= s:
            raise IndexError(f"slice_index {resolved_slice} is out of bounds for {s} slices")
        return resolved_slice

    resolved_strategy = str(strategy).lower()
    if resolved_strategy in {"middle", "geometric_middle", "center"}:
        return s // 2

    if resolved_strategy == "center_nonempty":
        lv_present = np.any(labels == 3, axis=(0, 1, 3))
        myo_present = np.any(labels == 2, axis=(0, 1, 3))
        candidate_slices = np.where(lv_present & myo_present)[0]

        if candidate_slices.size == 0:
            candidate_slices = np.where(np.any(labels != 0, axis=(0, 1, 3)))[0]

        if candidate_slices.size == 0:
            raise ValueError("No non-empty slices found in label volume.")

        center = (s - 1) / 2.0
        nearest_idx = int(np.argmin(np.abs(candidate_slices - center)))
        return int(candidate_slices[nearest_idx])

    raise ValueError(f"Unsupported slice selection strategy: {strategy}")


def choose_ed_frame_from_lv_area(
    labels: np.ndarray,
    slice_index: int,
) -> int:
    """
    Choose ED frame as the frame with maximum LV area in the given slice.
    """
    lv_areas = np.array([
        np.sum(labels[:, :, slice_index, f] == 3)
        for f in range(labels.shape[-1])
    ])

    if not np.any(lv_areas > 0):
        raise ValueError(f"No LV pixels found in slice {slice_index}.")

    return int(np.argmax(lv_areas))
