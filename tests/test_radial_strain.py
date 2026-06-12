import numpy as np
import pytest

from cardionet.features.aha_segments import sector_boundaries_from_anchor
from cardionet.features.radial_strain import (
    choose_es_frame_from_lv_areas,
    compute_radial_strain,
    compute_segment_centroid_trajectories,
    compute_segment_radial_distances,
    compute_sector_centroids,
    compute_weighted_global_curve,
)


def make_circular_slice(
    *,
    size: int = 64,
    lv_radius: float = 10.0,
    outer_radius: float = 18.0,
) -> np.ndarray:
    y, x = np.indices((size, size))
    yc = (size - 1) / 2.0
    xc = (size - 1) / 2.0
    radius = np.hypot(y - yc, x - xc)

    labels = np.zeros((size, size), dtype=np.uint8)
    labels[(radius > lv_radius) & (radius <= outer_radius)] = 2
    labels[radius <= lv_radius] = 3
    labels[((y - (yc - outer_radius - 6.0)) / 6.0) ** 2 + ((x - xc) / 8.0) ** 2 <= 1.0] = 1
    labels[radius <= lv_radius] = 3
    return labels


def test_compute_sector_centroids_returns_expected_centers():
    sector_map = np.array([
        [0, 0, -1],
        [1, 1, -1],
        [1, 1, 0],
    ])

    centroids = compute_sector_centroids(sector_map, n_sectors=3)

    assert centroids[0, 0] == pytest.approx((0 + 0 + 2) / 3)
    assert centroids[0, 1] == pytest.approx((0 + 1 + 2) / 3)
    assert centroids[1, 0] == pytest.approx((1 + 1 + 2 + 2) / 4)
    assert centroids[1, 1] == pytest.approx((0 + 1 + 0 + 1) / 4)
    assert np.all(np.isnan(centroids[2]))


def test_compute_radial_strain_has_zero_ed_and_positive_thickening():
    binned_wt = np.array([
        [10.0, 12.0, 15.0],
        [8.0, 8.0, 12.0],
    ])

    radial_strain = compute_radial_strain(binned_wt, ed_frame=0)

    assert np.allclose(radial_strain[:, 0], 0.0)
    assert radial_strain[0, 1] == pytest.approx(0.2)
    assert radial_strain[0, 2] == pytest.approx(0.5)
    assert radial_strain[1, 2] == pytest.approx(0.5)


def test_compute_segment_radial_distances_uses_lv_centroid_framewise():
    segment_centroids = np.array([
        [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],
        [[5.0, 7.0], [6.0, 8.0], [7.0, 9.0]],
    ])
    lv_centroids = np.array([
        [5.0, 4.0],
        [6.0, 5.0],
        [7.0, 6.0],
    ])

    radial_positions = compute_segment_radial_distances(segment_centroids, lv_centroids)

    assert np.allclose(radial_positions[0], [1.0, 1.0, 1.0])
    assert np.allclose(radial_positions[1], [3.0, 3.0, 3.0])


def test_compute_segment_centroid_trajectories_returns_expected_shapes():
    frame0 = make_circular_slice(lv_radius=11.0, outer_radius=18.0)
    frame1 = make_circular_slice(lv_radius=9.0, outer_radius=19.0)

    labels = np.zeros((64, 64, 1, 2), dtype=np.uint8)
    labels[:, :, 0, 0] = frame0
    labels[:, :, 0, 1] = frame1
    bounds = sector_boundaries_from_anchor(anchor_angle=0.0, n_sectors=6)

    segment_centroids, sector_areas, lv_centroids, lv_areas = compute_segment_centroid_trajectories(
        labels,
        slice_index=0,
        bounds=bounds,
        n_sectors=6,
    )

    assert segment_centroids.shape == (6, 2, 2)
    assert sector_areas.shape == (6, 2)
    assert lv_centroids.shape == (2, 2)
    assert lv_areas.shape == (2,)
    assert np.all(np.isfinite(lv_centroids))
    assert np.all(lv_areas > 0)
    assert np.all(sector_areas.sum(axis=0) > 0)


def test_choose_es_frame_from_lv_areas_picks_smallest_populated_frame():
    lv_areas = np.array([110.0, 72.0, 80.0, 0.0])
    assert choose_es_frame_from_lv_areas(lv_areas) == 1


def test_compute_weighted_global_curve_ignores_nan_values():
    values = np.array([
        [0.1, 0.2, np.nan],
        [0.3, np.nan, 0.5],
    ])
    weights = np.array([1.0, 3.0])

    curve = compute_weighted_global_curve(values, weights)

    assert curve[0] == pytest.approx(0.25)
    assert curve[1] == pytest.approx(0.2)
    assert curve[2] == pytest.approx(0.5)
