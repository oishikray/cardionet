import numpy as np
import pytest

from cardionet.features.wall_thickness import (
    choose_ed_frame_from_lv_area,
    choose_slice,
    compute_aha_binned_wt,
    compute_wt_matrix_and_sector_ids,
    mean_initial_epicardial_radius,
    normalize_by_ed,
    normalize_by_initial_epicardial_radius,
    thickness_from_transitions,
)


def test_thickness_from_transitions_simple_case():
    samples = [
        (0.0, 5, 5, 3),
        (1.0, 5, 6, 3),
        (2.0, 5, 7, 2),
        (3.0, 5, 8, 2),
        (4.0, 5, 9, 0),
    ]

    r_endo, r_epi, th = thickness_from_transitions(samples)

    assert r_endo == pytest.approx(2.0)
    assert r_epi == pytest.approx(4.0)
    assert th == pytest.approx(2.0)


def test_thickness_from_transitions_failure_returns_nan():
    samples = [
        (0.0, 5, 5, 3),
        (1.0, 5, 6, 3),
        (2.0, 5, 7, 3),
    ]

    r_endo, r_epi, th = thickness_from_transitions(samples)

    assert np.isnan(r_endo)
    assert np.isnan(r_epi)
    assert np.isnan(th)


def test_compute_aha_binned_wt_shape():
    wt_matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])
    sector_ids = np.array([0, 0, 1, 1])

    binned = compute_aha_binned_wt(wt_matrix, sector_ids, n_sectors=2)
    assert binned.shape == (2, 2)


def test_normalize_by_ed_makes_ed_column_one():
    binned_wt = np.array([
        [2.0, 4.0, 6.0],
        [5.0, 10.0, 15.0],
    ])

    nwt = normalize_by_ed(binned_wt, ed_frame=0)
    assert np.allclose(nwt[:, 0], 1.0, equal_nan=False)


def test_normalize_by_initial_epicardial_radius_uses_mean_radius():
    binned_wt = np.array([
        [2.0, 4.0],
        [3.0, 5.0],
    ])
    epicardial_radius_matrix = np.array([
        [10.0, 9.0],
        [12.0, 11.0],
        [np.nan, 10.0],
    ])

    baseline = mean_initial_epicardial_radius(epicardial_radius_matrix, initial_frame=0)
    nwt = normalize_by_initial_epicardial_radius(
        binned_wt,
        epicardial_radius_matrix,
        initial_frame=0,
    )

    assert baseline == pytest.approx(11.0)
    assert np.allclose(nwt, binned_wt / 11.0)


def test_choose_ed_frame_from_lv_area():
    labels = np.zeros((8, 8, 1, 3), dtype=np.uint8)
    labels[2:4, 2:4, 0, 0] = 3
    labels[2:6, 2:6, 0, 1] = 3
    labels[2:3, 2:3, 0, 2] = 3

    ed_frame = choose_ed_frame_from_lv_area(labels, slice_index=0)
    assert ed_frame == 1


def test_choose_slice_prefers_center_nonempty_slice():
    labels = np.zeros((8, 8, 5, 2), dtype=np.uint8)
    labels[2:4, 2:4, 1, 0] = 3
    labels[1:5, 1:5, 1, 0] = 2
    labels[2:4, 2:4, 3, 0] = 3
    labels[1:5, 1:5, 3, 0] = 2

    slice_index = choose_slice(labels, slice_index=None)

    assert slice_index == 1


def test_choose_ed_frame_from_lv_area_raises_for_empty_slice():
    labels = np.zeros((8, 8, 1, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        choose_ed_frame_from_lv_area(labels, slice_index=0)



def make_shifted_ring_slice(
    *,
    size: int = 64,
    center_y: float = 31.5,
    center_x: float = 31.5,
    lv_radius: float = 8.0,
    outer_radius: float = 14.0,
) -> np.ndarray:
    y, x = np.indices((size, size))
    radius = np.hypot(y - center_y, x - center_x)
    labels = np.zeros((size, size), dtype=np.uint8)
    labels[(radius > lv_radius) & (radius <= outer_radius)] = 2
    labels[radius <= lv_radius] = 3
    return labels


def test_compute_wt_matrix_can_use_fixed_reference_centroid():
    labels = np.zeros((64, 64, 1, 2), dtype=np.uint8)
    labels[:, :, 0, 0] = make_shifted_ring_slice(center_y=28.0, center_x=30.0)
    labels[:, :, 0, 1] = make_shifted_ring_slice(center_y=36.0, center_x=38.0)
    bounds = np.linspace(0.0, 2.0 * np.pi, 5)

    _, _, _, centroids, _ = compute_wt_matrix_and_sector_ids(
        labels,
        slice_index=0,
        bounds=bounds,
        n_rays=90,
        ray_step=0.5,
        max_radius=32.0,
        reference_frame=0,
    )

    assert np.allclose(centroids[0], centroids[1])
    assert centroids[0, 0] == pytest.approx(30.0, abs=0.25)
    assert centroids[0, 1] == pytest.approx(28.0, abs=0.25)
