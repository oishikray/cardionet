import numpy as np
import pytest

from cardionet.features.wall_thickness import (
    choose_ed_frame_from_lv_area,
    choose_slice,
    compute_aha_binned_wt,
    normalize_by_ed,
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
