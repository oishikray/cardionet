import numpy as np
import pytest

from cardionet.geometry.aha_reference import (
    compute_contact_centroid,
    compute_lv_centroid,
    find_rv_myo_contact_region,
)


def test_compute_lv_centroid_simple_mask():
    label_slice = np.zeros((10, 10), dtype=np.uint8)
    label_slice[4:6, 4:6] = 3

    yc, xc = compute_lv_centroid(label_slice)

    assert np.isfinite(yc)
    assert np.isfinite(xc)
    assert yc == pytest.approx(4.5)
    assert xc == pytest.approx(4.5)


def test_compute_lv_centroid_empty_lv_returns_nan():
    label_slice = np.zeros((10, 10), dtype=np.uint8)

    yc, xc = compute_lv_centroid(label_slice)

    assert np.isnan(yc)
    assert np.isnan(xc)


def test_compute_contact_centroid_simple_mask():
    contact = np.zeros((10, 10), dtype=bool)
    contact[2:4, 6:8] = True

    yc, xc = compute_contact_centroid(contact)

    assert yc == pytest.approx(2.5)
    assert xc == pytest.approx(6.5)


def test_find_rv_myo_contact_region_nonempty():
    rv = np.zeros((10, 10), dtype=bool)
    myo = np.zeros((10, 10), dtype=bool)

    rv[5, 4] = True
    myo[5, 5] = True

    contact = find_rv_myo_contact_region(rv, myo, dilation_iterations=1)
    assert contact.sum() >= 1