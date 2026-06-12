import numpy as np
import pytest

from cardionet.features.endocardial_excursion import (
    compute_aha_binned_endocardial_radius,
    compute_endocardial_excursion,
    compute_endocardial_radius_matrix_and_sector_ids,
)


def make_circular_slice(*, size: int = 64, lv_radius: float = 10.0, outer_radius: float = 16.0):
    y, x = np.indices((size, size))
    yc = (size - 1) / 2.0
    xc = (size - 1) / 2.0
    radius = np.hypot(y - yc, x - xc)

    labels = np.zeros((size, size), dtype=np.uint8)
    labels[(radius > lv_radius) & (radius <= outer_radius)] = 2
    labels[radius <= lv_radius] = 3
    return labels


def test_endocardial_excursion_tracks_lv_radius_decrease():
    labels = np.zeros((64, 64, 1, 2), dtype=np.uint8)
    labels[:, :, 0, 0] = make_circular_slice(lv_radius=10.0, outer_radius=16.0)
    labels[:, :, 0, 1] = make_circular_slice(lv_radius=8.0, outer_radius=16.0)
    bounds = np.linspace(0.0, 2.0 * np.pi, 5)

    _, radius_matrix, sector_ids, _, lv_areas = compute_endocardial_radius_matrix_and_sector_ids(
        labels,
        slice_index=0,
        bounds=bounds,
        n_rays=180,
        ray_step=0.5,
        max_radius=40.0,
    )
    binned_radius = compute_aha_binned_endocardial_radius(
        radius_matrix,
        sector_ids,
        n_sectors=4,
    )
    absolute, fractional = compute_endocardial_excursion(binned_radius, ed_frame=0)

    assert lv_areas[0] > lv_areas[1]
    assert np.nanmedian(binned_radius[:, 0]) == pytest.approx(10.0, abs=1.0)
    assert np.nanmedian(binned_radius[:, 1]) == pytest.approx(8.0, abs=1.0)
    assert np.nanmedian(absolute[:, 0]) == pytest.approx(0.0, abs=1e-6)
    assert np.nanmedian(absolute[:, 1]) == pytest.approx(2.0, abs=1.0)
    assert np.nanmedian(fractional[:, 1]) == pytest.approx(0.2, abs=0.12)



def test_endocardial_radius_matrix_can_use_fixed_reference_centroid():
    frame0 = make_circular_slice(lv_radius=10.0, outer_radius=16.0)
    frame1 = np.roll(make_circular_slice(lv_radius=8.0, outer_radius=16.0), shift=(6, 5), axis=(0, 1))
    labels = np.zeros((64, 64, 1, 2), dtype=np.uint8)
    labels[:, :, 0, 0] = frame0
    labels[:, :, 0, 1] = frame1
    bounds = np.linspace(0.0, 2.0 * np.pi, 5)

    _, _, _, centroids, _ = compute_endocardial_radius_matrix_and_sector_ids(
        labels,
        slice_index=0,
        bounds=bounds,
        n_rays=90,
        ray_step=0.5,
        max_radius=40.0,
        reference_frame=0,
    )

    assert np.allclose(centroids[0], centroids[1])
    assert centroids[0, 0] == pytest.approx(31.5, abs=0.25)
    assert centroids[0, 1] == pytest.approx(31.5, abs=0.25)
