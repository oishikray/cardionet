import numpy as np
import pytest
from scipy.ndimage import rotate as ndi_rotate

from cardionet.features.aha_segments import (
    build_sector_map,
    compute_aha_slice_reference,
    compute_sector_areas,
    get_segment_names,
    sector_boundaries_from_anchor,
)
from cardionet.features.wall_thickness import (
    compute_aha_binned_wt,
    compute_wt_matrix_and_sector_ids,
)
from cardionet.features.slices import label_slice_yx
from cardionet.geometry.aha_reference import get_masks, wrap_signed_angle


def make_synthetic_slice(
    *,
    size: int = 128,
    lv_radius: float = 12.0,
    outer_radius: float = 20.0,
    abnormality_angle: float = np.deg2rad(-100.0),
    abnormality_width: float = np.deg2rad(22.0),
    abnormality_gain: float = 4.0,
    include_rv: bool = True,
) -> np.ndarray:
    y, x = np.indices((size, size))
    yc = (size - 1) / 2.0
    xc = (size - 1) / 2.0
    dy = y - yc
    dx = x - xc
    radius = np.hypot(dy, dx)
    theta = np.arctan2(dy, dx)

    outer = np.full((size, size), outer_radius, dtype=float)
    anterior_wedge = np.abs(wrap_signed_angle(theta - abnormality_angle)) <= abnormality_width
    outer[anterior_wedge] += abnormality_gain

    lv = radius <= lv_radius
    myo = (radius > lv_radius) & (radius <= outer)

    labels = np.zeros((size, size), dtype=np.uint8)
    labels[myo] = 2
    labels[lv] = 3

    if include_rv:
        rv_center_y = yc - (outer_radius + 8.0)
        rv_center_x = xc
        rv = (
            ((y - rv_center_y) / 8.0) ** 2
            + ((x - rv_center_x) / 11.0) ** 2
            <= 1.0
        )
        labels[rv & ~lv] = 1

    return labels


def make_synthetic_stack() -> np.ndarray:
    basal = make_synthetic_slice(outer_radius=22.0, abnormality_gain=5.0, include_rv=True)
    mid = make_synthetic_slice(outer_radius=20.0, abnormality_gain=4.0, include_rv=True)
    apical = make_synthetic_slice(outer_radius=16.0, abnormality_gain=3.0, include_rv=False)
    return np.stack([basal.T, mid.T, apical.T], axis=2)[..., None]


def rotate_stack_k90(labels_4d: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(labels_4d, k=k, axes=(0, 1)).copy()


def rotate_stack_nearest_neighbor(labels_4d: np.ndarray, angle_deg: float) -> np.ndarray:
    rotated = np.zeros_like(labels_4d)
    for slice_index in range(labels_4d.shape[2]):
        for frame_index in range(labels_4d.shape[3]):
            rotated[:, :, slice_index, frame_index] = ndi_rotate(
                labels_4d[:, :, slice_index, frame_index],
                angle=angle_deg,
                reshape=False,
                order=0,
            )
    return rotated


def compute_mid_slice_metrics(labels_4d: np.ndarray) -> dict[str, object]:
    mid_index = 1
    mid_ref = compute_aha_slice_reference(
        labels_4d,
        slice_index=mid_index,
        slice_type="mid",
    )

    label_slice = label_slice_yx(labels_4d, mid_index, mid_ref.frame_index)
    _, myo, _ = get_masks(label_slice)

    segment_names = get_segment_names(mid_ref.slice_type)
    bounds = sector_boundaries_from_anchor(mid_ref.anchor_angle, len(segment_names))
    sector_map = build_sector_map(
        mid_ref.lv_centroid[0],
        mid_ref.lv_centroid[1],
        myo,
        bounds,
    )
    sector_areas = compute_sector_areas(sector_map, len(segment_names))
    angles, wt_matrix, sector_ids, _, _ = compute_wt_matrix_and_sector_ids(
        labels_4d,
        mid_index,
        bounds,
        n_rays=360,
        ray_step=0.5,
        max_radius=80.0,
    )
    binned_wt = compute_aha_binned_wt(
        wt_matrix,
        sector_ids,
        n_sectors=len(segment_names),
    )

    peak_idx = int(np.nanargmax(binned_wt[:, 0]))
    anchor_distance = float(np.hypot(
        mid_ref.anchor_point[0] - mid_ref.lv_centroid[0],
        mid_ref.anchor_point[1] - mid_ref.lv_centroid[1],
    ))

    return {
        "slice_type": mid_ref.slice_type,
        "anchor_source": mid_ref.anchor_source,
        "angles": angles,
        "anchor_angle": float(mid_ref.anchor_angle),
        "anchor_distance": anchor_distance,
        "sector_areas": sector_areas,
        "binned_wt": binned_wt,
        "peak_segment_name": segment_names[peak_idx],
    }


def angle_delta(theta: float, reference: float) -> float:
    return float(wrap_signed_angle(theta - reference))


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_aha_reference_rotates_with_anatomy_under_rot90(k: int):
    base_stack = make_synthetic_stack()
    base_metrics = compute_mid_slice_metrics(base_stack)
    rotated_stack = rotate_stack_k90(base_stack, k)
    rotated_metrics = compute_mid_slice_metrics(rotated_stack)

    expected_shift = k * (np.pi / 2.0)

    assert rotated_metrics["slice_type"] == "mid"
    assert rotated_metrics["anchor_source"] == "direct"
    assert rotated_metrics["peak_segment_name"] == "Mid Anterior"
    assert base_metrics["peak_segment_name"] == "Mid Anterior"
    assert angle_delta(
        rotated_metrics["anchor_angle"],
        base_metrics["anchor_angle"],
    ) == pytest.approx(angle_delta(expected_shift, 0.0), abs=0.12)
    assert rotated_metrics["anchor_distance"] == pytest.approx(
        base_metrics["anchor_distance"],
        abs=1.5,
    )
    assert np.allclose(
        rotated_metrics["sector_areas"],
        base_metrics["sector_areas"],
        atol=8,
    )
    assert np.allclose(
        rotated_metrics["binned_wt"][:, 0],
        base_metrics["binned_wt"][:, 0],
        atol=1.0,
    )


def test_aha_reference_is_stable_under_nearest_neighbor_rotation():
    base_stack = make_synthetic_stack()
    base_metrics = compute_mid_slice_metrics(base_stack)
    rotated_stack = rotate_stack_nearest_neighbor(base_stack, angle_deg=30.0)
    rotated_metrics = compute_mid_slice_metrics(rotated_stack)

    base_area_fraction = base_metrics["sector_areas"] / np.sum(base_metrics["sector_areas"])
    rotated_area_fraction = rotated_metrics["sector_areas"] / np.sum(rotated_metrics["sector_areas"])

    assert rotated_metrics["slice_type"] == "mid"
    assert rotated_metrics["peak_segment_name"] == "Mid Anterior"
    assert angle_delta(
        rotated_metrics["anchor_angle"],
        base_metrics["anchor_angle"],
    ) == pytest.approx(np.deg2rad(30.0), abs=np.deg2rad(12.0))
    assert rotated_metrics["anchor_distance"] == pytest.approx(
        base_metrics["anchor_distance"],
        abs=2.5,
    )
    assert np.allclose(rotated_area_fraction, base_area_fraction, atol=0.08)
    assert np.allclose(
        rotated_metrics["binned_wt"][:, 0],
        base_metrics["binned_wt"][:, 0],
        atol=1.5,
    )
