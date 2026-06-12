from pathlib import Path

import numpy as np

from cardionet.features.stack_aha import (
    analyze_stack_aha,
    choose_global_ed_frame_from_lv_volume,
)
from cardionet.geometry.aha_reference import wrap_signed_angle
from cardionet.visualization.aha_qc import save_stack_bullseye_summary


def make_synthetic_slice(
    *,
    size: int = 96,
    lv_radius: float = 10.0,
    outer_radius: float = 18.0,
    abnormality_angle: float = np.deg2rad(-50.0),
    abnormality_width: float = np.deg2rad(22.0),
    abnormality_gain: float = 3.0,
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
    wedge = np.abs(wrap_signed_angle(theta - abnormality_angle)) <= abnormality_width
    outer[wedge] += abnormality_gain

    lv = radius <= lv_radius
    myo = (radius > lv_radius) & (radius <= outer)

    labels = np.zeros((size, size), dtype=np.uint8)
    labels[myo] = 2
    labels[lv] = 3

    if include_rv:
        rv_center_y = yc - (outer_radius + 7.0)
        rv_center_x = xc
        rv = (((y - rv_center_y) / 7.0) ** 2 + ((x - rv_center_x) / 10.0) ** 2 <= 1.0)
        labels[rv & ~lv] = 1

    return labels


def make_synthetic_stack(num_frames: int = 2) -> np.ndarray:
    basal = make_synthetic_slice(outer_radius=20.0, abnormality_gain=4.0, include_rv=True)
    mid = make_synthetic_slice(outer_radius=18.0, abnormality_gain=3.0, include_rv=True)
    apical = make_synthetic_slice(outer_radius=14.0, abnormality_gain=2.0, include_rv=True)
    base_stack = np.stack([basal, mid, apical], axis=2)

    labels = np.zeros((*base_stack.shape, num_frames), dtype=np.uint8)
    labels[:, :, :, 0] = base_stack
    labels[:, :, :, 1] = np.where(base_stack == 3, 0, base_stack)
    return labels


def test_choose_global_ed_frame_from_lv_volume_prefers_fuller_frame():
    labels = make_synthetic_stack(num_frames=2)
    ed_frame = choose_global_ed_frame_from_lv_volume(labels)
    assert ed_frame == 0


def test_analyze_stack_aha_reports_all_chunks():
    labels = make_synthetic_stack(num_frames=2)
    stack_features = analyze_stack_aha(
        labels,
        slice_types={0: "basal", 1: "mid", 2: "apical"},
        n_rays=180,
        ray_step=0.5,
        max_radius=60.0,
    )

    assert stack_features.global_ed_frame == 0
    assert stack_features.overlay_frame == 0
    assert [feature.slice_type for feature in stack_features.slice_features] == [
        "basal",
        "mid",
        "apical",
    ]
    assert set(stack_features.chunk_features) == {"basal", "mid", "apical"}

    apical = stack_features.chunk_features["apical"]
    assert apical.slice_indices == (2,)
    assert apical.aggregated_wt.shape == (4, labels.shape[-1])
    assert apical.aggregated_nwt.shape == (4, labels.shape[-1])
    assert apical.weighted_mean_nwt_curve.shape == (labels.shape[-1],)


def test_save_stack_bullseye_summary_writes_file(tmp_path: Path):
    labels = make_synthetic_stack(num_frames=2)
    stack_features = analyze_stack_aha(
        labels,
        slice_types={0: "basal", 1: "mid", 2: "apical"},
        n_rays=180,
        ray_step=0.5,
        max_radius=60.0,
    )

    outpath = save_stack_bullseye_summary(
        stack_features,
        tmp_path,
        patient_id="patient999",
    )

    assert outpath.exists()
