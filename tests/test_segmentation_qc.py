from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cardionet.visualization.segmentation_qc import (
    _draw_label_contours,
    compute_ejection_fraction,
    compute_label_volumes_disk,
    compute_label_volumes_riemann,
    compute_lv_slice_quality_mask,
    plot_segmentations_per_slice,
    plot_volume_changes,
)


def test_compute_label_volumes_riemann_uses_physical_spacing():
    labels = np.zeros((10, 10, 3, 2), dtype=np.uint8)
    labels[:, :, :, 0] = 3
    labels[:5, :, :, 1] = 3

    volumes = compute_label_volumes_riemann(
        labels,
        label_value=3,
        spacing_mm=(2.0, 3.0, 5.0),
    )

    assert np.allclose(volumes, [9.0, 4.5])
    assert compute_ejection_fraction(volumes) == 50.0


def test_compute_label_volumes_disk_sums_slice_discs():
    labels = np.zeros((10, 10, 3, 1), dtype=np.uint8)
    labels[:2, :, 0, 0] = 3
    labels[:4, :, 1, 0] = 3
    labels[:6, :, 2, 0] = 3

    volumes = compute_label_volumes_disk(
        labels,
        label_value=3,
        spacing_mm=(2.0, 3.0, 5.0),
    )

    assert np.allclose(volumes, [3.6])


def test_compute_lv_slice_quality_mask_drops_bad_apical_slice():
    labels = np.zeros((20, 20, 5, 2), dtype=np.uint8)
    labels[:10, :10, 1, 0] = 3
    labels[:4, :10, 1, 1] = 3
    labels[:10, :10, 2, 0] = 3
    labels[:7, :10, 2, 1] = 3
    labels[:10, :10, 3, 0] = 3
    labels[:7, :10, 3, 1] = 3
    labels[:3, :10, 4, 0] = 3
    labels[:4, :10, 4, 1] = 3

    result = compute_lv_slice_quality_mask(
        labels,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.kept_indices == [1, 2, 3]
    assert result.dropped_indices == [0, 4]
    assert result.drop_reasons[4] == "es_area_exceeds_ed_area"


def test_plot_volume_changes_writes_png(tmp_path: Path):
    labels = np.zeros((16, 16, 4, 5), dtype=np.uint8)
    labels[..., 0] = 3
    labels[..., 1] = 2
    labels[..., 2] = 1

    out_path = tmp_path / "volume_qc.png"
    returned = plot_volume_changes(
        labels=labels,
        t_step=1,
        filepath=out_path,
        spacing_mm=(1.0, 1.0, 10.0),
    )

    assert returned.exists()
    assert returned == out_path


def test_draw_label_contours_does_not_add_filled_mask_images():
    label_slice = np.zeros((16, 16), dtype=np.uint8)
    label_slice[2:8, 2:8] = 1
    label_slice[6:14, 6:14] = 3

    fig, ax = plt.subplots()
    ax.imshow(np.zeros_like(label_slice), cmap="gray")
    _draw_label_contours(
        ax,
        label_slice,
        {
            1: np.array([1.0, 0.9, 0.0, 1.0]),
            3: np.array([1.0, 0.0, 0.0, 1.0]),
        },
    )

    assert len(ax.images) == 1
    assert len(ax.collections) > 0
    plt.close(fig)


def test_plot_segmentations_per_slice_writes_contour_gif(tmp_path: Path):
    images = np.zeros((16, 16, 1, 2), dtype=np.float32)
    labels = np.zeros((16, 16, 1, 2), dtype=np.uint8)
    labels[3:10, 3:10, 0, 0] = 1
    labels[5:13, 5:13, 0, 1] = 2

    paths = plot_segmentations_per_slice(
        images=images,
        labels=labels,
        t_step=1,
        output_dir=tmp_path,
        basename="contour_mask",
        label_colors={
            1: np.array([1.0, 0.9, 0.0, 1.0]),
            2: np.array([0.0, 0.85, 0.25, 1.0]),
        },
    )

    assert len(paths) == 1
    assert paths[0].exists()


def test_plot_segmentations_per_slice_can_limit_source_slices(tmp_path: Path):
    images = np.zeros((16, 16, 3, 2), dtype=np.float32)
    labels = np.zeros((16, 16, 3, 2), dtype=np.uint8)
    labels[3:10, 3:10, 2, :] = 3

    paths = plot_segmentations_per_slice(
        images=images,
        labels=labels,
        t_step=1,
        output_dir=tmp_path,
        basename="selected_slice",
        slice_indices=(2,),
    )

    assert len(paths) == 1
    assert paths[0].name == "selected_slice_slice_02.gif"
    assert paths[0].exists()
