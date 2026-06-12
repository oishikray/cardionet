from pathlib import Path

import numpy as np

from cardionet.visualization.canonical import (
    bullseye_wedge_angles,
    save_delta_t_plots,
    save_lv_volume_curve,
    save_metric_bullseye,
    save_ring_time_series_plots,
    segment_color,
)
from cardionet.features.delta_t import compute_delta_t


def test_segment_color_is_consistent_across_ring_prefixes():
    assert segment_color("Basal Anterolateral") == segment_color("Mid Anterolateral")
    assert segment_color("Basal Inferior") == segment_color("Apical Inferior")


def test_bullseye_wedge_angles_advance_counterclockwise():
    _, _, center_1 = bullseye_wedge_angles(index=0, n_segments=6)
    _, _, center_2 = bullseye_wedge_angles(index=1, n_segments=6)
    _, _, center_4 = bullseye_wedge_angles(index=3, n_segments=6)

    assert center_1 == 90.0
    assert center_2 == 150.0
    assert center_4 == 270.0


def test_save_metric_bullseye_writes_png(tmp_path: Path):
    outpath = save_metric_bullseye(
        {
            "basal": np.arange(1, 7, dtype=float),
            "mid": np.arange(7, 13, dtype=float),
            "apical": np.arange(13, 17, dtype=float),
        },
        tmp_path,
        patient_id="patient001",
        metric_name="Peak NWT",
    )
    assert outpath.exists()


def test_save_ring_time_series_plots_writes_one_file_per_ring(tmp_path: Path):
    series = {
        "basal": np.tile(np.linspace(1.0, 1.5, 5), (6, 1)),
        "mid": np.tile(np.linspace(0.0, 0.4, 5), (6, 1)),
        "apical": np.tile(np.linspace(0.0, 0.3, 5), (4, 1)),
    }
    paths = save_ring_time_series_plots(
        series,
        tmp_path,
        patient_id="patient001",
        metric_name="Radial Strain",
        ed_frame=0,
        es_frame=3,
        baseline_value=0.0,
        include_std_legend=True,
    )
    assert len(paths) == 3
    assert all(path.exists() for path in paths)


def test_save_lv_volume_curve_writes_png(tmp_path: Path):
    labels = np.zeros((12, 12, 3, 4), dtype=np.uint8)
    labels[:10, :10, :, 0] = 3
    labels[:8, :8, :, 1] = 3
    labels[:6, :6, :, 2] = 3
    labels[:8, :8, :, 3] = 3

    outpath = save_lv_volume_curve(
        labels,
        tmp_path,
        patient_id="patient001",
        spacing_mm=(1.0, 1.0, 10.0),
    )
    assert outpath.exists()


def test_save_delta_t_plots_writes_one_file_per_ring(tmp_path: Path):
    nwt_by_ring = {
        "basal": np.tile(np.linspace(1.0, 1.5, 5), (6, 1)),
    }
    delta_t = compute_delta_t(nwt_by_ring, es_frame=2, frame_interval_ms=30.0)

    paths = save_delta_t_plots(delta_t, tmp_path, patient_id="patient001")

    assert len(paths) == 1
    assert paths[0].exists()
