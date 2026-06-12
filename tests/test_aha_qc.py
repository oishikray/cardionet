from pathlib import Path

import numpy as np

from cardionet.visualization.aha_qc import bullseye_wedge_angles, save_debug_wt_plot


def test_save_debug_wt_plot_writes_file(tmp_path: Path):
    wt = np.linspace(1.0, 2.0, 20)
    wt_smooth = wt.copy()

    outpath = save_debug_wt_plot(
        wt_frame=wt,
        wt_frame_smooth=wt_smooth,
        ed_frame=3,
        outdir=tmp_path,
    )

    assert outpath.exists()

def test_bullseye_wedge_angles_center_anterior_at_top():
    theta1, theta2, center = bullseye_wedge_angles(index=0, n_segments=6)
    assert theta1 == 60.0
    assert theta2 == 120.0
    assert center == 90.0


def test_bullseye_wedge_angles_puts_inferior_at_bottom():
    _, _, center = bullseye_wedge_angles(index=3, n_segments=6)
    assert center == 270.0
