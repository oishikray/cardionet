from pathlib import Path

import numpy as np

from cardionet.visualization.aha_qc import save_debug_wt_plot


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