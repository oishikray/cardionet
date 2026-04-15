from pathlib import Path

import numpy as np

from cardionet.visualization.segmentation_qc import plot_volume_changes


def test_plot_volume_changes_writes_png(tmp_path: Path):
    labels = np.zeros((16, 16, 4, 5), dtype=np.uint8)
    labels[..., 0] = 3
    labels[..., 1] = 2
    labels[..., 2] = 1

    out_path = tmp_path / "volume_qc.png"
    returned = plot_volume_changes(labels=labels, t_step=1, filepath=out_path)

    assert returned.exists()
    assert returned == out_path