from pathlib import Path

import numpy as np
import pytest

from cardionet.io.labels import load_label_volume


def test_load_label_volume_ok(tmp_path: Path):
    arr = np.zeros((16, 16, 4, 5), dtype=np.uint8)
    path = tmp_path / "labels.npy"
    np.save(path, arr)

    loaded = load_label_volume(path)
    assert loaded.shape == arr.shape


def test_load_label_volume_raises_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_label_volume(tmp_path / "missing.npy")


def test_load_label_volume_raises_on_wrong_dim(tmp_path: Path):
    arr = np.zeros((16, 16, 4), dtype=np.uint8)
    path = tmp_path / "labels.npy"
    np.save(path, arr)

    with pytest.raises(ValueError):
        load_label_volume(path)