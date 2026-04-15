from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from cardionet.segmentation.io import load_cine_nifti


def test_load_cine_nifti_returns_4d_array(tmp_path: Path):
    arr = np.zeros((4, 3, 2, 1), dtype=np.float32)
    image = sitk.GetImageFromArray(arr)
    nifti_path = tmp_path / "cine.nii.gz"
    sitk.WriteImage(image, str(nifti_path))

    loaded = load_cine_nifti(nifti_path)
    assert loaded.ndim == 4
    assert loaded.shape[-1] == 1


def test_load_cine_nifti_raises_if_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_cine_nifti(tmp_path / "missing.nii.gz")
