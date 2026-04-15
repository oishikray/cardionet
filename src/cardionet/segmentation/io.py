from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813


def load_cine_nifti(path: str | Path) -> np.ndarray:
    """
    Load a 4D cine NIfTI and return an array shaped (x, y, z, t).

    Notes
    -----
    This currently preserves the old CineMA/ACDC script behavior:
    SimpleITK output is transposed blindly to recover (x, y, z, t).

    This is acceptable only as long as:
    - the input data follows the same preprocessing conventions
    - validation confirms the resulting shape semantics

    Parameters
    ----------
    path
        Path to the cine NIfTI file.

    Returns
    -------
    np.ndarray
        Cine volume shaped (x, y, z, t).

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the loaded array is not 4D after transpose.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input cine file not found: {path}")

    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    arr = np.transpose(arr)

    if arr.ndim == 3:
        arr = arr[..., None]

    if arr.ndim != 4:
        raise ValueError(
            f"Expected 4D cine stack after transpose, got shape {arr.shape}"
        )

    return arr


def save_inference_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    basename: str,
    *,
    save_inputs: bool = True,
    save_predictions: bool = True,
    image_suffix: str = "_images.npy",
    labels_suffix: str = "_pred_labels.npy",
) -> tuple[Path | None, Path | None]:
    """
    Save inference inputs and predicted labels as .npy files.

    Returns
    -------
    tuple[Path | None, Path | None]
        Paths to saved images and labels. Entries are None when saving is disabled.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_path = output_dir / f"{basename}{image_suffix}" if save_inputs else None
    labels_path = (
        output_dir / f"{basename}{labels_suffix}" if save_predictions else None
    )

    if images_path is not None:
        np.save(images_path, images.astype(np.float32))

    if labels_path is not None:
        np.save(labels_path, labels.astype(np.uint8))

    return images_path, labels_path
