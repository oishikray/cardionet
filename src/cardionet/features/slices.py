from __future__ import annotations

import numpy as np


def label_slice_yx(
    labels_4d: np.ndarray,
    slice_index: int,
    frame_index: int,
) -> np.ndarray:
    """
    Return one 2D label slice in row/column ``(y, x)`` order.

    CardioNet stores cine images and predicted labels in canonical
    ``(x, y, z, t)`` order for model compatibility. Geometry, ray-casting, and
    Matplotlib-style overlays operate on 2D arrays as ``(row, column)``, i.e.
    ``(y, x)``. This helper is the explicit boundary between those conventions.
    """
    if labels_4d.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels_4d.shape}")

    _, _, num_slices, num_frames = labels_4d.shape
    if slice_index < 0 or slice_index >= num_slices:
        raise IndexError(f"slice_index {slice_index} is out of bounds for {num_slices} slices")
    if frame_index < 0 or frame_index >= num_frames:
        raise IndexError(f"frame_index {frame_index} is out of bounds for {num_frames} frames")

    return labels_4d[:, :, slice_index, frame_index].T
