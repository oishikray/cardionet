from __future__ import annotations

from pathlib import Path

import numpy as np


def load_label_volume(path: str | Path) -> np.ndarray:
    """
    Load a 4D label volume from .npy.

    Expected shape: (x, y, z, t)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    labels = np.load(path)

    if labels.ndim != 4:
        raise ValueError(f"Expected labels shape (x, y, z, t), got {labels.shape}")

    return labels
