from __future__ import annotations

import numpy as np

from cardionet.geometry.aha_reference import wrap_angle


def get_segment_names(ring_type: str) -> list[str]:
    """
    Return AHA segment names for a ring type.
    """
    if ring_type == "basal":
        return [
            "Basal Anteroseptal",
            "Basal Anterior",
            "Basal Anterolateral",
            "Basal Inferolateral",
            "Basal Inferior",
            "Basal Inferoseptal",
        ]

    if ring_type == "mid":
        return [
            "Mid Anteroseptal",
            "Mid Anterior",
            "Mid Anterolateral",
            "Mid Inferolateral",
            "Mid Inferior",
            "Mid Inferoseptal",
        ]

    if ring_type == "apical":
        return [
            "Apical Septal",
            "Apical Anterior",
            "Apical Lateral",
            "Apical Inferior",
        ]

    raise ValueError(f"Unknown ring_type: {ring_type}")


def sector_boundaries_from_anchor(anchor_angle: float, n_sectors: int) -> np.ndarray:
    """
    Create evenly spaced sector boundaries centered around the anchor direction.
    """
    width = 2 * np.pi / n_sectors
    start = anchor_angle - width / 2
    bounds = np.array([start + k * width for k in range(n_sectors + 1)])
    return bounds


def assign_sector_index(theta: float, bounds: np.ndarray) -> int:
    """
    Assign a ray angle to a sector index using wrapped angular boundaries.
    """
    n_sectors = len(bounds) - 1
    width = 2 * np.pi / n_sectors
    start = bounds[0]

    theta_rel = wrap_angle(theta - start)
    idx = int(theta_rel // width)

    if idx == n_sectors:
        idx = n_sectors - 1

    return idx


def build_sector_map(
    yc: float,
    xc: float,
    myo: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """
    Assign each MYO pixel to an AHA sector. Non-MYO pixels remain -1.
    """
    sector_map = np.full(myo.shape, -1, dtype=int)
    coords = np.argwhere(myo)

    for y, x in coords:
        theta = np.arctan2(y - yc, x - xc)
        idx = assign_sector_index(theta, bounds)
        sector_map[y, x] = idx

    return sector_map