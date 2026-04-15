from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def get_masks(label_slice: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return binary masks for RV, MYO, LV from a 2D label slice.

    Current label convention:
    - 1 -> RV
    - 2 -> MYO
    - 3 -> LV
    """
    rv = label_slice == 1
    myo = label_slice == 2
    lv = label_slice == 3
    return rv, myo, lv


def compute_lv_centroid(label_slice: np.ndarray) -> tuple[float, float]:
    """
    Compute LV centroid in (y, x) coordinates from a 2D label slice.
    Returns (nan, nan) if LV is absent.
    """
    _, _, lv = get_masks(label_slice)
    coords = np.argwhere(lv)

    if len(coords) == 0:
        return np.nan, np.nan

    yc = coords[:, 0].mean()
    xc = coords[:, 1].mean()
    return yc, xc


def find_rv_myo_contact_region(
    rv: np.ndarray,
    myo: np.ndarray,
    *,
    dilation_iterations: int = 1,
) -> np.ndarray:
    """
    Estimate RV-MYO contact region by dilating RV and intersecting with MYO.
    """
    rv_dil = binary_dilation(rv, iterations=dilation_iterations)
    contact = myo & rv_dil
    return contact


def compute_contact_centroid(contact_mask: np.ndarray) -> tuple[float, float]:
    """
    Compute centroid of a binary contact mask in (y, x) coordinates.
    Returns (nan, nan) if mask is empty.
    """
    coords = np.argwhere(contact_mask)

    if len(coords) == 0:
        return np.nan, np.nan

    yc = coords[:, 0].mean()
    xc = coords[:, 1].mean()
    return yc, xc


def angle_from_centroid(yc: float, xc: float, y: float, x: float) -> float:
    """
    Angle from centroid (yc, xc) to point (y, x), in radians.
    """
    return float(np.arctan2(y - yc, x - xc))


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap angle(s) into [0, 2*pi).
    """
    return (theta + 2 * np.pi) % (2 * np.pi)


def compute_anchor_angle_from_rv_contact(
    label_slice: np.ndarray,
) -> tuple[float, tuple[float, float], tuple[float, float], np.ndarray]:
    """
    Compute AHA anchor angle using the RV-MYO contact centroid relative to LV centroid.

    Returns
    -------
    tuple
        (
            anchor_angle,
            lv_centroid,
            contact_centroid,
            contact_mask,
        )

    Raises
    ------
    RuntimeError
        If no RV-MYO contact region is found.
    """
    rv, myo, _ = get_masks(label_slice)

    lv_centroid = compute_lv_centroid(label_slice)
    contact_mask = find_rv_myo_contact_region(rv, myo)

    if int(contact_mask.sum()) == 0:
        raise RuntimeError("No RV-MYO contact region found.")

    contact_centroid = compute_contact_centroid(contact_mask)
    anchor_angle = angle_from_centroid(
        lv_centroid[0],
        lv_centroid[1],
        contact_centroid[0],
        contact_centroid[1],
    )

    return anchor_angle, lv_centroid, contact_centroid, contact_mask