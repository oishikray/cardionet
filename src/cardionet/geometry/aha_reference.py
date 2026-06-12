from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass(slots=True)
class RVMyocardiumContactGeometry:
    """
    Geometry derived from the RV-MYO contact region in one short-axis slice.

    The ``anchor_point`` represents the clockwise insertion/contact endpoint
    relative to the RV direction. Downstream AHA sector construction treats
    this point as the boundary with anterior on the clockwise side.
    """

    lv_centroid: tuple[float, float]
    rv_centroid: tuple[float, float]
    contact_centroid: tuple[float, float]
    anchor_point: tuple[float, float]
    inferior_point: tuple[float, float]
    anchor_angle: float
    inferior_angle: float
    rv_direction_angle: float
    contact_mask: np.ndarray


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


def compute_mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """
    Compute the centroid of a binary mask in (y, x) coordinates.

    Returns ``(nan, nan)`` if the mask is empty.
    """
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return np.nan, np.nan

    yc = coords[:, 0].mean()
    xc = coords[:, 1].mean()
    return float(yc), float(xc)


def compute_lv_centroid(label_slice: np.ndarray) -> tuple[float, float]:
    """
    Compute LV centroid in (y, x) coordinates from a 2D label slice.
    Returns (nan, nan) if LV is absent.
    """
    _, _, lv = get_masks(label_slice)
    return compute_mask_centroid(lv)


def compute_rv_centroid(label_slice: np.ndarray) -> tuple[float, float]:
    """
    Compute RV centroid in (y, x) coordinates from a 2D label slice.
    Returns ``(nan, nan)`` if RV is absent.
    """
    rv, _, _ = get_masks(label_slice)
    return compute_mask_centroid(rv)


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
    return compute_mask_centroid(contact_mask)


def angle_from_centroid(yc: float, xc: float, y: float, x: float) -> float:
    """
    Angle from centroid (yc, xc) to point (y, x), in radians.
    """
    return float(np.arctan2(y - yc, x - xc))


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap angle(s) into ``[0, 2*pi)``.
    """
    return (theta + 2 * np.pi) % (2 * np.pi)


def wrap_signed_angle(theta: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap angle(s) into ``[-pi, pi)``.
    """
    return wrap_angle(np.asarray(theta) + np.pi) - np.pi


def circular_mean(angles: np.ndarray | list[float]) -> float:
    """
    Compute the circular mean of one or more angles in radians.
    """
    values = np.asarray(angles, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return float("nan")

    return float(np.arctan2(np.sin(finite).mean(), np.cos(finite).mean()))


def point_from_angle(
    yc: float,
    xc: float,
    angle: float,
    radius: float,
) -> tuple[float, float]:
    """
    Convert a centroid, angle, and radius back into image coordinates.
    """
    return (
        float(yc + radius * np.sin(angle)),
        float(xc + radius * np.cos(angle)),
    )


def _select_contact_endpoint(
    contact_mask: np.ndarray,
    *,
    lv_centroid: tuple[float, float],
    rv_direction_angle: float,
    choose_clockwise: bool,
    angle_window: float = np.deg2rad(7.5),
) -> tuple[float, float]:
    """
    Select one RV insertion endpoint from the contact arc.

    Endpoints are chosen by ranking contact pixels by their signed angular
    offset from the LV->RV direction. The clockwise endpoint is the AHA anchor.
    """
    coords = np.argwhere(contact_mask).astype(float)
    yc, xc = lv_centroid
    angles = np.arctan2(coords[:, 0] - yc, coords[:, 1] - xc)
    rel = np.asarray(wrap_signed_angle(angles - rv_direction_angle), dtype=float)

    if choose_clockwise:
        target = float(rel.max())
        keep = rel >= (target - angle_window)
    else:
        target = float(rel.min())
        keep = rel <= (target + angle_window)

    endpoint_coords = coords[keep]
    if endpoint_coords.size == 0:
        endpoint_coords = coords[[
            int(np.argmax(rel) if choose_clockwise else np.argmin(rel))
        ]]

    yc_endpoint = float(endpoint_coords[:, 0].mean())
    xc_endpoint = float(endpoint_coords[:, 1].mean())
    return yc_endpoint, xc_endpoint


def compute_rv_myo_contact_geometry(
    label_slice: np.ndarray,
    *,
    dilation_iterations: int = 1,
    angle_window: float = np.deg2rad(7.5),
) -> RVMyocardiumContactGeometry:
    """
    Compute RV-contact-derived AHA reference geometry for one 2D label slice.

    The anchor is taken from the clockwise RV insertion point, which makes the
    AHA sector reference rotate with the anatomy instead of the image axes.
    """
    rv, myo, _ = get_masks(label_slice)
    lv_centroid = compute_lv_centroid(label_slice)
    rv_centroid = compute_mask_centroid(rv)

    if not np.isfinite(lv_centroid[0]) or not np.isfinite(lv_centroid[1]):
        raise RuntimeError("No LV cavity found for AHA reference geometry.")

    if not np.isfinite(rv_centroid[0]) or not np.isfinite(rv_centroid[1]):
        raise RuntimeError("No RV cavity found for AHA reference geometry.")

    contact_mask = find_rv_myo_contact_region(
        rv,
        myo,
        dilation_iterations=dilation_iterations,
    )
    if int(contact_mask.sum()) == 0:
        raise RuntimeError("No RV-MYO contact region found.")

    contact_centroid = compute_contact_centroid(contact_mask)
    rv_direction_angle = angle_from_centroid(
        lv_centroid[0],
        lv_centroid[1],
        rv_centroid[0],
        rv_centroid[1],
    )
    anchor_point = _select_contact_endpoint(
        contact_mask,
        lv_centroid=lv_centroid,
        rv_direction_angle=rv_direction_angle,
        choose_clockwise=True,
        angle_window=angle_window,
    )
    inferior_point = _select_contact_endpoint(
        contact_mask,
        lv_centroid=lv_centroid,
        rv_direction_angle=rv_direction_angle,
        choose_clockwise=False,
        angle_window=angle_window,
    )

    anchor_angle = angle_from_centroid(
        lv_centroid[0],
        lv_centroid[1],
        anchor_point[0],
        anchor_point[1],
    )
    inferior_angle = angle_from_centroid(
        lv_centroid[0],
        lv_centroid[1],
        inferior_point[0],
        inferior_point[1],
    )

    return RVMyocardiumContactGeometry(
        lv_centroid=lv_centroid,
        rv_centroid=rv_centroid,
        contact_centroid=contact_centroid,
        anchor_point=anchor_point,
        inferior_point=inferior_point,
        anchor_angle=anchor_angle,
        inferior_angle=inferior_angle,
        rv_direction_angle=rv_direction_angle,
        contact_mask=contact_mask,
    )


def compute_anchor_angle_from_rv_contact(
    label_slice: np.ndarray,
) -> tuple[float, tuple[float, float], tuple[float, float], np.ndarray]:
    """
    Backward-compatible helper returning the contact-centroid anchor summary.

    Newer code should prefer ``compute_rv_myo_contact_geometry`` so it can use
    the anatomically oriented anchor point instead of the contact centroid.
    """
    geometry = compute_rv_myo_contact_geometry(label_slice)
    return (
        geometry.anchor_angle,
        geometry.lv_centroid,
        geometry.contact_centroid,
        geometry.contact_mask,
    )
