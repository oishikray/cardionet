from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from cardionet.geometry.aha_reference import (
    compute_rv_myo_contact_geometry,
    wrap_angle,
)
from cardionet.features.slices import label_slice_yx


@dataclass(frozen=True, slots=True)
class AHASegment:
    """Canonical AHA segment metadata."""

    number: int
    name: str
    ring_type: str
    sector_index: int


AHA_SEGMENTS: tuple[AHASegment, ...] = (
    AHASegment(1, "Basal Anterior", "basal", 0),
    AHASegment(2, "Basal Anteroseptal", "basal", 1),
    AHASegment(3, "Basal Inferoseptal", "basal", 2),
    AHASegment(4, "Basal Inferior", "basal", 3),
    AHASegment(5, "Basal Inferolateral", "basal", 4),
    AHASegment(6, "Basal Anterolateral", "basal", 5),
    AHASegment(7, "Mid Anterior", "mid", 0),
    AHASegment(8, "Mid Anteroseptal", "mid", 1),
    AHASegment(9, "Mid Inferoseptal", "mid", 2),
    AHASegment(10, "Mid Inferior", "mid", 3),
    AHASegment(11, "Mid Inferolateral", "mid", 4),
    AHASegment(12, "Mid Anterolateral", "mid", 5),
    AHASegment(13, "Apical Anterior", "apical", 0),
    AHASegment(14, "Apical Septal", "apical", 1),
    AHASegment(15, "Apical Inferior", "apical", 2),
    AHASegment(16, "Apical Lateral", "apical", 3),
    AHASegment(17, "Apex", "apex", 0),
)

AHA_SEGMENTS_BY_RING: dict[str, tuple[AHASegment, ...]] = {
    ring_type: tuple(segment for segment in AHA_SEGMENTS if segment.ring_type == ring_type)
    for ring_type in ("basal", "mid", "apical", "apex")
}

VALID_AHA_SLICE_TYPES = tuple(AHA_SEGMENTS_BY_RING)

@dataclass(slots=True)
class AHASliceReference:
    """Resolved AHA reference geometry for one short-axis slice."""

    slice_index: int
    slice_type: str
    frame_index: int
    anchor_angle: float
    anchor_point: tuple[float, float]
    lv_centroid: tuple[float, float]
    contact_centroid: tuple[float, float]
    anchor_source: str


def normalize_aha_slice_type(
    slice_type: str | None,
    *,
    field_name: str = "slice_type",
) -> str:
    """Validate and normalize a clinician-selected AHA slice type."""
    if slice_type is None:
        raise ValueError(
            f"{field_name} is required for AHA analysis. Set it to one of "
            f"{', '.join(VALID_AHA_SLICE_TYPES)}."
        )

    normalized = str(slice_type).strip().lower()
    if normalized in {"", "auto", "none", "null"}:
        raise ValueError(
            f"{field_name} must be explicit for AHA analysis; automatic slice "
            "typing is no longer supported."
        )
    if normalized not in AHA_SEGMENTS_BY_RING:
        raise ValueError(
            f"Unknown {field_name}: {slice_type!r}. Expected one of "
            f"{', '.join(VALID_AHA_SLICE_TYPES)}."
        )
    return normalized


def resolve_manual_slice_index(slice_index: int | None, *, num_slices: int) -> int:
    """Validate a clinician-selected slice index."""
    if slice_index is None:
        raise ValueError(
            "slice_index is required for AHA analysis. Set it to the clinician-selected "
            "short-axis slice index."
        )

    resolved = int(slice_index)
    if resolved < 0 or resolved >= num_slices:
        raise IndexError(f"slice_index {resolved} is out of bounds for {num_slices} slices")
    return resolved


def resolve_manual_slice_selection(
    labels_4d: np.ndarray,
    *,
    slice_index: int | None,
    slice_type: str | None,
) -> tuple[int, str]:
    """Validate explicit AHA slice metadata against a label volume."""
    if labels_4d.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels_4d.shape}")

    return (
        resolve_manual_slice_index(slice_index, num_slices=labels_4d.shape[2]),
        normalize_aha_slice_type(slice_type),
    )


def get_segments(ring_type: str) -> tuple[AHASegment, ...]:
    """Return canonical AHA segment objects for one ring type."""
    normalized = normalize_aha_slice_type(ring_type, field_name="ring_type")
    return AHA_SEGMENTS_BY_RING[normalized]


def get_segment_names(ring_type: str) -> list[str]:
    """
    Return canonical AHA segment names for a ring type.

    Segment index order follows the standard AHA circumferential sequence:
    anterior first, septal sectors on the RV side, inferior, then lateral
    free-wall sectors.
    """
    return [segment.name for segment in get_segments(ring_type)]


def get_segment_numbers(ring_type: str) -> list[int]:
    """Return global AHA segment numbers for a ring type."""
    return [segment.number for segment in get_segments(ring_type)]


def get_num_segments(ring_type: str) -> int:
    """Return the configured number of angular AHA sectors for one ring."""
    return len(get_segment_names(ring_type))


def sector_boundaries_from_anchor(anchor_angle: float, n_sectors: int) -> np.ndarray:
    """
    Create evenly spaced sector boundaries from an inferred AHA anchor.

    The direct RV-MYO contact anchor is treated as the boundary with the
    anterior sector on the clockwise side and the next AHA sector on the
    counterclockwise side. Boundary angles are listed counterclockwise in the
    displayed short-axis view.

    Image-space angles use array coordinates, where increasing angle follows
    image-clockwise rotation.
    """
    width = 2 * np.pi / n_sectors
    bounds = np.array([anchor_angle - k * width for k in range(n_sectors + 1)])
    return bounds


def assign_sector_index(theta: float, bounds: np.ndarray) -> int:
    """
    Assign a ray angle to an anatomical sector index.

    AHA sector indices increase counterclockwise in the displayed short-axis
    view. Because image-space angles increase clockwise, counterclockwise
    angular distance is measured as ``start - theta``.

    The RV-contact anchor is the boundary after anterior in counterclockwise
    order, so raw counterclockwise bin 0 maps to AHA sector 1 and the raw bin
    immediately clockwise of the anchor maps to sector 0.
    """
    n_sectors = len(bounds) - 1
    width = 2 * np.pi / n_sectors
    start = bounds[0]

    theta_rel = wrap_angle(start - theta)
    raw_idx = int(theta_rel // width)

    if raw_idx == n_sectors:
        raw_idx = n_sectors - 1

    return (raw_idx + 1) % n_sectors


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


def compute_sector_areas(sector_map: np.ndarray, n_sectors: int) -> np.ndarray:
    """Count the number of MYO pixels assigned to each sector."""
    valid = sector_map[sector_map >= 0]
    return np.bincount(valid, minlength=n_sectors).astype(int)


def infer_slice_types(labels_4d: np.ndarray) -> list[str | None]:
    """Deprecated. Automatic AHA slice typing is no longer supported."""
    raise RuntimeError(
        "Automatic AHA slice typing is no longer supported. Provide explicit "
        "clinician-selected slice_index and slice_type metadata."
    )


def _coerce_slice_type_map(
    labels_4d: np.ndarray,
    slice_types: Mapping[int, str] | Sequence[str | None],
) -> list[str | None]:
    if labels_4d.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels_4d.shape}")

    _, _, num_slices, _ = labels_4d.shape
    resolved: list[str | None] = [None] * num_slices

    if isinstance(slice_types, Mapping):
        items = slice_types.items()
    else:
        if len(slice_types) != num_slices:
            raise ValueError(
                f"slice_types length {len(slice_types)} does not match "
                f"{num_slices} slices."
            )
        items = enumerate(slice_types)

    for slice_index, slice_type in items:
        if slice_type is None:
            continue
        resolved_index = resolve_manual_slice_index(int(slice_index), num_slices=num_slices)
        resolved[resolved_index] = normalize_aha_slice_type(slice_type)

    return resolved


def _resolve_geometry_frame(
    labels_4d: np.ndarray,
    slice_index: int,
    frame_index: int | None,
) -> int:
    if frame_index is not None:
        return int(frame_index)

    lv_areas = np.array([
        np.sum(labels_4d[:, :, slice_index, frame] == 3)
        for frame in range(labels_4d.shape[-1])
    ])

    if not np.any(lv_areas > 0):
        return 0

    return int(np.argmax(lv_areas))


def compute_aha_slice_reference(
    labels_4d: np.ndarray,
    *,
    slice_index: int,
    slice_type: str,
    frame_index: int | None = None,
    dilation_iterations: int = 1,
) -> AHASliceReference:
    """
    Resolve direct AHA reference geometry for one clinician-selected slice.

    This does not propagate anchors from neighboring slices. If the selected
    slice cannot provide an RV-MYO contact anchor, the caller gets a clear
    failure instead of silently guessing the orientation.
    """
    resolved_slice_index, resolved_slice_type = resolve_manual_slice_selection(
        labels_4d,
        slice_index=slice_index,
        slice_type=slice_type,
    )
    resolved_frame = _resolve_geometry_frame(labels_4d, resolved_slice_index, frame_index)
    label_slice = label_slice_yx(labels_4d, resolved_slice_index, resolved_frame)

    try:
        geometry = compute_rv_myo_contact_geometry(
            label_slice,
            dilation_iterations=dilation_iterations,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not compute direct AHA reference geometry for "
            f"slice {resolved_slice_index} ({resolved_slice_type}) at frame "
            f"{resolved_frame}. Check that the selected slice has LV, MYO, and "
            "RV contact anatomy in the segmentation, or choose a different "
            "overlay/reference frame."
        ) from exc

    return AHASliceReference(
        slice_index=resolved_slice_index,
        slice_type=resolved_slice_type,
        frame_index=resolved_frame,
        anchor_angle=geometry.anchor_angle,
        anchor_point=geometry.anchor_point,
        lv_centroid=geometry.lv_centroid,
        contact_centroid=geometry.contact_centroid,
        anchor_source="direct",
    )


def resolve_aha_slice_references(
    labels_4d: np.ndarray,
    *,
    slice_types: Mapping[int, str] | Sequence[str | None],
    frame_index: int | None = None,
    dilation_iterations: int = 1,
) -> list[AHASliceReference | None]:
    """
    Resolve AHA slice references from explicit clinician-selected slice types.

    The function name is retained for compatibility, but it no longer infers
    slice roles or propagates anchors across the stack.
    """
    if labels_4d.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels_4d.shape}")

    _, _, num_slices, _ = labels_4d.shape
    resolved_slice_types = _coerce_slice_type_map(labels_4d, slice_types)
    slice_refs: list[AHASliceReference | None] = [None] * num_slices

    for slice_index, slice_type in enumerate(resolved_slice_types):
        if slice_type is None:
            continue
        slice_refs[slice_index] = compute_aha_slice_reference(
            labels_4d,
            slice_index=slice_index,
            slice_type=slice_type,
            frame_index=frame_index,
            dilation_iterations=dilation_iterations,
        )

    return slice_refs


def infer_aha_slice_references(
    labels_4d: np.ndarray,
    *,
    slice_types: Mapping[int, str] | Sequence[str | None],
    frame_index: int | None = None,
    dilation_iterations: int = 1,
) -> list[AHASliceReference | None]:
    """Deprecated alias for resolve_aha_slice_references."""
    return resolve_aha_slice_references(
        labels_4d,
        slice_types=slice_types,
        frame_index=frame_index,
        dilation_iterations=dilation_iterations,
    )
