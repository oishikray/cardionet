"""Input/output utilities and artifact discovery helpers."""

from .artifacts import SegmentationPair, attach_input_metadata, discover_segmentation_pairs
from .common import as_path, normalize_patient_id, normalize_slice_type
from .selection import (
    SelectedSlice,
    SelectedSlicesByPatient,
    load_data_index_frame,
    parse_selected_slices,
    resolve_data_index_path,
    selected_slices_from_index_frame,
)

__all__ = [
    "SegmentationPair",
    "SelectedSlice",
    "SelectedSlicesByPatient",
    "as_path",
    "attach_input_metadata",
    "discover_segmentation_pairs",
    "load_data_index_frame",
    "normalize_patient_id",
    "normalize_slice_type",
    "parse_selected_slices",
    "resolve_data_index_path",
    "selected_slices_from_index_frame",
]
