import numpy as np
import pytest

from cardionet.features.aha_segments import (
    AHA_SEGMENTS,
    assign_sector_index,
    get_segments,
    get_segment_names,
    get_segment_numbers,
    infer_slice_types,
    sector_boundaries_from_anchor,
)


def test_get_segment_names_mid_has_canonical_aha_order():
    names = get_segment_names("mid")
    assert names == [
        "Mid Anterior",
        "Mid Anteroseptal",
        "Mid Inferoseptal",
        "Mid Inferior",
        "Mid Inferolateral",
        "Mid Anterolateral",
    ]


def test_get_segment_names_apical_has_canonical_aha_order():
    names = get_segment_names("apical")
    assert names == [
        "Apical Anterior",
        "Apical Septal",
        "Apical Inferior",
        "Apical Lateral",
    ]


def test_get_segment_numbers_are_global_aha_numbers():
    assert get_segment_numbers("basal") == [1, 2, 3, 4, 5, 6]
    assert get_segment_numbers("mid") == [7, 8, 9, 10, 11, 12]
    assert get_segment_numbers("apical") == [13, 14, 15, 16]


def test_get_segments_exposes_names_numbers_and_order():
    segments = get_segments("basal")
    assert [(segment.number, segment.name) for segment in segments] == [
        (1, "Basal Anterior"),
        (2, "Basal Anteroseptal"),
        (3, "Basal Inferoseptal"),
        (4, "Basal Inferior"),
        (5, "Basal Inferolateral"),
        (6, "Basal Anterolateral"),
    ]
    assert [segment.number for segment in AHA_SEGMENTS[:3]] == [1, 2, 3]


def test_get_segment_names_invalid_raises():
    with pytest.raises(ValueError):
        get_segment_names("nonsense")


def test_infer_slice_types_is_disabled():
    labels = np.zeros((8, 8, 1, 1), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Automatic AHA slice typing"):
        infer_slice_types(labels)


def test_sector_boundaries_from_anchor_count():
    bounds = sector_boundaries_from_anchor(anchor_angle=0.0, n_sectors=6)
    assert len(bounds) == 7


def test_assign_sector_index_uses_anterior_first_order():
    bounds = sector_boundaries_from_anchor(anchor_angle=0.0, n_sectors=6)
    width = 2 * np.pi / 6

    assert assign_sector_index(theta=0.01, bounds=bounds) == 0
    assert assign_sector_index(theta=-width / 2, bounds=bounds) == 1
    assert assign_sector_index(theta=-width - 0.01, bounds=bounds) == 2
    assert assign_sector_index(theta=np.pi, bounds=bounds) == 4
