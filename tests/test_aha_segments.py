import numpy as np
import pytest

from cardionet.features.aha_segments import (
    assign_sector_index,
    get_segment_names,
    sector_boundaries_from_anchor,
)


def test_get_segment_names_mid_has_six_segments():
    names = get_segment_names("mid")
    assert len(names) == 6


def test_get_segment_names_apical_has_four_segments():
    names = get_segment_names("apical")
    assert len(names) == 4


def test_get_segment_names_invalid_raises():
    with pytest.raises(ValueError):
        get_segment_names("nonsense")


def test_sector_boundaries_from_anchor_count():
    bounds = sector_boundaries_from_anchor(anchor_angle=0.0, n_sectors=6)
    assert len(bounds) == 7


def test_assign_sector_index_valid_range():
    bounds = sector_boundaries_from_anchor(anchor_angle=0.0, n_sectors=6)

    idx = assign_sector_index(theta=0.0, bounds=bounds)
    assert 0 <= idx < 6