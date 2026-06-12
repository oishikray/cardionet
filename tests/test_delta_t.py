import numpy as np

from cardionet.features.delta_t import compute_delta_t, safe_rowwise_nanargmax


def test_safe_rowwise_nanargmax_tolerates_all_nan_rows():
    values = np.array(
        [
            [1.0, 2.0, 1.5],
            [np.nan, np.nan, np.nan],
        ]
    )

    np.testing.assert_array_equal(safe_rowwise_nanargmax(values), np.array([1, -1]))


def test_compute_delta_t_uses_peak_nwt_frame_minus_es_frame():
    nwt_by_ring = {
        "basal": np.array(
            [
                [0.9, 1.2, 1.5, 1.3],
                [1.0, 1.1, 1.0, 1.4],
            ]
        )
    }

    result = compute_delta_t(nwt_by_ring, es_frame=2, frame_interval_ms=30.0)["basal"]

    np.testing.assert_array_equal(result.peak_nwt_frame, np.array([2, 3]))
    np.testing.assert_allclose(result.delta_t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(result.abs_delta_t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(result.delta_t_ms, np.array([0.0, 30.0]))
