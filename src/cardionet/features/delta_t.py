from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class DeltaTRingMetrics:
    """Per-segment timing difference between peak NWT and end systole."""

    peak_nwt_frame: np.ndarray
    es_frame: int
    delta_t: np.ndarray
    abs_delta_t: np.ndarray
    delta_t_ms: np.ndarray
    frame_interval_ms: float


def safe_rowwise_nanargmax(values: np.ndarray) -> np.ndarray:
    """Return row-wise nanargmax frame indices, using -1 for all-NaN rows."""
    if values.ndim != 2:
        raise ValueError(f"Expected values shaped (segments, frames), got {values.shape}")

    frames = np.full(values.shape[0], -1, dtype=int)
    for index, row in enumerate(values):
        if np.any(np.isfinite(row)):
            frames[index] = int(np.nanargmax(row))
    return frames


def compute_delta_t(
    nwt_by_ring: Mapping[str, np.ndarray],
    *,
    es_frame: int,
    frame_interval_ms: float,
) -> dict[str, DeltaTRingMetrics]:
    """Compute deltaT = peak NWT frame - ES frame for each AHA ring."""
    if frame_interval_ms <= 0:
        raise ValueError(f"frame_interval_ms must be positive, got {frame_interval_ms}")

    results: dict[str, DeltaTRingMetrics] = {}
    for ring, nwt in nwt_by_ring.items():
        values = np.asarray(nwt, dtype=float)
        peak_frames = safe_rowwise_nanargmax(values)
        valid = peak_frames >= 0

        delta_t = np.full(peak_frames.shape, np.nan, dtype=float)
        delta_t[valid] = peak_frames[valid].astype(float) - float(es_frame)
        abs_delta_t = np.abs(delta_t)
        delta_t_ms = delta_t * float(frame_interval_ms)

        results[ring] = DeltaTRingMetrics(
            peak_nwt_frame=peak_frames,
            es_frame=int(es_frame),
            delta_t=delta_t,
            abs_delta_t=abs_delta_t,
            delta_t_ms=delta_t_ms,
            frame_interval_ms=float(frame_interval_ms),
        )

    return results


def delta_t_metrics_to_json(metrics: Mapping[str, DeltaTRingMetrics]) -> dict[str, dict[str, object]]:
    """Convert deltaT metrics to JSON-serializable dictionaries."""
    return {
        ring: {
            "peak_nwt_frame": values.peak_nwt_frame.tolist(),
            "es_frame": values.es_frame,
            "deltaT": values.delta_t.tolist(),
            "absdeltaT": values.abs_delta_t.tolist(),
            "deltaTms": values.delta_t_ms.tolist(),
            "frame_interval_ms": values.frame_interval_ms,
        }
        for ring, values in metrics.items()
    }
