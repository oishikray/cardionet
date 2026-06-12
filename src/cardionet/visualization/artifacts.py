from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cardionet.visualization.canonical import save_metric_bullseye, save_ring_time_series_plots

SUMMARY_METRICS = (
    ("peak_nwt", "Peak NWT", "Peak NWT", "peak_nwt_bullseye.png"),
    (
        "peak_radial_strain",
        "Peak Radial Strain Proxy",
        "Peak radial strain proxy",
        "peak_radial_strain_proxy_bullseye.png",
    ),
    (
        "peak_endocardial_excursion",
        "Peak Endocardial Excursion",
        "Peak fractional inward endocardial excursion",
        "peak_endocardial_excursion_bullseye.png",
    ),
)
TIME_SERIES_METRICS = (
    ("nwt_time_series", "NWT", "NWT = WT(t) / mean ED epicardial radius", "nwt", 1.0),
    (
        "radial_strain_time_series",
        "Radial Strain Proxy",
        "Relative wall thickening from ED",
        "radial_strain_proxy",
        0.0,
    ),
    (
        "endocardial_excursion_time_series",
        "Endocardial Excursion",
        "Fractional inward endocardial excursion",
        "endocardial_excursion",
        0.0,
    ),
)


def _finite_float(value: str) -> float:
    text = str(value).strip()
    if not text:
        return np.nan
    numeric = float(text)
    return numeric if np.isfinite(numeric) else np.nan


def _json_series(value: str) -> list[float]:
    raw = json.loads(value or "[]")
    return [np.nan if item is None else float(item) for item in raw]


def load_segment_metric_rows(path: Path) -> list[dict[str, str]]:
    """Read a segment metrics CSV into dictionaries."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def group_rows_by_selected_slice(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, int, str], list[dict[str, str]]]:
    """Group rows by patient folder, patient id, slice index, and slice type."""
    grouped: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["patient_folder"],
            row["patient_id"],
            int(row["slice_index"]),
            row["slice_type"],
        )
        grouped[key].append(row)
    return dict(grouped)


def _ordered_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: int(row["aha_sector_index"]))


def _summary_by_ring(rows: list[dict[str, str]], column: str) -> dict[str, np.ndarray]:
    by_ring: dict[str, list[float]] = defaultdict(list)
    for row in _ordered_rows(rows):
        by_ring[row["slice_type"]].append(_finite_float(row[column]))
    return {ring: np.asarray(values, dtype=float) for ring, values in by_ring.items()}


def _series_by_ring(rows: list[dict[str, str]], column: str) -> dict[str, np.ndarray]:
    by_ring: dict[str, list[list[float]]] = defaultdict(list)
    for row in _ordered_rows(rows):
        by_ring[row["slice_type"]].append(_json_series(row[column]))
    return {ring: np.asarray(values, dtype=float) for ring, values in by_ring.items()}


def render_feature_visualisations(
    *,
    segment_metrics_path: Path,
    output_root: Path,
    classification_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Render plots from saved feature tables without recomputing mask-derived metrics."""
    rows = load_segment_metric_rows(segment_metrics_path)
    grouped = group_rows_by_selected_slice(rows)
    output_root.mkdir(parents=True, exist_ok=True)

    rendered: list[str] = []
    for (patient_folder, patient_id, slice_index, slice_type), slice_rows in grouped.items():
        output_dir = output_root / patient_folder / f"slice_{slice_index:02d}_{slice_type}"
        output_dir.mkdir(parents=True, exist_ok=True)
        title_patient_id = f"{patient_id} slice_{slice_index:02d}_{slice_type}"
        ed_frame = int(slice_rows[0]["ed_frame"])
        es_frame = int(slice_rows[0]["es_frame"])

        for column, metric_name, legend_label, filename in SUMMARY_METRICS:
            path = save_metric_bullseye(
                _summary_by_ring(slice_rows, column),
                output_dir,
                patient_id=title_patient_id,
                metric_name=metric_name,
                filename=filename,
                legend_label=legend_label,
            )
            rendered.append(str(path))

        for column, metric_name, ylabel, filename_prefix, baseline_value in TIME_SERIES_METRICS:
            paths = save_ring_time_series_plots(
                _series_by_ring(slice_rows, column),
                output_dir,
                patient_id=title_patient_id,
                metric_name=metric_name,
                ylabel=ylabel,
                filename_prefix=filename_prefix,
                ed_frame=ed_frame,
                es_frame=es_frame,
                baseline_value=baseline_value,
                peak_mode="max",
            )
            rendered.extend(str(path) for path in paths)

    manifest = {
        "segment_metrics_path": str(segment_metrics_path),
        "classification_summary_path": (
            str(classification_summary_path) if classification_summary_path else None
        ),
        "output_root": str(output_root),
        "selected_slice_count": len(grouped),
        "plot_count": len(rendered),
        "plots": rendered,
    }
    manifest_path = output_root / "visualisation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
