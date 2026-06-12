from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CLASS_ORDER = ("normal", "hypo", "aki_diski")
METRIC_COLUMNS = ("peak_nwt", "peak_endocardial_excursion")


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    metric: str
    low_threshold: float
    high_threshold: float
    macro_f1: float
    confusion_matrix: np.ndarray
    class_f1: dict[str, float]
    support: dict[str, int]


def collapse_diagnosis(value: str) -> str | None:
    """Map hand labels to the three-class threshold experiment."""
    text = value.strip().lower()
    if text == "normal":
        return "normal"
    if text == "hypokinesia":
        return "hypo"
    if text in {"akinesia", "dyskinesia"}:
        return "aki_diski"
    return None


def load_metric_rows(csv_path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Load finite metric values and collapsed labels from a labelled segment CSV."""
    values: list[float] = []
    labels: list[str] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"diagnosis", metric}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

        for row in reader:
            label = collapse_diagnosis(str(row.get("diagnosis", "")))
            if label is None:
                continue
            raw_value = str(row.get(metric, "")).strip()
            if raw_value == "":
                continue
            value = float(raw_value)
            if not np.isfinite(value):
                continue
            labels.append(label)
            values.append(value)

    if not values:
        raise ValueError(f"No finite values found for {metric!r} after class filtering.")
    return np.asarray(values, dtype=float), np.asarray(labels, dtype=object)


def predict_from_thresholds(
    values: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Classify high values as normal, middle values as hypo, and low values as aki/diski."""
    predictions = np.full(values.shape, "aki_diski", dtype=object)
    predictions[values >= low_threshold] = "hypo"
    predictions[values >= high_threshold] = "normal"
    return predictions


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a confusion count matrix with rows=true and columns=predicted."""
    index = {label: i for i, label in enumerate(CLASS_ORDER)}
    matrix = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[index[str(true_label)], index[str(pred_label)]] += 1
    return matrix


def f1_scores(matrix: np.ndarray) -> dict[str, float]:
    """Compute one-vs-rest F1 for each class from a confusion matrix."""
    scores: dict[str, float] = {}
    for i, label in enumerate(CLASS_ORDER):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - matrix[i, i])
        fn = float(matrix[i, :].sum() - matrix[i, i])
        denominator = (2.0 * tp) + fp + fn
        scores[label] = 0.0 if denominator == 0.0 else (2.0 * tp) / denominator
    return scores


def macro_f1(matrix: np.ndarray) -> float:
    scores = f1_scores(matrix)
    return float(np.mean([scores[label] for label in CLASS_ORDER]))


def threshold_grid(values: np.ndarray, n_steps: int) -> np.ndarray:
    """Build candidate thresholds spanning the observed finite metric range."""
    unique_values = np.unique(values[np.isfinite(values)])
    if unique_values.size < 2:
        raise ValueError("At least two unique metric values are required for a threshold sweep.")
    if unique_values.size <= n_steps:
        return unique_values
    return np.linspace(float(unique_values.min()), float(unique_values.max()), int(n_steps))


def sweep_thresholds(
    values: np.ndarray,
    labels: np.ndarray,
    metric: str,
    n_steps: int,
) -> ThresholdResult:
    """Find the two-threshold classifier with the highest macro-F1."""
    thresholds = threshold_grid(values, n_steps=n_steps)
    best: ThresholdResult | None = None

    for low_threshold in thresholds:
        for high_threshold in thresholds:
            if low_threshold >= high_threshold:
                continue
            predictions = predict_from_thresholds(
                values,
                low_threshold=float(low_threshold),
                high_threshold=float(high_threshold),
            )
            matrix = confusion_matrix(labels, predictions)
            score = macro_f1(matrix)
            if best is None or score > best.macro_f1:
                best = ThresholdResult(
                    metric=metric,
                    low_threshold=float(low_threshold),
                    high_threshold=float(high_threshold),
                    macro_f1=score,
                    confusion_matrix=matrix,
                    class_f1=f1_scores(matrix),
                    support={label: int(np.sum(labels == label)) for label in CLASS_ORDER},
                )

    if best is None:
        raise ValueError(f"Could not find valid low/high threshold pair for {metric}.")
    return best


def result_to_json(result: ThresholdResult, plot_path: Path | None = None) -> dict[str, Any]:
    """Convert a threshold result into JSON-safe output."""
    payload: dict[str, Any] = {
        "metric": result.metric,
        "low_threshold": result.low_threshold,
        "high_threshold": result.high_threshold,
        "prediction_rule": {
            "normal": f"{result.metric} >= high_threshold",
            "hypo": f"low_threshold <= {result.metric} < high_threshold",
            "aki_diski": f"{result.metric} < low_threshold",
        },
        "macro_f1": result.macro_f1,
        "class_f1": result.class_f1,
        "support": result.support,
        "class_order": list(CLASS_ORDER),
        "confusion_matrix": result.confusion_matrix.tolist(),
    }
    if plot_path is not None:
        payload["confusion_matrix_plot"] = str(plot_path)
    return payload


def run_threshold_search(
    *,
    input_csv: Path,
    output_dir: Path,
    metrics: tuple[str, ...] = METRIC_COLUMNS,
    n_steps: int = 200,
) -> list[dict[str, Any]]:
    """Run threshold searches and write a JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for metric in metrics:
        values, labels = load_metric_rows(input_csv, metric)
        result = sweep_thresholds(values, labels, metric, n_steps=n_steps)
        summaries.append(result_to_json(result))

    summary_path = output_dir / "threshold_sweep_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries
