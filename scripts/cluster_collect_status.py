from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cardionet.config import load_cardionet_config
from cardionet.io.common import as_path
from cardionet.pipelines.status import collect_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cluster stage status files and optionally merge feature tables."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--status-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--merge-features",
        action="store_true",
        help="Merge per-case feature CSVs into cluster.artifacts.feature_root.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_feature_tables(feature_root: Path) -> dict[str, str | int | None]:
    """Merge per-case feature CSVs into dataset-level CSVs."""
    outputs: dict[str, str | int | None] = {}
    for filename in (
        "segment_metrics.csv",
        "segment_metrics_labelled.csv",
        "metric_time_series.csv",
    ):
        all_rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
        for path in sorted(feature_root.glob(f"*/{filename}")):
            current_fieldnames, rows = _read_rows(path)
            if rows and not fieldnames:
                fieldnames = current_fieldnames
            all_rows.extend(rows)
        output_path = feature_root / filename
        if fieldnames:
            _write_rows(output_path, fieldnames, all_rows)
            outputs[filename] = str(output_path)
            outputs[f"{filename}_rows"] = len(all_rows)
        else:
            outputs[filename] = None
            outputs[f"{filename}_rows"] = 0
    return outputs


def main() -> None:
    args = parse_args()
    config = load_cardionet_config(args.config)
    status_root = as_path(args.status_root or str(config.cluster.artifacts.status_root))
    summary = collect_status(status_root)

    if args.merge_features:
        feature_root = as_path(str(config.cluster.artifacts.feature_root))
        summary["merged_features"] = merge_feature_tables(feature_root)

    output_path = as_path(args.output) if args.output else status_root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Status summary:", output_path)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
