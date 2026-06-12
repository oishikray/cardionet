from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from cardionet.config import load_cardionet_config, resolve_feature_raycast_settings
from cardionet.features.artifacts import run_feature_table_export
from cardionet.io.artifacts import attach_input_metadata, discover_segmentation_pairs
from cardionet.io.common import as_path
from cardionet.io.selection import (
    load_data_index_frame,
    parse_selected_slices,
    resolve_data_index_path,
    selected_slices_from_index_frame,
)

SCRIPT_NAME = "run_feature_extraction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract durable segment feature tables from saved segmentation arrays."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--segmentation-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ground-truth-labels", default=None)
    parser.add_argument("--data-index-path", default=None)
    parser.add_argument("--selected-slices-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cardionet_config(args.config)
    script_cfg = config.scripts[SCRIPT_NAME]
    raycast_cfg = resolve_feature_raycast_settings(config)

    input_root = as_path(args.input_root or str(script_cfg.input_root))
    segmentation_root = as_path(args.segmentation_root or str(script_cfg.segmentation_root))
    output_dir = as_path(args.output_dir or str(script_cfg.output_dir))
    ground_truth_labels = as_path(
        args.ground_truth_labels or str(script_cfg.ground_truth_labels_path)
    )

    data_index_path = resolve_data_index_path(input_root, args.data_index_path)
    if data_index_path is not None:
        selected_slices = selected_slices_from_index_frame(load_data_index_frame(data_index_path))
    else:
        selected_json = args.selected_slices_json or json.dumps(
            OmegaConf.to_container(script_cfg.selected_slices, resolve=True)
        )
        selected_slices = parse_selected_slices(selected_json)

    pairs = attach_input_metadata(
        [
            pair
            for pair in discover_segmentation_pairs(segmentation_root)
            if pair.patient_id in selected_slices
        ],
        input_root=input_root,
    )
    pairs_by_patient = {pair.patient_id: pair for pair in pairs}
    missing = sorted(set(selected_slices) - set(pairs_by_patient))
    if missing:
        raise FileNotFoundError("No saved segmentation arrays for: " + ", ".join(missing))

    manifest = run_feature_table_export(
        pairs_by_patient=pairs_by_patient,
        selected_slices=selected_slices,
        output_dir=output_dir,
        ground_truth_labels_path=ground_truth_labels,
        n_rays=int(raycast_cfg.n_rays),
        ray_step=float(raycast_cfg.ray_step),
        max_radius=float(raycast_cfg.max_radius),
    )
    print("Feature manifest:", manifest["manifest_path"])
    print("Segment metrics:", manifest["segment_metrics_path"])
    print("Time-series metrics:", manifest["metric_time_series_path"])


if __name__ == "__main__":
    main()
