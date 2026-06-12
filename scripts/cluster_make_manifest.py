from __future__ import annotations

import argparse
from pathlib import Path

from cardionet.config import load_cardionet_config
from cardionet.io.common import as_path
from cardionet.io.selection import (
    load_data_index_frame,
    parse_selected_slices,
    resolve_data_index_path,
    selected_slices_from_index_frame,
)
from cardionet.pipelines.cluster_manifest import (
    build_pipeline_cases,
    manifest_path_from_config,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cluster pipeline case manifest.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-index-path", default=None)
    parser.add_argument("--selected-slices-json", default=None)
    parser.add_argument(
        "--use-config-selected-slices",
        action="store_true",
        help="Use scripts.run_feature_extraction.selected_slices from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cardionet_config(args.config)
    manifest_path = as_path(args.output) if args.output else manifest_path_from_config(config)

    selected_slices = None
    if args.data_index_path:
        input_root = as_path(str(config.scripts.run_feature_extraction.input_root))
        data_index_path = resolve_data_index_path(input_root, args.data_index_path)
        if data_index_path is None:
            raise ValueError("--data-index-path did not resolve to a path.")
        selected_slices = selected_slices_from_index_frame(load_data_index_frame(data_index_path))
    elif args.selected_slices_json:
        selected_slices = parse_selected_slices(args.selected_slices_json)
    elif args.use_config_selected_slices:
        from cardionet.pipelines.cluster_manifest import selected_slices_from_config

        selected_slices = selected_slices_from_config(config)

    cases = build_pipeline_cases(config, selected_slices=selected_slices)
    write_manifest(manifest_path, cases)
    print("Manifest:", manifest_path)
    print("Cases:", len(cases))
    if selected_slices:
        print("Selected-slice patients:", len(selected_slices))


if __name__ == "__main__":
    main()
