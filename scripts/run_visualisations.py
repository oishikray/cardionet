from __future__ import annotations

import argparse
from pathlib import Path

from cardionet.config import load_cardionet_config
from cardionet.io.common import as_path
from cardionet.visualization.artifacts import render_feature_visualisations

SCRIPT_NAME = "run_visualisations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render visualisations from saved feature/classification artifacts."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--segment-metrics", default=None)
    parser.add_argument("--classification-summary", default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cardionet_config(args.config)
    script_cfg = config.scripts[SCRIPT_NAME]
    segment_metrics = as_path(args.segment_metrics or str(script_cfg.segment_metrics_path))
    classification_summary = (
        as_path(args.classification_summary)
        if args.classification_summary
        else as_path(str(script_cfg.classification_summary_path))
    )
    output_root = as_path(args.output_root or str(script_cfg.output_root))

    manifest = render_feature_visualisations(
        segment_metrics_path=segment_metrics,
        output_root=output_root,
        classification_summary_path=classification_summary if classification_summary.exists() else None,
    )
    print("Visualisation manifest:", manifest["manifest_path"])
    print("Plots:", manifest["plot_count"])


if __name__ == "__main__":
    main()
