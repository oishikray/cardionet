from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cardionet.config import load_cardionet_config

STAGES = ("segmentation", "features", "classification", "visualisation")
SCRIPT_BY_STAGE = {
    "segmentation": "run_segmentation.py",
    "features": "run_feature_extraction.py",
    "classification": "run_threshold_classifier.py",
    "visualisation": "run_visualisations.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular CardioNet pipeline stages.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--start-at", choices=STAGES, default=None)
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-visualisation", action="store_true")
    return parser.parse_args()


def selected_stages(start_at: str, stop_after: str) -> list[str]:
    start_index = STAGES.index(start_at)
    stop_index = STAGES.index(stop_after)
    if stop_index < start_index:
        raise ValueError("--stop-after must be the same stage or later than --start-at")
    return list(STAGES[start_index : stop_index + 1])


def main() -> None:
    args = parse_args()
    config = load_cardionet_config(args.config)
    script_cfg = config.scripts.run_full_pipeline
    start_at = str(args.start_at or script_cfg.start_at)
    stop_after = str(args.stop_after or script_cfg.stop_after)

    stages = selected_stages(start_at, stop_after)
    if args.skip_segmentation and "segmentation" in stages:
        stages.remove("segmentation")
    if args.skip_classification and "classification" in stages:
        stages.remove("classification")
    if args.skip_visualisation and "visualisation" in stages:
        stages.remove("visualisation")

    scripts_dir = Path(__file__).resolve().parent
    for stage in stages:
        script_path = scripts_dir / SCRIPT_BY_STAGE[stage]
        command = [sys.executable, str(script_path)]
        if args.config is not None:
            command.extend(["--config", str(args.config)])
        print("\n" + "=" * 72)
        print("Running stage:", stage)
        print("Command:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
