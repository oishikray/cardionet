from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardionet.config import load_cardionet_config
from cardionet.features.threshold_classifier import run_threshold_search
from cardionet.io.common import as_path

SCRIPT_NAME = "run_threshold_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search threshold classifiers from saved labelled feature tables."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cardionet_config(args.config)
    script_cfg = config.scripts[SCRIPT_NAME]
    input_csv = as_path(args.input_csv or str(script_cfg.input_csv))
    output_dir = as_path(args.output_dir or str(script_cfg.output_dir))
    n_steps = int(args.n_steps or script_cfg.n_steps)

    summaries = run_threshold_search(
        input_csv=input_csv,
        output_dir=output_dir,
        n_steps=n_steps,
    )
    print("Classification summary:", output_dir / "threshold_sweep_summary.json")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
