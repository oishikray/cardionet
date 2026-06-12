from __future__ import annotations

import argparse
import os
from pathlib import Path

from cardionet.config import load_cardionet_config
from cardionet.pipelines.cluster_manifest import manifest_path_from_config
from cardionet.pipelines.stage_runner import (
    CASE_STAGES,
    ALL_STAGES,
    case_index_from_env,
    run_case_stage,
    run_dataset_stage,
    select_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one cluster pipeline stage.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stage", choices=ALL_STAGES, default=None)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = os.environ.get("CARDIONET_CONFIG")
    config_path = args.config or (Path(env_config) if env_config else None)
    config = load_cardionet_config(config_path)
    stage = args.stage or os.environ.get("CARDIONET_STAGE")
    if stage not in ALL_STAGES:
        raise ValueError(f"Unsupported or missing stage: {stage!r}")
    force = bool(args.force or os.environ.get("CARDIONET_FORCE") == "1")

    if stage in CASE_STAGES:
        manifest = args.manifest or Path(
            os.environ.get("CARDIONET_MANIFEST") or manifest_path_from_config(config)
        )
        case_index = args.case_index
        if case_index is None:
            case_index = case_index_from_env(config)
        case = select_case(manifest, case_index=case_index)
        status = run_case_stage(
            stage=stage,
            case=case,
            config_path=config_path,
            config=config,
            force=force,
        )
        print(f"{stage} {case.patient_id}: {status}")
        return

    status = run_dataset_stage(
        stage=stage,
        config_path=config_path,
        config=config,
        force=force,
    )
    print(f"{stage}: {status}")


if __name__ == "__main__":
    main()
