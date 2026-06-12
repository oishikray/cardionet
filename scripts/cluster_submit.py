from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from cardionet.config import load_cardionet_config
from cardionet.pipelines.cluster_manifest import manifest_path_from_config, read_manifest
from cardionet.pipelines.slurm import build_sbatch_command, shell_join
from cardionet.pipelines.stage_runner import ALL_STAGES, CASE_STAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a configured cluster stage.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stage", choices=ALL_STAGES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cardionet_config(args.config)
    if str(config.cluster.scheduler).lower() != "slurm":
        raise ValueError(f"Unsupported scheduler: {config.cluster.scheduler!r}")

    manifest_path = args.manifest or manifest_path_from_config(config)
    array_size = None
    if args.stage in CASE_STAGES:
        cases = read_manifest(manifest_path)
        array_size = len(cases)
        if array_size == 0:
            raise ValueError(f"Manifest has no cases: {manifest_path}")

    command = build_sbatch_command(
        config,
        stage=args.stage,
        manifest_path=manifest_path,
        config_path=args.config,
        array_size=array_size,
        force=bool(args.force),
    )
    print(shell_join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
