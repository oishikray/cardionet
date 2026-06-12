from __future__ import annotations

import shlex
from pathlib import Path

from omegaconf import DictConfig

CASE_STAGES = {"segmentation", "features", "visualisation"}


def build_sbatch_command(
    config: DictConfig,
    *,
    stage: str,
    manifest_path: Path,
    config_path: Path | None,
    array_size: int | None,
    force: bool = False,
) -> list[str]:
    """Build an sbatch command for a configured stage."""
    resources = config.cluster.resources[stage]
    command = [
        str(config.cluster.slurm.sbatch),
        f"--job-name=cardionet-{stage}",
        f"--cpus-per-task={int(resources.cpus_per_task)}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={Path(str(config.cluster.artifacts.log_root)) / (stage + '_%A_%a.out')}",
        f"--error={Path(str(config.cluster.artifacts.log_root)) / (stage + '_%A_%a.err')}",
    ]
    partition = str(resources.get("partition", "")).strip()
    if partition:
        command.append(f"--partition={partition}")
    gpus_per_task = int(resources.get("gpus_per_task", 0))
    if gpus_per_task > 0:
        command.append(f"--gres=gpu:{gpus_per_task}")
    if stage in CASE_STAGES:
        if array_size is None or array_size <= 0:
            raise ValueError(f"Stage {stage!r} requires a positive array size.")
        command.append(f"--array=0-{array_size - 1}")

    export_values = [
        "ALL",
        f"CARDIONET_STAGE={stage}",
        f"CARDIONET_MANIFEST={manifest_path}",
        f"CARDIONET_ACTIVATE_COMMAND={config.cluster.environment.activate_command}",
    ]
    if config_path is not None:
        export_values.append(f"CARDIONET_CONFIG={config_path}")
    if force:
        export_values.append("CARDIONET_FORCE=1")
    command.append("--export=" + ",".join(export_values))
    command.append(str(config.cluster.slurm.template_path))
    return command


def shell_join(command: list[str]) -> str:
    """Return a shell-safe command string for dry-run output."""
    return " ".join(shlex.quote(part) for part in command)
