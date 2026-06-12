from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig

from cardionet.pipelines.cluster_manifest import PipelineCase, read_manifest
from cardionet.pipelines.status import format_exception, utc_now, write_status

CASE_STAGES = {"segmentation", "features", "visualisation"}
ALL_STAGES = ("segmentation", "features", "classification", "visualisation")


def segmentation_complete(case: PipelineCase) -> bool:
    return case.images_path.exists() and case.labels_path.exists()


def features_complete(case: PipelineCase) -> bool:
    return (case.feature_dir / "feature_manifest.json").exists()


def visualisation_complete(case: PipelineCase) -> bool:
    return (case.visualisation_dir / "visualisation_manifest.json").exists()


def stage_complete(stage: str, case: PipelineCase) -> bool:
    if stage == "segmentation":
        return segmentation_complete(case)
    if stage == "features":
        return features_complete(case)
    if stage == "visualisation":
        return visualisation_complete(case)
    raise ValueError(f"Stage {stage!r} is not a per-case stage.")


def case_index_from_env(config: DictConfig) -> int:
    """Resolve scheduler array index from the configured environment variable."""
    env_var = str(config.cluster.slurm.array_env_var)
    raw_value = os.environ.get(env_var)
    if raw_value is None:
        raise ValueError(f"Array index not provided and {env_var} is unset.")
    return int(raw_value)


def select_case(manifest_path: Path, *, case_index: int) -> PipelineCase:
    cases = read_manifest(manifest_path)
    for case in cases:
        if case.case_index == case_index:
            return case
    raise IndexError(f"Case index {case_index} not found in manifest {manifest_path}")


def command_for_case_stage(
    *,
    stage: str,
    case: PipelineCase,
    config_path: Path | None,
    config: DictConfig,
) -> list[str]:
    """Build the subprocess command for one case stage."""
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    command = [sys.executable]
    if stage == "segmentation":
        command.append(str(scripts_dir / "run_segmentation.py"))
        if config_path is not None:
            command.extend(["--config", str(config_path)])
        command.extend([
            "--patient-id",
            case.patient_id,
            "--output-root",
            str(config.cluster.artifacts.segmentation_root),
        ])
        return command

    if stage == "features":
        command.append(str(scripts_dir / "run_feature_extraction.py"))
        if config_path is not None:
            command.extend(["--config", str(config_path)])
        command.extend([
            "--segmentation-root",
            str(config.cluster.artifacts.segmentation_root),
            "--output-dir",
            str(case.feature_dir),
            "--selected-slices-json",
            case.selected_slices_json,
        ])
        return command

    if stage == "visualisation":
        command.append(str(scripts_dir / "run_visualisations.py"))
        if config_path is not None:
            command.extend(["--config", str(config_path)])
        command.extend([
            "--segment-metrics",
            str(case.feature_dir / "segment_metrics.csv"),
            "--classification-summary",
            str(Path(str(config.cluster.artifacts.classification_root)) / "threshold_sweep_summary.json"),
            "--output-root",
            str(case.visualisation_dir),
        ])
        return command

    raise ValueError(f"Unsupported case stage: {stage}")


def command_for_dataset_stage(
    *,
    stage: str,
    config_path: Path | None,
    config: DictConfig,
) -> list[str]:
    """Build the subprocess command for a dataset-level stage."""
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if stage != "classification":
        raise ValueError(f"Unsupported dataset stage: {stage}")
    command = [sys.executable, str(scripts_dir / "run_threshold_classifier.py")]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    command.extend([
        "--input-csv",
        str(Path(str(config.cluster.artifacts.feature_root)) / "segment_metrics_labelled.csv"),
        "--output-dir",
        str(config.cluster.artifacts.classification_root),
    ])
    return command


def run_case_stage(
    *,
    stage: str,
    case: PipelineCase,
    config_path: Path | None,
    config: DictConfig,
    force: bool = False,
) -> str:
    """Run or skip one case stage and return final status."""
    status_root = Path(str(config.cluster.artifacts.status_root))
    started_at = None
    try:
        if not force and stage_complete(stage, case):
            write_status(
                status_root,
                stage=stage,
                case=case,
                status="skipped",
                message="Expected outputs already exist.",
            )
            return "skipped"
        command = command_for_case_stage(
            stage=stage,
            case=case,
            config_path=config_path,
            config=config,
        )
        started_at = utc_now()
        write_status(
            status_root,
            stage=stage,
            case=case,
            status="running",
            message=" ".join(command),
            started_at=started_at,
        )
        subprocess.run(command, check=True)
        write_status(
            status_root,
            stage=stage,
            case=case,
            status="done",
            message="Completed.",
            started_at=started_at,
        )
        return "done"
    except Exception as exc:
        write_status(
            status_root,
            stage=stage,
            case=case,
            status="failed",
            message=format_exception(exc),
            started_at=str(started_at) if started_at else None,
        )
        raise


def run_dataset_stage(
    *,
    stage: str,
    config_path: Path | None,
    config: DictConfig,
    force: bool = False,
) -> str:
    """Run one dataset-level stage."""
    status_root = Path(str(config.cluster.artifacts.status_root))
    output_path = Path(str(config.cluster.artifacts.classification_root)) / "threshold_sweep_summary.json"
    started_at = None
    try:
        if not force and output_path.exists():
            write_status(
                status_root,
                stage=stage,
                case=None,
                status="skipped",
                message="Expected outputs already exist.",
            )
            return "skipped"
        command = command_for_dataset_stage(stage=stage, config_path=config_path, config=config)
        started_at = utc_now()
        write_status(
            status_root,
            stage=stage,
            case=None,
            status="running",
            message=" ".join(command),
            started_at=started_at,
        )
        subprocess.run(command, check=True)
        write_status(
            status_root,
            stage=stage,
            case=None,
            status="done",
            message="Completed.",
            started_at=started_at,
        )
        return "done"
    except Exception as exc:
        write_status(
            status_root,
            stage=stage,
            case=None,
            status="failed",
            message=format_exception(exc),
            started_at=str(started_at) if started_at else None,
        )
        raise
