from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

from cardionet.config import (
    resolve_dataset_root,
    resolve_output_basename,
    resolve_patient_ids,
    resolve_script_output_dir,
)
from cardionet.io.common import normalize_patient_id
from cardionet.io.selection import SelectedSlicesByPatient

MANIFEST_FIELDNAMES = [
    "case_index",
    "patient_id",
    "input_image_path",
    "segmentation_dir",
    "images_path",
    "labels_path",
    "feature_dir",
    "visualisation_dir",
    "selected_slices_json",
]


@dataclass(frozen=True, slots=True)
class PipelineCase:
    """One independently runnable cluster pipeline case."""

    case_index: int
    patient_id: str
    input_image_path: Path
    segmentation_dir: Path
    images_path: Path
    labels_path: Path
    feature_dir: Path
    visualisation_dir: Path
    selected_slices_json: str


def _get_cine_filename(config: DictConfig, *, dataset_name: str, patient_id: str) -> str:
    dataset_cfg = config.datasets[dataset_name]
    return str(dataset_cfg.file_patterns.cine).format(patient_id=patient_id)


def _selected_slices_payload(
    patient_id: str,
    selected_slices: SelectedSlicesByPatient | None,
) -> dict[str, object]:
    if not selected_slices or patient_id not in selected_slices:
        return {}
    patient_slices = selected_slices[patient_id]
    if len(patient_slices) == 1:
        selected = patient_slices[0]
        return {
            patient_id: {
                "slice_index": selected.slice_index,
                "slice_type": selected.slice_type,
            }
        }
    return {
        patient_id: [
            {"slice_index": selected.slice_index, "slice_type": selected.slice_type}
            for selected in patient_slices
        ]
    }


def _normalise_selected_payload(payload: dict[str, object]) -> str:
    if not payload:
        return "{}"
    return json.dumps(payload)


def build_pipeline_cases(
    config: DictConfig,
    *,
    selected_slices: SelectedSlicesByPatient | None = None,
) -> list[PipelineCase]:
    """Build pipeline cases from config patient selection and optional slice metadata."""
    infer_cfg = config.scripts.infer_acdc_cine
    dataset_name = str(infer_cfg.dataset_name)
    split = str(infer_cfg.split)
    view = str(config.segmentation.model.view)
    dataset_root = resolve_dataset_root(config, dataset_name=dataset_name, split=split)
    patient_ids = resolve_patient_ids(
        config,
        dataset_root=dataset_root,
        script_name="infer_acdc_cine",
        dataset_name=dataset_name,
    )
    if selected_slices:
        patient_ids = [patient_id for patient_id in patient_ids if patient_id in selected_slices]

    cluster_cfg = config.cluster.artifacts
    segmentation_root = Path(str(cluster_cfg.segmentation_root))
    feature_root = Path(str(cluster_cfg.feature_root))
    visualisation_root = Path(str(cluster_cfg.visualisation_root))

    cases: list[PipelineCase] = []
    for case_index, patient_id in enumerate(patient_ids):
        patient_id = normalize_patient_id(patient_id)
        input_image_path = dataset_root / patient_id / _get_cine_filename(
            config,
            dataset_name=dataset_name,
            patient_id=patient_id,
        )
        segmentation_dir = resolve_script_output_dir(
            config,
            script_name="infer_acdc_cine",
            patient_id=patient_id,
            view=view,
            output_root=segmentation_root,
        )
        basename = resolve_output_basename(config, patient_id=patient_id, view=view)
        images_path = segmentation_dir / f"{basename}{config.conventions.file_naming.saved_images_suffix}"
        labels_path = segmentation_dir / f"{basename}{config.conventions.file_naming.predicted_labels_suffix}"
        selected_payload = _selected_slices_payload(patient_id, selected_slices)
        cases.append(
            PipelineCase(
                case_index=case_index,
                patient_id=patient_id,
                input_image_path=input_image_path,
                segmentation_dir=segmentation_dir,
                images_path=images_path,
                labels_path=labels_path,
                feature_dir=feature_root / patient_id,
                visualisation_dir=visualisation_root / patient_id,
                selected_slices_json=_normalise_selected_payload(selected_payload),
            )
        )
    return cases


def write_manifest(path: Path, cases: Iterable[PipelineCase]) -> None:
    """Write pipeline cases to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for case in cases:
            row = asdict(case)
            for key in (
                "input_image_path",
                "segmentation_dir",
                "images_path",
                "labels_path",
                "feature_dir",
                "visualisation_dir",
            ):
                row[key] = str(row[key])
            writer.writerow(row)


def read_manifest(path: Path) -> list[PipelineCase]:
    """Read pipeline cases from a manifest CSV."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        PipelineCase(
            case_index=int(row["case_index"]),
            patient_id=str(row["patient_id"]),
            input_image_path=Path(row["input_image_path"]),
            segmentation_dir=Path(row["segmentation_dir"]),
            images_path=Path(row["images_path"]),
            labels_path=Path(row["labels_path"]),
            feature_dir=Path(row["feature_dir"]),
            visualisation_dir=Path(row["visualisation_dir"]),
            selected_slices_json=str(row.get("selected_slices_json", "{}")),
        )
        for row in rows
    ]


def manifest_path_from_config(config: DictConfig) -> Path:
    """Return the configured manifest path."""
    return Path(str(config.cluster.artifacts.manifest_path))


def selected_slices_from_config(config: DictConfig) -> SelectedSlicesByPatient | None:
    """Load default cluster selected slices from the feature-runner config."""
    from cardionet.io.selection import parse_selected_slices

    script_cfg = config.scripts.run_feature_extraction
    payload = OmegaConf.to_container(script_cfg.get("selected_slices", {}), resolve=True)
    if not payload:
        return None
    return parse_selected_slices(json.dumps(payload))
