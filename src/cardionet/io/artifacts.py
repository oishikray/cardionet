from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cardionet.io.common import normalize_patient_id
from cardionet.segmentation.io import NiftiFrameTiming, read_nifti_frame_timing, read_nifti_geometry

PREDICTED_LABELS_SUFFIX = "_pred_labels.npy"
IMAGES_SUFFIX = "_images.npy"


@dataclass(frozen=True, slots=True)
class SegmentationPair:
    """Saved image/label arrays for one segmentation output."""

    patient_folder: str
    patient_id: str
    images_path: Path
    labels_path: Path
    input_nifti_path: Path | None = None
    spacing_mm: tuple[float, float, float] | None = None
    frame_timing: NiftiFrameTiming | None = None


def infer_patient_id(folder_name: str) -> str:
    """Infer patient id from a segmentation output folder name."""
    return normalize_patient_id(folder_name[:-10] if folder_name.endswith("_inference") else folder_name)


def discover_segmentation_pairs(segmentation_root: Path) -> list[SegmentationPair]:
    """Find saved image and predicted-label arrays below a segmentation output root."""
    pairs: list[SegmentationPair] = []
    for labels_path in sorted(segmentation_root.glob(f"*/**/*{PREDICTED_LABELS_SUFFIX}")):
        if not labels_path.is_file():
            continue
        images_path = labels_path.with_name(
            labels_path.name.replace(PREDICTED_LABELS_SUFFIX, IMAGES_SUFFIX)
        )
        if not images_path.exists():
            continue

        patient_folder = labels_path.parent.name
        pairs.append(
            SegmentationPair(
                patient_folder=patient_folder,
                patient_id=infer_patient_id(patient_folder),
                images_path=images_path,
                labels_path=labels_path,
            )
        )
    return pairs


def attach_input_metadata(
    pairs: list[SegmentationPair],
    *,
    input_root: Path,
) -> list[SegmentationPair]:
    """Attach source NIfTI path, spacing, and frame timing when files exist."""
    enriched: list[SegmentationPair] = []
    for pair in pairs:
        input_nifti_path = input_root / f"{pair.patient_id}.nii.gz"
        if not input_nifti_path.exists():
            enriched.append(pair)
            continue

        geometry = read_nifti_geometry(input_nifti_path)
        frame_timing = read_nifti_frame_timing(input_nifti_path)
        enriched.append(
            SegmentationPair(
                patient_folder=pair.patient_folder,
                patient_id=pair.patient_id,
                images_path=pair.images_path,
                labels_path=pair.labels_path,
                input_nifti_path=input_nifti_path,
                spacing_mm=geometry.spatial_spacing_mm,
                frame_timing=frame_timing,
            )
        )
    return enriched
