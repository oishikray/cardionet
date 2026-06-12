from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813


@dataclass(frozen=True, slots=True)
class NiftiGeometry:
    """Physical geometry read from a NIfTI image header."""

    size: tuple[int, ...]
    spacing_mm: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]

    @property
    def spatial_spacing_mm(self) -> tuple[float, float, float]:
        """Return x/y/z spacing in millimeters."""
        if len(self.spacing_mm) < 3:
            raise ValueError(f"Expected at least 3 spacing values, got {self.spacing_mm}")
        return (
            float(self.spacing_mm[0]),
            float(self.spacing_mm[1]),
            float(self.spacing_mm[2]),
        )


@dataclass(frozen=True, slots=True)
class NiftiFrameTiming:
    """Temporal frame spacing read from a cine NIfTI header."""

    frame_interval_ms: float
    header_zooms: tuple[float, ...]
    header_spatial_unit: str
    header_time_unit: str
    interpreted_time_unit: str
    note: str | None = None


def read_nifti_geometry(path: str | Path) -> NiftiGeometry:
    """Read NIfTI size, spacing, origin, and direction without loading pixels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input NIfTI file not found: {path}")

    image = sitk.ReadImage(str(path))
    return NiftiGeometry(
        size=tuple(int(value) for value in image.GetSize()),
        spacing_mm=tuple(float(value) for value in image.GetSpacing()),
        origin=tuple(float(value) for value in image.GetOrigin()),
        direction=tuple(float(value) for value in image.GetDirection()),
    )


def _temporal_zoom_to_ms(temporal_zoom: float, time_unit: str) -> tuple[float, str, str | None]:
    """Convert NIfTI temporal zoom to milliseconds."""
    unit = str(time_unit).strip().lower()
    if temporal_zoom <= 0:
        raise ValueError(f"NIfTI temporal zoom must be positive, got {temporal_zoom}")

    if unit in {"msec", "ms", "millisecond", "milliseconds"}:
        return float(temporal_zoom), "msec", None
    if unit in {"usec", "us", "microsecond", "microseconds"}:
        return float(temporal_zoom) / 1000.0, "usec", None
    if unit in {"sec", "s", "second", "seconds"}:
        if temporal_zoom > 10.0:
            return (
                float(temporal_zoom),
                "msec",
                (
                    "Header time unit is sec, but temporal zoom is >10; "
                    "interpreting as milliseconds for cine MRI."
                ),
            )
        return float(temporal_zoom) * 1000.0, "sec", None

    raise ValueError(
        f"Unsupported or missing NIfTI time unit {time_unit!r}; cannot compute deltaTms."
    )


def read_nifti_frame_timing(path: str | Path) -> NiftiFrameTiming:
    """Read cine frame interval from NIfTI header and return milliseconds."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input NIfTI file not found: {path}")

    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nibabel is required to read NIfTI temporal units for deltaTms."
        ) from exc

    image = nib.load(str(path))
    zooms = tuple(float(value) for value in image.header.get_zooms())
    if len(zooms) < 4:
        raise ValueError(f"Expected 4D NIfTI zooms for frame timing, got {zooms}")

    spatial_unit, time_unit = image.header.get_xyzt_units()
    frame_interval_ms, interpreted_unit, note = _temporal_zoom_to_ms(zooms[3], time_unit)
    return NiftiFrameTiming(
        frame_interval_ms=frame_interval_ms,
        header_zooms=zooms,
        header_spatial_unit=str(spatial_unit),
        header_time_unit=str(time_unit),
        interpreted_time_unit=interpreted_unit,
        note=note,
    )


def load_cine_nifti(path: str | Path) -> np.ndarray:
    """
    Load a 4D cine NIfTI and return an array shaped (x, y, z, t).

    Notes
    -----
    This currently preserves the old CineMA/ACDC script behavior:
    SimpleITK output is transposed blindly to recover (x, y, z, t).

    This is acceptable only as long as:
    - the input data follows the same preprocessing conventions
    - validation confirms the resulting shape semantics

    Parameters
    ----------
    path
        Path to the cine NIfTI file.

    Returns
    -------
    np.ndarray
        Cine volume shaped (x, y, z, t).

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the loaded array is not 4D after transpose.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input cine file not found: {path}")

    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    arr = np.transpose(arr)

    if arr.ndim == 3:
        arr = arr[..., None]

    if arr.ndim != 4:
        raise ValueError(
            f"Expected 4D cine stack after transpose, got shape {arr.shape}"
        )

    return arr


def save_inference_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    basename: str,
    *,
    save_inputs: bool = True,
    save_predictions: bool = True,
    image_suffix: str = "_images.npy",
    labels_suffix: str = "_pred_labels.npy",
) -> tuple[Path | None, Path | None]:
    """
    Save inference inputs and predicted labels as .npy files.

    Returns
    -------
    tuple[Path | None, Path | None]
        Paths to saved images and labels. Entries are None when saving is disabled.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_path = output_dir / f"{basename}{image_suffix}" if save_inputs else None
    labels_path = (
        output_dir / f"{basename}{labels_suffix}" if save_predictions else None
    )

    if images_path is not None:
        np.save(images_path, images.astype(np.float32))

    if labels_path is not None:
        np.save(labels_path, labels.astype(np.uint8))

    return images_path, labels_path
