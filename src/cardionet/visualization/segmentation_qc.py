from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path

try:
    import imageio
except ModuleNotFoundError:
    imageio = None
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


LABEL_COLORS = {
    1: np.array([1.0, 0.9, 0.0, 1.0]),
    2: np.array([0.0, 0.85, 0.25, 1.0]),
    3: np.array([1.0, 0.0, 0.0, 1.0]),
}


@dataclass(frozen=True, slots=True)
class LVSliceQualityResult:
    """LV slice inclusion result for EF volume calculations."""

    include_slices: np.ndarray
    drop_reasons: tuple[str, ...]
    ed_frame: int
    es_frame: int
    slice_peak_areas_mm2: np.ndarray
    slice_ed_areas_mm2: np.ndarray
    slice_es_areas_mm2: np.ndarray

    @property
    def kept_indices(self) -> list[int]:
        """Return zero-based slice indices kept for volume integration."""
        return np.where(self.include_slices)[0].astype(int).tolist()

    @property
    def dropped_indices(self) -> list[int]:
        """Return zero-based slice indices excluded from volume integration."""
        return np.where(~self.include_slices)[0].astype(int).tolist()


def _draw_label_contours(
    ax: plt.Axes,
    label_slice: np.ndarray,
    label_colors: dict[int, np.ndarray],
    *,
    line_width: float = 1.5,
) -> None:
    """Draw label boundaries as contours on an existing image axis."""
    for label_value, color in label_colors.items():
        mask = label_slice == label_value
        if not np.any(mask):
            continue
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[tuple(np.asarray(color, dtype=float))],
            linewidths=line_width,
        )


def plot_segmentations_per_slice(
    images: np.ndarray,
    labels: np.ndarray,
    t_step: int,
    output_dir: str | Path,
    basename: str,
    *,
    figure_size: tuple[float, float] = (4, 4),
    dpi: int = 150,
    cmap: str = "gray",
    gif_loop: int = 0,
    gif_duration_base_ms: int = 50,
    label_colors: dict[int, np.ndarray] | None = None,
    contour_line_width: float = 1.5,
    slice_indices: list[int] | tuple[int, ...] | None = None,
) -> list[Path]:
    """
    Create one GIF per slice, showing temporal segmentation contour evolution.

    Parameters
    ----------
    images
        Input cine array shaped (x, y, z, t).
    labels
        Predicted labels shaped (x, y, z, t).
    t_step
        Temporal subsampling step used upstream.
    output_dir
        Directory in which GIFs will be written.
    basename
        Base filename prefix.
    slice_indices
        Optional source slice indices to render. If omitted, all slices are
        rendered.

    Returns
    -------
    list[Path]
        Paths to written GIF files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if images.shape != labels.shape:
        raise ValueError(
            f"images and labels must have same shape, got {images.shape} vs {labels.shape}"
        )

    n_slices, n_frames = labels.shape[-2:]
    if slice_indices is None:
        resolved_slice_indices = list(range(n_slices))
    else:
        resolved_slice_indices = [int(slice_index) for slice_index in slice_indices]
        invalid_indices = [
            slice_index
            for slice_index in resolved_slice_indices
            if slice_index < 0 or slice_index >= n_slices
        ]
        if invalid_indices:
            raise ValueError(
                f"slice_indices out of bounds for {n_slices} slices: {invalid_indices}"
            )

    gif_paths: list[Path] = []
    active_label_colors = LABEL_COLORS if label_colors is None else label_colors
    contour_line_width = float(contour_line_width)

    for z in resolved_slice_indices:
        frames = []

        frame_iter = range(n_frames)
        if tqdm is not None:
            frame_iter = tqdm(frame_iter, desc=f"Slice {z} frames")

        for t in frame_iter:
            fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
            ax.imshow(images[..., z, t].T, cmap=cmap)
            _draw_label_contours(
                ax,
                labels[..., z, t].T,
                active_label_colors,
                line_width=contour_line_width,
            )

            ax.set_title(f"Slice {z} | Frame {t}")
            ax.set_xticks([])
            ax.set_yticks([])

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi)
            buf.seek(0)

            img = Image.open(buf)
            frame = np.array(img.convert("RGB"))
            frames.append(frame)

            buf.close()
            plt.close(fig)

        gif_path = output_dir / f"{basename}_slice_{z:02d}.gif"
        if imageio is not None:
            with imageio.get_writer(
                gif_path,
                mode="I",
                duration=gif_duration_base_ms * t_step,
                loop=gif_loop,
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
        elif frames:
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=gif_duration_base_ms * t_step,
                loop=gif_loop,
            )

        gif_paths.append(gif_path)

    return gif_paths


def _label_slice_areas_mm2(
    labels: np.ndarray,
    *,
    label_value: int,
    spacing_mm: tuple[float, float, float],
    include_slices: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return per-slice label areas in mm^2, z spacing, and z positions."""
    if labels.ndim != 4:
        raise ValueError(f"Expected labels shaped (x, y, z, t), got {labels.shape}")

    sx, sy, sz = (float(value) for value in spacing_mm)
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"Spacing values must be positive, got {spacing_mm}")

    slice_areas_mm2 = np.sum(labels == label_value, axis=(0, 1)).astype(float) * sx * sy
    z_positions_mm = np.arange(slice_areas_mm2.shape[0], dtype=float) * sz
    if include_slices is not None:
        include_slices = np.asarray(include_slices, dtype=bool)
        if include_slices.shape != (slice_areas_mm2.shape[0],):
            raise ValueError(
                "include_slices must have one boolean per z slice, got "
                f"{include_slices.shape} for {slice_areas_mm2.shape[0]} slices"
            )
        slice_areas_mm2 = slice_areas_mm2[include_slices, :]
        z_positions_mm = z_positions_mm[include_slices]

    return slice_areas_mm2, sz, z_positions_mm


def compute_label_volumes_riemann(
    labels: np.ndarray,
    *,
    label_value: int,
    spacing_mm: tuple[float, float, float],
    include_slices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute framewise label volumes in ml by direct voxel-volume summation.

    This mirrors the CineMA SAX inference example: count label voxels in each
    frame and multiply by voxel volume. Unlike the example's ACDC-specific
    ``* 10 / 1000`` shortcut, this uses the source NIfTI ``sx * sy * sz``.
    """
    slice_areas_mm2, z_spacing_mm, _ = _label_slice_areas_mm2(
        labels,
        label_value=label_value,
        spacing_mm=spacing_mm,
        include_slices=include_slices,
    )
    if slice_areas_mm2.shape[0] == 0:
        return np.full(slice_areas_mm2.shape[1], np.nan, dtype=float)
    return np.sum(slice_areas_mm2, axis=0) * z_spacing_mm / 1000.0


def compute_label_volumes_disk(
    labels: np.ndarray,
    *,
    label_value: int,
    spacing_mm: tuple[float, float, float],
    include_slices: np.ndarray | None = None,
) -> np.ndarray:
    """Deprecated alias for direct Riemann voxel-volume summation."""
    return compute_label_volumes_riemann(
        labels,
        label_value=label_value,
        spacing_mm=spacing_mm,
        include_slices=include_slices,
    )


def compute_ejection_fraction(volumes_ml: np.ndarray) -> float:
    """Compute EF percentage from a framewise ventricular volume curve."""
    finite = np.asarray(volumes_ml, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or np.nanmax(finite) <= 0:
        return float("nan")

    end_diastolic_volume = float(np.nanmax(finite))
    end_systolic_volume = float(np.nanmin(finite))
    return (end_diastolic_volume - end_systolic_volume) / end_diastolic_volume * 100.0


def compute_lv_slice_quality_mask(
    labels: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    label_value: int = 3,
    min_peak_area_fraction: float = 0.08,
    apical_min_peak_area_fraction: float = 0.20,
    max_es_ed_area_ratio: float = 1.05,
    apical_max_es_ed_area_ratio: float = 1.0,
) -> LVSliceQualityResult:
    """
    Select LV slices for EF calculation using conservative area-curve QC.

    Empty slices are dropped. The smaller terminal side is treated as apical
    and uses stricter thresholds because terminal apical LV masks are most
    likely to be clinically excluded when the cavity is poorly segmented.
    """
    slice_areas_mm2, _, _ = _label_slice_areas_mm2(
        labels,
        label_value=label_value,
        spacing_mm=spacing_mm,
    )
    num_slices, _ = slice_areas_mm2.shape
    include = np.ones(num_slices, dtype=bool)
    reasons = [""] * num_slices

    frame_total_area = np.sum(slice_areas_mm2, axis=0)
    populated_frames = np.where(frame_total_area > 0)[0]
    if populated_frames.size == 0:
        return LVSliceQualityResult(
            include_slices=np.zeros(num_slices, dtype=bool),
            drop_reasons=tuple("empty_lv_stack" for _ in range(num_slices)),
            ed_frame=0,
            es_frame=0,
            slice_peak_areas_mm2=np.zeros(num_slices, dtype=float),
            slice_ed_areas_mm2=np.zeros(num_slices, dtype=float),
            slice_es_areas_mm2=np.zeros(num_slices, dtype=float),
        )

    ed_frame = int(populated_frames[np.argmax(frame_total_area[populated_frames])])
    es_frame = int(populated_frames[np.argmin(frame_total_area[populated_frames])])
    peak_areas = np.nanmax(slice_areas_mm2, axis=1)
    ed_areas = slice_areas_mm2[:, ed_frame]
    es_areas = slice_areas_mm2[:, es_frame]
    max_peak_area = float(np.nanmax(peak_areas)) if peak_areas.size else 0.0

    if max_peak_area <= 0:
        include[:] = False
        reasons = ["empty_lv_slice"] * num_slices
    else:
        nonempty_indices = np.where(peak_areas > 0)[0]
        apical_side = None
        if nonempty_indices.size:
            first = int(nonempty_indices[0])
            last = int(nonempty_indices[-1])
            apical_side = "low" if peak_areas[first] <= peak_areas[last] else "high"

        for slice_index in range(num_slices):
            is_empty = peak_areas[slice_index] <= 0
            is_apical_terminal = (
                (apical_side == "low" and slice_index == nonempty_indices[0])
                or (apical_side == "high" and slice_index == nonempty_indices[-1])
            ) if nonempty_indices.size else False

            peak_fraction = peak_areas[slice_index] / max_peak_area
            area_ratio = np.inf if ed_areas[slice_index] <= 0 else (
                es_areas[slice_index] / ed_areas[slice_index]
            )
            min_fraction = (
                apical_min_peak_area_fraction
                if is_apical_terminal
                else min_peak_area_fraction
            )
            max_ratio = (
                apical_max_es_ed_area_ratio
                if is_apical_terminal
                else max_es_ed_area_ratio
            )

            drop_reason = ""
            if is_empty:
                drop_reason = "empty_lv_slice"
            elif peak_fraction < min_fraction:
                drop_reason = "low_peak_lv_area"
            elif area_ratio > max_ratio:
                drop_reason = "es_area_exceeds_ed_area"

            if drop_reason:
                include[slice_index] = False
                reasons[slice_index] = drop_reason

    return LVSliceQualityResult(
        include_slices=include,
        drop_reasons=tuple(reasons),
        ed_frame=ed_frame,
        es_frame=es_frame,
        slice_peak_areas_mm2=peak_areas,
        slice_ed_areas_mm2=ed_areas,
        slice_es_areas_mm2=es_areas,
    )


def plot_volume_changes(
    labels: np.ndarray,
    t_step: int,
    filepath: str | Path,
    *,
    spacing_mm: tuple[float, float, float],
    drop_poor_lv_slices: bool = False,
    lv_min_peak_area_fraction: float = 0.08,
    lv_apical_min_peak_area_fraction: float = 0.20,
    lv_max_es_ed_area_ratio: float = 1.05,
    lv_apical_max_es_ed_area_ratio: float = 1.0,
    figsize: tuple[float, float] = (5, 4),
    dpi_screen: int = 120,
    dpi_save: int = 300,
    ylabel: str = "Volume (ml)",
) -> Path:
    """
    Plot physical label-volume changes over time and report EF.

    Volumes are computed by direct Riemann-style summation: count label voxels
    in each frame and multiply by the source NIfTI voxel volume
    ``sx * sy * sz``.

    Label convention currently assumed:
    - 1 -> RV
    - 2 -> MYO
    - 3 -> LV
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_frames = labels.shape[-1]
    xs = np.arange(n_frames) * t_step

    lv_slice_qc = None
    include_lv_slices = None
    if drop_poor_lv_slices:
        lv_slice_qc = compute_lv_slice_quality_mask(
            labels,
            spacing_mm=spacing_mm,
            min_peak_area_fraction=lv_min_peak_area_fraction,
            apical_min_peak_area_fraction=lv_apical_min_peak_area_fraction,
            max_es_ed_area_ratio=lv_max_es_ed_area_ratio,
            apical_max_es_ed_area_ratio=lv_apical_max_es_ed_area_ratio,
        )
        include_lv_slices = lv_slice_qc.include_slices

    rv_volumes = compute_label_volumes_riemann(
        labels,
        label_value=1,
        spacing_mm=spacing_mm,
    )
    myo_volumes = compute_label_volumes_riemann(
        labels,
        label_value=2,
        spacing_mm=spacing_mm,
    )
    lv_volumes = compute_label_volumes_riemann(
        labels,
        label_value=3,
        spacing_mm=spacing_mm,
        include_slices=include_lv_slices,
    )

    lvef = compute_ejection_fraction(lv_volumes)
    rvef = compute_ejection_fraction(rv_volumes)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi_screen)
    ax.plot(xs, rv_volumes, label="RV")
    ax.plot(xs, myo_volumes, label="MYO")
    ax.plot(xs, lv_volumes, label="LV")
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    slice_note = ""
    if lv_slice_qc is not None:
        slice_note = (
            f" | LV slices kept {lv_slice_qc.kept_indices}; "
            f"dropped {lv_slice_qc.dropped_indices}"
        )
    ax.set_title(
        f"LVEF = {lvef:.2f}% | RVEF = {rvef:.2f}%\n"
        f"Spacing = {spacing_mm[0]:.3g} x {spacing_mm[1]:.3g} x {spacing_mm[2]:.3g} mm"
        f"{slice_note}"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi_save, bbox_inches="tight")
    plt.close(fig)

    return filepath
