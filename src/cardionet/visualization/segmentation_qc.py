from __future__ import annotations

import io
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


LABEL_COLORS = {
    1: np.array([108 / 255, 142 / 255, 191 / 255, 0.6]),
    2: np.array([214 / 255, 182 / 255, 86 / 255, 0.6]),
    3: np.array([130 / 255, 179 / 255, 102 / 255, 0.6]),
}


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
) -> list[Path]:
    """
    Create one GIF per slice, showing temporal segmentation evolution.

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
    gif_paths: list[Path] = []
    active_label_colors = LABEL_COLORS if label_colors is None else label_colors

    for z in range(n_slices):
        frames = []

        for t in tqdm(range(n_frames), desc=f"Slice {z} frames"):
            fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
            ax.imshow(images[..., z, t], cmap=cmap)

            for label_value, color in active_label_colors.items():
                ax.imshow((labels[..., z, t, None] == label_value) * color)

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
        with imageio.get_writer(
            gif_path,
            mode="I",
            duration=gif_duration_base_ms * t_step,
            loop=gif_loop,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

        gif_paths.append(gif_path)

    return gif_paths


def plot_volume_changes(
    labels: np.ndarray,
    t_step: int,
    filepath: str | Path,
    *,
    voxel_volume_ml: float = 10 / 1000,
    figsize: tuple[float, float] = (5, 4),
    dpi_screen: int = 120,
    dpi_save: int = 300,
    ylabel: str = "Volume (ml, relative)",
) -> Path:
    """
    Plot rough label-volume changes over time for QC.

    Notes
    -----
    This is explicitly a QC plot, not a robust volumetric measurement module.
    The default voxel_volume_ml matches the old bespoke script assumption.

    Label convention currently assumed:
    - 1 -> RV
    - 2 -> MYO
    - 3 -> LV
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_frames = labels.shape[-1]
    xs = np.arange(n_frames) * t_step

    rv_volumes = np.sum(labels == 1, axis=(0, 1, 2)) * voxel_volume_ml
    myo_volumes = np.sum(labels == 2, axis=(0, 1, 2)) * voxel_volume_ml
    lv_volumes = np.sum(labels == 3, axis=(0, 1, 2)) * voxel_volume_ml

    lvef = (
        (max(lv_volumes) - min(lv_volumes)) / max(lv_volumes) * 100
        if max(lv_volumes) > 0
        else np.nan
    )
    rvef = (
        (max(rv_volumes) - min(rv_volumes)) / max(rv_volumes) * 100
        if max(rv_volumes) > 0
        else np.nan
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi_screen)
    ax.plot(xs, rv_volumes, label="RV")
    ax.plot(xs, myo_volumes, label="MYO")
    ax.plot(xs, lv_volumes, label="LV")
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"LVEF ~ {lvef:.2f}%\nRVEF ~ {rvef:.2f}%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi_save, bbox_inches="tight")
    plt.close(fig)

    return filepath
