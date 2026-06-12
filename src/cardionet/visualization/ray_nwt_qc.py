from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image

from cardionet.features.aha_segments import build_sector_map
from cardionet.features.slices import label_slice_yx
from cardionet.features.stack_aha import AHASliceFeatureSet
from cardionet.features.wall_thickness import sample_ray_labels, thickness_from_transitions
from cardionet.geometry.aha_reference import get_masks
from cardionet.visualization.segmentation_qc import LABEL_COLORS, _draw_label_contours


def _save_gif_with_global_palette(
    frames: list[Image.Image],
    gif_path: Path,
    *,
    duration_ms: int,
) -> None:
    """
    Save a GIF using one shared palette for all frames.

    GIF encoders often quantize RGB frames independently. That makes static
    elements such as colorbars appear to change hue even when the source RGB
    pixels are fixed. Building one palette from all frames keeps the colorbar
    and other static colors stable across the animation.
    """
    if not frames:
        raise ValueError("Cannot write a GIF with zero frames.")

    rgb_frames = [frame.convert("RGB") for frame in frames]
    samples = []
    for frame in rgb_frames:
        array = np.asarray(frame)
        flat = array.reshape(-1, 3)
        step = max(1, flat.shape[0] // 4096)
        samples.append(flat[::step])

    palette_samples = np.concatenate(samples, axis=0).astype(np.uint8)
    palette_source = Image.fromarray(palette_samples.reshape(1, -1, 3), mode="RGB")
    palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    paletted_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in rgb_frames
    ]
    paletted_frames[0].save(
        gif_path,
        save_all=True,
        append_images=paletted_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def raw_ray_nwt_matrix(
    feature: AHASliceFeatureSet,
    *,
    wt_matrix: np.ndarray | None = None,
    denominator: float | None = None,
) -> np.ndarray:
    """Return raw per-ray NWT without AHA-sector binning."""
    active_wt_matrix = feature.wt_matrix if wt_matrix is None else wt_matrix
    active_denominator = (
        float(feature.initial_epicardial_radius)
        if denominator is None
        else float(denominator)
    )
    return np.divide(
        active_wt_matrix,
        active_denominator,
        out=np.full_like(active_wt_matrix, np.nan, dtype=float),
        where=(
            np.isfinite(active_wt_matrix)
            & np.isfinite(active_denominator)
            & (active_denominator != 0)
        ),
    )


def _ray_segments_for_frame(
    feature: AHASliceFeatureSet,
    frame_index: int,
    *,
    wt_matrix: np.ndarray | None = None,
    epicardial_radius_matrix: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
    denominator: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build colored line segments from endocardium to epicardium for finite rays."""
    active_wt_matrix = feature.wt_matrix if wt_matrix is None else wt_matrix
    active_epi_matrix = (
        feature.epicardial_radius_matrix
        if epicardial_radius_matrix is None
        else epicardial_radius_matrix
    )
    active_centroids = feature.centroids if centroids is None else centroids
    wt_frame = active_wt_matrix[:, frame_index]
    epi_frame = active_epi_matrix[:, frame_index]
    nwt_frame = raw_ray_nwt_matrix(
        feature,
        wt_matrix=active_wt_matrix,
        denominator=denominator,
    )[:, frame_index]
    yc, xc = active_centroids[frame_index]

    if not np.isfinite(yc) or not np.isfinite(xc):
        return np.empty((0, 2, 2), dtype=float), np.empty(0, dtype=float)

    finite = np.isfinite(wt_frame) & np.isfinite(epi_frame) & np.isfinite(nwt_frame)
    if not np.any(finite):
        return np.empty((0, 2, 2), dtype=float), np.empty(0, dtype=float)

    angles = feature.angles[finite]
    r_epi = epi_frame[finite]
    r_endo = r_epi - wt_frame[finite]
    values = nwt_frame[finite]

    y0 = yc + r_endo * np.sin(angles)
    x0 = xc + r_endo * np.cos(angles)
    y1 = yc + r_epi * np.sin(angles)
    x1 = xc + r_epi * np.cos(angles)

    segments = np.stack(
        [
            np.stack([x0, y0], axis=1),
            np.stack([x1, y1], axis=1),
        ],
        axis=1,
    )
    return segments, values


def _boundary_segments_for_frame(
    label_slice: np.ndarray,
    feature: AHASliceFeatureSet,
    frame_index: int,
    *,
    ray_step: float,
    max_radius: float,
    centroids: np.ndarray | None = None,
) -> np.ndarray:
    """Return AHA segment boundary rays clipped to the myocardial wall."""
    active_centroids = feature.centroids if centroids is None else centroids
    yc, xc = active_centroids[frame_index]
    if not np.isfinite(yc) or not np.isfinite(xc):
        return np.empty((0, 2, 2), dtype=float)

    segments = []
    for theta in feature.bounds[:-1]:
        samples = sample_ray_labels(
            label_slice,
            float(yc),
            float(xc),
            float(theta),
            step=ray_step,
            max_radius=max_radius,
        )
        r_endo, r_epi, _ = thickness_from_transitions(samples)
        if not np.isfinite(r_endo) or not np.isfinite(r_epi):
            continue

        y0 = yc + r_endo * np.sin(theta)
        x0 = xc + r_endo * np.cos(theta)
        y1 = yc + r_epi * np.sin(theta)
        x1 = xc + r_epi * np.cos(theta)
        segments.append([[x0, y0], [x1, y1]])

    if not segments:
        return np.empty((0, 2, 2), dtype=float)
    return np.asarray(segments, dtype=float)


def _draw_segment_numbers(
    ax: plt.Axes,
    label_slice: np.ndarray,
    feature: AHASliceFeatureSet,
    frame_index: int,
    *,
    centroids: np.ndarray | None = None,
) -> None:
    """Draw AHA segment numbers at framewise myocardial sector centroids."""
    active_centroids = feature.centroids if centroids is None else centroids
    yc, xc = active_centroids[frame_index]
    if not np.isfinite(yc) or not np.isfinite(xc):
        return

    _, myo, _ = get_masks(label_slice)
    if not np.any(myo):
        return

    sector_map = build_sector_map(float(yc), float(xc), myo, feature.bounds)
    for sector_index, segment_number in enumerate(feature.segment_numbers):
        coords = np.argwhere(sector_map == sector_index)
        if coords.size == 0:
            continue

        y = float(coords[:, 0].mean())
        x = float(coords[:, 1].mean())
        ax.text(
            x,
            y,
            str(segment_number),
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="white",
            zorder=8,
            bbox={
                "boxstyle": "circle,pad=0.22",
                "facecolor": "black",
                "alpha": 0.68,
                "linewidth": 0,
            },
        )


def compute_fixed_center_ray_matrices(
    labels: np.ndarray,
    feature: AHASliceFeatureSet,
    *,
    fixed_frame: int,
    ray_step: float,
    max_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute raw ray WT/epicardial radii using one fixed LV centroid for all frames.

    Returns
    -------
    tuple
        ``(wt_matrix, epicardial_radius_matrix, centroids, denominator)`` where
        ``centroids`` repeats the fixed reference centroid across all frames.
    """
    if fixed_frame < 0 or fixed_frame >= labels.shape[-1]:
        raise IndexError(f"fixed_frame {fixed_frame} is out of bounds for {labels.shape[-1]} frames")

    fixed_centroid = feature.centroids[fixed_frame]
    if not np.all(np.isfinite(fixed_centroid)):
        raise ValueError(f"Reference frame {fixed_frame} does not have a finite LV centroid.")

    num_rays = len(feature.angles)
    num_frames = labels.shape[-1]
    wt_matrix = np.full((num_rays, num_frames), np.nan, dtype=float)
    epicardial_radius_matrix = np.full((num_rays, num_frames), np.nan, dtype=float)

    yc, xc = fixed_centroid
    for frame_index in range(num_frames):
        label_slice = label_slice_yx(labels, feature.slice_index, frame_index)
        for ray_index, theta in enumerate(feature.angles):
            samples = sample_ray_labels(
                label_slice,
                float(yc),
                float(xc),
                float(theta),
                step=ray_step,
                max_radius=max_radius,
            )
            _, r_epi, thickness = thickness_from_transitions(samples)
            wt_matrix[ray_index, frame_index] = thickness
            epicardial_radius_matrix[ray_index, frame_index] = r_epi

    fixed_centroids = np.repeat(fixed_centroid[None, :], num_frames, axis=0)
    ed_epicardial_radii = epicardial_radius_matrix[:, fixed_frame]
    finite_ed_radii = ed_epicardial_radii[np.isfinite(ed_epicardial_radii)]
    if finite_ed_radii.size == 0:
        raise ValueError(
            f"No finite fixed-center epicardial radii found at reference frame {fixed_frame}."
        )
    denominator = float(np.mean(finite_ed_radii))

    return wt_matrix, epicardial_radius_matrix, fixed_centroids, denominator


def _render_ray_nwt_frame(
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    feature: AHASliceFeatureSet,
    frame_index: int,
    *,
    norm: Normalize,
    cmap_name: str,
    ray_step: float,
    max_radius: float,
    wt_matrix: np.ndarray | None,
    epicardial_radius_matrix: np.ndarray | None,
    centroids: np.ndarray | None,
    denominator: float | None,
    figure_size: tuple[float, float],
    dpi: int,
    contour_line_width: float,
    ray_line_width: float,
    boundary_line_width: float,
) -> Image.Image:
    """Render one frame as a PIL image."""
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    ax.imshow(image_slice, cmap="gray")
    _draw_label_contours(
        ax,
        label_slice,
        LABEL_COLORS,
        line_width=contour_line_width,
    )

    ray_segments, ray_values = _ray_segments_for_frame(
        feature,
        frame_index,
        wt_matrix=wt_matrix,
        epicardial_radius_matrix=epicardial_radius_matrix,
        centroids=centroids,
        denominator=denominator,
    )
    if len(ray_segments) > 0:
        ray_collection = LineCollection(
            ray_segments,
            array=ray_values,
            cmap=plt.get_cmap(cmap_name),
            norm=norm,
            linewidths=ray_line_width,
            alpha=0.95,
            zorder=4,
        )
        ax.add_collection(ray_collection)

    boundary_segments = _boundary_segments_for_frame(
        label_slice,
        feature,
        frame_index,
        ray_step=ray_step,
        max_radius=max_radius,
        centroids=centroids,
    )
    if len(boundary_segments) > 0:
        ax.add_collection(
            LineCollection(
                boundary_segments,
                colors="white",
                linewidths=boundary_line_width + 1.2,
                alpha=0.95,
                zorder=5,
            )
        )
        ax.add_collection(
            LineCollection(
                boundary_segments,
                colors="black",
                linewidths=boundary_line_width,
                alpha=0.95,
                zorder=6,
            )
        )

    ax.set_title(
        f"Slice {feature.slice_index} {feature.slice_type} | Frame {frame_index} | raw ray NWT"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    cbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Raw ray NWT")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    copied = image.copy()
    buf.close()
    plt.close(fig)
    return copied


def save_ray_nwt_frame_series(
    images: np.ndarray,
    labels: np.ndarray,
    feature: AHASliceFeatureSet,
    output_dir: str | Path,
    *,
    ray_step: float,
    max_radius: float,
    basename: str,
    cmap_name: str = "coolwarm_r",
    figure_size: tuple[float, float] = (6, 6),
    dpi: int = 150,
    gif_duration_ms: int = 80,
    contour_line_width: float = 1.4,
    ray_line_width: float = 2.0,
    boundary_line_width: float = 1.7,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_slack_fraction: float = 0.03,
    save_frame_pngs: bool = True,
    wt_matrix: np.ndarray | None = None,
    epicardial_radius_matrix: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
    denominator: float | None = None,
) -> tuple[list[Path], Path]:
    """
    Save per-frame raw ray NWT PNGs and a GIF with a shared blue-red color scale.
    """
    if images.shape != labels.shape:
        raise ValueError(f"images and labels must match, got {images.shape} vs {labels.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if save_frame_pngs:
        frames_dir.mkdir(parents=True, exist_ok=True)

    active_wt_matrix = feature.wt_matrix if wt_matrix is None else wt_matrix
    raw_nwt = raw_ray_nwt_matrix(
        feature,
        wt_matrix=active_wt_matrix,
        denominator=denominator,
    )
    finite = raw_nwt[np.isfinite(raw_nwt)]
    if finite.size == 0:
        raise ValueError("No finite raw ray NWT values are available for rendering.")

    resolved_vmin = float(np.nanmin(finite)) if vmin is None else float(vmin)
    resolved_vmax = float(np.nanmax(finite)) if vmax is None else float(vmax)
    if vmin is None and vmax is None:
        value_range = resolved_vmax - resolved_vmin
        if value_range > 0:
            slack = value_range * float(colorbar_slack_fraction)
            resolved_vmin -= slack
            resolved_vmax += slack
    if resolved_vmax <= resolved_vmin:
        resolved_vmax = resolved_vmin + 1e-6
    norm = Normalize(vmin=resolved_vmin, vmax=resolved_vmax)

    frame_paths: list[Path] = []
    gif_frames: list[Image.Image] = []
    num_frames = labels.shape[-1]
    for frame_index in range(num_frames):
        image_slice = images[:, :, feature.slice_index, frame_index].T
        label_slice = label_slice_yx(labels, feature.slice_index, frame_index)
        frame_image = _render_ray_nwt_frame(
            image_slice,
            label_slice,
            feature,
            frame_index,
            norm=norm,
            cmap_name=cmap_name,
            ray_step=ray_step,
            max_radius=max_radius,
            wt_matrix=wt_matrix,
            epicardial_radius_matrix=epicardial_radius_matrix,
            centroids=centroids,
            denominator=denominator,
            figure_size=figure_size,
            dpi=dpi,
            contour_line_width=contour_line_width,
            ray_line_width=ray_line_width,
            boundary_line_width=boundary_line_width,
        )
        frame_path = frames_dir / f"{basename}_frame_{frame_index:03d}.png"
        if save_frame_pngs:
            frame_image.save(frame_path)
            frame_paths.append(frame_path)
        gif_frames.append(frame_image)

    gif_path = output_dir / f"{basename}.gif"
    _save_gif_with_global_palette(
        gif_frames,
        gif_path,
        duration_ms=gif_duration_ms,
    )

    return frame_paths, gif_path


def save_aha_boundary_contour_gif(
    images: np.ndarray,
    labels: np.ndarray,
    feature: AHASliceFeatureSet,
    output_dir: str | Path,
    *,
    ray_step: float,
    max_radius: float,
    basename: str,
    figure_size: tuple[float, float] = (6, 6),
    dpi: int = 150,
    gif_duration_ms: int = 80,
    contour_line_width: float = 1.4,
    boundary_line_width: float = 1.7,
    show_segment_numbers: bool = False,
) -> Path:
    """Save a contour GIF with AHA segment boundary rays clipped to the myocardium."""
    if images.shape != labels.shape:
        raise ValueError(f"images and labels must match, got {images.shape} vs {labels.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gif_frames: list[Image.Image] = []
    for frame_index in range(labels.shape[-1]):
        image_slice = images[:, :, feature.slice_index, frame_index].T
        label_slice = label_slice_yx(labels, feature.slice_index, frame_index)

        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
        ax.imshow(image_slice, cmap="gray")
        _draw_label_contours(
            ax,
            label_slice,
            LABEL_COLORS,
            line_width=contour_line_width,
        )

        boundary_segments = _boundary_segments_for_frame(
            label_slice,
            feature,
            frame_index,
            ray_step=ray_step,
            max_radius=max_radius,
        )
        if len(boundary_segments) > 0:
            ax.add_collection(
                LineCollection(
                    boundary_segments,
                    colors="white",
                    linewidths=boundary_line_width + 1.2,
                    alpha=0.95,
                    zorder=5,
                )
            )
            ax.add_collection(
                LineCollection(
                    boundary_segments,
                    colors="black",
                    linewidths=boundary_line_width,
                    alpha=0.95,
                    zorder=6,
                )
            )

        if show_segment_numbers:
            _draw_segment_numbers(ax, label_slice, feature, frame_index)

        ax.set_title(
            f"Slice {feature.slice_index} {feature.slice_type} | Frame {frame_index}"
        )
        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, dpi=dpi)
        buf.seek(0)
        frame_image = Image.open(buf).convert("RGB")
        gif_frames.append(frame_image.copy())
        buf.close()
        plt.close(fig)

    gif_path = output_dir / f"{basename}.gif"
    _save_gif_with_global_palette(
        gif_frames,
        gif_path,
        duration_ms=gif_duration_ms,
    )

    return gif_path
