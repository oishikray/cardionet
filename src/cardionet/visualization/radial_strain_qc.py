from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cardionet.geometry.aha_reference import get_masks
from cardionet.visualization.aha_qc import segment_numbers_for_names


def _safe_peak_metrics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peak_values = np.full(values.shape[0], np.nan, dtype=float)
    peak_frames = np.full(values.shape[0], -1, dtype=int)

    for index, row in enumerate(values):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        peak_frame = int(np.nanargmax(row))
        peak_values[index] = float(row[peak_frame])
        peak_frames[index] = peak_frame

    return peak_values, peak_frames


def _draw_sector_lines(ax, yc: float, xc: float, bounds: np.ndarray, radius: float) -> None:
    for theta in bounds[:-1]:
        y = yc + radius * np.sin(theta)
        x = xc + radius * np.cos(theta)
        ax.plot([xc, x], [yc, y], color="white", linewidth=2)


def save_segment_centroid_overlay(
    label_slice: np.ndarray,
    sector_map: np.ndarray,
    bounds: np.ndarray,
    lv_centroid: tuple[float, float],
    anchor_point: tuple[float, float],
    segment_centroids: np.ndarray,
    segment_names: list[str],
    radial_strain: np.ndarray,
    *,
    ed_frame: int,
    es_frame: int,
    overlay_frame: int,
    outdir: str | Path,
    line_radius: float = 90.0,
) -> Path:
    """
    Save a sector overlay that highlights the tracked segment centroids.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    yc, xc = lv_centroid
    ya, xa = anchor_point

    rv, myo, lv = get_masks(label_slice)
    anatomy_colors = {
        "RV": "#4DB6E2",
        "MYO": "#CDEB00",
        "LV": "#D9D9D9",
    }

    cmap = plt.get_cmap("tab10")
    sector_colors = [cmap(index) for index in range(len(segment_names))]
    segment_numbers = segment_numbers_for_names(segment_names)

    fig, ax = plt.subplots(figsize=(9, 9))

    base = np.zeros((*label_slice.shape, 4), dtype=float)
    base[rv] = plt.matplotlib.colors.to_rgba(anatomy_colors["RV"], alpha=0.95)
    base[myo] = plt.matplotlib.colors.to_rgba(anatomy_colors["MYO"], alpha=0.95)
    base[lv] = plt.matplotlib.colors.to_rgba(anatomy_colors["LV"], alpha=0.95)
    ax.imshow(base)

    overlay = np.zeros((*label_slice.shape, 4), dtype=float)
    for sector_index in range(len(segment_names)):
        overlay[sector_map == sector_index] = (*sector_colors[sector_index][:3], 0.55)
    ax.imshow(overlay)

    ax.scatter(xc, yc, c="red", s=45)
    ax.scatter(xa, ya, c="cyan", s=45)
    _draw_sector_lines(ax, yc, xc, bounds, line_radius)

    for sector_index, centroid in enumerate(segment_centroids):
        if not np.all(np.isfinite(centroid)):
            continue
        ax.scatter(
            centroid[1],
            centroid[0],
            s=55,
            color=sector_colors[sector_index],
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(
            centroid[1] + 1.5,
            centroid[0] + 1.5,
            str(segment_numbers[sector_index]),
            color="white",
            fontsize=9,
            weight="bold",
        )

    ax.set_title(
        f"AHA sector centroids at frame {overlay_frame} "
        f"(ED={ed_frame}, ES={es_frame})"
    )
    ax.axis("off")

    peak_strain, peak_frames = _safe_peak_metrics(radial_strain)

    structure_handles = [
        Patch(facecolor=anatomy_colors["RV"], edgecolor="black", label="RV"),
        Patch(facecolor=anatomy_colors["MYO"], edgecolor="black", label="MYO"),
        Patch(facecolor=anatomy_colors["LV"], edgecolor="black", label="LV"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="LV centroid"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="cyan", markersize=8, label="Anchor point"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=8, label="Segment centroid"),
    ]

    sector_handles = []
    for sector_index, name in enumerate(segment_names):
        label = (
            f"{segment_numbers[sector_index]} {name}\n"
            f"Peak RS={peak_strain[sector_index]:.3f} @ f{peak_frames[sector_index]}"
        )
        sector_handles.append(
            Patch(facecolor=sector_colors[sector_index], edgecolor="black", label=label)
        )

    legend1 = ax.legend(
        handles=structure_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        title="Structures / landmarks",
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=sector_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.48),
        frameon=True,
        fontsize=9,
        title="Segmental radial strain",
    )

    fig.tight_layout()
    outpath = outdir / "radial_strain_centroid_overlay.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return outpath


def save_radial_strain_time_series(
    lv_areas: np.ndarray,
    segment_radial_positions: np.ndarray,
    segment_radial_strain: np.ndarray,
    global_radial_strain: np.ndarray,
    segment_names: list[str],
    *,
    ed_frame: int,
    es_frame: int,
    outdir: str | Path,
) -> list[Path]:
    """
    Save time-series plots for LV area, tracked radial positions, and strain.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    num_frames = lv_areas.shape[0]
    xs = np.arange(num_frames)
    outpaths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, lv_areas, marker="o")
    ax.axvline(ed_frame, linestyle="--", color="black", alpha=0.6, label=f"ED = {ed_frame}")
    ax.axvline(es_frame, linestyle=":", color="black", alpha=0.6, label=f"ES = {es_frame}")
    ax.set_title("LV cavity area over time (chosen slice)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("LV pixels")
    ax.legend()
    fig.tight_layout()

    lv_area_path = outdir / "lv_area_over_time.png"
    fig.savefig(lv_area_path, dpi=150)
    plt.close(fig)
    outpaths.append(lv_area_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    for sector_index, name in enumerate(segment_names):
        ax.plot(xs, segment_radial_positions[sector_index], marker="o", label=name)
    ax.axvline(ed_frame, linestyle="--", color="black", alpha=0.6)
    ax.axvline(es_frame, linestyle=":", color="black", alpha=0.6)
    ax.set_title("Tracked segment centroid radial positions")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Radius from LV centroid (pixels)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    position_path = outdir / "segment_radial_positions_over_time.png"
    fig.savefig(position_path, dpi=150)
    plt.close(fig)
    outpaths.append(position_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    for sector_index, name in enumerate(segment_names):
        ax.plot(xs, segment_radial_strain[sector_index], marker="o", label=name)
    ax.plot(xs, global_radial_strain, color="black", linewidth=2, label="Global")
    ax.axvline(ed_frame, linestyle="--", color="black", alpha=0.6)
    ax.axvline(es_frame, linestyle=":", color="black", alpha=0.6)
    ax.axhline(0.0, linestyle=":", color="gray")
    ax.set_title("Segmental radial strain over time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Radial strain")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    strain_path = outdir / "segment_radial_strain_over_time.png"
    fig.savefig(strain_path, dpi=150)
    plt.close(fig)
    outpaths.append(strain_path)

    return outpaths
