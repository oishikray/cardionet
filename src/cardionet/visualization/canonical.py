from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge

from cardionet.features.aha_segments import get_segment_names, get_segment_numbers
from cardionet.features.delta_t import DeltaTRingMetrics
from cardionet.visualization.segmentation_qc import (
    compute_ejection_fraction,
    compute_label_volumes_riemann,
    compute_lv_slice_quality_mask,
    plot_segmentations_per_slice,
)

RING_ORDER = ("basal", "mid", "apical")
BULLSEYE_RING_SPECS = {
    "basal": {"r_inner": 1.9, "r_outer": 2.9},
    "mid": {"r_inner": 0.95, "r_outer": 1.9},
    "apical": {"r_inner": 0.0, "r_outer": 0.95},
}
SEGMENT_BASE_COLORS = {
    "anterior": "#1f77b4",
    "anteroseptal": "#ff7f0e",
    "inferoseptal": "#2ca02c",
    "inferior": "#d62728",
    "inferolateral": "#9467bd",
    "anterolateral": "#8c564b",
    "septal": "#17becf",
    "lateral": "#e377c2",
}
DEFAULT_MASK_COLORS = {
    1: np.array([1.0, 0.9, 0.0, 1.0]),
    2: np.array([0.0, 0.85, 0.25, 1.0]),
    3: np.array([1.0, 0.0, 0.0, 1.0]),
}
DEFAULT_BULLSEYE_CMAP = "coolwarm_r"


def segment_base_name(segment_name: str) -> str:
    """Return the circumferential segment name without basal/mid/apical prefix."""
    parts = segment_name.strip().lower().split(maxsplit=1)
    if len(parts) == 2 and parts[0] in {"basal", "mid", "apical"}:
        return parts[1]
    return segment_name.strip().lower()


def segment_color(segment_name: str) -> str:
    """Return the canonical time-series color for a segment name."""
    return SEGMENT_BASE_COLORS.get(segment_base_name(segment_name), "#4d4d4d")


def finite_metric_limits(
    values_by_ring: Mapping[str, np.ndarray],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    """Resolve heatmap limits across all finite ring values."""
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)

    finite_values = []
    for values in values_by_ring.values():
        arr = np.asarray(values, dtype=float)
        finite_values.extend(arr[np.isfinite(arr)].tolist())

    if not finite_values:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(np.min(finite_values)), float(np.max(finite_values))

    if vmin is not None:
        lo = float(vmin)
    if vmax is not None:
        hi = float(vmax)
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return lo, hi


def bullseye_wedge_angles(index: int, n_segments: int) -> tuple[float, float, float]:
    """Return AHA wedge bounds with segment numbers advancing counterclockwise."""
    width = 360.0 / n_segments
    center = 90.0 + index * width
    return center - width / 2.0, center + width / 2.0, center


def save_metric_bullseye(
    values_by_ring: Mapping[str, Sequence[float] | np.ndarray],
    outdir: str | Path,
    *,
    patient_id: str,
    metric_name: str,
    filename: str | None = None,
    cmap_name: str = DEFAULT_BULLSEYE_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    value_format: str = ".2f",
    legend_label: str | None = None,
) -> Path:
    """Save a canonical AHA bullseye heatmap for one segment-level metric."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_filename = filename or f"{metric_name.lower().replace(' ', '_')}_bullseye.png"

    ring_values = {
        ring: np.asarray(values, dtype=float)
        for ring, values in values_by_ring.items()
        if ring in BULLSEYE_RING_SPECS
    }
    lo, hi = finite_metric_limits(ring_values, vmin=vmin, vmax=vmax)
    norm = Normalize(vmin=lo, vmax=hi)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(10, 11.5))
    for ring in RING_ORDER:
        if ring not in ring_values:
            continue
        values = ring_values[ring]
        segment_names = get_segment_names(ring)
        segment_numbers = get_segment_numbers(ring)
        if len(values) != len(segment_names):
            raise ValueError(
                f"Expected {len(segment_names)} values for {ring}, got {len(values)}"
            )

        spec = BULLSEYE_RING_SPECS[ring]
        for index, value in enumerate(values):
            theta1, theta2, theta_mid_deg = bullseye_wedge_angles(index, len(values))
            facecolor = cmap(norm(value)) if np.isfinite(value) else (0.85, 0.85, 0.85, 1.0)
            wedge = Wedge(
                (0.0, 0.0),
                spec["r_outer"],
                theta1,
                theta2,
                width=spec["r_outer"] - spec["r_inner"],
                facecolor=facecolor,
                edgecolor="white",
                linewidth=2.0,
            )
            ax.add_patch(wedge)

            theta_mid = np.deg2rad(theta_mid_deg)
            radius_mid = (spec["r_inner"] + spec["r_outer"]) / 2.0
            x = radius_mid * np.cos(theta_mid)
            y = radius_mid * np.sin(theta_mid)
            value_text = format(float(value), value_format) if np.isfinite(value) else "NA"
            ax.text(
                x,
                y,
                f"{segment_numbers[index]}\n{value_text}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "boxstyle": "round,pad=0.25",
                },
            )

    ax.set_aspect("equal")
    ax.set_xlim(-3.15, 3.15)
    ax.set_ylim(-3.15, 3.15)
    ax.axis("off")
    ax.set_title(f"{patient_id} {metric_name} AHA bullseye")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(legend_label or metric_name)

    legend_handles = []
    for ring in RING_ORDER:
        if ring not in ring_values:
            continue
        for number, name in zip(get_segment_numbers(ring), get_segment_names(ring)):
            legend_handles.append(Patch(facecolor="white", edgecolor="black", label=f"{number}: {name}"))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        title="AHA segments",
        fontsize=8,
        frameon=True,
        ncol=3,
    )

    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    outpath = outdir / resolved_filename
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _peak_values(values: np.ndarray, peak_mode: str) -> tuple[np.ndarray, np.ndarray]:
    peaks = np.full(values.shape[0], np.nan, dtype=float)
    frames = np.full(values.shape[0], -1, dtype=int)
    for index, row in enumerate(values):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        if peak_mode == "min":
            frame = int(np.nanargmin(row))
        elif peak_mode == "max":
            frame = int(np.nanargmax(row))
        elif peak_mode == "absmax":
            frame = int(np.nanargmax(np.abs(row)))
        else:
            raise ValueError(f"Unsupported peak_mode: {peak_mode}")
        frames[index] = frame
        peaks[index] = float(row[frame])
    return peaks, frames


def save_ring_time_series_plots(
    series_by_ring: Mapping[str, np.ndarray],
    outdir: str | Path,
    *,
    patient_id: str,
    metric_name: str,
    ylabel: str | None = None,
    filename_prefix: str | None = None,
    ed_frame: int | None = None,
    es_frame: int | None = None,
    peak_mode: str = "max",
    baseline_value: float | None = None,
    frame_labels: Sequence[float] | None = None,
    include_std_legend: bool = False,
    std_legend_title: str = "NWT SD",
    y_limits: tuple[float, float] | None = None,
    title_suffix: str = "",
) -> list[Path]:
    """Save one canonical segment time-series plot per AHA ring."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or metric_name.lower().replace(" ", "_")
    outpaths: list[Path] = []

    for ring in RING_ORDER:
        if ring not in series_by_ring:
            continue
        values = np.asarray(series_by_ring[ring], dtype=float)
        if values.ndim != 2:
            raise ValueError(f"Expected {ring} values shaped (segments, frames), got {values.shape}")

        segment_names = get_segment_names(ring)
        segment_numbers = get_segment_numbers(ring)
        if values.shape[0] != len(segment_names):
            raise ValueError(
                f"Expected {len(segment_names)} {ring} segments, got {values.shape[0]}"
            )

        xs = np.asarray(frame_labels, dtype=float) if frame_labels is not None else np.arange(values.shape[1])
        if xs.shape[0] != values.shape[1]:
            raise ValueError(f"Expected {values.shape[1]} frame labels, got {xs.shape[0]}")

        _, peak_frames = _peak_values(values, peak_mode)
        fig, ax = plt.subplots(figsize=(12, 6.8 if include_std_legend else 6))
        for index, name in enumerate(segment_names):
            color = segment_color(name)
            label = f"{segment_numbers[index]} {segment_base_name(name).title()}"
            ax.plot(xs, values[index], marker="o", linewidth=1.8, color=color, label=label)
            peak_frame = peak_frames[index]
            if peak_frame >= 0:
                ax.scatter(
                    xs[peak_frame],
                    values[index, peak_frame],
                    s=70,
                    marker="*",
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=4,
                )

        if ed_frame is not None:
            ax.axvline(xs[ed_frame], color="black", linestyle="--", linewidth=1.4, label=f"ED f{ed_frame}")
        if es_frame is not None:
            ax.axvline(xs[es_frame], color="#54278f", linestyle=":", linewidth=1.8, label=f"ES f{es_frame}")
        if baseline_value is not None:
            ax.axhline(baseline_value, color="gray", linestyle=":", linewidth=1.1)
        if y_limits is not None:
            lo, hi = (float(y_limits[0]), float(y_limits[1]))
            if np.isclose(lo, hi):
                hi = lo + 1e-6
            ax.set_ylim(lo, hi)

        peak_handle = Line2D(
            [0],
            [0],
            marker="*",
            color="white",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="None",
            label=f"Per-segment {peak_mode} peak",
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(peak_handle)
        labels.append(peak_handle.get_label())

        ax.set_title(f"{patient_id} {ring.title()} {metric_name} over time{title_suffix}")
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel or metric_name)
        ax.grid(alpha=0.25)
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.80, 0.5),
            fontsize=9,
            frameon=True,
        )

        if include_std_legend:
            std_values = np.nanstd(values, axis=1)
            std_handles = [
                Line2D(
                    [0],
                    [0],
                    color=segment_color(name),
                    linewidth=2.0,
                    label=(
                        f"{segment_numbers[index]} "
                        f"{segment_base_name(name).title()}: {std_values[index]:.3f}"
                    ),
                )
                for index, name in enumerate(segment_names)
            ]
            fig.legend(
                handles=std_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(3, len(std_handles)),
                fontsize=8,
                title=std_legend_title,
                frameon=True,
            )

        fig.tight_layout(rect=(0.0, 0.11 if include_std_legend else 0.0, 0.78, 1.0))

        outpath = outdir / f"{prefix}_{ring}_time_series.png"
        fig.savefig(outpath, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outpaths.append(outpath)

    return outpaths


def save_mask_overlay_gifs(
    images: np.ndarray,
    labels: np.ndarray,
    outdir: str | Path,
    *,
    patient_id: str,
    t_step: int = 1,
    basename: str | None = None,
) -> list[Path]:
    """Save per-slice cine GIFs with segmentation masks overlaid on SAX images."""
    return plot_segmentations_per_slice(
        images=images,
        labels=labels,
        t_step=t_step,
        output_dir=outdir,
        basename=basename or f"{patient_id}_sax_mask_overlay",
        label_colors=DEFAULT_MASK_COLORS,
    )


def save_delta_t_plots(
    delta_t_by_ring: Mapping[str, DeltaTRingMetrics],
    outdir: str | Path,
    *,
    patient_id: str,
    filename_prefix: str = "delta_t",
) -> list[Path]:
    """Save one deltaT summary plot per AHA ring."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpaths: list[Path] = []

    for ring in RING_ORDER:
        if ring not in delta_t_by_ring:
            continue

        metrics = delta_t_by_ring[ring]
        segment_names = get_segment_names(ring)
        segment_numbers = get_segment_numbers(ring)
        x = np.arange(len(segment_names))
        labels = [
            f"{number}\n{segment_base_name(name).title()}"
            for number, name in zip(segment_numbers, segment_names)
        ]

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.15]},
        )
        bar_colors = [segment_color(name) for name in segment_names]

        axes[0].bar(x, metrics.delta_t, color=bar_colors, edgecolor="black", linewidth=0.6)
        axes[0].axhline(0.0, color="black", linewidth=1.0)
        axes[0].set_ylabel("deltaT\n(frames)")

        axes[1].bar(x, metrics.abs_delta_t, color=bar_colors, edgecolor="black", linewidth=0.6)
        axes[1].set_ylabel("absdeltaT\n(frames)")

        axes[2].bar(x, metrics.delta_t_ms, color=bar_colors, edgecolor="black", linewidth=0.6)
        axes[2].axhline(0.0, color="black", linewidth=1.0)
        axes[2].set_ylabel("deltaTms\n(ms)")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels)

        for ax in axes:
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(
            f"{patient_id} {ring.title()} deltaT: peak NWT frame - ES frame\n"
            f"ES frame = {metrics.es_frame}; frame interval = "
            f"{metrics.frame_interval_ms:.3f} ms"
        )

        legend_handles = [
            Patch(facecolor=segment_color(name), edgecolor="black", label=f"{number}: {name}")
            for number, name in zip(segment_numbers, segment_names)
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(3, len(legend_handles)),
            fontsize=8,
            title="AHA segments",
            frameon=True,
        )
        fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))

        outpath = outdir / f"{filename_prefix}_{ring}.png"
        fig.savefig(outpath, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outpaths.append(outpath)

    return outpaths


def save_lv_volume_curve(
    labels: np.ndarray,
    outdir: str | Path,
    *,
    patient_id: str,
    spacing_mm: tuple[float, float, float],
    ed_frame: int | None = None,
    es_frame: int | None = None,
    drop_poor_lv_slices: bool = False,
    lv_min_peak_area_fraction: float = 0.08,
    lv_apical_min_peak_area_fraction: float = 0.20,
    lv_max_es_ed_area_ratio: float = 1.05,
    lv_apical_max_es_ed_area_ratio: float = 1.0,
    filename: str = "lv_volume_ef_over_time.png",
) -> Path:
    """Save LV volume curves using direct Riemann voxel summation and report EF."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
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

    lv_volumes = compute_label_volumes_riemann(
        labels,
        label_value=3,
        spacing_mm=spacing_mm,
        include_slices=include_lv_slices,
    )
    ef = compute_ejection_fraction(lv_volumes)
    resolved_ed = int(np.nanargmax(lv_volumes)) if ed_frame is None and np.any(np.isfinite(lv_volumes)) else ed_frame
    resolved_es = int(np.nanargmin(lv_volumes)) if es_frame is None and np.any(np.isfinite(lv_volumes)) else es_frame

    xs = np.arange(lv_volumes.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, lv_volumes, marker="o", color="#08519c", label="LV")
    if resolved_ed is not None:
        ax.axvline(resolved_ed, color="black", linestyle="--", linewidth=1.5, label=f"ED f{resolved_ed}")
        ax.scatter(resolved_ed, lv_volumes[resolved_ed], s=80, color="black", zorder=4)
    if resolved_es is not None:
        ax.axvline(resolved_es, color="#54278f", linestyle=":", linewidth=1.8, label=f"ES f{resolved_es}")
        ax.scatter(resolved_es, lv_volumes[resolved_es], s=80, color="#54278f", zorder=4)
    slice_note = ""
    if lv_slice_qc is not None:
        slice_note = (
            f"\nLV slices kept {lv_slice_qc.kept_indices}; "
            f"dropped {lv_slice_qc.dropped_indices}"
        )
    ax.set_title(
        f"{patient_id} LV volume over time | EF = {ef:.2f}%\n"
        f"Spacing = {spacing_mm[0]:.3g} x {spacing_mm[1]:.3g} x {spacing_mm[2]:.3g} mm"
        f"{slice_note}"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("LV volume (ml)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()

    outpath = outdir / filename
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return outpath
