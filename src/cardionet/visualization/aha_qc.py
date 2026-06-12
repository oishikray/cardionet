from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge

from cardionet.features.stack_aha import AHAChunkFeatureSet, AHAStackFeatureSet, CHUNK_ORDER
from cardionet.features.aha_segments import get_segment_numbers
from cardionet.geometry.aha_reference import get_masks


BULLSEYE_RING_SPECS = {
    "basal": {"r_inner": 1.9, "r_outer": 2.9},
    "mid": {"r_inner": 0.95, "r_outer": 1.9},
    "apical": {"r_inner": 0.0, "r_outer": 0.95},
}


def bullseye_wedge_angles(index: int, n_segments: int) -> tuple[float, float, float]:
    """Return AHA wedge bounds with segment numbers advancing counterclockwise."""
    width_deg = 360.0 / n_segments
    center_deg = 90.0 + index * width_deg
    return center_deg - width_deg / 2.0, center_deg + width_deg / 2.0, center_deg


def _safe_peak_metrics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row peak values and frames while tolerating all-NaN rows."""
    peak_values = np.full(values.shape[0], np.nan, dtype=float)
    peak_frames = np.full(values.shape[0], -1, dtype=int)

    for index in range(values.shape[0]):
        row = values[index]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        peak_idx = int(np.nanargmax(row))
        peak_frames[index] = peak_idx
        peak_values[index] = float(row[peak_idx])

    return peak_values, peak_frames


def moving_average_nan(x: np.ndarray, window: int = 11) -> np.ndarray:
    """
    Simple NaN-aware moving average for QC visualization.
    """
    if window <= 1:
        return x.copy()

    out = np.full_like(x, np.nan, dtype=float)
    pad = window // 2

    for i in range(len(x)):
        lo = max(0, i - pad)
        hi = min(len(x), i + pad + 1)
        vals = x[lo:hi]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            out[i] = vals.mean()

    return out


def draw_sector_lines(ax, yc: float, xc: float, bounds: np.ndarray, radius: float) -> None:
    for theta in bounds[:-1]:
        y = yc + radius * np.sin(theta)
        x = xc + radius * np.cos(theta)
        ax.plot([xc, x], [yc, y], color="white", linewidth=2)


def _ring_type_from_segment_names(segment_names: list[str]) -> str | None:
    if not segment_names:
        return None
    first = segment_names[0].lower()
    if first.startswith("basal "):
        return "basal"
    if first.startswith("mid "):
        return "mid"
    if first.startswith("apical "):
        return "apical"
    if first == "apex":
        return "apex"
    return None


def segment_numbers_for_names(segment_names: list[str]) -> list[int]:
    ring_type = _ring_type_from_segment_names(segment_names)
    if ring_type is None:
        return list(range(1, len(segment_names) + 1))
    numbers = get_segment_numbers(ring_type)
    if len(numbers) != len(segment_names):
        return list(range(1, len(segment_names) + 1))
    return numbers


def _draw_sector_number_labels(ax, sector_map: np.ndarray, segment_numbers: list[int]) -> None:
    for sector_index, segment_number in enumerate(segment_numbers):
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
            bbox={"boxstyle": "circle,pad=0.22", "facecolor": "black", "alpha": 0.55, "linewidth": 0},
        )


def save_structure_and_sector_overlay(
    label_slice: np.ndarray,
    sector_map: np.ndarray,
    bounds: np.ndarray,
    lv_centroid: tuple[float, float],
    contact_centroid: tuple[float, float],
    segment_names: list[str],
    binned_wt: np.ndarray,
    nwt: np.ndarray,
    ed_frame: int,
    outdir: str | Path,
    *,
    line_radius: float = 90.0,
) -> Path:
    """
    Save anatomy + AHA sector overlay with summary metrics.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    yc, xc = lv_centroid
    yrv, xrv = contact_centroid

    rv, myo, lv = get_masks(label_slice)

    anatomy_colors = {
        "RV": "#4DB6E2",
        "MYO": "#CDEB00",
        "LV": "#D9D9D9",
    }

    cmap = plt.get_cmap("tab10")
    sector_colors = [cmap(i) for i in range(len(segment_names))]
    segment_numbers = segment_numbers_for_names(segment_names)

    fig, ax = plt.subplots(figsize=(9, 9))

    base = np.zeros((*label_slice.shape, 4), dtype=float)
    base[rv] = plt.matplotlib.colors.to_rgba(anatomy_colors["RV"], alpha=0.95)
    base[myo] = plt.matplotlib.colors.to_rgba(anatomy_colors["MYO"], alpha=0.95)
    base[lv] = plt.matplotlib.colors.to_rgba(anatomy_colors["LV"], alpha=0.95)
    ax.imshow(base)

    overlay = np.zeros((*label_slice.shape, 4), dtype=float)
    for s in range(len(segment_names)):
        mask = sector_map == s
        overlay[mask] = (*sector_colors[s][:3], 0.55)
    ax.imshow(overlay)

    ax.scatter(xc, yc, c="red", s=45)
    ax.scatter(xrv, yrv, c="cyan", s=45)
    draw_sector_lines(ax, yc, xc, bounds, line_radius)
    _draw_sector_number_labels(ax, sector_map, segment_numbers)

    ax.set_title("AHA-aligned sector map with anatomy + segment metrics")
    ax.axis("off")

    structure_handles = [
        Patch(facecolor=anatomy_colors["RV"], edgecolor="black", label="RV"),
        Patch(facecolor=anatomy_colors["MYO"], edgecolor="black", label="MYO"),
        Patch(facecolor=anatomy_colors["LV"], edgecolor="black", label="LV"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="LV centroid"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="cyan", markersize=8, label="Anchor point"),
    ]

    peak_nwt, peak_frame = _safe_peak_metrics(nwt)
    ed_wt = binned_wt[:, ed_frame]

    sector_handles = []
    for s, name in enumerate(segment_names):
        label = (
            f"{segment_numbers[s]} {name}\n"
            f"ED WT={ed_wt[s]:.2f}px | Peak NWT={peak_nwt[s]:.2f} @ f{peak_frame[s]}"
        )
        sector_handles.append(
            Patch(facecolor=sector_colors[s], edgecolor="black", label=label)
        )

    leg1 = ax.legend(
        handles=structure_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        frameon=True,
        title="Structures / landmarks",
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=sector_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.52),
        frameon=True,
        title="AHA sectors + metrics",
        fontsize=9,
    )

    fig.tight_layout()
    outpath = outdir / "aha_sector_map_with_legend.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return outpath


def save_debug_wt_plot(
    wt_frame: np.ndarray,
    wt_frame_smooth: np.ndarray,
    ed_frame: int,
    outdir: str | Path,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wt_frame, label="raw thickness")
    ax.plot(wt_frame_smooth, label="smoothed thickness")
    ax.set_title(f"WT vs angle at ED frame {ed_frame}")
    ax.set_xlabel("Ray index")
    ax.set_ylabel("Thickness (pixels)")
    ax.legend()

    fig.tight_layout()
    outpath = outdir / "debug_wt_vs_angle.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return outpath


def save_time_series_plots(
    lv_areas: np.ndarray,
    binned_wt: np.ndarray,
    nwt: np.ndarray,
    ed_frame: int,
    segment_names: list[str],
    outdir: str | Path,
) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t = len(lv_areas)
    xs = np.arange(t)

    outpaths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, lv_areas, marker="o")
    ax.axvline(ed_frame, linestyle="--", label=f"ED frame = {ed_frame}")
    ax.set_title("LV cavity area over time (chosen slice)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("LV pixels")
    ax.legend()
    fig.tight_layout()

    path1 = outdir / "lv_area_over_time.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    outpaths.append(path1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for s, name in enumerate(segment_names):
        ax.plot(xs, binned_wt[s], marker="o", label=name)
    ax.axvline(ed_frame, linestyle="--", color="k", alpha=0.5)
    ax.set_title("AHA-aligned binned wall thickness over time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Thickness (pixels)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    path2 = outdir / "aha_binned_wt_over_time.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    outpaths.append(path2)

    fig, ax = plt.subplots(figsize=(10, 5))
    for s, name in enumerate(segment_names):
        ax.plot(xs, nwt[s], marker="o", label=name)
    ax.axvline(ed_frame, linestyle="--", color="k", alpha=0.5)
    ax.axhline(1.0, linestyle=":", color="gray")
    ax.set_title("AHA-aligned normalized wall thickness (NWT) over time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("NWT = WT(t) / mean ED epicardial radius")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    path3 = outdir / "aha_nwt_over_time.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    outpaths.append(path3)

    return outpaths


def _draw_bullseye_panel(
    ax,
    chunk_features: dict[str, AHAChunkFeatureSet],
    *,
    values_by_chunk: dict[str, np.ndarray],
    title: str,
    cmap_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> ScalarMappable:
    all_values = np.concatenate([
        values.ravel() for values in values_by_chunk.values() if values.size > 0
    ])
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    resolved_vmin = float(finite.min()) if vmin is None else float(vmin)
    resolved_vmax = float(finite.max()) if vmax is None else float(vmax)
    if np.isclose(resolved_vmin, resolved_vmax):
        resolved_vmax = resolved_vmin + 1.0

    norm = Normalize(vmin=resolved_vmin, vmax=resolved_vmax)
    cmap = plt.get_cmap(cmap_name)

    for chunk_name in CHUNK_ORDER:
        if chunk_name not in chunk_features or chunk_name not in values_by_chunk:
            continue

        values = values_by_chunk[chunk_name]
        ring_spec = BULLSEYE_RING_SPECS[chunk_name]
        segment_numbers = get_segment_numbers(chunk_name)
        n_segments = len(values)

        for index, value in enumerate(values):
            theta1, theta2, theta_mid_deg = bullseye_wedge_angles(index, n_segments)
            wedge = Wedge(
                (0.0, 0.0),
                ring_spec["r_outer"],
                theta1,
                theta2,
                width=ring_spec["r_outer"] - ring_spec["r_inner"],
                facecolor=cmap(norm(value)),
                edgecolor="white",
                linewidth=2.0,
            )
            ax.add_patch(wedge)

            theta_mid = np.deg2rad(theta_mid_deg)
            radius_mid = (ring_spec["r_inner"] + ring_spec["r_outer"]) / 2.0
            x = radius_mid * np.cos(theta_mid)
            y = radius_mid * np.sin(theta_mid)
            segment_number = segment_numbers[index]
            ax.text(
                x,
                y,
                f"{segment_number}\n{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    ax.set_aspect("equal")
    ax.set_xlim(-3.15, 3.15)
    ax.set_ylim(-3.15, 3.15)
    ax.axis("off")
    ax.set_title(title)

    return ScalarMappable(norm=norm, cmap=cmap)


def _write_chunk_summary(ax, stack_features: AHAStackFeatureSet) -> None:
    ax.axis("off")

    lines = [
        f"Global ED frame: {stack_features.global_ed_frame}",
        f"Overlay frame: {stack_features.overlay_frame}",
        "",
    ]

    for chunk_name in CHUNK_ORDER:
        chunk_feature = stack_features.chunk_features.get(chunk_name)
        if chunk_feature is None:
            continue

        slice_label = ", ".join(str(index) for index in chunk_feature.slice_indices)
        lines.extend([
            f"{chunk_name.title()} slices: ({slice_label})",
            f"Weighted mean ED WT: {chunk_feature.weighted_mean_ed_wt:.2f} px",
            f"Weighted mean peak NWT: {chunk_feature.weighted_mean_peak_nwt:.2f}",
            "",
        ])

    ax.text(
        0.0,
        1.0,
        "\n".join(lines).rstrip(),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )


def save_stack_bullseye_summary(
    stack_features: AHAStackFeatureSet,
    outdir: str | Path,
    *,
    patient_id: str,
    ed_frame: int | None = None,
) -> Path:
    """
    Save a multi-slice bullseye summary using chunk-aggregated AHA features.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    resolved_ed_frame = stack_features.global_ed_frame if ed_frame is None else int(ed_frame)
    ed_values = {
        chunk_name: chunk_feature.aggregated_wt[:, resolved_ed_frame]
        for chunk_name, chunk_feature in stack_features.chunk_features.items()
    }
    peak_values = {
        chunk_name: _safe_peak_metrics(chunk_feature.aggregated_nwt)[0]
        for chunk_name, chunk_feature in stack_features.chunk_features.items()
    }

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9])
    ax_ed = fig.add_subplot(gs[0, 0])
    ax_peak = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])

    sm_ed = _draw_bullseye_panel(
        ax_ed,
        stack_features.chunk_features,
        values_by_chunk=ed_values,
        title="ED Wall Thickness",
        cmap_name="coolwarm",
    )
    sm_peak = _draw_bullseye_panel(
        ax_peak,
        stack_features.chunk_features,
        values_by_chunk=peak_values,
        title="Peak Normalized Wall Thickness",
        cmap_name="coolwarm",
    )
    _write_chunk_summary(ax_text, stack_features)

    fig.colorbar(sm_ed, ax=ax_ed, fraction=0.046, pad=0.04, label="Thickness (px)")
    fig.colorbar(sm_peak, ax=ax_peak, fraction=0.046, pad=0.04, label="Peak NWT")
    fig.suptitle(f"{patient_id} selected-slice AHA summary")
    fig.tight_layout()

    outpath = outdir / "aha_bullseye_summary.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return outpath
