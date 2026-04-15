from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cardionet.geometry.aha_reference import get_masks


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

    ax.set_title("AHA-aligned sector map with anatomy + segment metrics")
    ax.axis("off")

    structure_handles = [
        Patch(facecolor=anatomy_colors["RV"], edgecolor="black", label="RV"),
        Patch(facecolor=anatomy_colors["MYO"], edgecolor="black", label="MYO"),
        Patch(facecolor=anatomy_colors["LV"], edgecolor="black", label="LV"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="LV centroid"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="cyan", markersize=8, label="RV-contact centroid"),
    ]

    peak_nwt = np.nanmax(nwt, axis=1)
    peak_frame = np.nanargmax(nwt, axis=1)
    ed_wt = binned_wt[:, ed_frame]

    sector_handles = []
    for s, name in enumerate(segment_names):
        label = (
            f"{name}\n"
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
    ax.set_ylabel("NWT = WT(t) / WT(ED)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    path3 = outdir / "aha_nwt_over_time.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    outpaths.append(path3)

    return outpaths
