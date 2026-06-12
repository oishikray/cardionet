"""Visualization helpers for cardiac MRI analysis."""

from .canonical import (
    BULLSEYE_RING_SPECS,
    RING_ORDER,
    SEGMENT_BASE_COLORS,
    save_delta_t_plots,
    save_lv_volume_curve,
    save_mask_overlay_gifs,
    save_metric_bullseye,
    save_ring_time_series_plots,
    segment_base_name,
    segment_color,
)

from .aha_qc import (
    moving_average_nan,
    bullseye_wedge_angles,
    segment_numbers_for_names,
    save_debug_wt_plot,
    save_stack_bullseye_summary,
    save_structure_and_sector_overlay,
    save_time_series_plots,
)

from .ray_nwt_qc import (
    compute_fixed_center_ray_matrices,
    raw_ray_nwt_matrix,
    save_aha_boundary_contour_gif,
    save_ray_nwt_frame_series,
)

try:
    from .segmentation_qc import (
        LVSliceQualityResult,
        compute_ejection_fraction,
        compute_label_volumes_disk,
        compute_label_volumes_riemann,
        compute_lv_slice_quality_mask,
        plot_segmentations_per_slice,
        plot_volume_changes,
    )
except ModuleNotFoundError:
    LVSliceQualityResult = None
    compute_ejection_fraction = None
    compute_label_volumes_disk = None
    compute_label_volumes_riemann = None
    compute_lv_slice_quality_mask = None
    plot_segmentations_per_slice = None
    plot_volume_changes = None

__all__ = [
    "BULLSEYE_RING_SPECS",
    "RING_ORDER",
    "SEGMENT_BASE_COLORS",
    "save_delta_t_plots",
    "save_lv_volume_curve",
    "save_mask_overlay_gifs",
    "save_metric_bullseye",
    "save_ring_time_series_plots",
    "segment_base_name",
    "segment_color",
    "LVSliceQualityResult",
    "compute_ejection_fraction",
    "compute_label_volumes_disk",
    "compute_label_volumes_riemann",
    "compute_lv_slice_quality_mask",
    "moving_average_nan",
    "bullseye_wedge_angles",
    "segment_numbers_for_names",
    "plot_segmentations_per_slice",
    "plot_volume_changes",
    "save_debug_wt_plot",
    "save_stack_bullseye_summary",
    "save_structure_and_sector_overlay",
    "save_time_series_plots",
    "compute_fixed_center_ray_matrices",
    "raw_ray_nwt_matrix",
    "save_aha_boundary_contour_gif",
    "save_ray_nwt_frame_series",
]
