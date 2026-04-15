from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from cardionet.config import (
    load_cardionet_config,
    resolve_output_basename,
    resolve_dataset_root,
    resolve_patient_ids,
    resolve_runtime_device_and_dtype,
    resolve_script_output_dir,
)
from cardionet.segmentation.inference import infer_cine_frames, subsample_time_axis
from cardionet.segmentation.io import load_cine_nifti, save_inference_arrays
from cardionet.segmentation.model_loader import load_convunetr_from_local
from cardionet.segmentation.model_paths import resolve_finetuned_model_paths_from_config
from cardionet.segmentation.transforms import build_sax_inference_transform
from cardionet.visualization.segmentation_qc import (
    plot_segmentations_per_slice,
    plot_volume_changes,
)

SCRIPT_NAME = "infer_acdc_cine"


def get_output_paths(
    config: DictConfig,
    *,
    patient_id: str,
    view: str,
) -> tuple[Path, str]:
    """Resolve patient output directory and basename from config."""
    output_dir = resolve_script_output_dir(
        config,
        script_name=SCRIPT_NAME,
        patient_id=patient_id,
        view=view,
    )
    basename = resolve_output_basename(config, patient_id=patient_id, view=view)
    return output_dir, basename


def get_cine_filename(config: DictConfig, *, dataset_name: str, patient_id: str) -> str:
    """Resolve the cine filename for a dataset/patient pair."""
    dataset_cfg = config.datasets[dataset_name]
    return str(dataset_cfg.file_patterns.cine).format(patient_id=patient_id)


def get_qc_label_colors(config: DictConfig) -> dict[int, np.ndarray]:
    """Resolve QC overlay colors using the canonical CardioNet label mapping."""
    colors_cfg = config.qc.segmentation_overlays.colors_rgba
    return {
        int(config.conventions.labels.rv): np.array(colors_cfg.rv, dtype=np.float32),
        int(config.conventions.labels.myocardium): np.array(
            colors_cfg.myocardium,
            dtype=np.float32,
        ),
        int(config.conventions.labels.lv): np.array(colors_cfg.lv, dtype=np.float32),
    }


def process_patient(
    *,
    config: DictConfig,
    patient_id: str,
    dataset_name: str,
    dataset_root: Path,
    model,
    transform,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Run configured framewise inference and optional QC for one patient."""
    view = str(config.segmentation.model.view)
    inference_cfg = config.segmentation.inference
    qc_cfg = config.qc
    dev_cfg = config.dev
    t_step = int(inference_cfg.temporal_subsampling_step)

    input_nifti_path = dataset_root / patient_id / get_cine_filename(
        config,
        dataset_name=dataset_name,
        patient_id=patient_id,
    )

    if not input_nifti_path.exists():
        if bool(inference_cfg.validation.raise_on_missing_input):
            raise FileNotFoundError(f"Input cine file not found: {input_nifti_path}")

        print(f"[SKIP] {patient_id} not found.")
        return

    output_dir, output_basename = get_output_paths(
        config,
        patient_id=patient_id,
        view=view,
    )

    print("\n" + "=" * 60)
    print(f"Processing {patient_id}")
    print("=" * 60)

    images = load_cine_nifti(input_nifti_path)
    images = subsample_time_axis(images, t_step=t_step)

    if bool(dev_cfg.print_shapes):
        print(f"Input shape for inference: {images.shape}")

    labels = infer_cine_frames(
        model,
        images,
        transform=transform,
        view=str(inference_cfg.input_key),
        device=device,
        dtype=dtype,
        show_progress=bool(config.logging.tqdm_enabled),
    )

    if bool(inference_cfg.validation.enforce_output_shape_match) and labels.shape != images.shape:
        raise RuntimeError(
            f"Predicted labels shape {labels.shape} does not match input shape {images.shape}"
        )

    if bool(dev_cfg.print_shapes):
        print("Predicted label volume shape:", labels.shape)

    if bool(dev_cfg.print_unique_labels):
        print("Unique labels in prediction:", np.unique(labels))

    images_path, labels_path = save_inference_arrays(
        images=images,
        labels=labels,
        output_dir=output_dir,
        basename=output_basename,
        save_inputs=bool(inference_cfg.save_input_arrays and config.outputs.save_inputs),
        save_predictions=bool(
            inference_cfg.save_prediction_arrays and config.outputs.save_predictions
        ),
        image_suffix=str(config.conventions.file_naming.saved_images_suffix),
        labels_suffix=str(config.conventions.file_naming.predicted_labels_suffix),
    )

    if images_path is not None:
        print("Saved images:", images_path)
    if labels_path is not None:
        print("Saved labels:", labels_path)

    save_qc = bool(config.outputs.save_qc and qc_cfg.enabled)
    script_cfg = config.scripts[SCRIPT_NAME]

    if save_qc and bool(script_cfg.save_gif) and bool(qc_cfg.segmentation_overlays.enabled):
        gif_paths = plot_segmentations_per_slice(
            images=images,
            labels=labels,
            t_step=t_step,
            output_dir=output_dir,
            basename=output_basename,
            figure_size=tuple(qc_cfg.segmentation_overlays.figure_size),
            dpi=int(qc_cfg.segmentation_overlays.dpi),
            cmap=str(qc_cfg.segmentation_overlays.cmap),
            gif_loop=int(qc_cfg.segmentation_overlays.gif_loop),
            gif_duration_base_ms=int(qc_cfg.segmentation_overlays.gif_duration_base_ms),
            label_colors=get_qc_label_colors(config),
        )
        print(f"Saved {len(gif_paths)} slice GIF(s)")

    if save_qc and bool(script_cfg.save_volume_plot) and bool(qc_cfg.volume_plot.enabled):
        volume_plot_path = output_dir / (
            f"{output_basename}{config.conventions.file_naming.volume_plot_suffix}"
        )
        plot_volume_changes(
            labels=labels,
            t_step=t_step,
            filepath=volume_plot_path,
            voxel_volume_ml=float(qc_cfg.volume_plot.voxel_volume_ml),
            figsize=tuple(qc_cfg.volume_plot.figsize),
            dpi_screen=int(qc_cfg.volume_plot.dpi_screen),
            dpi_save=int(qc_cfg.volume_plot.dpi_save),
            ylabel=str(qc_cfg.volume_plot.ylabel),
        )
        print("Saved volume plot:", volume_plot_path)


def main() -> None:
    config = load_cardionet_config()
    script_cfg = config.scripts[SCRIPT_NAME]
    dataset_name = str(script_cfg.dataset_name)
    split = str(script_cfg.split)

    device, dtype = resolve_runtime_device_and_dtype(config)
    print("Using device:", device)
    print("Using dtype:", dtype)

    weights_path, config_path = resolve_finetuned_model_paths_from_config(config)

    model = load_convunetr_from_local(
        weights_path=weights_path,
        config_path=config_path,
        device=device,
        eval_mode=bool(config.segmentation.model.eval_mode),
    )

    transform = build_sax_inference_transform(
        view=str(config.segmentation.inference.input_key),
        scale_intensity=bool(config.segmentation.inference.preprocessing.scale_intensity),
        pad_spatial=bool(config.segmentation.inference.preprocessing.pad_spatial),
        spatial_size=tuple(config.segmentation.inference.preprocessing.pad_spatial_size),
        pad_method=str(config.segmentation.inference.preprocessing.pad_method),
    )

    dataset_root = resolve_dataset_root(
        config,
        dataset_name=dataset_name,
        split=split,
    )
    patient_ids = resolve_patient_ids(
        config,
        dataset_root=dataset_root,
        script_name=SCRIPT_NAME,
        dataset_name=dataset_name,
    )

    for patient_id in patient_ids:
        try:
            process_patient(
                config=config,
                patient_id=patient_id,
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                model=model,
                transform=transform,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            print(f"[ERROR] {patient_id}: {exc}")
            continue


if __name__ == "__main__":
    main()
