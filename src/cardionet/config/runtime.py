from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "cardionet_config.yaml"

_DTYPE_LOOKUP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def load_cardionet_config(config_path: str | Path | None = None) -> DictConfig:
    """Load the repo-level CardioNet runtime configuration."""
    resolved_path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"CardioNet config not found: {resolved_path}")

    return OmegaConf.load(resolved_path)


def resolve_runtime_device_and_dtype(
    config: DictConfig,
) -> tuple[torch.device, torch.dtype]:
    """Resolve runtime device and tensor dtype from config."""
    device_cfg = config.runtime.device
    precision_cfg = config.runtime.precision

    preferred = str(device_cfg.preferred).lower()
    fallback_to_cpu = bool(device_cfg.fallback_to_cpu)

    if preferred == "cpu":
        device = torch.device("cpu")
    elif preferred == "cuda":
        explicit_index = int(device_cfg.explicit_index)
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{explicit_index}")
        elif fallback_to_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError("CUDA requested but not available.")
    elif preferred == "auto":
        if torch.cuda.is_available():
            explicit_index = int(device_cfg.explicit_index)
            device = torch.device(f"cuda:{explicit_index}")
        else:
            device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported runtime.device.preferred value: {preferred}")

    if device.type == "cuda":
        prefer_bfloat16 = bool(precision_cfg.prefer_bfloat16_if_available)
        if prefer_bfloat16 and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16

        dtype_name = str(precision_cfg.default_cuda_dtype).lower()
    else:
        dtype_name = str(precision_cfg.default_cpu_dtype).lower()

    if dtype_name not in _DTYPE_LOOKUP:
        raise ValueError(f"Unsupported torch dtype in config: {dtype_name}")

    return device, _DTYPE_LOOKUP[dtype_name]


def resolve_dataset_root(
    config: DictConfig,
    dataset_name: str | None = None,
    split: str | None = None,
) -> Path:
    """Resolve the processed dataset split directory for the requested dataset."""
    resolved_dataset = str(dataset_name or config.datasets.active_dataset)
    dataset_cfg = config.datasets[resolved_dataset]
    resolved_split = str(split or dataset_cfg.split)
    return Path(dataset_cfg.processed_root) / resolved_split


def resolve_patient_ids(
    config: DictConfig,
    *,
    dataset_root: str | Path,
    script_name: str,
    dataset_name: str | None = None,
) -> list[str]:
    """Resolve patient IDs from script overrides or dataset defaults."""
    resolved_dataset = str(dataset_name or config.datasets.active_dataset)
    dataset_cfg = config.datasets[resolved_dataset]
    patient_cfg = dataset_cfg.patients
    script_cfg = config.scripts[script_name]

    dataset_root = Path(dataset_root)
    prefix = str(patient_cfg.prefix)
    zero_pad = int(patient_cfg.zero_pad)

    override_ids = [str(pid) for pid in script_cfg.override_patient_ids]
    if override_ids:
        return override_ids

    override_start = script_cfg.override_patient_start
    override_end = script_cfg.override_patient_end
    if override_start is not None or override_end is not None:
        if override_start is None or override_end is None:
            raise ValueError(
                f"{script_name} must set both override_patient_start and "
                "override_patient_end when using range overrides."
            )

        return [
            f"{prefix}{patient_idx:0{zero_pad}d}"
            for patient_idx in range(int(override_start), int(override_end) + 1)
        ]

    mode = str(patient_cfg.mode).lower()
    if mode == "range":
        start = int(patient_cfg.start)
        end = int(patient_cfg.end)
        return [
            f"{prefix}{patient_idx:0{zero_pad}d}"
            for patient_idx in range(start, end + 1)
        ]

    if mode == "explicit_list":
        return [str(pid) for pid in patient_cfg.explicit_ids]

    if mode == "all":
        return sorted(
            entry.name
            for entry in dataset_root.iterdir()
            if entry.is_dir() and entry.name.startswith(prefix)
        )

    raise ValueError(f"Unsupported patient selection mode: {patient_cfg.mode}")


def resolve_output_basename(
    config: DictConfig,
    *,
    patient_id: str,
    view: str,
) -> str:
    """Resolve the configured output basename for a patient/view pair."""
    return str(config.outputs.naming.basename_template).format(
        patient_id=patient_id,
        view=view,
    )


def resolve_script_output_dir(
    config: DictConfig,
    *,
    script_name: str,
    patient_id: str,
    view: str,
    output_root: str | Path | None = None,
    output_dir_template: str | None = None,
) -> Path:
    """Resolve a script-owned patient output directory."""
    base_root = (
        Path(output_root)
        if output_root is not None
        else Path(config.scripts[script_name].output_root)
    )
    dir_template = (
        output_dir_template
        if output_dir_template is not None
        else str(config.outputs.naming.patient_output_dir_template)
    )
    return base_root / dir_template.format(patient_id=patient_id, view=view)


def resolve_prediction_labels_path(
    config: DictConfig,
    *,
    patient_id: str,
    view: str,
    source_script_name: str = "infer_acdc_cine",
) -> Path:
    """Resolve the configured path to saved segmentation labels."""
    output_dir = resolve_script_output_dir(
        config,
        script_name=source_script_name,
        patient_id=patient_id,
        view=view,
    )
    basename = resolve_output_basename(config, patient_id=patient_id, view=view)
    return output_dir / (
        f"{basename}{config.conventions.file_naming.predicted_labels_suffix}"
    )
