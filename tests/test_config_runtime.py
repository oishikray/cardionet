from pathlib import Path

import torch
from omegaconf import OmegaConf

from cardionet.config import resolve_dataset_root, resolve_patient_ids
from cardionet.config import resolve_prediction_labels_path, resolve_script_output_dir
from cardionet.config import resolve_runtime_device_and_dtype


def _make_base_config(tmp_path: Path):
    dataset_root = tmp_path / "acdc" / "train"
    dataset_root.mkdir(parents=True)
    (dataset_root / "patient001").mkdir()
    (dataset_root / "patient002").mkdir()

    return OmegaConf.create(
        {
            "runtime": {
                "device": {
                    "preferred": "cpu",
                    "explicit_index": 0,
                    "fallback_to_cpu": True,
                },
                "precision": {
                    "prefer_bfloat16_if_available": True,
                    "default_cpu_dtype": "float32",
                    "default_cuda_dtype": "float32",
                },
            },
            "datasets": {
                "active_dataset": "acdc",
                "acdc": {
                    "processed_root": str(tmp_path / "acdc"),
                    "split": "train",
                    "patients": {
                        "mode": "range",
                        "start": 1,
                        "end": 2,
                        "explicit_ids": [],
                        "zero_pad": 3,
                        "prefix": "patient",
                    },
                },
            },
            "scripts": {
                "infer_acdc_cine": {
                    "override_patient_start": None,
                    "override_patient_end": None,
                    "override_patient_ids": [],
                    "output_root": str(tmp_path / "outputs" / "segmentation"),
                },
                "extract_aha_wt_nwt": {
                    "output_root": str(tmp_path / "outputs" / "features"),
                }
            },
            "outputs": {
                "naming": {
                    "patient_output_dir_template": "{patient_id}_inference",
                    "basename_template": "{patient_id}_{view}_t",
                }
            },
            "conventions": {
                "file_naming": {
                    "predicted_labels_suffix": "_pred_labels.npy",
                }
            },
        }
    )


def test_resolve_runtime_device_and_dtype_cpu(tmp_path: Path):
    config = _make_base_config(tmp_path)

    device, dtype = resolve_runtime_device_and_dtype(config)

    assert device == torch.device("cpu")
    assert dtype == torch.float32


def test_resolve_dataset_root_uses_dataset_split(tmp_path: Path):
    config = _make_base_config(tmp_path)

    dataset_root = resolve_dataset_root(config, dataset_name="acdc", split="train")

    assert dataset_root == tmp_path / "acdc" / "train"


def test_resolve_patient_ids_uses_dataset_range(tmp_path: Path):
    config = _make_base_config(tmp_path)
    dataset_root = resolve_dataset_root(config, dataset_name="acdc", split="train")

    patient_ids = resolve_patient_ids(
        config,
        dataset_root=dataset_root,
        script_name="infer_acdc_cine",
        dataset_name="acdc",
    )

    assert patient_ids == ["patient001", "patient002"]


def test_resolve_patient_ids_prefers_script_overrides(tmp_path: Path):
    config = _make_base_config(tmp_path)
    config.scripts.infer_acdc_cine.override_patient_ids = ["patient099"]
    dataset_root = resolve_dataset_root(config, dataset_name="acdc", split="train")

    patient_ids = resolve_patient_ids(
        config,
        dataset_root=dataset_root,
        script_name="infer_acdc_cine",
        dataset_name="acdc",
    )

    assert patient_ids == ["patient099"]


def test_resolve_script_output_dir_uses_shared_templates(tmp_path: Path):
    config = _make_base_config(tmp_path)

    output_dir = resolve_script_output_dir(
        config,
        script_name="infer_acdc_cine",
        patient_id="patient001",
        view="sax",
    )

    assert output_dir == tmp_path / "outputs" / "segmentation" / "patient001_inference"


def test_resolve_prediction_labels_path_matches_inference_layout(tmp_path: Path):
    config = _make_base_config(tmp_path)

    labels_path = resolve_prediction_labels_path(
        config,
        patient_id="patient001",
        view="sax",
        source_script_name="infer_acdc_cine",
    )

    assert labels_path == (
        tmp_path
        / "outputs"
        / "segmentation"
        / "patient001_inference"
        / "patient001_sax_t_pred_labels.npy"
    )
