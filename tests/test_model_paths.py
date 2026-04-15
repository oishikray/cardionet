from pathlib import Path

from omegaconf import OmegaConf

from cardionet.segmentation.model_paths import resolve_finetuned_model_paths
from cardionet.segmentation.model_paths import resolve_finetuned_model_paths_from_config


def test_resolve_finetuned_model_paths():
    model_root = Path("/tmp/models")
    weights_path, config_path = resolve_finetuned_model_paths(
        model_root=model_root,
        trained_dataset="acdc",
        view="sax",
        seed=0,
    )

    assert weights_path == model_root / "finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors"
    assert config_path == model_root / "finetuned/segmentation/acdc_sax/config.yaml"


def test_resolve_finetuned_model_paths_from_config_uses_templates():
    config = OmegaConf.create(
        {
            "paths": {"model_root": "D:/models"},
            "segmentation": {
                "model": {
                    "trained_dataset": "acdc",
                    "view": "sax",
                    "seed": 7,
                    "explicit_weights_path": "",
                    "explicit_config_path": "",
                    "local_layout": {
                        "finetuned_dir_template": "finetuned/segmentation/{trained_dataset}_{view}",
                        "weights_template": "{trained_dataset}_{view}_{seed}.safetensors",
                        "config_filename": "config.yaml",
                    },
                }
            },
        }
    )

    weights_path, config_path = resolve_finetuned_model_paths_from_config(config)

    assert weights_path == Path(
        "D:/models/finetuned/segmentation/acdc_sax/acdc_sax_7.safetensors"
    )
    assert config_path == Path("D:/models/finetuned/segmentation/acdc_sax/config.yaml")
