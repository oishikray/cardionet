from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


def resolve_finetuned_model_paths(
    model_root: str | Path,
    trained_dataset: str,
    view: str,
    seed: int,
    *,
    finetuned_dir_template: str = "finetuned/segmentation/{trained_dataset}_{view}",
    weights_template: str = "{trained_dataset}_{view}_{seed}.safetensors",
    config_filename: str = "config.yaml",
) -> tuple[Path, Path]:
    """
    Resolve local config and weights paths for a CineMA fine-tuned model.

    Expected layout
    ---------------
    {model_root}/{finetuned_dir_template.format(...)} /
        {config_filename}
        {weights_template.format(...)}
    """
    model_root = Path(model_root)
    base = model_root / finetuned_dir_template.format(
        trained_dataset=trained_dataset,
        view=view,
        seed=seed,
    )

    weights_path = base / weights_template.format(
        trained_dataset=trained_dataset,
        view=view,
        seed=seed,
    )
    config_path = base / config_filename

    return weights_path, config_path


def resolve_finetuned_model_paths_from_config(config: DictConfig) -> tuple[Path, Path]:
    """Resolve local fine-tuned model paths from the CardioNet config."""
    model_cfg = config.segmentation.model
    explicit_weights = str(model_cfg.explicit_weights_path).strip()
    explicit_config = str(model_cfg.explicit_config_path).strip()

    if explicit_weights or explicit_config:
        if not explicit_weights or not explicit_config:
            raise ValueError(
                "segmentation.model.explicit_weights_path and "
                "segmentation.model.explicit_config_path must be set together."
            )

        return Path(explicit_weights), Path(explicit_config)

    layout_cfg = model_cfg.local_layout
    return resolve_finetuned_model_paths(
        model_root=config.paths.model_root,
        trained_dataset=str(model_cfg.trained_dataset),
        view=str(model_cfg.view),
        seed=int(model_cfg.seed),
        finetuned_dir_template=str(layout_cfg.finetuned_dir_template),
        weights_template=str(layout_cfg.weights_template),
        config_filename=str(layout_cfg.config_filename),
    )
