from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open

from cinema.log import get_logger
from cinema.segmentation.convunetr import ConvUNetR, get_model

logger = get_logger(__name__)


def load_safetensors_state_dict(
    weights_path: str | Path,
) -> dict[str, torch.Tensor]:
    """
    Load a PyTorch state_dict from a .safetensors file.

    Parameters
    ----------
    weights_path
        Path to the .safetensors checkpoint.
    Returns
    -------
    dict[str, torch.Tensor]
        State dictionary suitable for model.load_state_dict(...).
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    logger.info("Loading model weights from %s", weights_path)

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    return state_dict


def load_model_config(config_path: str | Path) -> DictConfig:
    """
    Load the model configuration consumed by CineMA's get_model(...).

    Parameters
    ----------
    config_path
        Path to the YAML config file.

    Returns
    -------
    DictConfig
        OmegaConf config object compatible with get_model(...).
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    logger.info("Loading model config from %s", config_path)
    return OmegaConf.load(config_path)


def build_convunetr_from_config(config: DictConfig) -> ConvUNetR:
    """
    Build a ConvUNetR model from a CineMA config.
    """
    model = get_model(config)

    if not isinstance(model, ConvUNetR):
        raise TypeError(
            "Expected get_model(config) to return ConvUNetR, "
            f"got {type(model).__name__}."
        )

    return model


def resolve_inference_device(device: str | torch.device | None = None) -> torch.device:
    """
    Resolve the target device for inference.

    If no device is provided, prefer CUDA when available.
    """
    if device is None:
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
            logger.info("CUDA is available; using %s for inference", resolved_device)
            return resolved_device

        resolved_device = torch.device("cpu")
        logger.info("CUDA is not available; using %s for inference", resolved_device)
        return resolved_device

    return torch.device(device)


def load_convunetr_from_local(
    weights_path: str | Path,
    config_path: str | Path,
    *,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
) -> ConvUNetR:
    """
    Build a ConvUNetR from a local config and local safetensors weights.

    This is the clean replacement for the old monkey-patched
    ConvUNetR.from_local(...).

    Parameters
    ----------
    weights_path
        Path to local .safetensors model weights.
    config_path
        Path to local YAML config.
    device
        Target device for the instantiated model. If None, prefers CUDA when
        available and otherwise uses CPU.
    eval_mode
        If True, calls model.eval() before returning.

    Returns
    -------
    ConvUNetR
        Loaded model ready for inference.
    """
    target_device = resolve_inference_device(device)
    state_dict = load_safetensors_state_dict(weights_path)
    config = load_model_config(config_path)
    model = build_convunetr_from_config(config)

    logger.info("Loading state_dict into model with strict checkpoint validation")
    model.load_state_dict(state_dict)
    model = model.to(target_device)

    if eval_mode:
        model.eval()

    logger.info("Loaded local ConvUNetR weights successfully on %s", target_device)
    return model
