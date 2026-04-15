from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def subsample_time_axis(images: np.ndarray, t_step: int) -> np.ndarray:
    """
    Uniformly subsample the temporal axis of a cine stack.

    Parameters
    ----------
    images
        Cine array shaped (x, y, z, t).
    t_step
        Temporal step size. Must be >= 1.

    Returns
    -------
    np.ndarray
        Subsampled cine array.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (x, y, z, t), got {images.shape}")

    if t_step < 1:
        raise ValueError(f"t_step must be >= 1, got {t_step}")

    if t_step == 1:
        return images

    return images[..., ::t_step]


def infer_cine_frames(
    model,
    images: np.ndarray,
    *,
    transform,
    view: str = "sax",
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Run frame-wise segmentation inference on a 4D cine volume.

    Parameters
    ----------
    model
        Segmentation model. Expected to return a dict keyed by `view`.
    images
        Input array shaped (x, y, z, t).
    transform
        MONAI-style transform applied to each frame.
    view
        Input/output dict key used by the model.
    device
        Device on which inference will run.
    dtype
        Target input dtype.
    show_progress
        Whether to show a tqdm progress bar.

    Returns
    -------
    np.ndarray
        Predicted labels shaped (x, y, z, t), dtype uint8.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (x, y, z, t), got {images.shape}")

    x, y, z, t = images.shape
    labels_list: list[torch.Tensor] = []

    frame_iter = range(t)
    if show_progress:
        frame_iter = tqdm(frame_iter, total=t, desc="Inference frames")

    for frame_idx in frame_iter:
        frame_volume = images[None, ..., frame_idx]
        batch = transform({view: torch.from_numpy(frame_volume)})

        batch = {
            k: v[None, ...].to(device=device, dtype=dtype)
            for k, v in batch.items()
        }

        use_cuda_autocast = device.type == "cuda" and dtype in {
            torch.float16,
            torch.bfloat16,
        }

        with torch.no_grad(), torch.autocast(
            "cuda",
            dtype=dtype,
            enabled=use_cuda_autocast,
        ):
            logits = model(batch)[view]

        pred = torch.argmax(logits, dim=1)[0, ..., :z]
        labels_list.append(pred)

    labels = torch.stack(labels_list, dim=-1).detach().cpu().numpy().astype(np.uint8)

    expected_shape = (x, y, z, t)
    if labels.shape != expected_shape:
        raise RuntimeError(
            f"Inference output shape mismatch: expected {expected_shape}, got {labels.shape}"
        )

    return labels
