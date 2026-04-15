import numpy as np
import torch

from cardionet.segmentation.inference import infer_cine_frames, subsample_time_axis


class IdentityTransform:
    def __call__(self, batch):
        return batch


class DummyModel:
    def __call__(self, batch):
        x = batch["sax"]  # expected shape (1, 1, x, y, z)
        _, _, sx, sy, sz = x.shape

        logits = torch.zeros((1, 4, sx, sy, sz), dtype=torch.float32, device=x.device)
        logits[:, 3, ...] = 1.0
        return {"sax": logits}


def test_subsample_time_axis_step_1_returns_same_shape():
    images = np.zeros((192, 192, 10, 30), dtype=np.float32)
    out = subsample_time_axis(images, t_step=1)
    assert out.shape == images.shape


def test_subsample_time_axis_step_2_halves_time_approximately():
    images = np.zeros((192, 192, 10, 30), dtype=np.float32)
    out = subsample_time_axis(images, t_step=2)
    assert out.shape == (192, 192, 10, 15)


def test_infer_cine_frames_returns_xyz_t_uint8():
    model = DummyModel()
    transform = IdentityTransform()
    images = np.zeros((32, 32, 5, 4), dtype=np.float32)

    labels = infer_cine_frames(
        model=model,
        images=images,
        transform=transform,
        view="sax",
        device=torch.device("cpu"),
        dtype=torch.float32,
        show_progress=False,
    )

    assert labels.shape == (32, 32, 5, 4)
    assert labels.dtype == np.uint8
    assert np.all(labels == 3)


def test_infer_cine_frames_raises_on_non_4d_input():
    model = DummyModel()
    transform = IdentityTransform()
    images = np.zeros((32, 32, 5), dtype=np.float32)

    try:
        infer_cine_frames(
            model=model,
            images=images,
            transform=transform,
            view="sax",
            device=torch.device("cpu"),
            dtype=torch.float32,
            show_progress=False,
        )
    except ValueError as exc:
        assert "Expected images with shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-4D input")