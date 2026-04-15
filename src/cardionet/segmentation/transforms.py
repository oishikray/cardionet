from __future__ import annotations

from monai.transforms import Compose, ScaleIntensityd, SpatialPadd


def build_sax_inference_transform(
    *,
    view: str = "sax",
    scale_intensity: bool = True,
    pad_spatial: bool = True,
    spatial_size: tuple[int, int, int] = (192, 192, 16),
    pad_method: str = "end",
) -> Compose:
    """
    Build the MONAI transform used for SAX segmentation inference.

    Parameters
    ----------
    spatial_size
        Spatial padding target applied to the input frame volume.

    Returns
    -------
    Compose
        MONAI transform pipeline.
    """
    transforms = []

    if scale_intensity:
        transforms.append(ScaleIntensityd(keys=view))

    if pad_spatial:
        transforms.append(
            SpatialPadd(keys=view, spatial_size=spatial_size, method=pad_method)
        )

    return Compose(transforms)
