"""Segmentation mask placeholders for cardiac MRI analysis."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SegmentationLabels:
    """Canonical CardioNet label ids aligned with CineMA segmentation outputs."""

    background: int = 0
    right_ventricle: int = 1
    myocardium: int = 2
    left_ventricle: int = 3


@dataclass(slots=True)
class SegmentationMask:
    """Container describing a predicted or annotated segmentation mask."""

    shape: tuple[int, ...]
    labels: SegmentationLabels = field(default_factory=SegmentationLabels)
    spacing_mm: tuple[float, ...] | None = None

    def ndim(self) -> int:
        """Return the mask dimensionality."""
        return len(self.shape)
