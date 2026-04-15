"""Contour and landmark placeholders for cardiac MRI geometry."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class Landmark:
    """Named 2D landmark in image coordinates."""

    name: str
    x: float
    y: float


@dataclass(slots=True)
class CardiacContour:
    """Simple container for an ordered contour."""

    label: str
    points: list[tuple[float, float]] = field(default_factory=list)

    def num_points(self) -> int:
        """Return the number of sampled contour points."""
        return len(self.points)
