"""
Matte Configuration
===================

Configuration for matte combination and processing.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatteCombineConfig:
    """
    Configuration for matte combination.

    Attributes:
        core_erosion: Erosion pixels for core matte (cleaner edges)
        despill_strength: Despill strength (0-1, 0=disabled)
        despill_color: Spill color to remove ("green", "blue", "auto")
        preview_checker_size: Checkerboard square size for preview
    """
    core_erosion: int = 3
    despill_strength: float = 0.5
    despill_color: str = "auto"
    preview_checker_size: int = 16


@dataclass
class MatteBlendConfig:
    """
    Configuration for matte blending.

    Attributes:
        mode: Blend mode (maximum/minimum/average/weighted/multiply/screen)
        weights: Weights for weighted blend (must match source count)
    """
    mode: str = "maximum"
    weights: Optional[list] = None
