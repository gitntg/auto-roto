"""
Depth Expansion Configuration
=============================

Configuration for depth-guided mask expansion in the cinema preset workflow.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DepthExpansionConfig:
    """
    Configuration for depth-guided mask expansion.

    The depth expansion algorithm:
        1. Analyzes depth values within the initial SAM mask
        2. Computes foreground depth range (percentile-based)
        3. Expands mask to include nearby pixels within that depth range
        4. Optionally requires connected components to avoid distant regions

    This is a key feature of the cinema preset workflow, enabling
    complete subject coverage by capturing all pixels at similar depth.
    """

    # Enable/disable expansion
    enabled: bool = False

    # Depth analysis
    depth_tolerance: float = 0.1          # How much depth variation to include (0-1)
    percentile_range: Tuple[float, float] = (10, 90)  # Foreground depth percentiles

    # Expansion behavior
    require_connectivity: bool = True     # Only expand to connected regions
    max_expansion_px: int = 50            # Maximum expansion distance from original mask

    # Edge refinement
    edge_aware: bool = True               # Use RGB edges to limit expansion
    edge_threshold: float = 0.1           # Edge detection threshold (Canny)

    # Morphological cleanup
    cleanup_iterations: int = 2           # Morphological open/close iterations
    min_region_size: int = 100            # Minimum region size to keep (pixels)

    # Output
    save_expansion_mask: bool = True      # Save intermediate expansion result
    save_depth_profile: bool = False      # Save depth profile analysis

    def validate(self) -> None:
        """Validate configuration values."""
        if self.depth_tolerance < 0 or self.depth_tolerance > 1:
            raise ValueError("depth_tolerance must be between 0 and 1")

        if len(self.percentile_range) != 2:
            raise ValueError("percentile_range must be a tuple of (low, high)")

        if self.percentile_range[0] >= self.percentile_range[1]:
            raise ValueError("percentile_range[0] must be less than percentile_range[1]")

        if self.max_expansion_px < 0:
            raise ValueError("max_expansion_px must be non-negative")
