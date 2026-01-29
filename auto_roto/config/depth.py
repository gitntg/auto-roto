"""
Depth Refinement Configuration
==============================

Configuration for depth-guided alpha refinement.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DepthRefineConfig:
    """Configuration for depth-guided refinement."""

    # Input paths
    alpha_dir: str = ""           # Directory with alpha mattes
    frames_dir: str = ""          # Directory with RGB frames (optional)
    depth_dir: str = ""           # Pre-computed depth maps (optional)
    video_path: str = ""          # Video file (optional, for depth computation)

    # Output
    output_dir: str = "./refined"

    # Depth model settings
    depth_model: str = "large"    # large (DA3Mono) best for hair detail; nested smooths fine detail
    compute_depth: bool = True    # Compute depth if not provided
    save_depth: bool = True       # Save depth maps for reuse
    depth_only: bool = True       # ONLY output depth - no alpha refinement (pure depth mode)

    # DA3 Resolution Settings
    depth_process_res: Optional[int] = None   # None = auto (image size), or explicit value
    depth_process_method: str = "upper"       # "upper" or "lower" bound resize
    depth_norm_percentiles: Tuple[float, float] = (2.0, 98.0)  # Visualization percentiles
    use_depth_confidence: bool = False        # Extract confidence maps

    # Refinement settings
    edge_threshold: float = 0.1   # Depth gradient threshold for edges
    blend_strength: float = 0.7   # How much to blend depth-based refinement
    preserve_softness: bool = True # Preserve existing soft edges

    # Edge handling
    sharpen_hard_edges: bool = True   # Sharpen where depth changes abruptly
    soften_gradual_edges: bool = True # Soften where depth changes gradually

    # Hair detection settings
    hair_search_radius: int = 50      # How far outside mask to search for hair
    depth_tolerance: float = 0.15     # Depth similarity tolerance for hair
    min_hair_alpha: float = 0.3       # Minimum alpha value for detected hair

    # Output settings
    output_format: str = "exr"
    bit_depth: int = 16

    # Performance
    device: str = "cuda"
    batch_size: int = 4

    # Debug
    verbose: bool = False
    save_debug: bool = False      # Save intermediate visualizations

    def validate(self) -> None:
        """Validate configuration values."""
        if self.depth_model not in ("small", "base", "large", "nested-base", "nested-large"):
            raise ValueError("depth_model must be small, base, large, nested-base, or nested-large")

        if self.depth_process_method not in ("upper", "lower"):
            raise ValueError("depth_process_method must be 'upper' or 'lower'")

        if self.edge_threshold < 0 or self.edge_threshold > 1:
            raise ValueError("edge_threshold must be between 0 and 1")

        if self.blend_strength < 0 or self.blend_strength > 1:
            raise ValueError("blend_strength must be between 0 and 1")
