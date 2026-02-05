"""
ViTMatte Configuration
======================

Configuration for ViTMatte-based alpha refinement and trimap synthesis.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TrimapConfig:
    """
    Configuration for trimap synthesis.

    DEFAULT: Adaptive Mode (Formulaic Distance Approach)
    - Unknown width varies based on local depth complexity
    - Smooth regions: ~2px, Complex regions (hair): up to 60px
    - Optional motion-aware expansion using optical flow
    """

    # Core foreground (eroded SAM2 mask)
    core_erosion: int = 5           # Pixels to erode for definite foreground

    # Legacy mode: fixed dilation (used when adaptive_mode=False)
    unknown_dilation: int = 25      # Fixed dilation for legacy fallback

    # Depth high-pass for geometric strands
    highpass_threshold: float = 0.01  # Depth gradient threshold for "spikes"
    min_unknown_depth: float = 0.2    # Legacy: minimum depth for unknown zone

    # Depth confidence intervals (SOURCEOFTRUTH)
    depth_core_threshold: float = 0.8   # Depth above this = definite foreground
    depth_bg_threshold: float = 0.2     # Depth below this = definite background
    depth_unknown_low: float = 0.4      # Depth variance zone lower bound
    depth_unknown_high: float = 0.6     # Depth variance zone upper bound

    # Hair region focus
    hair_region_ratio: float = 0.5  # Top portion of bounding box (hair region)
    hair_dilation_boost: int = 15   # Extra dilation in hair region

    # Geometric continuity
    require_connectivity: bool = False
    min_strand_size: int = 20

    # Soft trimap options
    soft_unknown: bool = False      # SOURCEOFTRUTH uses hard 128 for unknown
    unknown_softness: float = 0.5

    # Simple mode: just dilate the edge without depth analysis
    simple_edge_mode: bool = False
    simple_edge_width: int = 30

    # Linear color space processing
    use_linear_colorspace: bool = True

    # === ADAPTIVE MODE (Formulaic Distance Approach) ===
    adaptive_mode: bool = True          # Adaptive trimap is now the default
    adaptive_blur_kernel: int = 21      # Gaussian blur for complexity field smoothing

    # Formulaic EROSION (Smart Core)
    erosion_base_px: float = 2.0        # Minimum erosion (smooth regions)
    erosion_max_px: float = 40.0        # Additional erosion for complex regions

    # Formulaic DILATION (Dynamic Reach)
    adaptive_base_px: float = 2.0       # Minimum reach (smooth regions)
    adaptive_max_px: float = 60.0       # Maximum additional reach (complex regions)

    # RGB Guidance
    rgb_complexity_weight: float = 1.0  # Influence of RGB texture on complexity (0-1)

    # === MOTION-AWARE MODE (Optical Flow Weighting) ===
    motion_aware: bool = False          # Enable motion-weighted trimap
    motion_max_speed: float = 30.0      # Max pixel motion to consider
    motion_blur_kernel: int = 15        # Gaussian blur for motion field
    motion_weight: float = 40.0         # Motion multiplier for unknown zone expansion


@dataclass
class ViTMatteConfig:
    """Configuration for ViTMatte model."""

    model_size: str = "base"        # "small" or "base"
    device: str = "cuda"
    max_resolution: int = 2048      # Max dimension for processing
    use_fp16: bool = True           # Use half precision for lower VRAM


@dataclass
class GeometricMatteConfig:
    """Full configuration for geometric matte refinement."""

    trimap: TrimapConfig = field(default_factory=TrimapConfig)
    vitmatte: ViTMatteConfig = field(default_factory=ViTMatteConfig)

    # Depth normalization (applied to raw depth from depth stage)
    depth_norm_percentiles: Tuple[float, float] = (2.0, 98.0)

    # Hair detail polish (applied to unknown/edge regions only)
    hair_gamma: float = 0.8           # Closer to 1.0 preserves gradients
    hair_black_point: float = 0.02    # Lower preserves faint tips
    hair_gain: float = 1.1            # Less aggressive solidification
    hair_polish_enabled: bool = True  # Allow disabling entirely

    # Guided Filter settings (edge-aware smoothing)
    guided_filter_radius: int = 4         # 1-2 for hair, 4 for bodies/clothes
    guided_filter_eps: float = 1e-5       # Lower = stricter edge adherence

    # Output
    output_dir: str = "./vitmatte_output"
    output_format: str = "exr"
    bit_depth: int = 16

    # Debug
    save_trimap: bool = True
    save_debug: bool = False
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if self.vitmatte.model_size not in ("small", "base"):
            raise ValueError("model_size must be 'small' or 'base'")

        if self.hair_gamma < 0 or self.hair_gamma > 2:
            raise ValueError("hair_gamma must be between 0 and 2")

        if self.hair_black_point < 0 or self.hair_black_point > 1:
            raise ValueError("hair_black_point must be between 0 and 1")
