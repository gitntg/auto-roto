"""
MatAnyone Configuration
=======================

Configuration for MatAnyone (formerly Mam2) alpha refinement.

Note: This replaces the old Mam2Config. The naming was corrected
to match the actual model being used (MatAnyone v1).
"""

from dataclasses import dataclass


@dataclass
class MatAnyoneConfig:
    """Configuration for MatAnyone alpha refinement."""

    # Input paths
    sam_mask: str = ""
    frames: str = ""
    output: str = ""

    # Model paths
    checkpoint: str = "./checkpoints/matanyone.pth"
    repo: str = "./MatAnyone"

    # Device
    device: str = "cuda"

    # Output settings
    output_format: str = "exr"
    bit_depth: int = 16

    # Processing settings
    mask_threshold: float = 0.5
    allow_fallback: bool = True

    # MatAnyone-specific settings
    ma1_warmup: int = 5
    ma1_erode: int = 3
    ma1_dilate: int = 5

    # Quality parameters
    mem_every: int = 3          # Memory frame interval (lower = better quality, slower)
    max_mem_frames: int = 10    # Max memory frames (higher = better quality, more VRAM)
    top_k: int = 50             # Top-k memory matching (higher = more accurate)
    use_long_term: bool = True  # Enable long-term memory for better temporal consistency
    max_internal_size: int = -1 # Max internal processing size (-1 = no resize)

    # Debug
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if self.mask_threshold < 0 or self.mask_threshold > 1:
            raise ValueError("mask_threshold must be between 0 and 1")

        if self.mem_every < 1:
            raise ValueError("mem_every must be at least 1")

        if self.max_mem_frames < 1:
            raise ValueError("max_mem_frames must be at least 1")


# Backward compatibility alias
Mam2Config = MatAnyoneConfig
