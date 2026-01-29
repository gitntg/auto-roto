"""
SAM3 Configuration
==================

Configuration for SAM3-based rotoscoping.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RotoConfig:
    """Configuration for the auto-roto pipeline using SAM3."""

    # Input/Output
    input_path: str = ""
    output_dir: str = "./output"

    # Detection mode
    prompt: Optional[str] = None  # Text prompt (SAM3 has built-in text prompting)
    box: Optional[str] = None     # Box prompt "x1,y1,x2,y2"
    point: Optional[str] = None   # Point prompt "x,y"
    interactive: bool = False     # Interactive selection mode

    # SAM3 settings (single model, no size choices like SAM2)
    propagate_forward: bool = True
    propagate_backward: bool = True

    # SAM3 inference settings
    sam_imgsz: int = 0              # Processing resolution (0 = auto from input, max 8192)
    sam_conf: float = 0.15          # Confidence threshold (0.0-1.0, lower = more detections/details)
    sam_retina_masks: bool = True   # High-resolution mask output
    sam_max_det: int = 100          # Maximum detections per frame

    # Alpha refinement
    refine_alpha: bool = True
    refine_iterations: int = 3
    edge_softness: float = 1.0

    # Output settings
    output_format: str = "exr"    # exr, png, tiff
    bit_depth: int = 16           # 8, 16, 32
    include_rgb: bool = True      # Include RGB in output
    frame_padding: int = 4        # Frame number padding (####)

    # Performance
    device: str = "cuda"
    batch_size: int = 1
    compile_model: bool = True    # Use torch.compile for speed

    # Debug
    verbose: bool = False
    save_preview: bool = True
    preview_scale: float = 0.5

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.input_path:
            raise ValueError("input_path is required")

        if self.sam_conf < 0 or self.sam_conf > 1:
            raise ValueError("sam_conf must be between 0 and 1")

        if self.bit_depth not in (8, 16, 32):
            raise ValueError("bit_depth must be 8, 16, or 32")

        if self.output_format not in ("exr", "png", "tiff", "tif"):
            raise ValueError("output_format must be exr, png, or tiff")
