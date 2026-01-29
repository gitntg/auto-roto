"""
Pipeline Configuration
======================

Configuration for the full auto-roto pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline using SAM3."""

    # Input/Output
    input_path: str = ""
    output_dir: str = "./output"

    # Detection
    prompt: str = ""
    box: str = ""
    interactive: bool = False
    interactive_points: bool = False  # Include/exclude point marking mode
    detail_points: bool = False       # Supplementary point marking for missed details

    # Quality preset
    quality: str = "standard"  # draft, standard, high, ultra

    # Stage control
    skip_sam: bool = False
    skip_depth: bool = False
    skip_vitmatte: bool = False
    skip_combine: bool = False
    skip_hair: bool = True      # Hair refinement optional, off by default
    clean_output: bool = False  # Clear output directories before running

    # Alpha refinement selection
    refiner: str = "vitmatte"   # vitmatte or ma1 (MatAnyone1)

    # ViTMatte settings (adaptive trimap)
    vitmatte_motion_aware: bool = False
    vitmatte_adaptive_base: float = 2.0
    vitmatte_adaptive_max: float = 60.0

    # MatAnyone1 settings
    mam2_repo: str = "./MatAnyone"
    mam2_checkpoint: str = "./checkpoints/matanyone.pth"

    # Hair polish settings (applied to edge/unknown regions)
    hair_gamma: Optional[float] = None
    hair_black_point: Optional[float] = None
    hair_gain: Optional[float] = None
    hair_polish_enabled: bool = True

    # Guided Filter settings
    guided_filter_radius: Optional[int] = None
    guided_filter_eps: Optional[float] = None

    # SAM3 inference settings
    sam_imgsz: int = 0              # Processing resolution (0 = auto)
    sam_conf: float = 0.25          # Confidence threshold
    sam_retina_masks: bool = True   # High-resolution mask output
    sam_max_det: int = 100          # Maximum detections per frame

    # Depth model (auto-set by quality preset if empty)
    depth_model: str = ""

    # DA3 Depth Sensitivity Settings
    depth_process_res: Optional[int] = None
    depth_process_method: str = "upper"
    depth_norm_percentiles: Tuple[float, float] = (2.0, 98.0)
    use_depth_confidence: bool = False

    # Pure depth pass (runs BEFORE SAM, no alpha input)
    depth_first: bool = False
    export_pointcloud: bool = False
    pointcloud_format: str = "ply"

    # Matte combine settings
    core_shrink: int = 3
    despill_strength: float = 0.5

    # Output settings
    output_format: str = "exr"
    bit_depth: int = 16

    # Performance
    device: str = "cuda"
    no_compile: bool = False
    force_compile: bool = False

    # Debug
    verbose: bool = False
    keep_intermediate: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.input_path:
            raise ValueError("input_path is required")

        if self.quality not in ("draft", "standard", "high", "ultra"):
            raise ValueError("quality must be draft, standard, high, or ultra")

        if self.refiner not in ("vitmatte", "ma1"):
            raise ValueError("refiner must be 'vitmatte' or 'ma1'")

        if self.sam_conf < 0 or self.sam_conf > 1:
            raise ValueError("sam_conf must be between 0 and 1")

        if self.bit_depth not in (8, 16, 32):
            raise ValueError("bit_depth must be 8, 16, or 32")

    def apply_quality_preset(self) -> None:
        """Apply quality preset settings to configuration."""
        from auto_roto.config.presets import get_quality_preset

        preset = get_quality_preset(self.quality)

        # Apply preset values if not explicitly set
        if not self.depth_model:
            self.depth_model = preset['depth_model']

        if self.depth_process_res is None:
            self.depth_process_res = preset.get('depth_process_res')

        if self.hair_gamma is None:
            self.hair_gamma = preset.get('hair_gamma', 0.8)

        if self.hair_black_point is None:
            self.hair_black_point = preset.get('hair_black_point', 0.02)

        if self.hair_gain is None:
            self.hair_gain = preset.get('hair_gain', 1.1)

        if self.guided_filter_radius is None:
            self.guided_filter_radius = preset.get('guided_filter_radius', 4)

        if self.guided_filter_eps is None:
            self.guided_filter_eps = preset.get('guided_filter_eps', 1e-5)
