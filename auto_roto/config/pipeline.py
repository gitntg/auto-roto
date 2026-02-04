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

    # Pipeline preset (overrides quality settings with full workflow config)
    preset: str = ""  # cinema, quick, or empty for manual config

    # Depth expansion settings (for cinema preset)
    depth_expansion_enabled: bool = False
    depth_expansion_tolerance: float = 0.1
    depth_expansion_percentiles: Tuple[float, float] = (10, 90)
    depth_expansion_connectivity: bool = True
    depth_expansion_max_px: int = 50
    depth_expansion_edge_aware: bool = True

    # Stage control
    skip_sam: bool = False
    skip_depth: bool = False
    skip_vitmatte: bool = False
    skip_combine: bool = False
    skip_hair: bool = True      # Hair refinement optional, off by default
    clean_output: bool = False  # Clear output directories before running

    # MatAnyone temporal propagation mode
    # When enabled, SAM/Depth/ViTMatte/Combine only process frame 1
    # and MatAnyone propagates that alpha temporally to all frames
    use_matanyone: bool = False

    # Alpha refinement selection
    refiner: str = "vitmatte"   # vitmatte or ma1 (MatAnyone1)

    # ViTMatte settings (adaptive trimap)
    vitmatte_motion_aware: bool = False
    vitmatte_adaptive_base: float = 2.0
    vitmatte_adaptive_max: float = 60.0

    # MatAnyone settings (both refiner=ma1 and use_matanyone modes)
    mam2_repo: str = "./MatAnyone"
    mam2_checkpoint: str = "./checkpoints/matanyone.pth"
    matanyone_mem_every: int = 3           # Memory frame interval (lower = better, slower)
    matanyone_max_mem_frames: int = 10     # Max memory frames (higher = better, more VRAM)
    matanyone_warmup: int = 5              # Number of warmup frames
    matanyone_erode: int = 3               # Erosion iterations for initial mask
    matanyone_dilate: int = 5              # Dilation iterations for initial mask
    matanyone_top_k: int = 50              # Top-k memory matching
    matanyone_use_long_term: bool = True   # Enable long-term memory
    matanyone_max_internal_size: int = -1  # Max internal processing size (-1 = full res)

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

        if self.preset and self.preset not in ("cinema", "quick"):
            raise ValueError("preset must be 'cinema', 'quick', or empty")

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

    def apply_pipeline_preset(self) -> None:
        """
        Apply pipeline preset settings to configuration.

        Pipeline presets (cinema, quick) override quality presets
        with full workflow configurations including stage selection,
        depth expansion, and temporal settings.
        """
        if not self.preset:
            return

        from auto_roto.config.presets import get_pipeline_preset

        preset_config = get_pipeline_preset(self.preset)
        if not preset_config:
            return

        settings = preset_config.get('settings', {})

        # Apply all settings from preset
        if 'depth_model' in settings:
            self.depth_model = settings['depth_model']

        if 'depth_process_res' in settings:
            self.depth_process_res = settings['depth_process_res']

        if 'depth_process_method' in settings:
            self.depth_process_method = settings['depth_process_method']

        if 'use_matanyone' in settings:
            self.use_matanyone = settings['use_matanyone']

        # Depth expansion settings
        if 'depth_expansion_enabled' in settings:
            self.depth_expansion_enabled = settings['depth_expansion_enabled']

        if 'depth_expansion_tolerance' in settings:
            self.depth_expansion_tolerance = settings['depth_expansion_tolerance']

        if 'depth_expansion_percentiles' in settings:
            self.depth_expansion_percentiles = settings['depth_expansion_percentiles']

        if 'depth_expansion_connectivity' in settings:
            self.depth_expansion_connectivity = settings['depth_expansion_connectivity']

        if 'depth_expansion_max_px' in settings:
            self.depth_expansion_max_px = settings['depth_expansion_max_px']

        # Hair polish settings
        if 'hair_gamma' in settings:
            self.hair_gamma = settings['hair_gamma']

        if 'hair_black_point' in settings:
            self.hair_black_point = settings['hair_black_point']

        if 'hair_gain' in settings:
            self.hair_gain = settings['hair_gain']

        # Guided filter settings
        if 'guided_filter_radius' in settings:
            self.guided_filter_radius = settings['guided_filter_radius']

        if 'guided_filter_eps' in settings:
            self.guided_filter_eps = settings['guided_filter_eps']

    def get_active_stages(self) -> list:
        """
        Get list of stages to run based on preset or manual configuration.

        Returns:
            List of stage names to execute
        """
        if self.preset:
            from auto_roto.config.presets import get_pipeline_preset
            preset_config = get_pipeline_preset(self.preset)
            if preset_config and 'stages' in preset_config:
                return preset_config['stages']

        # Default stage list based on manual configuration
        stages = []

        if not self.skip_sam:
            stages.append('sam')

        if not self.skip_depth:
            stages.append('depth')

        if self.depth_expansion_enabled:
            stages.append('depth_expand')

        if not self.skip_vitmatte:
            stages.append('vitmatte')

        if not self.skip_combine:
            stages.append('combine')

        if self.use_matanyone:
            stages.append('matanyone')

        return stages
