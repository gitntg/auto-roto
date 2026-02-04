"""
Full Pipeline CLI
=================

Command-line interface for the full auto-roto pipeline.

Usage:
    python -m auto_roto pipeline --input video.mp4 --prompt "person"
"""

import sys
from pathlib import Path

from auto_roto.cli.common import (
    check_conda_environment,
    create_base_parser,
    add_input_args,
    add_output_args,
    add_performance_args,
    add_debug_args,
    add_quality_args,
    add_prompt_args,
    add_stage_control_args,
    setup_cli_logging,
)


def parse_args():
    """Parse command line arguments."""
    parser = create_base_parser(
        description="AUTO-ROTO Full Pipeline - Production-grade automatic rotoscoping",
        epilog="""
EXAMPLES:
  # Cinema preset (production quality with depth expansion + temporal)
  %(prog)s --input video.mp4 --prompt "person" --preset cinema

  # Quick preview
  %(prog)s --input video.mp4 --prompt "person" --preset quick

  # Standard quality (manual config)
  %(prog)s --input video.mp4 --prompt "person" --output ./output

  # High quality with hair refinement
  %(prog)s --input video.mp4 --prompt "person" --quality high --with-hair

  # MatAnyone temporal mode: refine frame 1, propagate to all frames
  %(prog)s --input video.mp4 --prompt "person" --use-matanyone

  # Process PNG sequence
  %(prog)s --input /path/to/frames/ --prompt "person" --output ./output

PIPELINE PRESETS (recommended):
  cinema   - Production quality: depth expansion + MatAnyone temporal
  quick    - Fast single-frame preview

QUALITY PRESETS (for fine-tuning):
  draft    - Fastest, small depth model
  standard - Balanced quality and speed (default)
  high     - Best quality, large depth model
  ultra    - Maximum quality, large depth model
        """
    )

    add_input_args(parser)
    add_output_args(parser)
    add_prompt_args(parser)
    add_quality_args(parser)
    add_stage_control_args(parser)
    add_performance_args(parser)
    add_debug_args(parser)

    # Pipeline preset (overrides quality settings)
    parser.add_argument(
        "--preset",
        choices=["cinema", "quick"],
        default=None,
        help="Pipeline preset: cinema (production quality + temporal), quick (fast preview)"
    )

    # Depth expansion settings (for cinema preset or manual use)
    parser.add_argument(
        "--depth-expand",
        action="store_true",
        help="Enable depth-guided mask expansion"
    )
    parser.add_argument(
        "--depth-expand-tolerance",
        type=float,
        default=0.1,
        help="Depth expansion tolerance (0-1)"
    )
    parser.add_argument(
        "--depth-expand-max-px",
        type=int,
        default=50,
        help="Max expansion distance in pixels"
    )

    # SAM settings
    parser.add_argument(
        "--sam-imgsz",
        type=int,
        default=0,
        help="SAM processing resolution (0=auto, max 2048)"
    )
    parser.add_argument(
        "--sam-conf",
        type=float,
        default=0.25,
        help="SAM confidence threshold"
    )

    # Depth settings
    parser.add_argument(
        "--depth-model",
        choices=["small", "base", "large", "nested-base", "nested-large"],
        help="Override depth model"
    )
    parser.add_argument(
        "--depth-res",
        type=int,
        default=None,
        help="Depth processing resolution"
    )
    parser.add_argument(
        "--depth-method",
        choices=["upper", "lower"],
        default="upper",
        help="Depth resize method"
    )
    parser.add_argument(
        "--depth-percentiles",
        type=float,
        nargs=2,
        default=[2.0, 98.0],
        metavar=("LOW", "HIGH"),
        help="Depth normalization percentiles"
    )

    # ViTMatte settings
    parser.add_argument(
        "--vitmatte-motion",
        action="store_true",
        help="Enable motion-aware trimap"
    )
    parser.add_argument(
        "--vitmatte-base",
        type=float,
        default=2.0,
        help="Min unknown width for smooth regions"
    )
    parser.add_argument(
        "--vitmatte-max",
        type=float,
        default=60.0,
        help="Max unknown width for complex regions"
    )

    # Hair polish settings
    parser.add_argument(
        "--hair-gamma",
        type=float,
        default=None,
        help="Hair edge gamma correction"
    )
    parser.add_argument(
        "--hair-black-point",
        type=float,
        default=None,
        help="Hair black point threshold"
    )
    parser.add_argument(
        "--hair-gain",
        type=float,
        default=None,
        help="Hair gain multiplier"
    )
    parser.add_argument(
        "--no-hair-polish",
        action="store_true",
        help="Disable hair polish"
    )

    # Guided filter settings
    parser.add_argument(
        "--guided-radius",
        type=int,
        default=None,
        help="Guided filter radius"
    )
    parser.add_argument(
        "--guided-eps",
        type=float,
        default=None,
        help="Guided filter epsilon"
    )

    # Matte combine settings
    parser.add_argument(
        "--core-shrink",
        type=int,
        default=3,
        help="Core shrink pixels"
    )
    parser.add_argument(
        "--despill",
        type=float,
        default=0.5,
        help="Despill strength"
    )

    # Refiner selection
    parser.add_argument(
        "--refiner",
        choices=["vitmatte", "ma1"],
        default="vitmatte",
        help="Alpha refinement model"
    )
    parser.add_argument(
        "--mam2-checkpoint",
        default="./checkpoints/matanyone.pth",
        help="MatAnyone checkpoint path"
    )
    parser.add_argument(
        "--mam2-repo",
        default="./MatAnyone",
        help="MatAnyone repo path"
    )

    # MatAnyone temporal propagation mode
    parser.add_argument(
        "--use-matanyone",
        action="store_true",
        help="Enable MatAnyone temporal mode: refine frame 1 only, propagate to all"
    )
    parser.add_argument(
        "--matanyone-mem-every",
        type=int,
        default=3,
        help="MatAnyone memory interval (lower=better, slower)"
    )
    parser.add_argument(
        "--matanyone-max-mem-frames",
        type=int,
        default=10,
        help="MatAnyone max memory frames (higher=better, more VRAM)"
    )
    parser.add_argument(
        "--matanyone-warmup",
        type=int,
        default=5,
        help="MatAnyone warmup frames"
    )
    parser.add_argument(
        "--matanyone-erode",
        type=int,
        default=3,
        help="MatAnyone mask erosion iterations"
    )
    parser.add_argument(
        "--matanyone-dilate",
        type=int,
        default=5,
        help="MatAnyone mask dilation iterations"
    )

    return parser.parse_args()


def main():
    """Main entry point for full pipeline CLI."""
    # Check environment
    check_conda_environment()

    args = parse_args()

    # Setup logging
    logger = setup_cli_logging(args.verbose, getattr(args, 'debug', False))

    # Import pipeline
    from auto_roto.pipelines.full_pipeline import FullPipeline
    from auto_roto.config.pipeline import PipelineConfig

    # Create config
    config = PipelineConfig(
        prompt=args.prompt or "",
        box=args.box if hasattr(args, 'box') and args.box else "",
        quality=args.quality,
        preset=args.preset or "",
        device=args.device,
        output_format=args.format,
        bit_depth=args.bit_depth,
        # Depth expansion settings
        depth_expansion_enabled=args.depth_expand or (args.preset == "cinema"),
        depth_expansion_tolerance=args.depth_expand_tolerance,
        depth_expansion_max_px=args.depth_expand_max_px,
        # SAM settings
        sam_imgsz=args.sam_imgsz,
        sam_conf=args.sam_conf,
        # Depth settings
        depth_model=args.depth_model or "",
        depth_process_res=args.depth_res,
        depth_process_method=args.depth_method,
        depth_norm_percentiles=tuple(args.depth_percentiles),
        # ViTMatte settings
        vitmatte_motion_aware=args.vitmatte_motion,
        vitmatte_adaptive_base=args.vitmatte_base,
        vitmatte_adaptive_max=args.vitmatte_max,
        # Hair settings
        hair_gamma=args.hair_gamma,
        hair_black_point=args.hair_black_point,
        hair_gain=args.hair_gain,
        hair_polish_enabled=not args.no_hair_polish,
        # Guided filter
        guided_filter_radius=args.guided_radius,
        guided_filter_eps=args.guided_eps,
        # Matte combine
        core_shrink=args.core_shrink,
        despill_strength=args.despill,
        # Refiner
        refiner=args.refiner,
        mam2_checkpoint=args.mam2_checkpoint,
        mam2_repo=args.mam2_repo,
        # MatAnyone temporal mode
        use_matanyone=args.use_matanyone or (args.preset == "cinema"),
        matanyone_mem_every=args.matanyone_mem_every,
        matanyone_max_mem_frames=args.matanyone_max_mem_frames,
        matanyone_warmup=args.matanyone_warmup,
        matanyone_erode=args.matanyone_erode,
        matanyone_dilate=args.matanyone_dilate,
    )

    # Create pipeline
    pipeline = FullPipeline(config=config, logger=logger)

    # Run pipeline
    success = pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        skip_sam=args.skip_sam,
        skip_depth=args.skip_depth,
        skip_vitmatte=args.skip_vitmatte,
        skip_combine=args.skip_combine,
        keep_intermediate=args.keep_intermediate,
        verbose=args.verbose
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
