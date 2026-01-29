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
  # Standard quality
  %(prog)s --input video.mp4 --prompt "person" --output ./output

  # High quality with hair refinement
  %(prog)s --input video.mp4 --prompt "person" --quality high --with-hair

  # Ultra quality
  %(prog)s --input video.mp4 --prompt "person" --quality ultra

  # Process PNG sequence
  %(prog)s --input /path/to/frames/ --prompt "person" --output ./output

QUALITY PRESETS:
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
        device=args.device,
        output_format=args.format,
        bit_depth=args.bit_depth,
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
