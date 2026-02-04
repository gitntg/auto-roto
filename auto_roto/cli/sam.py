"""
SAM CLI
=======

Command-line interface for SAM3 segmentation.

Usage:
    python -m auto_roto sam --input video.mp4 --prompt "person"
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
    add_prompt_args,
    setup_cli_logging,
)


def parse_args():
    """Parse command line arguments."""
    parser = create_base_parser(
        description="SAM3 Segmentation - Extract alpha mattes using Segment Anything",
        epilog="""
EXAMPLES:
  # Basic text prompt
  %(prog)s --input video.mp4 --prompt "person"

  # Multiple prompts (use . separator)
  %(prog)s --input video.mp4 --prompt "person.dog"

  # Box prompt
  %(prog)s --input video.mp4 --box "100,100,500,500"

  # Interactive point selection
  %(prog)s --input video.mp4 --interactive-points
        """
    )

    add_input_args(parser)
    add_output_args(parser)
    add_prompt_args(parser)
    add_performance_args(parser)
    add_debug_args(parser)

    # SAM-specific arguments
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
        help="SAM confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--no-retina-masks",
        action="store_true",
        help="Disable high-resolution mask output"
    )
    parser.add_argument(
        "--sam-max-det",
        type=int,
        default=100,
        help="Maximum detections per frame (default: 100)"
    )

    return parser.parse_args()


def main():
    """Main entry point for SAM CLI."""
    # Check environment
    check_conda_environment()

    args = parse_args()

    # Setup logging
    logger = setup_cli_logging(args.verbose, getattr(args, 'debug', False))

    # Import pipeline
    from auto_roto.pipelines.sam_pipeline import AutoRotoPipeline
    from auto_roto.config.sam import RotoConfig

    # Create config
    config = RotoConfig(
        sam_imgsz=args.sam_imgsz,
        sam_conf=args.sam_conf,
        sam_retina_masks=not args.no_retina_masks,
        sam_max_det=args.sam_max_det
    )

    # Parse prompts
    prompts = None
    if args.prompt:
        prompts = args.prompt.split(".")

    # Create pipeline
    pipeline = AutoRotoPipeline(
        config=config,
        prompts=prompts,
        box=args.box if hasattr(args, 'box') and args.box else None,
        device=args.device,
        output_format=args.format,
        bit_depth=args.bit_depth,
        logger=logger
    )

    # Run pipeline
    result = pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        verbose=args.verbose
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
