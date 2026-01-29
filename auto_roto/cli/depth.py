"""
Depth CLI
=========

Command-line interface for depth estimation.

Usage:
    python -m auto_roto depth --input ./frames --output ./output
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
    setup_cli_logging,
)


def parse_args():
    """Parse command line arguments."""
    parser = create_base_parser(
        description="Depth Estimation - Generate depth maps using Depth Anything V3",
        epilog="""
EXAMPLES:
  # Basic depth estimation
  %(prog)s --input ./frames --output ./output

  # High quality with large model
  %(prog)s --input ./frames --output ./output --depth-model large

  # Custom processing resolution
  %(prog)s --input ./frames --output ./output --depth-res 2048 --depth-method lower
        """
    )

    add_input_args(parser)
    add_output_args(parser)
    add_performance_args(parser)
    add_debug_args(parser)

    # Depth-specific arguments
    parser.add_argument(
        "--depth-model",
        choices=["small", "base", "large", "nested-base", "nested-large"],
        default="base",
        help="Depth model size (default: base)"
    )
    parser.add_argument(
        "--depth-res",
        type=int,
        default=None,
        help="Processing resolution (default: auto)"
    )
    parser.add_argument(
        "--depth-method",
        choices=["upper", "lower"],
        default="upper",
        help="Resize method (default: upper)"
    )
    parser.add_argument(
        "--depth-percentiles",
        type=float,
        nargs=2,
        default=[2.0, 98.0],
        metavar=("LOW", "HIGH"),
        help="Normalization percentiles (default: 2.0 98.0)"
    )

    return parser.parse_args()


def main():
    """Main entry point for depth CLI."""
    # Check environment
    check_conda_environment()

    args = parse_args()

    # Setup logging
    logger = setup_cli_logging(args.verbose, getattr(args, 'debug', False))

    # Import pipeline
    from auto_roto.pipelines.depth_pipeline import DepthRefinePipeline

    # Create pipeline
    pipeline = DepthRefinePipeline(
        model=args.depth_model,
        process_res=args.depth_res,
        process_method=args.depth_method,
        percentiles=tuple(args.depth_percentiles),
        device=args.device,
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
