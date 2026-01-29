"""
Interactive CLI
===============

Command-line interface for interactive SAM3 → MatAnyone pipeline.

Usage:
    python -m auto_roto interactive --input video.mp4 --prompt "person"
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
        description="Interactive SAM3 → MatAnyone Pipeline",
        epilog="""
WORKFLOW:
  1. SAM3 segments first frame with your text prompt
  2. Preview opens - review the mask
  3. Choose: Accept, Add points to refine, or Quit
  4. If refining: click include/exclude points, re-preview
  5. Once approved, MatAnyone processes entire video

EXAMPLES:
  # Basic usage
  %(prog)s --input video.mp4 --prompt "person"

  # Custom MatAnyone settings
  %(prog)s --input video.mp4 --prompt "person" --warmup 10 --erode 15 --dilate 15

  # Multiple prompts
  %(prog)s --input video.mp4 --prompt "person.dog"

MATANYONE SETTINGS (auto-detected by resolution):
  Low-res (≤576p):  warmup=1,  erode=4,  dilate=4
  High-res (>576p): warmup=10, erode=15, dilate=15
        """
    )

    add_input_args(parser)
    add_output_args(parser)
    add_performance_args(parser)
    add_debug_args(parser)

    # Prompt
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Text prompt for SAM3 (use . to separate multiple)"
    )

    # MatAnyone settings
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=None,
        help="MatAnyone warmup iterations (default: auto by resolution)"
    )
    parser.add_argument(
        "--erode",
        type=int,
        default=None,
        help="Erosion kernel size (default: auto by resolution)"
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=None,
        help="Dilation kernel size (default: auto by resolution)"
    )

    # MatAnyone paths
    parser.add_argument(
        "--matanyone-repo",
        default="./MatAnyone",
        help="Path to MatAnyone repository"
    )
    parser.add_argument(
        "--matanyone-checkpoint",
        default="./checkpoints/matanyone.pth",
        help="Path to MatAnyone checkpoint"
    )

    # Auto settings
    parser.add_argument(
        "--no-auto-settings",
        action="store_true",
        help="Disable auto-adjustment of warmup/erode/dilate based on resolution"
    )

    return parser.parse_args()


def main():
    """Main entry point for interactive CLI."""
    try:
        # Check environment
        check_conda_environment()

        args = parse_args()

        # Setup logging
        logger = setup_cli_logging(args.verbose, getattr(args, 'debug', False))

        # Import pipeline
        from auto_roto.pipelines.interactive_matanyone import InteractiveMatAnyonePipeline

        # Determine settings
        warmup = args.warmup if args.warmup is not None else 10
        erode = args.erode if args.erode is not None else 10
        dilate = args.dilate if args.dilate is not None else 10

        # Create pipeline
        pipeline = InteractiveMatAnyonePipeline(
            prompt=args.prompt,
            warmup=warmup,
            erode_kernel=erode,
            dilate_kernel=dilate,
            device=args.device,
            output_format=args.format,
            bit_depth=args.bit_depth,
            matanyone_repo=args.matanyone_repo,
            matanyone_checkpoint=args.matanyone_checkpoint,
            logger=logger
        )

        # Run pipeline
        success = pipeline.run(
            input_path=args.input,
            output_dir=args.output,
            auto_resolution_settings=not args.no_auto_settings,
            verbose=args.verbose
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    finally:
        # Clean up any remaining OpenCV windows
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
