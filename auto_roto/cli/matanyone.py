"""
MatAnyone CLI
=============

Command-line interface for standalone MatAnyone temporal propagation.

Usage:
    python -m auto_roto matanyone --input ./frames --mask mask.exr --output ./alpha
"""

import sys
from pathlib import Path

from auto_roto.cli.common import (
    check_conda_environment,
    create_base_parser,
    add_output_args,
    add_performance_args,
    add_debug_args,
    setup_cli_logging,
)


def parse_args():
    """Parse command line arguments."""
    parser = create_base_parser(
        description="MatAnyone Temporal Propagation - Propagate a single-frame alpha to all frames",
        epilog="""
EXAMPLES:
  # Basic usage: propagate mask from frame 0
  %(prog)s --input ./frames --mask frame0_mask.exr --output ./alpha

  # From a video file
  %(prog)s --input video.mp4 --mask mask.exr --output ./alpha

  # High quality settings
  %(prog)s --input ./frames --mask mask.exr --mem-every 2 --max-mem-frames 15

  # Custom repo and checkpoint paths
  %(prog)s --input ./frames --mask mask.exr --repo ./MatAnyone --checkpoint ./checkpoints/matanyone.pth
        """
    )

    # Input arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file or directory of frames"
    )
    parser.add_argument(
        "--mask", "-m",
        required=True,
        help="Initial mask (from frame 0) for propagation"
    )

    add_output_args(parser)
    add_performance_args(parser)
    add_debug_args(parser)

    # MatAnyone model paths
    parser.add_argument(
        "--repo",
        default="./MatAnyone",
        help="Path to MatAnyone repository (default: ./MatAnyone)"
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/matanyone.pth",
        help="Path to MatAnyone checkpoint (default: ./checkpoints/matanyone.pth)"
    )

    # MatAnyone quality settings
    parser.add_argument(
        "--mem-every",
        type=int,
        default=3,
        help="Memory frame interval: lower = better quality, slower (default: 3)"
    )
    parser.add_argument(
        "--max-mem-frames",
        type=int,
        default=10,
        help="Max memory frames: higher = better quality, more VRAM (default: 10)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup frames (default: 5)"
    )

    # Morphological settings for initial mask
    parser.add_argument(
        "--erode",
        type=int,
        default=3,
        help="Erosion iterations on initial mask (default: 3)"
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=5,
        help="Dilation iterations on initial mask (default: 5)"
    )

    # Advanced settings
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k memory matching (default: 50)"
    )
    parser.add_argument(
        "--no-long-term",
        action="store_true",
        help="Disable long-term memory (use only short-term)"
    )
    parser.add_argument(
        "--max-internal-size",
        type=int,
        default=-1,
        help="Max internal processing size, -1 = full res (default: -1)"
    )

    return parser.parse_args()


def main():
    """Main entry point for MatAnyone CLI."""
    # Check environment
    check_conda_environment()

    args = parse_args()

    # Setup logging
    logger = setup_cli_logging(args.verbose, getattr(args, 'debug', False))

    from auto_roto.stages.matanyone import MatAnyoneStage
    from auto_roto.stages.base import StageContext, StageResult
    from auto_roto.io.alpha import load_alpha

    import cv2

    # Setup paths
    input_path = Path(args.input)
    mask_path = Path(args.mask)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    if not mask_path.exists():
        logger.error(f"Mask not found: {mask_path}")
        sys.exit(1)

    # Resolve frames directory
    frames_dir = None
    if input_path.is_dir():
        frames_dir = input_path
    else:
        # Extract frames from video
        frames_dir = output_dir / "00_frames"
        frames_dir.mkdir(exist_ok=True)

        logger.info(f"Extracting frames from: {input_path}")
        cap = cv2.VideoCapture(str(input_path))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = frames_dir / f"frame.{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            idx += 1

        cap.release()

        if idx == 0:
            logger.error(f"No frames extracted from: {input_path}")
            sys.exit(1)

        logger.info(f"Extracted {idx} frames")

    # Create context
    context = StageContext(
        input_path=input_path,
        output_dir=output_dir,
        frames_dir=frames_dir,
        device=args.device,
        quality="standard",
        logger=logger,
        verbose=args.verbose,
        config={
            "output_format": args.format,
            "bit_depth": args.bit_depth
        }
    )

    # We need to create a mock "combine" stage result that points to the mask
    # so MatAnyoneStage can find the initial alpha
    mock_combine_dir = output_dir / "mock_combine"
    mock_alpha_dir = mock_combine_dir / "alpha"
    mock_alpha_dir.mkdir(parents=True, exist_ok=True)

    # Copy/link the initial mask
    import shutil
    mask_ext = mask_path.suffix
    target_mask = mock_alpha_dir / f"alpha.0000{mask_ext}"
    shutil.copy2(mask_path, target_mask)

    # Add to context
    context.stage_outputs["combine"] = StageResult.success_result(
        output_dir=mock_combine_dir,
        message="Mock combine for MatAnyone",
        outputs={"alpha_dir": str(mock_alpha_dir)}
    )

    # Create and run MatAnyone stage
    matanyone_stage = MatAnyoneStage(
        repo_path=args.repo,
        checkpoint_path=args.checkpoint,
        mem_every=args.mem_every,
        max_mem_frames=args.max_mem_frames,
        warmup=args.warmup,
        erode=args.erode,
        dilate=args.dilate,
        top_k=args.top_k,
        use_long_term=not args.no_long_term,
        max_internal_size=args.max_internal_size,
        logger=logger
    )

    result = matanyone_stage.execute(context)

    # Cleanup mock directory
    shutil.rmtree(mock_combine_dir, ignore_errors=True)

    if result.success:
        logger.info(f"MatAnyone propagation complete!")
        logger.info(f"Output: {result.output_dir}")
        sys.exit(0)
    else:
        logger.error(f"MatAnyone failed: {result.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
