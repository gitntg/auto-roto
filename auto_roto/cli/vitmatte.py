"""
ViTMatte CLI
============

Command-line interface for standalone ViTMatte alpha refinement.

Usage:
    python -m auto_roto vitmatte --input frame.jpg --mask mask.exr --depth depth.exr
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
        description="ViTMatte Alpha Refinement - Refine SAM masks into high-quality alpha mattes",
        epilog="""
EXAMPLES:
  # Basic usage with SAM mask and depth
  %(prog)s --input frame.jpg --mask mask.exr --depth depth.exr

  # Process a directory of frames
  %(prog)s --input ./frames --mask ./masks --depth ./depth --output ./alpha

  # Custom trimap instead of auto-generation
  %(prog)s --input frame.jpg --trimap trimap.png

  # Adjust model size
  %(prog)s --input frame.jpg --mask mask.exr --depth depth.exr --model-size base
        """
    )

    # Input arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input RGB image or directory of frames"
    )
    parser.add_argument(
        "--mask", "-m",
        required=True,
        help="Input SAM mask or directory of masks"
    )
    parser.add_argument(
        "--depth", "-d",
        required=True,
        help="Input depth map or directory of depth maps"
    )
    parser.add_argument(
        "--trimap", "-t",
        default=None,
        help="Optional custom trimap (overrides auto-generation)"
    )

    add_output_args(parser)
    add_performance_args(parser)
    add_debug_args(parser)

    # ViTMatte-specific arguments
    parser.add_argument(
        "--model-size",
        choices=["small", "base"],
        default="base",
        help="ViTMatte model size (default: base)"
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=2048,
        help="Maximum processing resolution (default: 2048)"
    )
    parser.add_argument(
        "--save-trimap",
        action="store_true",
        help="Save generated trimaps"
    )

    # Trimap settings
    parser.add_argument(
        "--trimap-mode",
        choices=["adaptive", "simple"],
        default="adaptive",
        help="Trimap generation mode (default: adaptive)"
    )
    parser.add_argument(
        "--unknown-width",
        type=int,
        default=30,
        help="Unknown region width for simple mode (default: 30)"
    )

    # Hair polish settings
    parser.add_argument(
        "--hair-gamma",
        type=float,
        default=0.8,
        help="Hair edge gamma correction (default: 0.8)"
    )
    parser.add_argument(
        "--hair-black-point",
        type=float,
        default=0.02,
        help="Hair black point threshold (default: 0.02)"
    )
    parser.add_argument(
        "--hair-gain",
        type=float,
        default=1.1,
        help="Hair gain multiplier (default: 1.1)"
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
        default=4,
        help="Guided filter radius (default: 4)"
    )
    parser.add_argument(
        "--guided-eps",
        type=float,
        default=1e-5,
        help="Guided filter epsilon (default: 1e-5)"
    )

    return parser.parse_args()


def main():
    """Main entry point for ViTMatte CLI."""
    # Check environment
    check_conda_environment()

    args = parse_args()

    # Setup logging
    logger = setup_cli_logging(args.verbose, getattr(args, 'debug', False))

    import cv2
    import numpy as np

    from auto_roto.config.vitmatte import GeometricMatteConfig, TrimapConfig, ViTMatteConfig
    from auto_roto.refiners.geometric import GeometricMatteRefiner
    from auto_roto.io.alpha import load_alpha, save_alpha
    from auto_roto.io.depth import load_depth_float

    # Setup paths
    input_path = Path(args.input)
    mask_path = Path(args.mask)
    depth_path = Path(args.depth)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    trimap_config = TrimapConfig(
        adaptive_mode=(args.trimap_mode == "adaptive"),
        simple_edge_mode=(args.trimap_mode == "simple"),
        simple_edge_width=args.unknown_width
    )

    vitmatte_config = ViTMatteConfig(
        model_size=args.model_size,
        device=args.device,
        max_resolution=args.max_resolution
    )

    geometric_config = GeometricMatteConfig(
        trimap=trimap_config,
        vitmatte=vitmatte_config,
        hair_gamma=args.hair_gamma,
        hair_black_point=args.hair_black_point,
        hair_gain=args.hair_gain,
        hair_polish_enabled=not args.no_hair_polish,
        guided_filter_radius=args.guided_radius,
        guided_filter_eps=args.guided_eps,
        save_trimap=args.save_trimap
    )

    # Initialize refiner
    refiner = GeometricMatteRefiner(config=geometric_config, logger=logger)

    try:
        # Determine if processing single file or directory
        if input_path.is_file():
            # Single file processing
            logger.info(f"Processing single frame: {input_path}")

            # Load inputs
            frame = cv2.imread(str(input_path))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = load_alpha(mask_path)
            depth = load_depth_float(depth_path)

            # Set trimap save path
            trimap_path = None
            if args.save_trimap:
                trimap_path = output_dir / "trimap.png"

            # Process
            alpha = refiner.process_frame(
                rgb=rgb,
                sam_mask=mask,
                depth=depth,
                save_trimap_path=trimap_path
            )

            # Save output
            output_path = output_dir / f"alpha.{args.format}"
            save_alpha(output_path, alpha, args.bit_depth)
            logger.info(f"Saved alpha to: {output_path}")

        else:
            # Directory processing
            logger.info(f"Processing directory: {input_path}")

            # Get file lists
            frame_files = sorted(
                list(input_path.glob("*.png")) +
                list(input_path.glob("*.jpg"))
            )

            mask_files = sorted(
                list(mask_path.glob("*.exr")) +
                list(mask_path.glob("*.png"))
            )

            depth_files = sorted(depth_path.glob("*.exr"))

            if not frame_files or not mask_files or not depth_files:
                logger.error("No input files found")
                sys.exit(1)

            min_count = min(len(frame_files), len(mask_files), len(depth_files))
            logger.info(f"Processing {min_count} frames")

            # Create output directories
            alpha_out = output_dir / "alpha"
            alpha_out.mkdir(exist_ok=True)

            trimap_out = None
            if args.save_trimap:
                trimap_out = output_dir / "trimap"
                trimap_out.mkdir(exist_ok=True)

            for idx in range(min_count):
                # Load inputs
                frame = cv2.imread(str(frame_files[idx]))
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mask = load_alpha(mask_files[idx])
                depth = load_depth_float(depth_files[idx])

                # Set trimap save path
                trimap_path = None
                if trimap_out:
                    trimap_path = trimap_out / f"trimap.{idx:04d}.png"

                # Process
                alpha = refiner.process_frame(
                    rgb=rgb,
                    sam_mask=mask,
                    depth=depth,
                    save_trimap_path=trimap_path
                )

                # Save output
                output_path = alpha_out / f"alpha.{idx:04d}.{args.format}"
                save_alpha(output_path, alpha, args.bit_depth)

                logger.info(f"Processed frame {idx + 1}/{min_count}")

            logger.info(f"Output saved to: {alpha_out}")

        logger.info("ViTMatte refinement complete")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        refiner.release()


if __name__ == "__main__":
    main()
