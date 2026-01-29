"""
AUTO-ROTO CLI Entry Point
=========================

Unified command-line interface for auto-roto.

Usage:
    python -m auto_roto <command> [options]

Commands:
    sam         - SAM3 segmentation
    depth       - Depth estimation
    pipeline    - Full pipeline (SAM + Depth + ViTMatte + Combine)
    interactive - Interactive SAM3 → MatAnyone workflow
    version     - Show version

Examples:
    python -m auto_roto sam --input video.mp4 --prompt "person"
    python -m auto_roto depth --input ./frames --output ./output
    python -m auto_roto pipeline --input video.mp4 --prompt "person" --quality high
    python -m auto_roto interactive --input video.mp4 --prompt "person"
"""

import sys
import argparse


def main():
    """Main entry point for auto-roto CLI."""
    parser = argparse.ArgumentParser(
        description="AUTO-ROTO: Production-grade automatic rotoscoping",
        usage="python -m auto_roto <command> [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  sam          SAM3 segmentation
  depth        Depth estimation
  pipeline     Full pipeline (SAM + Depth + ViTMatte + Combine)
  interactive  Interactive SAM3 → MatAnyone (recommended for video)
  version      Show version

Examples:
  python -m auto_roto sam --input video.mp4 --prompt "person"
  python -m auto_roto pipeline --input video.mp4 --prompt "person" --quality high
  python -m auto_roto interactive --input video.mp4 --prompt "person"

For command-specific help:
  python -m auto_roto <command> --help
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["sam", "depth", "pipeline", "interactive", "version"],
        help="Command to run"
    )

    # Parse only the command
    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "version":
        from auto_roto import __version__
        print(f"auto-roto version {__version__}")
        sys.exit(0)

    elif args.command == "sam":
        # Pass remaining args to SAM CLI
        sys.argv = [sys.argv[0]] + remaining
        from auto_roto.cli.sam import main as sam_main
        sam_main()

    elif args.command == "depth":
        # Pass remaining args to depth CLI
        sys.argv = [sys.argv[0]] + remaining
        from auto_roto.cli.depth import main as depth_main
        depth_main()

    elif args.command == "pipeline":
        # Pass remaining args to pipeline CLI
        sys.argv = [sys.argv[0]] + remaining
        from auto_roto.cli.pipeline import main as pipeline_main
        pipeline_main()

    elif args.command == "interactive":
        # Pass remaining args to interactive CLI
        sys.argv = [sys.argv[0]] + remaining
        from auto_roto.cli.interactive import main as interactive_main
        interactive_main()


if __name__ == "__main__":
    main()
