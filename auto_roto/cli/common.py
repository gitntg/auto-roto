"""
Common CLI Utilities
====================

Shared utilities for command-line interfaces.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Environment verification
REQUIRED_ENV = "autoroto"


def check_conda_environment():
    """Verify we're running in the correct conda environment."""
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != REQUIRED_ENV:
        print("\n" + "="*60)
        print("WRONG CONDA ENVIRONMENT")
        print("="*60)
        print(f"\n  Current environment: {current_env or '(none/base)'}")
        print(f"  Required environment: {REQUIRED_ENV}")
        print(f"\n  Please activate the correct environment:")
        print(f"    conda activate {REQUIRED_ENV}")
        print("\n" + "="*60)
        sys.exit(1)


def create_base_parser(
    description: str,
    epilog: str = None
) -> argparse.ArgumentParser:
    """Create a base argument parser with common settings."""
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )


def add_input_args(parser: argparse.ArgumentParser):
    """Add input arguments to parser."""
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video or image sequence directory"
    )


def add_output_args(parser: argparse.ArgumentParser):
    """Add output arguments to parser."""
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--format",
        default="exr",
        choices=["exr", "png", "tiff"],
        help="Output format (default: exr)"
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=16,
        choices=[8, 16, 32],
        help="Output bit depth (default: 16)"
    )


def add_performance_args(parser: argparse.ArgumentParser):
    """Add performance arguments to parser."""
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (cuda/cpu, default: cuda)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile"
    )
    parser.add_argument(
        "--force-compile",
        action="store_true",
        help="Force torch.compile even on Windows"
    )


def add_debug_args(parser: argparse.ArgumentParser):
    """Add debug arguments to parser."""
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with extra logging"
    )


def add_quality_args(parser: argparse.ArgumentParser):
    """Add quality preset arguments to parser."""
    parser.add_argument(
        "--quality", "-q",
        choices=["draft", "standard", "high", "ultra"],
        default="standard",
        help="Quality preset (default: standard)"
    )


def add_prompt_args(parser: argparse.ArgumentParser):
    """Add prompt arguments to parser."""
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt", "-p",
        help="Text prompt for segmentation (use . to separate multiple)"
    )
    prompt_group.add_argument(
        "--box", "-b",
        help="Box prompt: x1,y1,x2,y2"
    )
    prompt_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive box selection"
    )
    prompt_group.add_argument(
        "--interactive-points",
        action="store_true",
        help="Interactive point marking (LEFT=include, RIGHT=exclude)"
    )


def add_stage_control_args(parser: argparse.ArgumentParser):
    """Add stage control arguments to parser."""
    parser.add_argument(
        "--skip-sam",
        action="store_true",
        help="Skip SAM segmentation (use existing output)"
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth estimation"
    )
    parser.add_argument(
        "--skip-vitmatte",
        action="store_true",
        help="Skip ViTMatte refinement"
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip matte combination"
    )
    parser.add_argument(
        "--with-hair",
        action="store_true",
        help="Enable hair refinement stage"
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate files"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear output directories before running"
    )


def setup_cli_logging(verbose: bool = False, debug: bool = False):
    """Setup logging for CLI."""
    import logging

    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    return logging.getLogger("AutoRoto")
