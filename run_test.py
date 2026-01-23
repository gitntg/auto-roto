#!/usr/bin/env python3
"""
AUTO-ROTO Test Runner
=====================

Simple test script with auto-versioning output directories.

Usage:
    python run_test.py                      # Run with defaults
    python run_test.py --input my_video.mp4 # Custom input
    python run_test.py --full               # Run full pipeline (all stages)
    python run_test.py --sam-only           # Run SAM2 only (fastest)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def get_next_version(output_base: Path, name: str) -> int:
    """Find the next available version number for a given name."""
    if not output_base.exists():
        return 1

    # Find all existing directories matching pattern: name_v#
    pattern = re.compile(rf"^{re.escape(name)}_v(\d+)$")
    max_version = 0

    for item in output_base.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)

    return max_version + 1


def get_input_name(input_path: str) -> str:
    """Extract a clean name from the input path."""
    path = Path(input_path)

    # If it's a directory, use the directory name
    if path.is_dir():
        return path.name

    # If it's a file, use the stem (filename without extension)
    return path.stem


def main():
    parser = argparse.ArgumentParser(description="AUTO-ROTO Test Runner")
    parser.add_argument("--input", "-i", default="test_input",
                        help="Input video/image sequence (default: test_input)")
    parser.add_argument("--prompt", "-p", default="person",
                        help="Detection prompt (default: person)")
    parser.add_argument("--quality", "-q", default="standard",
                        choices=["draft", "standard", "high", "ultra"],
                        help="Quality preset (default: standard)")
    parser.add_argument("--full", action="store_true",
                        help="Run full pipeline with all stages")
    parser.add_argument("--sam-only", action="store_true",
                        help="Run SAM2 only (skip all refinement)")
    parser.add_argument("--keep-intermediate", "-k", action="store_true",
                        help="Keep intermediate stage outputs")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show command without running")

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    output_base = script_dir / "output"
    output_base.mkdir(exist_ok=True)

    # Generate versioned output directory name
    input_name = get_input_name(args.input)
    version = get_next_version(output_base, input_name)
    output_dir = output_base / f"{input_name}_v{version}"

    print("=" * 60)
    print("AUTO-ROTO Test Runner")
    print("=" * 60)
    print(f"Input:   {args.input}")
    print(f"Output:  {output_dir}")
    print(f"Quality: {args.quality}")
    print(f"Prompt:  {args.prompt}")
    print("=" * 60)

    # Build command
    cmd = [
        sys.executable,
        str(script_dir / "full_pipeline_v5.py"),
        "--input", args.input,
        "--output", str(output_dir),
        "--prompt", args.prompt,
        "--quality", args.quality,
    ]

    # Add skip flags for sam-only mode
    if args.sam_only:
        cmd.extend([
            "--skip-depth",
            "--skip-vitmatte",
            "--skip-edge",
            "--skip-temporal",
            "--skip-combine",
        ])
        print("Mode:    SAM2 only (skipping refinement stages)")
    elif not args.full:
        # Default: skip some heavy stages for faster testing
        cmd.extend([
            "--skip-combine",  # Combine often causes issues
        ])
        print("Mode:    Standard (skipping combine stage)")
    else:
        print("Mode:    Full pipeline (all stages)")

    if args.keep_intermediate:
        cmd.append("--keep-intermediate")

    if args.verbose:
        cmd.append("--verbose")

    print("=" * 60)

    # Set environment to disable torch.compile (Windows compatibility)
    env = os.environ.copy()
    env["TORCHDYNAMO_DISABLE"] = "1"

    if args.dry_run:
        print("\nDry run - would execute:")
        print(" ".join(cmd))
        print(f"\nWith TORCHDYNAMO_DISABLE=1")
        return 0

    print("\nStarting pipeline...\n")

    # Run the pipeline
    try:
        result = subprocess.run(cmd, env=env, cwd=str(script_dir))

        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"Output saved to: {output_dir}")

            # List output contents
            if output_dir.exists():
                final_dir = output_dir / "final"
                if final_dir.exists():
                    print("\nFinal output contents:")
                    for subdir in final_dir.iterdir():
                        if subdir.is_dir():
                            count = len(list(subdir.glob("*")))
                            print(f"  {subdir.name}/  ({count} files)")
        else:
            print(f"\nPipeline exited with code: {result.returncode}")
            return result.returncode

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
