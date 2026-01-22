#!/usr/bin/env python
"""
AUTO-ROTO FULL PIPELINE
=======================

Complete workflow that chains:
    1. SAM2 segmentation (auto_roto.py)
    2. Depth Anything V2 refinement (depth_refine.py)

Since both models can't fit in VRAM simultaneously, this script:
    - Runs SAM2 first, releases GPU memory
    - Runs Depth Anything V2 refinement second

USAGE:
    # Full pipeline
    python full_pipeline.py --input video.mp4 --prompt "person" --output ./output

    # Skip depth refinement (SAM2 only)
    python full_pipeline.py --input video.mp4 --prompt "person" --output ./output --no-depth

    # Fast mode (smaller models)
    python full_pipeline.py --input video.mp4 --prompt "person" --output ./output --fast
"""

import os
import sys
import argparse
import subprocess
import shutil
import gc
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("FullPipeline")


def find_script(name: str) -> Path:
    """Find a script in the same directory."""
    script_dir = Path(__file__).parent
    script_path = script_dir / name
    
    if script_path.exists():
        return script_path
    
    raise FileNotFoundError(f"Cannot find {name} in {script_dir}")


def clear_gpu_memory():
    """Clear GPU memory between stages."""
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except ImportError:
        pass
    
    gc.collect()
    logger.info("GPU memory cleared")


def run_stage(cmd: list, stage_name: str) -> bool:
    """Run a pipeline stage and return success status."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE: {stage_name}")
    logger.info(f"{'='*60}")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )
        
        duration = time.time() - start_time
        logger.info(f"✓ {stage_name} completed in {duration:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {stage_name} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AUTO-ROTO Full Pipeline (SAM2 + Depth Refinement)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Full quality pipeline
  %(prog)s --input video.mp4 --prompt "person" --output ./output

  # Fast mode (smaller models, still good quality)
  %(prog)s --input video.mp4 --prompt "person" --output ./output --fast

  # SAM2 only (no depth refinement)
  %(prog)s --input video.mp4 --prompt "person" --output ./output --no-depth

  # Interactive selection
  %(prog)s --input video.mp4 --interactive --output ./output
        """
    )
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True,
                       help="Input video or image sequence")
    parser.add_argument("--output", "-o", default="./output",
                       help="Output directory")
    
    # Prompt type
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", "-p",
                             help="Text prompt for detection")
    prompt_group.add_argument("--box", "-b",
                             help="Box prompt: x1,y1,x2,y2")
    prompt_group.add_argument("--interactive", action="store_true",
                             help="Interactive selection")
    
    # Quality presets
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode: smaller models, lower quality")
    parser.add_argument("--quality", choices=["draft", "standard", "high"],
                       default="standard",
                       help="Quality preset")
    
    # Pipeline options
    parser.add_argument("--no-depth", action="store_true",
                       help="Skip depth refinement (SAM2 only)")
    parser.add_argument("--depth-only", action="store_true",
                       help="Run depth refinement only (assumes SAM2 already done)")
    
    # Advanced options
    parser.add_argument("--sam-model",
                       choices=["tiny", "small", "base_plus", "large"],
                       help="Override SAM2 model size")
    parser.add_argument("--depth-model",
                       choices=["small", "base", "large"],
                       help="Override Depth model size")
    parser.add_argument("--format", default="exr",
                       choices=["exr", "png", "tiff"])
    parser.add_argument("--bit-depth", type=int, default=16,
                       choices=[8, 16, 32])
    
    # Debug
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true",
                       help="Keep intermediate files")
    
    args = parser.parse_args()
    
    # Determine model sizes based on quality preset
    if args.fast:
        sam_model = args.sam_model or "small"
        depth_model = args.depth_model or "small"
    elif args.quality == "draft":
        sam_model = args.sam_model or "tiny"
        depth_model = args.depth_model or "small"
    elif args.quality == "high":
        sam_model = args.sam_model or "large"
        depth_model = args.depth_model or "large"
    else:  # standard
        sam_model = args.sam_model or "base_plus"
        depth_model = args.depth_model or "base"
    
    # Setup directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sam_output = output_dir / "sam_output"
    final_output = output_dir / "final"
    
    # Find scripts
    auto_roto_script = find_script("auto_roto.py")
    depth_refine_script = find_script("depth_refine.py")
    
    pipeline_start = time.time()
    
    # =========================================================================
    # STAGE 1: SAM2 Segmentation
    # =========================================================================
    if not args.depth_only:
        sam_cmd = [
            sys.executable,
            str(auto_roto_script),
            "--input", args.input,
            "--output", str(sam_output),
            "--sam-model", sam_model,
            "--format", args.format,
            "--bit-depth", str(args.bit_depth),
        ]
        
        if args.prompt:
            sam_cmd.extend(["--prompt", args.prompt])
        elif args.box:
            sam_cmd.extend(["--box", args.box])
        elif args.interactive:
            sam_cmd.append("--interactive")
        
        if args.verbose:
            sam_cmd.append("--verbose")
        
        # Run SAM2 without alpha refinement if we're doing depth refinement
        if not args.no_depth:
            sam_cmd.append("--no-refine")  # Let depth refinement handle it
        
        success = run_stage(sam_cmd, "SAM2 Segmentation")
        
        if not success:
            logger.error("Pipeline failed at SAM2 stage")
            sys.exit(1)
        
        # Clear GPU memory before next stage
        clear_gpu_memory()
    
    # =========================================================================
    # STAGE 2: Depth-Guided Refinement
    # =========================================================================
    if not args.no_depth:
        # Prepare depth refinement inputs
        alpha_dir = sam_output / "alpha"
        
        if not alpha_dir.exists():
            logger.error(f"Alpha directory not found: {alpha_dir}")
            logger.error("Run SAM2 stage first or check output directory")
            sys.exit(1)
        
        depth_cmd = [
            sys.executable,
            str(depth_refine_script),
            "--alpha", str(alpha_dir),
            "--video", args.input,  # Extract frames from input video
            "--output", str(final_output),
            "--depth-model", depth_model,
            "--format", args.format,
            "--bit-depth", str(args.bit_depth),
        ]
        
        if args.verbose:
            depth_cmd.append("--verbose")
            depth_cmd.append("--debug")
        
        success = run_stage(depth_cmd, "Depth-Guided Refinement")
        
        if not success:
            logger.error("Pipeline failed at Depth Refinement stage")
            sys.exit(1)
    else:
        # No depth refinement, just copy SAM output to final
        final_output.mkdir(parents=True, exist_ok=True)
        
        for subdir in ["alpha", "rgba", "preview"]:
            src = sam_output / subdir
            dst = final_output / subdir
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    if not args.keep_intermediate and not args.depth_only:
        # Remove intermediate SAM output if depth refinement was done
        if not args.no_depth and sam_output.exists():
            # Keep preview from SAM output
            sam_preview = sam_output / "preview"
            if sam_preview.exists():
                dst_preview = final_output / "sam_preview"
                shutil.move(str(sam_preview), str(dst_preview))
            
            # Remove the rest
            shutil.rmtree(sam_output, ignore_errors=True)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - pipeline_start
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Output: {final_output}")
    logger.info("")
    logger.info("Output structure:")
    logger.info(f"  {final_output}/alpha/   - Alpha mattes (for Nuke)")
    
    if (final_output / "rgba").exists():
        logger.info(f"  {final_output}/rgba/    - RGBA images")
    
    if (final_output / "depth").exists():
        logger.info(f"  {final_output}/depth/   - Depth maps")
    
    if (final_output / "preview").exists():
        logger.info(f"  {final_output}/preview/ - Preview images")
    
    logger.info("="*60)


if __name__ == "__main__":
    main()
