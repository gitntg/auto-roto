#!/usr/bin/env python
"""
AUTO-ROTO FULL PIPELINE v5
==========================

Production-grade automatic rotoscoping with professional quality enhancements.

This is the main entry point for v5 which chains:
    1. SAM2 segmentation (auto_roto.py)
    2. Depth Anything V2 refinement (depth_refine.py)
    3. Edge refinement (edge_refine.py) - NEW
    4. Temporal coherence (temporal_smooth.py) - NEW
    5. Matte combination (matte_combine.py) - NEW
    6. (Optional) Hair refinement (hair_refine.py)

NEW IN V5:
- Professional edge refinement with subpixel precision
- Temporal coherence to prevent flickering
- Multi-layer matte combination
- Color correction and despill
- Proper premultiplied alpha compositing

USAGE:
    # Full pipeline with all enhancements
    python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output

    # Quick mode (SAM2 + edge refinement only)
    python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output --quick

    # High quality with temporal smoothing
    python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output --quality high

    # Process PNG sequence
    python full_pipeline_v5.py --input /path/to/frames/ --prompt "car" --output ./output

Author: AUTO-ROTO v5
License: MIT
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
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("FullPipelineV5")


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    # Input/Output
    input_path: str = ""
    output_dir: str = "./output"

    # Detection
    prompt: str = ""
    box: str = ""
    interactive: bool = False

    # Quality preset
    quality: str = "standard"  # draft, standard, high, ultra

    # Stage control
    skip_sam: bool = False
    skip_depth: bool = False
    skip_vitmatte: bool = False  # ViTMatte alpha refinement
    skip_edge: bool = False
    skip_temporal: bool = False
    skip_combine: bool = False
    skip_hair: bool = True  # Hair refinement optional, off by default

    # ViTMatte settings (adaptive trimap)
    vitmatte_motion_aware: bool = False
    vitmatte_adaptive_base: float = 2.0
    vitmatte_adaptive_max: float = 60.0

    # Model sizes (auto-set by quality preset)
    sam_model: str = ""
    depth_model: str = ""

    # Edge refinement
    edge_softness: float = 1.0
    core_shrink: int = 3
    despill_strength: float = 0.5

    # Temporal smoothing
    temporal_window: int = 5
    keyframe_interval: int = 30

    # Output settings
    output_format: str = "exr"
    bit_depth: int = 16

    # Performance
    device: str = "cuda"

    # Debug
    verbose: bool = False
    keep_intermediate: bool = False


def get_quality_preset(quality: str) -> Dict[str, Any]:
    """Get model sizes and settings for quality preset."""
    presets = {
        'draft': {
            'sam_model': 'tiny',
            'depth_model': 'small',
            'temporal_window': 3,
            'edge_softness': 0.5,
        },
        'standard': {
            'sam_model': 'base_plus',
            'depth_model': 'base',
            'temporal_window': 5,
            'edge_softness': 1.0,
        },
        'high': {
            'sam_model': 'large',
            'depth_model': 'large',  # DA3Mono-Large preserves hair detail (not nested!)
            'temporal_window': 7,
            'edge_softness': 1.5,
        },
        'ultra': {
            'sam_model': 'large',
            'depth_model': 'large',  # DA3Mono-Large preserves hair detail (not nested!)
            'temporal_window': 9,
            'edge_softness': 2.0,
        }
    }
    return presets.get(quality, presets['standard'])


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
    logger.debug("GPU memory cleared")


def run_stage(cmd: list, stage_name: str, verbose: bool = False) -> bool:
    """Run a pipeline stage and return success status."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE: {stage_name}")
    logger.info(f"{'='*60}")

    if verbose:
        logger.info(f"Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=not verbose
        )

        duration = time.time() - start_time
        logger.info(f"  {stage_name} completed in {duration:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"  {stage_name} failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"  Error: {e.stderr[:500]}")
        return False


def run_python_stage(
    script_path: Path,
    args: List[str],
    stage_name: str,
    verbose: bool = False
) -> bool:
    """Run a Python script as a pipeline stage."""
    cmd = [sys.executable, str(script_path)] + args
    return run_stage(cmd, stage_name, verbose)


def run_pipeline(config: PipelineConfig):
    """Run the full v5 pipeline."""

    # Apply quality preset
    preset = get_quality_preset(config.quality)
    if not config.sam_model:
        config.sam_model = preset['sam_model']
    if not config.depth_model:
        config.depth_model = preset['depth_model']
    if config.temporal_window == 5:  # Default
        config.temporal_window = preset['temporal_window']
    if config.edge_softness == 1.0:  # Default
        config.edge_softness = preset['edge_softness']

    # Setup directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate directories
    sam_output = output_dir / "01_sam_output"
    depth_output = output_dir / "02_depth_output"
    vitmatte_output = output_dir / "03_vitmatte_output"
    edge_output = output_dir / "04_edge_output"
    temporal_output = output_dir / "05_temporal_output"
    combine_output = output_dir / "06_combine_output"
    final_output = output_dir / "final"

    # Find scripts
    script_dir = Path(__file__).parent

    pipeline_start = time.time()

    logger.info("="*60)
    logger.info("AUTO-ROTO v5 PIPELINE")
    logger.info("="*60)
    logger.info(f"Input: {config.input_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Quality: {config.quality}")
    logger.info(f"SAM Model: {config.sam_model}")
    logger.info(f"Depth Model: {config.depth_model}")
    logger.info("="*60)

    # =========================================================================
    # STAGE 1: SAM2 Segmentation
    # =========================================================================
    if not config.skip_sam:
        auto_roto_script = find_script("auto_roto.py")

        sam_args = [
            "--input", config.input_path,
            "--output", str(sam_output),
            "--sam-model", config.sam_model,
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
            "--no-refine",  # We'll do our own refinement
        ]

        if config.prompt:
            sam_args.extend(["--prompt", config.prompt])
        elif config.box:
            sam_args.extend(["--box", config.box])
        elif config.interactive:
            sam_args.append("--interactive")

        if config.verbose:
            sam_args.append("--verbose")

        success = run_python_stage(
            auto_roto_script, sam_args,
            "SAM2 Segmentation", config.verbose
        )

        if not success:
            logger.error("Pipeline failed at SAM2 stage")
            return False

        clear_gpu_memory()
    else:
        logger.info("Skipping SAM2 (--skip-sam)")

    # =========================================================================
    # STAGE 2: Depth Refinement
    # =========================================================================
    if not config.skip_depth:
        depth_script = find_script("depth_refine.py")

        # Determine alpha source
        alpha_source = sam_output / "alpha"
        if not alpha_source.exists():
            logger.warning(f"Alpha dir not found: {alpha_source}")
            alpha_source = sam_output

        depth_args = [
            "--alpha", str(alpha_source),
            "--frames", config.input_path,
            "--output", str(depth_output),
            "--depth-model", config.depth_model,
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
        ]

        if config.verbose:
            depth_args.extend(["--verbose", "--debug"])

        success = run_python_stage(
            depth_script, depth_args,
            "Depth Refinement", config.verbose
        )

        if not success:
            logger.warning("Depth refinement failed, continuing with SAM output")
            depth_output = sam_output

        clear_gpu_memory()
    else:
        logger.info("Skipping Depth Refinement (--skip-depth)")
        depth_output = sam_output

    # =========================================================================
    # STAGE 3: ViTMatte Alpha Refinement (Adaptive Trimap)
    # =========================================================================
    if not config.skip_vitmatte:
        vitmatte_script = find_script("vitmatte_refine.py")

        # Determine sources
        sam_alpha = sam_output / "alpha"
        if not sam_alpha.exists():
            sam_alpha = sam_output

        depth_maps = depth_output / "depth"
        if not depth_maps.exists():
            depth_maps = depth_output

        vitmatte_args = [
            "--sam-mask", str(sam_alpha),
            "--depth", str(depth_maps),
            "--frames", config.input_path,
            "--output", str(vitmatte_output),
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
            "--core-erosion", "10",
            "--adaptive-base", str(config.vitmatte_adaptive_base),
            "--adaptive-max", str(config.vitmatte_adaptive_max),
            "--save-trimap",
        ]

        if config.vitmatte_motion_aware:
            vitmatte_args.append("--motion-aware")

        if config.verbose:
            vitmatte_args.append("--verbose")

        success = run_python_stage(
            vitmatte_script, vitmatte_args,
            "ViTMatte Alpha Refinement", config.verbose
        )

        if not success:
            logger.warning("ViTMatte refinement failed, continuing with depth output")
            vitmatte_output = depth_output

        clear_gpu_memory()
    else:
        logger.info("Skipping ViTMatte (--skip-vitmatte)")
        vitmatte_output = depth_output

    # =========================================================================
    # STAGE 4: Edge Refinement
    # =========================================================================
    if not config.skip_edge:
        edge_script = find_script("edge_refine.py")

        # Determine alpha source (now from ViTMatte)
        alpha_source = vitmatte_output / "alpha"
        if not alpha_source.exists():
            alpha_source = vitmatte_output

        edge_args = [
            "--alpha", str(alpha_source),
            "--output", str(edge_output),
            "--frames", config.input_path,
            "--softness", str(config.edge_softness),
            "--core-shrink", str(config.core_shrink),
            "--despill", str(config.despill_strength),
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
        ]

        if config.verbose:
            edge_args.append("--verbose")

        success = run_python_stage(
            edge_script, edge_args,
            "Edge Refinement", config.verbose
        )

        if not success:
            logger.warning("Edge refinement failed, continuing with previous output")
            edge_output = vitmatte_output

        clear_gpu_memory()  # Clean up after Edge Refinement
    else:
        logger.info("Skipping Edge Refinement (--skip-edge)")
        edge_output = vitmatte_output

    # =========================================================================
    # STAGE 5: Temporal Smoothing
    # =========================================================================
    if not config.skip_temporal:
        temporal_script = find_script("temporal_smooth.py")

        # Determine alpha source
        alpha_source = edge_output / "alpha"
        if not alpha_source.exists():
            alpha_source = edge_output

        temporal_args = [
            "--alpha", str(alpha_source),
            "--output", str(temporal_output),
            "--frames", config.input_path,
            "--window", str(config.temporal_window),
            "--keyframe-interval", str(config.keyframe_interval),
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
        ]

        if config.verbose:
            temporal_args.append("--verbose")

        success = run_python_stage(
            temporal_script, temporal_args,
            "Temporal Smoothing", config.verbose
        )

        if not success:
            logger.warning("Temporal smoothing failed, continuing with previous output")
            temporal_output = edge_output

        clear_gpu_memory()  # Clean up after Temporal Smoothing
    else:
        logger.info("Skipping Temporal Smoothing (--skip-temporal)")
        temporal_output = edge_output

    # =========================================================================
    # STAGE 6: Matte Combination
    # =========================================================================
    if not config.skip_combine:
        combine_script = find_script("matte_combine.py")

        # Determine alpha source
        alpha_source = temporal_output / "alpha"
        if not alpha_source.exists():
            alpha_source = temporal_output

        combine_args = [
            "--alpha", str(alpha_source),
            "--output", str(combine_output),
            "--frames", config.input_path,
            "--core-erosion", str(config.core_shrink),
            "--despill", str(config.despill_strength),
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
        ]

        if config.verbose:
            combine_args.append("--verbose")

        success = run_python_stage(
            combine_script, combine_args,
            "Matte Combination", config.verbose
        )

        if not success:
            logger.warning("Matte combination failed, using previous output")
            combine_output = temporal_output

        clear_gpu_memory()  # Clean up after Matte Combination
    else:
        logger.info("Skipping Matte Combination (--skip-combine)")
        combine_output = temporal_output

    # =========================================================================
    # STAGE 7: Hair Refinement (Using Adaptive ViTMatte)
    # =========================================================================
    if not config.skip_hair:
        # POINT TO THE NEW SCRIPT
        hair_script = find_script("vitmatte_refine.py")

        # Determine inputs
        # 1. We need the original SAM mask (best source for core)
        sam_alpha = sam_output / "alpha"
        if not sam_alpha.exists():
            # Fallback to whatever alpha we have currently
            sam_alpha = alpha_source

        # 2. We need depth maps
        depth_maps = depth_output / "depth"
        if not depth_maps.exists():
            logger.warning("No depth maps found for hair refinement!")
            # In a real fix, you might want to skip or fail here

        hair_output = output_dir / "07_hair_output"

        # USE THE NEW ARGUMENTS
        hair_args = [
            "--sam-mask", str(sam_alpha),
            "--depth", str(depth_maps),
            "--frames", config.input_path,
            "--output", str(hair_output),
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
            "--adaptive-base", "2.0",
            "--adaptive-max", "60.0",
            "--motion-aware", # Enable the motion logic
            "--save-trimap",
        ]

        # Only run if we have frames (vitmatte needs frames folder, not video file)
        # Note: If input is a video file, you might need to point to the
        # temp frames extracted in Stage 1/2 if they exist, or extract them.
        # Assuming input_path is a sequence or we have temp frames:

        success = run_python_stage(
            hair_script, hair_args,
            "Hair Refinement (Adaptive)", config.verbose
        )

        if success:
            combine_output = hair_output

        clear_gpu_memory()
    else:
        logger.debug("Skipping Hair Refinement (default off)")

    # =========================================================================
    # FINAL: Copy to output
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("FINALIZING OUTPUT")
    logger.info("="*60)

    final_output.mkdir(parents=True, exist_ok=True)

    # Find the last successful output
    final_source = combine_output

    # Copy final results
    for subdir in ["alpha", "rgb", "rgba", "preview"]:
        src = final_source / subdir
        if src.exists():
            dst = final_output / subdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info(f"  Copied {subdir}/ to final output")

    # Cleanup intermediate if not keeping
    if not config.keep_intermediate:
        logger.info("Cleaning up intermediate files...")
        for intermediate in [sam_output, depth_output, vitmatte_output, edge_output,
                           temporal_output, combine_output]:
            if intermediate.exists() and intermediate != final_output:
                shutil.rmtree(intermediate, ignore_errors=True)

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

    for subdir in ["alpha", "rgb", "rgba", "preview", "depth"]:
        subpath = final_output / subdir
        if subpath.exists():
            count = len(list(subpath.glob("*")))
            logger.info(f"  {subdir}/  ({count} files)")

    logger.info("="*60)

    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AUTO-ROTO v5: Production-grade automatic rotoscoping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Full quality pipeline
  %(prog)s --input video.mp4 --prompt "person" --output ./output

  # Process PNG sequence with high quality
  %(prog)s --input /path/to/frames/ --prompt "car" --output ./output --quality high

  # Quick mode (SAM2 + edge only)
  %(prog)s --input video.mp4 --prompt "person" --output ./output --quality draft --skip-temporal

  # Interactive selection
  %(prog)s --input video.mp4 --interactive --output ./output

QUALITY PRESETS:
  draft    - Fastest, tiny/small models
  standard - Balanced quality and speed (default)
  high     - Best quality, large models
  ultra    - Maximum quality, longer temporal window
        """
    )

    # Input/Output
    parser.add_argument("--input", "-i", required=True,
                       help="Input video or image sequence directory")
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

    # Quality
    parser.add_argument("--quality", "-q",
                       choices=["draft", "standard", "high", "ultra"],
                       default="standard",
                       help="Quality preset (default: standard)")

    # Stage control
    parser.add_argument("--skip-sam", action="store_true",
                       help="Skip SAM2 (use existing alpha)")
    parser.add_argument("--skip-depth", action="store_true",
                       help="Skip depth refinement")
    parser.add_argument("--skip-vitmatte", action="store_true",
                       help="Skip ViTMatte alpha refinement")
    parser.add_argument("--skip-edge", action="store_true",
                       help="Skip edge refinement")
    parser.add_argument("--skip-temporal", action="store_true",
                       help="Skip temporal smoothing")
    parser.add_argument("--skip-combine", action="store_true",
                       help="Skip matte combination")
    parser.add_argument("--with-hair", action="store_true",
                       help="Enable hair refinement (off by default)")

    # Advanced
    parser.add_argument("--sam-model",
                       choices=["tiny", "small", "base_plus", "large"],
                       help="Override SAM2 model size")
    parser.add_argument("--depth-model",
                       choices=["small", "base", "large", "nested-base", "nested-large"],
                       help="Override Depth model (nested-large best for hair detail)")

    # ViTMatte settings (adaptive trimap)
    parser.add_argument("--vitmatte-motion", action="store_true",
                       help="Enable motion-aware trimap for ViTMatte")
    parser.add_argument("--vitmatte-base", type=float, default=2.0,
                       help="Min unknown width for smooth regions (default: 2)")
    parser.add_argument("--vitmatte-max", type=float, default=60.0,
                       help="Max unknown width for complex regions (default: 60)")

    # Edge settings
    parser.add_argument("--edge-softness", type=float, default=1.0,
                       help="Edge softness (default: 1.0)")
    parser.add_argument("--core-shrink", type=int, default=3,
                       help="Core shrink pixels (default: 3)")
    parser.add_argument("--despill", type=float, default=0.5,
                       help="Despill strength (default: 0.5)")

    # Temporal settings
    parser.add_argument("--temporal-window", type=int, default=5,
                       help="Temporal window size (default: 5)")
    parser.add_argument("--keyframe-interval", type=int, default=30,
                       help="Keyframe interval (default: 30)")

    # Output format
    parser.add_argument("--format", default="exr",
                       choices=["exr", "png", "tiff"])
    parser.add_argument("--bit-depth", type=int, default=16,
                       choices=[8, 16, 32])

    # Debug
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--keep-intermediate", action="store_true",
                       help="Keep intermediate files")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output,
        prompt=args.prompt or "",
        box=args.box or "",
        interactive=args.interactive,
        quality=args.quality,
        skip_sam=args.skip_sam,
        skip_depth=args.skip_depth,
        skip_vitmatte=args.skip_vitmatte,
        skip_edge=args.skip_edge,
        skip_temporal=args.skip_temporal,
        skip_combine=args.skip_combine,
        skip_hair=not args.with_hair,
        sam_model=args.sam_model or "",
        depth_model=args.depth_model or "",
        vitmatte_motion_aware=args.vitmatte_motion,
        vitmatte_adaptive_base=args.vitmatte_base,
        vitmatte_adaptive_max=args.vitmatte_max,
        edge_softness=args.edge_softness,
        core_shrink=args.core_shrink,
        despill_strength=args.despill,
        temporal_window=args.temporal_window,
        keyframe_interval=args.keyframe_interval,
        output_format=args.format,
        bit_depth=args.bit_depth,
        verbose=args.verbose,
        keep_intermediate=args.keep_intermediate,
    )

    success = run_pipeline(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
