#!/usr/bin/env python
"""
AUTO-ROTO FULL PIPELINE v5
==========================

Production-grade automatic rotoscoping with professional quality enhancements.

This is the main entry point for v5 which chains:
    1. SAM3 segmentation with built-in text prompting (auto_roto.py)
    2. Depth Anything V2 refinement (depth_refine.py)
    3. Edge refinement (edge_refine.py)
    4. Temporal coherence (temporal_smooth.py)
    5. Matte combination (matte_combine.py)
    6. (Optional) Hair refinement (hair_refine.py)

SAM3 has native Promptable Concept Segmentation (PCS) supporting 270k+ concepts,
eliminating the need for separate detection (GroundingDINO).

NEW IN V5:
- SAM3 with built-in text prompting (no GroundingDINO needed)
- Professional edge refinement with subpixel precision
- Temporal coherence to prevent flickering
- Multi-layer matte combination
- Color correction and despill
- Proper premultiplied alpha compositing

USAGE:
    # Standard quality (balanced)
    python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output

    # Ultra quality + hair detail (max quality, slower)
    python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output --quality ultra --with-hair --depth-res 2048 --depth-method lower --depth-percentiles 0 100 --use-depth-confidence --vitmatte-motion

    # MatAnyone2 refinement (Stage 3 replacement)
    python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output --refiner mam2

    # Fast preview (draft quality, no temporal)
    python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output --quality draft --skip-temporal

    # Process PNG sequence (ultra quality)
    python full_pipeline_v5.py --input /path/to/frames/ --prompt "car" --output ./output --quality ultra --with-hair

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
import platform
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


# Environment verification - ensures correct conda environment
REQUIRED_ENV = "autoroto"

def _check_conda_environment():
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

# Run environment check immediately on import
_check_conda_environment()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("FullPipelineV5")


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline using SAM3."""

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

    # Alpha refinement selection
    refiner: str = "vitmatte"  # vitmatte or mam2

    # ViTMatte settings (adaptive trimap)
    vitmatte_motion_aware: bool = False
    vitmatte_adaptive_base: float = 2.0
    vitmatte_adaptive_max: float = 60.0

    # MatAnyone2 settings
    mam2_repo: str = "MatAnyone2"
    mam2_checkpoint: str = "./checkpoints/mam2.pth"

    # Hair polish settings (applied to edge/unknown regions)
    # None = use quality preset value; explicit values override preset
    hair_gamma: float = None          # Lower = more visible strands (0.5-1.0)
    hair_black_point: float = None    # Lower = preserve faint tips (0.0-0.1)
    hair_gain: float = None           # Higher = more solid core (1.0-1.5)
    hair_polish_enabled: bool = True  # Allow disabling entirely

    # Guided Filter settings
    guided_filter_radius: int = None      # None = use quality preset
    guided_filter_eps: float = None       # None = use quality preset

    # SAM3 inference settings
    sam_imgsz: int = 0              # Processing resolution (0 = auto from input, max 2048)
    sam_conf: float = 0.25          # Confidence threshold (0.0-1.0, lower = more detections)
    sam_retina_masks: bool = True   # High-resolution mask output
    sam_max_det: int = 100          # Maximum detections per frame

    # Depth model (auto-set by quality preset)
    depth_model: str = ""

    # DA3 Depth Sensitivity Settings (NEW)
    depth_process_res: int = None         # None = auto, or explicit value like 1024, 2048
    depth_process_method: str = "upper"   # "upper" or "lower" bound resize
    depth_norm_percentiles: tuple = (2.0, 98.0)  # Normalization percentiles
    use_depth_confidence: bool = False     # Use DA3 confidence maps

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
    no_compile: bool = False
    force_compile: bool = False  # Force torch.compile even on unsupported platforms

    # Debug
    verbose: bool = False
    keep_intermediate: bool = False


def get_quality_preset(quality: str) -> Dict[str, Any]:
    """Get model sizes and settings for quality preset.

    Note: SAM3 has a single model architecture, so no sam_model setting needed.
    """
    presets = {
        'draft': {
            'depth_model': 'small',
            'temporal_window': 3,
            'edge_softness': 0.5,
            # DA3 settings: fast, basic detail
            'depth_process_res': None,  # auto (image size)
            'depth_process_method': 'upper',
            'depth_norm_percentiles': (2.0, 98.0),
            # Hair polish: more aggressive for speed
            'hair_gamma': 0.7,
            'hair_black_point': 0.03,
            'hair_gain': 1.2,
            # Guided Filter: fast, general purpose
            'guided_filter_radius': 4,
            'guided_filter_eps': 1e-4,
        },
        'standard': {
            'depth_model': 'base',
            'temporal_window': 5,
            'edge_softness': 1.0,
            # DA3 settings: balanced - explicit 1536 for reasonable quality/speed
            'depth_process_res': 1536,  # Explicit resolution (was None/auto)
            'depth_process_method': 'upper',
            'depth_norm_percentiles': (2.0, 98.0),
            # Hair polish: balanced
            'hair_gamma': 0.8,
            'hair_black_point': 0.02,
            'hair_gain': 1.1,
            # Guided Filter: balanced
            'guided_filter_radius': 4,
            'guided_filter_eps': 1e-5,
        },
        'high': {
            'depth_model': 'large',  # DA3Mono-Large preserves hair detail (not nested!)
            'temporal_window': 7,
            'edge_softness': 1.5,
            # DA3 settings: optimized for fine detail - explicit 2048 for quality
            'depth_process_res': 2048,  # Explicit high resolution (was None/auto)
            'depth_process_method': 'lower',  # process_res is min dimension
            'depth_norm_percentiles': (1.0, 99.0),  # Wider range preserves more detail
            # Hair polish: preserve detail
            'hair_gamma': 0.9,
            'hair_black_point': 0.01,
            'hair_gain': 1.05,
            # Guided Filter: better hair strand separation
            'guided_filter_radius': 2,
            'guided_filter_eps': 1e-5,
        },
        'ultra': {
            'depth_model': 'large',  # DA3Mono-Large preserves hair detail (not nested!)
            'temporal_window': 9,
            'edge_softness': 2.0,
            # DA3 settings: maximum detail capture
            'depth_process_res': 2048,  # Explicit high resolution
            'depth_process_method': 'lower',  # process_res is min dimension
            'depth_norm_percentiles': (0.0, 100.0),  # Full range to preserve depth variation
            # Hair polish: no adjustment (preserve all detail)
            'hair_gamma': 1.0,
            'hair_black_point': 0.0,
            'hair_gain': 1.0,
            # Guided Filter: maximum individual strand definition
            'guided_filter_radius': 1,
            'guided_filter_eps': 1e-6,
        }
    }
    return presets.get(quality, presets['standard'])


def should_disable_compile() -> bool:
    """Check if torch.compile should be disabled by default.

    torch.compile has known issues on Windows (Dynamo/Inductor incompatibility).
    Returns True if running on Windows.
    """
    if platform.system() == "Windows":
        return True
    return False


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


def resolve_frames_dir(
    input_path: str,
    output_dir: Path,
    logger: logging.Logger
) -> Tuple[Optional[Path], bool]:
    """Resolve or extract RGB frames for stages that require frame sequences."""
    input_path = Path(input_path)

    if input_path.is_dir():
        return input_path, False

    image_exts = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpx"}
    if input_path.is_file() and input_path.suffix.lower() in image_exts:
        return input_path.parent, False

    if "#" in str(input_path) or "*" in str(input_path):
        if input_path.parent.exists():
            return input_path.parent, False

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"}
    if input_path.is_file() and input_path.suffix.lower() in video_exts:
        frames_dir = output_dir / "00_frames"

        if frames_dir.exists():
            existing = sorted(list(frames_dir.glob("*.png")) +
                              list(frames_dir.glob("*.jpg")) +
                              list(frames_dir.glob("*.jpeg")))
            if existing:
                logger.info(f"Using existing extracted frames: {frames_dir}")
                return frames_dir, True
            shutil.rmtree(frames_dir, ignore_errors=True)

        frames_dir.mkdir(parents=True, exist_ok=True)

        import cv2
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video for frame extraction: {input_path}")
            return None, True

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
            logger.error(f"No frames extracted from video: {input_path}")
            return None, True

        logger.info(f"Extracted {idx} frames to: {frames_dir}")
        return frames_dir, True

    return None, False


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


def run_sam3_stage(config: PipelineConfig, output_dir: Path) -> bool:
    """
    Run SAM3 segmentation with built-in text prompting.

    SAM3 doesn't need Grounding DINO - it has native text understanding.

    Args:
        config: Pipeline configuration
        output_dir: Output directory for SAM3 results

    Returns:
        True if successful, False otherwise
    """
    import cv2
    import numpy as np

    logger.info("\n" + "="*60)
    logger.info("STAGE: SAM3 Segmentation (Built-in Text Prompting)")
    logger.info("="*60)

    start_time = time.time()

    try:
        # Import SAM3 from auto_roto
        sys.path.insert(0, str(Path(__file__).parent))
        from auto_roto import SAM3Segmenter, VideoReader, FrameWriter

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir = output_dir / "alpha"
        preview_dir = output_dir / "preview"
        alpha_dir.mkdir(exist_ok=True)
        preview_dir.mkdir(exist_ok=True)

        # Parse text prompts (support multiple via "." separator like SAM2 mode)
        text_prompts = config.prompt.split(".") if config.prompt else ["person"]
        logger.info(f"Text prompts: {text_prompts}")

        # Initialize SAM3 with inference settings
        sam3 = SAM3Segmenter(
            model_path="sam3.pt",
            device=config.device,
            logger=logger,
            imgsz=config.sam_imgsz,
            conf=config.sam_conf,
            retina_masks=config.sam_retina_masks,
            max_det=config.sam_max_det
        )

        # Setup frame writer
        writer = FrameWriter(
            output_dir=str(output_dir),
            prefix="roto",
            format=config.output_format,
            bit_depth=config.bit_depth,
            padding=4,
            logger=logger
        )

        # Check if input is video or frames directory
        input_path = Path(config.input_path)

        if input_path.is_file():
            # Process video file with SAM3 video predictor
            logger.info(f"Processing video: {input_path}")

            frame_idx = 0
            for idx, mask in sam3.segment_video_with_text(str(input_path), text_prompts):
                if mask is not None:
                    # Write alpha
                    writer.write_alpha(mask, idx)
                    frame_idx = idx

            logger.info(f"Processed {frame_idx + 1} frames")

        elif input_path.is_dir():
            # Process frames directory
            logger.info(f"Processing frames directory: {input_path}")

            for idx, mask in sam3.segment_frames_with_text(str(input_path), text_prompts):
                if mask is not None:
                    writer.write_alpha(mask, idx)

        else:
            # Try as video reader (handles sequences)
            reader = VideoReader(str(input_path), logger)
            frames = list(reader)

            for idx, frame in enumerate(frames):
                masks = sam3.segment_image_with_text(frame, text_prompts)
                if masks:
                    combined = np.zeros(frame.shape[:2], dtype=np.float32)
                    for m in masks:
                        combined = np.maximum(combined, m.astype(np.float32))
                    writer.write_alpha(combined, idx)

                if idx % 10 == 0:
                    logger.info(f"  Frame {idx}/{len(frames)}")

        # Cleanup
        sam3.release()

        duration = time.time() - start_time
        logger.info(f"  SAM3 Segmentation completed in {duration:.1f}s")
        return True

    except Exception as e:
        logger.error(f"SAM3 stage failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline(config: PipelineConfig):
    """Run the full v5 pipeline using SAM3."""

    # Apply quality preset
    preset = get_quality_preset(config.quality)
    if not config.depth_model:
        config.depth_model = preset['depth_model']
    if config.temporal_window == 5:  # Default
        config.temporal_window = preset['temporal_window']
    if config.edge_softness == 1.0:  # Default
        config.edge_softness = preset['edge_softness']
    # Apply DA3 depth sensitivity settings from preset (unless overridden)
    if config.depth_process_res is None and 'depth_process_res' in preset:
        config.depth_process_res = preset['depth_process_res']
    if config.depth_process_method == "upper" and 'depth_process_method' in preset:
        config.depth_process_method = preset['depth_process_method']
    if config.depth_norm_percentiles == (2.0, 98.0) and 'depth_norm_percentiles' in preset:
        config.depth_norm_percentiles = preset['depth_norm_percentiles']
    # Apply hair polish settings from preset (unless overridden via CLI)
    if config.hair_gamma is None and 'hair_gamma' in preset:
        config.hair_gamma = preset['hair_gamma']
    if config.hair_black_point is None and 'hair_black_point' in preset:
        config.hair_black_point = preset['hair_black_point']
    if config.hair_gain is None and 'hair_gain' in preset:
        config.hair_gain = preset['hair_gain']
    # Set defaults if still None (shouldn't happen with presets, but be safe)
    if config.hair_gamma is None:
        config.hair_gamma = 0.8
    if config.hair_black_point is None:
        config.hair_black_point = 0.02
    if config.hair_gain is None:
        config.hair_gain = 1.1
    # Apply guided filter settings from preset (unless overridden via CLI)
    if config.guided_filter_radius is None and 'guided_filter_radius' in preset:
        config.guided_filter_radius = preset['guided_filter_radius']
    if config.guided_filter_eps is None and 'guided_filter_eps' in preset:
        config.guided_filter_eps = preset['guided_filter_eps']
    # Set defaults if still None
    if config.guided_filter_radius is None:
        config.guided_filter_radius = 4
    if config.guided_filter_eps is None:
        config.guided_filter_eps = 1e-5

    # Auto-disable torch.compile on Windows unless forced
    if not config.no_compile and not config.force_compile and should_disable_compile():
        logger.info("Windows detected - disabling torch.compile (use --force-compile to override)")
        config.no_compile = True

    # Setup directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir, frames_is_temp = resolve_frames_dir(
        config.input_path,
        output_dir,
        logger
    )
    if frames_is_temp and frames_dir is None:
        logger.error("Failed to prepare frames for processing; aborting pipeline.")
        return False

    frames_arg = str(frames_dir) if frames_dir else config.input_path

    # Intermediate directories
    sam_output = output_dir / "01_sam_output"
    depth_output = output_dir / "02_depth_output"
    refine_stage_dir = "03_mam2_output" if config.refiner == "mam2" else "03_vitmatte_output"
    vitmatte_output = output_dir / refine_stage_dir
    edge_output = output_dir / "04_edge_output"
    temporal_output = output_dir / "05_temporal_output"
    combine_output = output_dir / "06_combine_output"
    final_output = output_dir / "final"

    # Find scripts
    script_dir = Path(__file__).parent

    pipeline_start = time.time()

    logger.info("="*60)
    logger.info("AUTO-ROTO v5 PIPELINE (SAM3)")
    logger.info("="*60)
    logger.info(f"Input: {config.input_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Quality: {config.quality}")
    logger.info("SAM Mode: SAM3 (built-in text prompting)")
    sam_imgsz_str = f"{config.sam_imgsz}" if config.sam_imgsz > 0 else "auto"
    logger.info(f"SAM3: imgsz={sam_imgsz_str}, conf={config.sam_conf}, retina={config.sam_retina_masks}")
    logger.info(f"Depth Model: {config.depth_model}")
    logger.info(f"Refiner: {config.refiner}")
    logger.info("="*60)

    # =========================================================================
    # STAGE 1: SAM3 Segmentation (built-in text prompting)
    # =========================================================================
    if not config.skip_sam:
        # Always use SAM3 with built-in text prompting
        success = run_sam3_stage(config, sam_output)

        if not success:
            logger.error("Pipeline failed at SAM3 stage")
            return False

        clear_gpu_memory()
    else:
        logger.info("Skipping SAM3 (--skip-sam)")

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
            "--frames", frames_arg,
            "--output", str(depth_output),
            "--depth-model", config.depth_model,
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
            # DA3 sensitivity settings
            "--depth-method", config.depth_process_method,
            "--depth-percentiles", str(config.depth_norm_percentiles[0]),
            str(config.depth_norm_percentiles[1]),
        ]

        # Add optional depth_process_res if explicitly set
        if config.depth_process_res is not None:
            depth_args.extend(["--depth-res", str(config.depth_process_res)])

        # Add confidence flag if enabled
        if config.use_depth_confidence:
            depth_args.append("--use-depth-confidence")

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
    # STAGE 3: Alpha Refinement (ViTMatte or MatAnyone2)
    # =========================================================================
    if not config.skip_vitmatte:
        # Determine sources
        sam_alpha = sam_output / "alpha"
        if not sam_alpha.exists():
            sam_alpha = sam_output

        if config.refiner == "mam2":
            refine_script = find_script("mam2_refine.py")
            refine_args = [
                "--sam-mask", str(sam_alpha),
                "--frames", frames_arg,
                "--output", str(vitmatte_output),
                "--checkpoint", config.mam2_checkpoint,
                "--repo", config.mam2_repo,
                "--format", config.output_format,
                "--bit-depth", str(config.bit_depth),
                "--device", config.device,
            ]
            stage_name = "MatAnyone2 Refinement"
        else:
            refine_script = find_script("vitmatte_refine.py")

            depth_maps = depth_output / "depth"
            if not depth_maps.exists():
                depth_maps = depth_output

            refine_args = [
                "--sam-mask", str(sam_alpha),
                "--depth", str(depth_maps),
                "--frames", frames_arg,
                "--output", str(vitmatte_output),
                "--format", config.output_format,
                "--bit-depth", str(config.bit_depth),
                "--core-erosion", "10",
                "--adaptive-base", str(config.vitmatte_adaptive_base),
                "--adaptive-max", str(config.vitmatte_adaptive_max),
                "--save-trimap",
                # Hair polish settings
                "--hair-gamma", str(config.hair_gamma),
                "--hair-black-point", str(config.hair_black_point),
                "--hair-gain", str(config.hair_gain),
                # Guided Filter settings
                "--guided-radius", str(config.guided_filter_radius),
                "--guided-eps", str(config.guided_filter_eps),
            ]
            stage_name = "ViTMatte Alpha Refinement"

            if config.vitmatte_motion_aware:
                refine_args.append("--motion-aware")

            if not config.hair_polish_enabled:
                refine_args.append("--no-hair-polish")

        if config.verbose:
            refine_args.append("--verbose")

        success = run_python_stage(
            refine_script, refine_args,
            stage_name, config.verbose
        )

        if not success:
            logger.warning(f"{stage_name} failed, continuing with depth output")
            vitmatte_output = depth_output

        clear_gpu_memory()
    else:
        logger.info("Skipping Alpha Refinement (--skip-vitmatte)")
        vitmatte_output = depth_output

    # =========================================================================
    # STAGE 4: Edge Refinement
    # =========================================================================
    if not config.skip_edge:
        edge_script = find_script("edge_refine.py")

        # Determine alpha source (now from refinement stage)
        alpha_source = vitmatte_output / "alpha"
        if not alpha_source.exists():
            alpha_source = vitmatte_output

        edge_args = [
            "--alpha", str(alpha_source),
            "--output", str(edge_output),
            "--frames", frames_arg,
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
            "--frames", frames_arg,
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
            "--frames", frames_arg,
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
            "--frames", frames_arg,
            "--output", str(hair_output),
            "--format", config.output_format,
            "--bit-depth", str(config.bit_depth),
            "--adaptive-base", "2.0",
            "--adaptive-max", "60.0",
            "--motion-aware",  # Enable the motion logic
            "--save-trimap",
            # Hair polish settings
            "--hair-gamma", str(config.hair_gamma),
            "--hair-black-point", str(config.hair_black_point),
            "--hair-gain", str(config.hair_gain),
            # Guided Filter settings
            "--guided-radius", str(config.guided_filter_radius),
            "--guided-eps", str(config.guided_filter_eps),
        ]

        if not config.hair_polish_enabled:
            hair_args.append("--no-hair-polish")

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

    # Copy depth outputs (if available)
    depth_src = depth_output / "depth"
    if depth_src.exists():
        depth_dst = final_output / "depth"
        if depth_dst.exists():
            shutil.rmtree(depth_dst)
        shutil.copytree(depth_src, depth_dst)
        logger.info("  Copied depth/ to final output")

    # Cleanup intermediate if not keeping
    if not config.keep_intermediate:
        logger.info("Cleaning up intermediate files...")
        for intermediate in [sam_output, depth_output, vitmatte_output, edge_output,
                           temporal_output, combine_output]:
            if intermediate.exists() and intermediate != final_output:
                shutil.rmtree(intermediate, ignore_errors=True)
        if frames_is_temp and frames_dir and frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)

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
        description="AUTO-ROTO v5: Production-grade automatic rotoscoping with SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Full quality pipeline (SAM3 has built-in text prompting)
  %(prog)s --input video.mp4 --prompt "person" --output ./output

  # Process PNG sequence with high quality
  %(prog)s --input /path/to/frames/ --prompt "car" --output ./output --quality high

  # Quick mode (SAM3 + edge only)
  %(prog)s --input video.mp4 --prompt "person" --output ./output --quality draft --skip-temporal

  # Interactive selection
  %(prog)s --input video.mp4 --interactive --output ./output

QUALITY PRESETS:
  draft    - Fastest, small depth model
  standard - Balanced quality and speed (default)
  high     - Best quality, large depth model
  ultra    - Maximum quality, longer temporal window

NOTE: SAM3 includes built-in text prompting (270k+ concepts).
      No separate GroundingDINO required.
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
                             help="Text prompt for SAM3 (use . to separate multiple)")
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
                       help="Skip SAM3 (use existing alpha)")
    parser.add_argument("--skip-depth", action="store_true",
                       help="Skip depth refinement")
    parser.add_argument("--skip-vitmatte", action="store_true",
                       help="Skip alpha refinement stage (ViTMatte/MatAnyone2)")
    parser.add_argument("--skip-edge", action="store_true",
                       help="Skip edge refinement")
    parser.add_argument("--skip-temporal", action="store_true",
                       help="Skip temporal smoothing")
    parser.add_argument("--skip-combine", action="store_true",
                       help="Skip matte combination")
    parser.add_argument("--with-hair", action="store_true",
                       help="Enable hair refinement (off by default)")

    # Alpha refinement selection
    parser.add_argument("--refiner", choices=["vitmatte", "mam2"], default="vitmatte",
                       help="Alpha refinement model (default: vitmatte)")
    parser.add_argument("--mam2-checkpoint", default="./checkpoints/mam2.pth",
                       help="MatAnyone2 checkpoint path")
    parser.add_argument("--mam2-repo", default="MatAnyone2",
                       help="MatAnyone2 repo path (relative or absolute)")

    # SAM3 settings
    parser.add_argument("--sam-imgsz", type=int, default=0,
                       help="SAM3 processing resolution (default: 0 = auto from input, max 2048)")
    parser.add_argument("--sam-conf", type=float, default=0.25,
                       help="SAM3 confidence threshold (default: 0.25, lower = more inclusive masks)")
    parser.add_argument("--no-retina-masks", action="store_true",
                       help="Disable high-resolution mask output")
    parser.add_argument("--sam-max-det", type=int, default=100,
                       help="Maximum detections per frame (default: 100)")

    # Depth settings
    parser.add_argument("--depth-model",
                       choices=["small", "base", "large", "nested-base", "nested-large"],
                       help="Override Depth model (large=DA3Mono best for hair detail)")

    # DA3 Depth Sensitivity Settings (NEW)
    parser.add_argument("--depth-res", type=int, default=None,
                       help="DA3 processing resolution (default: auto). "
                            "Higher values (1024, 2048) capture finer hair details")
    parser.add_argument("--depth-method", type=str, default="upper",
                       choices=["upper", "lower"],
                       help="DA3 resize method: 'lower' gives higher effective resolution")
    parser.add_argument("--depth-percentiles", type=float, nargs=2, default=[2.0, 98.0],
                       metavar=("LOW", "HIGH"),
                       help="Depth normalization percentiles. Wider (1 99) preserves more detail")
    parser.add_argument("--use-depth-confidence", action="store_true",
                       help="Use DA3 confidence maps for semi-transparent edges")

    # ViTMatte settings (adaptive trimap)
    parser.add_argument("--vitmatte-motion", action="store_true",
                       help="Enable motion-aware trimap for ViTMatte")
    parser.add_argument("--vitmatte-base", type=float, default=2.0,
                       help="Min unknown width for smooth regions (default: 2)")
    parser.add_argument("--vitmatte-max", type=float, default=60.0,
                       help="Max unknown width for complex regions (default: 60)")

    # Hair polish settings (for edge/unknown regions)
    parser.add_argument("--hair-gamma", type=float, default=None,
                       help="Hair edge gamma correction (0.5-1.0, lower=more visible, default: from quality preset)")
    parser.add_argument("--hair-black-point", type=float, default=None,
                       help="Hair black point threshold (0.0-0.1, lower=preserve faint tips, default: from quality preset)")
    parser.add_argument("--hair-gain", type=float, default=None,
                       help="Hair gain multiplier (1.0-1.5, higher=more solid, default: from quality preset)")
    parser.add_argument("--no-hair-polish", action="store_true",
                       help="Disable hair polish entirely (preserve raw ViTMatte output)")

    # Guided Filter settings
    parser.add_argument("--guided-radius", type=int, default=None,
                       help="Guided filter radius (1-2 for hair, 4 for general, default: from quality preset)")
    parser.add_argument("--guided-eps", type=float, default=None,
                       help="Guided filter epsilon (lower=stricter edges, default: from quality preset)")

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

    # Performance
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-compile", action="store_true",
                       help="Disable SAM model compilation (auto-enabled on Windows)")
    parser.add_argument("--force-compile", action="store_true",
                       help="Force torch.compile even on Windows (may fail)")

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
        refiner=args.refiner,
        mam2_repo=args.mam2_repo,
        mam2_checkpoint=args.mam2_checkpoint,
        depth_model=args.depth_model or "",
        # DA3 depth sensitivity settings
        depth_process_res=args.depth_res,
        depth_process_method=args.depth_method,
        depth_norm_percentiles=tuple(args.depth_percentiles),
        use_depth_confidence=args.use_depth_confidence,
        # SAM3 settings
        sam_imgsz=args.sam_imgsz,
        sam_conf=args.sam_conf,
        sam_retina_masks=not args.no_retina_masks,
        sam_max_det=args.sam_max_det,
        # ViTMatte settings
        vitmatte_motion_aware=args.vitmatte_motion,
        vitmatte_adaptive_base=args.vitmatte_base,
        vitmatte_adaptive_max=args.vitmatte_max,
        # Hair polish settings (None = use quality preset)
        hair_gamma=args.hair_gamma,
        hair_black_point=args.hair_black_point,
        hair_gain=args.hair_gain,
        hair_polish_enabled=not args.no_hair_polish,
        # Guided Filter settings (None = use quality preset)
        guided_filter_radius=args.guided_radius,
        guided_filter_eps=args.guided_eps,
        edge_softness=args.edge_softness,
        core_shrink=args.core_shrink,
        despill_strength=args.despill,
        temporal_window=args.temporal_window,
        keyframe_interval=args.keyframe_interval,
        output_format=args.format,
        bit_depth=args.bit_depth,
        device=args.device,
        no_compile=args.no_compile,
        force_compile=args.force_compile,
        verbose=args.verbose,
        keep_intermediate=args.keep_intermediate,
    )

    success = run_pipeline(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
