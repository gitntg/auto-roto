"""
Depth Expand Stage
==================

Depth-guided mask expansion stage for the auto-roto pipeline.

Expands SAM masks based on depth similarity to capture complete subject coverage.
This is a key component of the cinema preset workflow.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from auto_roto.stages.base import PipelineStage, StageContext, StageResult, StageRegistry
from auto_roto.config.depth_expansion import DepthExpansionConfig
from auto_roto.refiners.depth_expansion import DepthBasedExpander
from auto_roto.io.alpha import load_alpha, save_alpha
from auto_roto.io.depth import load_depth_float

logger = logging.getLogger("AutoRoto.Stages.DepthExpand")


@StageRegistry.register
class DepthExpandStage(PipelineStage):
    """
    Depth-guided mask expansion stage.

    Expands SAM masks based on depth similarity, capturing
    regions at the same depth plane as the subject.

    This enables complete subject coverage by including all
    pixels at similar depth to the initial SAM mask.

    Requires:
        - SAM mask (from sam stage)
        - Depth map (from depth stage)

    Outputs:
        - alpha/: Expanded masks
        - debug/: Optional expansion visualizations
    """

    def __init__(
        self,
        config: DepthExpansionConfig = None,
        first_frame_only: bool = True,
        logger: logging.Logger = None
    ):
        """
        Initialize depth expand stage.

        Args:
            config: DepthExpansionConfig instance (uses defaults if None)
            first_frame_only: If True, only process the first frame (cinema mode)
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.config = config or DepthExpansionConfig()
        self.first_frame_only = first_frame_only

    @property
    def name(self) -> str:
        return "depth_expand"

    @property
    def description(self) -> str:
        return "Depth-Guided Mask Expansion"

    def validate_inputs(self, context: StageContext) -> Optional[str]:
        """Validate that required inputs exist."""
        # Check for SAM output
        sam_alpha_dir = context.get_alpha_dir("sam")
        if not sam_alpha_dir or not sam_alpha_dir.exists():
            return "SAM alpha output not found - run sam stage first"

        # Check for depth output
        depth_dir = context.get_depth_dir("depth")
        if not depth_dir or not depth_dir.exists():
            return "Depth output not found - run depth stage first"

        # Check for frames directory (for edge-aware expansion)
        # Fall back to input_path if frames_dir is not set (matching other stages)
        frames_dir = context.frames_dir or context.input_path
        if self.config.edge_aware:
            if frames_dir is None or not frames_dir.exists():
                self._logger.warning("No frames directory - edge-aware expansion disabled")
            elif frames_dir.is_file():
                self._logger.warning("Input is a video file - edge-aware expansion disabled for video inputs")

        return None

    def run(self, context: StageContext) -> StageResult:
        """Execute depth-guided mask expansion."""
        # Setup output directory
        output_dir = context.output_dir / "02b_depth_expand_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_dir = output_dir / "alpha"
        alpha_dir.mkdir(exist_ok=True)

        debug_dir = None
        if self.config.save_expansion_mask:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)

        # Get input directories
        sam_alpha_dir = context.get_alpha_dir("sam")
        depth_dir = context.get_depth_dir("depth")

        # Initialize expander
        expander = DepthBasedExpander(self.config, self._logger)

        # Get output format from context
        output_format = context.config.get("output_format", "exr")
        bit_depth = context.config.get("bit_depth", 16)

        # Get list of SAM alpha files
        alpha_files = sorted(sam_alpha_dir.glob("*.exr")) + sorted(sam_alpha_dir.glob("*.png"))
        if not alpha_files:
            return StageResult.failure_result(
                ValueError("No alpha files found in SAM output"),
                message="No alpha files found"
            )

        frame_count = 0

        if self.first_frame_only:
            self._logger.info("First frame only mode (cinema preset)")
            alpha_files = alpha_files[:1]

        for alpha_file in alpha_files:
            # Extract frame index from filename
            frame_idx = self._extract_frame_index(alpha_file)

            # Load SAM mask
            sam_mask = load_alpha(alpha_file)

            # Find corresponding depth file
            depth_file = self._find_depth_file(depth_dir, frame_idx)
            if depth_file is None:
                self._logger.warning(f"No depth file for frame {frame_idx}, skipping")
                continue

            depth = load_depth_float(depth_file)

            # Load RGB frame for edge-aware expansion if available
            # Fall back to input_path if frames_dir is not set (matching other stages)
            rgb = None
            frames_dir = context.frames_dir or context.input_path
            if self.config.edge_aware and frames_dir and frames_dir.is_dir():
                rgb_file = self._find_frame_file(frames_dir, frame_idx)
                if rgb_file:
                    rgb = cv2.imread(str(rgb_file))
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Compute depth profile for logging
            profile = expander.compute_depth_profile(sam_mask, depth)

            # Expand mask
            expanded_mask = expander.expand_mask(sam_mask, depth, rgb)

            # Save expanded mask
            output_filename = f"roto_{frame_idx:04d}.{output_format}"
            output_path = alpha_dir / output_filename
            save_alpha(output_path, expanded_mask, bit_depth)

            # Save debug visualization if enabled
            if debug_dir:
                expander.save_debug_output(
                    debug_dir, sam_mask, expanded_mask, depth, profile, frame_idx
                )

            frame_count += 1
            self._logger.info(
                f"Frame {frame_idx}: expanded {np.sum(sam_mask > 0.5)} -> "
                f"{np.sum(expanded_mask > 0.5)} pixels"
            )

        return StageResult.success_result(
            output_dir=output_dir,
            message=f"Expanded {frame_count} frames",
            outputs={
                "alpha_dir": str(alpha_dir),
                "frame_count": frame_count,
            }
        )

    def _extract_frame_index(self, filepath: Path) -> int:
        """Extract frame index from filename."""
        stem = filepath.stem
        # Try to extract number from common patterns
        import re
        match = re.search(r'(\d+)', stem)
        if match:
            return int(match.group(1))
        return 0

    def _find_depth_file(self, depth_dir: Path, frame_idx: int) -> Optional[Path]:
        """Find depth file for given frame index."""
        # Order matters: most specific patterns first
        patterns = [
            f"depth.{frame_idx:04d}.exr",  # DepthStage default format (dot separator)
            f"roto_{frame_idx:04d}.exr",
            f"depth_{frame_idx:04d}.exr",
            f"frame_{frame_idx:04d}.exr",
            f"*{frame_idx:04d}*.exr",
        ]

        for pattern in patterns:
            matches = list(depth_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def _find_frame_file(self, frames_dir: Path, frame_idx: int) -> Optional[Path]:
        """Find RGB frame file for given frame index."""
        # Support common frame formats including EXR
        patterns = [
            f"frame.{frame_idx:04d}.png",
            f"frame.{frame_idx:04d}.jpg",
            f"frame.{frame_idx:04d}.exr",
            f"frame_{frame_idx:04d}.png",
            f"frame_{frame_idx:04d}.jpg",
            f"frame_{frame_idx:04d}.exr",
            f"*{frame_idx:04d}*.png",
            f"*{frame_idx:04d}*.jpg",
            f"*{frame_idx:04d}*.exr",
        ]

        for pattern in patterns:
            matches = list(frames_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def cleanup(self, context: StageContext):
        """Cleanup GPU memory."""
        from auto_roto.utils.gpu import clear_gpu_memory
        clear_gpu_memory()
