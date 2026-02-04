"""
ViTMatte Stage
==============

Alpha refinement stage using ViTMatte with trimap synthesis.

Uses depth-guided trimap generation for high-quality alpha matting.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from auto_roto.stages.base import PipelineStage, StageContext, StageResult, StageRegistry
from auto_roto.config.vitmatte import GeometricMatteConfig

logger = logging.getLogger("AutoRoto.Stages.ViTMatte")


@StageRegistry.register
class ViTMatteStage(PipelineStage):
    """
    ViTMatte alpha refinement stage.

    Uses geometric-guided trimap synthesis with depth maps
    to refine SAM masks into high-quality alpha mattes.

    Requires:
        - SAM masks (from sam stage)
        - Depth maps (from depth stage)
        - RGB frames

    Outputs:
        - alpha/: Refined alpha mattes
        - trimap/: Generated trimaps (optional)
    """

    def __init__(
        self,
        config: GeometricMatteConfig = None,
        save_trimap: bool = True,
        first_frame_only: bool = False,
        logger: logging.Logger = None
    ):
        """
        Initialize ViTMatte stage.

        Args:
            config: GeometricMatteConfig instance (uses defaults if None)
            save_trimap: Whether to save trimap visualizations
            first_frame_only: If True, only process the first frame (for MatAnyone mode)
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.config = config or GeometricMatteConfig()
        self.save_trimap = save_trimap
        self.first_frame_only = first_frame_only

    @property
    def name(self) -> str:
        return "vitmatte"

    @property
    def description(self) -> str:
        return "ViTMatte Alpha Refinement"

    def validate_inputs(self, context: StageContext) -> Optional[str]:
        """Validate that SAM masks and depth maps exist."""
        # Check for input masks - prefer depth_expand, fall back to sam
        alpha_dir = context.get_alpha_dir("depth_expand")
        if not alpha_dir or not alpha_dir.exists():
            alpha_dir = context.get_alpha_dir("sam")

        if not alpha_dir or not alpha_dir.exists():
            return "No input masks found (need sam or depth_expand output)"

        # Check for depth output
        depth_result = context.get_stage_output("depth")
        if not depth_result or not depth_result.success:
            return "Depth stage must complete successfully first"

        depth_dir = context.get_depth_dir("depth")
        if not depth_dir or not depth_dir.exists():
            return f"Depth output not found"

        # Check for frames
        frames_dir = context.frames_dir or context.input_path
        if not frames_dir.exists():
            return f"Frames directory not found: {frames_dir}"

        return None

    def run(self, context: StageContext) -> StageResult:
        """Execute ViTMatte refinement."""
        import cv2

        from auto_roto.refiners.geometric import GeometricMatteRefiner
        from auto_roto.io.alpha import load_alpha, save_alpha
        from auto_roto.io.depth import load_depth_float

        # Setup output directory
        output_dir = context.output_dir / "03_vitmatte_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_out_dir = output_dir / "alpha"
        alpha_out_dir.mkdir(exist_ok=True)

        trimap_dir = None
        if self.save_trimap:
            trimap_dir = output_dir / "trimap"
            trimap_dir.mkdir(exist_ok=True)

        # Get input directories - prefer depth_expand masks when available
        input_alpha_dir = context.get_alpha_dir("depth_expand")
        if not input_alpha_dir or not input_alpha_dir.exists():
            input_alpha_dir = context.get_alpha_dir("sam")
            self._logger.info("Using SAM masks as input")
        else:
            self._logger.info("Using depth-expanded masks as input")

        depth_dir = context.get_depth_dir("depth")
        frames_dir = context.frames_dir or context.input_path

        # Get output format
        output_format = context.config.get("output_format", "exr")
        bit_depth = context.config.get("bit_depth", 16)

        # Initialize refiner
        refiner = GeometricMatteRefiner(
            config=self.config,
            logger=self._logger
        )

        try:
            # Build frame index -> file mappings for proper alignment
            # This handles cases where depth_expand may have skipped frames
            frame_files_by_idx = self._build_file_index_map(
                frames_dir, ["*.png", "*.jpg", "*.jpeg"]
            )
            mask_files_by_idx = self._build_file_index_map(
                input_alpha_dir, ["*.exr", "*.png"]
            )
            depth_files_by_idx = self._build_file_index_map(
                depth_dir, ["*.exr"]
            )

            if not frame_files_by_idx or not mask_files_by_idx or not depth_files_by_idx:
                return StageResult.failure_result(
                    ValueError("Missing input files"),
                    message="No frame, mask, or depth files found"
                )

            # Find common frame indices across all inputs
            common_indices = sorted(
                set(frame_files_by_idx.keys()) &
                set(mask_files_by_idx.keys()) &
                set(depth_files_by_idx.keys())
            )

            if not common_indices:
                return StageResult.failure_result(
                    ValueError("No matching frames"),
                    message="No frame indices match across frames, masks, and depth files"
                )

            if self.first_frame_only:
                common_indices = common_indices[:1]
                self._logger.info("First frame only mode (for MatAnyone)")

            self._logger.info(f"Processing {len(common_indices)} frames with matching indices")

            prev_rgb = None
            frame_count = 0

            for frame_idx in common_indices:
                # Load inputs using frame index (not list index)
                frame = cv2.imread(str(frame_files_by_idx[frame_idx]))
                if frame is None:
                    self._logger.warning(f"Could not read frame {frame_idx}, skipping")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_mask = load_alpha(mask_files_by_idx[frame_idx])
                depth = load_depth_float(depth_files_by_idx[frame_idx])

                # Set trimap save path (use frame_idx for proper alignment)
                trimap_path = None
                if trimap_dir:
                    trimap_path = trimap_dir / f"trimap.{frame_idx:04d}.png"

                # Process frame
                alpha = refiner.process_frame(
                    rgb=rgb,
                    sam_mask=input_mask,
                    depth=depth,
                    save_trimap_path=trimap_path,
                    prev_rgb=prev_rgb
                )

                # Save alpha (use frame_idx for proper alignment with inputs)
                alpha_path = alpha_out_dir / f"alpha.{frame_idx:04d}.{output_format}"
                save_alpha(alpha_path, alpha, bit_depth=bit_depth)

                prev_rgb = rgb
                frame_count += 1

                if frame_count % 10 == 0:
                    self._logger.info(f"  Frame {frame_idx} ({frame_count}/{len(common_indices)})")

            self._logger.info(f"Refined {frame_count} frames")

            outputs = {
                "alpha_dir": str(alpha_out_dir),
                "frame_count": frame_count
            }
            if trimap_dir:
                outputs["trimap_dir"] = str(trimap_dir)

            return StageResult.success_result(
                output_dir=output_dir,
                message=f"Refined {frame_count} frames",
                outputs=outputs
            )

        finally:
            refiner.release()

    def _build_file_index_map(self, directory: Path, patterns: list) -> dict:
        """
        Build a mapping of frame indices to file paths.

        Extracts numeric frame indices from filenames and maps them to paths.
        This enables frame-index-based matching instead of list-index pairing.

        Args:
            directory: Directory to search
            patterns: List of glob patterns (e.g., ["*.png", "*.jpg"])

        Returns:
            Dict mapping frame_index (int) -> file_path (Path)
        """
        file_map = {}

        for pattern in patterns:
            for filepath in directory.glob(pattern):
                # Extract frame index from filename
                # Handles patterns like: frame.0001.png, depth_0001.exr, roto_0001.exr
                match = re.search(r'(\d+)', filepath.stem)
                if match:
                    frame_idx = int(match.group(1))
                    # Don't overwrite if already found (first pattern wins)
                    if frame_idx not in file_map:
                        file_map[frame_idx] = filepath

        return file_map

    def cleanup(self, context: StageContext):
        """Clear GPU memory after ViTMatte."""
        from auto_roto.utils.gpu import clear_gpu_memory
        clear_gpu_memory()
