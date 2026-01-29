"""
ViTMatte Stage
==============

Alpha refinement stage using ViTMatte with trimap synthesis.

Uses depth-guided trimap generation for high-quality alpha matting.
"""

import logging
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
        logger: logging.Logger = None
    ):
        """
        Initialize ViTMatte stage.

        Args:
            config: GeometricMatteConfig instance (uses defaults if None)
            save_trimap: Whether to save trimap visualizations
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.config = config or GeometricMatteConfig()
        self.save_trimap = save_trimap

    @property
    def name(self) -> str:
        return "vitmatte"

    @property
    def description(self) -> str:
        return "ViTMatte Alpha Refinement"

    def validate_inputs(self, context: StageContext) -> Optional[str]:
        """Validate that SAM masks and depth maps exist."""
        # Check for SAM output
        sam_result = context.get_stage_output("sam")
        if not sam_result or not sam_result.success:
            return "SAM stage must complete successfully first"

        alpha_dir = context.get_alpha_dir("sam")
        if not alpha_dir or not alpha_dir.exists():
            return f"SAM alpha output not found"

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

        # Get input directories
        sam_alpha_dir = context.get_alpha_dir("sam")
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
            # Get frame files
            frame_files = sorted(
                list(frames_dir.glob("*.png")) +
                list(frames_dir.glob("*.jpg")) +
                list(frames_dir.glob("*.jpeg"))
            )

            # Get SAM mask files
            sam_files = sorted(
                list(sam_alpha_dir.glob("*.exr")) +
                list(sam_alpha_dir.glob("*.png"))
            )

            # Get depth files
            depth_files = sorted(depth_dir.glob("*.exr"))

            if not frame_files or not sam_files or not depth_files:
                return StageResult.failure_result(
                    ValueError("Missing input files"),
                    message="No frame, SAM, or depth files found"
                )

            # Ensure counts match
            min_count = min(len(frame_files), len(sam_files), len(depth_files))
            self._logger.info(f"Processing {min_count} frames")

            prev_rgb = None
            frame_count = 0

            for idx in range(min_count):
                # Load inputs
                frame = cv2.imread(str(frame_files[idx]))
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sam_mask = load_alpha(sam_files[idx])
                depth = load_depth_float(depth_files[idx])

                # Set trimap save path
                trimap_path = None
                if trimap_dir:
                    trimap_path = trimap_dir / f"trimap.{idx:04d}.png"

                # Process frame
                alpha = refiner.process_frame(
                    rgb=rgb,
                    sam_mask=sam_mask,
                    depth=depth,
                    save_trimap_path=trimap_path,
                    prev_rgb=prev_rgb
                )

                # Save alpha
                alpha_path = alpha_out_dir / f"alpha.{idx:04d}.{output_format}"
                save_alpha(alpha_path, alpha, bit_depth=bit_depth)

                prev_rgb = rgb
                frame_count = idx + 1

                if idx % 10 == 0:
                    self._logger.info(f"  Frame {idx}/{min_count}")

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

    def cleanup(self, context: StageContext):
        """Clear GPU memory after ViTMatte."""
        from auto_roto.utils.gpu import clear_gpu_memory
        clear_gpu_memory()
