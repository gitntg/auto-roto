"""
SAM Stage
=========

SAM3 segmentation stage for the auto-roto pipeline.

Performs object segmentation using SAM3 with text, box, or point prompts.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from auto_roto.stages.base import PipelineStage, StageContext, StageResult, StageRegistry
from auto_roto.config.sam import RotoConfig

logger = logging.getLogger("AutoRoto.Stages.SAM")


@StageRegistry.register
class SAMStage(PipelineStage):
    """
    SAM3 segmentation stage.

    Performs object segmentation using SAM3 with:
    - Text prompts (270k+ concepts)
    - Box prompts
    - Point prompts (include/exclude)

    Outputs:
        - alpha/: Alpha masks for each frame
        - preview/: Preview composites (optional)
    """

    def __init__(
        self,
        config: RotoConfig = None,
        prompts: List[str] = None,
        box: str = None,
        points: List[tuple] = None,
        point_labels: List[int] = None,
        first_frame_only: bool = False,
        logger: logging.Logger = None
    ):
        """
        Initialize SAM stage.

        Args:
            config: RotoConfig instance (uses defaults if None)
            prompts: Text prompts (use "." to separate multiple)
            box: Box prompt as "x1,y1,x2,y2"
            points: Point coordinates [(x1,y1), (x2,y2), ...]
            point_labels: Point labels (1=include, 0=exclude)
            first_frame_only: If True, only process the first frame (for MatAnyone mode)
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.config = config or RotoConfig()
        self.prompts = prompts or ["person"]
        self.box = box
        self.points = points
        self.point_labels = point_labels
        self.first_frame_only = first_frame_only

    @property
    def name(self) -> str:
        return "sam"

    @property
    def description(self) -> str:
        return "SAM3 Segmentation"

    def validate_inputs(self, context: StageContext) -> Optional[str]:
        """Validate that input exists."""
        if not context.input_path.exists():
            return f"Input not found: {context.input_path}"

        # Need at least one prompt type
        if not self.prompts and not self.box and not self.points:
            return "No prompt specified (text, box, or points required)"

        return None

    def run(self, context: StageContext) -> StageResult:
        """Execute SAM3 segmentation."""
        import cv2

        from auto_roto.models.sam3 import SAM3Segmenter
        from auto_roto.io.frames import FrameWriter

        # Setup output directory
        output_dir = context.output_dir / "01_sam_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_dir = output_dir / "alpha"
        alpha_dir.mkdir(exist_ok=True)

        # Initialize SAM3
        sam3 = SAM3Segmenter(
            model_path="sam3.pt",
            device=context.device,
            logger=self._logger,
            imgsz=self.config.sam_imgsz,
            conf=self.config.sam_conf,
            retina_masks=self.config.sam_retina_masks,
            max_det=self.config.sam_max_det
        )

        # Setup frame writer
        output_format = context.config.get("output_format", "exr")
        bit_depth = context.config.get("bit_depth", 16)

        writer = FrameWriter(
            output_dir=str(output_dir),
            prefix="roto",
            format=output_format,
            bit_depth=bit_depth,
            padding=4,
            logger=self._logger
        )

        try:
            input_path = context.input_path
            frame_count = 0

            if self.first_frame_only:
                self._logger.info("First frame only mode (for MatAnyone)")

            # Determine segmentation mode
            if self.points and self.point_labels:
                # Point-based segmentation
                self._logger.info(f"Mode: Point prompts ({len(self.points)} points)")

                if input_path.is_dir():
                    for idx, mask in sam3.segment_frames_with_points(
                        str(input_path), self.points, self.point_labels
                    ):
                        if mask is not None:
                            writer.write_alpha(mask, idx)
                            frame_count = idx + 1
                        if self.first_frame_only:
                            break
                else:
                    for idx, mask in sam3.segment_video_with_points(
                        str(input_path), self.points, self.point_labels
                    ):
                        if mask is not None:
                            writer.write_alpha(mask, idx)
                            frame_count = idx + 1
                        if self.first_frame_only:
                            break

            elif self.box:
                # Box-based segmentation
                self._logger.info(f"Mode: Box prompt ({self.box})")
                box_coords = [int(x) for x in self.box.split(",")]

                if input_path.is_dir():
                    for idx, mask in sam3.segment_frames_with_box(
                        str(input_path), box_coords
                    ):
                        if mask is not None:
                            writer.write_alpha(mask, idx)
                            frame_count = idx + 1
                        if self.first_frame_only:
                            break
                else:
                    for idx, mask in sam3.segment_video_with_box(
                        str(input_path), box_coords
                    ):
                        if mask is not None:
                            writer.write_alpha(mask, idx)
                            frame_count = idx + 1
                        if self.first_frame_only:
                            break

            else:
                # Text-based segmentation (default)
                self._logger.info(f"Mode: Text prompts {self.prompts}")

                if input_path.is_file():
                    for idx, mask in sam3.segment_video_with_text(
                        str(input_path), self.prompts
                    ):
                        if mask is not None:
                            writer.write_alpha(mask, idx)
                            frame_count = idx + 1
                        if self.first_frame_only:
                            break

                elif input_path.is_dir():
                    for idx, mask in sam3.segment_frames_with_text(
                        str(input_path), self.prompts
                    ):
                        if mask is not None:
                            writer.write_alpha(mask, idx)
                            frame_count = idx + 1
                        if self.first_frame_only:
                            break

            self._logger.info(f"Processed {frame_count} frames")

            return StageResult.success_result(
                output_dir=output_dir,
                message=f"Segmented {frame_count} frames",
                outputs={
                    "alpha_dir": str(alpha_dir),
                    "frame_count": frame_count
                }
            )

        finally:
            sam3.release()

    def cleanup(self, context: StageContext):
        """Clear GPU memory after SAM."""
        from auto_roto.utils.gpu import clear_gpu_memory
        clear_gpu_memory()
