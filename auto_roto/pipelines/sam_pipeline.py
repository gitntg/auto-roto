"""
SAM Pipeline
============

Simple pipeline for SAM3 segmentation only.

This is the most basic pipeline - just runs SAM3 to generate masks.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from auto_roto.stages.base import StageContext, StageResult
from auto_roto.stages.sam import SAMStage
from auto_roto.config.sam import RotoConfig

logger = logging.getLogger("AutoRoto.Pipelines.SAM")


class AutoRotoPipeline:
    """
    Simple SAM3 segmentation pipeline.

    Runs SAM3 to generate alpha masks from text, box, or point prompts.

    Usage:
        pipeline = AutoRotoPipeline(prompts=["person"])
        result = pipeline.run(
            input_path="video.mp4",
            output_dir="./output"
        )
    """

    def __init__(
        self,
        config: RotoConfig = None,
        prompts: List[str] = None,
        box: str = None,
        points: List[tuple] = None,
        point_labels: List[int] = None,
        device: str = "cuda",
        output_format: str = "exr",
        bit_depth: int = 16,
        logger: logging.Logger = None
    ):
        """
        Initialize SAM pipeline.

        Args:
            config: RotoConfig instance (uses defaults if None)
            prompts: Text prompts for segmentation
            box: Box prompt as "x1,y1,x2,y2"
            points: Point coordinates [(x1,y1), ...]
            point_labels: Point labels (1=include, 0=exclude)
            device: Compute device (cuda/cpu)
            output_format: Output format (exr/png)
            bit_depth: Bit depth for output (8/16/32)
            logger: Optional logger instance
        """
        self.config = config or RotoConfig()
        self.prompts = prompts or ["person"]
        self.box = box
        self.points = points
        self.point_labels = point_labels
        self.device = device
        self.output_format = output_format
        self.bit_depth = bit_depth
        self._logger = logger or logging.getLogger("AutoRoto.SAMPipeline")

    def run(
        self,
        input_path: str,
        output_dir: str,
        frames_dir: str = None,
        verbose: bool = False
    ) -> StageResult:
        """
        Run the SAM pipeline.

        Args:
            input_path: Input video or frames directory
            output_dir: Output directory
            frames_dir: Optional pre-extracted frames directory
            verbose: Verbose output

        Returns:
            StageResult with execution status
        """
        start_time = time.time()

        self._logger.info("="*60)
        self._logger.info("AUTO-ROTO SAM PIPELINE")
        self._logger.info("="*60)
        self._logger.info(f"Input: {input_path}")
        self._logger.info(f"Output: {output_dir}")

        # Create context
        context = StageContext(
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            frames_dir=Path(frames_dir) if frames_dir else None,
            device=self.device,
            logger=self._logger,
            verbose=verbose,
            config={
                "output_format": self.output_format,
                "bit_depth": self.bit_depth
            }
        )

        # Create output directory
        context.output_dir.mkdir(parents=True, exist_ok=True)

        # Create and run SAM stage
        sam_stage = SAMStage(
            config=self.config,
            prompts=self.prompts,
            box=self.box,
            points=self.points,
            point_labels=self.point_labels,
            logger=self._logger
        )

        result = sam_stage.execute(context)

        # Summary
        total_time = time.time() - start_time

        self._logger.info("\n" + "="*60)
        self._logger.info("PIPELINE COMPLETE")
        self._logger.info("="*60)
        self._logger.info(f"Total time: {total_time:.1f}s")

        if result.success:
            self._logger.info(f"Output: {result.output_dir}")
        else:
            self._logger.error(f"Failed: {result.message}")

        return result
