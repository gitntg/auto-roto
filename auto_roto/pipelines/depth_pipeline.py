"""
Depth Pipeline
==============

Pipeline for depth estimation and depth-guided refinement.

Runs Depth Anything V3 to estimate depth maps from RGB frames.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from auto_roto.stages.base import StageContext, StageResult
from auto_roto.stages.depth import DepthStage
from auto_roto.config.depth import DepthRefineConfig

logger = logging.getLogger("AutoRoto.Pipelines.Depth")


class DepthRefinePipeline:
    """
    Depth estimation pipeline.

    Runs Depth Anything V3 to estimate depth maps from RGB frames.

    Usage:
        pipeline = DepthRefinePipeline(model="large")
        result = pipeline.run(
            input_path="./frames",
            output_dir="./output"
        )
    """

    def __init__(
        self,
        config: DepthRefineConfig = None,
        model: str = "base",
        process_res: int = None,
        process_method: str = "upper",
        percentiles: tuple = (2.0, 98.0),
        device: str = "cuda",
        logger: logging.Logger = None
    ):
        """
        Initialize depth pipeline.

        Args:
            config: DepthRefineConfig instance (uses defaults if None)
            model: Depth model size (small/base/large/nested-base/nested-large)
            process_res: Processing resolution (None=auto)
            process_method: Resize method (upper/lower)
            percentiles: Normalization percentiles (low, high)
            device: Compute device (cuda/cpu)
            logger: Optional logger instance
        """
        if config:
            self.config = config
        else:
            self.config = DepthRefineConfig(
                depth_model=model,
                depth_process_res=process_res,
                depth_process_method=process_method,
                depth_norm_percentiles=percentiles
            )

        self.device = device
        self._logger = logger or logging.getLogger("AutoRoto.DepthPipeline")

    def run(
        self,
        input_path: str,
        output_dir: str,
        frames_dir: str = None,
        verbose: bool = False
    ) -> StageResult:
        """
        Run the depth estimation pipeline.

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
        self._logger.info("DEPTH ESTIMATION PIPELINE")
        self._logger.info("="*60)
        self._logger.info(f"Input: {input_path}")
        self._logger.info(f"Output: {output_dir}")
        self._logger.info(f"Model: {self.config.depth_model}")

        # Create context
        context = StageContext(
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            frames_dir=Path(frames_dir) if frames_dir else None,
            device=self.device,
            logger=self._logger,
            verbose=verbose
        )

        # Create output directory
        context.output_dir.mkdir(parents=True, exist_ok=True)

        # Create and run depth stage
        depth_stage = DepthStage(
            config=self.config,
            logger=self._logger
        )

        result = depth_stage.execute(context)

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
