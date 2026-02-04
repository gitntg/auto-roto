"""
Depth Stage
===========

Depth estimation stage for the auto-roto pipeline.

Runs Depth Anything V3 to estimate depth maps from RGB frames.
"""

import logging
from pathlib import Path
from typing import Optional

from auto_roto.stages.base import PipelineStage, StageContext, StageResult, StageRegistry
from auto_roto.config.depth import DepthRefineConfig

logger = logging.getLogger("AutoRoto.Stages.Depth")


@StageRegistry.register
class DepthStage(PipelineStage):
    """
    Depth estimation stage.

    Runs Depth Anything V3 to estimate depth maps from RGB frames.

    Outputs:
        - depth/: Depth maps for each frame (EXR float32)
    """

    def __init__(
        self,
        config: DepthRefineConfig = None,
        first_frame_only: bool = False,
        logger: logging.Logger = None
    ):
        """
        Initialize depth stage.

        Args:
            config: DepthRefineConfig instance (uses defaults if None)
            first_frame_only: If True, only process the first frame (for MatAnyone mode)
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.config = config or DepthRefineConfig()
        self.first_frame_only = first_frame_only

    @property
    def name(self) -> str:
        return "depth"

    @property
    def description(self) -> str:
        return "Depth Estimation (DA3)"

    def validate_inputs(self, context: StageContext) -> Optional[str]:
        """Validate that frames exist."""
        frames_dir = context.frames_dir or context.input_path

        if frames_dir.is_file():
            # Video file - will be processed directly
            return None

        if not frames_dir.exists():
            return f"Frames directory not found: {frames_dir}"

        # Check for frame files
        frame_files = (
            list(frames_dir.glob("*.png")) +
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg")) +
            list(frames_dir.glob("*.exr"))
        )
        if not frame_files:
            return f"No frame files found in: {frames_dir}"

        return None

    def run(self, context: StageContext) -> StageResult:
        """Execute depth estimation."""
        import cv2

        from auto_roto.models.depth_anything import DepthEstimator
        from auto_roto.io.depth import save_depth_float

        # Setup output directory
        output_dir = context.output_dir / "02_depth_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        depth_dir = output_dir / "depth"
        depth_dir.mkdir(exist_ok=True)

        # Get frames directory
        frames_dir = context.frames_dir or context.input_path

        # Initialize depth estimator
        estimator = DepthEstimator(
            model_size=self.config.depth_model,
            device=context.device,
            process_res=self.config.depth_process_res,
            process_method=self.config.depth_process_method,
            logger=self._logger
        )

        try:
            frame_count = 0

            if self.first_frame_only:
                self._logger.info("First frame only mode (for MatAnyone)")

            if frames_dir.is_file():
                # Video file - extract and process frames
                cap = cv2.VideoCapture(str(frames_dir))
                idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Estimate depth
                    depth = estimator.estimate(rgb)

                    # Save depth (raw DA3 output, unnormalized)
                    depth_path = depth_dir / f"depth.{idx:04d}.exr"
                    save_depth_float(depth_path, depth)

                    if idx % 10 == 0:
                        self._logger.info(f"  Frame {idx}")

                    idx += 1
                    frame_count = idx

                    if self.first_frame_only:
                        break

                cap.release()

            else:
                # Frames directory
                frame_files = sorted(
                    list(frames_dir.glob("*.png")) +
                    list(frames_dir.glob("*.jpg")) +
                    list(frames_dir.glob("*.jpeg"))
                )

                for idx, frame_path in enumerate(frame_files):
                    # Read frame
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Estimate depth
                    depth = estimator.estimate(rgb)

                    # Save depth (raw DA3 output, unnormalized)
                    depth_out_path = depth_dir / f"depth.{idx:04d}.exr"
                    save_depth_float(depth_out_path, depth)

                    if idx % 10 == 0:
                        self._logger.info(f"  Frame {idx}/{len(frame_files)}")

                    frame_count = idx + 1

                    if self.first_frame_only:
                        break

            self._logger.info(f"Processed {frame_count} frames")

            return StageResult.success_result(
                output_dir=output_dir,
                message=f"Estimated depth for {frame_count} frames",
                outputs={
                    "depth_dir": str(depth_dir),
                    "frame_count": frame_count
                }
            )

        finally:
            estimator.release()

    def cleanup(self, context: StageContext):
        """Clear GPU memory after depth estimation."""
        from auto_roto.utils.gpu import clear_gpu_memory
        clear_gpu_memory()
