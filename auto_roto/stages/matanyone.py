"""
MatAnyone Stage
===============

MatAnyone temporal propagation stage for the auto-roto pipeline.

Uses the best alpha from frame 1 (produced by SAM/Depth/ViTMatte/Combine)
and propagates it temporally through all frames using MatAnyone's
video object segmentation model.
"""

import logging
from pathlib import Path
from typing import Optional

from auto_roto.stages.base import PipelineStage, StageContext, StageResult, StageRegistry

logger = logging.getLogger("AutoRoto.Stages.MatAnyone")


@StageRegistry.register
class MatAnyoneStage(PipelineStage):
    """
    MatAnyone temporal propagation stage.

    Takes the refined alpha from frame 1 (combine/vitmatte/sam output)
    and propagates it through all frames using MatAnyone's temporal
    consistency model.

    Requires:
        - Alpha mask from previous stage (only frame 0)
        - RGB frames (all frames)

    Outputs:
        - alpha/: Propagated alpha mattes for all frames
    """

    def __init__(
        self,
        repo_path: str = "./MatAnyone",
        checkpoint_path: str = "./checkpoints/matanyone.pth",
        mem_every: int = 3,
        max_mem_frames: int = 10,
        warmup: int = 5,
        erode: int = 3,
        dilate: int = 5,
        top_k: int = 50,
        use_long_term: bool = True,
        max_internal_size: int = -1,
        logger: logging.Logger = None
    ):
        """
        Initialize MatAnyone stage.

        Args:
            repo_path: Path to MatAnyone repository
            checkpoint_path: Path to MatAnyone model checkpoint
            mem_every: Memory frame interval (lower = better quality, slower)
            max_mem_frames: Max memory frames (higher = better quality, more VRAM)
            warmup: Number of warmup frames
            erode: Erosion iterations for initial mask
            dilate: Dilation iterations for initial mask
            top_k: Top-k memory matching
            use_long_term: Enable long-term memory
            max_internal_size: Max internal processing size (-1 = full resolution)
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.mem_every = mem_every
        self.max_mem_frames = max_mem_frames
        self.warmup = warmup
        self.erode = erode
        self.dilate = dilate
        self.top_k = top_k
        self.use_long_term = use_long_term
        self.max_internal_size = max_internal_size

    @property
    def name(self) -> str:
        return "matanyone"

    @property
    def description(self) -> str:
        return "MatAnyone Temporal Propagation"

    def validate_inputs(self, context: StageContext) -> Optional[str]:
        """Validate that alpha mask and frames exist."""
        # Check for alpha output from combine (preferred), vitmatte, or sam
        alpha_dir = context.get_alpha_dir("combine")
        if not alpha_dir:
            alpha_dir = context.get_alpha_dir("vitmatte")
        if not alpha_dir:
            alpha_dir = context.get_alpha_dir("sam")

        if not alpha_dir or not alpha_dir.exists():
            return "No alpha output found from previous stages"

        # Check that there's at least one alpha file
        alpha_files = list(alpha_dir.glob("*.exr")) + list(alpha_dir.glob("*.png"))
        if not alpha_files:
            return f"No alpha files found in {alpha_dir}"

        # Check for frames
        frames_dir = context.frames_dir or context.input_path
        if not frames_dir.exists():
            return f"Frames directory not found: {frames_dir}"

        return None

    def run(self, context: StageContext) -> StageResult:
        """Execute MatAnyone temporal propagation."""
        from auto_roto.models.matanyone import (
            load_matanyone_adapter,
            process_matanyone_v1_sequence
        )
        from auto_roto.io.alpha import load_alpha, save_alpha

        # Setup output directory
        output_dir = context.output_dir / "05_matanyone_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_out_dir = output_dir / "alpha"
        alpha_out_dir.mkdir(exist_ok=True)

        # Get input directories
        # Prefer combine output, then vitmatte, then sam
        alpha_dir = context.get_alpha_dir("combine")
        source_stage = "combine"
        if not alpha_dir:
            alpha_dir = context.get_alpha_dir("vitmatte")
            source_stage = "vitmatte"
        if not alpha_dir:
            alpha_dir = context.get_alpha_dir("sam")
            source_stage = "sam"

        frames_dir = context.frames_dir or context.input_path

        self._logger.info(f"Using alpha from: {source_stage}")
        self._logger.info(f"Frames directory: {frames_dir}")

        # Get output format
        output_format = context.config.get("output_format", "exr")
        bit_depth = context.config.get("bit_depth", 16)

        # Get frame files (all frames)
        frame_files = sorted(
            list(frames_dir.glob("*.png")) +
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg"))
        )

        if not frame_files:
            return StageResult.failure_result(
                ValueError("No frame files found"),
                message=f"No frame files found in {frames_dir}"
            )

        # Get alpha files (should be only frame 0 in MatAnyone mode)
        alpha_files = sorted(
            list(alpha_dir.glob("*.exr")) +
            list(alpha_dir.glob("*.png"))
        )

        if not alpha_files:
            return StageResult.failure_result(
                ValueError("No alpha files found"),
                message=f"No alpha files found in {alpha_dir}"
            )

        self._logger.info(f"Processing {len(frame_files)} frames")
        self._logger.info(f"Using {len(alpha_files)} initial alpha(s) from {source_stage}")

        try:
            # Load MatAnyone adapter
            self._logger.info("Loading MatAnyone model...")
            adapter = load_matanyone_adapter(
                repo_path=self.repo_path,
                checkpoint_path=self.checkpoint_path,
                device=context.device,
                logger=self._logger,
                mem_every=self.mem_every,
                max_mem_frames=self.max_mem_frames,
                top_k=self.top_k,
                use_long_term=self.use_long_term,
                max_internal_size=self.max_internal_size
            )

            # Process sequence
            result_code = process_matanyone_v1_sequence(
                adapter=adapter,
                frame_files=frame_files,
                alpha_files=alpha_files,
                alpha_output_dir=alpha_out_dir,
                warmup=self.warmup,
                erode=self.erode,
                dilate=self.dilate,
                output_format=output_format,
                bit_depth=bit_depth,
                log=self._logger
            )

            if result_code != 0:
                return StageResult.failure_result(
                    RuntimeError("MatAnyone processing failed"),
                    message="MatAnyone processing failed"
                )

            # Count output files
            output_files = list(alpha_out_dir.glob("*"))
            frame_count = len(output_files)

            self._logger.info(f"Propagated to {frame_count} frames")

            return StageResult.success_result(
                output_dir=output_dir,
                message=f"Propagated alpha to {frame_count} frames",
                outputs={
                    "alpha_dir": str(alpha_out_dir),
                    "frame_count": frame_count
                }
            )

        except Exception as e:
            self._logger.error(f"MatAnyone processing failed: {e}")
            import traceback
            traceback.print_exc()
            return StageResult.failure_result(e, message=str(e))

    def cleanup(self, context: StageContext):
        """Clear GPU memory after MatAnyone."""
        from auto_roto.utils.gpu import clear_gpu_memory
        clear_gpu_memory()
