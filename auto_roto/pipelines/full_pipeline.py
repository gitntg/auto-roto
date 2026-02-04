"""
Full Pipeline
=============

Complete auto-roto pipeline: SAM + Depth + ViTMatte + Combine.

This is the main production pipeline that chains all stages together.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from auto_roto.stages.base import StageContext, StageResult, StageStatus
from auto_roto.stages.sam import SAMStage
from auto_roto.stages.depth import DepthStage
from auto_roto.stages.depth_expand import DepthExpandStage
from auto_roto.stages.vitmatte import ViTMatteStage
from auto_roto.stages.combine import CombineStage
from auto_roto.stages.matanyone import MatAnyoneStage
from auto_roto.config.pipeline import PipelineConfig
from auto_roto.config.presets import get_quality_preset, get_pipeline_preset
from auto_roto.utils.gpu import clear_gpu_memory

logger = logging.getLogger("AutoRoto.Pipelines.Full")


class FullPipeline:
    """
    Full auto-roto pipeline.

    Chains SAM + Depth + ViTMatte + Combine stages into a complete workflow.

    Usage:
        pipeline = FullPipeline(
            prompts=["person"],
            quality="high"
        )
        success = pipeline.run(
            input_path="video.mp4",
            output_dir="./output"
        )
    """

    def __init__(
        self,
        config: PipelineConfig = None,
        prompts: List[str] = None,
        box: str = None,
        quality: str = "standard",
        device: str = "cuda",
        output_format: str = "exr",
        bit_depth: int = 16,
        logger: logging.Logger = None
    ):
        """
        Initialize full pipeline.

        Args:
            config: PipelineConfig instance (overrides other args if provided)
            prompts: Text prompts for SAM segmentation
            box: Box prompt as "x1,y1,x2,y2"
            quality: Quality preset (draft/standard/high/ultra)
            device: Compute device (cuda/cpu)
            output_format: Output format (exr/png)
            bit_depth: Bit depth for output (8/16/32)
            logger: Optional logger instance
        """
        # Use provided config or create from args
        if config:
            self.config = config
        else:
            self.config = PipelineConfig(
                prompt=".".join(prompts) if prompts else "person",
                box=box or "",
                quality=quality,
                device=device,
                output_format=output_format,
                bit_depth=bit_depth
            )

        self._logger = logger or logging.getLogger("AutoRoto.FullPipeline")

        # Apply quality preset
        self._apply_quality_preset()

    def _apply_quality_preset(self):
        """Apply quality preset settings to config."""
        # Apply pipeline preset first if specified (overrides quality preset)
        if self.config.preset:
            self._apply_pipeline_preset()
            return

        preset = get_quality_preset(self.config.quality)

        # Apply depth model if not set
        if not self.config.depth_model:
            self.config.depth_model = preset.get('depth_model', 'base')

        # Apply depth settings if not set
        if self.config.depth_process_res is None:
            self.config.depth_process_res = preset.get('depth_process_res')
        if self.config.depth_process_method == "upper":
            self.config.depth_process_method = preset.get('depth_process_method', 'upper')
        if self.config.depth_norm_percentiles == (2.0, 98.0):
            self.config.depth_norm_percentiles = preset.get('depth_norm_percentiles', (2.0, 98.0))

        # Apply hair polish settings
        if self.config.hair_gamma is None:
            self.config.hair_gamma = preset.get('hair_gamma', 0.8)
        if self.config.hair_black_point is None:
            self.config.hair_black_point = preset.get('hair_black_point', 0.02)
        if self.config.hair_gain is None:
            self.config.hair_gain = preset.get('hair_gain', 1.1)

        # Apply guided filter settings
        if self.config.guided_filter_radius is None:
            self.config.guided_filter_radius = preset.get('guided_filter_radius', 4)
        if self.config.guided_filter_eps is None:
            self.config.guided_filter_eps = preset.get('guided_filter_eps', 1e-5)

    def _apply_pipeline_preset(self):
        """Apply pipeline preset settings to config."""
        preset_config = get_pipeline_preset(self.config.preset)
        if not preset_config:
            self._logger.warning(f"Unknown preset: {self.config.preset}")
            return

        self._logger.info(f"Applying pipeline preset: {preset_config.get('name', self.config.preset)}")

        settings = preset_config.get('settings', {})

        # Apply all settings from preset
        for key, value in settings.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def run(
        self,
        input_path: str,
        output_dir: str,
        frames_dir: str = None,
        skip_sam: bool = False,
        skip_depth: bool = False,
        skip_vitmatte: bool = False,
        skip_combine: bool = False,
        keep_intermediate: bool = False,
        verbose: bool = False
    ) -> bool:
        """
        Run the full pipeline.

        Args:
            input_path: Input video or frames directory
            output_dir: Output directory
            frames_dir: Optional pre-extracted frames directory
            skip_sam: Skip SAM stage (use existing output)
            skip_depth: Skip depth stage
            skip_vitmatte: Skip ViTMatte stage
            skip_combine: Skip combine stage
            keep_intermediate: Keep intermediate outputs
            verbose: Verbose output

        Returns:
            True if pipeline completed successfully
        """
        start_time = time.time()

        self._logger.info("="*60)
        self._logger.info("AUTO-ROTO FULL PIPELINE")
        self._logger.info("="*60)
        self._logger.info(f"Input: {input_path}")
        self._logger.info(f"Output: {output_dir}")
        self._logger.info(f"Quality: {self.config.quality}")
        self._logger.info(f"Depth Model: {self.config.depth_model}")
        self._logger.info("="*60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Resolve frames directory
        frames_path = self._resolve_frames_dir(input_path, output_path)

        # Create context
        context = StageContext(
            input_path=Path(input_path),
            output_dir=output_path,
            frames_dir=frames_path,
            device=self.config.device,
            quality=self.config.quality,
            logger=self._logger,
            verbose=verbose,
            config={
                "output_format": self.config.output_format,
                "bit_depth": self.config.bit_depth
            }
        )

        # Parse prompts
        prompts = self.config.prompt.split(".") if self.config.prompt else ["person"]

        # Track results
        results: Dict[str, StageResult] = {}

        # Determine if we're in MatAnyone temporal mode
        use_matanyone = self.config.use_matanyone
        first_frame_only = use_matanyone

        if use_matanyone:
            self._logger.info("MatAnyone mode: Processing only frame 1 for refinement stages")

        # =================================================================
        # STAGE 1: SAM Segmentation
        # =================================================================
        if not skip_sam:
            from auto_roto.config.sam import RotoConfig
            sam_config = RotoConfig(
                sam_imgsz=self.config.sam_imgsz,
                sam_conf=self.config.sam_conf,
                sam_retina_masks=self.config.sam_retina_masks,
                sam_max_det=self.config.sam_max_det
            )

            sam_stage = SAMStage(
                config=sam_config,
                prompts=prompts,
                box=self.config.box if self.config.box else None,
                first_frame_only=first_frame_only,
                logger=self._logger
            )

            result = sam_stage.execute(context)
            results["sam"] = result

            if result.failed:
                self._logger.error("Pipeline failed at SAM stage")
                return False

            clear_gpu_memory()
        else:
            self._logger.info("Skipping SAM stage (--skip-sam)")
            # Create mock result for skipped stage
            sam_output = output_path / "01_sam_output"
            if sam_output.exists():
                results["sam"] = StageResult.success_result(
                    output_dir=sam_output,
                    message="Using existing SAM output",
                    outputs={"alpha_dir": str(sam_output / "alpha")}
                )
                context.stage_outputs["sam"] = results["sam"]

        # =================================================================
        # STAGE 2: Depth Estimation
        # =================================================================
        if not skip_depth:
            from auto_roto.config.depth import DepthRefineConfig
            depth_config = DepthRefineConfig(
                depth_model=self.config.depth_model,
                depth_process_res=self.config.depth_process_res,
                depth_process_method=self.config.depth_process_method,
                depth_norm_percentiles=self.config.depth_norm_percentiles
            )

            depth_stage = DepthStage(
                config=depth_config,
                first_frame_only=first_frame_only,
                logger=self._logger
            )

            result = depth_stage.execute(context)
            results["depth"] = result

            if result.failed:
                self._logger.warning("Depth stage failed, continuing with SAM output")

            clear_gpu_memory()
        else:
            self._logger.info("Skipping depth stage (--skip-depth)")

        # =================================================================
        # STAGE 2b: Depth-Guided Mask Expansion (optional - cinema preset)
        # =================================================================
        if self.config.depth_expansion_enabled and "depth" in results and results["depth"].success:
            from auto_roto.config.depth_expansion import DepthExpansionConfig

            expansion_config = DepthExpansionConfig(
                enabled=True,
                depth_tolerance=self.config.depth_expansion_tolerance,
                percentile_range=self.config.depth_expansion_percentiles,
                require_connectivity=self.config.depth_expansion_connectivity,
                max_expansion_px=self.config.depth_expansion_max_px,
                edge_aware=self.config.depth_expansion_edge_aware,
            )

            depth_expand_stage = DepthExpandStage(
                config=expansion_config,
                first_frame_only=first_frame_only,
                logger=self._logger
            )

            result = depth_expand_stage.execute(context)
            results["depth_expand"] = result

            if result.failed:
                self._logger.warning("Depth expansion failed, continuing with SAM output")
            else:
                self._logger.info("Depth expansion complete - expanded masks will be used")

            clear_gpu_memory()

        # =================================================================
        # STAGE 3: ViTMatte Refinement
        # =================================================================
        if not skip_vitmatte and "depth" in results and results["depth"].success:
            from auto_roto.config.vitmatte import GeometricMatteConfig, TrimapConfig, ViTMatteConfig

            trimap_config = TrimapConfig(
                adaptive_mode=True,
                motion_aware=self.config.vitmatte_motion_aware,
                adaptive_base_px=self.config.vitmatte_adaptive_base,
                adaptive_max_px=self.config.vitmatte_adaptive_max
            )

            vitmatte_config = ViTMatteConfig()

            geometric_config = GeometricMatteConfig(
                trimap=trimap_config,
                vitmatte=vitmatte_config,
                hair_gamma=self.config.hair_gamma,
                hair_black_point=self.config.hair_black_point,
                hair_gain=self.config.hair_gain,
                hair_polish_enabled=self.config.hair_polish_enabled,
                guided_filter_radius=self.config.guided_filter_radius,
                guided_filter_eps=self.config.guided_filter_eps
            )

            vitmatte_stage = ViTMatteStage(
                config=geometric_config,
                save_trimap=True,
                first_frame_only=first_frame_only,
                logger=self._logger
            )

            result = vitmatte_stage.execute(context)
            results["vitmatte"] = result

            if result.failed:
                self._logger.warning("ViTMatte stage failed, continuing with SAM output")

            clear_gpu_memory()
        else:
            if skip_vitmatte:
                self._logger.info("Skipping ViTMatte stage (--skip-vitmatte)")
            else:
                self._logger.info("Skipping ViTMatte stage (depth not available)")

        # =================================================================
        # STAGE 4: Matte Combination
        # =================================================================
        if not skip_combine:
            from auto_roto.config.matte import MatteCombineConfig

            combine_config = MatteCombineConfig(
                core_erosion=self.config.core_shrink,
                despill_strength=self.config.despill_strength
            )

            combine_stage = CombineStage(
                config=combine_config,
                first_frame_only=first_frame_only,
                logger=self._logger
            )

            result = combine_stage.execute(context)
            results["combine"] = result

            if result.failed:
                self._logger.warning("Combine stage failed")
        else:
            self._logger.info("Skipping combine stage (--skip-combine)")

        # =================================================================
        # STAGE 5: MatAnyone Temporal Propagation (optional)
        # =================================================================
        if use_matanyone:
            matanyone_stage = MatAnyoneStage(
                repo_path=self.config.mam2_repo,
                checkpoint_path=self.config.mam2_checkpoint,
                mem_every=self.config.matanyone_mem_every,
                max_mem_frames=self.config.matanyone_max_mem_frames,
                warmup=self.config.matanyone_warmup,
                erode=self.config.matanyone_erode,
                dilate=self.config.matanyone_dilate,
                top_k=self.config.matanyone_top_k,
                use_long_term=self.config.matanyone_use_long_term,
                max_internal_size=self.config.matanyone_max_internal_size,
                logger=self._logger
            )

            result = matanyone_stage.execute(context)
            results["matanyone"] = result

            if result.failed:
                self._logger.error("MatAnyone stage failed")
                # Don't return False - still have combine output as fallback

            clear_gpu_memory()

        # =================================================================
        # FINALIZE
        # =================================================================
        final_output = output_path / "final"
        self._finalize_output(context, results, final_output)

        # Cleanup intermediate
        if not keep_intermediate:
            self._cleanup_intermediate(output_path, final_output)

        # Summary
        total_time = time.time() - start_time

        self._logger.info("\n" + "="*60)
        self._logger.info("PIPELINE COMPLETE")
        self._logger.info("="*60)
        self._logger.info(f"Total time: {total_time/60:.1f} minutes")
        self._logger.info(f"Output: {final_output}")

        # Check overall success
        success = all(
            r.success or r.skipped
            for r in results.values()
        )

        return success

    def _resolve_frames_dir(
        self,
        input_path: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Resolve or extract RGB frames directory."""
        import cv2

        input_path = Path(input_path)

        # Already a directory
        if input_path.is_dir():
            return input_path

        # Image file - use parent
        image_exts = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpx"}
        if input_path.is_file() and input_path.suffix.lower() in image_exts:
            return input_path.parent

        # Video file - extract frames
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"}
        if input_path.is_file() and input_path.suffix.lower() in video_exts:
            frames_dir = output_dir / "00_frames"

            # Check for existing frames
            if frames_dir.exists():
                existing = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
                if existing:
                    self._logger.info(f"Using existing frames: {frames_dir}")
                    return frames_dir
                shutil.rmtree(frames_dir, ignore_errors=True)

            frames_dir.mkdir(parents=True, exist_ok=True)

            # Extract frames
            self._logger.info(f"Extracting frames from: {input_path}")
            cap = cv2.VideoCapture(str(input_path))

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
                self._logger.error(f"No frames extracted from: {input_path}")
                return None

            self._logger.info(f"Extracted {idx} frames")
            return frames_dir

        return None

    def _finalize_output(
        self,
        context: StageContext,
        results: Dict[str, StageResult],
        final_output: Path
    ):
        """Copy final outputs to the final directory."""
        self._logger.info("\n" + "="*60)
        self._logger.info("FINALIZING OUTPUT")
        self._logger.info("="*60)

        final_output.mkdir(parents=True, exist_ok=True)

        # Determine source - use the last successful stage
        # MatAnyone takes priority when available since it has all frames
        source = None
        for stage_name in ["matanyone", "combine", "vitmatte", "sam"]:
            if stage_name in results and results[stage_name].success:
                source = results[stage_name].output_dir
                self._logger.info(f"Using output from: {stage_name}")
                break

        if not source:
            self._logger.warning("No successful stage output to finalize")
            return

        # Copy outputs
        for subdir in ["alpha", "rgb", "rgba", "preview"]:
            src = source / subdir
            if src.exists():
                dst = final_output / subdir
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                self._logger.info(f"  Copied {subdir}/ to final output")

        # Copy depth if available
        if "depth" in results and results["depth"].success:
            depth_src = results["depth"].output_dir / "depth"
            if depth_src.exists():
                depth_dst = final_output / "depth"
                if depth_dst.exists():
                    shutil.rmtree(depth_dst)
                shutil.copytree(depth_src, depth_dst)
                self._logger.info("  Copied depth/ to final output")

    def _cleanup_intermediate(self, output_dir: Path, final_output: Path):
        """Remove intermediate stage outputs."""
        self._logger.info("Cleaning up intermediate files...")

        intermediate_dirs = [
            "00_frames",
            "01_sam_output",
            "02_depth_output",
            "02b_depth_expand_output",
            "03_vitmatte_output",
            "04_combine_output",
            "05_matanyone_output"
        ]

        for dirname in intermediate_dirs:
            dirpath = output_dir / dirname
            if dirpath.exists() and dirpath != final_output:
                shutil.rmtree(dirpath, ignore_errors=True)
                self._logger.debug(f"  Removed {dirname}")
