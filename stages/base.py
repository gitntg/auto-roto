"""
Stage Pattern Base Classes
==========================

Provides abstract base class for pipeline stages and a registry for stage management.
This eliminates the repeated pattern in full_pipeline_v5.py where each stage follows:
    1. Check if skipped
    2. Find script
    3. Build arguments
    4. Run stage
    5. Handle failure with fallback
    6. Clear GPU memory

Now each stage encapsulates its own configuration and execution logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import sys
import subprocess
import time
import gc
import logging

logger = logging.getLogger("AutoRoto.Stages")


@dataclass
class StageResult:
    """Result of a stage execution."""
    success: bool
    output_dir: Path
    duration: float = 0.0
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, fallback_dir: Path, error: str = "") -> "StageResult":
        """Create a failure result with fallback."""
        return cls(success=False, output_dir=fallback_dir, error_message=error)

    @classmethod
    def skipped(cls, fallback_dir: Path) -> "StageResult":
        """Create a skipped result (not an error)."""
        return cls(success=True, output_dir=fallback_dir, metrics={"skipped": True})


@dataclass
class StageContext:
    """Shared context passed between pipeline stages."""

    # Input configuration
    input_path: str
    output_base: Path
    verbose: bool = False

    # Output format
    output_format: str = "exr"
    bit_depth: int = 16

    # Quality settings (populated from preset)
    sam_model: str = "base_plus"
    depth_model: str = "base"

    # Stage-specific settings (populated from config)
    settings: Dict[str, Any] = field(default_factory=dict)

    # Runtime state - outputs from previous stages
    sam_output: Optional[Path] = None
    depth_output: Optional[Path] = None
    vitmatte_output: Optional[Path] = None
    edge_output: Optional[Path] = None
    temporal_output: Optional[Path] = None
    combine_output: Optional[Path] = None
    hair_output: Optional[Path] = None

    # Current alpha source (updated after each stage)
    current_alpha: Optional[Path] = None

    def get_alpha_source(self) -> Path:
        """Get the most recent alpha output directory."""
        if self.current_alpha:
            return self.current_alpha
        raise ValueError("No alpha source available yet")


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage must implement:
        - name: Human-readable stage name
        - order: Execution order (lower = earlier)
        - script_name: Python script to execute
        - build_args(): Build command-line arguments
        - is_enabled(): Check if stage should run

    Optional overrides:
        - get_output_subdir(): Directory name for stage output
        - on_success(): Post-processing after successful run
        - on_failure(): Handling when stage fails
        - requires_gpu_cleanup: Whether to clear GPU after stage
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name for logging."""
        pass

    @property
    @abstractmethod
    def order(self) -> int:
        """Execution order (lower numbers run first)."""
        pass

    @property
    @abstractmethod
    def script_name(self) -> str:
        """Name of the Python script to execute."""
        pass

    @property
    def requires_gpu_cleanup(self) -> bool:
        """Whether to clear GPU memory after this stage. Default True."""
        return True

    @property
    def is_critical(self) -> bool:
        """If True, pipeline stops on failure. If False, continues with fallback."""
        return False

    def get_output_subdir(self, order_prefix: bool = True) -> str:
        """Get the output subdirectory name for this stage."""
        base_name = self.name.lower().replace(" ", "_")
        if order_prefix:
            return f"{self.order:02d}_{base_name}"
        return base_name

    @abstractmethod
    def is_enabled(self, context: StageContext) -> bool:
        """Check if this stage should run based on config."""
        pass

    @abstractmethod
    def build_args(self, context: StageContext, output_dir: Path) -> List[str]:
        """Build command-line arguments for the stage script."""
        pass

    def get_fallback_output(self, context: StageContext) -> Path:
        """Get the fallback output directory if this stage fails or is skipped."""
        # Default: use the current alpha source
        return context.get_alpha_source()

    def on_success(self, context: StageContext, result: StageResult) -> None:
        """Called after successful stage execution. Update context here."""
        pass

    def on_failure(self, context: StageContext, result: StageResult) -> None:
        """Called when stage fails. Can log warnings or adjust context."""
        logger.warning(f"{self.name} failed, continuing with fallback output")

    def execute(self, context: StageContext, script_dir: Path) -> StageResult:
        """
        Execute this pipeline stage.

        This is the main entry point that handles:
        - Skip check
        - Output directory creation
        - Script execution
        - Success/failure handling
        - GPU memory cleanup
        """
        # Check if stage should run
        if not self.is_enabled(context):
            logger.info(f"Skipping {self.name} (disabled)")
            fallback = self.get_fallback_output(context)
            return StageResult.skipped(fallback)

        # Setup output directory
        output_dir = context.output_base / self.get_output_subdir()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find and validate script
        script_path = script_dir / self.script_name
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            fallback = self.get_fallback_output(context)
            return StageResult.failure(fallback, f"Script not found: {self.script_name}")

        # Build arguments
        args = self.build_args(context, output_dir)

        # Execute
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE: {self.name}")
        logger.info(f"{'='*60}")

        if context.verbose:
            cmd_str = f"python {script_path} {' '.join(args)}"
            logger.info(f"Command: {cmd_str}")

        start_time = time.time()
        success = self._run_script(script_path, args, context.verbose)
        duration = time.time() - start_time

        if success:
            logger.info(f"  {self.name} completed in {duration:.1f}s")
            result = StageResult(
                success=True,
                output_dir=output_dir,
                duration=duration
            )
            self.on_success(context, result)
        else:
            fallback = self.get_fallback_output(context)
            result = StageResult.failure(fallback, f"{self.name} execution failed")
            result.duration = duration
            self.on_failure(context, result)

            if self.is_critical:
                logger.error(f"Critical stage {self.name} failed - stopping pipeline")

        # GPU cleanup
        if self.requires_gpu_cleanup:
            clear_gpu_memory()

        return result

    def _run_script(self, script_path: Path, args: List[str], verbose: bool) -> bool:
        """Run the stage script as a subprocess."""
        cmd = [sys.executable, str(script_path)] + args

        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=not verbose
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"  {self.name} failed with exit code {e.returncode}")
            if e.stderr:
                logger.error(f"  Error: {e.stderr[:500]}")
            return False


class StageRegistry:
    """
    Registry for pipeline stages.

    Manages stage registration, ordering, and execution.
    Replaces the hardcoded stage blocks in run_pipeline().
    """

    def __init__(self):
        self._stages: Dict[str, PipelineStage] = {}

    def register(self, stage: PipelineStage) -> "StageRegistry":
        """Register a stage. Returns self for chaining."""
        self._stages[stage.name] = stage
        return self

    def get_stages_in_order(self) -> List[PipelineStage]:
        """Get all registered stages sorted by execution order."""
        return sorted(self._stages.values(), key=lambda s: s.order)

    def get_enabled_stages(self, context: StageContext) -> List[PipelineStage]:
        """Get only the enabled stages in execution order."""
        return [s for s in self.get_stages_in_order() if s.is_enabled(context)]

    def run_all(
        self,
        context: StageContext,
        script_dir: Path,
        stop_on_critical_failure: bool = True
    ) -> List[StageResult]:
        """
        Run all registered stages in order.

        Args:
            context: Shared pipeline context
            script_dir: Directory containing stage scripts
            stop_on_critical_failure: Stop if a critical stage fails

        Returns:
            List of StageResult objects for each stage
        """
        results = []

        for stage in self.get_stages_in_order():
            result = stage.execute(context, script_dir)
            results.append(result)

            # Update context with output
            self._update_context_output(stage, context, result)

            # Check for critical failure
            if not result.success and stage.is_critical and stop_on_critical_failure:
                logger.error(f"Pipeline stopped due to critical failure in {stage.name}")
                break

        return results

    def _update_context_output(
        self,
        stage: PipelineStage,
        context: StageContext,
        result: StageResult
    ) -> None:
        """Update context with stage output directory."""
        # Update the appropriate context field based on stage name
        stage_name_lower = stage.name.lower()

        if "sam" in stage_name_lower:
            context.sam_output = result.output_dir
            context.current_alpha = result.output_dir / "alpha"
        elif "depth" in stage_name_lower:
            context.depth_output = result.output_dir
            context.current_alpha = result.output_dir / "alpha"
        elif "vitmatte" in stage_name_lower:
            context.vitmatte_output = result.output_dir
            context.current_alpha = result.output_dir / "alpha"
        elif "edge" in stage_name_lower:
            context.edge_output = result.output_dir
            context.current_alpha = result.output_dir / "alpha"
        elif "temporal" in stage_name_lower:
            context.temporal_output = result.output_dir
            context.current_alpha = result.output_dir / "alpha"
        elif "combine" in stage_name_lower or "matte" in stage_name_lower:
            context.combine_output = result.output_dir
            context.current_alpha = result.output_dir / "alpha"
        elif "hair" in stage_name_lower:
            context.hair_output = result.output_dir
            context.current_alpha = result.output_dir / "alpha"

        # Ensure alpha path exists, fallback to output_dir
        if context.current_alpha and not context.current_alpha.exists():
            context.current_alpha = result.output_dir


def clear_gpu_memory():
    """Clear GPU memory between stages."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    gc.collect()
    logger.debug("GPU memory cleared")
