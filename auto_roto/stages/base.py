"""
Pipeline Stage Base Classes
===========================

Base classes for pipeline stages with result tracking and context management.

Classes:
    - StageResult: Result of a stage execution
    - StageContext: Shared context passed between stages
    - PipelineStage: Abstract base class for stages
    - StageRegistry: Registry for stage management
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger("AutoRoto.Stages")


class StageStatus(Enum):
    """Status of a stage execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """
    Result of a stage execution.

    Attributes:
        status: Execution status
        output_dir: Directory containing stage outputs
        message: Human-readable status message
        duration: Execution time in seconds
        outputs: Dictionary of output paths/values
        error: Exception if failed
    """
    status: StageStatus = StageStatus.PENDING
    output_dir: Optional[Path] = None
    message: str = ""
    duration: float = 0.0
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        """Check if stage completed successfully."""
        return self.status == StageStatus.SUCCESS

    @property
    def failed(self) -> bool:
        """Check if stage failed."""
        return self.status == StageStatus.FAILED

    @property
    def skipped(self) -> bool:
        """Check if stage was skipped."""
        return self.status == StageStatus.SKIPPED

    @classmethod
    def success_result(
        cls,
        output_dir: Path,
        message: str = "",
        duration: float = 0.0,
        outputs: Dict[str, Any] = None
    ) -> "StageResult":
        """Create a success result."""
        return cls(
            status=StageStatus.SUCCESS,
            output_dir=output_dir,
            message=message,
            duration=duration,
            outputs=outputs or {}
        )

    @classmethod
    def failure_result(
        cls,
        error: Exception,
        message: str = "",
        duration: float = 0.0
    ) -> "StageResult":
        """Create a failure result."""
        return cls(
            status=StageStatus.FAILED,
            message=message or str(error),
            duration=duration,
            error=error
        )

    @classmethod
    def skipped_result(cls, message: str = "Skipped") -> "StageResult":
        """Create a skipped result."""
        return cls(
            status=StageStatus.SKIPPED,
            message=message
        )


@dataclass
class StageContext:
    """
    Shared context passed between pipeline stages.

    Attributes:
        input_path: Original input video/frames path
        output_dir: Base output directory
        frames_dir: Directory containing RGB frames
        device: Compute device (cuda/cpu)
        quality: Quality preset name
        logger: Logger instance
        verbose: Verbose output flag
        stage_outputs: Results from previous stages
        config: Additional configuration dictionary
    """
    input_path: Path
    output_dir: Path
    frames_dir: Optional[Path] = None
    device: str = "cuda"
    quality: str = "standard"
    logger: logging.Logger = None
    verbose: bool = False
    stage_outputs: Dict[str, StageResult] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure paths are Path objects and logger exists."""
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        if self.frames_dir:
            self.frames_dir = Path(self.frames_dir)
        if self.logger is None:
            self.logger = logging.getLogger("AutoRoto.Pipeline")

    def get_stage_output(self, stage_name: str) -> Optional[StageResult]:
        """Get the result of a previous stage."""
        return self.stage_outputs.get(stage_name)

    def get_alpha_dir(self, stage_name: str) -> Optional[Path]:
        """Get the alpha output directory from a stage."""
        result = self.get_stage_output(stage_name)
        if result and result.success:
            alpha_dir = result.outputs.get("alpha_dir")
            if alpha_dir:
                return Path(alpha_dir)
            # Fallback to output_dir/alpha
            if result.output_dir:
                alpha_path = result.output_dir / "alpha"
                if alpha_path.exists():
                    return alpha_path
        return None

    def get_depth_dir(self, stage_name: str) -> Optional[Path]:
        """Get the depth output directory from a stage."""
        result = self.get_stage_output(stage_name)
        if result and result.success:
            depth_dir = result.outputs.get("depth_dir")
            if depth_dir:
                return Path(depth_dir)
            # Fallback to output_dir/depth
            if result.output_dir:
                depth_path = result.output_dir / "depth"
                if depth_path.exists():
                    return depth_path
        return None


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Subclasses must implement:
        - name: Stage name property
        - run(): Execute the stage

    Optionally override:
        - validate_inputs(): Validate stage inputs
        - cleanup(): Cleanup after execution
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize stage.

        Args:
            logger: Optional logger instance
        """
        self._logger = logger or logging.getLogger(f"AutoRoto.Stage.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name used for identification and output directories."""
        pass

    @property
    def description(self) -> str:
        """Human-readable stage description."""
        return f"{self.name} stage"

    @abstractmethod
    def run(self, context: StageContext) -> StageResult:
        """
        Execute the stage.

        Args:
            context: Pipeline context with inputs and configuration

        Returns:
            StageResult with execution status and outputs
        """
        pass

    def validate_inputs(self, context: StageContext) -> Optional[str]:
        """
        Validate stage inputs.

        Args:
            context: Pipeline context

        Returns:
            Error message if validation fails, None if valid
        """
        return None

    def cleanup(self, context: StageContext):
        """
        Cleanup after stage execution.

        Override to release resources, clear GPU memory, etc.
        """
        pass

    def execute(self, context: StageContext) -> StageResult:
        """
        Execute stage with timing, validation, and error handling.

        This is the main entry point - it wraps run() with:
        - Input validation
        - Timing
        - Error handling
        - Logging
        - Cleanup

        Args:
            context: Pipeline context

        Returns:
            StageResult with execution status
        """
        self._logger.info(f"\n{'='*60}")
        self._logger.info(f"STAGE: {self.description}")
        self._logger.info(f"{'='*60}")

        # Validate inputs
        validation_error = self.validate_inputs(context)
        if validation_error:
            self._logger.error(f"Validation failed: {validation_error}")
            return StageResult.failure_result(
                ValueError(validation_error),
                message=f"Validation failed: {validation_error}"
            )

        start_time = time.time()

        try:
            result = self.run(context)
            result.duration = time.time() - start_time

            if result.success:
                self._logger.info(
                    f"  {self.name} completed in {result.duration:.1f}s"
                )
            elif result.skipped:
                self._logger.info(f"  {self.name} skipped: {result.message}")
            else:
                self._logger.error(f"  {self.name} failed: {result.message}")

            # Store result in context
            context.stage_outputs[self.name] = result

            return result

        except Exception as e:
            duration = time.time() - start_time
            self._logger.error(f"  {self.name} failed with exception: {e}")
            import traceback
            traceback.print_exc()

            result = StageResult.failure_result(e, duration=duration)
            context.stage_outputs[self.name] = result
            return result

        finally:
            self.cleanup(context)


class StageRegistry:
    """
    Registry for managing pipeline stages.

    Allows registration and retrieval of stage classes by name.
    """

    _stages: Dict[str, Type[PipelineStage]] = {}

    @classmethod
    def register(cls, stage_class: Type[PipelineStage]) -> Type[PipelineStage]:
        """
        Register a stage class.

        Can be used as a decorator:
            @StageRegistry.register
            class MyStage(PipelineStage):
                ...
        """
        # Create instance to get name
        instance = stage_class.__new__(stage_class)
        if hasattr(stage_class, 'name') and isinstance(stage_class.name, property):
            # Need to call __init__ first for property
            stage_class.__init__(instance)
            name = instance.name
        else:
            name = getattr(stage_class, 'name', stage_class.__name__)

        cls._stages[name] = stage_class
        return stage_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[PipelineStage]]:
        """Get a stage class by name."""
        return cls._stages.get(name)

    @classmethod
    def list_stages(cls) -> List[str]:
        """List all registered stage names."""
        return list(cls._stages.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[PipelineStage]:
        """Create a stage instance by name."""
        stage_class = cls.get(name)
        if stage_class:
            return stage_class(**kwargs)
        return None
