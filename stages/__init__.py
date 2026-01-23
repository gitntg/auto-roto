"""
AUTO-ROTO Stage Pattern Implementation
======================================

Provides a registry-based pipeline architecture that eliminates repeated patterns
in the orchestrator and enables procedural stage execution.

Usage:
    from stages import StageRegistry, StageContext, create_default_registry

    registry = create_default_registry()
    context = StageContext(
        input_path="/path/to/input",
        output_base=Path("/path/to/output"),
        settings={"skip_depth": True, "prompt": "person"}
    )

    results = registry.run_all(context, script_dir=Path(__file__).parent)
"""

from .base import (
    PipelineStage,
    StageResult,
    StageContext,
    StageRegistry,
    clear_gpu_memory,
)

from .pipeline_stages import (
    SAMStage,
    DepthStage,
    ViTMatteStage,
    EdgeStage,
    TemporalStage,
    CombineStage,
    HairStage,
    create_default_registry,
)

__all__ = [
    # Base classes
    'PipelineStage',
    'StageResult',
    'StageContext',
    'StageRegistry',
    'clear_gpu_memory',
    # Concrete stages
    'SAMStage',
    'DepthStage',
    'ViTMatteStage',
    'EdgeStage',
    'TemporalStage',
    'CombineStage',
    'HairStage',
    # Factory
    'create_default_registry',
]
