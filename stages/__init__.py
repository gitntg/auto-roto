"""
AUTO-ROTO Stage Pattern Implementation
======================================

Provides a registry-based pipeline architecture that eliminates repeated patterns
in the orchestrator and enables procedural stage execution.

Usage:
    from pathlib import Path
    from stages import StageRegistry, StageContext, create_default_registry
    from full_pipeline_v5 import PipelineConfig

    # Create config (single source of truth for all settings)
    config = PipelineConfig(
        input_path="/path/to/input",
        output_dir="/path/to/output",
        prompt="person",
        quality="high",
    )

    # Create context from config (no manual settings dict needed)
    context = StageContext.from_pipeline_config(config, Path(config.output_dir))

    # Run all enabled stages
    registry = create_default_registry()
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
