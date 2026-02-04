"""
Pipeline Stages
===============

Modular stage implementations for the auto-roto pipeline.

Each stage encapsulates a specific processing step and can be
composed into pipelines with proper error handling and progress tracking.

Stages:
    - base: Base stage classes and utilities
    - sam: SAM3 segmentation stage
    - depth: Depth estimation stage
    - depth_expand: Depth-guided mask expansion stage (cinema preset)
    - vitmatte: ViTMatte alpha refinement stage
    - combine: Matte combination stage
    - matanyone: MatAnyone temporal propagation stage
"""

from auto_roto.stages.base import (
    StageResult,
    StageContext,
    PipelineStage,
    StageRegistry,
)
from auto_roto.stages.sam import SAMStage
from auto_roto.stages.depth import DepthStage
from auto_roto.stages.depth_expand import DepthExpandStage
from auto_roto.stages.vitmatte import ViTMatteStage
from auto_roto.stages.combine import CombineStage
from auto_roto.stages.matanyone import MatAnyoneStage

__all__ = [
    # Base
    'StageResult',
    'StageContext',
    'PipelineStage',
    'StageRegistry',
    # Concrete stages
    'SAMStage',
    'DepthStage',
    'DepthExpandStage',
    'ViTMatteStage',
    'CombineStage',
    'MatAnyoneStage',
]
