"""
Configuration Classes
=====================

Centralized configuration dataclasses for all pipeline stages.

Modules:
    base - Base configuration with validation
    pipeline - Full pipeline configuration
    presets - Quality presets (draft/standard/high/ultra)
    sam - SAM3 segmentation configuration
    depth - Depth refinement configuration
    vitmatte - ViTMatte and trimap configuration
    matanyone - MatAnyone adapter configuration
    matte - Matte combination configuration
"""

from auto_roto.config.sam import RotoConfig
from auto_roto.config.depth import DepthRefineConfig
from auto_roto.config.vitmatte import TrimapConfig, ViTMatteConfig, GeometricMatteConfig
from auto_roto.config.matanyone import MatAnyoneConfig
from auto_roto.config.pipeline import PipelineConfig
from auto_roto.config.presets import get_quality_preset, QUALITY_PRESETS
from auto_roto.config.matte import MatteCombineConfig, MatteBlendConfig

__all__ = [
    # Configuration classes
    "RotoConfig",
    "DepthRefineConfig",
    "TrimapConfig",
    "ViTMatteConfig",
    "GeometricMatteConfig",
    "MatAnyoneConfig",
    "PipelineConfig",
    "MatteCombineConfig",
    "MatteBlendConfig",
    # Presets
    "get_quality_preset",
    "QUALITY_PRESETS",
]
