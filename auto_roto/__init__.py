"""
AUTO-ROTO: Production-Grade Automatic Rotoscoping Pipeline
==========================================================

A comprehensive video rotoscoping library using SAM3, Depth Anything 3,
ViTMatte, and MatAnyone for professional-quality alpha matte extraction.

Basic Usage:
    from auto_roto import AutoRotoPipeline, RotoConfig

    config = RotoConfig(
        input_path="video.mp4",
        output_dir="./output",
        prompt="person"
    )
    pipeline = AutoRotoPipeline(config)
    pipeline.run()

Full Pipeline:
    from auto_roto import FullPipeline, PipelineConfig

    config = PipelineConfig(
        input_path="video.mp4",
        output_dir="./output",
        prompt="person",
        quality="high"
    )
    pipeline = FullPipeline(config)
    pipeline.run()

Author: Built for Core Form VFX
License: MIT
"""

__version__ = "5.0.0"
__author__ = "Core Form VFX"

# Lazy imports to avoid heavy dependencies on import
def __getattr__(name):
    """Lazy import of heavy modules."""

    # Configuration classes
    if name == "RotoConfig":
        from auto_roto.config.sam import RotoConfig
        return RotoConfig
    if name == "DepthRefineConfig":
        from auto_roto.config.depth import DepthRefineConfig
        return DepthRefineConfig
    if name == "PipelineConfig":
        from auto_roto.config.pipeline import PipelineConfig
        return PipelineConfig
    if name == "TrimapConfig":
        from auto_roto.config.vitmatte import TrimapConfig
        return TrimapConfig
    if name == "ViTMatteConfig":
        from auto_roto.config.vitmatte import ViTMatteConfig
        return ViTMatteConfig
    if name == "GeometricMatteConfig":
        from auto_roto.config.vitmatte import GeometricMatteConfig
        return GeometricMatteConfig
    if name == "MatAnyoneConfig":
        from auto_roto.config.matanyone import MatAnyoneConfig
        return MatAnyoneConfig

    # Pipeline classes
    if name == "AutoRotoPipeline":
        from auto_roto.pipelines.sam_pipeline import AutoRotoPipeline
        return AutoRotoPipeline
    if name == "DepthRefinePipeline":
        from auto_roto.pipelines.depth_pipeline import DepthRefinePipeline
        return DepthRefinePipeline
    if name == "FullPipeline":
        from auto_roto.pipelines.full_pipeline import FullPipeline
        return FullPipeline

    # Model wrappers
    if name == "SAM3Segmenter":
        from auto_roto.models.sam3 import SAM3Segmenter
        return SAM3Segmenter
    if name == "DepthEstimator":
        from auto_roto.models.depth_anything import DepthEstimator
        return DepthEstimator
    if name == "ViTMatteRefiner":
        from auto_roto.models.vitmatte import ViTMatteRefiner
        return ViTMatteRefiner
    if name == "MatAnyoneAdapter":
        from auto_roto.models.matanyone import MatAnyoneAdapter
        return MatAnyoneAdapter

    # Refiners
    if name == "AlphaRefiner":
        from auto_roto.refiners.alpha import AlphaRefiner
        return AlphaRefiner
    if name == "DepthGuidedRefiner":
        from auto_roto.refiners.depth import DepthGuidedRefiner
        return DepthGuidedRefiner
    if name == "TrimapSynthesizer":
        from auto_roto.refiners.trimap import TrimapSynthesizer
        return TrimapSynthesizer
    if name == "GeometricMatteRefiner":
        from auto_roto.refiners.geometric import GeometricMatteRefiner
        return GeometricMatteRefiner

    # Stages
    if name == "StageResult":
        from auto_roto.stages.base import StageResult
        return StageResult
    if name == "StageContext":
        from auto_roto.stages.base import StageContext
        return StageContext
    if name == "PipelineStage":
        from auto_roto.stages.base import PipelineStage
        return PipelineStage
    if name == "SAMStage":
        from auto_roto.stages.sam import SAMStage
        return SAMStage
    if name == "DepthStage":
        from auto_roto.stages.depth import DepthStage
        return DepthStage
    if name == "ViTMatteStage":
        from auto_roto.stages.vitmatte import ViTMatteStage
        return ViTMatteStage
    if name == "CombineStage":
        from auto_roto.stages.combine import CombineStage
        return CombineStage

    # Matte operations
    if name == "MatteCombiner":
        from auto_roto.matte.combiner import MatteCombiner
        return MatteCombiner
    if name == "Despill":
        from auto_roto.matte.despill import Despill
        return Despill
    if name == "MatteBlender":
        from auto_roto.matte.blender import MatteBlender
        return MatteBlender

    # I/O classes
    if name == "VideoReader":
        from auto_roto.io.video import VideoReader
        return VideoReader
    if name == "FrameWriter":
        from auto_roto.io.frames import FrameWriter
        return FrameWriter

    # Utilities
    if name == "setup_logging":
        from auto_roto.utils.logging import setup_logging
        return setup_logging
    if name == "clear_gpu_memory":
        from auto_roto.utils.gpu import clear_gpu_memory
        return clear_gpu_memory

    raise AttributeError(f"module 'auto_roto' has no attribute '{name}'")


# Define what's available for tab completion
__all__ = [
    # Version
    "__version__",
    "__author__",

    # Configuration
    "RotoConfig",
    "DepthRefineConfig",
    "PipelineConfig",
    "TrimapConfig",
    "ViTMatteConfig",
    "GeometricMatteConfig",
    "MatAnyoneConfig",

    # Pipelines
    "AutoRotoPipeline",
    "DepthRefinePipeline",
    "FullPipeline",

    # Models
    "SAM3Segmenter",
    "DepthEstimator",
    "ViTMatteRefiner",
    "MatAnyoneAdapter",

    # Refiners
    "AlphaRefiner",
    "DepthGuidedRefiner",
    "TrimapSynthesizer",
    "GeometricMatteRefiner",

    # Stages
    "StageResult",
    "StageContext",
    "PipelineStage",
    "SAMStage",
    "DepthStage",
    "ViTMatteStage",
    "CombineStage",

    # Matte operations
    "MatteCombiner",
    "Despill",
    "MatteBlender",

    # I/O
    "VideoReader",
    "FrameWriter",

    # Utilities
    "setup_logging",
    "clear_gpu_memory",
]
