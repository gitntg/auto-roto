"""
Pipelines
=========

High-level pipeline orchestrators for auto-roto.

Pipelines compose stages into complete workflows:
    - AutoRotoPipeline: SAM-only segmentation
    - DepthRefinePipeline: Depth-guided refinement
    - FullPipeline: Complete SAM + Depth + ViTMatte + Combine workflow
    - InteractiveMatAnyonePipeline: Interactive SAM3 → MatAnyone workflow
"""

from auto_roto.pipelines.sam_pipeline import AutoRotoPipeline
from auto_roto.pipelines.depth_pipeline import DepthRefinePipeline
from auto_roto.pipelines.full_pipeline import FullPipeline
from auto_roto.pipelines.interactive_matanyone import InteractiveMatAnyonePipeline

__all__ = [
    'AutoRotoPipeline',
    'DepthRefinePipeline',
    'FullPipeline',
    'InteractiveMatAnyonePipeline',
]
