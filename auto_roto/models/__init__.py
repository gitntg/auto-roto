"""
Model Wrappers
==============

Clean inference-only wrappers for ML models.

Each model wrapper:
- Handles model loading and device management
- Provides a consistent inference interface
- Manages GPU memory cleanup
- Supports context manager usage
"""

from auto_roto.models.sam3 import SAM3Segmenter
from auto_roto.models.depth_anything import DepthEstimator
from auto_roto.models.vitmatte import ViTMatteRefiner
from auto_roto.models.matanyone import MatAnyoneAdapter, load_matanyone_adapter

__all__ = [
    'SAM3Segmenter',
    'DepthEstimator',
    'ViTMatteRefiner',
    'MatAnyoneAdapter',
    'load_matanyone_adapter',
]
