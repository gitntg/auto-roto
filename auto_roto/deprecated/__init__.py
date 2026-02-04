"""
Deprecated Module Wrappers
==========================

Backward-compatible wrappers for old import paths.

This module provides thin wrappers that emit deprecation warnings
and redirect to the new module locations.

Usage:
    # Old import (deprecated, but still works)
    from auto_roto.deprecated import SAM3Segmenter

    # New import (preferred)
    from auto_roto.models.sam3 import SAM3Segmenter
"""

import warnings

# Re-export with deprecation warnings


def _deprecated_import(name, new_module, old_module="auto_roto.deprecated"):
    """Emit deprecation warning for old imports."""
    warnings.warn(
        f"Importing {name} from {old_module} is deprecated. "
        f"Use 'from {new_module} import {name}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


def __getattr__(name):
    """Lazy import with deprecation warnings."""

    # Model wrappers
    if name == "SAM3Segmenter":
        _deprecated_import(name, "auto_roto.models.sam3")
        from auto_roto.models.sam3 import SAM3Segmenter
        return SAM3Segmenter

    if name == "DepthEstimator":
        _deprecated_import(name, "auto_roto.models.depth_anything")
        from auto_roto.models.depth_anything import DepthEstimator
        return DepthEstimator

    if name == "ViTMatteRefiner":
        _deprecated_import(name, "auto_roto.models.vitmatte")
        from auto_roto.models.vitmatte import ViTMatteRefiner
        return ViTMatteRefiner

    if name == "Mam2Adapter":
        _deprecated_import(name, "auto_roto.models.matanyone", "auto_roto.deprecated")
        warnings.warn(
            "Mam2Adapter is renamed to MatAnyoneAdapter. "
            "Use 'from auto_roto.models.matanyone import MatAnyoneAdapter' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from auto_roto.models.matanyone import MatAnyoneAdapter
        return MatAnyoneAdapter

    # Refiners
    if name == "AlphaRefiner":
        _deprecated_import(name, "auto_roto.refiners.alpha")
        from auto_roto.refiners.alpha import AlphaRefiner
        return AlphaRefiner

    if name == "DepthGuidedRefiner":
        _deprecated_import(name, "auto_roto.refiners.depth")
        from auto_roto.refiners.depth import DepthGuidedRefiner
        return DepthGuidedRefiner

    if name == "TrimapSynthesizer":
        _deprecated_import(name, "auto_roto.refiners.trimap")
        from auto_roto.refiners.trimap import TrimapSynthesizer
        return TrimapSynthesizer

    # I/O
    if name == "VideoReader":
        _deprecated_import(name, "auto_roto.io.video")
        from auto_roto.io.video import VideoReader
        return VideoReader

    if name == "FrameWriter":
        _deprecated_import(name, "auto_roto.io.frames")
        from auto_roto.io.frames import FrameWriter
        return FrameWriter

    raise AttributeError(f"module 'auto_roto.deprecated' has no attribute '{name}'")


__all__ = [
    "SAM3Segmenter",
    "DepthEstimator",
    "ViTMatteRefiner",
    "Mam2Adapter",
    "AlphaRefiner",
    "DepthGuidedRefiner",
    "TrimapSynthesizer",
    "VideoReader",
    "FrameWriter",
]
