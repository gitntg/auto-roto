"""
Matte Blender
=============

Blend multiple alpha sources with various blend modes.

Useful for combining results from different segmentation methods.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger("AutoRoto.Matte.Blender")


class MatteBlender:
    """
    Blend multiple alpha sources.

    Supports various blend modes:
        - maximum: Take maximum alpha value
        - minimum: Take minimum alpha value
        - average: Average all alpha values
        - weighted: Weighted average
        - multiply: Multiply alpha values
        - screen: Screen blend mode
    """

    def __init__(
        self,
        mode: str = "maximum",
        weights: List[float] = None,
        logger: logging.Logger = None
    ):
        """
        Initialize matte blender.

        Args:
            mode: Blend mode (maximum/minimum/average/weighted/multiply/screen)
            weights: Weights for weighted blend (must match number of sources)
            logger: Optional logger instance
        """
        self.mode = mode
        self.weights = weights
        self.logger = logger or logging.getLogger("MatteBlender")

    def blend(self, sources: List[np.ndarray]) -> np.ndarray:
        """
        Blend multiple alpha sources.

        Args:
            sources: List of alpha mattes (H, W), float 0-1

        Returns:
            Blended alpha matte (H, W)
        """
        if not sources:
            raise ValueError("No sources provided")

        if len(sources) == 1:
            return sources[0].copy()

        # Ensure all sources are float32
        sources = [s.astype(np.float32) for s in sources]

        # Ensure all sources are same shape
        shapes = [s.shape for s in sources]
        if len(set(shapes)) > 1:
            raise ValueError(f"Source shapes don't match: {shapes}")

        # Apply blend mode
        if self.mode == "maximum":
            return self._blend_maximum(sources)
        elif self.mode == "minimum":
            return self._blend_minimum(sources)
        elif self.mode == "average":
            return self._blend_average(sources)
        elif self.mode == "weighted":
            return self._blend_weighted(sources)
        elif self.mode == "multiply":
            return self._blend_multiply(sources)
        elif self.mode == "screen":
            return self._blend_screen(sources)
        else:
            self.logger.warning(f"Unknown blend mode: {self.mode}, using maximum")
            return self._blend_maximum(sources)

    def _blend_maximum(self, sources: List[np.ndarray]) -> np.ndarray:
        """Take maximum value across all sources."""
        result = sources[0].copy()
        for src in sources[1:]:
            result = np.maximum(result, src)
        return result

    def _blend_minimum(self, sources: List[np.ndarray]) -> np.ndarray:
        """Take minimum value across all sources."""
        result = sources[0].copy()
        for src in sources[1:]:
            result = np.minimum(result, src)
        return result

    def _blend_average(self, sources: List[np.ndarray]) -> np.ndarray:
        """Average all sources."""
        stack = np.stack(sources, axis=-1)
        return np.mean(stack, axis=-1)

    def _blend_weighted(self, sources: List[np.ndarray]) -> np.ndarray:
        """Weighted average of sources."""
        if self.weights is None or len(self.weights) != len(sources):
            self.logger.warning("Invalid weights, using average")
            return self._blend_average(sources)

        # Normalize weights
        weights = np.array(self.weights, dtype=np.float32)
        weights = weights / weights.sum()

        # Weighted sum
        result = np.zeros_like(sources[0])
        for src, w in zip(sources, weights):
            result += src * w

        return result

    def _blend_multiply(self, sources: List[np.ndarray]) -> np.ndarray:
        """Multiply all sources (like masking)."""
        result = sources[0].copy()
        for src in sources[1:]:
            result = result * src
        return result

    def _blend_screen(self, sources: List[np.ndarray]) -> np.ndarray:
        """Screen blend (1 - (1-a) * (1-b))."""
        result = sources[0].copy()
        for src in sources[1:]:
            result = 1.0 - (1.0 - result) * (1.0 - src)
        return np.clip(result, 0, 1)
