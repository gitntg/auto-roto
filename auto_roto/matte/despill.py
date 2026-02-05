"""
Despill
=======

Remove color spill from semi-transparent edges.

Useful for removing green/blue screen spill or color bleeding
from edge regions.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("AutoRoto.Matte.Despill")


class Despill:
    """
    Remove color spill from semi-transparent regions.

    Supports:
        - Green screen despill
        - Blue screen despill
        - Auto-detect spill color
    """

    def __init__(
        self,
        strength: float = 0.5,
        spill_color: str = "auto",
        logger: logging.Logger = None
    ):
        """
        Initialize despill.

        Args:
            strength: Despill strength (0-1)
            spill_color: Spill color ("green", "blue", "auto")
            logger: Optional logger instance
        """
        self.strength = strength
        self.spill_color = spill_color
        self.logger = logger or logging.getLogger("Despill")

    def apply(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """
        Apply despill to RGB image.

        Args:
            rgb: RGB image (H, W, 3), float 0-1
            alpha: Alpha matte (H, W), float 0-1

        Returns:
            Despilled RGB image (H, W, 3)
        """
        if self.strength <= 0:
            return rgb

        # Detect spill color if auto
        spill = self.spill_color
        if spill == "auto":
            spill = self._detect_spill_color(rgb, alpha)

        # Apply despill based on color
        if spill == "green":
            return self._despill_green(rgb, alpha)
        elif spill == "blue":
            return self._despill_blue(rgb, alpha)
        else:
            return rgb

    def _detect_spill_color(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray
    ) -> str:
        """Detect dominant spill color in edge regions."""
        # Get edge region (semi-transparent)
        edge_mask = (alpha > 0.1) & (alpha < 0.9)

        if not np.any(edge_mask):
            return "green"  # Default

        # Sample colors in edge region
        edge_pixels = rgb[edge_mask]

        # Compare green vs blue dominance
        avg_color = np.mean(edge_pixels, axis=0)
        r, g, b = avg_color

        # Green spill: G > R and G > B
        # Blue spill: B > R and B > G
        green_excess = g - max(r, b)
        blue_excess = b - max(r, g)

        if green_excess > blue_excess:
            return "green"
        elif blue_excess > green_excess:
            return "blue"
        else:
            return "green"  # Default

    def _despill_green(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """Remove green spill."""
        result = rgb.copy()

        r = result[:, :, 0]
        g = result[:, :, 1]
        b = result[:, :, 2]

        # Calculate green excess over average of R and B
        avg_rb = (r + b) / 2.0
        green_excess = np.maximum(0, g - avg_rb)

        # Scale by strength and inverse alpha (more despill at edges)
        edge_factor = 1.0 - alpha
        despill_amount = green_excess * self.strength * edge_factor

        # Reduce green channel
        result[:, :, 1] = g - despill_amount

        # Optionally boost red slightly to compensate
        result[:, :, 0] = r + despill_amount * 0.1

        return np.clip(result, 0, 1)

    def _despill_blue(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """Remove blue spill."""
        result = rgb.copy()

        r = result[:, :, 0]
        g = result[:, :, 1]
        b = result[:, :, 2]

        # Calculate blue excess over average of R and G
        avg_rg = (r + g) / 2.0
        blue_excess = np.maximum(0, b - avg_rg)

        # Scale by strength and inverse alpha
        edge_factor = 1.0 - alpha
        despill_amount = blue_excess * self.strength * edge_factor

        # Reduce blue channel
        result[:, :, 2] = b - despill_amount

        # Optionally boost red slightly
        result[:, :, 0] = r + despill_amount * 0.1

        return np.clip(result, 0, 1)
