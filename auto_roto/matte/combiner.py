"""
Matte Combiner
==============

Combine alpha mattes with RGB frames to produce final output.

Features:
    - Core erosion for cleaner edges
    - Premultiplied alpha compositing
    - Preview generation with checkerboard background
"""

import logging
from typing import Dict, Optional

import numpy as np

from auto_roto.config.matte import MatteCombineConfig

logger = logging.getLogger("AutoRoto.Matte.Combiner")


class MatteCombiner:
    """
    Combine alpha mattes with RGB frames.

    Produces:
        - Final alpha matte (with core erosion)
        - RGBA composite with premultiplied alpha
        - Preview composite on checkerboard
    """

    def __init__(
        self,
        config: MatteCombineConfig = None,
        logger: logging.Logger = None
    ):
        """
        Initialize matte combiner.

        Args:
            config: MatteCombineConfig instance (uses defaults if None)
            logger: Optional logger instance
        """
        self.config = config or MatteCombineConfig()
        self.logger = logger or logging.getLogger("MatteCombiner")

    def combine(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        apply_despill: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Combine RGB and alpha into final output.

        Args:
            rgb: RGB image (H, W, 3), uint8 or float
            alpha: Alpha matte (H, W), float 0-1
            apply_despill: Apply despill to semi-transparent regions

        Returns:
            Dictionary with:
                - "alpha": Final alpha matte (H, W)
                - "rgba": RGBA composite (H, W, 4)
                - "preview": Preview on checkerboard (H, W, 3)
        """
        import cv2

        # Ensure float32
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        else:
            rgb = rgb.astype(np.float32)

        alpha = alpha.astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0

        h, w = alpha.shape

        # Apply core erosion
        if self.config.core_erosion > 0:
            alpha = self._apply_core_erosion(alpha)

        # Apply despill
        if apply_despill and self.config.despill_strength > 0:
            from auto_roto.matte.despill import Despill
            despill = Despill(strength=self.config.despill_strength)
            rgb = despill.apply(rgb, alpha)

        # Create RGBA with premultiplied alpha
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = rgb * alpha[:, :, np.newaxis]
        rgba[:, :, 3] = alpha

        # Create preview
        preview = self._create_preview(rgb, alpha)

        return {
            "alpha": alpha,
            "rgba": rgba,
            "preview": preview
        }

    def _apply_core_erosion(self, alpha: np.ndarray) -> np.ndarray:
        """Apply core erosion to clean up edges."""
        import cv2

        if self.config.core_erosion <= 0:
            return alpha

        # Create binary core mask
        core_mask = (alpha > 0.99).astype(np.uint8)

        # Erode core
        kernel_size = self.config.core_erosion * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        eroded_core = cv2.erode(core_mask, kernel, iterations=1)

        # Blend: preserve eroded core, soften edges
        result = alpha.copy()

        # The eroded core stays solid
        result[eroded_core > 0] = 1.0

        return result

    def _create_preview(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        checker_size: int = 16
    ) -> np.ndarray:
        """Create preview composite on checkerboard."""
        h, w = alpha.shape

        # Create checkerboard
        checker = self._create_checkerboard(h, w, checker_size)

        # Composite RGB over checkerboard using alpha
        alpha_3ch = alpha[:, :, np.newaxis]
        preview = rgb * alpha_3ch + checker * (1 - alpha_3ch)

        # Convert to uint8 for preview
        preview = (np.clip(preview, 0, 1) * 255).astype(np.uint8)

        return preview

    def _create_checkerboard(
        self,
        height: int,
        width: int,
        size: int = 16
    ) -> np.ndarray:
        """Create a checkerboard pattern."""
        # Create grid
        y_idx = np.arange(height) // size
        x_idx = np.arange(width) // size

        checker = (y_idx[:, np.newaxis] + x_idx[np.newaxis, :]) % 2

        # Colors: light gray and mid gray
        light = 0.7
        dark = 0.5

        checker_rgb = np.where(
            checker[:, :, np.newaxis] == 0,
            light,
            dark
        ).astype(np.float32)

        return np.broadcast_to(checker_rgb, (height, width, 3)).copy()
