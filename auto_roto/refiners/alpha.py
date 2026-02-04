"""
Alpha Refiner
=============

Refine binary/rough masks into smooth, production-quality alpha mattes.

Techniques:
    - Guided filter for edge-aware smoothing
    - Morphological operations for cleanup
    - Edge softening for natural falloff
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("AutoRoto.Refiners.Alpha")


class AlphaRefiner:
    """
    Refine binary/rough masks into smooth, production-quality alpha mattes.

    Techniques:
        - Guided filter for edge-aware smoothing
        - Morphological operations for cleanup
        - Edge softening for natural falloff
    """

    def __init__(
        self,
        iterations: int = 3,
        edge_softness: float = 1.0,
        logger: logging.Logger = None
    ):
        """
        Initialize alpha refiner.

        Args:
            iterations: Number of refinement iterations
            edge_softness: Edge softening amount (0 = no softening)
            logger: Optional logger instance
        """
        self.iterations = iterations
        self.edge_softness = edge_softness
        self.logger = logger or logging.getLogger("AlphaRefiner")

    def refine(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        trimap: np.ndarray = None
    ) -> np.ndarray:
        """
        Refine a binary mask into a smooth alpha matte.

        Args:
            mask: Binary or rough mask (0-1)
            image: RGB image for guided filtering
            trimap: Optional trimap (0=BG, 0.5=unknown, 1=FG)

        Returns:
            Refined alpha matte (0-1, float32)
        """
        import cv2

        mask = mask.astype(np.float32)
        if mask.max() > 1:
            mask = mask / 255.0

        # Step 1: Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Remove small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Remove small specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Step 2: Generate trimap if not provided
        if trimap is None:
            trimap = self._generate_trimap(mask)

        # Step 3: Apply guided filter for edge refinement
        alpha = self._guided_filter_matting(image, mask, trimap)

        # Step 4: Edge softening
        if self.edge_softness > 0:
            alpha = self._soften_edges(alpha, self.edge_softness)

        # Step 5: Final cleanup
        alpha = np.clip(alpha, 0, 1).astype(np.float32)

        return alpha

    def _generate_trimap(
        self,
        mask: np.ndarray,
        erosion: int = 10,
        dilation: int = 20
    ) -> np.ndarray:
        """Generate trimap from binary mask."""
        import cv2

        mask_uint8 = (mask * 255).astype(np.uint8)

        # Create foreground (eroded mask)
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion))
        fg = cv2.erode(mask_uint8, kernel_e, iterations=1)

        # Create background (inverted dilated mask)
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        dilated = cv2.dilate(mask_uint8, kernel_d, iterations=1)
        bg = 255 - dilated

        # Trimap: 0 = bg, 0.5 = unknown, 1 = fg
        trimap = np.ones_like(mask, dtype=np.float32) * 0.5
        trimap[fg > 127] = 1.0
        trimap[bg > 127] = 0.0

        return trimap

    def _guided_filter_matting(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        trimap: np.ndarray,
        radius: int = 16,
        eps: float = 1e-4
    ) -> np.ndarray:
        """Apply guided filter for alpha matting."""
        import cv2

        # Convert image to float
        if image.dtype == np.uint8:
            guide = image.astype(np.float32) / 255.0
        else:
            guide = image.astype(np.float32)

        # Use grayscale guide
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)

        # Apply guided filter
        try:
            alpha = cv2.ximgproc.guidedFilter(guide, mask, radius, eps)
        except AttributeError:
            # Fallback: simple bilateral filter
            mask_uint8 = (mask * 255).astype(np.uint8)
            alpha_uint8 = cv2.bilateralFilter(mask_uint8, 9, 75, 75)
            alpha = alpha_uint8.astype(np.float32) / 255.0

        # Preserve known regions from trimap
        alpha = np.where(trimap > 0.9, 1.0, alpha)
        alpha = np.where(trimap < 0.1, 0.0, alpha)

        return alpha

    def _soften_edges(self, alpha: np.ndarray, softness: float) -> np.ndarray:
        """Apply edge softening for natural falloff."""
        import cv2

        # Find edges
        edges = cv2.Canny((alpha * 255).astype(np.uint8), 50, 150)

        # Dilate edges
        kernel_size = max(3, int(softness * 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edge_mask = cv2.dilate(edges, kernel, iterations=1)
        edge_mask = edge_mask.astype(np.float32) / 255.0

        # Apply gaussian blur to alpha at edges
        blur_size = max(3, int(softness * 5))
        if blur_size % 2 == 0:
            blur_size += 1

        alpha_blurred = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)

        # Blend
        alpha_soft = alpha * (1 - edge_mask) + alpha_blurred * edge_mask

        return alpha_soft
