"""
Trimap Synthesizer
==================

Builds trimap from SAM mask + Depth map for ViTMatte.

Trimap values:
    0   = Definite Background (black)
    128 = Unknown (gray) - where hair/fine detail lives
    255 = Definite Foreground (white)

The unknown zone is found using depth high-pass filtering:
    - High-pass reveals "depth spikes" (sudden depth changes)
    - Hair strands show as spikes extending from the head
    - We mark these as unknown for ViTMatte to solve
"""

import logging
from typing import Optional, Tuple

import numpy as np

from auto_roto.config.vitmatte import TrimapConfig

logger = logging.getLogger("AutoRoto.Refiners.Trimap")


class TrimapSynthesizer:
    """
    Builds trimap from SAM mask + Depth map.

    Supports multiple modes:
        - Adaptive mode (default): Variable unknown width based on local complexity
        - Simple edge mode: Fixed dilation around mask edge
        - Scientific mode: SOURCEOFTRUTH golden rule implementation
    """

    def __init__(self, config: TrimapConfig = None, logger: logging.Logger = None):
        """
        Initialize trimap synthesizer.

        Args:
            config: TrimapConfig instance (uses defaults if None)
            logger: Optional logger instance
        """
        self.config = config or TrimapConfig()
        self.logger = logger or logging.getLogger("TrimapSynth")

    def compute_depth_highpass(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to depth map to find "depth spikes".

        Hair strands create sudden depth discontinuities that show up
        as high values in the Laplacian (high-pass) of the depth map.
        """
        import cv2

        laplacian = cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F, ksize=5)
        highpass = np.abs(laplacian)

        grad_x = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        combined = highpass + gradient_mag * 0.5

        if combined.max() > 0:
            combined = combined / combined.max()

        return combined.astype(np.float32)

    def detect_hair_region(self, mask: np.ndarray) -> np.ndarray:
        """Detect the hair region (top portion of the subject)."""
        import cv2

        mask_binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros_like(mask, dtype=np.float32)

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        hair_height = int(h * self.config.hair_region_ratio)

        hair_region = np.zeros_like(mask, dtype=np.float32)
        y_start = max(0, y - int(h * 0.15))
        hair_region[y_start:y + hair_height, x:x + w] = 1.0

        dilate_size = self.config.hair_dilation_boost * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        hair_region = cv2.dilate(hair_region.astype(np.uint8), kernel).astype(np.float32)

        return hair_region

    def _compute_complexity_map(self, depth: np.ndarray) -> np.ndarray:
        """Compute depth gradient magnitude as complexity factor."""
        import cv2

        gX = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        gY = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gX, gY)

        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

        kernel_size = self.config.adaptive_blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        complexity_map = cv2.GaussianBlur(magnitude, (kernel_size, kernel_size), 0)

        return complexity_map

    def _compute_rgb_complexity(self, rgb: np.ndarray) -> np.ndarray:
        """Compute RGB gradient magnitude as texture complexity factor."""
        import cv2

        if rgb.ndim == 3:
            gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb.astype(np.float32)

        if gray.max() > 1.0:
            gray = gray / 255.0

        gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gX, gY)

        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

        kernel_size = self.config.adaptive_blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        complexity_map = cv2.GaussianBlur(magnitude, (kernel_size, kernel_size), 0)

        return complexity_map

    def _adaptive_trimap(
        self,
        mask_binary: np.ndarray,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        prev_frame_gray: Optional[np.ndarray] = None,
        curr_frame_gray: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Adaptive Trimap using Formulaic Distance Approach.

        Formula:
            dynamic_threshold = base_px + (max_px * complexity_map) + (motion_weight * motion_factor)
        """
        import cv2

        h, w = mask_binary.shape

        # Compute complexity
        depth_complexity = self._compute_complexity_map(depth)
        if rgb is not None and self.config.rgb_complexity_weight > 0:
            rgb_complexity = self._compute_rgb_complexity(rgb)
            if rgb_complexity.shape != depth_complexity.shape:
                rgb_complexity = cv2.resize(rgb_complexity, (w, h))
            complexity_map = np.maximum(
                depth_complexity,
                rgb_complexity * self.config.rgb_complexity_weight
            )
        else:
            complexity_map = depth_complexity

        # Optional motion factor
        motion_factor = np.zeros((h, w), dtype=np.float32)

        # Distance transforms
        mask_255 = (mask_binary * 255).astype(np.uint8)
        dist_inside = cv2.distanceTransform(mask_255, cv2.DIST_L2, 5)
        inv_mask = cv2.bitwise_not(mask_255)
        dist_outside = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

        # Adaptive thresholds
        erosion_threshold = (
            self.config.erosion_base_px +
            (self.config.erosion_max_px * complexity_map)
        )
        dilation_threshold = (
            self.config.adaptive_base_px +
            (self.config.adaptive_max_px * complexity_map) +
            (self.config.motion_weight * motion_factor)
        )

        # Structural core
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_dist = cv2.dilate(dist_inside, kernel)
        min_width = 10.0
        ridges = (dist_inside == dilated_dist) & (dist_inside > min_width)
        skeleton = ridges.astype(np.uint8) * 255
        bone_thickness = 15
        protection_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (bone_thickness * 2 + 1, bone_thickness * 2 + 1)
        )
        structural_core = cv2.dilate(skeleton, protection_kernel)

        # Assemble trimap
        adaptive_core = (dist_inside > erosion_threshold).astype(np.uint8) * 255
        core_mask = cv2.bitwise_or(adaptive_core, structural_core)
        core_mask = cv2.bitwise_and(core_mask, mask_255)

        is_potentially_foreground = (mask_binary > 0) | (dist_outside < dilation_threshold)

        trimap = np.zeros((h, w), dtype=np.uint8)
        trimap[is_potentially_foreground] = 128
        trimap[core_mask == 255] = 255

        return trimap

    def _simple_edge_trimap(self, mask_binary: np.ndarray) -> np.ndarray:
        """Create a simple trimap by dilating the mask edge."""
        import cv2

        h, w = mask_binary.shape

        erosion_size = self.config.core_erosion * 2 + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        core_fg = cv2.erode(mask_binary, erode_kernel, iterations=1)

        dilation_size = self.config.simple_edge_width * 2 + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        dilated = cv2.dilate(mask_binary, dilate_kernel, iterations=1)

        unknown_zone = dilated.astype(np.float32) - core_fg.astype(np.float32)
        unknown_zone = np.clip(unknown_zone, 0, 1)

        trimap = np.zeros((h, w), dtype=np.uint8)
        trimap[unknown_zone > 0] = 128
        trimap[core_fg > 0] = 255

        return trimap

    def synthesize(
        self,
        sam_mask: np.ndarray,
        depth: np.ndarray,
        rgb: np.ndarray = None,
        prev_frame_gray: Optional[np.ndarray] = None,
        curr_frame_gray: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesize trimap from SAM mask and depth map.

        Args:
            sam_mask: SAM alpha mask (H, W), float 0-1
            depth: Depth map (H, W), float 0-1 (higher = closer)
            rgb: Optional RGB image for additional analysis
            prev_frame_gray: Previous frame grayscale (for motion-aware mode)
            curr_frame_gray: Current frame grayscale (for motion-aware mode)

        Returns:
            Trimap (H, W), uint8 with values 0, 128, 255
        """
        sam_mask = sam_mask.astype(np.float32)
        depth = depth.astype(np.float32)

        mask_binary = (sam_mask > 0.5).astype(np.uint8)

        if self.config.simple_edge_mode:
            return self._simple_edge_trimap(mask_binary)

        if self.config.adaptive_mode:
            return self._adaptive_trimap(mask_binary, depth, rgb, prev_frame_gray, curr_frame_gray)

        # Legacy mode fallback
        return self._simple_edge_trimap(mask_binary)
