"""
Depth-Guided Alpha Refiner
==========================

Refine alpha mattes using depth information.

Uses depth to DETECT hair strands outside the SAM mask by finding
foreground-depth pixels in the edge region around the mask.

Algorithm:
    1. Keep SAM alpha as the core shape (interior stays untouched)
    2. Get foreground depth reference from inside the SAM mask
    3. Create search region: dilated mask minus original mask
    4. In search region, find pixels with foreground-like depth
    5. Those pixels are hair strands - add them with soft blending
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("AutoRoto.Refiners.Depth")

# Constants
DEEP_CORE_MIN_DISTANCE = 30
DEEP_CORE_MIN_PIXELS = 100


@dataclass
class DepthStatistics:
    """Statistics computed from depth analysis."""
    fg_depth_median: float
    fg_depth_std: float
    fg_depth_min: float
    fg_depth_max: float
    bg_depth_median: float
    bg_depth_min: float
    bg_depth_max: float
    foreground_threshold: float
    hair_foreground_threshold: float
    dist_inside: np.ndarray


@dataclass
class HairProcessingContext:
    """Context for hair matte computation."""
    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray
    dist_inside: np.ndarray
    dist_outside: np.ndarray
    outside_mask: np.ndarray
    is_foreground_depth: np.ndarray
    depth_stats: DepthStatistics


class DepthGuidedRefiner:
    """
    Refine alpha mattes using depth information.

    Uses depth to detect hair strands outside the SAM mask.
    """

    def __init__(
        self,
        edge_threshold: float = 0.1,
        blend_strength: float = 0.5,
        preserve_softness: bool = True,
        sharpen_hard_edges: bool = True,
        soften_gradual_edges: bool = True,
        hair_search_radius: int = 50,
        depth_tolerance: float = 0.15,
        min_hair_alpha: float = 0.3,
        logger: logging.Logger = None
    ):
        """
        Initialize depth-guided refiner.

        Args:
            edge_threshold: Depth gradient threshold for edges
            blend_strength: Refinement blend strength
            preserve_softness: Preserve existing soft edges
            sharpen_hard_edges: Sharpen hard depth edges
            soften_gradual_edges: Soften gradual edges
            hair_search_radius: Search radius for hair detection
            depth_tolerance: Depth similarity tolerance
            min_hair_alpha: Minimum alpha for detected hair
            logger: Optional logger instance
        """
        self.edge_threshold = edge_threshold
        self.blend_strength = blend_strength
        self.preserve_softness = preserve_softness
        self.sharpen_hard_edges = sharpen_hard_edges
        self.soften_gradual_edges = soften_gradual_edges
        self.hair_search_radius = hair_search_radius
        self.depth_tolerance = depth_tolerance
        self.min_hair_alpha = min_hair_alpha
        self.logger = logger or logging.getLogger("DepthRefiner")

    def detect_hair_region(self, alpha: np.ndarray, extended: bool = False) -> np.ndarray:
        """Detect the hair region (top portion of mask)."""
        import cv2

        mask_binary = (alpha > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros_like(alpha)

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        hair_region = np.zeros_like(alpha)
        hair_percent = 0.6 if extended else 0.4
        hair_height = int(h * hair_percent)

        y_start = max(0, y - int(h * 0.1))
        hair_region[y_start:y+hair_height, x:x+w] = 1.0

        dilate_size = self.hair_search_radius * (3 if extended else 2)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        hair_region = cv2.dilate(hair_region.astype(np.uint8), dilate_kernel).astype(np.float32)

        return hair_region

    def detect_hair_texture(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Detect hair-like texture using high-frequency analysis."""
        import cv2

        if rgb.max() > 1:
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        texture_energy = np.abs(laplacian)

        mask_binary = alpha > 0.5
        if np.sum(mask_binary) > 100:
            fg_texture_mean = np.mean(texture_energy[mask_binary])
            fg_texture_std = np.std(texture_energy[mask_binary])
        else:
            fg_texture_mean = 0.05
            fg_texture_std = 0.02

        texture_threshold = max(fg_texture_mean - fg_texture_std, 0.01)
        hair_texture = (texture_energy > texture_threshold).astype(np.float32)
        hair_confidence = np.clip(texture_energy / (fg_texture_mean + 0.01), 0, 1)

        return hair_confidence * hair_texture

    def detect_hair_color(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        depth: np.ndarray,
        fg_depth_min: float,
        fg_depth_max: float,
        foreground_threshold: float
    ) -> np.ndarray:
        """Detect hair by color similarity to foreground."""
        import cv2

        if rgb.max() > 1:
            rgb_float = rgb.astype(np.float32) / 255.0
        else:
            rgb_float = rgb.astype(np.float32)

        hair_sample_region = self.detect_hair_region(alpha)
        hair_sample_region = hair_sample_region * (alpha > 0.5).astype(np.float32)

        if np.sum(hair_sample_region > 0.5) < 100:
            return np.zeros_like(alpha)

        hair_pixels = rgb_float[hair_sample_region > 0.5]
        hair_color_mean = np.mean(hair_pixels, axis=0)
        hair_color_std = np.std(hair_pixels, axis=0) + 0.05

        color_diff = np.sqrt(np.sum((rgb_float - hair_color_mean) ** 2, axis=2))
        color_tolerance = np.sqrt(np.sum(hair_color_std ** 2))
        color_similarity = np.exp(-(color_diff ** 2) / (2 * (color_tolerance ** 2)))

        depth_ok = depth < foreground_threshold
        hair_depth_boost = np.where(depth < fg_depth_min + 0.05, 1.5, 1.0)
        hair_color_confidence = color_similarity * depth_ok.astype(np.float32) * hair_depth_boost

        return np.clip(hair_color_confidence, 0, 1).astype(np.float32)

    def _create_core_matte(self, alpha: np.ndarray, core_erode_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Create the core matte via erosion."""
        import cv2

        mask_binary = (alpha > 0.5).astype(np.uint8)
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (core_erode_size * 2 + 1, core_erode_size * 2 + 1)
        )
        core_matte = cv2.erode(mask_binary, erode_kernel, iterations=1).astype(np.float32)

        return core_matte, mask_binary

    def _analyze_depth_statistics(
        self,
        depth: np.ndarray,
        alpha: np.ndarray,
        mask_binary: np.ndarray
    ) -> DepthStatistics:
        """Analyze depth statistics for foreground and background regions."""
        import cv2

        dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        deep_core = dist_inside > DEEP_CORE_MIN_DISTANCE

        if np.sum(deep_core) > DEEP_CORE_MIN_PIXELS:
            fg_depth_median = float(np.median(depth[deep_core]))
            fg_depth_std = float(np.std(depth[deep_core]))
            fg_depth_min = float(np.percentile(depth[deep_core], 5))
            fg_depth_max = float(np.percentile(depth[deep_core], 95))
        else:
            fg_depth_median = float(np.median(depth[alpha > 0.5]))
            fg_depth_std = 0.1
            fg_depth_min = fg_depth_median - 0.15
            fg_depth_max = fg_depth_median + 0.15

        bg_region = dist_inside == 0
        if np.sum(bg_region) > DEEP_CORE_MIN_PIXELS:
            bg_depth_median = float(np.median(depth[bg_region]))
            bg_depth_min = float(np.percentile(depth[bg_region], 5))
            bg_depth_max = float(np.percentile(depth[bg_region], 95))
        else:
            bg_depth_median = 0.7
            bg_depth_min = 0.6
            bg_depth_max = 0.8

        foreground_threshold = fg_depth_max + self.depth_tolerance
        hair_foreground_threshold = fg_depth_max + self.depth_tolerance * 1.5

        return DepthStatistics(
            fg_depth_median=fg_depth_median,
            fg_depth_std=fg_depth_std,
            fg_depth_min=fg_depth_min,
            fg_depth_max=fg_depth_max,
            bg_depth_median=bg_depth_median,
            bg_depth_min=bg_depth_min,
            bg_depth_max=bg_depth_max,
            foreground_threshold=foreground_threshold,
            hair_foreground_threshold=hair_foreground_threshold,
            dist_inside=dist_inside,
        )

    def _compute_edge_matte(
        self,
        alpha: np.ndarray,
        depth: np.ndarray,
        mask_binary: np.ndarray,
        depth_stats: DepthStatistics
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute depth-gated edge matte."""
        import cv2

        dist_inside = depth_stats.dist_inside
        foreground_threshold = depth_stats.foreground_threshold
        fg_depth_max = depth_stats.fg_depth_max
        fg_depth_min = depth_stats.fg_depth_min

        dist_outside = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)

        inner_edge_dist = 5
        outer_edge_dist = float(self.hair_search_radius)
        spatial_edge_region = (dist_inside <= inner_edge_dist) | (dist_outside <= outer_edge_dist)

        is_foreground_depth = depth < foreground_threshold
        depth_margin = foreground_threshold - depth
        depth_confidence = np.clip(depth_margin / (foreground_threshold - fg_depth_min + 0.01), 0, 1)

        in_fg_range = (depth >= fg_depth_min - 0.05) & (depth <= fg_depth_max + 0.1)
        depth_confidence = np.where(in_fg_range, np.maximum(depth_confidence, 0.8), depth_confidence)

        edge_matte = np.zeros_like(alpha)
        valid_edge = spatial_edge_region & is_foreground_depth
        edge_matte[valid_edge] = depth_confidence[valid_edge]

        outside_mask = dist_outside > 0
        if np.any(outside_mask & valid_edge):
            distance_falloff = 1.0 - (dist_outside / outer_edge_dist)
            distance_falloff = np.clip(distance_falloff, 0, 1)
            edge_matte[outside_mask] *= distance_falloff[outside_mask]

        edge_matte = edge_matte * self.blend_strength * 0.7

        return edge_matte, dist_outside, is_foreground_depth

    def _compute_hair_matte(self, ctx: HairProcessingContext) -> np.ndarray:
        """Compute hair-specific matte using texture, color, and depth analysis."""
        import cv2

        rgb = ctx.rgb
        alpha = ctx.alpha
        depth = ctx.depth
        dist_inside = ctx.dist_inside
        dist_outside = ctx.dist_outside
        outside_mask = ctx.outside_mask
        is_foreground_depth = ctx.is_foreground_depth
        depth_stats = ctx.depth_stats

        fg_depth_min = depth_stats.fg_depth_min
        fg_depth_max = depth_stats.fg_depth_max
        fg_depth_median = depth_stats.fg_depth_median
        foreground_threshold = depth_stats.foreground_threshold
        hair_depth_threshold = depth_stats.hair_foreground_threshold

        hair_matte = np.zeros_like(alpha)

        hair_region = self.detect_hair_region(alpha, extended=True)
        hair_region_normal = self.detect_hair_region(alpha, extended=False)
        hair_region_strict = self.detect_hair_region(alpha, extended=False)

        hair_texture = self.detect_hair_texture(rgb, alpha)
        hair_color = self.detect_hair_color(
            rgb, alpha, depth,
            fg_depth_min, fg_depth_max, foreground_threshold
        )

        # METHOD 1: Texture + Color evidence
        hair_evidence_tc = hair_texture * hair_color * is_foreground_depth.astype(np.float32)
        hair_evidence_tc = hair_evidence_tc * hair_region_normal * outside_mask.astype(np.float32)

        # METHOD 2: Direct depth-based hair
        depth_range = hair_depth_threshold - fg_depth_min
        depth_hair_evidence = np.where(
            depth < hair_depth_threshold,
            np.clip((hair_depth_threshold - depth) / (depth_range + 0.01), 0, 1),
            0
        )
        depth_hair_evidence = depth_hair_evidence * hair_region * outside_mask.astype(np.float32)

        # METHOD 3: Edge-focused hair refinement
        max_hair_distance = self.hair_search_radius
        max_edge_distance = 15
        hair_edge_band = (dist_outside > 0) & (dist_outside < max_hair_distance)

        edge_inside = (dist_inside > 0) & (dist_inside < 5)
        if np.sum(edge_inside & (hair_region_strict > 0.5)) > 100:
            edge_depth = depth[edge_inside & (hair_region_strict > 0.5)]
            edge_depth_mean = np.mean(edge_depth)
            edge_depth_std = np.std(edge_depth)
        else:
            edge_depth_mean = fg_depth_median
            edge_depth_std = 0.15

        depth_similar_to_edge = (depth > edge_depth_mean - edge_depth_std * 3) & \
                                (depth < edge_depth_mean + edge_depth_std * 4)

        # Apply rope filter
        fg_outside = (depth < hair_depth_threshold) & outside_mask
        fg_outside_uint8 = fg_outside.astype(np.uint8) * 255
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_opened = cv2.morphologyEx(fg_outside_uint8, cv2.MORPH_OPEN, open_kernel)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_cleaned = cv2.morphologyEx(fg_opened, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_cleaned = cv2.dilate(fg_cleaned, dilate_kernel, iterations=1)
        rope_filter = (fg_cleaned > 0)

        valid_hair = hair_edge_band & (depth < hair_depth_threshold) & depth_similar_to_edge & rope_filter
        valid_hair_region = valid_hair & (hair_region_strict > 0.5)

        edge_band_tiny = (dist_outside > 0) & (dist_outside < max_edge_distance)
        valid_edge = edge_band_tiny & (depth < foreground_threshold)
        valid_pixels = valid_hair_region | (valid_edge & (hair_region_strict < 0.5))

        if np.sum(valid_hair_region) > 100:
            hair_depths = depth[valid_hair_region]
            hair_local_min = np.percentile(hair_depths, 2)
            hair_local_max = np.percentile(hair_depths, 99)
        else:
            hair_local_min = hair_depth_threshold
            hair_local_max = fg_depth_max

        # Convert depth to alpha
        depth_range_local = max(hair_local_max - hair_local_min, 0.1)
        depth_normalized = np.clip((hair_local_max - depth) / depth_range_local, 0, 1)
        depth_contrasted = depth_normalized ** 0.8
        depth_as_alpha = depth_contrasted * valid_pixels.astype(np.float32)

        # Texture preservation
        blur_size = 15
        depth_blurred = cv2.GaussianBlur(depth_normalized, (blur_size, blur_size), 0)
        depth_detail = depth_normalized - depth_blurred
        detail_boost = 2.0
        depth_as_alpha = depth_as_alpha + (depth_detail * detail_boost * valid_pixels.astype(np.float32))
        depth_as_alpha = np.clip(depth_as_alpha, 0, 1)

        # Apply distance falloff
        hair_falloff = np.clip(1.0 - (dist_outside / max_hair_distance) ** 0.6, 0, 1)
        edge_falloff = np.clip(1.0 - (dist_outside / max_edge_distance), 0, 1)
        falloff = np.where(hair_region_strict > 0.5, hair_falloff, edge_falloff)
        depth_as_alpha = depth_as_alpha * falloff

        # Combine all methods
        hair_evidence = depth_as_alpha.copy()
        hair_evidence = np.maximum(hair_evidence, hair_evidence_tc * 0.7)
        hair_evidence = np.maximum(hair_evidence, depth_hair_evidence * 0.5)

        hair_matte = hair_evidence * self.blend_strength
        hair_matte = np.where(hair_matte > 0.05, hair_matte, 0)

        return hair_matte

    def _combine_mattes(
        self,
        core_matte: np.ndarray,
        edge_matte: np.ndarray,
        hair_matte: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """Combine all mattes using MAX operation."""
        import cv2

        combined = core_matte.copy()
        combined = np.maximum(combined, edge_matte)
        combined = np.maximum(combined, hair_matte)
        combined = np.maximum(combined, alpha * 0.9)

        combined = np.where(core_matte > 0.5, 1.0, combined)
        combined = cv2.GaussianBlur(combined, (3, 3), 0.5)
        combined = np.where(core_matte > 0.5, 1.0, combined)

        return np.clip(combined, 0, 1).astype(np.float32)

    def refine(
        self,
        alpha: np.ndarray,
        depth: np.ndarray,
        rgb: np.ndarray = None,
        normalize_depth: bool = False
    ) -> np.ndarray:
        """
        Refine alpha matte using depth information.

        Args:
            alpha: Input alpha matte (H, W), 0-1
            depth: Input depth map (H, W), 0-1 (closer = LOWER)
            rgb: Optional RGB image for color-guided refinement
            normalize_depth: If True, expand foreground depth detail

        Returns:
            Refined alpha matte (H, W), 0-1
        """
        import cv2

        core_matte, mask_binary = self._create_core_matte(alpha)
        depth = depth.astype(np.float32)

        depth_stats = self._analyze_depth_statistics(depth, alpha, mask_binary)

        edge_matte, dist_outside, is_foreground_depth = self._compute_edge_matte(
            alpha, depth, mask_binary, depth_stats
        )
        outside_mask = dist_outside > 0

        hair_matte = np.zeros_like(alpha)
        if rgb is not None:
            ctx = HairProcessingContext(
                rgb=rgb,
                alpha=alpha,
                depth=depth,
                dist_inside=depth_stats.dist_inside,
                dist_outside=dist_outside,
                outside_mask=outside_mask,
                is_foreground_depth=is_foreground_depth,
                depth_stats=depth_stats
            )
            hair_matte = self._compute_hair_matte(ctx)

        return self._combine_mattes(core_matte, edge_matte, hair_matte, alpha)

    def refine_with_rgb(
        self,
        alpha: np.ndarray,
        depth: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """Refine alpha using both depth and RGB information."""
        import cv2

        refined = self.refine(alpha, depth, rgb)

        if rgb is not None:
            if len(rgb.shape) == 3:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = rgb

            gray = gray.astype(np.float32) / 255.0 if gray.max() > 1 else gray.astype(np.float32)

            try:
                refined = cv2.ximgproc.guidedFilter(
                    guide=gray,
                    src=refined,
                    radius=4,
                    eps=1e-4
                )
            except AttributeError:
                pass

        return refined
