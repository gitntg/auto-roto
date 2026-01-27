#!/usr/bin/env python
"""
PROFESSIONAL EDGE REFINEMENT MODULE
====================================

Advanced alpha edge processing using VFX compositing techniques.

This module provides production-quality edge operations that go beyond
simple morphological operations:

1. SUBPIXEL EDGE DETECTION
   - Laplacian-of-Gaussian for precise edge localization
   - Subpixel interpolation using local gradients
   - Anti-aliased edge masks

2. COLOR DIFFERENCE KEYING
   - Edge-band color sampling (like Primatte/Keylight)
   - Complementary color suppression for despill
   - Luminance-weighted edge detection

3. MORPHOLOGICAL GRADIENT REFINEMENT
   - Precise edge band extraction
   - Soft falloff generation
   - Core/edge separation

4. PREMULTIPLIED ALPHA OPERATIONS
   - Proper compositing math
   - Gamma-correct blending
   - Edge color correction

Author: AUTO-ROTO v5 Enhancement
License: MIT
"""

import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EdgeRefineConfig:
    """Configuration for edge refinement operations."""

    # Subpixel detection
    subpixel_precision: int = 4          # Upscale factor for subpixel work
    laplacian_sigma: float = 1.5         # LoG sigma for edge detection

    # Edge bands
    inner_band_width: int = 8            # Pixels inside mask edge
    outer_band_width: int = 12           # Pixels outside mask edge

    # Color keying
    color_sample_radius: int = 5         # Radius for edge color sampling
    despill_strength: float = 0.5        # 0-1, strength of color suppression

    # Softness
    edge_softness: float = 1.0           # Edge blur amount
    falloff_gamma: float = 1.0           # Gamma curve for edge falloff

    # Anti-aliasing
    antialias_strength: float = 1.0      # 0-2, AA filter strength

    # Core operations
    core_shrink: int = 3                 # Pixels to shrink core
    edge_extend: int = 2                 # Pixels to extend edge outward


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("EdgeRefine")


class SubpixelEdgeDetector:
    """
    Detect edges with subpixel precision using Laplacian-of-Gaussian.

    Standard edge detection operates at pixel resolution, causing
    aliased edges. This class upscales, detects, then interpolates
    for smooth, anti-aliased results.
    """

    def __init__(self, precision: int = 4, sigma: float = 1.5):
        self.precision = precision
        self.sigma = sigma

    def detect(self, alpha: np.ndarray) -> np.ndarray:
        """
        Detect edges with subpixel accuracy.

        Args:
            alpha: Input alpha (H, W), values 0-1

        Returns:
            Edge magnitude map (H, W), values 0-1
        """
        h, w = alpha.shape

        # Upscale for subpixel precision
        upscale_h = h * self.precision
        upscale_w = w * self.precision
        alpha_up = cv2.resize(alpha, (upscale_w, upscale_h),
                              interpolation=cv2.INTER_CUBIC)

        # Apply Laplacian of Gaussian
        # First blur with Gaussian
        sigma_up = self.sigma * self.precision
        blurred = cv2.GaussianBlur(alpha_up, (0, 0), sigma_up)

        # Then apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)

        # Take absolute value and normalize
        edges_up = np.abs(laplacian)
        edges_up = edges_up / (edges_up.max() + 1e-8)

        # Downscale with area averaging for anti-aliasing
        edges = cv2.resize(edges_up, (w, h), interpolation=cv2.INTER_AREA)

        return edges.astype(np.float32)

    def detect_zero_crossings(self, alpha: np.ndarray) -> np.ndarray:
        """
        Detect zero crossings of Laplacian for precise edge localization.

        Zero crossings mark the exact edge position, not just high gradient.
        """
        h, w = alpha.shape

        # Upscale
        upscale_h = h * self.precision
        upscale_w = w * self.precision
        alpha_up = cv2.resize(alpha, (upscale_w, upscale_h),
                              interpolation=cv2.INTER_CUBIC)

        # Gaussian blur
        sigma_up = self.sigma * self.precision
        blurred = cv2.GaussianBlur(alpha_up, (0, 0), sigma_up)

        # Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=5)

        # Find zero crossings
        # A zero crossing occurs where adjacent pixels have opposite signs
        zero_cross = np.zeros_like(laplacian, dtype=np.float32)

        # Check horizontal neighbors
        sign_diff_h = laplacian[:, :-1] * laplacian[:, 1:]
        zero_h = sign_diff_h < 0
        zero_cross[:, :-1] = np.maximum(zero_cross[:, :-1],
                                         zero_h.astype(np.float32))
        zero_cross[:, 1:] = np.maximum(zero_cross[:, 1:],
                                        zero_h.astype(np.float32))

        # Check vertical neighbors
        sign_diff_v = laplacian[:-1, :] * laplacian[1:, :]
        zero_v = sign_diff_v < 0
        zero_cross[:-1, :] = np.maximum(zero_cross[:-1, :],
                                         zero_v.astype(np.float32))
        zero_cross[1:, :] = np.maximum(zero_cross[1:, :],
                                        zero_v.astype(np.float32))

        # Downscale
        edges = cv2.resize(zero_cross, (w, h), interpolation=cv2.INTER_AREA)

        return edges.astype(np.float32)


class ColorDifferenceKeyer:
    """
    Edge refinement using color difference keying.

    Samples foreground and background colors at the edge band,
    then uses color difference to refine alpha values.
    Similar to professional keyers like Primatte and Keylight.
    """

    def __init__(self, sample_radius: int = 5, despill_strength: float = 0.5):
        self.sample_radius = sample_radius
        self.despill_strength = despill_strength

    def sample_edge_colors(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        inner_band: np.ndarray,
        outer_band: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample foreground and background colors from edge bands.

        Returns:
            (fg_color, bg_color) as (3,) arrays
        """
        # Sample foreground from inner band
        fg_mask = inner_band > 0.5
        if np.sum(fg_mask) > 0:
            fg_color = np.median(rgb[fg_mask], axis=0)
        else:
            # Fallback to high-alpha region
            fg_mask = alpha > 0.8
            if np.sum(fg_mask) > 0:
                fg_color = np.median(rgb[fg_mask], axis=0)
            else:
                fg_color = np.array([0.5, 0.5, 0.5])

        # Sample background from outer band
        bg_mask = outer_band > 0.5
        if np.sum(bg_mask) > 0:
            bg_color = np.median(rgb[bg_mask], axis=0)
        else:
            # Fallback to low-alpha region
            bg_mask = alpha < 0.2
            if np.sum(bg_mask) > 0:
                bg_color = np.median(rgb[bg_mask], axis=0)
            else:
                bg_color = np.array([0.5, 0.5, 0.5])

        return fg_color.astype(np.float32), bg_color.astype(np.float32)

    def compute_color_alpha(
        self,
        rgb: np.ndarray,
        fg_color: np.ndarray,
        bg_color: np.ndarray,
        edge_mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute alpha values based on color distance to FG/BG.

        Uses the principle: pixel closer to FG color = higher alpha
        """
        # Normalize RGB if needed
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0

        # Distance to foreground and background colors
        dist_fg = np.sqrt(np.sum((rgb - fg_color) ** 2, axis=2))
        dist_bg = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))

        # Alpha = how much closer to FG than BG
        total_dist = dist_fg + dist_bg + 1e-8
        color_alpha = dist_bg / total_dist

        # Only apply in edge region
        color_alpha = color_alpha * edge_mask

        return color_alpha.astype(np.float32)

    def despill(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        bg_color: np.ndarray,
        edge_mask: np.ndarray
    ) -> np.ndarray:
        """
        Remove background color contamination from edges.

        Uses complementary color suppression at semi-transparent edges.
        """
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0

        # Find the dominant background channel
        bg_channel = np.argmax(bg_color)

        # Compute spill amount (how much bg color is present)
        spill = rgb[:, :, bg_channel] - np.mean(
            np.delete(rgb, bg_channel, axis=2), axis=2
        )
        spill = np.clip(spill, 0, 1)

        # Apply despill only at semi-transparent edges
        semi_transparent = (alpha > 0.1) & (alpha < 0.9)
        despill_mask = semi_transparent & (edge_mask > 0.5)

        # Reduce the dominant bg channel
        despilled = rgb.copy()
        reduction = spill * self.despill_strength * despill_mask.astype(np.float32)
        despilled[:, :, bg_channel] = np.clip(
            despilled[:, :, bg_channel] - reduction, 0, 1
        )

        return despilled


class MorphologicalGradient:
    """
    Extract precise edge bands using morphological gradient.

    The morphological gradient (dilation - erosion) gives a clean
    edge band that's useful for targeted operations.
    """

    def __init__(self, inner_width: int = 8, outer_width: int = 12):
        self.inner_width = inner_width
        self.outer_width = outer_width

    def extract_bands(
        self,
        alpha: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract core, inner edge, outer edge, and combined edge bands.

        Returns:
            (core, inner_edge, outer_edge, full_edge)
        """
        # Binarize for morphology
        binary = (alpha > 0.5).astype(np.uint8)

        # Create kernels
        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.inner_width * 2 + 1, self.inner_width * 2 + 1)
        )
        outer_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.outer_width * 2 + 1, self.outer_width * 2 + 1)
        )

        # Erode for core
        eroded = cv2.erode(binary, inner_kernel, iterations=1)
        core = eroded.astype(np.float32)

        # Dilate for outer boundary
        dilated = cv2.dilate(binary, outer_kernel, iterations=1)

        # Inner edge = original - eroded
        inner_edge = (binary.astype(np.float32) - eroded.astype(np.float32))
        inner_edge = np.clip(inner_edge, 0, 1)

        # Outer edge = dilated - original
        outer_edge = (dilated.astype(np.float32) - binary.astype(np.float32))
        outer_edge = np.clip(outer_edge, 0, 1)

        # Full edge band
        full_edge = inner_edge + outer_edge
        full_edge = np.clip(full_edge, 0, 1)

        return core, inner_edge, outer_edge, full_edge

    def compute_gradient(self, alpha: np.ndarray) -> np.ndarray:
        """
        Compute morphological gradient (dilation - erosion).

        This gives a single-pixel-wide edge representation.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        dilated = cv2.dilate(alpha, kernel, iterations=1)
        eroded = cv2.erode(alpha, kernel, iterations=1)

        gradient = dilated - eroded
        return gradient.astype(np.float32)


class PremultipliedAlpha:
    """
    Proper premultiplied alpha operations for compositing.

    Premultiplied alpha stores RGB * A, which:
    - Preserves edge color information
    - Enables proper additive compositing
    - Avoids dark halos at edges
    """

    @staticmethod
    def premultiply(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Convert straight alpha to premultiplied.

        premultiplied = RGB * A
        """
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0

        alpha_3d = alpha[:, :, np.newaxis] if len(alpha.shape) == 2 else alpha
        premult = rgb * alpha_3d

        return premult.astype(np.float32)

    @staticmethod
    def unpremultiply(premult_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Convert premultiplied to straight alpha.

        straight = premultiplied / A (where A > 0)
        """
        alpha_3d = alpha[:, :, np.newaxis] if len(alpha.shape) == 2 else alpha

        # Avoid division by zero
        safe_alpha = np.maximum(alpha_3d, 1e-8)

        straight = premult_rgb / safe_alpha

        # Where alpha is zero, RGB should be zero
        straight = np.where(alpha_3d > 1e-8, straight, 0)

        return np.clip(straight, 0, 1).astype(np.float32)

    @staticmethod
    def over(
        fg_premult: np.ndarray,
        fg_alpha: np.ndarray,
        bg_premult: np.ndarray,
        bg_alpha: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proper 'over' composite operation.

        result = fg + bg * (1 - fg_alpha)
        """
        fg_a = fg_alpha[:, :, np.newaxis] if len(fg_alpha.shape) == 2 else fg_alpha
        bg_a = bg_alpha[:, :, np.newaxis] if len(bg_alpha.shape) == 2 else bg_alpha

        # RGB composite
        result_rgb = fg_premult + bg_premult * (1 - fg_a)

        # Alpha composite
        result_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)

        return result_rgb.astype(np.float32), result_alpha.astype(np.float32)

    @staticmethod
    def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Screen blend mode: 1 - (1-a)(1-b)

        Useful for light wrap and additive effects.
        """
        return 1 - (1 - a) * (1 - b)

    @staticmethod
    def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction for linear workflow."""
        return np.power(np.clip(image, 0, 1), gamma)


class SoftEdgeGenerator:
    """
    Generate soft edge falloff with proper curves.

    Creates natural-looking edge transitions using:
    - Distance-based falloff
    - Gamma curves for falloff shape
    - Gaussian smoothing for anti-aliasing
    """

    def __init__(self, softness: float = 1.0, gamma: float = 1.0):
        self.softness = softness
        self.gamma = gamma

    def generate_falloff(
        self,
        alpha: np.ndarray,
        inner_dist: int = 5,
        outer_dist: int = 10
    ) -> np.ndarray:
        """
        Generate distance-based soft falloff at edges.
        """
        # Binarize
        binary = (alpha > 0.5).astype(np.uint8)

        # Distance transforms
        dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)

        # Normalize distances
        inner_falloff = np.clip(dist_inside / inner_dist, 0, 1)
        outer_falloff = np.clip(1 - dist_outside / outer_dist, 0, 1)

        # Combine: 1 inside core, falloff at edges, 0 outside
        soft_alpha = np.where(dist_inside > inner_dist, 1.0, inner_falloff)
        soft_alpha = np.where(dist_outside > 0, outer_falloff, soft_alpha)

        # Apply gamma for falloff curve shaping
        if self.gamma != 1.0:
            # Only apply to intermediate values
            edge_mask = (soft_alpha > 0) & (soft_alpha < 1)
            soft_alpha[edge_mask] = np.power(soft_alpha[edge_mask], self.gamma)

        # Optional softening
        if self.softness > 0:
            blur_size = max(3, int(self.softness * 3))
            if blur_size % 2 == 0:
                blur_size += 1
            soft_alpha = cv2.GaussianBlur(soft_alpha, (blur_size, blur_size), 0)

        return soft_alpha.astype(np.float32)

    def antialias_edge(
        self,
        alpha: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply anti-aliasing to alpha edges.

        Uses a small Gaussian blur weighted by edge proximity.
        """
        # Detect edges
        binary = (alpha > 0.5).astype(np.uint8)
        dist_to_edge = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        dist_to_edge += cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)

        # Create edge proximity mask (closer to edge = more blur)
        edge_proximity = np.exp(-dist_to_edge / (strength * 3))

        # Apply Gaussian blur
        blur_size = max(3, int(strength * 2 + 1))
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)

        # Blend based on edge proximity
        antialiased = alpha * (1 - edge_proximity) + blurred * edge_proximity

        return antialiased.astype(np.float32)


class EdgeRefiner:
    """
    Main edge refinement class combining all techniques.
    """

    def __init__(self, config: EdgeRefineConfig = None, logger: logging.Logger = None):
        self.config = config or EdgeRefineConfig()
        self.logger = logger or logging.getLogger("EdgeRefiner")

        # Initialize components
        self.subpixel_detector = SubpixelEdgeDetector(
            precision=self.config.subpixel_precision,
            sigma=self.config.laplacian_sigma
        )
        self.color_keyer = ColorDifferenceKeyer(
            sample_radius=self.config.color_sample_radius,
            despill_strength=self.config.despill_strength
        )
        self.morph_gradient = MorphologicalGradient(
            inner_width=self.config.inner_band_width,
            outer_width=self.config.outer_band_width
        )
        self.soft_edge = SoftEdgeGenerator(
            softness=self.config.edge_softness,
            gamma=self.config.falloff_gamma
        )

    def refine(
        self,
        alpha: np.ndarray,
        rgb: np.ndarray = None,
        depth: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Full edge refinement pipeline.

        Args:
            alpha: Input alpha matte (H, W), values 0-1
            rgb: Optional RGB image for color keying
            depth: Optional depth map for depth-aware refinement

        Returns:
            Dict with 'alpha', 'core', 'edge', 'despilled_rgb' (if rgb provided)
        """
        self.logger.debug("Starting edge refinement...")

        alpha = alpha.astype(np.float32)
        if alpha.max() > 1:
            alpha = alpha / 255.0

        results = {}

        # Step 1: Extract morphological bands
        core, inner_edge, outer_edge, full_edge = self.morph_gradient.extract_bands(alpha)
        self.logger.debug(f"Extracted bands: core={np.sum(core > 0.5)}, edge={np.sum(full_edge > 0.5)} pixels")

        # Step 2: Subpixel edge detection for precision
        subpixel_edges = self.subpixel_detector.detect(alpha)

        # Step 3: Apply core shrink
        if self.config.core_shrink > 0:
            shrink_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.core_shrink * 2 + 1, self.config.core_shrink * 2 + 1)
            )
            core = cv2.erode(core, shrink_kernel, iterations=1)

        # Step 4: Color-based refinement if RGB available
        if rgb is not None:
            if rgb.max() > 1:
                rgb = rgb.astype(np.float32) / 255.0

            # Sample edge colors
            fg_color, bg_color = self.color_keyer.sample_edge_colors(
                rgb, alpha, inner_edge, outer_edge
            )
            self.logger.debug(f"FG color: {fg_color}, BG color: {bg_color}")

            # Compute color-based alpha in edge region
            color_alpha = self.color_keyer.compute_color_alpha(
                rgb, fg_color, bg_color, full_edge
            )

            # Blend color alpha with original edge alpha
            edge_alpha = alpha * (1 - full_edge) + color_alpha * full_edge

            # Despill
            despilled = self.color_keyer.despill(rgb, alpha, bg_color, full_edge)
            results['despilled_rgb'] = despilled
        else:
            edge_alpha = alpha

        # Step 5: Generate soft falloff
        soft_alpha = self.soft_edge.generate_falloff(
            edge_alpha,
            inner_dist=self.config.inner_band_width,
            outer_dist=self.config.outer_band_width
        )

        # Step 6: Anti-alias edges
        if self.config.antialias_strength > 0:
            soft_alpha = self.soft_edge.antialias_edge(
                soft_alpha, self.config.antialias_strength
            )

        # Step 7: Combine core and soft edge
        # Core is always solid, edge region uses soft alpha
        final_alpha = np.maximum(core, soft_alpha)

        # Apply edge extend if configured
        if self.config.edge_extend > 0:
            extend_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.edge_extend * 2 + 1, self.config.edge_extend * 2 + 1)
            )
            # Dilate slightly then blend
            extended = cv2.dilate(final_alpha, extend_kernel, iterations=1)
            # Only extend at edges, not core
            extend_mask = (extended > final_alpha) & (final_alpha < 0.9)
            final_alpha[extend_mask] = extended[extend_mask] * 0.7  # Soft extend

        # Store results
        results['alpha'] = np.clip(final_alpha, 0, 1).astype(np.float32)
        results['core'] = core.astype(np.float32)
        results['edge'] = full_edge.astype(np.float32)
        results['subpixel_edges'] = subpixel_edges

        self.logger.debug("Edge refinement complete")

        return results


# ==============================================================================
# PIPELINE INTEGRATION
# ==============================================================================

class EdgeRefinePipeline:
    """
    Pipeline for batch edge refinement of alpha sequences.
    """

    def __init__(
        self,
        config: EdgeRefineConfig = None,
        output_format: str = "exr",
        bit_depth: int = 16,
        verbose: bool = False
    ):
        self.config = config or EdgeRefineConfig()
        self.output_format = output_format
        self.bit_depth = bit_depth
        self.logger = setup_logging(verbose)
        self.refiner = EdgeRefiner(config, self.logger)

    def process_frame(
        self,
        alpha: np.ndarray,
        rgb: np.ndarray = None,
        depth: np.ndarray = None
    ) -> np.ndarray:
        """Process a single frame."""
        results = self.refiner.refine(alpha, rgb, depth)
        return results['alpha']

    def process_sequence(
        self,
        alpha_dir: Path,
        output_dir: Path,
        frames_dir: Path = None
    ):
        """
        Process an entire sequence of alpha mattes.

        Args:
            alpha_dir: Directory containing alpha mattes
            output_dir: Output directory for refined mattes
            frames_dir: Optional directory with RGB frames for color keying
        """
        import cv2

        output_dir.mkdir(parents=True, exist_ok=True)
        refined_dir = output_dir / "alpha"
        refined_dir.mkdir(exist_ok=True)

        # Find alpha files
        alpha_files = sorted(list(alpha_dir.glob("*.exr")) +
                           list(alpha_dir.glob("*.png")) +
                           list(alpha_dir.glob("*.tif")))

        # Find frame files if available
        frame_files = []
        if frames_dir and frames_dir.exists():
            frame_files = sorted(list(frames_dir.glob("*.png")) +
                               list(frames_dir.glob("*.jpg")) +
                               list(frames_dir.glob("*.exr")))

        self.logger.info(f"Processing {len(alpha_files)} frames...")

        for idx, alpha_path in enumerate(alpha_files):
            # Load alpha
            alpha = self._load_alpha(alpha_path)

            # Load RGB if available
            rgb = None
            if idx < len(frame_files):
                rgb = cv2.imread(str(frame_files[idx]), cv2.IMREAD_UNCHANGED)
                if rgb is not None:
                    if len(rgb.shape) == 3:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    if rgb.dtype == np.uint16:
                        rgb = (rgb / 256).astype(np.uint8)

            # Process
            refined = self.process_frame(alpha, rgb)

            # Save
            output_path = refined_dir / f"refined.{idx:04d}.{self.output_format}"
            self._save_alpha(refined, output_path)

            if idx % 20 == 0:
                self.logger.info(f"  Processed frame {idx}/{len(alpha_files)}")

        self.logger.info("Edge refinement complete!")

    def _load_alpha(self, path: Path) -> np.ndarray:
        """Load alpha from file."""
        import cv2

        if path.suffix.lower() == '.exr':
            try:
                import OpenEXR
                import Imath

                exr = OpenEXR.InputFile(str(path))
                header = exr.header()
                dw = header['dataWindow']
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1

                channels = list(header['channels'].keys())
                channel = 'A' if 'A' in channels else channels[0]

                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                data = exr.channel(channel, pt)
                alpha = np.frombuffer(data, dtype=np.float32).reshape((h, w))
                return alpha
            except ImportError:
                pass

        # Fallback to OpenCV
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read: {path}")

        if len(img.shape) == 3:
            img = img[:, :, -1]  # Last channel (alpha or grayscale)

        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        else:
            return img.astype(np.float32)

    def _save_alpha(self, alpha: np.ndarray, path: Path):
        """Save alpha to file."""
        import cv2

        if path.suffix.lower() == '.exr':
            try:
                import OpenEXR
                import Imath

                h, w = alpha.shape
                header = OpenEXR.Header(w, h)
                header['channels'] = {'A': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

                exr = OpenEXR.OutputFile(str(path), header)
                exr.writePixels({'A': alpha.astype(np.float32).tobytes()})
                exr.close()
                return
            except ImportError:
                pass

            if cv2.imwrite(str(path), alpha.astype(np.float32)):
                return

            path = path.with_suffix('.png')

        if self.bit_depth == 16:
            alpha_int = (alpha * 65535).astype(np.uint16)
        else:
            alpha_int = (alpha * 255).astype(np.uint8)

        cv2.imwrite(str(path), alpha_int)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Professional edge refinement for alpha mattes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--alpha", "-a", required=True,
                       help="Directory with alpha mattes")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory")
    parser.add_argument("--frames", "-f",
                       help="Directory with RGB frames for color keying")

    # Edge settings
    parser.add_argument("--inner-band", type=int, default=8,
                       help="Inner edge band width (default: 8)")
    parser.add_argument("--outer-band", type=int, default=12,
                       help="Outer edge band width (default: 12)")
    parser.add_argument("--core-erosion", type=int, dest="core_shrink", default=3,
                       help="Core erosion pixels (alias for core-shrink)")
    parser.add_argument("--core-shrink", type=int, default=3,
                       help="Core shrink pixels (default: 3)")
    parser.add_argument("--edge-extend", type=int, default=2,
                       help="Edge extend pixels (default: 2)")

    # Softness
    parser.add_argument("--softness", type=float, default=1.0,
                       help="Edge softness (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="Falloff gamma curve (default: 1.0)")

    # Color keying
    parser.add_argument("--despill", type=float, default=0.5,
                       help="Despill strength 0-1 (default: 0.5)")

    # Output
    parser.add_argument("--format", default="exr",
                       choices=["exr", "png", "tiff"])
    parser.add_argument("--bit-depth", type=int, default=16,
                       choices=[8, 16, 32])

    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = EdgeRefineConfig(
        inner_band_width=args.inner_band,
        outer_band_width=args.outer_band,
        core_shrink=args.core_shrink,
        edge_extend=args.edge_extend,
        edge_softness=args.softness,
        falloff_gamma=args.gamma,
        despill_strength=args.despill
    )

    pipeline = EdgeRefinePipeline(
        config=config,
        output_format=args.format,
        bit_depth=args.bit_depth,
        verbose=args.verbose
    )

    pipeline.process_sequence(
        alpha_dir=Path(args.alpha),
        output_dir=Path(args.output),
        frames_dir=Path(args.frames) if args.frames else None
    )


if __name__ == "__main__":
    main()
