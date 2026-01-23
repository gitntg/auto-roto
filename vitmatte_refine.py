#!/usr/bin/env -S conda run -n autoroto python
"""
VITMATTE-BASED ALPHA REFINEMENT
===============================

Geometric-Guided Image Matting using ViTMatte.

This module implements the SOURCEOFTRUTH.md approach:
1. Use SAM2 mask as core foreground
2. Use Depth high-pass to find "geometric strands" (hair detail)
3. Build a Trimap (Foreground/Background/Unknown)
4. Let ViTMatte solve for actual alpha values

The key insight: Don't GUESS alpha from heuristics, let a trained
matting network SOLVE for it given the right trimap.

USAGE:
    # Standalone
    python vitmatte_refine.py --sam-mask ./sam/alpha/ --depth ./depth/ \\
        --frames ./frames/ --output ./vitmatte_output/

    # As module
    from vitmatte_refine import GeometricMatteRefiner
    refiner = GeometricMatteRefiner(config)
    alpha = refiner.process_frame(rgb, sam_mask, depth)
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Generator
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class TrimapConfig:
    """Configuration for trimap synthesis.

    DEFAULT: Adaptive Mode (Formulaic Distance Approach)
    - Unknown width varies based on local depth complexity
    - Smooth regions: ~2px, Complex regions (hair): up to 60px
    - Optional motion-aware expansion using optical flow

    Core Parameters:
    - Core Erosion: 10px for definite foreground
    - Depth Logic: High-pass (Laplacian) for geometric strands
    """

    # Core foreground (eroded SAM2 mask)
    core_erosion: int = 10          # Pixels to erode for definite foreground

    # Legacy mode: fixed dilation (used when adaptive_mode=False)
    unknown_dilation: int = 25      # Fixed dilation for legacy fallback

    # Depth high-pass for geometric strands
    highpass_threshold: float = 0.01  # Depth gradient threshold for "spikes"
    min_unknown_depth: float = 0.2    # Legacy: minimum depth for unknown zone

    # Depth confidence intervals (SOURCEOFTRUTH)
    # "Core Body: Depth > 0.8, Background: Depth < 0.2, Matting Zone: 0.4-0.6"
    depth_core_threshold: float = 0.8   # Depth above this = definite foreground
    depth_bg_threshold: float = 0.2     # Depth below this = definite background
    depth_unknown_low: float = 0.4      # Depth variance zone lower bound
    depth_unknown_high: float = 0.6     # Depth variance zone upper bound

    # Hair region focus (extended search in hair zone)
    hair_region_ratio: float = 0.5  # Top portion of bounding box (hair region)
    hair_dilation_boost: int = 15   # Extra dilation in hair region

    # Geometric continuity - DISABLED for maximum hair capture
    require_connectivity: bool = False
    min_strand_size: int = 20

    # Soft trimap options (gradual transition)
    soft_unknown: bool = False      # SOURCEOFTRUTH uses hard 128 for unknown
    unknown_softness: float = 0.5

    # Simple mode: just dilate the edge without depth analysis
    simple_edge_mode: bool = False
    simple_edge_width: int = 30

    # Linear color space processing
    use_linear_colorspace: bool = True  # Convert sRGB to linear for processing

    # === ADAPTIVE MODE (Formulaic Distance Approach) ===
    # Instead of hard-coded dilation, use depth gradient magnitude for local variance
    adaptive_mode: bool = True          # Adaptive trimap is now the default
    adaptive_base_px: float = 2.0       # Minimum reach (smooth regions like shoulders)
    adaptive_max_px: float = 60.0       # Maximum reach (complex regions like hair)
    adaptive_blur_kernel: int = 21      # Gaussian blur for complexity field smoothing

    # === MOTION-AWARE MODE (Optical Flow Weighting) ===
    # Expand unknown zone based on motion blur
    motion_aware: bool = False          # Enable motion-weighted trimap
    motion_max_speed: float = 30.0      # Max pixel motion to consider (clips outliers)
    motion_blur_kernel: int = 15        # Gaussian blur for motion field
    motion_weight: float = 20.0         # Motion multiplier for unknown zone expansion


@dataclass
class ViTMatteConfig:
    """Configuration for ViTMatte model."""

    model_size: str = "base"        # "small" or "base"
    device: str = "cuda"
    max_resolution: int = 2048      # Max dimension for processing (resize if larger)


@dataclass
class GeometricMatteConfig:
    """Full configuration for geometric matte refinement."""

    trimap: TrimapConfig = field(default_factory=TrimapConfig)
    vitmatte: ViTMatteConfig = field(default_factory=ViTMatteConfig)

    # Output
    output_dir: str = "./vitmatte_output"
    output_format: str = "exr"
    bit_depth: int = 16

    # Debug
    save_trimap: bool = True
    save_debug: bool = False
    verbose: bool = False


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("ViTMatteRefine")


# ==============================================================================
# COLOR SPACE UTILITIES (SOURCEOFTRUTH: Linear Color Space)
# ==============================================================================

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """
    Convert sRGB image to linear color space.

    SOURCEOFTRUTH: "Hair is dark. In sRGB (Gamma 2.2), the difference between
    dark hair and darker background is compressed into 2-3 integer values.
    Process in Linear Color Space (Gamma 1.0) to see sub-pixel differences."

    Args:
        img: sRGB image, float32 range [0, 1]

    Returns:
        Linear color space image, float32 range [0, 1]
    """
    # Standard sRGB to linear conversion
    # For values <= 0.04045: linear = srgb / 12.92
    # For values > 0.04045: linear = ((srgb + 0.055) / 1.055) ^ 2.4
    img = np.clip(img, 0, 1)
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((img + 0.055) / 1.055, 2.4)
    )
    return linear.astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """
    Convert linear color space image back to sRGB.

    Args:
        img: Linear image, float32 range [0, 1]

    Returns:
        sRGB image, float32 range [0, 1]
    """
    img = np.clip(img, 0, 1)
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(img, 1/2.4) - 0.055
    )
    return srgb.astype(np.float32)


# ==============================================================================
# TRIMAP SYNTHESIS
# ==============================================================================

class TrimapSynthesizer:
    """
    Builds trimap from SAM2 mask + Depth map.

    Trimap values:
        0   = Definite Background (black)
        128 = Unknown (gray) - where hair/fine detail lives
        255 = Definite Foreground (white)

    The unknown zone is found using depth high-pass filtering:
    - High-pass reveals "depth spikes" (sudden depth changes)
    - Hair strands show as spikes extending from the head
    - We mark these as unknown for ViTMatte to solve
    """

    def __init__(self, config: TrimapConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger("TrimapSynth")

    def compute_depth_highpass(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to depth map to find "depth spikes".

        Hair strands create sudden depth discontinuities that show up
        as high values in the Laplacian (high-pass) of the depth map.

        Args:
            depth: Float depth map (H, W), values 0-1

        Returns:
            High-pass magnitude (H, W), normalized 0-1
        """
        import cv2

        # Laplacian is the classic high-pass filter
        # It highlights regions of rapid intensity change
        laplacian = cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F, ksize=5)

        # We want magnitude (absolute value)
        highpass = np.abs(laplacian)

        # Also compute gradient magnitude (Sobel) for edges
        grad_x = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Combine Laplacian and gradient for robust edge detection
        combined = highpass + gradient_mag * 0.5

        # Normalize
        if combined.max() > 0:
            combined = combined / combined.max()

        return combined.astype(np.float32)

    def detect_hair_region(self, mask: np.ndarray) -> np.ndarray:
        """
        Detect the hair region (top portion of the subject).

        Hair is typically at the top of a person's bounding box.
        We give this region extra unknown zone dilation.

        Args:
            mask: Binary mask (H, W)

        Returns:
            Hair region mask (H, W), float 0-1
        """
        import cv2

        mask_binary = (mask > 0.5).astype(np.uint8)

        # Find bounding box of the mask
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros_like(mask, dtype=np.float32)

        # Get bounding box of largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Hair region = top portion of bounding box
        hair_height = int(h * self.config.hair_region_ratio)

        hair_region = np.zeros_like(mask, dtype=np.float32)
        # Extend slightly above the bounding box (hair may go up)
        y_start = max(0, y - int(h * 0.15))
        hair_region[y_start:y + hair_height, x:x + w] = 1.0

        # Dilate to include area around hair
        dilate_size = self.config.hair_dilation_boost * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        hair_region = cv2.dilate(hair_region.astype(np.uint8), kernel).astype(np.float32)

        return hair_region

    def filter_by_connectivity(
        self,
        unknown_mask: np.ndarray,
        core_mask: np.ndarray
    ) -> np.ndarray:
        """
        Filter unknown zone to only include regions connected to core.

        This implements the SOURCEOFTRUTH constraint:
        "A strand of hair must have a continuous depth path back to the scalp."

        Args:
            unknown_mask: Binary unknown zone mask
            core_mask: Binary core foreground mask

        Returns:
            Filtered unknown mask (only connected components)
        """
        import cv2

        # Dilate core slightly to ensure connection detection
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core_dilated = cv2.dilate(core_mask.astype(np.uint8), dilate_kernel)

        # Combine core and unknown for connectivity analysis
        combined = np.maximum(unknown_mask, core_dilated).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined, connectivity=8
        )

        # Find which labels touch the core
        core_labels = set(np.unique(labels[core_dilated > 0]))

        # Keep only unknown pixels that are in core-connected components
        filtered = np.zeros_like(unknown_mask)
        for label in core_labels:
            if label == 0:  # Skip background
                continue
            component_mask = (labels == label)
            # Only keep the unknown portion (not the core itself)
            filtered[component_mask & (unknown_mask > 0)] = 1

        # Also filter by minimum size
        # Re-analyze just the unknown regions
        unknown_only = (filtered > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unknown_only)

        final = np.zeros_like(unknown_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= self.config.min_strand_size:
                final[labels == label] = 1

        return final.astype(np.float32)

    def _compute_complexity_map(self, depth: np.ndarray) -> np.ndarray:
        """
        Compute depth gradient magnitude as a "complexity" or "messiness" factor.

        Areas with high depth variance (hair, fine detail) get high values.
        Smooth areas (shoulders, walls) get low values.

        This is the core of the Formulaic Distance Approach.
        """
        import cv2

        # Sobel gradients in X and Y
        gX = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        gY = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gX, gY)

        # Normalize to 0-1
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

        # Gaussian blur to create "field of influence"
        # Hair strands expand the complexity zone around them
        kernel_size = self.config.adaptive_blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd
        complexity_map = cv2.GaussianBlur(magnitude, (kernel_size, kernel_size), 0)

        self.logger.debug(f"ADAPTIVE: Complexity map range: {complexity_map.min():.3f} - {complexity_map.max():.3f}")

        return complexity_map

    def _compute_motion_factor(
        self,
        prev_frame_gray: np.ndarray,
        curr_frame_gray: np.ndarray
    ) -> np.ndarray:
        """
        Compute motion factor using Farneback optical flow.

        Fast-moving areas (motion blur) get expanded unknown zones.
        Static areas use base unknown width.
        """
        import cv2

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame_gray, curr_frame_gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Convert vectors to magnitude (speed)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Clip extreme values (camera pans, etc.)
        mag = np.clip(mag, 0, self.config.motion_max_speed)

        # Normalize to 0-1
        motion_factor = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

        # Blur to expand influence (motion blur trails behind movement)
        kernel_size = self.config.motion_blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        motion_factor = cv2.GaussianBlur(motion_factor, (kernel_size, kernel_size), 0)

        self.logger.debug(f"MOTION: Motion factor range: {motion_factor.min():.3f} - {motion_factor.max():.3f}")

        return motion_factor

    def _adaptive_trimap(
        self,
        mask_binary: np.ndarray,
        depth: np.ndarray,
        prev_frame_gray: Optional[np.ndarray] = None,
        curr_frame_gray: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Adaptive Trimap using Formulaic Distance Approach.

        Instead of hard-coded dilation values, the unknown zone width
        varies based on local depth complexity and motion.

        Formula:
            dynamic_threshold = base_px + (max_px * complexity_map) + (motion_weight * motion_factor)

        - Smooth regions (shoulders, walls): ~2px unknown
        - Complex regions (hair, fine detail): up to 60px unknown
        - Motion areas: additional expansion for blur
        """
        import cv2

        h, w = mask_binary.shape

        # =====================================================================
        # STEP 1: COMPUTE COMPLEXITY MAP (Depth Gradient Magnitude)
        # =====================================================================
        complexity_map = self._compute_complexity_map(depth)

        # =====================================================================
        # STEP 2: COMPUTE MOTION FACTOR (Optional)
        # =====================================================================
        motion_factor = np.zeros((h, w), dtype=np.float32)
        if self.config.motion_aware and prev_frame_gray is not None and curr_frame_gray is not None:
            motion_factor = self._compute_motion_factor(prev_frame_gray, curr_frame_gray)

        # =====================================================================
        # STEP 3: COMPUTE DISTANCE FROM CORE BODY
        # =====================================================================
        # Invert mask: 0 is body, 255 is background
        inv_mask = cv2.bitwise_not(mask_binary * 255)
        # Distance transform: each pixel's distance from the body
        dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

        # =====================================================================
        # STEP 4: THE DYNAMIC FORMULA
        # =====================================================================
        # dynamic_threshold = base + (max * complexity) + (weight * motion)
        dynamic_threshold = (
            self.config.adaptive_base_px +
            (self.config.adaptive_max_px * complexity_map) +
            (self.config.motion_weight * motion_factor)
        )

        self.logger.debug(
            f"ADAPTIVE: Dynamic threshold range: {dynamic_threshold.min():.1f}px - {dynamic_threshold.max():.1f}px"
        )

        # =====================================================================
        # STEP 5: CORE FOREGROUND (Erode for definite FG)
        # =====================================================================
        erosion_size = self.config.core_erosion * 2 + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        core_fg = cv2.erode(mask_binary, erode_kernel, iterations=1)

        # =====================================================================
        # STEP 6: BUILD TRIMAP
        # =====================================================================
        # Unknown zone = pixels where distance < dynamic_threshold AND distance > 0
        unknown_zone = (dist_map < dynamic_threshold) & (dist_map > 0)

        # Build final trimap
        trimap = np.zeros((h, w), dtype=np.uint8)
        trimap[core_fg > 0] = 255           # Definite Foreground
        trimap[unknown_zone] = 128          # Adaptive Unknown

        # Stats
        fg_pixels = np.sum(trimap == 255)
        unknown_pixels = np.sum(trimap == 128)
        bg_pixels = np.sum(trimap == 0)

        # Compute average unknown width for logging
        unknown_widths = dynamic_threshold[unknown_zone] if np.any(unknown_zone) else np.array([0])
        avg_width = np.mean(unknown_widths) if len(unknown_widths) > 0 else 0

        self.logger.info(
            f"ADAPTIVE Trimap: FG={fg_pixels:,}, Unknown={unknown_pixels:,}, BG={bg_pixels:,}"
        )
        self.logger.info(f"ADAPTIVE: Average unknown width: {avg_width:.1f}px")

        return trimap

    def _simple_edge_trimap(self, mask_binary: np.ndarray) -> np.ndarray:
        """
        Create a simple trimap by dilating the mask edge.

        This is a fallback mode that doesn't use depth analysis,
        just creates a band of unknown around the mask edge.
        """
        import cv2

        h, w = mask_binary.shape

        # Erode for core foreground
        erosion_size = self.config.core_erosion * 2 + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        core_fg = cv2.erode(mask_binary, erode_kernel, iterations=1)

        # Dilate for outer boundary
        dilation_size = self.config.simple_edge_width * 2 + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        dilated = cv2.dilate(mask_binary, dilate_kernel, iterations=1)

        # Unknown = dilated minus core
        unknown_zone = dilated.astype(np.float32) - core_fg.astype(np.float32)
        unknown_zone = np.clip(unknown_zone, 0, 1)

        if self.config.soft_unknown:
            # Create soft trimap with gradual transition
            dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
            dist_outside = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)

            # Normalize distances
            edge_width = self.config.simple_edge_width
            inside_factor = np.clip(dist_inside / edge_width, 0, 1)
            outside_factor = np.clip(dist_outside / edge_width, 0, 1)

            # Build soft trimap: 0 = bg, 0.5 = unknown, 1 = fg
            trimap_soft = np.zeros((h, w), dtype=np.float32)
            trimap_soft[mask_binary > 0] = 0.5 + inside_factor[mask_binary > 0] * 0.5
            trimap_soft[mask_binary == 0] = 0.5 - outside_factor[mask_binary == 0] * 0.5

            # Convert to 0-255 range
            trimap = (trimap_soft * 255).astype(np.uint8)
        else:
            # Hard trimap
            trimap = np.zeros((h, w), dtype=np.uint8)
            trimap[unknown_zone > 0] = 128
            trimap[core_fg > 0] = 255

        unknown_pixels = np.sum((trimap > 10) & (trimap < 245))
        self.logger.info(f"Simple edge trimap: FG={np.sum(trimap > 245)}, Unknown={unknown_pixels}, BG={np.sum(trimap < 10)}")

        return trimap

    def _scientific_trimap(self, mask_binary: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        SOURCEOFTRUTH Golden Rule Implementation.

        This implements the "Science" of trimap synthesis as specified in SOURCEOFTRUTH.md:

        Hard Numbers:
        - Core Erosion: 10px (definite foreground)
        - Wide Dilation: 25px (context window for ViTMatte attention)
        - Unknown Width: 25-40px minimum (essential for Transformer)
        - Depth Logic: High-pass (Laplacian) for edges, NOT absolute thresholds

        The Golden Rule:
        ```
        core_mask = erode(sam_mask, 10px)
        wide_search_area = dilate(sam_mask, 25px)
        trimap[core_mask == 1] = 255       # Definite FG
        trimap[wide_search_area == 0] = 0  # Definite BG
        # Everything else = Unknown (128)
        ```
        """
        import cv2

        h, w = mask_binary.shape

        # =====================================================================
        # STEP 1: CORE FOREGROUND (Erode by 10px)
        # SOURCEOFTRUTH: "Erode SAM2 by 10px for definite foreground"
        # =====================================================================
        erosion_size = self.config.core_erosion * 2 + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        core_fg = cv2.erode(mask_binary, erode_kernel, iterations=1)

        self.logger.debug(f"SCIENTIFIC: Core FG (erode {self.config.core_erosion}px): {np.sum(core_fg > 0):,} pixels")

        # =====================================================================
        # STEP 2: WIDE SEARCH AREA (Dilate by 25px)
        # SOURCEOFTRUTH: "Dilate SAM2 by 25px for wide context window"
        # "Unknown region must be 25-40px wide for ViTMatte attention"
        # =====================================================================
        dilation_size = self.config.unknown_dilation * 2 + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        wide_search_area = cv2.dilate(mask_binary, dilate_kernel, iterations=1)

        self.logger.debug(f"SCIENTIFIC: Wide search (dilate {self.config.unknown_dilation}px): {np.sum(wide_search_area > 0):,} pixels")

        # =====================================================================
        # STEP 3: DEPTH HIGH-PASS (Laplacian for geometric strands)
        # SOURCEOFTRUTH: "Find high-frequency depth details (strands) using Laplacian"
        # =====================================================================
        depth_highpass = self.compute_depth_highpass(depth)
        depth_spikes = (depth_highpass > self.config.highpass_threshold).astype(np.uint8)

        self.logger.debug(f"SCIENTIFIC: Depth spikes (Laplacian): {np.sum(depth_spikes > 0):,} pixels")

        # =====================================================================
        # STEP 4: DEPTH CONFIDENCE INTERVALS (Optional enhancement)
        # SOURCEOFTRUTH: "Core: Depth > 0.8, BG: Depth < 0.2, Unknown: 0.4-0.6"
        # =====================================================================
        # Pixels in the "ambiguous" depth range should be treated as Unknown
        depth_ambiguous = (
            (depth >= self.config.depth_unknown_low) &
            (depth <= self.config.depth_unknown_high)
        )

        self.logger.debug(f"SCIENTIFIC: Depth ambiguous zone ({self.config.depth_unknown_low}-{self.config.depth_unknown_high}): {np.sum(depth_ambiguous):,} pixels")

        # =====================================================================
        # STEP 5: BUILD TRIMAP (The Golden Rule)
        # SOURCEOFTRUTH: "Everything between core and wide area = Unknown (128)"
        # =====================================================================
        trimap = np.zeros((h, w), dtype=np.uint8)

        # Start with the basic zone-based trimap
        # Everything inside the wide search area but outside core = Unknown
        base_unknown = (wide_search_area > 0) & (core_fg == 0)

        # Enhance unknown zone with depth spikes that extend beyond the search area
        # (hair strands that weren't caught by simple dilation)
        # These must be near the mask edge (within reasonable distance)
        dist_outside = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)
        max_strand_reach = self.config.unknown_dilation + self.config.hair_dilation_boost
        extended_spikes = (
            (depth_spikes > 0) &
            (dist_outside > 0) &
            (dist_outside < max_strand_reach) &
            (core_fg == 0)
        )

        self.logger.debug(f"SCIENTIFIC: Extended spikes beyond dilation: {np.sum(extended_spikes & ~base_unknown):,} pixels")

        # Combine base unknown zone with depth spike extensions
        unknown_zone = base_unknown | extended_spikes

        # Also add depth-ambiguous pixels near the boundary
        near_mask = dist_outside < max_strand_reach
        ambiguous_unknown = depth_ambiguous & near_mask & (core_fg == 0)
        unknown_zone = unknown_zone | ambiguous_unknown

        # Apply to trimap
        trimap[unknown_zone] = 128
        trimap[core_fg > 0] = 255

        # Stats
        fg_pixels = np.sum(trimap == 255)
        unknown_pixels = np.sum(trimap == 128)
        bg_pixels = np.sum(trimap == 0)

        self.logger.info(f"SCIENTIFIC Trimap: FG={fg_pixels:,}, Unknown={unknown_pixels:,}, BG={bg_pixels:,}")
        self.logger.info(f"Unknown zone width: ~{self.config.core_erosion + self.config.unknown_dilation}px (min 25px required)")

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
        Synthesize trimap from SAM2 mask and depth map.

        Args:
            sam_mask: SAM2 alpha mask (H, W), float 0-1
            depth: Depth map (H, W), float 0-1 (higher = closer)
            rgb: Optional RGB image for additional analysis
            prev_frame_gray: Previous frame grayscale (for motion-aware mode)
            curr_frame_gray: Current frame grayscale (for motion-aware mode)

        Returns:
            Trimap (H, W), uint8 with values 0, 128, 255 (or gradual if soft_unknown)
        """
        import cv2

        sam_mask = sam_mask.astype(np.float32)
        depth = depth.astype(np.float32)

        h, w = sam_mask.shape
        mask_binary = (sam_mask > 0.5).astype(np.uint8)

        # =====================================================================
        # SIMPLE EDGE MODE: Just dilate the edge
        # =====================================================================
        if self.config.simple_edge_mode:
            return self._simple_edge_trimap(mask_binary)

        # =====================================================================
        # ADAPTIVE MODE: Formulaic Distance + Motion-Aware Trimap (DEFAULT)
        # Uses local depth complexity for variable unknown width
        # =====================================================================
        if self.config.adaptive_mode:
            return self._adaptive_trimap(mask_binary, depth, prev_frame_gray, curr_frame_gray)

        # =====================================================================
        # LEGACY MODE: Complex depth analysis (fallback if adaptive disabled)
        # =====================================================================
        # STEP 1: DEFINITE FOREGROUND (Core)
        # Erode SAM2 mask to get pixels we're 100% sure are foreground
        erosion_size = self.config.core_erosion * 2 + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        core_fg = cv2.erode(mask_binary, erode_kernel, iterations=1)

        self.logger.debug(f"Core foreground: {np.sum(core_fg > 0)} pixels")

        # STEP 2: DEPTH HIGH-PASS (Find Geometric Strands)
        depth_highpass = self.compute_depth_highpass(depth)

        # Threshold to find significant depth spikes
        depth_spikes = (depth_highpass > self.config.highpass_threshold).astype(np.float32)

        self.logger.debug(f"Depth spikes: {np.sum(depth_spikes > 0)} pixels")

        # STEP 3: UNKNOWN ZONE (Near boundary + depth spikes + fg depth)

        # Distance from SAM2 boundary
        dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)

        # Hair region gets extra dilation
        hair_region = self.detect_hair_region(sam_mask)

        # Base unknown radius + hair boost where applicable
        effective_radius = self.config.unknown_dilation + \
                          (hair_region * self.config.hair_dilation_boost)

        # Spatial constraint: near the SAM2 boundary
        near_boundary = (dist_outside < effective_radius) | (dist_inside < self.config.core_erosion)

        # Depth constraint: must have foreground-like depth
        # Get foreground depth reference from core
        if np.sum(core_fg > 0) > 100:
            fg_depth_values = depth[core_fg > 0]
            fg_depth_min = np.percentile(fg_depth_values, 5)
            fg_depth_median = np.median(fg_depth_values)
        else:
            fg_depth_min = 0.3
            fg_depth_median = 0.5

        # Background depth reference
        far_from_mask = dist_outside > 100
        if np.sum(far_from_mask) > 100:
            bg_depth_max = np.percentile(depth[far_from_mask], 95)
        else:
            bg_depth_max = 0.3

        # Foreground depth threshold (above background)
        fg_depth_threshold = max(bg_depth_max + 0.05, self.config.min_unknown_depth)

        has_fg_depth = depth > fg_depth_threshold

        self.logger.debug(f"FG depth threshold: {fg_depth_threshold:.3f}")
        self.logger.debug(f"Pixels with FG depth: {np.sum(has_fg_depth)}")

        # =====================================================================
        # STEP 4: COMBINE CONSTRAINTS FOR UNKNOWN ZONE
        # =====================================================================
        # Key insight from SOURCEOFTRUTH: Use depth GRADIENT, not absolute depth.
        # Hair strands have lower absolute depth but show gradient variation.

        # Near the actual edge (within a few pixels)
        # Note: dist transforms return 0 for "obstacle" pixels, so we need to mask properly
        near_edge_outside = (dist_outside < 5) & (dist_outside > 0)  # BG pixels near edge
        near_edge_inside = (dist_inside < 5) & (dist_inside > 0)    # FG pixels near edge (not core)
        very_near_edge = near_edge_outside | near_edge_inside

        # Depth spike regions follow gradient regardless of absolute depth
        # This captures hair strands that have depth variation even if lower than core
        # Key: don't require has_fg_depth for spike regions - hair has lower depth
        depth_spike_unknown = (
            near_boundary &           # Near SAM2 boundary (within dilation distance)
            (depth_spikes > 0) &      # Has depth gradient (hair strand signature)
            (core_fg == 0)            # Not in core foreground
        )

        # Thin edge band for non-spike regions (standard edge refinement)
        # This is just a narrow band for general edge softening
        edge_band_unknown = (
            very_near_edge &          # Within 5px of edge
            (core_fg == 0)            # Not in core
        )

        # Combine: depth spikes (for hair) + thin edge band (for general edges)
        unknown_zone = (depth_spike_unknown | edge_band_unknown).astype(np.float32)

        self.logger.debug(f"Depth spike unknown: {np.sum(depth_spike_unknown)} pixels")
        self.logger.debug(f"Edge band unknown: {np.sum(edge_band_unknown)} pixels")

        self.logger.debug(f"Unknown zone (before connectivity): {np.sum(unknown_zone > 0)} pixels")

        # =====================================================================
        # STEP 5: GEOMETRIC CONTINUITY FILTER
        # =====================================================================
        if self.config.require_connectivity:
            unknown_zone = self.filter_by_connectivity(unknown_zone, core_fg)
            self.logger.debug(f"Unknown zone (after connectivity): {np.sum(unknown_zone > 0)} pixels")

        # =====================================================================
        # STEP 6: BUILD TRIMAP
        # =====================================================================

        if self.config.soft_unknown:
            # Create soft trimap with gradual transition
            # Use distance transforms for smooth falloff

            # Distance from unknown zone
            unknown_dist = cv2.distanceTransform(
                (1 - unknown_zone).astype(np.uint8), cv2.DIST_L2, 5
            )

            # Create soft unknown: peaks at 128, falls off with distance
            softness = self.config.unknown_softness * 30  # Scale factor
            unknown_soft = np.exp(-unknown_dist / max(softness, 1)) * 0.5

            # Build trimap as float first
            trimap_float = np.zeros((h, w), dtype=np.float32)

            # Background starts at 0
            # Add soft unknown zone
            trimap_float += unknown_soft

            # SAM2 mask interior (not eroded) gets ramped to 1
            sam_interior = sam_mask > 0.5
            trimap_float[sam_interior] = np.maximum(
                trimap_float[sam_interior],
                sam_mask[sam_interior]
            )

            # Core foreground is definitely 1
            trimap_float[core_fg > 0] = 1.0

            # Clamp and convert to uint8
            trimap = (np.clip(trimap_float, 0, 1) * 255).astype(np.uint8)

        else:
            # Hard trimap
            trimap = np.zeros((h, w), dtype=np.uint8)

            # Unknown = 128
            trimap[unknown_zone > 0] = 128

            # Foreground = 255 (overwrites unknown if overlapping)
            trimap[core_fg > 0] = 255

            # Also mark original SAM2 mask interior (not just eroded) as foreground
            # This preserves SAM2's edge in non-hair regions
            # But only where we don't have unknown zone
            sam_interior = (sam_mask > 0.5) & (unknown_zone == 0)
            trimap[sam_interior] = 255

        # Count regions
        fg_pixels = np.sum(trimap > 245)
        unknown_pixels = np.sum((trimap > 10) & (trimap < 245))
        bg_pixels = np.sum(trimap < 10)

        self.logger.info(f"Trimap: FG={fg_pixels}, Unknown={unknown_pixels}, BG={bg_pixels}")

        return trimap


# ==============================================================================
# VITMATTE REFINER
# ==============================================================================

class ViTMatteRefiner:
    """
    Runs ViTMatte on RGB + Trimap to produce final alpha matte.

    ViTMatte is a Vision Transformer-based matting model that excels
    at resolving fine detail (hair, fur, transparency) given a good trimap.
    """

    MODEL_IDS = {
        'small': 'hustvl/vitmatte-small-composition-1k',
        'base': 'hustvl/vitmatte-base-composition-1k',
    }

    def __init__(self, config: ViTMatteConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger("ViTMatte")

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load ViTMatte model from HuggingFace."""
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        model_id = self.MODEL_IDS.get(self.config.model_size, self.MODEL_IDS['base'])

        self.logger.info(f"Loading ViTMatte ({self.config.model_size}) from {model_id}...")

        self.processor = VitMatteImageProcessor.from_pretrained(model_id)
        self.model = VitMatteForImageMatting.from_pretrained(model_id)
        self.model = self.model.to(self.config.device).eval()

        self.logger.info("ViTMatte loaded successfully")

    def refine(
        self,
        rgb: np.ndarray,
        trimap: np.ndarray
    ) -> np.ndarray:
        """
        Run ViTMatte to produce alpha matte.

        Args:
            rgb: RGB image (H, W, 3), uint8 or float
            trimap: Trimap (H, W), uint8 with values 0, 128, 255

        Returns:
            Alpha matte (H, W), float32 0-1
        """
        import torch
        from PIL import Image
        import cv2

        # Ensure correct formats
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        if len(rgb.shape) == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        h_orig, w_orig = rgb.shape[:2]

        # Ensure trimap is single channel uint8
        if len(trimap.shape) == 3:
            trimap = trimap[:, :, 0]
        trimap = trimap.astype(np.uint8)

        # Check if we need to resize for memory efficiency
        max_dim = max(h_orig, w_orig)
        scale_factor = 1.0
        if max_dim > self.config.max_resolution:
            scale_factor = self.config.max_resolution / max_dim
            new_h = int(h_orig * scale_factor)
            new_w = int(w_orig * scale_factor)

            self.logger.info(f"Resizing {w_orig}x{h_orig} -> {new_w}x{new_h} for processing")

            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        h, w = rgb.shape[:2]

        # Convert to PIL - trimap should be grayscale ('L' mode)
        rgb_pil = Image.fromarray(rgb)
        trimap_pil = Image.fromarray(trimap, mode='L')

        # Process inputs
        inputs = self.processor(
            images=rgb_pil,
            trimaps=trimap_pil,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract alpha
        alpha = outputs.alphas

        # Handle potential size mismatch (ViTMatte may output different size)
        if alpha.shape[-2:] != (h, w):
            alpha = torch.nn.functional.interpolate(
                alpha,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

        # Convert to numpy
        alpha = alpha.squeeze().cpu().numpy()

        # Upscale back to original resolution if we downscaled
        if scale_factor < 1.0:
            self.logger.debug(f"Upscaling alpha back to {w_orig}x{h_orig}")
            alpha = cv2.resize(alpha, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Ensure 0-1 range
        alpha = np.clip(alpha, 0, 1).astype(np.float32)

        return alpha

    def release(self):
        """Release GPU memory."""
        import torch

        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None

        torch.cuda.empty_cache()


# ==============================================================================
# GEOMETRIC MATTE REFINER (Main Pipeline)
# ==============================================================================

class GeometricMatteRefiner:
    """
    Full Geometric-Guided Image Matting pipeline.

    Orchestrates:
        SAM2 Mask + Depth → TrimapSynthesizer → Trimap → ViTMatte → Alpha
    """

    def __init__(self, config: GeometricMatteConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or setup_logging(config.verbose)

        self.trimap_synth = TrimapSynthesizer(config.trimap, self.logger)
        self.vitmatte = ViTMatteRefiner(config.vitmatte, self.logger)

    def process_frame(
        self,
        rgb: np.ndarray,
        sam_mask: np.ndarray,
        depth: np.ndarray,
        save_trimap_path: Path = None,
        prev_rgb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process a single frame through the full pipeline.

        Args:
            rgb: RGB image (H, W, 3)
            sam_mask: SAM2 alpha mask (H, W), float 0-1
            depth: Depth map (H, W), float 0-1
            save_trimap_path: Optional path to save trimap visualization
            prev_rgb: Previous frame RGB (for motion-aware adaptive mode)

        Returns:
            Final alpha matte (H, W), float32 0-1
        """
        import cv2

        # Step 1: Prepare frames for motion-aware mode
        prev_frame_gray = None
        curr_frame_gray = None
        if self.config.trimap.motion_aware and prev_rgb is not None:
            # Convert to grayscale for optical flow
            curr_frame_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            prev_frame_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)

        # Step 2: Synthesize trimap
        self.logger.debug("Synthesizing trimap...")
        trimap = self.trimap_synth.synthesize(
            sam_mask, depth, rgb,
            prev_frame_gray=prev_frame_gray,
            curr_frame_gray=curr_frame_gray
        )

        # Save trimap if requested
        if save_trimap_path is not None:
            # Create color visualization
            trimap_vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
            trimap_vis[trimap == 0] = [0, 0, 0]       # Black = background
            trimap_vis[trimap == 128] = [128, 128, 128]  # Gray = unknown
            trimap_vis[trimap == 255] = [255, 255, 255]  # White = foreground
            cv2.imwrite(str(save_trimap_path), trimap_vis)

        # Step 2: Run ViTMatte
        self.logger.debug("Running ViTMatte...")
        alpha = self.vitmatte.refine(rgb, trimap)

        # Step 3: Guided Filter (SOURCEOFTRUTH recommendation)
        # "Takes your rough AI-generated Alpha and says: 'Smooth this out,
        # but respect the edges found in the original high-res RGB footage.'
        # This is how you get that professional 'snap' where the matte
        # perfectly follows the texture of the hair."
        if self.config.trimap.use_linear_colorspace:
            try:
                from cv2 import ximgproc
                # Convert RGB to linear for better edge detection
                rgb_linear = srgb_to_linear(rgb.astype(np.float32) / 255.0)

                # SOURCEOFTRUTH: radius 2-4, eps 1e-6
                alpha_refined = ximgproc.guidedFilter(
                    guide=rgb_linear,
                    src=alpha.astype(np.float32),
                    radius=4,
                    eps=1e-6
                )
                alpha = np.clip(alpha_refined, 0, 1).astype(np.float32)
                self.logger.debug("Applied Guided Filter refinement")
            except ImportError:
                self.logger.warning("cv2.ximgproc not available - skipping Guided Filter")
            except Exception as e:
                self.logger.warning(f"Guided Filter failed: {e} - using raw alpha")

        # Step 4: Ensure core foreground is solid
        # ViTMatte should handle this, but we enforce it
        core_fg = trimap == 255
        alpha[core_fg] = np.maximum(alpha[core_fg], 0.99)

        return alpha

    def release(self):
        """Release GPU memory."""
        self.vitmatte.release()


# ==============================================================================
# FILE I/O
# ==============================================================================

def load_image(path: Path) -> np.ndarray:
    """Load an image file (handles EXR, PNG, etc.)."""
    import cv2

    ext = path.suffix.lower()

    if ext == '.exr':
        # Try OpenEXR first
        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()

            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            channels = header['channels'].keys()
            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            if 'R' in channels and 'G' in channels and 'B' in channels:
                r = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape((height, width))
                g = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape((height, width))
                b = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape((height, width))
                return np.stack([r, g, b], axis=-1)
            elif 'Y' in channels:
                y = np.frombuffer(exr_file.channel('Y', pt), dtype=np.float32).reshape((height, width))
                return y
            elif 'A' in channels:
                a = np.frombuffer(exr_file.channel('A', pt), dtype=np.float32).reshape((height, width))
                return a
            else:
                ch = list(channels)[0]
                data = np.frombuffer(exr_file.channel(ch, pt), dtype=np.float32).reshape((height, width))
                return data

        except ImportError:
            pass

    # Fallback to OpenCV
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise IOError(f"Cannot read image: {path}")

    # Handle different formats
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # Convert BGR to RGB if color
    if len(img.shape) == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_alpha(alpha: np.ndarray, path: Path, bit_depth: int = 16):
    """Save alpha matte to file."""
    import cv2

    ext = path.suffix.lower()

    if ext == '.exr':
        try:
            import OpenEXR
            import Imath

            h, w = alpha.shape

            header = OpenEXR.Header(w, h)
            if bit_depth == 32:
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                data = alpha.astype(np.float32)
            else:
                pixel_type = Imath.PixelType(Imath.PixelType.HALF)
                data = alpha.astype(np.float16)

            header['channels'] = {'A': Imath.Channel(pixel_type)}

            exr = OpenEXR.OutputFile(str(path), header)
            exr.writePixels({'A': data.tobytes()})
            exr.close()
            return

        except ImportError:
            pass

    # Fallback to OpenCV
    if bit_depth == 16:
        alpha_int = (alpha * 65535).astype(np.uint16)
    else:
        alpha_int = (alpha * 255).astype(np.uint8)

    cv2.imwrite(str(path), alpha_int)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ViTMatte-based Alpha Refinement using Geometric Trimap Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process with SAM masks and depth maps
  %(prog)s --sam-mask ./sam/alpha/ --depth ./depth/ --frames ./frames/ --output ./output/

  # Adjust trimap parameters
  %(prog)s --sam-mask ./sam/alpha/ --depth ./depth/ --frames ./frames/ \\
      --unknown-radius 60 --highpass-threshold 0.015 --output ./output/

  # Use smaller model for speed
  %(prog)s --sam-mask ./sam/alpha/ --depth ./depth/ --frames ./frames/ \\
      --model-size small --output ./output/
        """
    )

    # Input
    parser.add_argument("--sam-mask", "-s", required=True,
                       help="Directory with SAM2 alpha masks")
    parser.add_argument("--depth", "-d", required=True,
                       help="Directory with depth maps (float EXR)")
    parser.add_argument("--frames", "-f", required=True,
                       help="Directory with RGB frames")

    # Output
    parser.add_argument("--output", "-o", default="./vitmatte_output",
                       help="Output directory")

    # Trimap settings
    parser.add_argument("--no-adaptive", action="store_true",
                       help="Disable adaptive mode, use legacy fixed-dilation trimap")
    parser.add_argument("--core-erosion", type=int, default=10,
                       help="Erosion for core foreground (default: 10px)")
    parser.add_argument("--highpass-threshold", type=float, default=0.01,
                       help="Depth high-pass threshold (default: 0.01)")
    parser.add_argument("--hair-boost", type=int, default=15,
                       help="Extra dilation in hair region (default: 15)")
    parser.add_argument("--connectivity", action="store_true",
                       help="Enable geometric connectivity filter (disabled by default)")
    parser.add_argument("--min-depth", type=float, default=0.2,
                       help="Minimum depth for unknown zone (default: 0.2)")

    # Depth confidence intervals (SOURCEOFTRUTH)
    parser.add_argument("--depth-core", type=float, default=0.8,
                       help="Depth threshold for core body (SOURCEOFTRUTH: 0.8)")
    parser.add_argument("--depth-bg", type=float, default=0.2,
                       help="Depth threshold for background (SOURCEOFTRUTH: 0.2)")
    parser.add_argument("--depth-unknown-low", type=float, default=0.4,
                       help="Lower bound of depth ambiguous zone (SOURCEOFTRUTH: 0.4)")
    parser.add_argument("--depth-unknown-high", type=float, default=0.6,
                       help="Upper bound of depth ambiguous zone (SOURCEOFTRUTH: 0.6)")

    # Soft trimap options (SOURCEOFTRUTH recommends hard trimap)
    parser.add_argument("--soft-trimap", action="store_true",
                       help="Use soft trimap with gradual falloff (default: hard)")
    parser.add_argument("--hard-trimap", action="store_true", default=True,
                       help="Use hard trimap (0/128/255) - SOURCEOFTRUTH recommended")
    parser.add_argument("--unknown-softness", type=float, default=0.5,
                       help="Softness of unknown zone falloff (default: 0.5)")

    # Simple edge mode (basic, no depth)
    parser.add_argument("--simple-mode", action="store_true",
                       help="Simple edge mode: just dilate mask without depth analysis")
    parser.add_argument("--simple-width", type=int, default=30,
                       help="Width of simple edge band (default: 30)")

    # Adaptive mode tuning (Formulaic Distance Approach - enabled by default)
    parser.add_argument("--adaptive-base", type=float, default=2.0,
                       help="Minimum unknown width for smooth regions (default: 2)")
    parser.add_argument("--adaptive-max", type=float, default=60.0,
                       help="Maximum unknown width for complex regions like hair (default: 60)")
    parser.add_argument("--adaptive-blur", type=int, default=21,
                       help="Blur kernel for complexity field smoothing (default: 21)")

    # Motion-aware mode (Optical Flow)
    parser.add_argument("--motion-aware", action="store_true",
                       help="Enable motion-weighted trimap using optical flow")
    parser.add_argument("--motion-max-speed", type=float, default=30.0,
                       help="Max pixel motion before clipping (default: 30)")
    parser.add_argument("--motion-weight", type=float, default=20.0,
                       help="Motion multiplier for unknown zone expansion (default: 20)")

    # ViTMatte settings
    parser.add_argument("--model-size", choices=["small", "base"], default="base",
                       help="ViTMatte model size (default: base)")
    parser.add_argument("--max-resolution", type=int, default=2048,
                       help="Max resolution for processing (resize if larger, default: 2048)")

    # Output settings
    parser.add_argument("--format", choices=["exr", "png"], default="exr",
                       help="Output format (default: exr)")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16, 32], default=16,
                       help="Output bit depth (default: 16)")

    # Debug
    parser.add_argument("--save-trimap", action="store_true",
                       help="Save trimap visualizations")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--device", default="cuda",
                       help="Device (cuda or cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config - adaptive mode is now the default
    config = GeometricMatteConfig(
        trimap=TrimapConfig(
            # Core parameters
            core_erosion=args.core_erosion,
            highpass_threshold=args.highpass_threshold,
            # Depth confidence intervals
            depth_core_threshold=args.depth_core,
            depth_bg_threshold=args.depth_bg,
            depth_unknown_low=args.depth_unknown_low,
            depth_unknown_high=args.depth_unknown_high,
            # Hair and connectivity
            hair_dilation_boost=args.hair_boost,
            require_connectivity=args.connectivity,
            # Soft trimap options
            soft_unknown=args.soft_trimap and not args.hard_trimap,
            unknown_softness=args.unknown_softness,
            # Simple mode
            simple_edge_mode=args.simple_mode,
            simple_edge_width=args.simple_width,
            # Linear color space
            use_linear_colorspace=True,
            # Adaptive mode (Formulaic Distance Approach) - DEFAULT
            adaptive_mode=not args.no_adaptive,
            adaptive_base_px=args.adaptive_base,
            adaptive_max_px=args.adaptive_max,
            adaptive_blur_kernel=args.adaptive_blur,
            # Motion-aware mode (Optical Flow)
            motion_aware=args.motion_aware,
            motion_max_speed=args.motion_max_speed,
            motion_weight=args.motion_weight,
        ),
        vitmatte=ViTMatteConfig(
            model_size=args.model_size,
            device=args.device,
            max_resolution=args.max_resolution,
        ),
        output_dir=args.output,
        output_format=args.format,
        bit_depth=args.bit_depth,
        save_trimap=args.save_trimap,
        verbose=args.verbose,
    )

    logger = setup_logging(config.verbose)

    logger.info("=" * 60)
    logger.info("ViTMatte Geometric Matte Refinement")
    logger.info("=" * 60)

    # Setup output directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_dir = output_dir / "alpha"
    alpha_dir.mkdir(exist_ok=True)

    if config.save_trimap:
        trimap_dir = output_dir / "trimap"
        trimap_dir.mkdir(exist_ok=True)

    # Initialize refiner
    refiner = GeometricMatteRefiner(config, logger)

    # Find input files
    sam_dir = Path(args.sam_mask)
    depth_dir = Path(args.depth)
    frames_dir = Path(args.frames)

    # Get sorted file lists
    sam_files = sorted(sam_dir.glob("*.exr")) + sorted(sam_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.exr"))
    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))

    logger.info(f"Found {len(sam_files)} SAM masks, {len(depth_files)} depth maps, {len(frame_files)} frames")

    # Log mode info
    if config.trimap.adaptive_mode:
        logger.info(f"ADAPTIVE MODE: base={config.trimap.adaptive_base_px}px, max={config.trimap.adaptive_max_px}px")
        if config.trimap.motion_aware:
            logger.info(f"MOTION-AWARE: weight={config.trimap.motion_weight}, max_speed={config.trimap.motion_max_speed}")

    # Track previous frame for motion-aware mode
    prev_rgb = None

    # Process each frame
    for idx, (sam_path, depth_path, frame_path) in enumerate(zip(sam_files, depth_files, frame_files)):
        logger.info(f"Processing frame {idx}: {frame_path.name}")

        # Load inputs
        sam_mask = load_image(sam_path)
        if len(sam_mask.shape) == 3:
            sam_mask = sam_mask[:, :, 0] if sam_mask.shape[2] == 1 else sam_mask.mean(axis=2)

        depth = load_image(depth_path)
        if len(depth.shape) == 3:
            depth = depth[:, :, 0] if depth.shape[2] == 1 else depth.mean(axis=2)

        rgb = load_image(frame_path)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

        # Process (pass prev_rgb for motion-aware mode)
        trimap_path = trimap_dir / f"trimap.{idx:04d}.png" if config.save_trimap else None
        alpha = refiner.process_frame(rgb, sam_mask, depth, trimap_path, prev_rgb=prev_rgb)

        # Store current frame for next iteration (motion-aware mode)
        if config.trimap.motion_aware:
            prev_rgb = rgb.copy()

        # Save
        output_path = alpha_dir / f"vitmatte.{idx:04d}.{config.output_format}"
        save_alpha(alpha, output_path, config.bit_depth)

        logger.debug(f"  Saved: {output_path}")

    # Cleanup
    refiner.release()

    logger.info("=" * 60)
    logger.info("ViTMatte Refinement Complete!")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
