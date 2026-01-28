#!/usr/bin/env python
# Enable OpenCV's OpenEXR codec before importing cv2
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

"""
DEPTH-GUIDED ALPHA REFINEMENT
=============================

Uses Depth Anything 3 (DA3) to refine alpha mattes from AUTO-ROTO.

Depth discontinuities indicate object boundaries, which helps:
  - Sharpen edges where depth changes abruptly
  - Preserve soft edges where depth is continuous (hair, motion blur)
  - Separate overlapping objects based on depth ordering
  - Handle semi-transparent objects by depth context

This runs as a SEPARATE PASS after SAM2 segmentation because:
  1. Can't fit both models in VRAM simultaneously
  2. Allows iterative refinement with different settings
  3. Depth can be cached and reused for multiple objects

WORKFLOW:
    1. Run auto_roto.py to get initial masks
    2. Run depth_refine.py to enhance edges using depth

USAGE:
    # Basic refinement
    python depth_refine.py --alpha ./output/alpha/ --frames ./frames/ --output ./refined/
    
    # With pre-computed depth maps
    python depth_refine.py --alpha ./output/alpha/ --depth ./depth_maps/ --output ./refined/
    
    # Compute depth from video, then refine
    python depth_refine.py --alpha ./output/alpha/ --video input.mp4 --output ./refined/
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Generator, Union
from dataclasses import dataclass
import time

import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class DepthStatistics:
    """Statistics computed from depth analysis for foreground/background separation."""

    # Foreground depth stats (close to camera = LOW values)
    fg_depth_median: float
    fg_depth_std: float
    fg_depth_min: float
    fg_depth_max: float

    # Background depth stats (far from camera = HIGH values)
    bg_depth_median: float
    bg_depth_min: float
    bg_depth_max: float

    # Computed thresholds
    foreground_threshold: float      # Depth threshold for general foreground
    hair_foreground_threshold: float  # More permissive threshold for hair

    # Distance transform (for core detection)
    dist_inside: np.ndarray


@dataclass
class HairProcessingContext:
    """Context object for hair matte computation, grouping related parameters."""

    # Input images
    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray

    # Distance transforms
    dist_inside: np.ndarray
    dist_outside: np.ndarray

    # Masks
    outside_mask: np.ndarray
    is_foreground_depth: np.ndarray

    # Statistics
    depth_stats: 'DepthStatistics'


@dataclass
class DepthRefineConfig:
    """Configuration for depth-guided refinement."""

    # Input paths
    alpha_dir: str = ""           # Directory with alpha mattes
    frames_dir: str = ""          # Directory with RGB frames (optional)
    depth_dir: str = ""           # Pre-computed depth maps (optional)
    video_path: str = ""          # Video file (optional, for depth computation)

    # Output
    output_dir: str = "./refined"

    # Depth model settings
    depth_model: str = "large"    # large (DA3Mono) best for hair detail; nested smooths fine detail
    compute_depth: bool = True    # Compute depth if not provided
    save_depth: bool = True       # Save depth maps for reuse
    depth_only: bool = True       # ONLY output depth - no alpha refinement (pure depth mode)

    # DA3 Resolution Settings (NEW)
    # These control how DA3 processes images for fine detail capture
    depth_process_res: Optional[int] = None   # None = auto (image size), or explicit value like 1024, 2048
    depth_process_method: str = "upper"       # "upper" or "lower" bound resize
    depth_norm_percentiles: Tuple[float, float] = (2.0, 98.0)  # Percentiles for normalization
    use_depth_confidence: bool = False        # Use DA3 confidence maps for filtering

    # Refinement settings
    edge_threshold: float = 0.1   # Depth gradient threshold for edges
    blend_strength: float = 0.7   # How much to blend depth-based refinement
    preserve_softness: bool = True # Preserve existing soft edges

    # Edge handling
    sharpen_hard_edges: bool = True   # Sharpen where depth changes abruptly
    soften_gradual_edges: bool = True # Soften where depth changes gradually

    # Hair detection settings (NEW)
    hair_search_radius: int = 50      # How far outside mask to search for hair
    depth_tolerance: float = 0.15     # Depth similarity tolerance for hair
    min_hair_alpha: float = 0.3       # Minimum alpha value for detected hair

    # Output settings
    output_format: str = "exr"
    bit_depth: int = 16

    # Performance
    device: str = "cuda"
    batch_size: int = 4

    # Debug
    verbose: bool = False
    save_debug: bool = False      # Save intermediate visualizations


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Tiled depth estimation
DEPTH_DEFAULT_TILE_SIZE = 2048       # Default tile size for high-res depth (was 1024)
DEPTH_MIN_TILE_SIZE = 1024           # Minimum tile size for quality
DEPTH_MAX_PROCESS_RES = 8192         # Maximum processing resolution (8K cap)

# Distance transform thresholds
DEEP_CORE_MIN_DISTANCE = 30          # Minimum distance for deep core pixels
DEEP_CORE_MIN_PIXELS = 100           # Minimum pixels to use core statistics

# Bit depth conversion
BIT_DEPTH_16_TO_8_DIVISOR = 256      # Divide 16-bit values by 256 for 8-bit


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
    return logging.getLogger("DepthRefine")


# ==============================================================================
# DEPTH ANYTHING V3 WRAPPER
# ==============================================================================

class DepthEstimator:
    """
    Wrapper for Depth Anything 3 monocular depth estimation.
    
    Outputs relative depth maps (not metric) which is perfect for
    edge detection and refinement.
    """

    MODEL_NAME_BY_SIZE = {
        "small": "depth-anything/DA3-Small",
        "base": "depth-anything/DA3-Base",
        "large": "depth-anything/DA3Mono-Large",
        # Nested models - designed for higher resolution, use larger tiles
        "nested-large": "depth-anything/DA3NESTED-GIANT-LARGE",
        "nested-base": "depth-anything/DA3NESTED-Base",
    }

    # Resolution settings
    MAX_PROCESS_RES = 8192  # 8K cap to prevent extreme memory usage
    DEFAULT_PROCESS_RES = None  # None = use image size (capped at MAX_PROCESS_RES)

    def __init__(
        self,
        model_size: str = 'large',
        device: str = 'cuda',
        logger: logging.Logger = None,
        # New DA3 sensitivity parameters
        process_res: Optional[int] = None,          # Explicit resolution or None for auto
        process_method: str = "upper",              # "upper" or "lower" bound resize
        norm_percentiles: Tuple[float, float] = (2.0, 98.0),  # Percentiles for normalization
        use_confidence: bool = False                 # Use DA3 confidence maps
    ):
        self.model_size = model_size
        self.device = device
        self.logger = logger or logging.getLogger("DepthEstimator")

        # DA3 sensitivity settings
        self.process_res = process_res
        self.process_method = process_method  # "upper" or "lower"
        self.norm_percentiles = norm_percentiles
        self.use_confidence = use_confidence

        self.model = None
        self._load_model()

    
    def _load_model(self):
        """Load Depth Anything 3 model."""
        import sys

        model_name = self.MODEL_NAME_BY_SIZE.get(self.model_size, self.MODEL_NAME_BY_SIZE["large"])
        self.logger.info(f"Loading Depth Anything 3 ({model_name})...")

        # Add Depth-Anything-3 repo to path if it exists locally
        script_dir = Path(__file__).parent
        depth_repo = script_dir / "Depth-Anything-3"
        if depth_repo.exists() and str(depth_repo) not in sys.path:
            sys.path.insert(0, str(depth_repo))

        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

        self.logger.info("Depth Anything 3 loaded successfully")
    
    def estimate(self, image: np.ndarray, process_res: int = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate depth from RGB image.

        Args:
            image: RGB image (H, W, 3), uint8 or float
            process_res: Processing resolution override (default: use instance setting or image size)

        Returns:
            Depth map (H, W), float32, normalized 0-1 (closer = LOWER values)
            This follows DA3's native depth convention where:
            - Foreground (close to camera) = LOW values (e.g., 0.1-0.3)
            - Background (far from camera) = HIGH values (e.g., 0.6-0.9)

            If use_confidence is True, returns tuple: (depth, confidence)
        """
        import torch
        import cv2

        # Ensure proper format
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # Determine process_res: argument > instance setting > auto (image size)
        if process_res is None:
            if self.process_res is not None:
                process_res = min(self.process_res, self.MAX_PROCESS_RES)
            else:
                process_res = min(max(h, w), self.MAX_PROCESS_RES)
        else:
            process_res = min(process_res, self.MAX_PROCESS_RES)

        # Determine process_res_method from instance setting
        # "upper" -> "upper_bound_resize": process_res is max dimension (standard)
        # "lower" -> "lower_bound_resize": process_res is min dimension (higher effective res)
        process_method = f"{self.process_method}_bound_resize"

        self.logger.debug(
            f"Estimating depth at process_res={process_res}, method={process_method} "
            f"(image: {w}x{h}, percentiles={self.norm_percentiles})"
        )

        with torch.no_grad():
            prediction = self.model.inference(
                [image],
                process_res=process_res,
                process_res_method=process_method,
            )
        depth = prediction.depth[0]

        # Extract confidence map if available and requested
        confidence = None
        if self.use_confidence and hasattr(prediction, 'conf') and prediction.conf is not None:
            confidence = prediction.conf[0]
            if confidence.shape != (h, w):
                confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
            confidence = confidence.astype(np.float32)
            self.logger.debug(f"Confidence map extracted: range [{confidence.min():.3f}, {confidence.max():.3f}]")

        # Resize depth to match input image size
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to 0-1 using configurable percentiles
        # Wider percentiles (e.g., 0.5, 99.5) preserve more depth variation for fine details
        # Narrower percentiles (e.g., 2, 98) are more robust to outliers
        p_low, p_high = np.percentile(depth, list(self.norm_percentiles))
        depth = np.clip((depth - p_low) / (p_high - p_low + 1e-8), 0, 1)

        # Optional: Apply confidence weighting to reduce noise in uncertain regions
        if confidence is not None and self.use_confidence:
            # Weight depth by confidence (high confidence = keep depth, low = smooth toward mean)
            # This helps reduce noise at fine edges like hair strands
            depth_mean = np.mean(depth)
            confidence_weight = np.clip(confidence, 0.3, 1.0)  # Floor at 0.3 to avoid zeroing out
            depth = depth * confidence_weight + depth_mean * (1 - confidence_weight)
            self.logger.debug(f"Applied confidence weighting to depth")

        # Note: DA3 outputs depth (not disparity), so closer = LOWER values
        # This is the standard depth convention - no inversion needed

        if self.use_confidence and confidence is not None:
            return depth.astype(np.float32), confidence
        return depth.astype(np.float32)

    def normalize_for_foreground(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        foreground_range: tuple = (0.0, 0.4)
    ) -> np.ndarray:
        """
        Re-normalize depth to expand foreground detail.

        Standard depth normalization may compress foreground variation.
        This method expands the foreground region (where mask > 0.5) to use
        more of the 0-1 range, preserving fine detail like hair.

        Note: Depth convention is closer = LOWER values (foreground is low).

        Args:
            depth: Depth map (H, W), normalized 0-1 (closer = lower)
            mask: Alpha mask (H, W) where foreground > 0.5
            foreground_range: Target range for foreground depths (default 0.0-0.4)

        Returns:
            Re-normalized depth map with expanded foreground detail
        """
        import cv2

        # Get foreground pixels
        fg_mask = mask > 0.5
        if not np.any(fg_mask):
            return depth

        fg_depths = depth[fg_mask]

        # Find foreground depth range (use percentiles for robustness)
        # Foreground (close) has LOW depth values
        fg_min = np.percentile(fg_depths, 5)
        fg_max = np.percentile(fg_depths, 95)
        fg_range = fg_max - fg_min

        if fg_range < 0.01:
            # Foreground has no depth variation, can't enhance
            return depth

        # Target range for foreground
        target_min, target_max = foreground_range
        target_range = target_max - target_min

        # Create enhanced depth
        enhanced = depth.copy()

        # Scale foreground depths to target range
        # fg_depths: [fg_min, fg_max] -> [target_min, target_max]
        scale = target_range / fg_range
        enhanced = (depth - fg_min) * scale + target_min

        # Background depths (> fg_max) get compressed into remaining range [target_max, 1.0]
        # With standard depth: background = HIGH values, foreground = LOW values
        bg_mask = depth > fg_max
        if np.any(bg_mask):
            bg_min_orig = fg_max
            bg_max_orig = depth.max()
            bg_range_orig = bg_max_orig - bg_min_orig + 1e-8

            # Map background to [target_max, 1.0]
            bg_scale = (1.0 - target_max) / bg_range_orig
            enhanced[bg_mask] = (depth[bg_mask] - bg_min_orig) * bg_scale + target_max

        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 1)

        self.logger.debug(
            f"Foreground depth enhancement: [{fg_min:.3f}, {fg_max:.3f}] -> "
            f"[{target_min:.3f}, {target_max:.3f}], scale={scale:.2f}x"
        )

        return enhanced.astype(np.float32)
    
    def estimate_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate depth for a batch of images."""
        # For now, process sequentially (batching requires more memory management)
        return [self.estimate(img) for img in images]

    def release(self) -> None:
        """Release model and free GPU memory."""
        if self.model is not None:
            try:
                # Move model to CPU first to free GPU memory
                self.model = self.model.to('cpu')
            except Exception:
                pass
            del self.model
            self.model = None
            self.logger.debug("DepthEstimator model released")

        # Clear GPU memory
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except ImportError:
            pass

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.release()
        return False

    def estimate_high_res(
        self,
        image: np.ndarray,
        tile_size: int = None,
        overlap: int = None,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Estimate depth at higher resolution.

        For nested models (DA3NESTED), uses direct high-res processing.
        For other models, uses tiled processing with blending.

        Args:
            image: RGB image (H, W, 3), uint8
            tile_size: Size of each tile (default: 2048, or process_res if set)
            overlap: Overlap between tiles for blending (default: tile_size // 4)
            mask: Optional alpha mask - prioritize tiles overlapping mask edges

        Returns:
            High-resolution depth map (H, W), float32, normalized 0-1
        """
        import cv2
        import torch

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # Determine tile size based on process_res setting or default to 2048
        # Larger tiles = better quality but more VRAM
        if tile_size is None:
            if self.process_res is not None and self.process_res >= DEPTH_MIN_TILE_SIZE:
                tile_size = min(self.process_res, DEPTH_DEFAULT_TILE_SIZE)
            else:
                tile_size = DEPTH_DEFAULT_TILE_SIZE

        if overlap is None:
            overlap = tile_size // 2  # 50% overlap for seamless blending (was 25%)

        self.logger.debug(f"High-res depth: tile_size={tile_size}, overlap={overlap}, image={w}x{h}")

        # Tiled processing for memory efficiency on large images
        # If image is smaller than tile size, just use regular estimation
        if h <= tile_size and w <= tile_size:
            return self.estimate(image)

        # Create output arrays
        depth_sum = np.zeros((h, w), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)

        # Create weight mask for blending (smooth cosine feathered edges)
        def create_weight_mask(th, tw):
            """Create a weight mask with smooth cosine-feathered edges for seamless blending."""
            # Use full overlap as feather zone for smoother transitions
            feather = min(overlap, th // 2, tw // 2)
            if feather <= 0:
                return np.ones((th, tw), dtype=np.float32)
            
            # Create 1D cosine ramps (smoother than linear)
            def cosine_ramp(length, feather_size):
                ramp = np.ones(length, dtype=np.float32)
                if feather_size > 0 and length > 2 * feather_size:
                    # Smooth cosine fade at edges: 0 -> 1 over feather zone
                    t = np.linspace(0, np.pi / 2, feather_size)
                    fade_in = np.sin(t) ** 2  # Smooth S-curve
                    ramp[:feather_size] = fade_in
                    ramp[-feather_size:] = fade_in[::-1]
                return ramp
            
            # Create 2D weight mask from outer product of 1D ramps
            ramp_h = cosine_ramp(th, feather)
            ramp_w = cosine_ramp(tw, feather)
            mask_w = np.outer(ramp_h, ramp_w).astype(np.float32)
            
            return mask_w

        # Calculate tile grid
        step = tile_size - overlap
        tiles_processed = 0

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Tile boundaries
                y1, y2 = y, min(y + tile_size, h)
                x1, x2 = x, min(x + tile_size, w)

                # Extract tile
                tile = image[y1:y2, x1:x2]
                tile_h, tile_w = tile.shape[:2]

                # Pad tile to tile_size if at edge
                if tile_h < tile_size or tile_w < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:tile_h, :tile_w] = tile
                    # Mirror padding for better edge handling
                    # Use available rows/columns, not the missing amount
                    if tile_h < tile_size:
                        # Mirror available rows to fill bottom
                        mirror_h = min(tile_h, tile_size - tile_h)
                        padded[tile_h:tile_h + mirror_h, :tile_w] = tile[tile_h - mirror_h:tile_h, :][::-1]
                    if tile_w < tile_size:
                        # Mirror available columns to fill right
                        mirror_w = min(tile_w, tile_size - tile_w)
                        padded[:tile_h, tile_w:tile_w + mirror_w] = tile[:, tile_w - mirror_w:tile_w][:, ::-1]
                    tile = padded

                # Estimate depth for this tile
                # Handle tuple return when use_confidence is enabled
                result = self.estimate(tile)
                tile_depth = result[0] if isinstance(result, tuple) else result

                # DA3 may return different dimensions - resize to match tile
                if tile_depth.shape != (tile_size, tile_size):
                    tile_depth = cv2.resize(tile_depth, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

                # Crop back to original tile size
                tile_depth = tile_depth[:tile_h, :tile_w]

                # Create weight mask
                weight = create_weight_mask(tile_h, tile_w)

                # Accumulate
                depth_sum[y1:y2, x1:x2] += tile_depth * weight
                weight_sum[y1:y2, x1:x2] += weight

                tiles_processed += 1

        # Handle any zero weights (shouldn't happen but safety check)
        weight_sum = np.maximum(weight_sum, 1e-8)

        # Normalize
        depth = (depth_sum / weight_sum).astype(np.float32)

        # Normalize to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Post-process: Edge-preserving smoothing to reduce noise in uniform regions
        # Bilateral filter preserves depth edges while smoothing noise
        # Use float32 directly for full precision (sigmaColor in 0-1 range)
        depth = cv2.bilateralFilter(depth, d=9, sigmaColor=0.1, sigmaSpace=25)

        self.logger.info(f"High-res depth: processed {tiles_processed} tiles at {tile_size}x{tile_size}")

        return depth


# ==============================================================================
# DEPTH-GUIDED REFINEMENT
# ==============================================================================

class DepthGuidedRefiner:
    """
    Refine alpha mattes using depth information.

    NEW APPROACH: Use depth to DETECT hair strands outside the SAM2 mask.

    The key insight is that hair strands will have FOREGROUND depth (same as head)
    even though they're outside the hard SAM2 boundary. We detect them by finding
    foreground-depth pixels in the edge region around the SAM2 mask.

    Algorithm:
    1. Keep SAM2 alpha as the core shape (interior stays untouched)
    2. Get foreground depth reference from inside the SAM2 mask
    3. Create search region: dilated mask minus original mask (the edge band)
    4. In search region, find pixels with foreground-like depth
    5. Those pixels are hair strands - add them to the alpha with soft blending
    """

    def __init__(
        self,
        edge_threshold: float = 0.1,
        blend_strength: float = 0.5,
        preserve_softness: bool = True,
        sharpen_hard_edges: bool = True,
        soften_gradual_edges: bool = True,
        # New hair detection params
        hair_search_radius: int = 50,  # How far outside mask to search for hair
        depth_tolerance: float = 0.15,  # How close to foreground depth to consider hair
        min_hair_alpha: float = 0.3,    # Minimum alpha for detected hair
        logger: logging.Logger = None
    ):
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
        """
        Detect the hair region (top portion of mask where head is).

        Args:
            alpha: Alpha mask
            extended: If True, use larger region (60% height) with more dilation

        Returns a mask focused on where hair is likely to be.
        """
        import cv2

        mask_binary = (alpha > 0.5).astype(np.uint8)

        # Find contours to locate the main subject
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros_like(alpha)

        # Get bounding box of largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Hair region - use more of the bounding box if extended
        hair_region = np.zeros_like(alpha)
        hair_percent = 0.6 if extended else 0.4
        hair_height = int(h * hair_percent)

        # Also extend above the bounding box (hair may extend upward)
        y_start = max(0, y - int(h * 0.1))
        hair_region[y_start:y+hair_height, x:x+w] = 1.0

        # Dilate to include area around hair - larger if extended
        dilate_size = self.hair_search_radius * (3 if extended else 2)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        hair_region = cv2.dilate(hair_region.astype(np.uint8), dilate_kernel).astype(np.float32)

        return hair_region

    def detect_hair_texture(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Detect hair-like texture using high-frequency analysis.

        Hair has characteristic fine texture patterns that differ from
        smooth surfaces. We use Laplacian to detect this.

        Args:
            rgb: RGB image (H, W, 3), uint8 or float
            alpha: Current alpha mask

        Returns:
            Hair texture confidence map (H, W), float32 0-1
        """
        import cv2

        # Convert to grayscale
        if rgb.max() > 1:
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Compute Laplacian (high-frequency detector)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        texture_energy = np.abs(laplacian)

        # Get texture energy stats from inside mask (foreground reference)
        mask_binary = alpha > 0.5
        if np.sum(mask_binary) > 100:
            fg_texture_mean = np.mean(texture_energy[mask_binary])
            fg_texture_std = np.std(texture_energy[mask_binary])
        else:
            fg_texture_mean = 0.05
            fg_texture_std = 0.02

        # Hair has higher texture energy than smooth surfaces
        # Pixels with texture similar to foreground are candidates
        texture_threshold = max(fg_texture_mean - fg_texture_std, 0.01)
        hair_texture = (texture_energy > texture_threshold).astype(np.float32)

        # Weight by texture energy (more texture = more likely hair)
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
        """
        Detect hair by color similarity to foreground.

        Sample hair color from inside the mask (top region), then find
        similar colors outside the mask that also have foreground depth.

        Args:
            rgb: RGB image
            alpha: Current alpha mask
            depth: Float depth map (0-1)
            fg_depth_min: Minimum foreground depth (furthest fg)
            fg_depth_max: Maximum foreground depth (closest fg, likely hair)
            foreground_threshold: Depth threshold separating fg from bg

        Returns:
            Hair color confidence map (H, W), float32 0-1
        """
        import cv2

        if rgb.max() > 1:
            rgb_float = rgb.astype(np.float32) / 255.0
        else:
            rgb_float = rgb.astype(np.float32)

        # Get hair region (top of mask)
        hair_sample_region = self.detect_hair_region(alpha)
        hair_sample_region = hair_sample_region * (alpha > 0.5).astype(np.float32)

        if np.sum(hair_sample_region > 0.5) < 100:
            return np.zeros_like(alpha)

        # Sample hair color from the hair region
        hair_pixels = rgb_float[hair_sample_region > 0.5]
        hair_color_mean = np.mean(hair_pixels, axis=0)
        hair_color_std = np.std(hair_pixels, axis=0) + 0.05

        # Compute color distance from hair color
        color_diff = np.sqrt(np.sum((rgb_float - hair_color_mean) ** 2, axis=2))
        color_tolerance = np.sqrt(np.sum(hair_color_std ** 2))

        # Color similarity (Gaussian falloff)
        color_similarity = np.exp(-(color_diff ** 2) / (2 * (color_tolerance ** 2)))

        # RANGE-BASED depth gating - hair can be CLOSER than body
        # Accept any depth in foreground range or closer (close = LOW depth)
        depth_ok = depth < foreground_threshold

        # Extra boost for pixels at hair-like depth (very close to camera)
        # Hair is typically the closest thing (LOWEST depth value)
        hair_depth_boost = np.where(
            depth < fg_depth_min + 0.05,  # very close to camera (low depth)
            1.5,  # boost
            1.0
        )

        # Combine: similar color AND foreground depth
        hair_color_confidence = color_similarity * depth_ok.astype(np.float32) * hair_depth_boost

        return np.clip(hair_color_confidence, 0, 1).astype(np.float32)

    def compute_depth_edges(self, depth: np.ndarray) -> np.ndarray:
        """
        Compute depth edge map (magnitude of depth gradient).

        Uses FLOAT depth values directly for precision.
        Higher values = stronger depth discontinuity = likely object edge
        """
        import cv2

        # Ensure we're working with float depth (not colorized)
        depth_float = depth.astype(np.float32)

        # Compute gradients on float depth
        grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)

        # Magnitude
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)

        return edge_magnitude
    
    def compute_alpha_edges(self, alpha: np.ndarray) -> np.ndarray:
        """Compute alpha edge map."""
        import cv2
        
        grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
        
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)
        
        return edge_magnitude
    
    def _create_core_matte(
        self,
        alpha: np.ndarray,
        core_erode_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the core matte (inner key) via erosion.

        Returns:
            Tuple of (core_matte, mask_binary)
        """
        import cv2

        mask_binary = (alpha > 0.5).astype(np.uint8)

        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (core_erode_size * 2 + 1, core_erode_size * 2 + 1)
        )
        core_matte = cv2.erode(mask_binary, erode_kernel, iterations=1).astype(np.float32)

        self.logger.debug(f"Core matte: {np.sum(core_matte > 0.5)} solid pixels")

        return core_matte, mask_binary

    def _analyze_depth_statistics(
        self,
        depth: np.ndarray,
        alpha: np.ndarray,
        mask_binary: np.ndarray
    ) -> DepthStatistics:
        """
        Analyze depth statistics for foreground and background regions.

        Note: Depth convention is closer = LOWER values.
        - Foreground (close to camera) has LOW depth values
        - Background (far from camera) has HIGH depth values

        Returns:
            DepthStatistics dataclass with depth statistics and thresholds.
        """
        import cv2

        # Distance transform for finding deep core
        dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        deep_core = dist_inside > DEEP_CORE_MIN_DISTANCE

        # Foreground depth statistics (close = LOW values)
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

        # Background depth statistics (far = HIGH values)
        bg_region = dist_inside == 0
        if np.sum(bg_region) > DEEP_CORE_MIN_PIXELS:
            bg_depth_median = float(np.median(depth[bg_region]))
            bg_depth_min = float(np.percentile(depth[bg_region], 5))
            bg_depth_max = float(np.percentile(depth[bg_region], 95))
        else:
            # Defaults for when no background region detected
            # Background should have HIGH values (far from camera)
            bg_depth_median = 0.7
            bg_depth_min = 0.6
            bg_depth_max = 0.8

        # Compute thresholds for foreground detection
        # Foreground = depth < threshold (close objects have LOW depth)
        # Threshold is just above the maximum foreground depth
        foreground_threshold = fg_depth_max + self.depth_tolerance

        # Hair threshold is slightly more permissive (allows slightly further depths)
        hair_foreground_threshold = fg_depth_max + self.depth_tolerance * 1.5

        self.logger.debug(f"Foreground depth (float): median={fg_depth_median:.4f}, "
                         f"range=[{fg_depth_min:.4f}, {fg_depth_max:.4f}]")
        self.logger.debug(f"Background depth (float): median={bg_depth_median:.4f}, "
                         f"range=[{bg_depth_min:.4f}, {bg_depth_max:.4f}]")
        self.logger.debug(f"Foreground threshold: {foreground_threshold:.4f}")
        self.logger.debug(f"Hair foreground threshold: {hair_foreground_threshold:.4f}")

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
    ) -> np.ndarray:
        """
        Compute depth-gated edge matte.

        Uses range-based depth matching for precise foreground/background separation.
        """
        import cv2

        dist_inside = depth_stats.dist_inside
        foreground_threshold = depth_stats.foreground_threshold
        fg_depth_max = depth_stats.fg_depth_max
        fg_depth_min = depth_stats.fg_depth_min

        # Distance transforms
        dist_outside = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)

        # Spatial edge region
        inner_edge_dist = 5
        outer_edge_dist = float(self.hair_search_radius)
        spatial_edge_region = (dist_inside <= inner_edge_dist) | (dist_outside <= outer_edge_dist)

        # Depth-based foreground detection (close = LOW depth)
        is_foreground_depth = depth < foreground_threshold

        # Compute confidence: higher confidence for depths closer to fg_depth_min (closest)
        depth_margin = foreground_threshold - depth  # How much BELOW threshold
        depth_confidence = np.clip(depth_margin / (foreground_threshold - fg_depth_min + 0.01), 0, 1)

        # Boost for pixels in foreground range
        in_fg_range = (depth >= fg_depth_min - 0.05) & (depth <= fg_depth_max + 0.1)
        depth_confidence = np.where(in_fg_range, np.maximum(depth_confidence, 0.8), depth_confidence)

        # Edge matte with strict depth gating
        edge_matte = np.zeros_like(alpha)
        valid_edge = spatial_edge_region & is_foreground_depth
        edge_matte[valid_edge] = depth_confidence[valid_edge]

        # Distance-based falloff for outside pixels
        outside_mask = dist_outside > 0
        if np.any(outside_mask & valid_edge):
            distance_falloff = 1.0 - (dist_outside / outer_edge_dist)
            distance_falloff = np.clip(distance_falloff, 0, 1)
            edge_matte[outside_mask] *= distance_falloff[outside_mask]

        edge_matte = edge_matte * self.blend_strength * 0.7

        self.logger.debug(f"Edge matte (depth-gated): {np.sum(edge_matte > 0.1)} pixels")

        return edge_matte, dist_outside, is_foreground_depth

    def _apply_rope_filter(
        self,
        depth: np.ndarray,
        hair_depth_threshold: float,
        outside_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply morphological filter to remove thin linear structures (ropes).

        Ropes are thin linear structures that should not be included in hair detection.
        Note: Foreground = depth < threshold (close = LOW depth values)
        """
        import cv2

        fg_outside = (depth < hair_depth_threshold) & outside_mask
        fg_outside_uint8 = fg_outside.astype(np.uint8) * 255

        # Opening removes thin structures
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_opened = cv2.morphologyEx(fg_outside_uint8, cv2.MORPH_OPEN, open_kernel)

        # Closing fills small gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_cleaned = cv2.morphologyEx(fg_opened, cv2.MORPH_CLOSE, close_kernel)

        # Dilate to recover edge detail
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_cleaned = cv2.dilate(fg_cleaned, dilate_kernel, iterations=1)

        rope_filter = (fg_cleaned > 0)
        self.logger.debug(f"Rope filter: {np.sum(fg_outside)} -> {np.sum(rope_filter)} pixels")

        return rope_filter

    def _compute_depth_as_alpha(
        self,
        depth: np.ndarray,
        valid_pixels: np.ndarray,
        hair_local_min: float,
        hair_local_max: float
    ) -> np.ndarray:
        """
        Convert depth values to alpha values with texture preservation.

        Note: Depth convention is closer = LOWER values.
        Closer objects (low depth) should have HIGHER alpha (more opaque).
        """
        import cv2

        depth_range = max(hair_local_max - hair_local_min, 0.1)

        # Normalize depth to 0-1 within hair range
        # INVERT: closer (low depth) = higher alpha
        depth_normalized = np.clip((hair_local_max - depth) / depth_range, 0, 1)

        # Contrast curve to enhance hair strand visibility
        depth_contrasted = depth_normalized ** 0.8

        # Apply validity mask
        depth_as_alpha = depth_contrasted * valid_pixels.astype(np.float32)

        # Texture preservation: high-pass filter
        blur_size = 15
        depth_blurred = cv2.GaussianBlur(depth_normalized, (blur_size, blur_size), 0)
        depth_detail = depth_normalized - depth_blurred
        detail_boost = 2.0
        depth_as_alpha = depth_as_alpha + (depth_detail * detail_boost * valid_pixels.astype(np.float32))
        depth_as_alpha = np.clip(depth_as_alpha, 0, 1)

        return depth_as_alpha

    def _combine_mattes(
        self,
        core_matte: np.ndarray,
        edge_matte: np.ndarray,
        hair_matte: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """
        Combine all mattes using MAX operation.
        """
        import cv2

        combined = core_matte.copy()
        combined = np.maximum(combined, edge_matte)
        combined = np.maximum(combined, hair_matte)
        combined = np.maximum(combined, alpha * 0.9)

        self.logger.debug(f"Combined: {np.sum(combined > 0.5)} pixels > 0.5")

        # Ensure solid core and clean edges
        combined = np.where(core_matte > 0.5, 1.0, combined)
        combined = cv2.GaussianBlur(combined, (3, 3), 0.5)
        combined = np.where(core_matte > 0.5, 1.0, combined)

        return np.clip(combined, 0, 1).astype(np.float32)

    def _compute_hair_matte(
        self,
        ctx: HairProcessingContext
    ) -> np.ndarray:
        """
        Compute hair-specific matte using texture, color, and depth analysis.

        Args:
            ctx: HairProcessingContext containing all input images, masks, and statistics.

        This is the most complex stage combining multiple methods:
        - Method 1: Texture + Color evidence
        - Method 2: Direct depth-based hair detection
        - Method 3: Edge-focused hair refinement with rope filtering
        """
        import cv2

        # Unpack context for local variable access
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

        # Get hair regions
        hair_region = self.detect_hair_region(alpha, extended=True)
        hair_region_normal = self.detect_hair_region(alpha, extended=False)
        hair_region_strict = self.detect_hair_region(alpha, extended=False)

        # Texture and color-based detection
        hair_texture = self.detect_hair_texture(rgb, alpha)
        hair_color = self.detect_hair_color(
            rgb, alpha, depth,
            fg_depth_min, fg_depth_max, foreground_threshold
        )

        # METHOD 1: Texture + Color evidence
        hair_evidence_tc = hair_texture * hair_color * is_foreground_depth.astype(np.float32)
        hair_evidence_tc = hair_evidence_tc * hair_region_normal * outside_mask.astype(np.float32)

        # METHOD 2: Direct depth-based hair (close = LOW depth)
        self.logger.debug(f"Hair depth threshold: {hair_depth_threshold:.4f}")
        depth_range = hair_depth_threshold - fg_depth_min
        depth_hair_evidence = np.where(
            depth < hair_depth_threshold,  # Foreground is BELOW threshold
            np.clip((hair_depth_threshold - depth) / (depth_range + 0.01), 0, 1),
            0
        )
        depth_hair_evidence = depth_hair_evidence * hair_region * outside_mask.astype(np.float32)
        self.logger.debug(f"Depth-based hair pixels (raw): {np.sum(depth_hair_evidence > 0.1)}")

        # METHOD 3: Edge-focused hair refinement
        max_hair_distance = self.hair_search_radius
        max_edge_distance = 15
        hair_edge_band = (dist_outside > 0) & (dist_outside < max_hair_distance)

        # Get edge depth statistics
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

        self.logger.debug(f"Edge depth: mean={edge_depth_mean:.4f}, std={edge_depth_std:.4f}")

        # Apply rope filter
        rope_filter = self._apply_rope_filter(depth, hair_depth_threshold, outside_mask)

        # Compute valid pixels (close = LOW depth)
        valid_hair = hair_edge_band & (depth < hair_depth_threshold) & depth_similar_to_edge & rope_filter
        valid_hair_region = valid_hair & (hair_region_strict > 0.5)

        edge_band_tiny = (dist_outside > 0) & (dist_outside < max_edge_distance)
        valid_edge = edge_band_tiny & (depth < foreground_threshold)
        valid_pixels = valid_hair_region | (valid_edge & (hair_region_strict < 0.5))

        self.logger.debug(f"Valid hair pixels: {np.sum(valid_hair_region)}")
        self.logger.debug(f"Valid edge pixels: {np.sum(valid_edge & (hair_region_strict < 0.5))}")

        # Get local depth range for hair
        if np.sum(valid_hair_region) > 100:
            hair_depths = depth[valid_hair_region]
            hair_local_min = np.percentile(hair_depths, 2)
            hair_local_max = np.percentile(hair_depths, 99)
        else:
            hair_local_min = hair_depth_threshold
            hair_local_max = fg_depth_max

        self.logger.debug(f"Hair LOCAL depth range: [{hair_local_min:.4f}, {hair_local_max:.4f}]")

        # Convert depth to alpha
        depth_as_alpha = self._compute_depth_as_alpha(
            depth, valid_pixels, hair_local_min, hair_local_max
        )

        # Apply distance falloff
        hair_falloff = np.clip(1.0 - (dist_outside / max_hair_distance) ** 0.6, 0, 1)
        edge_falloff = np.clip(1.0 - (dist_outside / max_edge_distance), 0, 1)
        falloff = np.where(hair_region_strict > 0.5, hair_falloff, edge_falloff)
        depth_as_alpha = depth_as_alpha * falloff

        self.logger.debug(f"Depth-as-alpha pixels > 0.3: {np.sum(depth_as_alpha > 0.3)}")

        # Combine all methods
        hair_evidence = depth_as_alpha.copy()
        hair_evidence = np.maximum(hair_evidence, hair_evidence_tc * 0.7)
        hair_evidence = np.maximum(hair_evidence, depth_hair_evidence * 0.5)

        hair_matte = hair_evidence * self.blend_strength
        hair_matte = np.where(hair_matte > 0.05, hair_matte, 0)

        self.logger.debug(f"Hair matte (combined): {np.sum(hair_matte > 0.1)} pixels detected")

        return hair_matte

    def refine(
        self,
        alpha: np.ndarray,
        depth: np.ndarray,
        rgb: np.ndarray = None
    ) -> np.ndarray:
        """
        Refine alpha matte using PROFESSIONAL KEYING WORKFLOW with FLOAT DEPTH.

        This follows the industry-standard multi-matte approach:

        Stage 1 - CORE MATTE (Inner Key):
            Eroded SAM2 mask → 100% solid, never touches edges

        Stage 2 - DEPTH-GATED EDGE MATTE:
            Uses FLOAT depth values for precise foreground/background separation
            Only pixels with matching foreground depth are considered

        Stage 3 - HAIR-SPECIFIC DETECTION:
            Texture analysis for fine hair strands
            Color matching for hair color continuity
            Focused on hair region (top of subject)

        Stage 4 - COMBINE (Matte Logic):
            MAX(core, edge, hair) with proper weighting

        Args:
            alpha: Alpha matte (H, W), values 0-1
            depth: FLOAT depth map (H, W), values 0-1 (NOT colorized!)
            rgb: Optional RGB image for hair detection

        Returns:
            Combined alpha matte (H, W), values 0-1
        """
        import cv2

        # Normalize inputs
        alpha = alpha.astype(np.float32)
        depth = depth.astype(np.float32)
        if depth.max() > 1.0:
            self.logger.warning("Depth values > 1.0 detected - normalizing. Ensure float depth is used!")
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Stage 1: Create core matte
        core_matte, mask_binary = self._create_core_matte(alpha)

        # Stage 2: Analyze depth statistics
        depth_stats = self._analyze_depth_statistics(depth, alpha, mask_binary)

        # Stage 3: Compute edge matte
        edge_matte, dist_outside, is_foreground_depth = self._compute_edge_matte(
            alpha, depth, mask_binary, depth_stats
        )
        outside_mask = dist_outside > 0

        # Stage 4: Compute hair matte (if RGB available)
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

        # Stage 5: Combine all mattes
        return self._combine_mattes(core_matte, edge_matte, hair_matte, alpha)
    
    def refine_with_rgb(
        self,
        alpha: np.ndarray,
        depth: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """
        Refine alpha using both depth and RGB information.
        
        RGB helps preserve fine details like hair strands that might
        not be captured by depth alone.
        """
        import cv2
        
        # First pass: depth-based refinement
        refined = self.refine(alpha, depth)
        
        # Second pass: RGB-guided filtering
        if rgb is not None:
            # Convert RGB to grayscale for guiding
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
                pass  # Skip if not available
        
        return refined


# ==============================================================================
# FILE I/O
# ==============================================================================

def _read_exr_alpha(filepath: Path) -> np.ndarray:
    """Read alpha channel from EXR file using OpenEXR module."""
    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(filepath))
        header = exr_file.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Try to read alpha channel, fall back to Y (luminance) or R
        channels = header['channels'].keys()

        if 'A' in channels:
            channel = 'A'
        elif 'Y' in channels:
            channel = 'Y'
        elif 'R' in channels:
            channel = 'R'
        else:
            channel = list(channels)[0]

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        data = exr_file.channel(channel, pt)

        alpha = np.frombuffer(data, dtype=np.float32).reshape((height, width))
        return alpha

    except Exception as e:
        print(f"Error reading EXR {filepath}: {e}")
        return None


def load_alpha_sequence(alpha_dir: str) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """Load alpha images from directory."""
    import cv2

    alpha_path = Path(alpha_dir)

    # Find all image files
    extensions = {'.exr', '.png', '.tif', '.tiff', '.jpg', '.jpeg'}
    files = []
    for ext in extensions:
        files.extend(alpha_path.glob(f"*{ext}"))
        files.extend(alpha_path.glob(f"*{ext.upper()}"))

    files = sorted(set(files))

    for idx, filepath in enumerate(files):
        # Read alpha - use OpenEXR for EXR files
        if filepath.suffix.lower() == '.exr':
            alpha = _read_exr_alpha(filepath)
        else:
            alpha = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

        if alpha is None:
            continue

        # Handle different formats
        if len(alpha.shape) == 3:
            # Use alpha channel if RGBA, otherwise convert to grayscale
            if alpha.shape[2] == 4:
                alpha = alpha[:, :, 3]
            else:
                alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)

        # Normalize to 0-1
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0
        elif alpha.dtype == np.uint16:
            alpha = alpha.astype(np.float32) / 65535.0
        else:
            alpha = alpha.astype(np.float32)

        yield idx, alpha, filepath


def load_depth_map(depth_path: Path) -> np.ndarray:
    """Load a depth map from file."""
    import cv2
    
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        raise IOError(f"Cannot read depth map: {depth_path}")
    
    # Handle different formats
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    
    # Normalize to 0-1
    if depth.dtype == np.uint8:
        depth = depth.astype(np.float32) / 255.0
    elif depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 65535.0
    else:
        depth = depth.astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    return depth


def save_alpha(
    alpha: np.ndarray,
    filepath: Path,
    bit_depth: int = 16
):
    """Save alpha to file."""
    import cv2
    
    ext = filepath.suffix.lower()
    
    if ext == '.exr':
        try:
            import OpenEXR
            import Imath
            
            h, w = alpha.shape
            
            if bit_depth == 32:
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                data = alpha.astype(np.float32)
            else:
                pixel_type = Imath.PixelType(Imath.PixelType.HALF)
                data = alpha.astype(np.float16)
            
            header = OpenEXR.Header(w, h)
            header['channels'] = {'A': Imath.Channel(pixel_type)}
            
            exr = OpenEXR.OutputFile(str(filepath), header)
            exr.writePixels({'A': data.tobytes()})
            exr.close()
            
        except ImportError:
            # Fallback to OpenCV
            cv2.imwrite(str(filepath), alpha.astype(np.float32))
    
    elif ext == '.png':
        if bit_depth == 16:
            alpha_int = (alpha * 65535).astype(np.uint16)
        else:
            alpha_int = (alpha * 255).astype(np.uint8)
        cv2.imwrite(str(filepath), alpha_int)
    
    elif ext in ('.tif', '.tiff'):
        if bit_depth == 16:
            alpha_int = (alpha * 65535).astype(np.uint16)
        else:
            alpha_int = (alpha * 255).astype(np.uint8)
        cv2.imwrite(str(filepath), alpha_int)


def save_depth_float(
    depth: np.ndarray,
    filepath: Path,
):
    """Save depth map as float EXR for proper precision."""
    try:
        import OpenEXR
        import Imath

        h, w = depth.shape
        depth_float = depth.astype(np.float32)

        header = OpenEXR.Header(w, h)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        header['channels'] = {'Y': Imath.Channel(pixel_type)}

        exr = OpenEXR.OutputFile(str(filepath), header)
        exr.writePixels({'Y': depth_float.tobytes()})
        exr.close()

    except ImportError:
        # Fallback to 16-bit PNG (loses some precision but still usable)
        import cv2
        depth_16 = (depth * 65535).astype(np.uint16)
        cv2.imwrite(str(filepath), depth_16)


def save_depth_visualization(
    depth: np.ndarray,
    filepath: Path,
    colormap: bool = True
):
    """Save depth map as colorized visualization (for preview only).
    
    Uses 16-bit output to minimize banding artifacts in gradients.
    """
    import cv2

    if colormap:
        # Use matplotlib's colormap for smooth gradients
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            # Apply inferno colormap with full float precision
            cmap = cm.get_cmap('inferno')
            depth_colored = cmap(depth)  # Returns RGBA float [0,1]
            
            # Convert to BGR 16-bit for minimal banding (65536 levels per channel)
            depth_vis = (depth_colored[:, :, :3] * 65535).astype(np.uint16)
            depth_vis = depth_vis[:, :, ::-1]  # RGB to BGR
            
        except ImportError:
            # Fallback: grayscale 16-bit if matplotlib unavailable
            depth_vis = (depth * 65535).astype(np.uint16)
    else:
        # Grayscale 16-bit (no colormap)
        depth_vis = (depth * 65535).astype(np.uint16)

    cv2.imwrite(str(filepath), depth_vis)


def load_depth_float(filepath: Path) -> np.ndarray:
    """Load depth map as float values (0-1 range)."""
    import cv2

    ext = filepath.suffix.lower()

    if ext == '.exr':
        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(str(filepath))
            header = exr_file.header()

            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Try Y channel first, then R
            channels = header['channels'].keys()
            channel = 'Y' if 'Y' in channels else 'R' if 'R' in channels else list(channels)[0]

            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            data = exr_file.channel(channel, pt)

            depth = np.frombuffer(data, dtype=np.float32).reshape((height, width))
            return depth

        except ImportError:
            pass

    # Fallback: load with OpenCV
    depth = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise IOError(f"Cannot read depth: {filepath}")

    # Handle different formats - convert to float 0-1
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    if depth.dtype == np.uint8:
        depth = depth.astype(np.float32) / 255.0
    elif depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 65535.0
    else:
        depth = depth.astype(np.float32)
        # Normalize if not already 0-1
        if depth.max() > 1.0:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    return depth


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class DepthRefinePipeline:
    """Main pipeline for depth-guided alpha refinement."""
    
    def __init__(self, config: DepthRefineConfig):
        self.config = config
        self.logger = setup_logging(config.verbose)
        
        self._depth_estimator: Optional[DepthEstimator] = None
        self._refiner: Optional[DepthGuidedRefiner] = None
    
    @property
    def depth_estimator(self) -> DepthEstimator:
        if self._depth_estimator is None:
            self._depth_estimator = DepthEstimator(
                model_size=self.config.depth_model,
                device=self.config.device,
                logger=self.logger,
                # New DA3 sensitivity parameters
                process_res=self.config.depth_process_res,
                process_method=self.config.depth_process_method,
                norm_percentiles=self.config.depth_norm_percentiles,
                use_confidence=self.config.use_depth_confidence,
            )
        return self._depth_estimator
    
    @property
    def refiner(self) -> DepthGuidedRefiner:
        if self._refiner is None:
            self._refiner = DepthGuidedRefiner(
                edge_threshold=self.config.edge_threshold,
                blend_strength=self.config.blend_strength,
                preserve_softness=self.config.preserve_softness,
                sharpen_hard_edges=self.config.sharpen_hard_edges,
                soften_gradual_edges=self.config.soften_gradual_edges,
                # Hair detection parameters
                hair_search_radius=self.config.hair_search_radius,
                depth_tolerance=self.config.depth_tolerance,
                min_hair_alpha=self.config.min_hair_alpha,
                logger=self.logger
            )
        return self._refiner
    
    def run(self):
        """Run the depth refinement pipeline."""
        import cv2
        
        self.logger.info("="*60)
        self.logger.info("Depth-Guided Alpha Refinement")
        self.logger.info("="*60)
        
        # Setup output directories
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        refined_dir = output_dir / "alpha"
        refined_dir.mkdir(exist_ok=True)
        
        depth_dir = output_dir / "depth"
        if self.config.save_depth:
            depth_dir.mkdir(exist_ok=True)
        
        debug_dir = output_dir / "debug"
        if self.config.save_debug:
            debug_dir.mkdir(exist_ok=True)
        
        # Determine depth source
        has_precomputed_depth = self.config.depth_dir and Path(self.config.depth_dir).exists()

        # Load frames if needed for depth computation
        frames_path = None
        frame_files = []
        if self.config.compute_depth and not has_precomputed_depth:
            if self.config.frames_dir:
                frames_path = Path(self.config.frames_dir)
            elif self.config.video_path:
                # Extract frames from video
                self.logger.info("Extracting frames from video...")
                frames_path = output_dir / "temp_frames"
                self._extract_frames(self.config.video_path, frames_path)
            else:
                self.logger.error("Need frames_dir or video_path to compute depth")
                return

            # Pre-load sorted frame file list (handles any naming convention)
            if frames_path and frames_path.exists():
                frame_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG']
                for ext in frame_extensions:
                    frame_files.extend(frames_path.glob(f"*{ext}"))
                frame_files = sorted(set(frame_files))
                self.logger.info(f"Found {len(frame_files)} source frames")

        # Process each alpha
        self.logger.info(f"Loading alpha mattes from: {self.config.alpha_dir}")
        
        for idx, alpha, alpha_path in load_alpha_sequence(self.config.alpha_dir):
            self.logger.info(f"Processing frame {idx}: {alpha_path.name}")
            
            # Get or compute depth
            if has_precomputed_depth:
                # Load precomputed depth
                depth_files = list(Path(self.config.depth_dir).glob(f"*{idx:04d}*"))
                if depth_files:
                    depth = load_depth_map(depth_files[0])
                else:
                    self.logger.warning(f"No depth map for frame {idx}, skipping")
                    continue
            else:
                # Compute depth from frame - use sorted frame file list by index
                if idx < len(frame_files):
                    frame_path = frame_files[idx]
                    frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                    if frame is None:
                        self.logger.warning(f"Could not read frame {frame_path}, skipping")
                        continue
                    if frame.dtype == np.uint16:
                        frame = (frame / 256).astype(np.uint8)
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    self.logger.debug(f"  Computing depth from {frame_path.name}...")
                    self.logger.debug(f"  Model: {self.config.depth_model}")
                    # Use high-res depth estimation (pure depth, no alpha influence)
                    depth = self.depth_estimator.estimate_high_res(frame)

                    if self.config.save_depth:
                        # Save float EXR for actual depth data (for reuse/processing)
                        depth_float_path = depth_dir / f"depth.{idx:04d}.exr"
                        save_depth_float(depth, depth_float_path)
                        # Also save colorized visualization for preview
                        depth_vis_path = depth_dir / f"depth_preview.{idx:04d}.png"
                        save_depth_visualization(depth, depth_vis_path)
                else:
                    self.logger.warning(f"No frame for depth computation at index {idx} (have {len(frame_files)} frames), skipping")
                    continue

            # Skip alpha refinement if depth_only mode (pure depth output)
            if not self.config.depth_only:
                # Refine alpha - pass RGB frame for guided filtering
                self.logger.debug(f"  Refining alpha with hair detection...")
                rgb_for_refine = frame if 'frame' in locals() else None
                refined = self.refiner.refine(alpha, depth, rgb=rgb_for_refine)
                
                # Save refined alpha
                output_path = refined_dir / f"refined.{idx:04d}.{self.config.output_format}"
                save_alpha(refined, output_path, self.config.bit_depth)
                
                # Save debug visualization
                if self.config.save_debug:
                    self._save_debug_vis(
                        alpha, depth, refined,
                        debug_dir / f"debug.{idx:04d}.jpg"
                    )
            
            if idx % 10 == 0:
                self.logger.info(f"  Processed {idx} frames...")
        
        self.logger.info("="*60)
        self.logger.info("Depth Refinement Complete!")
        self.logger.info(f"Output: {output_dir}")
        self.logger.info("="*60)
    
    def _extract_frames(self, video_path: str, output_dir: Path):
        """Extract frames from video."""
        import cv2
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = output_dir / f"frame.{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            idx += 1
        
        cap.release()
        self.logger.info(f"Extracted {idx} frames")
    
    def _save_debug_vis(
        self,
        original: np.ndarray,
        depth: np.ndarray,
        refined: np.ndarray,
        output_path: Path
    ):
        """Save debug visualization."""
        import cv2
        
        h, w = original.shape
        
        # Create side-by-side comparison
        vis = np.zeros((h, w * 4, 3), dtype=np.uint8)
        
        # Original alpha
        vis[:, :w, :] = (original * 255).astype(np.uint8)[..., np.newaxis].repeat(3, axis=-1)
        
        # Depth map (colorized)
        depth_color = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis[:, w:w*2, :] = depth_color
        
        # Refined alpha
        vis[:, w*2:w*3, :] = (refined * 255).astype(np.uint8)[..., np.newaxis].repeat(3, axis=-1)
        
        # Difference (green = added, red = removed)
        diff = refined - original
        diff_vis = np.zeros((h, w, 3), dtype=np.uint8)
        diff_vis[:, :, 1] = np.clip(diff * 500, 0, 255).astype(np.uint8)  # Green = added
        diff_vis[:, :, 2] = np.clip(-diff * 500, 0, 255).astype(np.uint8)  # Red = removed
        vis[:, w*3:, :] = diff_vis
        
        cv2.imwrite(str(output_path), vis)

    def release(self) -> None:
        """Release all resources and free GPU memory."""
        try:
            if self._depth_estimator is not None:
                self._depth_estimator.release()
                self._depth_estimator = None
            
            if self._refiner is not None:
                self._refiner = None
            
            # Clear GPU memory
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except Exception:
            pass

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.release()
        return False


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth-Guided Alpha Refinement using Depth Anything V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Refine with frames (computes depth automatically)
  %(prog)s --alpha ./output/alpha/ --frames ./frames/ --output ./refined/

  # Use pre-computed depth maps
  %(prog)s --alpha ./output/alpha/ --depth ./depth_maps/ --output ./refined/

  # Compute depth from video
  %(prog)s --alpha ./output/alpha/ --video input.mp4 --output ./refined/

  # Adjust refinement settings
  %(prog)s --alpha ./output/alpha/ --frames ./frames/ \\
      --edge-threshold 0.15 --blend-strength 0.7 --output ./refined/

  # HIGH RESOLUTION for fine hair detail (NEW)
  %(prog)s --alpha ./alpha --frames ./frames --output ./refined \\
      --depth-res 2048 --depth-method lower --depth-percentiles 1 99

  # With confidence filtering for semi-transparent edges
  %(prog)s --alpha ./alpha --frames ./frames --output ./refined \\
      --use-depth-confidence --depth-percentiles 0.5 99.5
        """
    )
    
    # Input
    parser.add_argument("--alpha", "-a", required=True,
                       help="Directory with alpha mattes from AUTO-ROTO")
    parser.add_argument("--frames", "-f", help="Directory with RGB frames")
    parser.add_argument("--depth", "-d", help="Directory with pre-computed depth maps")
    parser.add_argument("--video", "-v", help="Video file for depth computation")
    
    # Output
    parser.add_argument("--output", "-o", default="./refined",
                       help="Output directory")
    
    # Depth model
    parser.add_argument("--depth-model", default="large",
                       choices=["small", "base", "large", "nested-base", "nested-large"],
                       help="Depth Anything 3 model (large=DA3Mono best for hair detail)")
    parser.add_argument("--no-save-depth", action="store_true",
                       help="Don't save computed depth maps")
    parser.add_argument("--depth-only", action="store_true", default=True,
                       help="Output ONLY depth maps - no alpha refinement (default: True)")
    parser.add_argument("--refine-alpha", action="store_true",
                       help="Enable alpha refinement using depth (disables depth-only mode)")

    # DA3 Resolution/Sensitivity Settings (NEW)
    parser.add_argument("--depth-res", type=int, default=None,
                       help="DA3 processing resolution (default: auto=image size). "
                            "Higher values (1024, 2048) capture finer details like hair strands")
    parser.add_argument("--depth-method", type=str, default="upper",
                       choices=["upper", "lower"],
                       help="DA3 resize method: 'upper' = process_res is max dimension (standard), "
                            "'lower' = process_res is min dimension (higher effective resolution)")
    parser.add_argument("--depth-percentiles", type=float, nargs=2, default=[2.0, 98.0],
                       metavar=("LOW", "HIGH"),
                       help="Normalization percentiles for depth. Wider range (1, 99) preserves "
                            "more depth variation for fine details. Default: 2 98")
    parser.add_argument("--use-depth-confidence", action="store_true",
                       help="Use DA3 confidence maps to filter uncertain depth values "
                            "(helps with semi-transparent edges)")

    # Refinement settings
    parser.add_argument("--edge-threshold", type=float, default=0.1,
                       help="Depth gradient threshold for hard edges (0.05-0.2)")
    parser.add_argument("--blend-strength", type=float, default=0.7,
                       help="Hair detection blend strength (0-1, higher=more hair)")
    parser.add_argument("--no-sharpen", action="store_true",
                       help="Disable edge sharpening")
    parser.add_argument("--no-soften", action="store_true",
                       help="Disable soft edge preservation")

    # Hair detection settings (NEW)
    parser.add_argument("--hair-search-radius", type=int, default=50,
                       help="How far outside mask to search for hair (pixels)")
    parser.add_argument("--depth-tolerance", type=float, default=0.15,
                       help="Depth similarity tolerance for hair detection (0.1-0.3)")
    parser.add_argument("--min-hair-alpha", type=float, default=0.3,
                       help="Minimum alpha value for detected hair (0.2-0.5)")
    
    # Output format
    parser.add_argument("--format", default="exr",
                       choices=["exr", "png", "tiff"])
    parser.add_argument("--bit-depth", type=int, default=16,
                       choices=[8, 16, 32])
    
    # Performance
    parser.add_argument("--device", default="cuda")
    
    # Debug
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true",
                       help="Save debug visualizations")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # depth_only is True by default; --refine-alpha disables it
    depth_only = not args.refine_alpha

    config = DepthRefineConfig(
        alpha_dir=args.alpha,
        frames_dir=args.frames or "",
        depth_dir=args.depth or "",
        video_path=args.video or "",
        output_dir=args.output,
        depth_model=args.depth_model,
        save_depth=not args.no_save_depth,
        depth_only=depth_only,  # Pure depth mode (no alpha refinement)
        # DA3 sensitivity settings (NEW)
        depth_process_res=args.depth_res,
        depth_process_method=args.depth_method,
        depth_norm_percentiles=tuple(args.depth_percentiles),
        use_depth_confidence=args.use_depth_confidence,
        # Refinement settings
        edge_threshold=args.edge_threshold,
        blend_strength=args.blend_strength,
        sharpen_hard_edges=not args.no_sharpen,
        soften_gradual_edges=not args.no_soften,
        # Hair detection settings
        hair_search_radius=args.hair_search_radius,
        depth_tolerance=args.depth_tolerance,
        min_hair_alpha=args.min_hair_alpha,
        # Output
        output_format=args.format,
        bit_depth=args.bit_depth,
        device=args.device,
        verbose=args.verbose,
        save_debug=args.debug,
    )

    pipeline = DepthRefinePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
