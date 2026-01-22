#!/usr/bin/env python
# Enable OpenCV's OpenEXR codec before importing cv2
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

"""
DEPTH-GUIDED ALPHA REFINEMENT
=============================

Uses Depth Anything V2 to refine alpha mattes from AUTO-ROTO.

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
from typing import Optional, List, Tuple, Dict, Generator
from dataclasses import dataclass
import time

import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

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
    depth_model: str = "large"    # small, base, large
    compute_depth: bool = True    # Compute depth if not provided
    save_depth: bool = True       # Save depth maps for reuse
    
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
# DEPTH ANYTHING V2 WRAPPER
# ==============================================================================

class DepthEstimator:
    """
    Wrapper for Depth Anything V2 monocular depth estimation.
    
    Outputs relative depth maps (not metric) which is perfect for
    edge detection and refinement.
    """
    
    MODEL_CONFIGS = {
        'small': ('depth_anything_v2_vits.pth', 'vits'),
        'base': ('depth_anything_v2_vitb.pth', 'vitb'),
        'large': ('depth_anything_v2_vitl.pth', 'vitl'),
    }
    
    CHECKPOINT_URLS = {
        'small': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
        'base': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
        'large': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
    }
    
    def __init__(
        self,
        model_size: str = 'large',
        device: str = 'cuda',
        logger: logging.Logger = None
    ):
        self.model_size = model_size
        self.device = device
        self.logger = logger or logging.getLogger("DepthEstimator")
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Depth Anything V2 model."""
        import torch
        import sys

        self.logger.info(f"Loading Depth Anything V2 ({self.model_size})...")

        checkpoint_name, encoder = self.MODEL_CONFIGS[self.model_size]
        checkpoint_path = self._ensure_checkpoint(checkpoint_name)

        try:
            # Add Depth-Anything-V2 repo to path if it exists
            script_dir = Path(__file__).parent
            depth_repo = script_dir / "Depth-Anything-V2"
            if depth_repo.exists() and str(depth_repo) not in sys.path:
                sys.path.insert(0, str(depth_repo))

            # Try loading via the official depth_anything_v2 module
            from depth_anything_v2.dpt import DepthAnythingV2
            
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            self.model = DepthAnythingV2(**model_configs[encoder])
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            self.model = self.model.to(self.device).eval()
            
        except ImportError:
            # Fallback: use Hugging Face Transformers
            self.logger.info("Using Hugging Face Transformers for Depth Anything V2")
            
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            model_id = f"depth-anything/Depth-Anything-V2-{self.model_size.capitalize()}"
            
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.model = self.model.to(self.device).eval()
            self._use_transformers = True
        
        self.logger.info("Depth Anything V2 loaded successfully")
    
    def _ensure_checkpoint(self, checkpoint_name: str) -> Path:
        """Ensure checkpoint exists, download if needed."""
        import urllib.request
        
        cache_dir = Path.home() / ".cache" / "depth_anything_v2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = cache_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            url = self.CHECKPOINT_URLS[self.model_size]
            self.logger.info(f"Downloading {checkpoint_name}...")
            urllib.request.urlretrieve(url, checkpoint_path)
            self.logger.info("Download complete")
        
        return checkpoint_path
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Args:
            image: RGB image (H, W, 3), uint8 or float
            
        Returns:
            Depth map (H, W), float32, normalized 0-1 (closer = lower values)
        """
        import torch
        import cv2
        
        # Ensure proper format
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        h, w = image.shape[:2]
        
        if hasattr(self, '_use_transformers') and self._use_transformers:
            # Hugging Face Transformers path
            from PIL import Image
            
            pil_image = Image.fromarray(image)
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth
            
            # Resize to original size
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            
            depth = depth.cpu().numpy()
            
        else:
            # Official DepthAnythingV2 path
            # Resize to model input size (518 is default)
            input_size = 518
            img_resized = cv2.resize(image, (input_size, input_size))
            
            # Normalize
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
            img_tensor = img_tensor / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            img_tensor = (img_tensor - mean) / std
            
            with torch.no_grad():
                depth = self.model(img_tensor)
            
            # Resize back to original size
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            
            depth = depth.cpu().numpy()
        
        # Normalize to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth.astype(np.float32)
    
    def estimate_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate depth for a batch of images."""
        # For now, process sequentially (batching requires more memory management)
        return [self.estimate(img) for img in images]

    def estimate_high_res(
        self,
        image: np.ndarray,
        tile_size: int = 1024,
        overlap: int = 256,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Estimate depth at higher resolution using tiled processing.

        For 4K footage, the standard 518x518 resolution loses fine detail.
        This method processes overlapping tiles and blends them together.

        If a mask is provided, only process tiles that overlap with the mask
        (optimization for rotoscoping where we only care about subject edges).

        Args:
            image: RGB image (H, W, 3), uint8
            tile_size: Size of each tile (default 1024 for good detail)
            overlap: Overlap between tiles for blending (default 256)
            mask: Optional alpha mask - only process tiles overlapping mask edges

        Returns:
            High-resolution depth map (H, W), float32, normalized 0-1
        """
        import cv2
        import torch

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # If image is smaller than tile size, just use regular estimation
        if h <= tile_size and w <= tile_size:
            return self.estimate(image)

        # Create output arrays
        depth_sum = np.zeros((h, w), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)

        # Create weight mask for blending (feathered edges)
        def create_weight_mask(th, tw):
            """Create a weight mask with feathered edges for blending."""
            mask = np.ones((th, tw), dtype=np.float32)
            feather = min(overlap // 2, th // 4, tw // 4)
            if feather > 0:
                for i in range(feather):
                    weight = (i + 1) / feather
                    mask[i, :] *= weight
                    mask[-(i+1), :] *= weight
                    mask[:, i] *= weight
                    mask[:, -(i+1)] *= weight
            return mask

        # If mask provided, find regions of interest (mask edges)
        if mask is not None:
            # Dilate mask to get region around edges
            mask_binary = (mask > 0.1).astype(np.uint8)
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
            mask_dilated = cv2.dilate(mask_binary, dilate_kernel, iterations=1)
        else:
            mask_dilated = np.ones((h, w), dtype=np.uint8)

        # Calculate tile grid
        step = tile_size - overlap
        tiles_processed = 0

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Tile boundaries
                y1, y2 = y, min(y + tile_size, h)
                x1, x2 = x, min(x + tile_size, w)

                # Check if this tile overlaps with mask region
                if mask is not None:
                    tile_mask = mask_dilated[y1:y2, x1:x2]
                    if np.sum(tile_mask) < 100:  # Skip tiles with no mask overlap
                        continue

                # Extract tile
                tile = image[y1:y2, x1:x2]
                tile_h, tile_w = tile.shape[:2]

                # Pad tile to tile_size if at edge
                if tile_h < tile_size or tile_w < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:tile_h, :tile_w] = tile
                    # Mirror padding for better edge handling
                    if tile_h < tile_size:
                        padded[tile_h:, :tile_w] = tile[-(tile_size-tile_h):, :][::-1]
                    if tile_w < tile_size:
                        padded[:tile_h, tile_w:] = tile[:, -(tile_size-tile_w):][:, ::-1]
                    tile = padded

                # Estimate depth for this tile
                tile_depth = self.estimate(tile)

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
    
    def compute_depth_edges(self, depth: np.ndarray) -> np.ndarray:
        """
        Compute depth edge map (magnitude of depth gradient).
        
        Higher values = stronger depth discontinuity = likely object edge
        """
        import cv2
        
        # Compute gradients
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        
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
    
    def refine(
        self,
        alpha: np.ndarray,
        depth: np.ndarray,
        rgb: np.ndarray = None
    ) -> np.ndarray:
        """
        Refine alpha matte using PROFESSIONAL KEYING WORKFLOW.

        This follows the industry-standard multi-matte approach:

        Stage 1 - CORE MATTE (Inner Key):
            Eroded SAM2 mask → 100% solid, never touches edges
            Prevents "crunchy" artifacts at boundaries

        Stage 2 - EDGE MATTE (Detail Key):
            Depth-based detection → captures hair/fine detail
            Interior has holes - that's intentional!
            Preserves every pixel of delicate edges

        Stage 3 - COMBINE (Matte Logic):
            MAX(core, edge) → core provides solid, edge provides detail
            Like Nuke's Keymix or Max operation

        Stage 4 - ADDITIVE (Color Difference Key):
            Luminance-based recovery for missed transparency

        Args:
            alpha: Alpha matte (H, W), values 0-1
            depth: Depth map (H, W), values 0-1
            rgb: Optional RGB image for additive key

        Returns:
            Combined alpha matte (H, W), values 0-1
        """
        import cv2

        alpha = alpha.astype(np.float32)
        depth = depth.astype(np.float32)

        h, w = alpha.shape

        # =====================================================================
        # STAGE 1: CORE MATTE (The Inner Key)
        # =====================================================================
        # Erode the SAM2 mask to create solid interior that never touches edges
        # This prevents all edge artifacts - the core is 100% clean

        mask_binary = (alpha > 0.5).astype(np.uint8)

        # Significant erosion to stay away from problematic edges
        core_erode_size = 15  # pixels to erode inward
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (core_erode_size * 2 + 1, core_erode_size * 2 + 1))
        core_matte = cv2.erode(mask_binary, erode_kernel, iterations=1).astype(np.float32)

        self.logger.debug(f"Core matte: {np.sum(core_matte > 0.5)} solid pixels")

        # =====================================================================
        # STAGE 2: EDGE MATTE (The Detail Key)
        # =====================================================================
        # Create a separate matte focused ONLY on edge detail
        # CRITICAL: Must be DEPTH-GATED to exclude background scene elements
        # Only pixels with FOREGROUND DEPTH should be included

        # Get foreground depth from deep inside the mask
        dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        deep_core = dist_inside > 30  # well inside

        if np.sum(deep_core) > 100:
            fg_depth_median = np.median(depth[deep_core])
            fg_depth_std = np.std(depth[deep_core])
        else:
            fg_depth_median = np.median(depth[alpha > 0.5])
            fg_depth_std = 0.1

        self.logger.debug(f"Foreground depth: {fg_depth_median:.3f} ± {fg_depth_std:.3f}")

        # Distance transforms
        dist_outside = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)

        inner_edge_dist = 10   # pixels inside the SAM2 edge
        outer_edge_dist = float(self.hair_search_radius)  # pixels outside

        # Spatial edge region (near the mask boundary)
        spatial_edge_region = (dist_inside <= inner_edge_dist) | (dist_outside <= outer_edge_dist)

        # Compute depth confidence
        depth_diff = np.abs(depth - fg_depth_median)
        tolerance = max(self.depth_tolerance, fg_depth_std * 2.5)

        # Gaussian falloff for smooth depth confidence
        depth_confidence = np.exp(-(depth_diff ** 2) / (2 * (tolerance ** 2)))

        # STRICT DEPTH GATE: Only pixels with foreground-like depth
        # This is what excludes the background scene!
        depth_threshold = 0.3  # minimum depth confidence to be considered foreground
        is_foreground_depth = depth_confidence > depth_threshold

        # Edge matte = ONLY pixels that are:
        # 1. In spatial edge region (near mask boundary)
        # 2. AND have foreground-like depth (not background)
        edge_matte = np.zeros_like(alpha)
        valid_edge = spatial_edge_region & is_foreground_depth
        edge_matte[valid_edge] = depth_confidence[valid_edge]

        # Apply distance-based falloff for pixels OUTSIDE the original mask
        outside_mask = dist_outside > 0
        if np.any(outside_mask & valid_edge):
            distance_falloff = 1.0 - (dist_outside / outer_edge_dist)
            distance_falloff = np.clip(distance_falloff, 0, 1)
            # Only apply to valid edge pixels
            edge_matte[outside_mask] *= distance_falloff[outside_mask]

        # Scale by blend strength
        edge_matte = edge_matte * self.blend_strength

        self.logger.debug(f"Edge matte: {np.sum(edge_matte > 0.1)} detail pixels (depth-gated)")

        # =====================================================================
        # STAGE 3: COMBINE (Matte Logic - MAX Operation)
        # =====================================================================
        # Like Nuke's Max or Keymix: take the higher value at each pixel
        # Core provides solid interior, Edge provides fine detail

        combined = np.maximum(core_matte, edge_matte)

        # Also incorporate original SAM2 alpha where it's useful
        # (in case SAM2 captured some edge detail we'd lose)
        combined = np.maximum(combined, alpha * 0.8)  # slight reduction to prefer our mattes

        self.logger.debug(f"Combined (after MAX): {np.sum(combined > 0.5)} pixels > 0.5")

        # =====================================================================
        # STAGE 4: ADDITIVE KEY (Color Difference / Luminance Recovery)
        # =====================================================================
        # Recover fine transparency details using luminance differences
        # CRITICAL: Must also be DEPTH-GATED to exclude background!

        if rgb is not None:
            # Convert to grayscale for luminance analysis
            if len(rgb.shape) == 3:
                if rgb.max() > 1:
                    rgb_norm = rgb.astype(np.float32) / 255.0
                else:
                    rgb_norm = rgb.astype(np.float32)

                # Luminance
                luminance = 0.299 * rgb_norm[:,:,0] + 0.587 * rgb_norm[:,:,1] + 0.114 * rgb_norm[:,:,2]

                # Get background luminance (from truly outside region)
                bg_region = dist_outside > outer_edge_dist
                if np.sum(bg_region) > 100:
                    bg_luminance = np.median(luminance[bg_region])
                else:
                    bg_luminance = 0.1

                # Foreground luminance (from core)
                if np.sum(deep_core) > 100:
                    fg_luminance = np.median(luminance[deep_core])
                else:
                    fg_luminance = 0.5

                # Additive key based on luminance difference
                if fg_luminance > bg_luminance:
                    lum_diff = luminance - bg_luminance
                    lum_range = fg_luminance - bg_luminance + 0.01
                    additive_matte = np.clip(lum_diff / lum_range, 0, 1)
                else:
                    lum_diff = bg_luminance - luminance
                    lum_range = bg_luminance - fg_luminance + 0.01
                    additive_matte = np.clip(lum_diff / lum_range, 0, 1)

                # DEPTH-GATE the additive key: only apply where depth is foreground-like
                # This prevents background scene elements from being included!
                additive_matte = additive_matte * is_foreground_depth.astype(np.float32)

                # Also restrict to spatial edge region
                additive_matte = additive_matte * spatial_edge_region.astype(np.float32)

                # Reduced strength
                additive_matte = additive_matte * 0.3

                # Combine with MAX
                combined = np.maximum(combined, additive_matte)

                self.logger.debug(f"After additive key (depth-gated): {np.sum(combined > 0.5)} pixels > 0.5")

        # =====================================================================
        # FINAL: Clean up and ensure solid core
        # =====================================================================
        # Make sure core is absolutely solid (no floating point errors)
        combined = np.where(core_matte > 0.5, 1.0, combined)

        # Clip to valid range
        refined = np.clip(combined, 0, 1).astype(np.float32)

        return refined
    
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


def save_depth_visualization(
    depth: np.ndarray,
    filepath: Path,
    colormap: bool = True
):
    """Save depth map as visualization."""
    import cv2
    
    # Normalize
    depth_vis = (depth * 255).astype(np.uint8)
    
    if colormap:
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    
    cv2.imwrite(str(filepath), depth_vis)


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
                logger=self.logger
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

                    self.logger.debug(f"  Computing high-res depth (tiled) from {frame_path.name}...")
                    # Use high-res tiled depth estimation, optimized with mask
                    depth = self.depth_estimator.estimate_high_res(
                        frame,
                        tile_size=1024,
                        overlap=256,
                        mask=alpha  # Only process tiles near the subject
                    )

                    if self.config.save_depth:
                        depth_path = depth_dir / f"depth.{idx:04d}.png"
                        save_depth_visualization(depth, depth_path)
                else:
                    self.logger.warning(f"No frame for depth computation at index {idx} (have {len(frame_files)} frames), skipping")
                    continue

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


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth-Guided Alpha Refinement using Depth Anything V2",
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
                       choices=["small", "base", "large"],
                       help="Depth Anything V2 model size")
    parser.add_argument("--no-save-depth", action="store_true",
                       help="Don't save computed depth maps")
    
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
    
    config = DepthRefineConfig(
        alpha_dir=args.alpha,
        frames_dir=args.frames or "",
        depth_dir=args.depth or "",
        video_path=args.video or "",
        output_dir=args.output,
        depth_model=args.depth_model,
        save_depth=not args.no_save_depth,
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
