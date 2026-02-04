"""
Geometric Matte Refiner
=======================

Full Geometric-Guided Image Matting pipeline.

Orchestrates:
    SAM Mask + Depth → TrimapSynthesizer → Trimap → ViTMatte → Alpha
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from auto_roto.config.vitmatte import GeometricMatteConfig
from auto_roto.refiners.trimap import TrimapSynthesizer
from auto_roto.models.vitmatte import ViTMatteRefiner
from auto_roto.utils.color import srgb_to_linear

logger = logging.getLogger("AutoRoto.Refiners.Geometric")


class GeometricMatteRefiner:
    """
    Full Geometric-Guided Image Matting pipeline.

    Orchestrates:
        SAM Mask + Depth → TrimapSynthesizer → Trimap → ViTMatte → Alpha
    """

    def __init__(self, config: GeometricMatteConfig = None, logger: logging.Logger = None):
        """
        Initialize geometric matte refiner.

        Args:
            config: GeometricMatteConfig instance (uses defaults if None)
            logger: Optional logger instance
        """
        self.config = config or GeometricMatteConfig()
        self.logger = logger or logging.getLogger("GeometricMatte")

        # Initialize to None for safe cleanup if init fails
        self.trimap_synth = None
        self.vitmatte = None

        self.trimap_synth = TrimapSynthesizer(self.config.trimap, self.logger)
        self.vitmatte = ViTMatteRefiner(
            model_size=self.config.vitmatte.model_size,
            device=self.config.vitmatte.device,
            max_resolution=self.config.vitmatte.max_resolution,
            use_fp16=getattr(self.config.vitmatte, 'use_fp16', True),
            logger=self.logger
        )

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
            sam_mask: SAM alpha mask (H, W), float 0-1
            depth: Depth map (H, W), float 0-1
            save_trimap_path: Optional path to save trimap visualization
            prev_rgb: Previous frame RGB (for motion-aware adaptive mode)

        Returns:
            Final alpha matte (H, W), float32 0-1
        """
        import cv2

        # Prepare frames for motion-aware mode
        prev_frame_gray = None
        curr_frame_gray = None
        if self.config.trimap.motion_aware and prev_rgb is not None:
            curr_frame_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            prev_frame_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)

        # Normalize depth using configured percentiles
        depth = depth.astype(np.float32)
        percentiles = self.config.depth_norm_percentiles
        p_low, p_high = np.percentile(depth, list(percentiles))
        depth = np.clip((depth - p_low) / (p_high - p_low + 1e-8), 0, 1)
        depth = 1.0 - depth

        # Synthesize trimap
        self.logger.debug("Synthesizing trimap...")
        trimap = self.trimap_synth.synthesize(
            sam_mask, depth, rgb,
            prev_frame_gray=prev_frame_gray,
            curr_frame_gray=curr_frame_gray
        )

        # Save trimap if requested
        if save_trimap_path is not None:
            trimap_vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
            trimap_vis[trimap == 0] = [0, 0, 0]
            trimap_vis[trimap == 128] = [128, 128, 128]
            trimap_vis[trimap == 255] = [255, 255, 255]
            cv2.imwrite(str(save_trimap_path), trimap_vis)

        # Run ViTMatte
        self.logger.debug("Running ViTMatte...")
        alpha = self.vitmatte.refine(rgb, trimap)

        # Apply Guided Filter with optional hair polish
        if self.config.trimap.use_linear_colorspace:
            try:
                from cv2 import ximgproc

                rgb_linear = srgb_to_linear(rgb.astype(np.float32) / 255.0)

                alpha_refined = ximgproc.guidedFilter(
                    guide=rgb_linear,
                    src=alpha.astype(np.float32),
                    radius=self.config.guided_filter_radius,
                    eps=self.config.guided_filter_eps
                )

                # Hair density polish
                detail_region = (trimap == 128)

                if np.any(detail_region) and self.config.hair_polish_enabled:
                    hair_alpha = alpha_refined[detail_region]

                    if self.config.hair_gamma != 1.0:
                        hair_alpha = np.power(hair_alpha, self.config.hair_gamma)

                    if self.config.hair_black_point > 0:
                        hair_alpha = np.maximum(0, hair_alpha - self.config.hair_black_point)

                    if self.config.hair_gain != 1.0:
                        hair_alpha = hair_alpha * self.config.hair_gain

                    alpha_refined[detail_region] = hair_alpha

                alpha = np.clip(alpha_refined, 0, 1).astype(np.float32)
                self.logger.debug("Applied Guided Filter refinement")
            except ImportError:
                self.logger.warning("cv2.ximgproc not available - skipping Guided Filter")
            except Exception as e:
                self.logger.warning(f"Guided Filter failed: {e} - using raw alpha")

        # Ensure core foreground is solid
        core_fg = trimap == 255
        alpha[core_fg] = np.maximum(alpha[core_fg], 0.99)

        return alpha

    def release(self):
        """Release GPU memory."""
        if self.vitmatte is not None:
            self.vitmatte.release()

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
