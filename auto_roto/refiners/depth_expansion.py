"""
Depth-Based Mask Expansion
==========================

Expands masks based on depth similarity to capture complete subject coverage.

Cinema Preset Core Algorithm:
    1. Analyze depth values within initial SAM mask
    2. Build depth histogram of foreground pixels
    3. Identify depth range covered by subject (percentile-based)
    4. Expand mask to include nearby pixels within that depth range
    5. Optionally use connected components to avoid distant regions
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from auto_roto.config.depth_expansion import DepthExpansionConfig

logger = logging.getLogger("AutoRoto.Refiners.DepthExpansion")


class DepthBasedExpander:
    """
    Expand mask to include pixels at similar depth values.

    This is the core algorithm for the cinema preset workflow,
    ensuring complete subject coverage by capturing all pixels
    at the same depth plane as the initial SAM mask.
    """

    def __init__(
        self,
        config: DepthExpansionConfig = None,
        logger: logging.Logger = None
    ):
        """
        Initialize depth-based expander.

        Args:
            config: DepthExpansionConfig instance (uses defaults if None)
            logger: Optional logger instance
        """
        self.config = config or DepthExpansionConfig()
        self.logger = logger or logging.getLogger("DepthExpander")

    def compute_depth_profile(
        self,
        mask: np.ndarray,
        depth: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze depth distribution within mask.

        Args:
            mask: Binary/soft mask (H, W), values 0-1
            depth: Depth map (H, W), float 0-1

        Returns:
            Dictionary with depth statistics:
                - min_depth: Minimum depth in foreground
                - max_depth: Maximum depth in foreground
                - mean_depth: Mean depth in foreground
                - std_depth: Standard deviation of depth
                - percentile_low: Lower percentile depth
                - percentile_high: Upper percentile depth
        """
        # Get foreground pixels (mask > 0.5)
        fg_mask = mask > 0.5
        fg_depths = depth[fg_mask]

        if len(fg_depths) == 0:
            self.logger.warning("No foreground pixels found in mask")
            return {
                'min_depth': 0.0,
                'max_depth': 1.0,
                'mean_depth': 0.5,
                'std_depth': 0.0,
                'percentile_low': 0.0,
                'percentile_high': 1.0,
            }

        p_low, p_high = self.config.percentile_range
        percentile_low = np.percentile(fg_depths, p_low)
        percentile_high = np.percentile(fg_depths, p_high)

        profile = {
            'min_depth': float(np.min(fg_depths)),
            'max_depth': float(np.max(fg_depths)),
            'mean_depth': float(np.mean(fg_depths)),
            'std_depth': float(np.std(fg_depths)),
            'percentile_low': float(percentile_low),
            'percentile_high': float(percentile_high),
        }

        self.logger.debug(
            f"Depth profile: mean={profile['mean_depth']:.3f}, "
            f"range=[{profile['percentile_low']:.3f}, {profile['percentile_high']:.3f}]"
        )

        return profile

    def expand_mask(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Expand mask based on depth similarity.

        Args:
            mask: Initial SAM mask (H, W), float 0-1
            depth: Depth map (H, W), float 0-1
            rgb: Optional RGB image for edge-aware expansion (H, W, 3)

        Returns:
            Expanded mask (H, W), float 0-1
        """
        self.logger.info("Starting depth-based mask expansion...")

        # Ensure float32
        mask = mask.astype(np.float32)
        depth = depth.astype(np.float32)

        # Normalize depth if needed
        if depth.max() > 1.0:
            depth = depth / depth.max()

        # Compute depth profile
        profile = self.compute_depth_profile(mask, depth)

        # Define depth range for expansion
        depth_low = profile['percentile_low'] - self.config.depth_tolerance
        depth_high = profile['percentile_high'] + self.config.depth_tolerance
        depth_low = max(0.0, depth_low)
        depth_high = min(1.0, depth_high)

        self.logger.debug(f"Expansion depth range: [{depth_low:.3f}, {depth_high:.3f}]")

        # Create depth-matching mask
        depth_match = (depth >= depth_low) & (depth <= depth_high)
        depth_match = depth_match.astype(np.float32)

        # Apply edge-aware filtering if enabled and RGB provided
        if self.config.edge_aware and rgb is not None:
            depth_match = self._apply_edge_aware_filter(depth_match, rgb)

        # Limit expansion distance from original mask
        if self.config.max_expansion_px > 0:
            depth_match = self._limit_expansion_distance(mask, depth_match)

        # Require connectivity to original mask if enabled
        if self.config.require_connectivity:
            depth_match = self._require_connectivity(mask, depth_match)

        # Combine with original mask (union)
        expanded = np.maximum(mask, depth_match)

        # Morphological cleanup
        expanded = self._morphological_cleanup(expanded)

        # Remove small regions
        expanded = self._remove_small_regions(expanded)

        self.logger.info(
            f"Mask expansion complete: "
            f"{np.sum(mask > 0.5)} -> {np.sum(expanded > 0.5)} pixels "
            f"({np.sum(expanded > 0.5) / max(1, np.sum(mask > 0.5)) * 100:.1f}%)"
        )

        return expanded.astype(np.float32)

    def _apply_edge_aware_filter(
        self,
        depth_match: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """Apply edge-aware filtering to prevent crossing strong edges."""
        # Convert to grayscale
        if rgb.ndim == 3:
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb.astype(np.uint8)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0

        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Reduce depth match at strong edges
        edge_penalty = 1.0 - (edges * self.config.edge_threshold * 10)
        edge_penalty = np.clip(edge_penalty, 0, 1)

        return depth_match * edge_penalty

    def _limit_expansion_distance(
        self,
        original_mask: np.ndarray,
        depth_match: np.ndarray
    ) -> np.ndarray:
        """Limit expansion to max_expansion_px from original mask."""
        # Compute distance transform from original mask
        original_binary = (original_mask > 0.5).astype(np.uint8)
        dist_transform = cv2.distanceTransform(
            1 - original_binary,
            cv2.DIST_L2,
            5
        )

        # Create expansion limit mask
        expansion_limit = dist_transform <= self.config.max_expansion_px

        return depth_match * expansion_limit.astype(np.float32)

    def _require_connectivity(
        self,
        original_mask: np.ndarray,
        depth_match: np.ndarray
    ) -> np.ndarray:
        """Only keep depth-matched pixels connected to original mask."""
        # Combine original and depth-matched
        combined = np.maximum(original_mask, depth_match) > 0.5

        # Label connected components
        labeled, num_features = ndimage.label(combined)

        if num_features == 0:
            return depth_match

        # Find which labels touch the original mask
        original_binary = original_mask > 0.5
        connected_labels = set(np.unique(labeled[original_binary]))
        connected_labels.discard(0)  # Remove background label

        # Keep only connected regions
        connected_mask = np.isin(labeled, list(connected_labels))

        return (depth_match * connected_mask).astype(np.float32)

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up mask."""
        if self.config.cleanup_iterations <= 0:
            return mask

        mask_binary = (mask > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Close (fill small holes)
        mask_binary = cv2.morphologyEx(
            mask_binary, cv2.MORPH_CLOSE, kernel,
            iterations=self.config.cleanup_iterations
        )

        # Open (remove small protrusions)
        mask_binary = cv2.morphologyEx(
            mask_binary, cv2.MORPH_OPEN, kernel,
            iterations=self.config.cleanup_iterations
        )

        return mask_binary.astype(np.float32)

    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove small disconnected regions."""
        if self.config.min_region_size <= 0:
            return mask

        mask_binary = (mask > 0.5).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_binary, connectivity=8
        )

        # Keep only large enough regions
        result = np.zeros_like(mask_binary)
        for label in range(1, num_labels):  # Skip background (0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.config.min_region_size:
                result[labels == label] = 1

        return result.astype(np.float32)

    def save_debug_output(
        self,
        output_dir: Path,
        mask: np.ndarray,
        expanded_mask: np.ndarray,
        depth: np.ndarray,
        profile: Dict[str, float],
        frame_idx: int = 0
    ) -> None:
        """Save debug visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save expansion comparison
        comparison = np.zeros((*mask.shape, 3), dtype=np.uint8)
        comparison[..., 0] = (mask > 0.5).astype(np.uint8) * 255  # Original in red
        comparison[..., 1] = (expanded_mask > 0.5).astype(np.uint8) * 255  # Expanded in green
        comparison[..., 2] = (expanded_mask > mask).astype(np.uint8) * 255  # New pixels in blue

        cv2.imwrite(
            str(output_dir / f"expansion_comparison_{frame_idx:05d}.png"),
            cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        )

        # Save depth profile as text
        if self.config.save_depth_profile:
            profile_path = output_dir / f"depth_profile_{frame_idx:05d}.txt"
            with open(profile_path, 'w') as f:
                for key, value in profile.items():
                    f.write(f"{key}: {value:.6f}\n")
