#!/usr/bin/env python
"""
PROFESSIONAL MATTE COMBINATION MODULE
======================================

Multi-layer matte architecture and compositing operations.

This module implements industry-standard matte combination techniques:

1. MULTI-LAYER MATTE ARCHITECTURE
   - Core Matte: 100% solid interior, never touches edges
   - Detail Matte: Hair, transparency, fine edges
   - Soft Edge Matte: Anti-aliased boundary transitions

2. COMPOSITING OPERATIONS
   - Proper OVER operation with premultiplied alpha
   - SCREEN/ADD for additive light effects
   - MIN/MAX for matte logic
   - DIFFERENCE for matte comparison

3. DESPILL & COLOR CORRECTION
   - Complementary color suppression
   - Edge color replacement
   - Light wrap simulation

4. HOLDOUT & GARBAGE MATTES
   - Region-specific processing
   - Matte intersection/union
   - Priority-based layering

Author: AUTO-ROTO v5 Enhancement
License: MIT
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class BlendMode(Enum):
    """Standard blend modes for matte combination."""
    OVER = "over"
    UNDER = "under"
    MAX = "max"
    MIN = "min"
    ADD = "add"
    SCREEN = "screen"
    MULTIPLY = "multiply"
    DIFFERENCE = "difference"
    AVERAGE = "average"


@dataclass
class MatteCombineConfig:
    """Configuration for matte combination."""

    # Core matte settings
    core_erosion: int = 5              # Pixels to erode for solid core
    core_feather: int = 2              # Feather at core boundary

    # Detail matte settings
    detail_blend: float = 0.8          # Strength of detail matte
    detail_threshold: float = 0.1      # Min alpha for detail inclusion

    # Soft edge settings
    soft_edge_width: int = 10          # Width of soft edge band
    soft_edge_gamma: float = 1.0       # Gamma curve for falloff

    # Despill settings
    despill_enabled: bool = True
    despill_strength: float = 0.5      # 0-1, color suppression strength
    despill_limit: float = 0.8         # Max despill amount

    # Light wrap settings
    light_wrap_enabled: bool = False
    light_wrap_strength: float = 0.3   # 0-1, wrap intensity
    light_wrap_width: int = 5          # Pixel width of wrap

    # Compositing
    premultiply: bool = True           # Use premultiplied alpha
    linear_workflow: bool = True       # Work in linear color space
    gamma: float = 2.2                 # Display gamma


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("MatteCombine")


# ==============================================================================
# MATTE LAYER CLASSES
# ==============================================================================

class CoreMatte:
    """
    Generate solid core matte from input alpha.

    The core matte is eroded from the input to ensure it never
    touches problematic edge regions. It's 100% solid (1.0) inside.
    """

    def __init__(self, erosion: int = 5, feather: int = 2):
        self.erosion = erosion
        self.feather = feather

    def extract(self, alpha: np.ndarray) -> np.ndarray:
        """
        Extract core matte from alpha.

        Args:
            alpha: Input alpha (H, W), values 0-1

        Returns:
            Core matte (H, W), values 0-1
        """
        # Threshold to binary
        binary = (alpha > 0.5).astype(np.uint8)

        # Erode to create core
        if self.erosion > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.erosion * 2 + 1, self.erosion * 2 + 1)
            )
            core = cv2.erode(binary, kernel, iterations=1)
        else:
            core = binary

        # Apply feather at boundary
        if self.feather > 0:
            # Distance from core edge
            dist = cv2.distanceTransform(core, cv2.DIST_L2, 3)
            # Normalize feather region
            feather_mask = (dist > 0) & (dist < self.feather)
            core = core.astype(np.float32)
            core[feather_mask] = dist[feather_mask] / self.feather
        else:
            core = core.astype(np.float32)

        return core


class DetailMatte:
    """
    Extract detail matte capturing hair, transparency, and fine edges.

    The detail matte contains all the semi-transparent pixels
    that make up fine edge detail.
    """

    def __init__(self, threshold: float = 0.1, blend: float = 0.8):
        self.threshold = threshold
        self.blend = blend

    def extract(
        self,
        alpha: np.ndarray,
        core: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract detail matte.

        Args:
            alpha: Input alpha (H, W)
            core: Optional core matte to exclude from detail

        Returns:
            Detail matte (H, W)
        """
        # Detail is semi-transparent regions
        detail = alpha.copy()

        # Exclude solid core
        if core is not None:
            detail = detail * (1 - core)

        # Threshold to remove noise
        detail[detail < self.threshold] = 0

        # Scale by blend factor
        detail = detail * self.blend

        return detail.astype(np.float32)


class SoftEdgeMatte:
    """
    Generate soft edge matte for anti-aliased boundaries.

    Creates a distance-based falloff at mask edges for
    smooth transitions.
    """

    def __init__(self, width: int = 10, gamma: float = 1.0):
        self.width = width
        self.gamma = gamma

    def generate(self, alpha: np.ndarray) -> np.ndarray:
        """
        Generate soft edge from alpha.

        Args:
            alpha: Input alpha (H, W)

        Returns:
            Soft edge matte (H, W)
        """
        # Get edge region
        binary = (alpha > 0.5).astype(np.uint8)

        # Distance transforms
        dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)

        # Combined distance to edge
        dist_to_edge = np.minimum(dist_inside, dist_outside)

        # Create soft edge: 1 at edge, 0 away from edge
        soft_edge = np.clip(1 - dist_to_edge / self.width, 0, 1)

        # Apply gamma for curve shaping
        if self.gamma != 1.0:
            soft_edge = np.power(soft_edge, self.gamma)

        return soft_edge.astype(np.float32)


# ==============================================================================
# BLEND OPERATIONS
# ==============================================================================

class MatteBlender:
    """
    Combine mattes using various blend modes.
    """

    @staticmethod
    def blend(
        a: np.ndarray,
        b: np.ndarray,
        mode: BlendMode,
        opacity: float = 1.0
    ) -> np.ndarray:
        """
        Blend two mattes using specified mode.

        Args:
            a: First matte (typically base)
            b: Second matte (typically layer)
            mode: Blend mode
            opacity: Opacity of b layer

        Returns:
            Blended result
        """
        b_opacity = b * opacity

        if mode == BlendMode.OVER:
            # Standard over: b + a * (1 - b)
            result = b_opacity + a * (1 - b_opacity)

        elif mode == BlendMode.UNDER:
            # Under: a + b * (1 - a)
            result = a + b_opacity * (1 - a)

        elif mode == BlendMode.MAX:
            # Maximum
            result = np.maximum(a, b_opacity)

        elif mode == BlendMode.MIN:
            # Minimum
            result = np.minimum(a, b_opacity)

        elif mode == BlendMode.ADD:
            # Additive (clamped)
            result = np.clip(a + b_opacity, 0, 1)

        elif mode == BlendMode.SCREEN:
            # Screen: 1 - (1-a)(1-b)
            result = 1 - (1 - a) * (1 - b_opacity)

        elif mode == BlendMode.MULTIPLY:
            # Multiply
            result = a * b_opacity

        elif mode == BlendMode.DIFFERENCE:
            # Difference
            result = np.abs(a - b_opacity)

        elif mode == BlendMode.AVERAGE:
            # Average
            result = (a + b_opacity) / 2

        else:
            result = a

        return np.clip(result, 0, 1).astype(np.float32)

    @staticmethod
    def blend_rgb(
        fg: np.ndarray,
        fg_alpha: np.ndarray,
        bg: np.ndarray,
        bg_alpha: np.ndarray = None,
        premultiplied: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend RGB images with alpha using proper OVER operation.

        Args:
            fg: Foreground RGB
            fg_alpha: Foreground alpha
            bg: Background RGB
            bg_alpha: Background alpha (optional, default 1.0)
            premultiplied: Whether inputs are premultiplied

        Returns:
            (result_rgb, result_alpha)
        """
        if bg_alpha is None:
            bg_alpha = np.ones_like(fg_alpha)

        # Ensure 3D alpha
        fg_a = fg_alpha[:, :, np.newaxis] if len(fg_alpha.shape) == 2 else fg_alpha
        bg_a = bg_alpha[:, :, np.newaxis] if len(bg_alpha.shape) == 2 else bg_alpha

        if premultiplied:
            # Inputs already premultiplied
            result_rgb = fg + bg * (1 - fg_a)
            result_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
        else:
            # Need to premultiply
            fg_premult = fg * fg_a
            bg_premult = bg * bg_a

            result_rgb = fg_premult + bg_premult * (1 - fg_a)
            result_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)

        return result_rgb.astype(np.float32), result_alpha.astype(np.float32)


# ==============================================================================
# COLOR CORRECTION
# ==============================================================================

class Despill:
    """
    Remove background color contamination from edges.

    Uses complementary color suppression to reduce spill
    from green/blue screens or background colors.
    """

    def __init__(self, strength: float = 0.5, limit: float = 0.8):
        self.strength = strength
        self.limit = limit

    def process(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        bg_color: np.ndarray = None,
        edge_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply despill to RGB image.

        Args:
            rgb: RGB image (H, W, 3), values 0-1
            alpha: Alpha matte (H, W)
            bg_color: Background color to suppress (R, G, B)
            edge_mask: Optional mask for edge region (apply despill only there)

        Returns:
            Despilled RGB image
        """
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0

        despilled = rgb.copy()

        # Auto-detect background color if not provided
        if bg_color is None:
            bg_color = self._detect_bg_color(rgb, alpha)

        # Find dominant color channel
        dominant_channel = np.argmax(bg_color)

        # Compute spill amount
        # Spill = dominant channel - average of others
        other_channels = np.delete(np.arange(3), dominant_channel)
        avg_others = np.mean(rgb[:, :, other_channels], axis=2)
        spill = rgb[:, :, dominant_channel] - avg_others
        spill = np.clip(spill, 0, self.limit)

        # Create despill mask
        if edge_mask is not None:
            # Only despill at edges
            despill_mask = edge_mask * self.strength
        else:
            # Despill in semi-transparent regions
            semi_transparent = (alpha > 0.1) & (alpha < 0.9)
            despill_mask = semi_transparent.astype(np.float32) * self.strength

        # Apply despill
        reduction = spill * despill_mask
        despilled[:, :, dominant_channel] = np.clip(
            despilled[:, :, dominant_channel] - reduction, 0, 1
        )

        return despilled

    def _detect_bg_color(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """Detect background color from low-alpha regions."""
        bg_mask = alpha < 0.2

        if np.sum(bg_mask) > 100:
            bg_color = np.median(rgb[bg_mask], axis=0)
        else:
            # Fallback: assume neutral gray
            bg_color = np.array([0.5, 0.5, 0.5])

        return bg_color


class LightWrap:
    """
    Simulate light bleeding from background onto foreground edges.

    Creates a more natural composite by wrapping background
    illumination around foreground edges.
    """

    def __init__(self, strength: float = 0.3, width: int = 5):
        self.strength = strength
        self.width = width

    def apply(
        self,
        fg: np.ndarray,
        bg: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """
        Apply light wrap effect.

        Args:
            fg: Foreground RGB (H, W, 3)
            bg: Background RGB (H, W, 3)
            alpha: Alpha matte (H, W)

        Returns:
            Foreground with light wrap applied
        """
        if fg.max() > 1:
            fg = fg.astype(np.float32) / 255.0
        if bg.max() > 1:
            bg = bg.astype(np.float32) / 255.0

        # Create edge mask for wrap region
        binary = (alpha > 0.5).astype(np.uint8)
        dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

        # Wrap region: just inside the edge
        wrap_mask = (dist_inside > 0) & (dist_inside < self.width)
        wrap_strength = 1 - (dist_inside / self.width)
        wrap_strength[~wrap_mask] = 0
        wrap_strength = wrap_strength * self.strength

        # Blur background for soft light wrap
        blur_size = self.width * 2 + 1
        bg_blurred = cv2.GaussianBlur(bg, (blur_size, blur_size), 0)

        # Apply wrap using SCREEN blend mode
        wrap_strength_3d = wrap_strength[:, :, np.newaxis]
        wrapped = self._screen_blend(fg, bg_blurred, wrap_strength_3d)

        return wrapped

    def _screen_blend(
        self,
        base: np.ndarray,
        blend: np.ndarray,
        opacity: np.ndarray
    ) -> np.ndarray:
        """Apply screen blend with opacity."""
        screen = 1 - (1 - base) * (1 - blend)
        result = base * (1 - opacity) + screen * opacity
        return np.clip(result, 0, 1).astype(np.float32)


class EdgeColorCorrection:
    """
    Correct edge colors to match foreground better.

    Samples foreground color and extends it to semi-transparent edges
    to reduce color fringing.
    """

    def __init__(self, sample_radius: int = 10):
        self.sample_radius = sample_radius

    def correct(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """
        Correct edge colors.

        Args:
            rgb: RGB image (H, W, 3)
            alpha: Alpha matte (H, W)

        Returns:
            Color-corrected RGB
        """
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0

        corrected = rgb.copy()

        # Get edge region
        binary = (alpha > 0.5).astype(np.uint8)
        dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

        # Sample foreground color from well inside mask
        fg_mask = dist_inside > self.sample_radius
        if np.sum(fg_mask) > 100:
            fg_color = np.median(rgb[fg_mask], axis=0)
        else:
            return rgb  # Not enough samples

        # Edge region: semi-transparent pixels
        edge_mask = (alpha > 0.1) & (alpha < 0.9)

        # Blend edge colors towards foreground color
        blend_strength = 0.3 * edge_mask.astype(np.float32)
        blend_strength = blend_strength[:, :, np.newaxis]

        corrected = corrected * (1 - blend_strength) + fg_color * blend_strength

        return corrected.astype(np.float32)


# ==============================================================================
# HOLDOUT & GARBAGE MATTES
# ==============================================================================

class HoldoutMatte:
    """
    Apply holdout (garbage) mattes to exclude/include regions.
    """

    @staticmethod
    def apply_holdout(
        alpha: np.ndarray,
        holdout: np.ndarray,
        mode: str = "subtract"
    ) -> np.ndarray:
        """
        Apply holdout matte.

        Args:
            alpha: Main alpha matte
            holdout: Holdout matte (regions to affect)
            mode: "subtract" (remove), "intersect" (keep only), "union" (add)

        Returns:
            Modified alpha
        """
        if mode == "subtract":
            # Remove holdout region
            result = alpha * (1 - holdout)

        elif mode == "intersect":
            # Keep only intersection
            result = np.minimum(alpha, holdout)

        elif mode == "union":
            # Add holdout region
            result = np.maximum(alpha, holdout)

        else:
            result = alpha

        return np.clip(result, 0, 1).astype(np.float32)

    @staticmethod
    def create_from_box(
        shape: Tuple[int, int],
        box: Tuple[int, int, int, int],
        feather: int = 10
    ) -> np.ndarray:
        """
        Create holdout matte from bounding box.

        Args:
            shape: (H, W) of output matte
            box: (x1, y1, x2, y2) bounding box
            feather: Edge feathering in pixels

        Returns:
            Holdout matte
        """
        h, w = shape
        x1, y1, x2, y2 = box

        holdout = np.zeros((h, w), dtype=np.float32)
        holdout[y1:y2, x1:x2] = 1.0

        if feather > 0:
            holdout = cv2.GaussianBlur(holdout, (feather * 2 + 1, feather * 2 + 1), 0)

        return holdout


# ==============================================================================
# MAIN COMBINER CLASS
# ==============================================================================

class MatteCombiner:
    """
    Main class for combining multiple matte layers.
    """

    def __init__(
        self,
        config: MatteCombineConfig = None,
        logger: logging.Logger = None
    ):
        self.config = config or MatteCombineConfig()
        self.logger = logger or logging.getLogger("MatteCombiner")

        # Initialize components
        self.core_extractor = CoreMatte(
            self.config.core_erosion,
            self.config.core_feather
        )
        self.detail_extractor = DetailMatte(
            self.config.detail_threshold,
            self.config.detail_blend
        )
        self.soft_edge_generator = SoftEdgeMatte(
            self.config.soft_edge_width,
            self.config.soft_edge_gamma
        )
        self.despill = Despill(
            self.config.despill_strength,
            self.config.despill_limit
        )
        self.light_wrap = LightWrap(
            self.config.light_wrap_strength,
            self.config.light_wrap_width
        )
        self.edge_color = EdgeColorCorrection()
        self.blender = MatteBlender()

    def combine(
        self,
        alpha: np.ndarray,
        detail_alpha: np.ndarray = None,
        holdout: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Combine mattes using multi-layer architecture.

        Args:
            alpha: Primary alpha matte (from SAM2 + depth)
            detail_alpha: Optional detail matte (from ViTMatte/hair)
            holdout: Optional holdout matte

        Returns:
            Dict with 'final_alpha', 'core', 'detail', 'soft_edge'
        """
        self.logger.debug("Extracting matte layers...")

        # Extract layers from primary alpha
        core = self.core_extractor.extract(alpha)
        soft_edge = self.soft_edge_generator.generate(alpha)

        # Get detail from primary alpha
        detail_primary = self.detail_extractor.extract(alpha, core)

        # If separate detail alpha provided, blend it
        if detail_alpha is not None:
            detail_combined = self.blender.blend(
                detail_primary, detail_alpha,
                BlendMode.MAX, self.config.detail_blend
            )
        else:
            detail_combined = detail_primary

        # Combine layers:
        # 1. Start with core (always solid)
        # 2. Add detail (soft blend)
        # 3. Use soft edge for transitions

        self.logger.debug("Combining layers...")

        # Core + Detail with MAX blend
        combined = self.blender.blend(core, detail_combined, BlendMode.MAX)

        # Apply soft edge influence
        # Soft edge defines WHERE transitions happen, not the alpha value
        # Use it to ensure smooth transition at boundaries
        edge_influence = soft_edge * 0.5  # Reduce to subtle influence
        combined = combined * (1 - edge_influence) + alpha * edge_influence

        # Apply holdout if provided
        if holdout is not None:
            combined = HoldoutMatte.apply_holdout(combined, holdout, "subtract")

        final = np.clip(combined, 0, 1).astype(np.float32)

        return {
            'final_alpha': final,
            'core': core,
            'detail': detail_combined,
            'soft_edge': soft_edge
        }

    def combine_with_rgb(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        detail_alpha: np.ndarray = None,
        bg_rgb: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Combine mattes with RGB color correction.

        Args:
            rgb: Foreground RGB
            alpha: Primary alpha
            detail_alpha: Optional detail alpha
            bg_rgb: Optional background for light wrap

        Returns:
            Dict with 'final_alpha', 'final_rgb', layers
        """
        # Get matte combination
        matte_result = self.combine(alpha, detail_alpha)

        # Color correct RGB
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0

        corrected_rgb = rgb.copy()

        # Apply despill
        if self.config.despill_enabled:
            self.logger.debug("Applying despill...")
            edge_mask = matte_result['soft_edge']
            corrected_rgb = self.despill.process(
                corrected_rgb, alpha, edge_mask=edge_mask
            )

        # Apply edge color correction
        corrected_rgb = self.edge_color.correct(corrected_rgb, alpha)

        # Apply light wrap
        if self.config.light_wrap_enabled and bg_rgb is not None:
            self.logger.debug("Applying light wrap...")
            corrected_rgb = self.light_wrap.apply(
                corrected_rgb, bg_rgb, alpha
            )

        # Premultiply if configured
        if self.config.premultiply:
            alpha_3d = matte_result['final_alpha'][:, :, np.newaxis]
            corrected_rgb = corrected_rgb * alpha_3d

        matte_result['final_rgb'] = corrected_rgb

        return matte_result


# ==============================================================================
# PIPELINE
# ==============================================================================

class MatteCombinePipeline:
    """
    Pipeline for batch matte combination.
    """

    def __init__(
        self,
        config: MatteCombineConfig = None,
        output_format: str = "exr",
        bit_depth: int = 16,
        verbose: bool = False
    ):
        self.config = config or MatteCombineConfig()
        self.output_format = output_format
        self.bit_depth = bit_depth
        self.logger = setup_logging(verbose)
        self.combiner = MatteCombiner(self.config, self.logger)

    def process_sequence(
        self,
        alpha_dir: Path,
        output_dir: Path,
        detail_dir: Path = None,
        frames_dir: Path = None
    ):
        """
        Process alpha sequence with matte combination.

        Args:
            alpha_dir: Primary alpha mattes (from SAM2/depth)
            output_dir: Output directory
            detail_dir: Optional detail mattes (from ViTMatte)
            frames_dir: Optional RGB frames for color correction
        """
        import cv2

        output_dir.mkdir(parents=True, exist_ok=True)
        combined_dir = output_dir / "alpha"
        combined_dir.mkdir(exist_ok=True)

        if frames_dir:
            rgb_dir = output_dir / "rgb"
            rgb_dir.mkdir(exist_ok=True)

        # Find files
        alpha_files = sorted(list(alpha_dir.glob("*.exr")) +
                           list(alpha_dir.glob("*.png")))

        detail_files = []
        if detail_dir and detail_dir.exists():
            detail_files = sorted(list(detail_dir.glob("*.exr")) +
                                list(detail_dir.glob("*.png")))

        frame_files = []
        if frames_dir and frames_dir.exists():
            frame_files = sorted(list(frames_dir.glob("*.png")) +
                               list(frames_dir.glob("*.jpg")))

        self.logger.info(f"Processing {len(alpha_files)} frames...")

        for i, alpha_path in enumerate(alpha_files):
            # Load primary alpha
            alpha = self._load_alpha(alpha_path)

            # Load detail if available
            detail = None
            if i < len(detail_files):
                detail = self._load_alpha(detail_files[i])

            # Load RGB if available
            rgb = None
            if i < len(frame_files):
                rgb = cv2.imread(str(frame_files[i]))
                if rgb is not None:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    rgb = rgb.astype(np.float32) / 255.0

            # Process
            if rgb is not None:
                result = self.combiner.combine_with_rgb(rgb, alpha, detail)
                # Save RGB
                rgb_path = rgb_dir / f"combined.{i:04d}.{self.output_format}"
                self._save_rgb(result['final_rgb'], result['final_alpha'], rgb_path)
            else:
                result = self.combiner.combine(alpha, detail)

            # Save alpha
            output_path = combined_dir / f"combined.{i:04d}.{self.output_format}"
            self._save_alpha(result['final_alpha'], output_path)

            if i % 20 == 0:
                self.logger.info(f"  Processed frame {i}/{len(alpha_files)}")

        self.logger.info("Matte combination complete!")

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
                return np.frombuffer(data, dtype=np.float32).reshape((h, w))
            except ImportError:
                pass

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read: {path}")

        if len(img.shape) == 3:
            img = img[:, :, -1]

        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
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
                path = path.with_suffix('.png')

        if self.bit_depth == 16:
            alpha_int = (alpha * 65535).astype(np.uint16)
        else:
            alpha_int = (alpha * 255).astype(np.uint8)

        cv2.imwrite(str(path), alpha_int)

    def _save_rgb(self, rgb: np.ndarray, alpha: np.ndarray, path: Path):
        """Save RGBA to file."""
        import cv2

        # Unpremultiply if needed
        if self.config.premultiply:
            alpha_3d = alpha[:, :, np.newaxis]
            safe_alpha = np.maximum(alpha_3d, 1e-8)
            rgb = rgb / safe_alpha
            rgb = np.where(alpha_3d > 1e-8, rgb, 0)

        # Combine RGBA
        rgba = np.concatenate([rgb, alpha[:, :, np.newaxis]], axis=2)

        if path.suffix.lower() == '.exr':
            try:
                import OpenEXR
                import Imath

                h, w = alpha.shape
                header = OpenEXR.Header(w, h)
                header['channels'] = {
                    'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'A': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                }

                exr = OpenEXR.OutputFile(str(path), header)
                exr.writePixels({
                    'R': rgba[:, :, 0].astype(np.float32).tobytes(),
                    'G': rgba[:, :, 1].astype(np.float32).tobytes(),
                    'B': rgba[:, :, 2].astype(np.float32).tobytes(),
                    'A': rgba[:, :, 3].astype(np.float32).tobytes(),
                })
                exr.close()
                return
            except ImportError:
                path = path.with_suffix('.png')

        # Convert to BGR for OpenCV
        bgra = np.zeros_like(rgba)
        bgra[:, :, 0] = rgba[:, :, 2]  # B
        bgra[:, :, 1] = rgba[:, :, 1]  # G
        bgra[:, :, 2] = rgba[:, :, 0]  # R
        bgra[:, :, 3] = rgba[:, :, 3]  # A

        if self.bit_depth == 16:
            bgra_int = (np.clip(bgra, 0, 1) * 65535).astype(np.uint16)
        else:
            bgra_int = (np.clip(bgra, 0, 1) * 255).astype(np.uint8)

        cv2.imwrite(str(path), bgra_int)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Professional matte combination and color correction"
    )

    parser.add_argument("--alpha", "-a", required=True,
                       help="Primary alpha directory")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory")
    parser.add_argument("--detail", "-d",
                       help="Detail alpha directory (from hair refinement)")
    parser.add_argument("--frames", "-f",
                       help="RGB frames directory for color correction")

    # Core settings
    parser.add_argument("--core-erosion", type=int, default=5,
                       help="Core erosion pixels (default: 5)")
    parser.add_argument("--core-feather", type=int, default=2,
                       help="Core feather pixels (default: 2)")

    # Color correction
    parser.add_argument("--despill", type=float, default=0.5,
                       help="Despill strength 0-1 (default: 0.5)")
    parser.add_argument("--no-despill", action="store_true",
                       help="Disable despill")
    parser.add_argument("--light-wrap", type=float, default=0.0,
                       help="Light wrap strength 0-1 (default: 0, disabled)")

    # Output
    parser.add_argument("--format", default="exr",
                       choices=["exr", "png", "tiff"])
    parser.add_argument("--bit-depth", type=int, default=16,
                       choices=[8, 16, 32])

    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = MatteCombineConfig(
        core_erosion=args.core_erosion,
        core_feather=args.core_feather,
        despill_enabled=not args.no_despill,
        despill_strength=args.despill,
        light_wrap_enabled=args.light_wrap > 0,
        light_wrap_strength=args.light_wrap
    )

    pipeline = MatteCombinePipeline(
        config=config,
        output_format=args.format,
        bit_depth=args.bit_depth,
        verbose=args.verbose
    )

    pipeline.process_sequence(
        alpha_dir=Path(args.alpha),
        output_dir=Path(args.output),
        detail_dir=Path(args.detail) if args.detail else None,
        frames_dir=Path(args.frames) if args.frames else None
    )


if __name__ == "__main__":
    main()
