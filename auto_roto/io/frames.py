"""
Frame Writer
============

Write frames with alpha to various formats.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("AutoRoto.IO.Frames")


class FrameWriter:
    """
    Write frames with alpha to various formats.

    Supports:
        - EXR: 16/32-bit float, industry standard
        - PNG: 8/16-bit, good for web preview
        - TIFF: 8/16-bit, good compatibility
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str = "roto",
        format: str = "exr",
        bit_depth: int = 16,
        padding: int = 4,
        logger: logging.Logger = None
    ):
        """
        Initialize frame writer.

        Args:
            output_dir: Base output directory
            prefix: Filename prefix (e.g., "roto" -> "roto.0001.exr")
            format: Output format (exr, png, tiff)
            bit_depth: Output bit depth (8, 16, 32)
            padding: Frame number padding (4 -> ####)
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.format = format.lower()
        self.bit_depth = bit_depth
        self.padding = padding
        self.logger = logger or logging.getLogger("FrameWriter")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.alpha_dir = self.output_dir / "alpha"
        self.rgba_dir = self.output_dir / "rgba"
        self.preview_dir = self.output_dir / "preview"

        self.alpha_dir.mkdir(exist_ok=True)
        self.rgba_dir.mkdir(exist_ok=True)
        self.preview_dir.mkdir(exist_ok=True)

    def _get_filename(self, frame_idx: int, subdir: str = "alpha") -> Path:
        """Generate output filename."""
        frame_str = str(frame_idx).zfill(self.padding)

        if self.format == "exr":
            ext = ".exr"
        elif self.format == "png":
            ext = ".png"
        elif self.format in ("tiff", "tif"):
            ext = ".tif"
        else:
            ext = f".{self.format}"

        base_dir = getattr(self, f"{subdir}_dir", self.output_dir)
        return base_dir / f"{self.prefix}.{frame_str}{ext}"

    def write_alpha(self, alpha: np.ndarray, frame_idx: int) -> Path:
        """
        Write alpha channel to file.

        Args:
            alpha: 2D numpy array, values 0-1
            frame_idx: Frame number

        Returns:
            Path to written file
        """
        import cv2
        from auto_roto.io.exr import write_exr_channel

        filepath = self._get_filename(frame_idx, "alpha")

        # Ensure proper range
        alpha = np.clip(alpha, 0, 1)

        if self.format == "exr":
            write_exr_channel(alpha, filepath, channel='A', bit_depth=self.bit_depth)
        elif self.format == "png":
            if self.bit_depth == 16:
                alpha_int = (alpha * 65535).astype(np.uint16)
            else:
                alpha_int = (alpha * 255).astype(np.uint8)
            cv2.imwrite(str(filepath), alpha_int)
        elif self.format in ("tiff", "tif"):
            if self.bit_depth == 16:
                alpha_int = (alpha * 65535).astype(np.uint16)
            else:
                alpha_int = (alpha * 255).astype(np.uint8)
            cv2.imwrite(str(filepath), alpha_int)

        return filepath

    def write_rgba(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        frame_idx: int
    ) -> Path:
        """
        Write RGBA image to file.

        Args:
            rgb: RGB image, uint8 or float
            alpha: Alpha channel, values 0-1
            frame_idx: Frame number

        Returns:
            Path to written file
        """
        import cv2
        from auto_roto.io.exr import write_exr_rgba

        filepath = self._get_filename(frame_idx, "rgba")

        # Normalize inputs
        if rgb.dtype == np.uint8:
            rgb_float = rgb.astype(np.float32) / 255.0
        else:
            rgb_float = rgb.astype(np.float32)

        alpha = np.clip(alpha, 0, 1).astype(np.float32)

        if self.format == "exr":
            write_exr_rgba(rgb_float, alpha, filepath, bit_depth=self.bit_depth)
        elif self.format == "png":
            # Combine RGBA
            if len(alpha.shape) == 2:
                alpha = alpha[..., np.newaxis]

            rgba = np.concatenate([rgb_float, alpha], axis=-1)

            # Convert to BGR for OpenCV
            bgra = np.zeros_like(rgba)
            bgra[..., 0] = rgba[..., 2]  # B
            bgra[..., 1] = rgba[..., 1]  # G
            bgra[..., 2] = rgba[..., 0]  # R
            bgra[..., 3] = rgba[..., 3]  # A

            if self.bit_depth == 16:
                bgra_int = (bgra * 65535).astype(np.uint16)
            else:
                bgra_int = (bgra * 255).astype(np.uint8)

            cv2.imwrite(str(filepath), bgra_int)
        elif self.format in ("tiff", "tif"):
            # Similar to PNG
            if len(alpha.shape) == 2:
                alpha = alpha[..., np.newaxis]

            rgba = np.concatenate([rgb_float, alpha], axis=-1)

            bgra = np.zeros_like(rgba)
            bgra[..., 0] = rgba[..., 2]
            bgra[..., 1] = rgba[..., 1]
            bgra[..., 2] = rgba[..., 0]
            bgra[..., 3] = rgba[..., 3]

            if self.bit_depth == 16:
                bgra_int = (bgra * 65535).astype(np.uint16)
            else:
                bgra_int = (bgra * 255).astype(np.uint8)

            cv2.imwrite(str(filepath), bgra_int)

        return filepath

    def write_preview(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        frame_idx: int,
        scale: float = 0.5
    ) -> Path:
        """
        Write preview image with alpha overlay.

        Args:
            rgb: RGB image
            alpha: Alpha channel
            frame_idx: Frame number
            scale: Preview scale factor

        Returns:
            Path to written file
        """
        import cv2

        filepath = self.preview_dir / f"{self.prefix}.{str(frame_idx).zfill(self.padding)}.jpg"

        # Ensure RGB is uint8
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        # Create side-by-side preview
        h, w = rgb.shape[:2]

        # Alpha visualization (green = FG, red = BG)
        alpha_vis = np.zeros((h, w, 3), dtype=np.uint8)
        alpha_vis[..., 1] = (alpha * 255).astype(np.uint8)  # Green channel

        # Composite
        composite = (rgb * alpha[..., np.newaxis]).astype(np.uint8)

        # Combine
        preview = np.hstack([rgb, alpha_vis, composite])

        # Scale
        if scale != 1.0:
            new_w = int(preview.shape[1] * scale)
            new_h = int(preview.shape[0] * scale)
            preview = cv2.resize(preview, (new_w, new_h))

        cv2.imwrite(str(filepath), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

        return filepath

    def get_alpha_dir(self) -> Path:
        """Get the alpha output directory."""
        return self.alpha_dir

    def get_rgba_dir(self) -> Path:
        """Get the RGBA output directory."""
        return self.rgba_dir

    def get_preview_dir(self) -> Path:
        """Get the preview output directory."""
        return self.preview_dir
