#!/usr/bin/env python
"""
AUTO-ROTO: Production-Grade Automatic Rotoscoping Pipeline
===========================================================

Combines SAM2 (video segmentation) + GroundingDINO (text detection) + Alpha Refinement
for bulletproof automatic rotoscoping that generates perfect alphas from complex clips.

Author: Built for Core Form VFX
License: MIT

ARCHITECTURE:
    Input Video/Sequence
           |
           v
    ┌─────────────────┐
    │ Frame Extraction │
    └────────┬────────┘
             │
             v
    ┌─────────────────┐     ┌──────────────────┐
    │  GroundingDINO  │ OR  │  Interactive     │
    │  (text prompt)  │     │  (point/box)     │
    └────────┬────────┘     └────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         v
              ┌─────────────────┐
              │     SAM 2.1     │
              │ (video predict) │
              └────────┬────────┘
                       │
                       v
              ┌─────────────────┐
              │ Alpha Refinement│
              │ (edge matting)  │
              └────────┬────────┘
                       │
                       v
              ┌─────────────────┐
              │  EXR Export     │
              │  (16-bit float) │
              └─────────────────┘

USAGE:
    # Text-based detection (automatic)
    python auto_roto.py --input video.mp4 --prompt "person" --output ./output

    # Interactive mode (click to select)
    python auto_roto.py --input video.mp4 --interactive --output ./output

    # Box prompt on first frame
    python auto_roto.py --input video.mp4 --box "100,100,500,400" --output ./output

    # Multiple objects
    python auto_roto.py --input video.mp4 --prompt "person.dog.car" --output ./output
"""

import os
import sys
import argparse
import logging
import platform
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time

# Lazy imports for faster startup and better error messages
def _check_dependencies():
    """Check and report missing dependencies with installation instructions."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append(("torch", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"))
    
    try:
        import numpy
    except ImportError:
        missing.append(("numpy", "pip install numpy"))
    
    try:
        import cv2
    except ImportError:
        missing.append(("opencv-python", "pip install opencv-python"))
    
    if missing:
        print("\n" + "="*60)
        print("MISSING DEPENDENCIES")
        print("="*60)
        for pkg, cmd in missing:
            print(f"\n  {pkg}:")
            print(f"    {cmd}")
        print("\n" + "="*60)
        sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class RotoConfig:
    """Configuration for the auto-roto pipeline."""
    
    # Input/Output
    input_path: str = ""
    output_dir: str = "./output"
    
    # Detection mode
    prompt: Optional[str] = None  # Text prompt for GroundingDINO
    box: Optional[str] = None     # Box prompt "x1,y1,x2,y2"
    point: Optional[str] = None   # Point prompt "x,y"
    interactive: bool = False     # Interactive selection mode
    
    # SAM2 settings
    sam_model: str = "large"      # tiny, small, base_plus, large
    propagate_forward: bool = True
    propagate_backward: bool = True
    
    # GroundingDINO settings
    detection_threshold: float = 0.3
    text_threshold: float = 0.25
    
    # Alpha refinement
    refine_alpha: bool = True
    refine_iterations: int = 3
    edge_softness: float = 1.0
    
    # Output settings
    output_format: str = "exr"    # exr, png, tiff
    bit_depth: int = 16           # 8, 16, 32
    include_rgb: bool = True      # Include RGB in output
    frame_padding: int = 4        # Frame number padding (####)
    
    # Performance
    device: str = "cuda"
    batch_size: int = 1
    compile_model: bool = True    # Use torch.compile for speed
    
    # Debug
    verbose: bool = False
    save_preview: bool = True
    preview_scale: float = 0.5


class PromptType(Enum):
    """Types of prompts for segmentation."""
    TEXT = "text"
    BOX = "box"
    POINT = "point"
    INTERACTIVE = "interactive"


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logger = logging.getLogger("AutoRoto")
    return logger


# ==============================================================================
# VIDEO I/O
# ==============================================================================

class VideoReader:
    """
    Read video files or image sequences into frames.
    
    Supports:
        - Video files: .mp4, .mov, .avi, .mkv
        - Image sequences: frame.####.exr, frame_####.png, etc.
    """
    
    SUPPORTED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf'}
    SUPPORTED_IMAGE = {'.exr', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.dpx'}
    
    def __init__(self, path: str, logger: logging.Logger):
        self.path = Path(path)
        self.logger = logger
        self.frames: List[Path] = []
        self.fps: float = 24.0
        self.width: int = 0
        self.height: int = 0
        self._is_sequence = False
        self._cap = None
        
        self._detect_input_type()
    
    def _detect_input_type(self):
        """Detect if input is video file or image sequence."""
        if self.path.is_file():
            ext = self.path.suffix.lower()
            if ext in self.SUPPORTED_VIDEO:
                self._init_video()
            elif ext in self.SUPPORTED_IMAGE:
                self._init_sequence_from_file()
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        elif self.path.is_dir():
            self._init_sequence_from_dir()
        else:
            # Try pattern matching for sequences
            self._init_sequence_from_pattern()
    
    def _init_video(self):
        """Initialize from video file."""
        import cv2
        
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video: {self.path.name}")
        self.logger.info(f"  Resolution: {self.width}x{self.height}")
        self.logger.info(f"  FPS: {self.fps:.2f}")
        self.logger.info(f"  Frames: {frame_count}")
    
    def _init_sequence_from_dir(self):
        """Initialize from directory of images."""
        self._is_sequence = True
        
        # Find all image files
        for ext in self.SUPPORTED_IMAGE:
            self.frames.extend(sorted(self.path.glob(f"*{ext}")))
            self.frames.extend(sorted(self.path.glob(f"*{ext.upper()}")))
        
        self.frames = sorted(set(self.frames))
        
        if not self.frames:
            raise ValueError(f"No image files found in: {self.path}")
        
        # Get dimensions from first frame
        self._read_first_frame_info()
    
    def _init_sequence_from_file(self):
        """Initialize sequence from single file (find related frames)."""
        self._is_sequence = True
        
        # Extract pattern from filename
        import re
        name = self.path.stem
        ext = self.path.suffix
        parent = self.path.parent
        
        # Match common patterns: name.0001.ext, name_0001.ext, name0001.ext
        patterns = [
            r'^(.+?)[\._]?(\d+)$',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                prefix = match.group(1)
                # Find all related frames
                for f in sorted(parent.glob(f"{prefix}*{ext}")):
                    if re.match(rf'^{re.escape(prefix)}[\._]?\d+$', f.stem):
                        self.frames.append(f)
                break
        
        if not self.frames:
            self.frames = [self.path]
        
        self._read_first_frame_info()
    
    def _init_sequence_from_pattern(self):
        """Initialize from glob pattern."""
        self._is_sequence = True
        
        # Replace # with glob pattern
        pattern = str(self.path).replace('#', '[0-9]')
        import glob
        self.frames = sorted([Path(f) for f in glob.glob(pattern)])
        
        if not self.frames:
            raise ValueError(f"No files match pattern: {self.path}")
        
        self._read_first_frame_info()
    
    def _read_first_frame_info(self):
        """Read first frame to get dimensions."""
        import cv2
        
        frame = cv2.imread(str(self.frames[0]), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise IOError(f"Cannot read: {self.frames[0]}")
        
        self.height, self.width = frame.shape[:2]
        
        self.logger.info(f"Sequence: {self.frames[0].parent.name}/")
        self.logger.info(f"  Resolution: {self.width}x{self.height}")
        self.logger.info(f"  Frames: {len(self.frames)}")
    
    def __len__(self) -> int:
        if self._is_sequence:
            return len(self.frames)
        else:
            return int(self._cap.get(cv2.CV_CAP_PROP_FRAME_COUNT) if self._cap else 0)
    
    def __iter__(self):
        """Iterate over frames as numpy arrays (RGB)."""
        import cv2
        import numpy as np
        
        if self._is_sequence:
            for frame_path in self.frames:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                if frame is None:
                    self.logger.warning(f"Skipping unreadable frame: {frame_path}")
                    continue
                
                # Convert 16-bit to 8-bit if needed
                if frame.dtype == np.uint16:
                    frame = (frame / 256).astype(np.uint8)
                elif frame.dtype in (np.float32, np.float64):
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

                # Convert to RGB
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield frame
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_frame(self, idx: int):
        """Get specific frame by index."""
        import cv2
        import numpy as np

        if self._is_sequence:
            if 0 <= idx < len(self.frames):
                frame = cv2.imread(str(self.frames[idx]), cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    # Convert 16-bit to 8-bit if needed
                    if frame.dtype == np.uint16:
                        frame = (frame / 256).astype(np.uint8)
                    elif frame.dtype in (np.float32, np.float64):
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def __del__(self):
        if self._cap is not None:
            self._cap.release()


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
    
    def write_alpha(self, alpha: 'np.ndarray', frame_idx: int) -> Path:
        """
        Write alpha channel to file.
        
        Args:
            alpha: 2D numpy array, values 0-1
            frame_idx: Frame number
            
        Returns:
            Path to written file
        """
        import cv2
        import numpy as np
        
        filepath = self._get_filename(frame_idx, "alpha")
        
        # Ensure proper range
        alpha = np.clip(alpha, 0, 1)
        
        if self.format == "exr":
            self._write_exr(alpha, filepath)
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
        rgb: 'np.ndarray',
        alpha: 'np.ndarray',
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
        import numpy as np
        
        filepath = self._get_filename(frame_idx, "rgba")
        
        # Normalize inputs
        if rgb.dtype == np.uint8:
            rgb_float = rgb.astype(np.float32) / 255.0
        else:
            rgb_float = rgb.astype(np.float32)
        
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        
        # Combine RGBA
        if len(alpha.shape) == 2:
            alpha = alpha[..., np.newaxis]
        
        rgba = np.concatenate([rgb_float, alpha], axis=-1)
        
        if self.format == "exr":
            self._write_exr(rgba, filepath)
        elif self.format == "png":
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
        rgb: 'np.ndarray',
        alpha: 'np.ndarray',
        frame_idx: int,
        scale: float = 0.5
    ) -> Path:
        """Write preview image with alpha overlay."""
        import cv2
        import numpy as np
        
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
    
    def _write_exr(self, data: 'np.ndarray', filepath: Path):
        """Write EXR file using OpenEXR or cv2."""
        import numpy as np
        
        try:
            import OpenEXR
            import Imath
            
            h, w = data.shape[:2]
            channels = data.shape[2] if len(data.shape) > 2 else 1
            
            # Determine pixel type
            if self.bit_depth == 32:
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                data = data.astype(np.float32)
            else:
                pixel_type = Imath.PixelType(Imath.PixelType.HALF)
                data = data.astype(np.float16)
            
            # Create header
            header = OpenEXR.Header(w, h)
            
            # Set channel names
            if channels == 1:
                channel_names = ['A']  # Alpha only
            elif channels == 3:
                channel_names = ['R', 'G', 'B']
            else:
                channel_names = ['R', 'G', 'B', 'A']
            
            header['channels'] = {
                name: Imath.Channel(pixel_type)
                for name in channel_names
            }
            
            # Write file
            exr = OpenEXR.OutputFile(str(filepath), header)
            
            channel_data = {}
            if channels == 1:
                channel_data['A'] = data.tobytes()
            else:
                for i, name in enumerate(channel_names):
                    if len(data.shape) > 2:
                        channel_data[name] = data[..., i].tobytes()
                    else:
                        channel_data[name] = data.tobytes()
            
            exr.writePixels(channel_data)
            exr.close()
            
        except ImportError:
            # Fallback to cv2 (limited EXR support)
            import cv2
            
            if len(data.shape) == 2:
                data = data[..., np.newaxis]
            
            # cv2 expects BGR order for color
            if data.shape[2] >= 3:
                bgr_data = data.copy()
                bgr_data[..., 0] = data[..., 2]  # B
                bgr_data[..., 2] = data[..., 0]  # R
                data = bgr_data
            
            cv2.imwrite(str(filepath), data.astype(np.float32))


# ==============================================================================
# ALPHA REFINEMENT
# ==============================================================================

class AlphaRefiner:
    """
    Refine binary/rough masks into smooth, production-quality alpha mattes.
    
    Techniques:
        - Guided filter for edge-aware smoothing
        - Morphological operations for cleanup
        - Edge softening for natural falloff
    """
    
    def __init__(
        self,
        iterations: int = 3,
        edge_softness: float = 1.0,
        logger: logging.Logger = None
    ):
        self.iterations = iterations
        self.edge_softness = edge_softness
        self.logger = logger or logging.getLogger("AlphaRefiner")
    
    def refine(
        self,
        mask: 'np.ndarray',
        image: 'np.ndarray',
        trimap: 'np.ndarray' = None
    ) -> 'np.ndarray':
        """
        Refine a binary mask into a smooth alpha matte.
        
        Args:
            mask: Binary or rough mask (0-1)
            image: RGB image for guided filtering
            trimap: Optional trimap (0=BG, 0.5=unknown, 1=FG)
            
        Returns:
            Refined alpha matte (0-1, float32)
        """
        import cv2
        import numpy as np
        
        mask = mask.astype(np.float32)
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Step 1: Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Step 2: Generate trimap if not provided
        if trimap is None:
            trimap = self._generate_trimap(mask)
        
        # Step 3: Apply guided filter for edge refinement
        alpha = self._guided_filter_matting(image, mask, trimap)
        
        # Step 4: Edge softening
        if self.edge_softness > 0:
            alpha = self._soften_edges(alpha, self.edge_softness)
        
        # Step 5: Final cleanup
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        
        return alpha
    
    def _generate_trimap(self, mask: 'np.ndarray', erosion: int = 10, dilation: int = 20) -> 'np.ndarray':
        """Generate trimap from binary mask."""
        import cv2
        import numpy as np
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Create foreground (eroded mask)
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion))
        fg = cv2.erode(mask_uint8, kernel_e, iterations=1)
        
        # Create background (inverted dilated mask)
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        dilated = cv2.dilate(mask_uint8, kernel_d, iterations=1)
        bg = 255 - dilated
        
        # Trimap: 0 = bg, 0.5 = unknown, 1 = fg
        trimap = np.ones_like(mask, dtype=np.float32) * 0.5
        trimap[fg > 127] = 1.0
        trimap[bg > 127] = 0.0
        
        return trimap
    
    def _guided_filter_matting(
        self,
        image: 'np.ndarray',
        mask: 'np.ndarray',
        trimap: 'np.ndarray',
        radius: int = 16,
        eps: float = 1e-4
    ) -> 'np.ndarray':
        """Apply guided filter for alpha matting."""
        import cv2
        import numpy as np
        
        # Convert image to float
        if image.dtype == np.uint8:
            guide = image.astype(np.float32) / 255.0
        else:
            guide = image.astype(np.float32)
        
        # Use grayscale guide
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        
        # Apply guided filter
        try:
            # OpenCV's guided filter (if available)
            alpha = cv2.ximgproc.guidedFilter(
                guide, mask, radius, eps
            )
        except AttributeError:
            # Fallback: simple bilateral filter
            mask_uint8 = (mask * 255).astype(np.uint8)
            alpha_uint8 = cv2.bilateralFilter(mask_uint8, 9, 75, 75)
            alpha = alpha_uint8.astype(np.float32) / 255.0
        
        # Preserve known regions from trimap
        alpha = np.where(trimap > 0.9, 1.0, alpha)
        alpha = np.where(trimap < 0.1, 0.0, alpha)
        
        return alpha
    
    def _soften_edges(self, alpha: 'np.ndarray', softness: float) -> 'np.ndarray':
        """Apply edge softening for natural falloff."""
        import cv2
        import numpy as np
        
        # Find edges
        edges = cv2.Canny((alpha * 255).astype(np.uint8), 50, 150)
        
        # Dilate edges
        kernel_size = max(3, int(softness * 3))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edge_mask = cv2.dilate(edges, kernel, iterations=1)
        edge_mask = edge_mask.astype(np.float32) / 255.0
        
        # Apply gaussian blur to alpha at edges
        blur_size = max(3, int(softness * 5))
        if blur_size % 2 == 0:
            blur_size += 1
        
        alpha_blurred = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
        
        # Blend
        alpha_soft = alpha * (1 - edge_mask) + alpha_blurred * edge_mask
        
        return alpha_soft


# ==============================================================================
# SAM2 WRAPPER
# ==============================================================================

class SAM2Segmenter:
    """
    Wrapper for SAM2 video segmentation.
    
    Features:
        - Temporal consistency via memory bank
        - Multi-object tracking
        - Point, box, and mask prompts
    """
    
    MODEL_CONFIGS = {
        'tiny': ('sam2.1_hiera_tiny.pt', 'configs/sam2.1/sam2.1_hiera_t.yaml'),
        'small': ('sam2.1_hiera_small.pt', 'configs/sam2.1/sam2.1_hiera_s.yaml'),
        'base_plus': ('sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
        'large': ('sam2.1_hiera_large.pt', 'configs/sam2.1/sam2.1_hiera_l.yaml'),
    }
    
    CHECKPOINT_URLS = {
        'tiny': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        'small': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
        'base_plus': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'large': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
    }
    
    def __init__(
        self,
        model_size: str = 'large',
        device: str = 'cuda',
        compile_model: bool = True,
        logger: logging.Logger = None
    ):
        self.model_size = model_size
        self.device = device
        self.logger = logger or logging.getLogger("SAM2")

        # Auto-disable compile on Windows (torch.compile has known issues)
        if compile_model and platform.system() == "Windows":
            self.logger.info("Windows detected - disabling torch.compile for SAM2")
            compile_model = False

        self.compile_model = compile_model

        self.predictor = None
        self.state = None

        self._load_model()
    
    def _load_model(self):
        """Load SAM2 model."""
        import torch
        
        self.logger.info(f"Loading SAM2 {self.model_size} model...")
        
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            self.logger.error(
                "SAM2 not installed. Install with:\n"
                "  git clone https://github.com/facebookresearch/sam2.git\n"
                "  cd sam2 && pip install -e ."
            )
            raise
        
        checkpoint, config = self.MODEL_CONFIGS[self.model_size]

        # Check if checkpoint exists, download if not
        checkpoint_path = self._ensure_checkpoint(checkpoint)

        # Hydra expects relative config name, not absolute path
        # The config is searched in sam2's package config search path
        self.predictor = build_sam2_video_predictor(
            config,  # Just the relative config name like "configs/sam2.1/sam2.1_hiera_s.yaml"
            str(checkpoint_path),
            device=self.device,
            vos_optimized=self.compile_model
        )
        
        self.logger.info("SAM2 model loaded successfully")
    
    def _ensure_checkpoint(self, checkpoint: str) -> Path:
        """Ensure checkpoint exists, download if needed."""
        import urllib.request
        
        # Check common locations
        possible_paths = [
            Path(checkpoint),
            Path("checkpoints") / checkpoint,
            Path.home() / ".cache" / "sam2" / checkpoint,
        ]
        
        for p in possible_paths:
            if p.exists():
                return p
        
        # Download
        cache_dir = Path.home() / ".cache" / "sam2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = cache_dir / checkpoint
        url = self.CHECKPOINT_URLS[self.model_size]
        
        self.logger.info(f"Downloading {checkpoint}...")
        urllib.request.urlretrieve(url, target_path)
        self.logger.info("Download complete")
        
        return target_path
    
    def init_video(self, frames_dir: str):
        """Initialize video state from frames directory."""
        import torch
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.state = self.predictor.init_state(frames_dir)
    
    def add_point_prompt(
        self,
        frame_idx: int,
        points: List[Tuple[int, int]],
        labels: List[int],
        object_id: int = 1
    ):
        """
        Add point prompt(s) to a frame.
        
        Args:
            frame_idx: Frame to add prompt to
            points: List of (x, y) coordinates
            labels: List of labels (1 = foreground, 0 = background)
            object_id: ID for this object
        """
        import torch
        import numpy as np
        
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            _, _, masks = self.predictor.add_new_points_or_box(
                self.state,
                frame_idx=frame_idx,
                obj_id=object_id,
                points=points_np,
                labels=labels_np,
            )
        
        return masks
    
    def add_box_prompt(
        self,
        frame_idx: int,
        box: Tuple[int, int, int, int],
        object_id: int = 1
    ):
        """
        Add box prompt to a frame.
        
        Args:
            frame_idx: Frame to add prompt to
            box: (x1, y1, x2, y2) coordinates
            object_id: ID for this object
        """
        import torch
        import numpy as np
        
        box_np = np.array(box, dtype=np.float32)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            _, _, masks = self.predictor.add_new_points_or_box(
                self.state,
                frame_idx=frame_idx,
                obj_id=object_id,
                box=box_np,
            )
        
        return masks
    
    def propagate(self, reverse: bool = False):
        """
        Propagate masks through video.
        
        Yields:
            (frame_idx, object_ids, masks) tuples
        """
        import torch
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(
                self.state,
                reverse=reverse
            ):
                # Convert masks to numpy
                masks_np = masks.cpu().numpy()
                yield frame_idx, object_ids, masks_np

    def release(self) -> None:
        """Release model and free GPU memory."""
        try:
            if self.predictor is not None:
                del self.predictor
                self.predictor = None
            
            if self.state is not None:
                del self.state
                self.state = None
            
            self.logger.debug("SAM2Segmenter model released")
            
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
# GROUNDING DINO WRAPPER
# ==============================================================================

class GroundingDINODetector:
    """
    Wrapper for GroundingDINO text-to-bbox detection.
    
    Allows detecting objects via text prompts for automatic roto setup.
    """
    
    def __init__(
        self,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: str = 'cuda',
        logger: logging.Logger = None
    ):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.logger = logger or logging.getLogger("GroundingDINO")
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load GroundingDINO model."""
        self.logger.info("Loading GroundingDINO model...")
        
        try:
            from groundingdino.util.inference import load_model, predict
            self.predict_fn = predict
            
            # Load model
            # Check for model files
            config_path = self._find_config()
            checkpoint_path = self._find_checkpoint()
            
            self.model = load_model(config_path, checkpoint_path, device=self.device)
            self.logger.info("GroundingDINO loaded successfully")
            
        except ImportError:
            self.logger.warning(
                "GroundingDINO not installed. Text prompts won't be available.\n"
                "Install with:\n"
                "  git clone https://github.com/IDEA-Research/GroundingDINO.git\n"
                "  cd GroundingDINO && pip install -e ."
            )
            self.model = None
    
    def _find_config(self) -> str:
        """Find GroundingDINO config file."""
        possible = [
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ]
        
        for p in possible:
            if Path(p).exists():
                return p
        
        # Try to find in installed package
        try:
            import groundingdino
            pkg_dir = Path(groundingdino.__file__).parent
            config = pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
            if config.exists():
                return str(config)
        except:
            pass
        
        raise FileNotFoundError("GroundingDINO config not found")
    
    def _find_checkpoint(self) -> str:
        """Find or download GroundingDINO checkpoint."""
        import urllib.request
        
        checkpoint_name = "groundingdino_swint_ogc.pth"
        
        possible = [
            Path(checkpoint_name),
            Path("weights") / checkpoint_name,
            Path.home() / ".cache" / "groundingdino" / checkpoint_name,
        ]
        
        for p in possible:
            if p.exists():
                return str(p)
        
        # Download
        cache_dir = Path.home() / ".cache" / "groundingdino"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        target = cache_dir / checkpoint_name
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        
        self.logger.info(f"Downloading GroundingDINO checkpoint...")
        urllib.request.urlretrieve(url, target)
        
        return str(target)
    
    def detect(
        self,
        image: 'np.ndarray',
        prompt: str
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect objects matching text prompt.
        
        Args:
            image: RGB image (numpy array)
            prompt: Text description of objects to find
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        if self.model is None:
            raise RuntimeError("GroundingDINO not available")
        
        import torch
        import numpy as np
        from PIL import Image
        import groundingdino.datasets.transforms as T

        # Prepare image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Convert 16-bit to 8-bit if needed
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        elif image.dtype in (np.float32, np.float64):
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        image_transformed, _ = transform(pil_image, None)
        
        # Run detection
        boxes, logits, phrases = self.predict_fn(
            self.model,
            image_transformed,
            prompt,
            self.box_threshold,
            self.text_threshold,
            device=self.device
        )
        
        # Convert normalized boxes to pixel coordinates
        h, w = image.shape[:2]
        boxes_pixel = []
        
        for box in boxes:
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            boxes_pixel.append((x1, y1, x2, y2))
        
        self.logger.info(f"Detected {len(boxes_pixel)} objects for prompt: '{prompt}'")
        
        return boxes_pixel, phrases

    def release(self) -> None:
        """Release model and free GPU memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                self.predict_fn = None
            
            self.logger.debug("GroundingDINODetector model released")
            
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
# MAIN PIPELINE
# ==============================================================================

class AutoRotoPipeline:
    """
    Main auto-rotoscoping pipeline.
    
    Coordinates all components to extract clean alpha mattes from video.
    """
    
    def __init__(self, config: RotoConfig):
        self.config = config
        self.logger = setup_logging(config.verbose)
        
        # Components (lazy loaded)
        self._sam: Optional[SAM2Segmenter] = None
        self._detector: Optional[GroundingDINODetector] = None
        self._refiner: Optional[AlphaRefiner] = None
        self._reader: Optional[VideoReader] = None
        self._writer: Optional[FrameWriter] = None
    
    @property
    def sam(self) -> SAM2Segmenter:
        if self._sam is None:
            self._sam = SAM2Segmenter(
                model_size=self.config.sam_model,
                device=self.config.device,
                compile_model=self.config.compile_model,
                logger=self.logger
            )
        return self._sam
    
    @property
    def detector(self) -> GroundingDINODetector:
        if self._detector is None:
            self._detector = GroundingDINODetector(
                box_threshold=self.config.detection_threshold,
                text_threshold=self.config.text_threshold,
                device=self.config.device,
                logger=self.logger
            )
        return self._detector
    
    @property
    def refiner(self) -> AlphaRefiner:
        if self._refiner is None:
            self._refiner = AlphaRefiner(
                iterations=self.config.refine_iterations,
                edge_softness=self.config.edge_softness,
                logger=self.logger
            )
        return self._refiner
    
    def run(self):
        """Run the full pipeline."""
        import numpy as np
        import cv2
        from pathlib import Path
        import tempfile
        import shutil
        
        self.logger.info("="*60)
        self.logger.info("AUTO-ROTO Pipeline Started")
        self.logger.info("="*60)
        
        # Setup I/O
        self._reader = VideoReader(self.config.input_path, self.logger)
        
        self._writer = FrameWriter(
            output_dir=self.config.output_dir,
            prefix="roto",
            format=self.config.output_format,
            bit_depth=self.config.bit_depth,
            padding=self.config.frame_padding,
            logger=self.logger
        )
        
        # Extract frames to temp directory for SAM2
        temp_dir = Path(tempfile.mkdtemp(prefix="autoroto_"))
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        self.logger.info("Extracting frames...")
        frames = []
        for idx, frame in enumerate(self._reader):
            frame_path = frames_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frames.append(frame)
        
        self.logger.info(f"Extracted {len(frames)} frames")
        
        try:
            # Initialize SAM2 with video
            self.logger.info("Initializing SAM2...")
            self.sam.init_video(str(frames_dir))
            
            # Get initial prompt
            first_frame = frames[0]
            boxes = []
            
            if self.config.prompt:
                # Text-based detection
                self.logger.info(f"Detecting objects: '{self.config.prompt}'")
                boxes, phrases = self.detector.detect(first_frame, self.config.prompt)
                
            elif self.config.box:
                # Manual box
                coords = [int(x) for x in self.config.box.split(',')]
                boxes = [tuple(coords)]
                
            elif self.config.point:
                # Point prompt
                coords = [int(x) for x in self.config.point.split(',')]
                self.sam.add_point_prompt(
                    frame_idx=0,
                    points=[tuple(coords)],
                    labels=[1],
                    object_id=1
                )
                
            elif self.config.interactive:
                # Interactive selection
                boxes = self._interactive_selection(first_frame)
            
            else:
                raise ValueError("No prompt specified. Use --prompt, --box, --point, or --interactive")
            
            # Add box prompts to SAM2
            for obj_id, box in enumerate(boxes, start=1):
                self.logger.info(f"Adding object {obj_id}: box {box}")
                self.sam.add_box_prompt(
                    frame_idx=0,
                    box=box,
                    object_id=obj_id
                )
            
            # Propagate through video
            self.logger.info("Propagating masks through video...")
            
            all_masks = {}  # frame_idx -> combined mask
            
            # Forward propagation
            if self.config.propagate_forward:
                for frame_idx, obj_ids, masks in self.sam.propagate(reverse=False):
                    # Combine all object masks
                    # SAM2 outputs masks as logits (num_objects, 1, H, W), need sigmoid to convert to probabilities
                    combined = np.zeros(masks.shape[-2:], dtype=np.float32)
                    for mask in masks:
                        # Squeeze out any extra dimensions (e.g., channel dim)
                        mask_2d = mask.squeeze()
                        # Apply sigmoid to convert logits to probabilities (0-1)
                        mask_prob = 1.0 / (1.0 + np.exp(-mask_2d.astype(np.float32)))
                        combined = np.maximum(combined, mask_prob)
                    all_masks[frame_idx] = combined
                    
                    if frame_idx % 10 == 0:
                        self.logger.info(f"  Frame {frame_idx}/{len(frames)}")
            
            # Backward propagation
            if self.config.propagate_backward:
                self.logger.info("Backward propagation...")
                for frame_idx, obj_ids, masks in self.sam.propagate(reverse=True):
                    if frame_idx not in all_masks:
                        combined = np.zeros(masks.shape[-2:], dtype=np.float32)
                        for mask in masks:
                            mask_2d = mask.squeeze()
                            # Apply sigmoid to convert logits to probabilities (0-1)
                            mask_prob = 1.0 / (1.0 + np.exp(-mask_2d.astype(np.float32)))
                            combined = np.maximum(combined, mask_prob)
                        all_masks[frame_idx] = combined
            
            # Process and write frames
            self.logger.info("Writing output frames...")
            
            for frame_idx in sorted(all_masks.keys()):
                mask = all_masks[frame_idx]
                rgb = frames[frame_idx]
                
                # Refine alpha
                if self.config.refine_alpha:
                    alpha = self.refiner.refine(mask, rgb)
                else:
                    alpha = mask.astype(np.float32)
                
                # Write outputs
                self._writer.write_alpha(alpha, frame_idx)
                
                if self.config.include_rgb:
                    self._writer.write_rgba(rgb, alpha, frame_idx)
                
                if self.config.save_preview:
                    self._writer.write_preview(
                        rgb, alpha, frame_idx,
                        scale=self.config.preview_scale
                    )
                
                if frame_idx % 10 == 0:
                    self.logger.info(f"  Wrote frame {frame_idx}/{len(frames)}")
            
            self.logger.info("="*60)
            self.logger.info("Pipeline Complete!")
            self.logger.info(f"Output: {self.config.output_dir}")
            self.logger.info("="*60)
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _interactive_selection(self, frame: 'np.ndarray') -> List[Tuple[int, int, int, int]]:
        """Interactive box selection using OpenCV."""
        import cv2
        
        boxes = []
        current_box = []
        drawing = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_box, drawing
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                current_box = [x, y]
                
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                display = frame.copy()
                cv2.rectangle(display, tuple(current_box), (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Objects (ESC to finish)", display[:, :, ::-1])
                
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                current_box.extend([x, y])
                boxes.append(tuple(current_box))
                
                display = frame.copy()
                for box in boxes:
                    cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.imshow("Select Objects (ESC to finish)", display[:, :, ::-1])
        
        cv2.namedWindow("Select Objects (ESC to finish)")
        cv2.setMouseCallback("Select Objects (ESC to finish)", mouse_callback)
        cv2.imshow("Select Objects (ESC to finish)", frame[:, :, ::-1])
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        
        return boxes

    def release(self) -> None:
        """Release all resources and free GPU memory."""
        if self._sam is not None:
            try:
                self._sam.release()
            except Exception:
                pass
            self._sam = None
        
        if self._detector is not None:
            try:
                self._detector.release()
            except Exception:
                pass
            self._detector = None
        
        if self._refiner is not None:
            self._refiner = None
        
        if self._reader is not None:
            self._reader = None
        
        if self._writer is not None:
            self._writer = None
        
        # Clear GPU memory (defensive - may fail during shutdown)
        try:
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AUTO-ROTO: Production-Grade Automatic Rotoscoping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Detect and roto person in video
  %(prog)s --input video.mp4 --prompt "person" --output ./roto_output

  # Multiple objects
  %(prog)s --input video.mp4 --prompt "person.dog.car" --output ./output

  # Interactive box selection
  %(prog)s --input video.mp4 --interactive --output ./output

  # Manual box on first frame
  %(prog)s --input video.mp4 --box "100,100,500,400" --output ./output

  # Image sequence input
  %(prog)s --input ./frames/ --prompt "person" --output ./output
        """
    )
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input video or image sequence")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    
    # Prompt type
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", "-p", help="Text prompt for object detection (use . to separate multiple)")
    prompt_group.add_argument("--box", "-b", help="Box prompt: x1,y1,x2,y2")
    prompt_group.add_argument("--point", help="Point prompt: x,y")
    prompt_group.add_argument("--interactive", action="store_true", help="Interactive box selection")
    
    # SAM2 settings
    parser.add_argument("--sam-model", default="large", 
                       choices=["tiny", "small", "base_plus", "large"],
                       help="SAM2 model size")
    parser.add_argument("--no-backward", action="store_true", help="Disable backward propagation")
    
    # Detection settings
    parser.add_argument("--detection-threshold", type=float, default=0.3,
                       help="GroundingDINO detection threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                       help="GroundingDINO text threshold")
    
    # Alpha refinement
    parser.add_argument("--no-refine", action="store_true", help="Disable alpha refinement")
    parser.add_argument("--refine-iterations", type=int, default=3,
                       help="Alpha refinement iterations")
    parser.add_argument("--edge-softness", type=float, default=1.0,
                       help="Edge softness (0 = sharp, higher = softer)")
    
    # Output settings
    parser.add_argument("--format", default="exr", choices=["exr", "png", "tiff"],
                       help="Output format")
    parser.add_argument("--bit-depth", type=int, default=16, choices=[8, 16, 32],
                       help="Output bit depth")
    parser.add_argument("--no-rgb", action="store_true", help="Don't include RGB output")
    parser.add_argument("--no-preview", action="store_true", help="Don't generate previews")
    
    # Performance
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-compile", action="store_true", help="Disable model compilation")
    
    # Debug
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    _check_dependencies()
    
    args = parse_args()
    
    config = RotoConfig(
        input_path=args.input,
        output_dir=args.output,
        prompt=args.prompt,
        box=args.box,
        point=args.point,
        interactive=args.interactive,
        sam_model=args.sam_model,
        propagate_backward=not args.no_backward,
        detection_threshold=args.detection_threshold,
        text_threshold=args.text_threshold,
        refine_alpha=not args.no_refine,
        refine_iterations=args.refine_iterations,
        edge_softness=args.edge_softness,
        output_format=args.format,
        bit_depth=args.bit_depth,
        include_rgb=not args.no_rgb,
        device=args.device,
        compile_model=not args.no_compile,
        verbose=args.verbose,
        save_preview=not args.no_preview,
    )
    
    pipeline = AutoRotoPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
