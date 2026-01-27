#!/usr/bin/env python
"""
AUTO-ROTO: Production-Grade Automatic Rotoscoping Pipeline
===========================================================

Uses SAM3 (Segment Anything Model 3) with built-in text prompting for
bulletproof automatic rotoscoping that generates perfect alphas from complex clips.

SAM3 has native Promptable Concept Segmentation (PCS) supporting 270k+ concepts,
eliminating the need for separate detection (GroundingDINO).

Author: Built for Core Form VFX
License: MIT

ARCHITECTURE:
    Input Video/Sequence
           │
           v
    ┌─────────────────┐
    │ Frame Extraction │
    └────────┬────────┘
             │
             v
    ┌─────────────────────────┐
    │         SAM3            │
    │  (unified text/box/     │
    │   point prompting)      │
    └────────┬────────────────┘
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
    # Text-based detection (SAM3 native - no GroundingDINO needed)
    python auto_roto.py --input video.mp4 --prompt "person" --output ./output

    # Interactive mode (click to select)
    python auto_roto.py --input video.mp4 --interactive --output ./output

    # Box prompt on first frame
    python auto_roto.py --input video.mp4 --box "100,100,500,400" --output ./output

    # Multiple objects
    python auto_roto.py --input video.mp4 --prompt "person.dog.car" --output ./output

REQUIREMENTS:
    - ultralytics >= 8.3.237
    - sam3.pt checkpoint (place in checkpoints/sam3/ or working directory)
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

# Environment verification - ensures correct conda environment
REQUIRED_ENV = "autoroto"

def _check_conda_environment():
    """Verify we're running in the correct conda environment."""
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != REQUIRED_ENV:
        print("\n" + "="*60)
        print("WRONG CONDA ENVIRONMENT")
        print("="*60)
        print(f"\n  Current environment: {current_env or '(none/base)'}")
        print(f"  Required environment: {REQUIRED_ENV}")
        print(f"\n  Please activate the correct environment:")
        print(f"    conda activate {REQUIRED_ENV}")
        print("\n" + "="*60)
        sys.exit(1)

# Run environment check immediately on import
_check_conda_environment()

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
# CONSTANTS
# ==============================================================================

# SAM3 inference resolution limits
SAM3_DEFAULT_IMGSZ = 1024           # Default processing resolution
SAM3_MAX_AUTO_IMGSZ = 2048          # Maximum auto-detected resolution (VRAM limited)
SAM3_MIN_IMGSZ = 640                # Minimum practical resolution

# Bit depth conversion
BIT_DEPTH_16_TO_8_DIVISOR = 256     # Divide 16-bit values by 256 for 8-bit

# Default thresholds
DEFAULT_SAM3_CONFIDENCE = 0.25      # Default detection confidence threshold
DEFAULT_SAM3_MAX_DETECTIONS = 100   # Maximum objects per frame


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class RotoConfig:
    """Configuration for the auto-roto pipeline using SAM3."""

    # Input/Output
    input_path: str = ""
    output_dir: str = "./output"

    # Detection mode
    prompt: Optional[str] = None  # Text prompt (SAM3 has built-in text prompting)
    box: Optional[str] = None     # Box prompt "x1,y1,x2,y2"
    point: Optional[str] = None   # Point prompt "x,y"
    interactive: bool = False     # Interactive selection mode

    # SAM3 settings (single model, no size choices like SAM2)
    propagate_forward: bool = True
    propagate_backward: bool = True

    # SAM3 inference settings
    sam_imgsz: int = 0              # Processing resolution (0 = auto from input, max 8192)
    sam_conf: float = 0.25          # Confidence threshold (0.0-1.0, lower = more detections)
    sam_retina_masks: bool = True   # High-resolution mask output
    sam_max_det: int = 100          # Maximum detections per frame

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

def normalize_frame_to_rgb(frame, cv2_module=None, np_module=None):
    """
    Normalize a frame to 8-bit RGB format.

    Handles:
        - 16-bit to 8-bit conversion
        - Float to uint8 conversion
        - Grayscale to RGB conversion
        - BGRA/BGR to RGB conversion

    Args:
        frame: Input frame (numpy array)
        cv2_module: Optional cv2 module (for lazy import)
        np_module: Optional numpy module (for lazy import)

    Returns:
        RGB uint8 numpy array
    """
    import cv2 as cv2_mod
    import numpy as np_mod
    cv2_mod = cv2_module or cv2_mod
    np_mod = np_module or np_mod

    # Convert to 8-bit if needed
    if frame.dtype == np_mod.uint16:
        frame = (frame / BIT_DEPTH_16_TO_8_DIVISOR).astype(np_mod.uint8)
    elif frame.dtype in (np_mod.float32, np_mod.float64):
        frame = (np_mod.clip(frame, 0, 1) * 255).astype(np_mod.uint8)

    # Convert to RGB
    if len(frame.shape) == 2:
        frame = cv2_mod.cvtColor(frame, cv2_mod.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGRA2RGB)
    else:
        frame = cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGR2RGB)

    return frame


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

        if self._is_sequence:
            for frame_path in self.frames:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                if frame is None:
                    self.logger.warning(f"Skipping unreadable frame: {frame_path}")
                    continue
                yield normalize_frame_to_rgb(frame)
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

        if self._is_sequence:
            if 0 <= idx < len(self.frames):
                frame = cv2.imread(str(self.frames[idx]), cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    return normalize_frame_to_rgb(frame)
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
# SAM3 WRAPPER (Ultralytics - with built-in text prompting)
# ==============================================================================

class SAM3Segmenter:
    """
    SAM3 segmenter with built-in text prompting via Ultralytics.

    SAM3 (released November 2025) has native Promptable Concept Segmentation (PCS)
    which allows direct text-to-segmentation without needing Grounding DINO.

    Features:
        - Built-in text prompting (no Grounding DINO needed)
        - Video tracking with temporal consistency
        - 270k+ trained concepts
        - Single-step inference pipeline

    Requirements:
        - ultralytics >= 8.3.237
        - sam3.pt checkpoint (download from HuggingFace after approval)
    """

    def __init__(
        self,
        model_path: str = "sam3.pt",
        device: str = "cuda",
        compile_model: bool = True,
        half_precision: bool = True,
        logger: logging.Logger = None,
        # Inference settings
        imgsz: int = 1024,
        conf: float = 0.25,
        retina_masks: bool = True,
        max_det: int = 100
    ):
        self.model_path = model_path
        self.device = device
        self.compile_model = compile_model
        self.half_precision = half_precision
        self.logger = logger or logging.getLogger("SAM3")

        # Inference settings
        self._imgsz_setting = imgsz  # 0 = auto
        self._resolved_imgsz = None  # Will be set based on first image
        self.conf = conf
        self.retina_masks = retina_masks
        self.max_det = max_det

        # Predictors (lazy loaded)
        self._image_predictor = None
        self._video_predictor = None

        # Check if sam3.pt exists
        self._verify_checkpoint()

    def _resolve_imgsz(self, image_shape: tuple) -> int:
        """
        Resolve imgsz from image dimensions.

        Args:
            image_shape: (H, W, C) or (H, W) tuple

        Returns:
            Resolved imgsz (max dimension, with practical GPU memory limits)
        """
        if self._imgsz_setting > 0:
            return self._imgsz_setting

        # Auto mode: use larger dimension with practical limits
        # SAM3 attention layers scale quadratically with image size
        # Practical limits based on VRAM:
        #   - 8GB VRAM:  ~1024 max
        #   - 12GB VRAM: ~1536 max
        #   - 24GB VRAM: ~2048 max
        #   - 48GB VRAM: ~3072 max
        # Default cap for RTX 4090 class GPUs

        h, w = image_shape[:2]
        max_dim = max(h, w)
        resolved = min(max_dim, SAM3_MAX_AUTO_IMGSZ)

        self.logger.info(f"Auto imgsz: input={w}x{h}, using imgsz={resolved} (max={SAM3_MAX_AUTO_IMGSZ})")
        return resolved

    @property
    def imgsz(self) -> int:
        """Get resolved imgsz (may be 0 if not yet resolved)."""
        return self._resolved_imgsz if self._resolved_imgsz else self._imgsz_setting

    def _verify_checkpoint(self):
        """Verify SAM3 checkpoint exists."""
        from pathlib import Path

        possible_paths = [
            Path(self.model_path),
            Path("checkpoints/sam3") / Path(self.model_path).name,  # NEW: checkpoints/sam3/sam3.pt
            Path("checkpoints") / self.model_path,
            Path.home() / ".cache" / "sam3" / self.model_path,
        ]

        for p in possible_paths:
            if p.exists():
                self.model_path = str(p)
                self.logger.info(f"Found SAM3 checkpoint: {p}")
                return

        self.logger.warning(
            f"SAM3 checkpoint '{self.model_path}' not found.\n"
            "Download from HuggingFace (requires approval):\n"
            "  1. Request access at https://huggingface.co/facebook/sam3\n"
            "  2. Download sam3.pt after approval\n"
            "  3. Place in working directory or checkpoints/sam3/"
        )

    @property
    def image_predictor(self):
        """Lazy-load image predictor."""
        if self._image_predictor is None:
            self.logger.info("Loading SAM3 image predictor...")
            self.logger.info(f"  imgsz={self.imgsz}, conf={self.conf}, retina_masks={self.retina_masks}")
            try:
                from ultralytics.models.sam import SAM3SemanticPredictor

                overrides = dict(
                    conf=self.conf,
                    imgsz=self.imgsz,
                    retina_masks=self.retina_masks,
                    max_det=self.max_det,
                    task="segment",
                    mode="predict",
                    model=self.model_path,
                    half=self.half_precision,
                    device=self.device,
                )
                self._image_predictor = SAM3SemanticPredictor(overrides=overrides)
                self.logger.info("SAM3 image predictor loaded")
            except ImportError as e:
                raise ImportError(
                    "SAM3 requires ultralytics >= 8.3.237. Install with:\n"
                    "  pip install -U ultralytics>=8.3.237"
                ) from e
        return self._image_predictor

    @property
    def video_predictor(self):
        """Lazy-load video predictor."""
        if self._video_predictor is None:
            self.logger.info("Loading SAM3 video predictor...")
            self.logger.info(f"  imgsz={self.imgsz}, conf={self.conf}, retina_masks={self.retina_masks}")
            try:
                from ultralytics.models.sam import SAM3VideoSemanticPredictor

                overrides = dict(
                    conf=self.conf,
                    imgsz=self.imgsz,
                    retina_masks=self.retina_masks,
                    max_det=self.max_det,
                    task="segment",
                    mode="predict",
                    model=self.model_path,
                    half=self.half_precision,
                    device=self.device,
                    save=False,  # We handle saving ourselves
                )
                self._video_predictor = SAM3VideoSemanticPredictor(overrides=overrides)
                self.logger.info("SAM3 video predictor loaded")
            except ImportError as e:
                raise ImportError(
                    "SAM3 requires ultralytics >= 8.3.237. Install with:\n"
                    "  pip install -U ultralytics>=8.3.237"
                ) from e
        return self._video_predictor

    def segment_image_with_text(
        self,
        image: 'np.ndarray',
        text_prompts: List[str]
    ) -> List['np.ndarray']:
        """
        Segment image using text prompts.

        Args:
            image: RGB image (numpy array)
            text_prompts: List of text descriptions (e.g., ["person", "dog"])

        Returns:
            List of binary masks for each detected concept
        """
        import numpy as np

        self.logger.info(f"Segmenting with text prompts: {text_prompts}")

        # Resolve imgsz from image dimensions (before predictor init)
        if self._resolved_imgsz is None:
            self._resolved_imgsz = self._resolve_imgsz(image.shape)

        # Set image
        self.image_predictor.set_image(image)

        # Run prediction (settings already configured in predictor)
        results = self.image_predictor(text=text_prompts)

        # Extract masks
        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    masks.append(mask.cpu().numpy())

        self.logger.info(f"Found {len(masks)} masks")
        return masks

    def segment_image_with_box(
        self,
        image: 'np.ndarray',
        boxes: List[Tuple[int, int, int, int]]
    ) -> List['np.ndarray']:
        """
        Segment image using bounding box prompts.

        Args:
            image: RGB image (numpy array)
            boxes: List of bounding boxes as (x1, y1, x2, y2)

        Returns:
            List of binary masks for each box
        """
        import numpy as np

        self.logger.info(f"Segmenting with {len(boxes)} box prompts")

        # Set image
        self.image_predictor.set_image(image)

        # Convert boxes to numpy array
        bboxes = np.array(boxes, dtype=np.float32)

        # Run prediction with bboxes
        results = self.image_predictor(bboxes=bboxes)

        # Extract masks
        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    masks.append(mask.cpu().numpy())

        self.logger.info(f"Found {len(masks)} masks from box prompts")
        return masks

    def segment_image_with_points(
        self,
        image: 'np.ndarray',
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> List['np.ndarray']:
        """
        Segment image using point prompts.

        Args:
            image: RGB image (numpy array)
            points: List of (x, y) coordinates
            labels: List of labels (1 = foreground, 0 = background)

        Returns:
            List of binary masks
        """
        import numpy as np

        self.logger.info(f"Segmenting with {len(points)} point prompts")

        # Set image
        self.image_predictor.set_image(image)

        # Convert to numpy arrays
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)

        # Run prediction with points
        results = self.image_predictor(points=points_np, labels=labels_np)

        # Extract masks
        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    masks.append(mask.cpu().numpy())

        self.logger.info(f"Found {len(masks)} masks from point prompts")
        return masks

    def segment_video_with_box(
        self,
        video_path: str,
        boxes: List[Tuple[int, int, int, int]],
        output_masks: bool = True
    ):
        """
        Segment video using bounding box prompts with temporal tracking.

        Args:
            video_path: Path to video file or frames directory
            boxes: List of bounding boxes as (x1, y1, x2, y2)
            output_masks: Whether to yield masks for each frame

        Yields:
            (frame_idx, masks) tuples
        """
        import numpy as np

        self.logger.info(f"Processing video with box prompts: {video_path}")
        self.logger.info(f"Boxes: {boxes}")

        # Convert boxes to numpy array
        bboxes = np.array(boxes, dtype=np.float32)

        # Run video prediction with streaming
        results = self.video_predictor(
            source=video_path,
            bboxes=bboxes,
            stream=True
        )

        frame_idx = 0
        for result in results:
            if output_masks and result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()

                if len(masks_data) > 0:
                    combined = np.zeros(masks_data.shape[-2:], dtype=np.float32)
                    for mask in masks_data:
                        combined = np.maximum(combined, mask.astype(np.float32))
                    yield frame_idx, combined
                else:
                    yield frame_idx, None
            else:
                yield frame_idx, None

            frame_idx += 1

            if frame_idx % 10 == 0:
                self.logger.info(f"  Processed frame {frame_idx}")

    def segment_video_with_text(
        self,
        video_path: str,
        text_prompts: List[str],
        output_masks: bool = True
    ):
        """
        Segment video using text prompts with temporal tracking.

        This is the main method for video rotoscoping with SAM3.

        Args:
            video_path: Path to video file or frames directory
            text_prompts: List of text descriptions (e.g., ["person"])
            output_masks: Whether to yield masks for each frame

        Yields:
            (frame_idx, masks) tuples where masks is a dict of concept->mask
        """
        import numpy as np
        from pathlib import Path

        path = Path(video_path)

        # If it's a directory of frames, use frame-by-frame processing
        if path.is_dir():
            self.logger.info(f"Processing frames directory: {video_path}")
            yield from self.segment_frames_with_text(video_path, text_prompts)
            return

        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Text prompts: {text_prompts}")

        # Run video prediction with streaming
        results = self.video_predictor(
            source=video_path,
            text=text_prompts,
            stream=True
        )

        frame_idx = 0
        for result in results:
            if output_masks and result.masks is not None:
                # Combine all masks for this frame
                masks_data = result.masks.data.cpu().numpy()

                # Return combined mask (union of all detected objects)
                if len(masks_data) > 0:
                    combined = np.zeros(masks_data.shape[-2:], dtype=np.float32)
                    for mask in masks_data:
                        combined = np.maximum(combined, mask.astype(np.float32))
                    yield frame_idx, combined
                else:
                    yield frame_idx, None
            else:
                yield frame_idx, None

            frame_idx += 1

            if frame_idx % 10 == 0:
                self.logger.info(f"  Processed frame {frame_idx}")

    def segment_frames_with_text(
        self,
        frames_dir: str,
        text_prompts: List[str]
    ):
        """
        Segment a directory of frames using text prompts.

        Args:
            frames_dir: Path to directory containing frame images
            text_prompts: List of text descriptions

        Yields:
            (frame_idx, mask) tuples
        """
        from pathlib import Path
        import cv2
        import numpy as np

        frames_path = Path(frames_dir)

        # Find all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        frame_files = []
        for ext in extensions:
            frame_files.extend(sorted(frames_path.glob(ext)))

        if not frame_files:
            raise ValueError(f"No image files found in {frames_dir}")

        self.logger.info(f"Processing {len(frame_files)} frames from {frames_dir}")

        for idx, frame_path in enumerate(frame_files):
            # Read frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                self.logger.warning(f"Could not read frame: {frame_path}")
                yield idx, None
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Segment with text
            masks = self.segment_image_with_text(frame, text_prompts)

            # Combine masks
            if masks:
                combined = np.zeros(frame.shape[:2], dtype=np.float32)
                for mask in masks:
                    combined = np.maximum(combined, mask.astype(np.float32))
                yield idx, combined
            else:
                yield idx, None

            if idx % 10 == 0:
                self.logger.info(f"  Processed frame {idx}/{len(frame_files)}")

    def release(self) -> None:
        """Release models and free GPU memory."""
        try:
            if self._image_predictor is not None:
                del self._image_predictor
                self._image_predictor = None

            if self._video_predictor is not None:
                del self._video_predictor
                self._video_predictor = None

            self.logger.debug("SAM3Segmenter models released")

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
    Main auto-rotoscoping pipeline using SAM3.

    SAM3 provides built-in text prompting with 270k+ concepts,
    eliminating the need for separate detection (GroundingDINO).

    Coordinates all components to extract clean alpha mattes from video.
    """

    def __init__(self, config: RotoConfig):
        self.config = config
        self.logger = setup_logging(config.verbose)

        # Components (lazy loaded)
        self._sam: Optional[SAM3Segmenter] = None
        self._refiner: Optional[AlphaRefiner] = None
        self._reader: Optional[VideoReader] = None
        self._writer: Optional[FrameWriter] = None

    @property
    def sam(self) -> SAM3Segmenter:
        """SAM3 segmenter with built-in text prompting."""
        if self._sam is None:
            self._sam = SAM3Segmenter(
                model_path="sam3.pt",
                device=self.config.device,
                compile_model=self.config.compile_model,
                logger=self.logger,
                # Inference settings from config
                imgsz=self.config.sam_imgsz,
                conf=self.config.sam_conf,
                retina_masks=self.config.sam_retina_masks,
                max_det=self.config.sam_max_det
            )
        return self._sam

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
        """Run the full pipeline using SAM3."""
        import numpy as np
        import cv2
        from pathlib import Path
        import tempfile
        import shutil

        self.logger.info("="*60)
        self.logger.info("AUTO-ROTO Pipeline Started (SAM3)")
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

        # Extract frames to temp directory
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
            all_masks = {}  # frame_idx -> combined mask

            if self.config.prompt:
                # Text prompt - SAM3 handles this natively (no GroundingDINO needed)
                text_prompts = self.config.prompt.split(".")
                self.logger.info(f"Processing with SAM3 text prompts: {text_prompts}")

                # Use SAM3 video predictor for temporal consistency
                for frame_idx, mask in self.sam.segment_video_with_text(
                    str(frames_dir), text_prompts
                ):
                    if mask is not None:
                        all_masks[frame_idx] = mask

                    if frame_idx % 10 == 0:
                        self.logger.info(f"  Frame {frame_idx}/{len(frames)}")

            elif self.config.box:
                # Box prompt
                coords = [int(x) for x in self.config.box.split(',')]
                boxes = [tuple(coords)]
                self.logger.info(f"Processing with box prompt: {boxes}")

                for frame_idx, mask in self.sam.segment_video_with_box(
                    str(frames_dir), boxes
                ):
                    if mask is not None:
                        all_masks[frame_idx] = mask

                    if frame_idx % 10 == 0:
                        self.logger.info(f"  Frame {frame_idx}/{len(frames)}")

            elif self.config.point:
                # Point prompt - process frame by frame
                coords = [int(x) for x in self.config.point.split(',')]
                points = [tuple(coords)]
                labels = [1]  # Foreground
                self.logger.info(f"Processing with point prompt: {points}")

                for idx, frame in enumerate(frames):
                    masks = self.sam.segment_image_with_points(frame, points, labels)
                    if masks:
                        combined = np.zeros(frame.shape[:2], dtype=np.float32)
                        for m in masks:
                            combined = np.maximum(combined, m.astype(np.float32))
                        all_masks[idx] = combined

                    if idx % 10 == 0:
                        self.logger.info(f"  Frame {idx}/{len(frames)}")

            elif self.config.interactive:
                # Interactive box selection
                first_frame = frames[0]
                boxes = self._interactive_selection(first_frame)
                self.logger.info(f"Processing with interactive boxes: {boxes}")

                for frame_idx, mask in self.sam.segment_video_with_box(
                    str(frames_dir), boxes
                ):
                    if mask is not None:
                        all_masks[frame_idx] = mask

                    if frame_idx % 10 == 0:
                        self.logger.info(f"  Frame {frame_idx}/{len(frames)}")

            else:
                raise ValueError("No prompt specified. Use --prompt, --box, --point, or --interactive")

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
        description="AUTO-ROTO: Production-Grade Automatic Rotoscoping with SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Detect and roto person in video (SAM3 has built-in text prompting)
  %(prog)s --input video.mp4 --prompt "person" --output ./roto_output

  # Multiple objects
  %(prog)s --input video.mp4 --prompt "person.dog.car" --output ./output

  # Interactive box selection
  %(prog)s --input video.mp4 --interactive --output ./output

  # Manual box on first frame
  %(prog)s --input video.mp4 --box "100,100,500,400" --output ./output

  # Image sequence input
  %(prog)s --input ./frames/ --prompt "person" --output ./output

NOTE: SAM3 includes built-in text prompting (270k+ concepts).
      No separate GroundingDINO required.
        """
    )

    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input video or image sequence")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")

    # Prompt type
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", "-p", help="Text prompt for SAM3 (use . to separate multiple)")
    prompt_group.add_argument("--box", "-b", help="Box prompt: x1,y1,x2,y2")
    prompt_group.add_argument("--point", help="Point prompt: x,y")
    prompt_group.add_argument("--interactive", action="store_true", help="Interactive box selection")

    # SAM3 settings (no model size choices - SAM3 has single architecture)
    parser.add_argument("--no-backward", action="store_true", help="Disable backward propagation")

    # SAM3 inference settings
    parser.add_argument("--sam-imgsz", type=int, default=0,
                       help="SAM3 processing resolution (default: 0 = auto from input, max 8192)")
    parser.add_argument("--sam-conf", type=float, default=0.25,
                       help="SAM3 confidence threshold (default: 0.25, lower = more detections)")
    parser.add_argument("--no-retina-masks", action="store_true",
                       help="Disable high-resolution mask output")
    parser.add_argument("--sam-max-det", type=int, default=100,
                       help="Maximum detections per frame (default: 100)")

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
        propagate_backward=not args.no_backward,
        # SAM3 inference settings
        sam_imgsz=args.sam_imgsz,
        sam_conf=args.sam_conf,
        sam_retina_masks=not args.no_retina_masks,
        sam_max_det=args.sam_max_det,
        # Alpha refinement
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
