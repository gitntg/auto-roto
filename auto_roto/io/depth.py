"""
Depth Map I/O
=============

Functions for reading and writing depth maps.
Depth maps are stored as float EXR for proper precision.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple
import logging

logger = logging.getLogger("AutoRoto.IO.Depth")


def save_depth_float(
    depth: np.ndarray,
    filepath: Union[str, Path]
) -> None:
    """
    Save depth map as float EXR (single Y channel) for proper precision.

    Args:
        depth: Depth map (H, W), float32
        filepath: Output path
    """
    filepath = Path(filepath)

    try:
        import OpenEXR
        import Imath

        h, w = depth.shape
        depth_float = depth.astype(np.float32)

        header = OpenEXR.Header(w, h)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        # Strictly single channel 'Y' for depth data
        header['channels'] = {'Y': Imath.Channel(pixel_type)}

        exr = OpenEXR.OutputFile(str(filepath), header)
        exr.writePixels({'Y': depth_float.tobytes()})
        exr.close()

    except ImportError:
        # Fallback to 16-bit PNG (grayscale)
        import cv2
        depth_16 = (depth * 65535).astype(np.uint16)
        cv2.imwrite(str(filepath), depth_16)


def load_depth_float(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load depth map as float values (raw DA3 depth).

    Args:
        filepath: Path to depth file

    Returns:
        Depth map (H, W), float32
    """
    import cv2

    filepath = Path(filepath)
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
            channels = list(header['channels'].keys())
            channel = 'Y' if 'Y' in channels else 'R' if 'R' in channels else channels[0]

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

    # Handle different formats - convert to float
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    return depth.astype(np.float32)


def save_depth_visualization(
    depth: np.ndarray,
    filepath: Union[str, Path],
    colormap: bool = True,
    percentiles: Tuple[float, float] = (2.0, 98.0)
) -> None:
    """
    Save depth map as colorized visualization (for preview only).

    Uses 16-bit output to minimize banding artifacts in gradients.

    Args:
        depth: Depth map (H, W), float
        filepath: Output path
        colormap: If True, apply inferno colormap
        percentiles: Percentiles for normalization
    """
    import cv2

    # Visualization only: normalize for display
    depth_float = depth.astype(np.float32)
    p_low, p_high = np.percentile(depth_float, list(percentiles))
    depth_vis_norm = np.clip((depth_float - p_low) / (p_high - p_low + 1e-8), 0, 1)

    if colormap:
        # Use matplotlib's colormap for smooth gradients
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            # Apply inferno colormap with full float precision
            cmap = cm.get_cmap('inferno')
            depth_colored = cmap(depth_vis_norm)  # Returns RGBA float [0,1]

            # Convert to BGR 16-bit for minimal banding (65536 levels per channel)
            depth_vis = (depth_colored[:, :, :3] * 65535).astype(np.uint16)
            depth_vis = depth_vis[:, :, ::-1]  # RGB to BGR

        except ImportError:
            # Fallback: grayscale 16-bit if matplotlib unavailable
            depth_vis = (depth_vis_norm * 65535).astype(np.uint16)
    else:
        # Grayscale 16-bit (no colormap)
        depth_vis = (depth_vis_norm * 65535).astype(np.uint16)

    cv2.imwrite(str(filepath), depth_vis)


def load_depth_map(depth_path: Union[str, Path]) -> np.ndarray:
    """
    Load a depth map from file (any format).

    Args:
        depth_path: Path to depth file

    Returns:
        Depth map (H, W), float32
    """
    import cv2

    depth_path = Path(depth_path)

    if depth_path.suffix.lower() == '.exr':
        return load_depth_float(depth_path)

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise IOError(f"Cannot read depth map: {depth_path}")

    # Handle different formats
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    return depth.astype(np.float32)


def normalize_depth(
    depth: np.ndarray,
    percentiles: Tuple[float, float] = (2.0, 98.0)
) -> np.ndarray:
    """
    Normalize depth map to 0-1 range using percentiles.

    Args:
        depth: Raw depth map
        percentiles: Low and high percentiles for normalization

    Returns:
        Normalized depth map (0-1)
    """
    depth_float = depth.astype(np.float32)
    p_low, p_high = np.percentile(depth_float, list(percentiles))
    normalized = np.clip((depth_float - p_low) / (p_high - p_low + 1e-8), 0, 1)
    return normalized.astype(np.float32)
