"""
EXR File Format Utilities
=========================

High-level utilities for reading and writing EXR files.
Supports both OpenEXR library and OpenCV fallback.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
import logging

logger = logging.getLogger("AutoRoto.IO.EXR")


def read_exr_channel(
    filepath: Union[str, Path],
    channel: str = None
) -> np.ndarray:
    """
    Read a single channel from an EXR file.

    Args:
        filepath: Path to EXR file
        channel: Channel name ('A', 'Y', 'R', 'G', 'B') or None for auto-detect

    Returns:
        2D numpy array (float32)

    Raises:
        IOError: If file cannot be read
    """
    filepath = Path(filepath)

    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(filepath))
        header = exr_file.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        channels = list(header['channels'].keys())

        # Auto-detect channel if not specified
        if channel is None:
            for ch in ['A', 'Y', 'R', 'G', 'B']:
                if ch in channels:
                    channel = ch
                    break
            if channel is None and channels:
                channel = channels[0]

        if channel not in channels:
            raise ValueError(f"Channel '{channel}' not found in {filepath}. Available: {channels}")

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        data = exr_file.channel(channel, pt)
        result = np.frombuffer(data, dtype=np.float32).reshape((height, width))
        return result

    except ImportError:
        # Fallback to OpenCV
        import cv2
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read EXR: {filepath}")

        if len(img.shape) == 3:
            if channel == 'A' and img.shape[2] == 4:
                return img[:, :, 3].astype(np.float32)
            elif channel in ['R', 'Y']:
                return img[:, :, 0].astype(np.float32)
            elif channel == 'G':
                return img[:, :, 1].astype(np.float32)
            elif channel == 'B':
                return img[:, :, 2].astype(np.float32)
            else:
                return img[:, :, 0].astype(np.float32)
        return img.astype(np.float32)


def write_exr_channel(
    data: np.ndarray,
    filepath: Union[str, Path],
    channel: str = 'A',
    bit_depth: int = 16
) -> None:
    """
    Write a single channel to an EXR file.

    Args:
        data: 2D numpy array (H, W)
        filepath: Output path
        channel: Channel name ('A', 'Y', 'R', etc.)
        bit_depth: 16 for half float, 32 for full float
    """
    filepath = Path(filepath)

    try:
        import OpenEXR
        import Imath

        h, w = data.shape

        if bit_depth == 32:
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            data_bytes = data.astype(np.float32).tobytes()
        else:
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            data_bytes = data.astype(np.float16).tobytes()

        header = OpenEXR.Header(w, h)
        header['channels'] = {channel: Imath.Channel(pixel_type)}

        exr = OpenEXR.OutputFile(str(filepath), header)
        exr.writePixels({channel: data_bytes})
        exr.close()

    except ImportError:
        # Fallback to OpenCV
        import cv2
        cv2.imwrite(str(filepath), data.astype(np.float32))


def read_exr_rgb(filepath: Union[str, Path]) -> np.ndarray:
    """
    Read RGB channels from an EXR file.

    Args:
        filepath: Path to EXR file

    Returns:
        3D numpy array (H, W, 3), float32
    """
    filepath = Path(filepath)

    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(filepath))
        header = exr_file.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        channels = list(header['channels'].keys())
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        if 'R' in channels and 'G' in channels and 'B' in channels:
            r = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape((height, width))
            g = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape((height, width))
            b = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape((height, width))
            return np.stack([r, g, b], axis=-1)
        elif 'Y' in channels:
            # Grayscale - replicate to RGB
            y = np.frombuffer(exr_file.channel('Y', pt), dtype=np.float32).reshape((height, width))
            return np.stack([y, y, y], axis=-1)
        else:
            raise ValueError(f"No RGB or Y channels in {filepath}")

    except ImportError:
        # Fallback to OpenCV
        import cv2
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read EXR: {filepath}")
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def write_exr_rgb(
    data: np.ndarray,
    filepath: Union[str, Path],
    bit_depth: int = 16
) -> None:
    """
    Write RGB data to an EXR file.

    Args:
        data: 3D numpy array (H, W, 3)
        filepath: Output path
        bit_depth: 16 for half float, 32 for full float
    """
    filepath = Path(filepath)

    try:
        import OpenEXR
        import Imath

        h, w = data.shape[:2]

        if bit_depth == 32:
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            r_bytes = data[:, :, 0].astype(np.float32).tobytes()
            g_bytes = data[:, :, 1].astype(np.float32).tobytes()
            b_bytes = data[:, :, 2].astype(np.float32).tobytes()
        else:
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            r_bytes = data[:, :, 0].astype(np.float16).tobytes()
            g_bytes = data[:, :, 1].astype(np.float16).tobytes()
            b_bytes = data[:, :, 2].astype(np.float16).tobytes()

        header = OpenEXR.Header(w, h)
        header['channels'] = {
            'R': Imath.Channel(pixel_type),
            'G': Imath.Channel(pixel_type),
            'B': Imath.Channel(pixel_type),
        }

        exr = OpenEXR.OutputFile(str(filepath), header)
        exr.writePixels({'R': r_bytes, 'G': g_bytes, 'B': b_bytes})
        exr.close()

    except ImportError:
        # Fallback to OpenCV
        import cv2
        bgr = cv2.cvtColor(data.astype(np.float32), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), bgr)


def write_exr_rgba(
    rgb: np.ndarray,
    alpha: np.ndarray,
    filepath: Union[str, Path],
    bit_depth: int = 16
) -> None:
    """
    Write RGBA data to an EXR file.

    Args:
        rgb: 3D numpy array (H, W, 3)
        alpha: 2D numpy array (H, W)
        filepath: Output path
        bit_depth: 16 for half float, 32 for full float
    """
    filepath = Path(filepath)

    try:
        import OpenEXR
        import Imath

        h, w = rgb.shape[:2]

        if bit_depth == 32:
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            r_bytes = rgb[:, :, 0].astype(np.float32).tobytes()
            g_bytes = rgb[:, :, 1].astype(np.float32).tobytes()
            b_bytes = rgb[:, :, 2].astype(np.float32).tobytes()
            a_bytes = alpha.astype(np.float32).tobytes()
        else:
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            r_bytes = rgb[:, :, 0].astype(np.float16).tobytes()
            g_bytes = rgb[:, :, 1].astype(np.float16).tobytes()
            b_bytes = rgb[:, :, 2].astype(np.float16).tobytes()
            a_bytes = alpha.astype(np.float16).tobytes()

        header = OpenEXR.Header(w, h)
        header['channels'] = {
            'R': Imath.Channel(pixel_type),
            'G': Imath.Channel(pixel_type),
            'B': Imath.Channel(pixel_type),
            'A': Imath.Channel(pixel_type),
        }

        exr = OpenEXR.OutputFile(str(filepath), header)
        exr.writePixels({'R': r_bytes, 'G': g_bytes, 'B': b_bytes, 'A': a_bytes})
        exr.close()

    except ImportError:
        # Fallback to OpenCV
        import cv2
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
        rgba[:, :, 3] = alpha
        cv2.imwrite(str(filepath), rgba)
