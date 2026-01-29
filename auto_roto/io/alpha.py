"""
Alpha Channel I/O
=================

Functions for reading and writing alpha matte files.
"""

import numpy as np
from pathlib import Path
from typing import Union, Generator, Tuple
import logging

logger = logging.getLogger("AutoRoto.IO.Alpha")


def save_alpha(
    alpha: np.ndarray,
    filepath: Union[str, Path],
    bit_depth: int = 16
) -> None:
    """
    Save alpha matte to file.

    Args:
        alpha: Alpha matte (H, W), values 0-1
        filepath: Output path (format determined by extension)
        bit_depth: Output bit depth (8, 16, or 32)
    """
    import cv2
    from auto_roto.io.exr import write_exr_channel

    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    # Ensure proper range
    alpha = np.clip(alpha, 0, 1).astype(np.float32)

    if ext == '.exr':
        write_exr_channel(alpha, filepath, channel='A', bit_depth=bit_depth)

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

    else:
        # Default to PNG-like behavior
        alpha_int = (alpha * 255).astype(np.uint8)
        cv2.imwrite(str(filepath), alpha_int)


def load_alpha(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load alpha matte from file.

    Args:
        filepath: Path to alpha file

    Returns:
        Alpha matte (H, W), float32 0-1

    Raises:
        IOError: If file cannot be read
    """
    import cv2
    from auto_roto.io.exr import read_exr_channel

    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == '.exr':
        return read_exr_channel(filepath, channel='A')

    # Use OpenCV for other formats
    alpha = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

    if alpha is None:
        raise IOError(f"Cannot read alpha: {filepath}")

    # Handle multi-channel images
    if len(alpha.shape) == 3:
        if alpha.shape[2] == 4:
            # Use alpha channel
            alpha = alpha[:, :, 3]
        else:
            # Convert to grayscale
            alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)

    # Normalize to 0-1
    if alpha.dtype == np.uint8:
        alpha = alpha.astype(np.float32) / 255.0
    elif alpha.dtype == np.uint16:
        alpha = alpha.astype(np.float32) / 65535.0
    else:
        alpha = alpha.astype(np.float32)

    return alpha


def load_alpha_sequence(
    alpha_dir: Union[str, Path]
) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """
    Load alpha images from directory.

    Args:
        alpha_dir: Directory containing alpha files

    Yields:
        Tuples of (index, alpha_array, filepath)
    """
    alpha_path = Path(alpha_dir)

    # Find all image files
    extensions = {'.exr', '.png', '.tif', '.tiff', '.jpg', '.jpeg'}
    files = []
    for ext in extensions:
        files.extend(alpha_path.glob(f"*{ext}"))
        files.extend(alpha_path.glob(f"*{ext.upper()}"))

    files = sorted(set(files))

    for idx, filepath in enumerate(files):
        try:
            alpha = load_alpha(filepath)
            yield idx, alpha, filepath
        except IOError as e:
            logger.warning(f"Skipping unreadable file: {filepath} ({e})")
            continue


def normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    """
    Normalize alpha values to 0-1 range.

    Args:
        alpha: Alpha matte (any range)

    Returns:
        Alpha matte normalized to 0-1
    """
    alpha = alpha.astype(np.float32)

    if alpha.max() > 1.0:
        if alpha.max() <= 255.0:
            alpha = alpha / 255.0
        elif alpha.max() <= 65535.0:
            alpha = alpha / 65535.0
        else:
            alpha = alpha / alpha.max()

    return np.clip(alpha, 0, 1)
