"""
Color Space Utilities
=====================

Conversions between sRGB and linear color space.

From SOURCEOFTRUTH.md:
"Hair is dark. In sRGB (Gamma 2.2), the difference between dark hair and
darker background is compressed into 2-3 integer values. Process in Linear
Color Space (Gamma 1.0) to see sub-pixel differences."
"""

import numpy as np


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """
    Convert sRGB image to linear color space.

    This is essential for proper color processing in matting workflows.
    Dark regions (like hair) have much better tonal separation in linear space.

    Args:
        img: sRGB image, float32 range [0, 1]

    Returns:
        Linear color space image, float32 range [0, 1]
    """
    # Standard sRGB to linear conversion
    # For values <= 0.04045: linear = srgb / 12.92
    # For values > 0.04045: linear = ((srgb + 0.055) / 1.055) ^ 2.4
    img = np.clip(img, 0, 1)
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((img + 0.055) / 1.055, 2.4)
    )
    return linear.astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """
    Convert linear color space image back to sRGB.

    Args:
        img: Linear image, float32 range [0, 1]

    Returns:
        sRGB image, float32 range [0, 1]
    """
    # Standard linear to sRGB conversion
    # For values <= 0.0031308: srgb = linear * 12.92
    # For values > 0.0031308: srgb = 1.055 * (linear ^ (1/2.4)) - 0.055
    img = np.clip(img, 0, 1)
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(img, 1/2.4) - 0.055
    )
    return srgb.astype(np.float32)


def gamma_correct(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Apply gamma correction to an image.

    Args:
        img: Input image, float32 range [0, 1]
        gamma: Gamma value (default 2.2 for sRGB-like correction)

    Returns:
        Gamma-corrected image
    """
    img = np.clip(img, 0, 1)
    return np.power(img, gamma).astype(np.float32)


def inverse_gamma(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Apply inverse gamma correction (linearize).

    Args:
        img: Input image with gamma encoding, float32 range [0, 1]
        gamma: Gamma value to remove

    Returns:
        Linearized image
    """
    img = np.clip(img, 0, 1)
    return np.power(img, 1.0 / gamma).astype(np.float32)
