"""
File I/O Operations
===================

Modules for reading and writing video, image, and alpha data.

Modules:
    video - VideoReader for video files and image sequences
    frames - FrameWriter for output alpha/rgba/preview frames
    alpha - Alpha channel I/O utilities
    depth - Depth map I/O utilities
    exr - EXR file format utilities
"""

from auto_roto.io.alpha import save_alpha, load_alpha, load_alpha_sequence
from auto_roto.io.depth import save_depth_float, load_depth_float, save_depth_visualization
from auto_roto.io.exr import read_exr_channel, write_exr_channel, read_exr_rgb

__all__ = [
    # Alpha I/O
    "save_alpha",
    "load_alpha",
    "load_alpha_sequence",
    # Depth I/O
    "save_depth_float",
    "load_depth_float",
    "save_depth_visualization",
    # EXR utilities
    "read_exr_channel",
    "write_exr_channel",
    "read_exr_rgb",
]
