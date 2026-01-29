"""
Command Line Interfaces
=======================

CLI modules for auto-roto commands.

Modules:
    - common: Shared CLI utilities
    - sam: SAM segmentation CLI (auto_roto command)
    - depth: Depth estimation CLI (depth_refine command)
    - pipeline: Full pipeline CLI (full_pipeline command)
"""

from auto_roto.cli.common import (
    create_base_parser,
    add_output_args,
    add_performance_args,
    add_debug_args,
)

__all__ = [
    'create_base_parser',
    'add_output_args',
    'add_performance_args',
    'add_debug_args',
]
