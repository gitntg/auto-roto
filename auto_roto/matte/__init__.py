"""
Matte Operations
================

Alpha matte processing, combination, and compositing.

Modules:
    - combiner: Combine alpha mattes with RGB to produce final output
    - despill: Remove color spill from semi-transparent edges
    - blender: Blend multiple alpha sources
"""

from auto_roto.matte.combiner import MatteCombiner
from auto_roto.matte.despill import Despill
from auto_roto.matte.blender import MatteBlender

__all__ = [
    'MatteCombiner',
    'Despill',
    'MatteBlender',
]
