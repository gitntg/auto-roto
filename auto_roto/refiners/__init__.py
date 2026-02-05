"""
Refinement Algorithms
=====================

Alpha matte refinement and enhancement algorithms.

Modules:
    - alpha: Basic alpha refinement (guided filter, morphological ops)
    - depth: Depth-guided alpha refinement
    - trimap: Trimap synthesis from masks and depth
    - geometric: Full geometric matte refinement pipeline
"""

from auto_roto.refiners.alpha import AlphaRefiner
from auto_roto.refiners.depth import DepthGuidedRefiner
from auto_roto.refiners.trimap import TrimapSynthesizer
from auto_roto.refiners.geometric import GeometricMatteRefiner

__all__ = [
    'AlphaRefiner',
    'DepthGuidedRefiner',
    'TrimapSynthesizer',
    'GeometricMatteRefiner',
]
