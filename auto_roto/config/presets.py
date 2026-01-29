"""
Quality Presets
===============

Predefined quality settings for different use cases.
"""

from typing import Dict, Any


QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    'draft': {
        'depth_model': 'small',
        # DA3 settings: fast, basic detail
        'depth_process_res': None,  # auto (image size)
        'depth_process_method': 'upper',
        'depth_norm_percentiles': (2.0, 98.0),
        # Hair polish: more aggressive for speed
        'hair_gamma': 0.7,
        'hair_black_point': 0.03,
        'hair_gain': 1.2,
        # Guided Filter: fast, general purpose
        'guided_filter_radius': 4,
        'guided_filter_eps': 1e-4,
    },
    'standard': {
        'depth_model': 'base',
        # DA3 settings: balanced - explicit 1536 for reasonable quality/speed
        'depth_process_res': 1536,
        'depth_process_method': 'upper',
        'depth_norm_percentiles': (2.0, 98.0),
        # Hair polish: balanced
        'hair_gamma': 0.8,
        'hair_black_point': 0.02,
        'hair_gain': 1.1,
        # Guided Filter: balanced
        'guided_filter_radius': 4,
        'guided_filter_eps': 1e-5,
    },
    'high': {
        'depth_model': 'large',  # DA3Mono-Large preserves hair detail
        # DA3 settings: optimized for fine detail
        'depth_process_res': 2048,
        'depth_process_method': 'lower',  # process_res is min dimension
        'depth_norm_percentiles': (1.0, 99.0),  # Wider range preserves more detail
        # Hair polish: preserve detail
        'hair_gamma': 0.9,
        'hair_black_point': 0.01,
        'hair_gain': 1.05,
        # Guided Filter: better hair strand separation
        'guided_filter_radius': 2,
        'guided_filter_eps': 1e-5,
    },
    'ultra': {
        'depth_model': 'large',  # DA3Mono-Large preserves hair detail
        # DA3 settings: maximum detail capture
        'depth_process_res': 2048,
        'depth_process_method': 'lower',  # process_res is min dimension
        'depth_norm_percentiles': (0.0, 100.0),  # Full range to preserve depth variation
        # Hair polish: no adjustment (preserve all detail)
        'hair_gamma': 1.0,
        'hair_black_point': 0.0,
        'hair_gain': 1.0,
        # Guided Filter: maximum individual strand definition
        'guided_filter_radius': 1,
        'guided_filter_eps': 1e-6,
    }
}


def get_quality_preset(quality: str) -> Dict[str, Any]:
    """
    Get model sizes and settings for quality preset.

    Args:
        quality: One of 'draft', 'standard', 'high', 'ultra'

    Returns:
        Dictionary of preset settings

    Note: SAM3 has a single model architecture, so no sam_model setting needed.
    """
    return QUALITY_PRESETS.get(quality, QUALITY_PRESETS['standard'])


def list_presets() -> list:
    """Return list of available preset names."""
    return list(QUALITY_PRESETS.keys())


def get_preset_description(quality: str) -> str:
    """Get human-readable description of a preset."""
    descriptions = {
        'draft': "Fast preview quality - basic detail, quick processing",
        'standard': "Balanced quality - good detail, reasonable speed",
        'high': "High quality - fine hair detail, slower processing",
        'ultra': "Maximum quality - best hair preservation, slowest processing",
    }
    return descriptions.get(quality, "Unknown preset")
