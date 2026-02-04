"""
Quality Presets
===============

Predefined quality settings for different use cases.

Contains two preset types:
    - QUALITY_PRESETS: Per-parameter quality tuning (draft/standard/high/ultra)
    - PIPELINE_PRESETS: Full workflow configurations (cinema/quick)
"""

from typing import Dict, Any, List


# Pipeline workflow presets - full workflow configurations
PIPELINE_PRESETS: Dict[str, Dict[str, Any]] = {
    'cinema': {
        'name': 'Cinema',
        'description': 'Maximum quality with depth-guided expansion + temporal propagation',
        'stages': ['sam', 'depth', 'depth_expand', 'vitmatte', 'combine', 'matanyone'],
        'settings': {
            # Depth model: nested-large for best quality
            'depth_model': 'nested-large',
            'depth_process_res': 2048,
            'depth_process_method': 'lower',
            # use_matanyone=True causes stages to process only first frame,
            # then MatAnyone propagates temporally to all frames
            'use_matanyone': True,
            # Enable depth expansion
            'depth_expansion_enabled': True,
            'depth_expansion_tolerance': 0.1,
            'depth_expansion_percentiles': (5, 95),
            'depth_expansion_connectivity': True,
            'depth_expansion_max_px': 100,
            # Maximum detail preservation
            'hair_gamma': 1.0,
            'hair_black_point': 0.0,
            'hair_gain': 1.0,
            'guided_filter_radius': 1,
            'guided_filter_eps': 1e-6,
        }
    },
    'quick': {
        'name': 'Quick Preview',
        'description': 'Fast single-frame processing for preview',
        'stages': ['sam', 'depth', 'vitmatte', 'combine'],
        'settings': {
            'depth_model': 'small',
            'depth_process_res': None,
            # No temporal propagation - processes all frames individually
            'use_matanyone': False,
            'depth_expansion_enabled': False,
            'hair_gamma': 0.7,
            'hair_black_point': 0.03,
            'hair_gain': 1.2,
            'guided_filter_radius': 4,
            'guided_filter_eps': 1e-4,
        }
    },
}


def get_pipeline_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get a pipeline preset configuration.

    Args:
        preset_name: One of 'cinema', 'quick'

    Returns:
        Dictionary with preset configuration
    """
    return PIPELINE_PRESETS.get(preset_name, {})


def list_pipeline_presets() -> List[str]:
    """Return list of available pipeline preset names."""
    return list(PIPELINE_PRESETS.keys())


def get_pipeline_preset_description(preset_name: str) -> str:
    """Get human-readable description of a pipeline preset."""
    preset = PIPELINE_PRESETS.get(preset_name, {})
    return preset.get('description', 'Unknown preset')


# Quality presets - per-parameter quality tuning
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
